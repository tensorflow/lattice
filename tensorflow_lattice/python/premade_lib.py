# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of algorithms required for premade models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import enum
import itertools

from absl import logging
import numpy as np
import six

import tensorflow as tf
# pylint: disable=g-import-not-at-top
# Use Keras 2.
version_fn = getattr(tf.keras, 'version', None)
if version_fn and version_fn().startswith('3.'):
  import tf_keras as keras
else:
  keras = tf.keras

from . import aggregation_layer
from . import categorical_calibration_layer
from . import configs
from . import kronecker_factored_lattice_layer as kfll
from . import kronecker_factored_lattice_lib as kfl_lib
from . import lattice_layer
from . import lattice_lib
from . import linear_layer
from . import pwl_calibration_layer
from . import rtl_layer
from . import utils


# Layer names used for layers in the premade models.
AGGREGATION_LAYER_NAME = 'tfl_aggregation'
CALIB_LAYER_NAME = 'tfl_calib'
INPUT_LAYER_NAME = 'tfl_input'
KFL_LAYER_NAME = 'tfl_kronecker_factored_lattice'
LATTICE_LAYER_NAME = 'tfl_lattice'
LINEAR_LAYER_NAME = 'tfl_linear'
OUTPUT_LINEAR_COMBINATION_LAYER_NAME = 'tfl_output_linear_combination'
OUTPUT_CALIB_LAYER_NAME = 'tfl_output_calib'
RTL_LAYER_NAME = 'tfl_rtl'
RTL_INPUT_NAME = 'tfl_rtl_input'

# Prefix for passthrough (identity) nodes for shared calibration.
# These nodes pass shared calibrated values to submodels in an ensemble.
CALIB_PASSTHROUGH_NAME = 'tfl_calib_passthrough'

# Prefix for defining feature calibrator regularizers.
_INPUT_CALIB_REGULARIZER_PREFIX = 'calib_'

# Prefix for defining output calibrator regularizers.
_OUTPUT_CALIB_REGULARIZER_PREFIX = 'output_calib_'

# Weight of laplacian in feature importance for the crystal algorithm.
_LAPLACIAN_WEIGHT_IN_IMPORTANCE = 6.0

# Discount amount for repeated co-occurrence of pairs of features in crystals.
_REPEATED_PAIR_DISCOUNT_IN_CRYSTALS_SCORE = 0.5

# Maximum number of swaps for the crystals algorithm.
_MAX_CRYSTALS_SWAPS = 1000


def _input_calibration_regularizers(model_config, feature_config):
  """Returns pwl layer regularizers defined in the model and feature configs."""
  regularizer_configs = []
  regularizer_configs.extend(feature_config.regularizer_configs or [])
  regularizer_configs.extend(model_config.regularizer_configs or [])
  return [(r.name.replace(_INPUT_CALIB_REGULARIZER_PREFIX, ''), r.l1, r.l2)
          for r in regularizer_configs
          if r.name.startswith(_INPUT_CALIB_REGULARIZER_PREFIX)]


def _middle_calibration_regularizers(model_config):
  """Returns pwl layer regularizers defined in the model config."""
  regularizer_configs = []
  regularizer_configs.extend(model_config.regularizer_configs or [])
  return [(r.name.replace(_INPUT_CALIB_REGULARIZER_PREFIX, ''), r.l1, r.l2)
          for r in regularizer_configs
          if r.name.startswith(_INPUT_CALIB_REGULARIZER_PREFIX)]


def _output_calibration_regularizers(model_config):
  """Returns output calibration regularizers defined in the model config."""
  return [(r.name.replace(_OUTPUT_CALIB_REGULARIZER_PREFIX, ''), r.l1, r.l2)
          for r in model_config.regularizer_configs or []
          if r.name.startswith(_OUTPUT_CALIB_REGULARIZER_PREFIX)]


def _lattice_regularizers(model_config, feature_configs):
  """Returns lattice regularizers defined in the model and feature configs."""
  # dict from regularizer name to pair of per feature l1 and l2 amounts.
  regularizers_dict = {}
  n_dims = len(feature_configs)
  for index, feature_config in enumerate(feature_configs):
    for regularizer_config in feature_config.regularizer_configs or []:
      if not (
          regularizer_config.name.startswith(_INPUT_CALIB_REGULARIZER_PREFIX) or
          regularizer_config.name.startswith(_OUTPUT_CALIB_REGULARIZER_PREFIX)):
        if regularizer_config.name not in regularizers_dict:
          regularizers_dict[regularizer_config.name] = ([0.0] * n_dims,
                                                        [0.0] * n_dims)
        regularizers_dict[
            regularizer_config.name][0][index] += regularizer_config.l1
        regularizers_dict[
            regularizer_config.name][1][index] += regularizer_config.l2

  regularizers = [(k,) + v for k, v in regularizers_dict.items()]

  for regularizer_config in model_config.regularizer_configs or []:
    if not (
        regularizer_config.name.startswith(_INPUT_CALIB_REGULARIZER_PREFIX) or
        regularizer_config.name.startswith(_OUTPUT_CALIB_REGULARIZER_PREFIX)):
      regularizers.append((regularizer_config.name, regularizer_config.l1,
                           regularizer_config.l2))
  return regularizers


class LayerOutputRange(enum.Enum):
  """Enum to indicate the output range based on the input of the next layers."""
  MODEL_OUTPUT = 1
  INPUT_TO_LATTICE = 2
  INPUT_TO_FINAL_CALIBRATION = 3


def _output_range(layer_output_range, model_config, feature_config=None):
  """Returns min/max/init_min/init_max for a given output range."""
  if layer_output_range == LayerOutputRange.INPUT_TO_LATTICE:
    if feature_config is None:
      raise ValueError('Expecting feature config for lattice inputs.')
    output_init_min = output_min = 0.0
    output_init_max = output_max = feature_config.lattice_size - 1.0
  elif layer_output_range == LayerOutputRange.MODEL_OUTPUT:
    output_min = model_config.output_min
    output_max = model_config.output_max
    # Note: due to the multiplicative nature of KroneckerFactoredLattice layers,
    # the initialization min/max do not correspond directly to the output
    # min/max. Thus we follow the same scheme as the KroneckerFactoredLattice
    # lattice layer to properly initialize the kernel and scale such that
    # the output does in fact respect the requested bounds.
    if ((isinstance(model_config, configs.CalibratedLatticeEnsembleConfig) or
         isinstance(model_config, configs.CalibratedLatticeConfig)) and
        model_config.parameterization == 'kronecker_factored'):
      output_init_min, output_init_max = kfl_lib.default_init_params(
          output_min, output_max)
    else:
      output_init_min = np.min(model_config.output_initialization)
      output_init_max = np.max(model_config.output_initialization)
  elif layer_output_range == LayerOutputRange.INPUT_TO_FINAL_CALIBRATION:
    output_init_min = output_min = 0.0
    output_init_max = output_max = 1.0
  else:
    raise ValueError('Unsupported layer output range.')
  return output_min, output_max, output_init_min, output_init_max


def build_input_layer(feature_configs, dtype, ragged=False):
  """Creates a mapping from feature name to `keras.Input`.

  Args:
    feature_configs: A list of `tfl.configs.FeatureConfig` instances that
      specify configurations for each feature.
    dtype: dtype
    ragged: If the inputs are ragged tensors.

  Returns:
    Mapping from feature name to `keras.Input` for the inputs specified by
      `feature_configs`.
  """
  input_layer = {}
  shape = (None,) if ragged else (1,)
  for feature_config in feature_configs:
    layer_name = '{}_{}'.format(INPUT_LAYER_NAME, feature_config.name)
    if feature_config.num_buckets:
      input_layer[feature_config.name] = keras.Input(
          shape=shape, ragged=ragged, dtype=tf.int32, name=layer_name)
    else:
      input_layer[feature_config.name] = keras.Input(
          shape=shape, ragged=ragged, dtype=dtype, name=layer_name)
  return input_layer


def build_multi_unit_calibration_layers(calibration_input_layer,
                                        calibration_output_units, model_config,
                                        layer_output_range,
                                        output_single_tensor, dtype):
  """Creates a mapping from feature names to calibration outputs.

  Args:
    calibration_input_layer: A mapping from feature name to `keras.Input`.
    calibration_output_units: A mapping from feature name to units.
    model_config: Model configuration object describing model architecture.
      Should be one of the model configs in `tfl.configs`.
    layer_output_range: A `tfl.premade_lib.LayerOutputRange` enum.
    output_single_tensor: If output for each feature should be a single tensor.
    dtype: dtype

  Returns:
    A mapping from feature name to calibration output Tensors.
  """
  calibration_output = {}
  for feature_name, units in calibration_output_units.items():
    if units == 0:
      raise ValueError(
          'Feature {} is not used. Calibration output units is 0.'.format(
              feature_name))
    feature_config = model_config.feature_config_by_name(feature_name)
    calibration_input = calibration_input_layer[feature_name]
    layer_name = '{}_{}'.format(CALIB_LAYER_NAME, feature_name)

    (output_min, output_max, output_init_min,
     output_init_max) = _output_range(layer_output_range, model_config,
                                      feature_config)

    if feature_config.num_buckets:
      kernel_initializer = keras.initializers.RandomUniform(
          output_init_min, output_init_max)
      calibrated = (
          categorical_calibration_layer.CategoricalCalibration(
              num_buckets=feature_config.num_buckets,
              units=units,
              output_min=output_min,
              output_max=output_max,
              kernel_initializer=kernel_initializer,
              monotonicities=feature_config.monotonicity if isinstance(
                  feature_config.monotonicity, list) else None,
              default_input_value=feature_config.default_value,
              split_outputs=(units > 1 and not output_single_tensor),
              dtype=dtype,
              name=layer_name)(calibration_input))
    else:
      kernel_regularizer = _input_calibration_regularizers(
          model_config, feature_config)
      monotonicity = feature_config.monotonicity
      if (utils.canonicalize_monotonicity(monotonicity) == 0 and
          feature_config.pwl_calibration_always_monotonic):
        monotonicity = 1
      kernel_initializer = pwl_calibration_layer.UniformOutputInitializer(
          output_min=output_init_min,
          output_max=output_init_max,
          monotonicity=monotonicity,
          keypoints=feature_config.pwl_calibration_input_keypoints)
      calibrated = (
          pwl_calibration_layer.PWLCalibration(
              units=units,
              input_keypoints=feature_config.pwl_calibration_input_keypoints,
              output_min=output_min,
              output_max=output_max,
              clamp_min=feature_config.pwl_calibration_clamp_min,
              clamp_max=feature_config.pwl_calibration_clamp_max,
              missing_input_value=feature_config.default_value,
              impute_missing=(feature_config.default_value is not None),
              kernel_initializer=kernel_initializer,
              kernel_regularizer=kernel_regularizer,
              monotonicity=monotonicity,
              convexity=feature_config.pwl_calibration_convexity,
              split_outputs=(units > 1 and not output_single_tensor),
              input_keypoints_type=feature_config
              .pwl_calibration_input_keypoints_type,
              dtype=dtype,
              name=layer_name)(calibration_input))
    if output_single_tensor:
      calibration_output[feature_name] = calibrated
    elif units == 1:
      calibration_output[feature_name] = [calibrated]
    else:
      # calibrated will have already been split in this case.
      calibration_output[feature_name] = calibrated
  return calibration_output


def build_calibration_layers(calibration_input_layer, model_config,
                             layer_output_range, submodels,
                             separate_calibrators, dtype):
  """Creates a calibration layer for `submodels` as list of list of features.

  Args:
    calibration_input_layer: A mapping from feature name to `keras.Input`.
    model_config: Model configuration object describing model architecture.
      Should be one of the model configs in `tfl.configs`.
    layer_output_range: A `tfl.premade_lib.LayerOutputRange` enum.
    submodels: A list of list of feature names.
    separate_calibrators: If features should be separately calibrated for each
      lattice in an ensemble.
    dtype: dtype

  Returns:
    A list of list of Tensors representing a calibration layer for `submodels`.
  """
  # Create a list of (feature_name, calibration_output_idx) pairs for each
  # submodel. When using shared calibration, all submodels will have
  # calibration_output_idx = 0.
  submodels_input_features = []
  calibration_last_index = collections.defaultdict(int)
  for submodel in submodels:
    submodel_input_features = []
    submodels_input_features.append(submodel_input_features)
    for feature_name in submodel:
      submodel_input_features.append(
          (feature_name, calibration_last_index[feature_name]))
      if separate_calibrators:
        calibration_last_index[feature_name] += 1

  # This is to account for shared calibration.
  calibration_output_units = {
      name: max(index, 1) for name, index in calibration_last_index.items()
  }
  calibration_output = build_multi_unit_calibration_layers(
      calibration_input_layer=calibration_input_layer,
      calibration_output_units=calibration_output_units,
      model_config=model_config,
      layer_output_range=layer_output_range,
      output_single_tensor=False,
      dtype=dtype)

  # Create passthrough nodes for each submodel input so that we can recover
  # the model structure for plotting and analysis.
  # {CALIB_PASSTHROUGH_NAME}_{feature_name}_
  #   {calibration_output_idx}_{submodel_idx}_{submodel_input_idx}
  submodels_inputs = []
  for submodel_idx, submodel_input_features in enumerate(
      submodels_input_features):
    submodel_inputs = []
    submodels_inputs.append(submodel_inputs)
    for (submodel_input_idx,
         (feature_name,
          calibration_output_idx)) in enumerate(submodel_input_features):
      passthrough_name = '{}_{}_{}_{}_{}'.format(CALIB_PASSTHROUGH_NAME,
                                                 feature_name,
                                                 calibration_output_idx,
                                                 submodel_idx,
                                                 submodel_input_idx)
      submodel_inputs.append(
          tf.identity(
              calibration_output[feature_name][calibration_output_idx],
              name=passthrough_name))

  return submodels_inputs


def build_aggregation_layer(aggregation_input_layer, model_config,
                            calibrated_lattice_models, layer_output_range,
                            submodel_index, dtype):
  """Creates an aggregation layer using the given calibrated lattice models.

  Args:
    aggregation_input_layer: A list or a mapping from feature name to
      `keras.Input`, in the order or format expected by
      `calibrated_lattice_models`.
    model_config: Model configuration object describing model architecture.
      Should be one of the model configs in `tfl.configs`.
    calibrated_lattice_models: A list of calibrated lattice models of size
      model_config.middle_diemnsion, where each calbirated lattice model
      instance is constructed using the same model configuration object.
    layer_output_range: A `tfl.premade_lib.LayerOutputRange` enum.
    submodel_index: Corresponding index into submodels.
    dtype: dtype

  Returns:
    A list of list of Tensors representing a calibration layer for `submodels`.
  """
  (output_min, output_max, output_init_min,
   output_init_max) = _output_range(layer_output_range, model_config)

  lattice_sizes = [model_config.middle_lattice_size
                  ] * model_config.middle_dimension
  lattice_monotonicities = [1] * model_config.middle_dimension

  # Create the aggergated embeddings to pass to the middle lattice.
  lattice_inputs = []
  for i in range(model_config.middle_dimension):
    agg_layer_name = '{}_{}'.format(AGGREGATION_LAYER_NAME, i)
    agg_output = aggregation_layer.Aggregation(
        calibrated_lattice_models[i], name=agg_layer_name)(
            aggregation_input_layer)
    agg_output = keras.layers.Reshape((1,))(agg_output)
    if model_config.middle_calibration:
      agg_output = pwl_calibration_layer.PWLCalibration(
          input_keypoints=np.linspace(
              -1.0,
              1.0,
              num=model_config.middle_calibration_num_keypoints,
              dtype=np.float32),
          output_min=0.0,
          output_max=lattice_sizes[i] - 1.0,
          monotonicity=utils.canonicalize_monotonicity(
              model_config.middle_monotonicity),
          kernel_regularizer=_middle_calibration_regularizers(model_config),
          input_keypoints_type=model_config
          .middle_calibration_input_keypoints_type,
          dtype=dtype,
      )(
          agg_output)
      agg_output = keras.layers.Reshape((1,))(agg_output)
    lattice_inputs.append(agg_output)

  # We use random monotonic initialization here to break the symmetry that we
  # would otherwise have between middle lattices. Since we use the same
  # CalibratedLattice for each of the middle dimensions, if we do not randomly
  # initialize the middle lattice we will have the same gradient flow back for
  # each middle dimension, thus acting the same as if there was only one middle
  # dimension.
  kernel_initializer = lattice_layer.RandomMonotonicInitializer(
      lattice_sizes=lattice_sizes,
      output_min=output_init_min,
      output_max=output_init_max)
  lattice_layer_name = '{}_{}'.format(LATTICE_LAYER_NAME, submodel_index)
  return lattice_layer.Lattice(
      lattice_sizes=lattice_sizes,
      monotonicities=lattice_monotonicities,
      output_min=output_min,
      output_max=output_max,
      clip_inputs=False,
      interpolation=model_config.middle_lattice_interpolation,
      kernel_initializer=kernel_initializer,
      dtype=dtype,
      name=lattice_layer_name,
  )(
      lattice_inputs)


def _monotonicities_from_feature_configs(feature_configs):
  """Returns list of monotonicities defined in the given feature_configs."""
  monotonicities = []
  for feature_config in feature_configs:
    if not feature_config.monotonicity:
      monotonicities.append(0)
    elif (isinstance(feature_config.monotonicity, six.string_types) and
          feature_config.monotonicity.lower() == 'none'):
      monotonicities.append(0)
    else:
      monotonicities.append(1)
  return monotonicities


def _dominance_constraints_from_feature_configs(feature_configs):
  """Returns list of dominance constraints in the given feature_configs."""
  feature_names = [feature_config.name for feature_config in feature_configs]
  monotonic_dominances = []
  for dominant_idx, dominant_feature_config in enumerate(feature_configs):
    for dominance_config in dominant_feature_config.dominates or []:
      if dominance_config.feature_name in feature_names:
        weak_idx = feature_names.index(dominance_config.feature_name)
        if dominance_config.dominance_type == 'monotonic':
          monotonic_dominances.append((dominant_idx, weak_idx))
        else:
          raise ValueError('Unrecognized dominance type: {}'.format(
              dominance_config.dominance_type))
  return monotonic_dominances


def _canonical_feature_names(model_config, feature_names=None):
  if feature_names is not None:
    return feature_names
  if model_config.feature_configs is None:
    raise ValueError(
        'Feature configs must be specified if feature names are not provided.')
  return [
      feature_config.name for feature_config in model_config.feature_configs
  ]


def build_linear_layer(linear_input, feature_configs, model_config,
                       weighted_average, submodel_index, dtype):
  """Creates a `tfl.layers.Linear` layer initialized to be an average.

  Args:
    linear_input: Input to the linear layer.
    feature_configs: A list of `tfl.configs.FeatureConfig` instances that
      specify configurations for each feature.
    model_config: Model configuration object describing model architecture.
      Should be one of the model configs in `tfl.configs`.
    weighted_average: If the linear coefficients should be positive and sum up
      to one.
    submodel_index: Corresponding index into submodels.
    dtype: dtype

  Returns:
    A `tfl.layers.Linear` instance.
  """
  layer_name = '{}_{}'.format(LINEAR_LAYER_NAME, submodel_index)

  linear_input = keras.layers.Concatenate(axis=1)(linear_input)
  num_input_dims = len(feature_configs)
  kernel_initializer = keras.initializers.Constant([1.0 / num_input_dims] *
                                                      num_input_dims)
  bias_initializer = keras.initializers.Constant(0)

  if weighted_average:
    # Linear coefficients should be possitive and sum up to one.
    linear_monotonicities = [1] * num_input_dims
    normalization_order = 1
    use_bias = False
  else:
    linear_monotonicities = _monotonicities_from_feature_configs(
        feature_configs)
    normalization_order = None
    use_bias = model_config.use_bias

  monotonic_dominances = _dominance_constraints_from_feature_configs(
      feature_configs)

  return linear_layer.Linear(
      num_input_dims=num_input_dims,
      monotonicities=linear_monotonicities,
      monotonic_dominances=monotonic_dominances,
      use_bias=use_bias,
      normalization_order=normalization_order,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      dtype=dtype,
      name=layer_name)(
          linear_input)


def build_lattice_layer(lattice_input, feature_configs, model_config,
                        layer_output_range, submodel_index, is_inside_ensemble,
                        dtype):
  """Creates a `tfl.layers.Lattice` layer.

  Args:
    lattice_input: Input to the lattice layer.
    feature_configs: A list of `tfl.configs.FeatureConfig` instances that
      specify configurations for each feature.
    model_config: Model configuration object describing model architecture.
      Should be one of the model configs in `tfl.configs`.
    layer_output_range: A `tfl.premade_lib.LayerOutputRange` enum.
    submodel_index: Corresponding index into submodels.
    is_inside_ensemble: If this layer is inside an ensemble.
    dtype: dtype

  Returns:
    A `tfl.layers.Lattice` instance if `model_config.parameterization` is set to
    `'all_vertices'` or a `tfl.layers.KroneckerFactoredLattice` instance if
    set to `'kronecker_factored'`.

  Raises:
    ValueError: If `model_config.parameterization` is not one of
      `'all_vertices'` or `'kronecker_factored'`.
  """
  layer_name = '{}_{}'.format(LATTICE_LAYER_NAME, submodel_index)

  (output_min, output_max, output_init_min,
   output_init_max) = _output_range(layer_output_range, model_config)

  feature_names = [feature_config.name for feature_config in feature_configs]
  lattice_sizes = [
      feature_config.lattice_size for feature_config in feature_configs
  ]
  lattice_monotonicities = _monotonicities_from_feature_configs(feature_configs)
  lattice_unimodalities = [
      feature_config.unimodality for feature_config in feature_configs
  ]
  lattice_regularizers = _lattice_regularizers(model_config,
                                               feature_configs) or None

  # Construct trust constraints within this lattice.
  edgeworth_trusts = []
  trapezoid_trusts = []
  for conditional_idx, conditional_feature_config in enumerate(feature_configs):
    for trust_config in conditional_feature_config.reflects_trust_in or []:
      if trust_config.feature_name in feature_names:
        main_idx = feature_names.index(trust_config.feature_name)
        if trust_config.trust_type == 'edgeworth':
          edgeworth_trusts.append(
              (main_idx, conditional_idx, trust_config.direction))
        elif trust_config.trust_type == 'trapezoid':
          trapezoid_trusts.append(
              (main_idx, conditional_idx, trust_config.direction))
        else:
          raise ValueError('Unrecognized trust type: {}'.format(
              trust_config.trust_type))
      elif is_inside_ensemble and trust_config.trust_type == 'trapezoid':
        logging.warning(
            'A "main" feature (%s) for a trapezoid trust constraint is not '
            'present in a lattice that includes the "conditional" feature '
            '(%s). In an ensemble model, this can result in constraint '
            'violations. Consider manually setting the ensemble structure if '
            'this constraint needs to be satisfied.', trust_config.feature_name,
            conditional_feature_config.name)

  monotonic_dominances = _dominance_constraints_from_feature_configs(
      feature_configs)

  if model_config.parameterization == 'all_vertices':
    layer_name = '{}_{}'.format(LATTICE_LAYER_NAME, submodel_index)
    kernel_initializer = lattice_layer.LinearInitializer(
        lattice_sizes=lattice_sizes,
        monotonicities=lattice_monotonicities,
        unimodalities=lattice_unimodalities,
        output_min=output_init_min,
        output_max=output_init_max)
    return lattice_layer.Lattice(
        lattice_sizes=lattice_sizes,
        monotonicities=lattice_monotonicities,
        unimodalities=lattice_unimodalities,
        edgeworth_trusts=edgeworth_trusts,
        trapezoid_trusts=trapezoid_trusts,
        monotonic_dominances=monotonic_dominances,
        output_min=output_min,
        output_max=output_max,
        clip_inputs=False,
        interpolation=model_config.interpolation,
        kernel_regularizer=lattice_regularizers,
        kernel_initializer=kernel_initializer,
        dtype=dtype,
        name=layer_name)(
            lattice_input)
  elif model_config.parameterization == 'kronecker_factored':
    layer_name = '{}_{}'.format(KFL_LAYER_NAME, submodel_index)
    kernel_initializer = kfll.KFLRandomMonotonicInitializer(
        monotonicities=lattice_monotonicities,
        init_min=output_init_min,
        init_max=output_init_max,
        seed=model_config.random_seed)
    scale_initializer = kfll.ScaleInitializer(
        output_min=output_min, output_max=output_max)
    return kfll.KroneckerFactoredLattice(
        lattice_sizes=lattice_sizes[0],
        num_terms=model_config.num_terms,
        monotonicities=lattice_monotonicities,
        output_min=output_min,
        output_max=output_max,
        clip_inputs=False,
        kernel_initializer=kernel_initializer,
        scale_initializer=scale_initializer,
        dtype=dtype,
        name=layer_name)(
            lattice_input)
  else:
    raise ValueError('Unknown type of parameterization: {}'.format(
        model_config.parameterization))


def build_lattice_ensemble_layer(submodels_inputs, model_config, dtype):
  """Creates an ensemble of `tfl.layers.Lattice` layers.

  Args:
    submodels_inputs: List of inputs to each of the lattice layers in the
      ensemble. The order corresponds to the elements of model_config.lattices.
    model_config: Model configuration object describing model architecture.
      Should be one of the model configs in `tfl.configs`.
    dtype: dtype

  Returns:
    A list of `tfl.layers.Lattice` instances.
  """
  lattice_outputs = []
  for submodel_index, (lattice_feature_names, lattice_input) in enumerate(
      zip(model_config.lattices, submodels_inputs)):
    lattice_feature_configs = [
        model_config.feature_config_by_name(feature_name)
        for feature_name in lattice_feature_names
    ]
    lattice_layer_output_range = (
        LayerOutputRange.INPUT_TO_FINAL_CALIBRATION
        if model_config.output_calibration else LayerOutputRange.MODEL_OUTPUT)
    lattice_outputs.append(
        build_lattice_layer(
            lattice_input=lattice_input,
            feature_configs=lattice_feature_configs,
            model_config=model_config,
            layer_output_range=lattice_layer_output_range,
            submodel_index=submodel_index,
            is_inside_ensemble=True,
            dtype=dtype))
  return lattice_outputs


def build_rtl_layer(calibration_outputs, model_config, submodel_index,
                    average_outputs, dtype):
  """Creates a `tfl.layers.RTL` layer.

  This function expects that all features defined in
  model_config.feature_configs are used and present in calibration_outputs.

  Args:
    calibration_outputs: A mapping from feature name to calibration output.
    model_config: Model configuration object describing model architecture.
      Should be one of the model configs in `tfl.configs`.
    submodel_index: Corresponding index into submodels.
    average_outputs: Whether to average the outputs of this layer.
    dtype: dtype

  Returns:
    A `tfl.layers.RTL` instance.

  Raises:
    ValueError: If `model_config.parameterization` is not one of
      `'all_vertices'` or `'kronecker_factored'`.
  """
  layer_name = '{}_{}'.format(RTL_LAYER_NAME, submodel_index)

  rtl_layer_output_range = (
      LayerOutputRange.INPUT_TO_FINAL_CALIBRATION
      if model_config.output_calibration else LayerOutputRange.MODEL_OUTPUT)

  (output_min, output_max, output_init_min,
   output_init_max) = _output_range(rtl_layer_output_range, model_config)

  lattice_regularizers = _lattice_regularizers(
      model_config, model_config.feature_configs) or None

  rtl_inputs = collections.defaultdict(list)
  for feature_config in model_config.feature_configs:
    passthrough_name = '{}_{}'.format(RTL_INPUT_NAME, feature_config.name)
    calibration_output = tf.identity(
        calibration_outputs[feature_config.name], name=passthrough_name)
    if feature_config.monotonicity in [1, -1, 'increasing', 'decreasing']:
      rtl_inputs['increasing'].append(calibration_output)
    else:
      rtl_inputs['unconstrained'].append(calibration_output)

  lattice_size = model_config.feature_configs[0].lattice_size
  if model_config.parameterization == 'all_vertices':
    kernel_initializer = 'random_monotonic_initializer'
  elif model_config.parameterization == 'kronecker_factored':
    kernel_initializer = 'kfl_random_monotonic_initializer'
  else:
    raise ValueError('Unknown type of parameterization: {}'.format(
        model_config.parameterization))
  return rtl_layer.RTL(
      num_lattices=model_config.num_lattices,
      lattice_rank=model_config.lattice_rank,
      lattice_size=lattice_size,
      output_min=output_min,
      output_max=output_max,
      init_min=output_init_min,
      init_max=output_init_max,
      random_seed=model_config.random_seed,
      clip_inputs=False,
      interpolation=model_config.interpolation,
      parameterization=model_config.parameterization,
      num_terms=model_config.num_terms,
      kernel_regularizer=lattice_regularizers,
      kernel_initializer=kernel_initializer,
      average_outputs=average_outputs,
      dtype=dtype,
      name=layer_name)(
          rtl_inputs)


def build_calibrated_lattice_ensemble_layer(calibration_input_layer,
                                            model_config, average_outputs,
                                            dtype):
  """Creates a calibration layer followed by a lattice ensemble layer.

  Args:
    calibration_input_layer: A mapping from feature name to `keras.Input`.
    model_config: Model configuration object describing model architecture.
      Should be one of the model configs in `tfl.configs`.
    average_outputs: Whether to average the outputs of this layer.
    dtype: dtype

  Returns:
    A `tfl.layers.RTL` instance if model_config.lattices is 'rtl_layer.
    Otherwise a list of `tfl.layers.Lattice` instances.
  """
  if model_config.lattices == 'rtl_layer':
    num_features = len(model_config.feature_configs)
    units = [1] * num_features
    if model_config.separate_calibrators:
      num_inputs = model_config.num_lattices * model_config.lattice_rank
      # We divide the number of inputs semi-evenly by the number of features.
      # TODO: support setting number of calibration units.
      for i in range(num_features):
        units[i] = ((i + 1) * num_inputs // num_features -
                    i * num_inputs // num_features)
    calibration_output_units = {
        feature_config.name: units[i]
        for i, feature_config in enumerate(model_config.feature_configs)
    }
    calibration_outputs = build_multi_unit_calibration_layers(
        calibration_input_layer=calibration_input_layer,
        calibration_output_units=calibration_output_units,
        model_config=model_config,
        layer_output_range=LayerOutputRange.INPUT_TO_LATTICE,
        output_single_tensor=True,
        dtype=dtype)

    lattice_outputs = build_rtl_layer(
        calibration_outputs=calibration_outputs,
        model_config=model_config,
        submodel_index=0,
        average_outputs=average_outputs,
        dtype=dtype)
  else:
    submodels_inputs = build_calibration_layers(
        calibration_input_layer=calibration_input_layer,
        model_config=model_config,
        layer_output_range=LayerOutputRange.INPUT_TO_LATTICE,
        submodels=model_config.lattices,
        separate_calibrators=model_config.separate_calibrators,
        dtype=dtype)

    lattice_outputs = build_lattice_ensemble_layer(
        submodels_inputs=submodels_inputs,
        model_config=model_config,
        dtype=dtype)

    if average_outputs:
      lattice_outputs = keras.layers.Average()(lattice_outputs)

  return lattice_outputs


def build_linear_combination_layer(ensemble_outputs, model_config, dtype):
  """Creates a `tfl.layers.Linear` layer initialized to be an average.

  Args:
    ensemble_outputs: Ensemble outputs to be linearly combined.
    model_config: Model configuration object describing model architecture.
      Should be one of the model configs in `tfl.configs`.
    dtype: dtype

  Returns:
    A `tfl.layers.Linear` instance.
  """
  if isinstance(ensemble_outputs, list):
    num_input_dims = len(ensemble_outputs)
    linear_input = keras.layers.Concatenate(axis=1)(ensemble_outputs)
  else:
    num_input_dims = int(ensemble_outputs.shape[1])
    linear_input = ensemble_outputs
  kernel_initializer = keras.initializers.Constant(1.0 / num_input_dims)
  bias_initializer = keras.initializers.Constant(0)

  if (not model_config.output_calibration and
      model_config.output_min is None and model_config.output_max is None):
    normalization_order = None
  else:
    # We need to use weighted average to keep the output range.
    normalization_order = 1
    # Bias term cannot be used when this layer should have bounded output.
    if model_config.use_bias:
      raise ValueError('Cannot use a bias term in linear combination with '
                       'output bounds or output calibration')

  return linear_layer.Linear(
      num_input_dims=num_input_dims,
      monotonicities=['increasing'] * num_input_dims,
      normalization_order=normalization_order,
      use_bias=model_config.use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      dtype=dtype,
      name=OUTPUT_LINEAR_COMBINATION_LAYER_NAME)(
          linear_input)


def build_output_calibration_layer(output_calibration_input, model_config,
                                   dtype):
  """Creates a monotonic output calibration layer with inputs range [0, 1].

  Args:
    output_calibration_input: Input to the output calibration layer.
    model_config: Model configuration object describing model architecture.
      Should be one of the model configs in `tfl.configs`.
    dtype: dtype

  Returns:
    A `tfl.layers.PWLCalibration` instance.
  """
  # kernel format: bias followed by diffs between consecutive keypoint outputs.
  kernel_init_values = np.ediff1d(
      model_config.output_initialization,
      to_begin=model_config.output_initialization[0])
  input_keypoints = np.linspace(0.0, 1.0, num=len(kernel_init_values))
  kernel_initializer = keras.initializers.Constant(kernel_init_values)
  kernel_regularizer = _output_calibration_regularizers(model_config)
  return pwl_calibration_layer.PWLCalibration(
      input_keypoints=input_keypoints,
      output_min=model_config.output_min,
      output_max=model_config.output_max,
      kernel_initializer=kernel_initializer,
      kernel_regularizer=kernel_regularizer,
      monotonicity=1,
      input_keypoints_type=model_config.output_calibration_input_keypoints_type,
      dtype=dtype,
      name=OUTPUT_CALIB_LAYER_NAME)(
          output_calibration_input)


def set_categorical_monotonicities(feature_configs):
  """Maps categorical monotonicities to indices based on specified vocab list.

  Args:
    feature_configs: A list of `tfl.configs.FeatureConfig` objects.
  """
  if not isinstance(feature_configs, list) or any(
      not isinstance(fc, configs.FeatureConfig) for fc in feature_configs):
    raise ValueError(
        'feature_configs must be a list of tfl.configs.FeatureConfig objects: '
        '{}'.format(feature_configs))
  for feature_config in feature_configs:
    if feature_config.num_buckets and isinstance(feature_config.monotonicity,
                                                 list):
      # Make sure the vocabulary list exists. If not, assume user has already
      # properly set monotonicity as proper indices for this calibrator.
      if not feature_config.vocabulary_list:
        continue
      if not all(
          isinstance(m, (list, tuple)) and len(m) == 2
          for m in feature_config.monotonicity):
        raise ValueError(
            'Monotonicities should be a list of pairs (list/tuples): {}'.format(
                feature_config.monotonicity))
      indexed_monotonicities = []
      index_map = {
          category: index
          for (index, category) in enumerate(feature_config.vocabulary_list)
      }
      if feature_config.default_value is not None:
        index_map[feature_config.default_value] = feature_config.num_buckets - 1
      for left, right in feature_config.monotonicity:
        for category in [left, right]:
          if category not in index_map:
            raise ValueError(
                'Category `{}` not found in vocabulary list for feature `{}`'
                .format(category, feature_config.name))
        indexed_monotonicities.append((index_map[left], index_map[right]))

      feature_config.monotonicity = indexed_monotonicities


def set_random_lattice_ensemble(model_config, feature_names=None):
  """Sets random lattice ensemble in the given model_config.

  Args:
    model_config: Model configuration object describing model architecture.
      Should be one of the model configs in `tfl.configs`.
    feature_names: A list of feature names. If not provided, feature names will
      be extracted from the feature configs contained in the model_config.
  """
  if not isinstance(model_config, configs.CalibratedLatticeEnsembleConfig):
    raise ValueError(
        'model_config must be a tfl.configs.CalibratedLatticeEnsembleConfig: {}'
        .format(type(model_config)))
  if model_config.lattices != 'random':
    raise ValueError('model_config.lattices must be set to \'random\'.')
  feature_names = _canonical_feature_names(model_config, feature_names)
  # Start by using each feature once.
  np.random.seed(model_config.random_seed)
  model_config.lattices = [[] for _ in range(model_config.num_lattices)]
  for feature_name in feature_names:
    non_full_indices = [
        i for (i, lattice) in enumerate(model_config.lattices)
        if len(lattice) < model_config.lattice_rank
    ]
    model_config.lattices[np.random.choice(non_full_indices)].append(
        feature_name)

  # Fill up lattices avoiding repeated features.
  for lattice in model_config.lattices:
    feature_names_not_in_lattice = [
        feature_name for feature_name in feature_names
        if feature_name not in lattice
    ]
    remaining_size = model_config.lattice_rank - len(lattice)
    lattice.extend(
        np.random.choice(
            feature_names_not_in_lattice, size=remaining_size, replace=False))


def _add_pair_to_ensemble(lattices, lattice_rank, i, j):
  """Adds pair (i, j) to the ensemble heuristically."""
  # First check if (i, j) pair is already present in a lattice.
  for lattice in lattices:
    if i in lattice and j in lattice:
      return

  # Try adding to a lattice that already has either i or j.
  for lattice in lattices:
    if len(lattice) < lattice_rank:
      if i in lattice:
        lattice.add(j)
        return
      if j in lattice:
        lattice.add(i)
        return

  # Add both i and j to a lattice that has enough space left.
  for lattice in lattices:
    if len(lattice) < lattice_rank - 1:
      lattice.add(i)
      lattice.add(j)
      return

  # Create a new lattice with pair (i, j).
  lattices.append(set([i, j]))


def _set_all_pairs_cover_lattices(prefitting_model_config, feature_names):
  """Sets prefitting lattice ensemble such that it covers all feature pairs."""
  # Pairs of co-occurrence that need to exist in the all-pairs cover.
  to_cover = list(itertools.combinations(range(len(feature_names)), 2))
  np.random.seed(prefitting_model_config.random_seed)
  np.random.shuffle(to_cover)

  lattices = []

  for (i, j) in to_cover:
    _add_pair_to_ensemble(lattices, prefitting_model_config.lattice_rank, i, j)

  prefitting_model_config.lattices = [
      [feature_names[i] for i in lattice] for lattice in lattices
  ]


def construct_prefitting_model_config(model_config, feature_names=None):
  """Constructs a model config for a prefitting model for crystal extraction.

  Args:
    model_config: Model configuration object describing model architecture.
      Should be a `tfl.configs.CalibratedLatticeEnsemble` instance.
    feature_names: A list of feature names. If not provided, feature names will
      be extracted from the feature configs contained in the model_config.

  Returns:
    A `tfl.configs.CalibratedLatticeEnsembleConfig` instance.
  """
  if not isinstance(model_config, configs.CalibratedLatticeEnsembleConfig):
    raise ValueError(
        'model_config must be a tfl.configs.CalibratedLatticeEnsembleConfig: {}'
        .format(type(model_config)))
  if model_config.lattices != 'crystals':
    raise ValueError('model_config.lattices must be set to \'crystals\'.')
  feature_names = _canonical_feature_names(model_config, feature_names)

  if len(feature_names) <= model_config.lattice_rank:
    raise ValueError(
        'model_config.lattice_rank must be less than the number of features '
        'when using \'crystals\' algorithm. If you want to use all features in '
        'every lattice, set model_config.lattices to \'random\'.')

  # Make a copy of the model config provided and set all pairs covered.
  prefitting_model_config = copy.deepcopy(model_config)
  # Set parameterization of prefitting model to 'all_vertices' to extract
  # crystals using normal lattice because we do not have laplacian/torsion
  # regularizers for KFL. This should still extract could feature combinations.
  prefitting_model_config.parameterization = 'all_vertices'
  _set_all_pairs_cover_lattices(
      prefitting_model_config=prefitting_model_config,
      feature_names=feature_names)

  # Trim the model for faster prefitting.
  for feature_config in prefitting_model_config.feature_configs:
    feature_config.lattice_size = 2
    # Unimodality requires lattice_size > 2.
    feature_config.unimodality = 0
    # Disable 2d constraints to avoid potential constraint violations.
    feature_config.dominates = None
    feature_config.reflects_trust_in = None

  # Return our properly constructed prefitting model config.
  return prefitting_model_config


def _verify_prefitting_model(prefitting_model, feature_names):
  """Checks that prefitting_model has the proper input layer."""
  if isinstance(prefitting_model, keras.Model):
    layer_names = [layer.name for layer in prefitting_model.layers]
  elif hasattr(prefitting_model, 'get_variable_names'):  # estimator
    layer_names = prefitting_model.get_variable_names()
  else:
    raise ValueError('Invalid model type for prefitting_model: {}'.format(
        type(prefitting_model)))
  for feature_name in feature_names:
    if isinstance(prefitting_model, keras.Model):
      input_layer_name = '{}_{}'.format(INPUT_LAYER_NAME, feature_name)
      if input_layer_name not in layer_names:
        raise ValueError(
            'prefitting_model does not match prefitting_model_config. Make '
            'sure that prefitting_model is the proper type and constructed '
            'from the prefitting_model_config: {}'.format(
                type(prefitting_model)))
    else:
      pwl_input_layer_name = '{}_{}/{}'.format(
          CALIB_LAYER_NAME, feature_name,
          pwl_calibration_layer.PWL_CALIBRATION_KERNEL_NAME)
      cat_input_layer_name = '{}_{}/{}'.format(
          CALIB_LAYER_NAME, feature_name,
          categorical_calibration_layer.CATEGORICAL_CALIBRATION_KERNEL_NAME)
      if (pwl_input_layer_name not in layer_names and
          cat_input_layer_name not in layer_names):
        raise ValueError(
            'prefitting_model does not match prefitting_model_config. Make '
            'sure that prefitting_model is the proper type and constructed '
            'from the prefitting_model_config: {}'.format(
                type(prefitting_model)))


def _get_lattice_weights(prefitting_model, lattice_index):
  """Gets the weights of the lattice at the specfied index."""
  if isinstance(prefitting_model, keras.Model):
    lattice_layer_name = '{}_{}'.format(LATTICE_LAYER_NAME, lattice_index)
    weights = keras.backend.get_value(
        prefitting_model.get_layer(lattice_layer_name).weights[0])
  else:
    # We have already checked the types by this point, so if prefitting_model
    # is not a keras Model it must be an Estimator.
    lattice_kernel_variable_name = '{}_{}/{}'.format(
        LATTICE_LAYER_NAME, lattice_index, lattice_layer.LATTICE_KERNEL_NAME)
    weights = prefitting_model.get_variable_value(lattice_kernel_variable_name)
  return weights


def _get_torsions_and_laplacians(prefitting_model_config, prefitting_model,
                                 feature_names):
  """Returns average torsion and laplacian regularizers in prefitted model."""
  num_fatures = len(feature_names)
  laplacians = [[] for _ in range(num_fatures)]
  torsions = [[[] for _ in range(num_fatures)] for _ in range(num_fatures)]
  for (lattice_index, lattice) in enumerate(prefitting_model_config.lattices):
    # Get lattice weights and normalize them.
    weights = _get_lattice_weights(prefitting_model, lattice_index)
    weights -= np.min(weights)
    weights /= np.max(weights)
    weights = tf.constant(weights)

    # Convert feature names in the lattice to their index in feature_names.
    lattice = [feature_names.index(feature_name) for feature_name in lattice]
    lattice_sizes = [2] * len(lattice)
    # feature_* refers to feature index in feature_names.
    # within_lattice_index_* is the index of input dimenstion of the lattice.
    for within_lattice_index_0, feature_0 in enumerate(lattice):
      l2 = [0] * len(lattice)
      l2[within_lattice_index_0] = 1
      laplacians[feature_0].append(
          lattice_lib.laplacian_regularizer(
              weights=weights, lattice_sizes=lattice_sizes, l2=l2))
      for within_lattice_index_1, feature_1 in enumerate(lattice):
        if within_lattice_index_1 > within_lattice_index_0:
          l2 = [0] * len(lattice)
          l2[within_lattice_index_0] = 1
          l2[within_lattice_index_1] = 1
          torsion = lattice_lib.torsion_regularizer(
              weights=weights, lattice_sizes=lattice_sizes, l2=l2)
          torsions[feature_0][feature_1].append(torsion)
          torsions[feature_1][feature_0].append(torsion)

  if not tf.executing_eagerly():
    with tf.compat.v1.Session() as sess:
      laplacians = sess.run(laplacians)
      torsions = sess.run(torsions)

  laplacians = [np.mean(v) for v in laplacians]
  torsions = [[np.mean(v) if v else 0.0 for v in row] for row in torsions]
  return torsions, laplacians


def _get_final_crystal_lattices(model_config, prefitting_model_config,
                                prefitting_model, feature_names):
  """Extracts the lattice ensemble structure from the prefitting model."""
  torsions, laplacians = _get_torsions_and_laplacians(
      prefitting_model_config=prefitting_model_config,
      prefitting_model=prefitting_model,
      feature_names=feature_names)

  # Calculate features' importance_score = lambda * laplacians + torsion.
  # Used to allocate slots to useful features with more non-linear interactions.
  num_features = len(feature_names)
  importance_scores = np.array(laplacians) * _LAPLACIAN_WEIGHT_IN_IMPORTANCE
  for feature_0, feature_1 in itertools.combinations(range(num_features), 2):
    importance_scores[feature_0] += torsions[feature_0][feature_1]
    importance_scores[feature_1] += torsions[feature_0][feature_1]

  # Each feature is used at least once, and the remaining slots are distributed
  # proportional to the importance_scores.
  features_uses = [1] * num_features
  total_feature_use = model_config.num_lattices * model_config.lattice_rank
  remaining_uses = total_feature_use - num_features
  remaining_scores = np.sum(importance_scores)
  for feature in np.argsort(-importance_scores):
    added_uses = int(
        round(remaining_uses * importance_scores[feature] / remaining_scores))
    # Each feature cannot be used more than once in a finalized lattice.
    added_uses = min(added_uses, model_config.num_lattices - 1)
    features_uses[feature] += added_uses
    remaining_uses -= added_uses
    remaining_scores -= importance_scores[feature]
  assert np.sum(features_uses) == total_feature_use

  # Add features to add list in round-robin order.
  add_list = []
  for use in range(1, max(features_uses) + 1):
    for feature_index, feature_use in enumerate(features_uses):
      if use <= feature_use:
        add_list.append(feature_index)
  assert len(add_list) == total_feature_use

  # Setup initial lattices that will be optimized by swapping later.
  lattices = [[] for _ in range(model_config.num_lattices)]
  cooccurrence_counts = [[0] * num_features for _ in range(num_features)]
  for feature_to_be_added in add_list:
    # List of pairs of (addition_score, candidate_lattice_to_add_to).
    score_candidates_pairs = []
    for candidate_lattice_to_add_to in range(model_config.num_lattices):
      # addition_score indicates the priority of an addition.
      if len(
          lattices[candidate_lattice_to_add_to]) >= model_config.lattice_rank:
        # going out of bound on the lattice
        addition_score = -2.0
      elif feature_to_be_added in lattices[candidate_lattice_to_add_to]:
        # repeates (fixed repeats later by swapping)
        addition_score = -1.0
      elif not lattices[candidate_lattice_to_add_to]:
        # adding a new lattice roughly has an "average" lattice score
        addition_score = np.mean(torsions) * model_config.lattice_rank**2 / 2
      else:
        # all other cases: change in total discounted torsion after addition.
        addition_score = 0.0
        for other_feature in lattices[candidate_lattice_to_add_to]:
          addition_score += (
              torsions[feature_to_be_added][other_feature] *
              _REPEATED_PAIR_DISCOUNT_IN_CRYSTALS_SCORE
              **(cooccurrence_counts[feature_to_be_added][other_feature]))

      score_candidates_pairs.append(
          (addition_score, candidate_lattice_to_add_to))

    # Use the highest scoring addition.
    score_candidates_pairs.sort(reverse=True)
    best_candidate_lattice_to_add_to = score_candidates_pairs[0][1]
    for other_feature in lattices[best_candidate_lattice_to_add_to]:
      cooccurrence_counts[feature_to_be_added][other_feature] += 1
      cooccurrence_counts[other_feature][feature_to_be_added] += 1
    lattices[best_candidate_lattice_to_add_to].append(feature_to_be_added)

  # Apply swapping operations to increase within-lattice torsion.
  changed = True
  iteration = 0
  while changed:
    if iteration > _MAX_CRYSTALS_SWAPS:
      logging.info('Crystals algorithm did not fully converge.')
      break
    changed = False
    iteration += 1
    for lattice_0, lattice_1 in itertools.combinations(lattices, 2):
      # For every pair of lattices: lattice_0, lattice_1
      for index_0, index_1 in itertools.product(
          range(len(lattice_0)), range(len(lattice_1))):
        # Consider swapping lattice_0[index_0] with lattice_1[index_1]
        rest_lattice_0 = list(lattice_0)
        rest_lattice_1 = list(lattice_1)
        feature_0 = rest_lattice_0.pop(index_0)
        feature_1 = rest_lattice_1.pop(index_1)
        if feature_0 == feature_1:
          continue

        # Calculate the change in the overall discounted sum of torsion terms.
        added_cooccurrence = set(
            [tuple(sorted((feature_1, other))) for other in rest_lattice_0] +
            [tuple(sorted((feature_0, other))) for other in rest_lattice_1])
        removed_cooccurrence = set(
            [tuple(sorted((feature_0, other))) for other in rest_lattice_0] +
            [tuple(sorted((feature_1, other))) for other in rest_lattice_1])
        wash = added_cooccurrence.intersection(removed_cooccurrence)
        added_cooccurrence = added_cooccurrence.difference(wash)
        removed_cooccurrence = removed_cooccurrence.difference(wash)
        swap_diff_torsion = (
            sum(torsions[i][j] * _REPEATED_PAIR_DISCOUNT_IN_CRYSTALS_SCORE**
                cooccurrence_counts[i][j] for (i, j) in added_cooccurrence) -
            sum(torsions[i][j] * _REPEATED_PAIR_DISCOUNT_IN_CRYSTALS_SCORE**
                (cooccurrence_counts[i][j] - 1)
                for (i, j) in removed_cooccurrence))

        # Swap if a feature is repeated or if the score change is positive.
        if (feature_0 not in lattice_1 and feature_1 not in lattice_0 and
            (lattice_0.count(feature_0) > 1 or lattice_1.count(feature_1) > 1 or
             swap_diff_torsion > 0)):
          for (i, j) in added_cooccurrence:
            cooccurrence_counts[i][j] += 1
            cooccurrence_counts[j][i] += 1
          for (i, j) in removed_cooccurrence:
            cooccurrence_counts[i][j] -= 1
            cooccurrence_counts[j][i] -= 1
          lattice_0[index_0], lattice_1[index_1] = (lattice_1[index_1],
                                                    lattice_0[index_0])
          changed = True
  # Return the extracted lattice structure.
  return lattices


def set_crystals_lattice_ensemble(model_config,
                                  prefitting_model_config,
                                  prefitting_model,
                                  feature_names=None):
  """Extracts crystals from a prefitting model and finalizes model_config.

  Args:
    model_config: Model configuration object describing model architecture.
      Should be a `tfl.configs.CalibratedLatticeEnsemble` instance.
    prefitting_model_config: Model configuration object describing prefitting
      model architecture. Should be a `tfl.configs.CalibratedLatticeEnsemble`
      insance constructed using
      `tfl.premade_lib.construct_prefitting_model_config`.
    prefitting_model: A trained `tfl.premade.CalibratedLatticeEnsemble`,
      `tfl.estimators.CannedEstimator`, `tfl.estimators.CannedClassifier`, or
      `tfl.estiamtors.CannedRegressor` instance.
    feature_names: A list of feature names. If not provided, feature names will
      be extracted from the feature configs contained in the model_config.
  """
  # Error checking parameter types.
  if not isinstance(model_config, configs.CalibratedLatticeEnsembleConfig):
    raise ValueError(
        'model_config must be a tfl.configs.CalibratedLatticeEnsembleConfig: {}'
        .format(type(model_config)))
  if not isinstance(prefitting_model_config,
                    configs.CalibratedLatticeEnsembleConfig):
    raise ValueError('prefitting_model_config must be a '
                     'tfl.configs.CalibratedLatticeEnsembleConfig: {}'.format(
                         type(model_config)))
  if model_config.lattices != 'crystals':
    raise ValueError('model_config.lattices must be set to \'crystals\'.')
  # Note that we cannot check the type of the prefitting model without importing
  # premade/estimators, which would cause a cyclic dependency. However, we can
  # check that the model is a keras.Model or tf.Estimator instance that has
  # the proper input layers matching prefitting_model_config feature_configs.
  # Beyond that, a prefitting_model with proper input layer names that is not of
  # the proper type will have undefined behavior.
  # To perform this check, we must first extract feature names if they are not
  # provided, which we need for later steps anyway.
  feature_names = _canonical_feature_names(model_config, feature_names)
  _verify_prefitting_model(prefitting_model, feature_names)

  # Now we can extract the crystals and finalize model_config.
  lattices = _get_final_crystal_lattices(
      model_config=model_config,
      prefitting_model_config=prefitting_model_config,
      prefitting_model=prefitting_model,
      feature_names=feature_names)
  model_config.lattices = [[
      feature_names[features_index] for features_index in lattice
  ] for lattice in lattices]


def _weighted_quantile(sorted_values, quantiles, weights):
  """Calculates weighted quantiles of the given sorted and unique values."""
  if len(sorted_values) < len(quantiles):
    raise ValueError(
        'Not enough unique values ({}) to calculate {} quantiles.'.format(
            len(sorted_values), len(quantiles)))
  # Weighted quantiles of the observed (sorted) values.
  # Weights are spread equaly before and after the observed values.
  weighted_quantiles = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights)

  # Use linear interpolation to find index of the quantile values.
  index_values = np.arange(len(sorted_values))
  quantiles_idx = np.interp(x=quantiles, xp=weighted_quantiles, fp=index_values)
  quantiles_idx = np.rint(quantiles_idx).astype(int)

  # Replace repeated quantile values with neighbouring values.
  unique_idx, first_use = np.unique(quantiles_idx, return_index=True)
  used_idx = set(unique_idx)
  num_values = len(sorted_values)
  for i in range(len(quantiles_idx)):
    if i not in first_use:
      # Since this is not the first use of a (repeated) quantile value, we will
      # need to find an unused neighbouring value.
      for delta, direction in itertools.product(range(1, num_values), [-1, 1]):
        candidate_idx = quantiles_idx[i] + direction * delta
        if (candidate_idx >= 0 and candidate_idx < num_values and
            candidate_idx not in used_idx):
          used_idx.add(candidate_idx)
          quantiles_idx[i] = candidate_idx
          break
  quantiles_idx = np.sort(quantiles_idx)

  return sorted_values[quantiles_idx]


def compute_keypoints(values,
                      num_keypoints,
                      keypoints='quantiles',
                      clip_min=None,
                      clip_max=None,
                      default_value=None,
                      weights=None,
                      weight_reduction='mean',
                      feature_name=''):
  """Calculates keypoints for the given set of values.

  Args:
    values: Values to use for quantile calculation.
    num_keypoints: Number of keypoints to compute.
    keypoints: String `'quantiles'` or `'uniform'`.
    clip_min: Input values are lower clipped by this value.
    clip_max: Input values are upper clipped by this value.
    default_value: If provided, occurances will be removed from values.
    weights: Weights to be used for quantile calculation.
    weight_reduction: Reduction applied to weights for repeated values. Must be
      either 'mean' or 'sum'.
    feature_name: Name to use for error logs.

  Returns:
    A list of keypoints of `num_keypoints` length.
  """
  # Remove default values before calculating stats.
  non_default_idx = values != default_value
  values = values[non_default_idx]
  if weights is not None:
    weights = weights[non_default_idx]

  # Clip min and max if requested. Note that we add clip bounds to the values
  # so that the first and last keypoints are set to those values.
  if clip_min is not None:
    values = np.maximum(values, clip_min)
    values = np.append(values, clip_min)
    if weights is not None:
      weights = np.append(weights, 0)
  if clip_max is not None:
    values = np.minimum(values, clip_max)
    values = np.append(values, clip_max)
    if weights is not None:
      weights = np.append(weights, 0)

  # We do not allow nans in the data, even as default_value.
  if np.isnan(values).any():
    raise ValueError(
        'NaN values were observed for numeric feature `{}`. '
        'Consider replacing the values in transform or input_fn.'.format(
            feature_name))

  # Remove duplicates and sort value before calculating stats.
  # This is emperically useful as we use of keypoints more efficiently.
  if weights is None:
    sorted_values = np.unique(values)
  else:
    # First sort the values and reorder weights.
    idx = np.argsort(values)
    values = values[idx]
    weights = weights[idx]

    # Set the weight of each unique element to be the sum or average of the
    # weights of repeated instances. Using 'mean' reduction results in parity
    # between unweighted calculation and having equal weights for all values.
    sorted_values, idx, counts = np.unique(
        values, return_index=True, return_counts=True)
    weights = np.add.reduceat(weights, idx)
    if weight_reduction == 'mean':
      weights = weights / counts
    elif weight_reduction != 'sum':
      raise ValueError('Invalid weight reduction: {}'.format(weight_reduction))

  if keypoints == 'quantiles':
    if sorted_values.size < num_keypoints:
      logging.info(
          'Not enough unique values observed for feature `%s` to '
          'construct %d keypoints for pwl calibration. Using %d unique '
          'values as keypoints.', feature_name, num_keypoints,
          sorted_values.size)
      return sorted_values.astype(float)

    quantiles = np.linspace(0., 1., num_keypoints)
    if weights is not None:
      return _weighted_quantile(
          sorted_values=sorted_values, quantiles=quantiles,
          weights=weights).astype(float)
    else:
      return np.quantile(
          sorted_values, quantiles, interpolation='nearest').astype(float)

  elif keypoints == 'uniform':
    return np.linspace(sorted_values[0], sorted_values[-1], num_keypoints)
  else:
    raise ValueError('Invalid keypoint generation mode: {}'.format(keypoints))


def _feature_config_by_name(feature_configs, feature_name, add_if_missing):
  """Returns feature_config with the given name."""
  for feature_config in feature_configs:
    if feature_config.name == feature_name:
      return feature_config
  # Use the default FeatureConfig if not present.
  feature_config = configs.FeatureConfig(feature_name)
  if add_if_missing:
    feature_configs.append(feature_config)
  return feature_config


def compute_feature_keypoints(feature_configs,
                              features,
                              weights=None,
                              weight_reduction='mean'):
  """Computes feature keypoints with the data provide in `features` dict."""
  # Calculate feature keypoitns.
  feature_keypoints = {}
  for feature_name, values in six.iteritems(features):
    feature_config = _feature_config_by_name(
        feature_configs=feature_configs,
        feature_name=feature_name,
        add_if_missing=False)

    if feature_config.num_buckets:
      # Skip categorical features.
      continue
    if isinstance(feature_config.pwl_calibration_input_keypoints, str):
      feature_keypoints[feature_name] = compute_keypoints(
          values,
          num_keypoints=feature_config.pwl_calibration_num_keypoints,
          keypoints=feature_config.pwl_calibration_input_keypoints,
          clip_min=feature_config.pwl_calibration_clip_min,
          clip_max=feature_config.pwl_calibration_clip_max,
          default_value=feature_config.default_value,
          weights=weights,
          weight_reduction=weight_reduction,
          feature_name=feature_name,
      )
    else:
      # User-specified keypoint values.
      feature_keypoints[
          feature_name] = feature_config.pwl_calibration_input_keypoints
  return feature_keypoints


def set_feature_keypoints(feature_configs, feature_keypoints,
                          add_missing_feature_configs):
  """Updates the feature configs with provided keypoints."""
  for feature_name, keypoints in six.iteritems(feature_keypoints):
    feature_config = _feature_config_by_name(
        feature_configs=feature_configs,
        feature_name=feature_name,
        add_if_missing=add_missing_feature_configs)
    feature_config.pwl_calibration_input_keypoints = keypoints


def compute_label_keypoints(model_config,
                            labels,
                            logits_output,
                            weights=None,
                            weight_reduction='mean'):
  """Computes label keypoints with the data provide in `lables` array."""
  if not np.issubdtype(labels[0], np.number):
    # Default feature_values to [0, ... n_class-1] for string labels.
    labels = np.arange(len(set(labels)))
    weights = None

  if isinstance(model_config.output_initialization, str):
    # If model is expected to produce logits, initialize linearly in the
    # range [-2, 2], ignoring the label distribution.
    if logits_output:
      return np.linspace(-2, 2, model_config.output_calibration_num_keypoints)

    return compute_keypoints(
        labels,
        num_keypoints=model_config.output_calibration_num_keypoints,
        keypoints=model_config.output_initialization,
        clip_min=model_config.output_min,
        clip_max=model_config.output_max,
        weights=weights,
        weight_reduction=weight_reduction,
        feature_name='label',
    )
  else:
    # User-specified keypoint values.
    return model_config.output_initialization


def set_label_keypoints(model_config, label_keypoints):
  """Updates the label keypoints in the `model_config`."""
  model_config.output_initialization = label_keypoints


def _verify_ensemble_config(model_config):
  """Verifies that an ensemble model and feature configs are properly specified.

  Args:
    model_config: Model configuration object describing model architecture.
      Should be one of the model configs in `tfl.configs`.

  Raises:
    ValueError: If `model_config.lattices` is set to 'rtl_layer' and
      `model_config.num_lattices` is not specified.
    ValueError: If `model_config.num_lattices < 2`.
    ValueError: If `model_config.lattices` is set to 'rtl_layer' and
      `lattice_size` is not the same for all features.
    ValueError: If `model_config.lattices` is set to 'rtl_layer' and
      there are features with unimodality constraints.
    ValueError: If `model_config.lattices` is set to 'rtl_layer' and
      there are features with trust constraints.
    ValueError: If `model_config.lattices` is set to 'rtl_layer' and
      there are features with dominance constraints.
    ValueError: If `model_config.lattices` is set to 'rtl_layer' and
      there are per-feature lattice regularizers.
    ValueError: If `model_config.lattices` is not iterable or constaints
      non-string values.
    ValueError: If `model_config.lattices` is not set to 'rtl_layer' or a fully
      specified list of lists of feature names.
  """
  if model_config.lattices == 'rtl_layer':
    # RTL must have num_lattices specified and >= 2.
    if model_config.num_lattices is None:
      raise ValueError('model_config.num_lattices must be specified when '
                       'model_config.lattices is set to \'rtl_layer\'.')
    if model_config.num_lattices < 2:
      raise ValueError(
          'CalibratedLatticeEnsemble must have >= 2 lattices. For single '
          'lattice models, use CalibratedLattice instead.')
    # Check that all lattices sizes for all features are the same.
    if any(feature_config.lattice_size !=
           model_config.feature_configs[0].lattice_size
           for feature_config in model_config.feature_configs):
      raise ValueError('RTL Layer must have the same lattice size for all '
                       'features.')
    # Check that there are only monotonicity and bound constraints.
    if any(
        feature_config.unimodality != 'none' and feature_config.unimodality != 0
        for feature_config in model_config.feature_configs):
      raise ValueError(
          'RTL Layer does not currently support unimodality constraints.')
    if any(feature_config.reflects_trust_in is not None
           for feature_config in model_config.feature_configs):
      raise ValueError(
          'RTL Layer does not currently support trust constraints.')
    if any(feature_config.dominates is not None
           for feature_config in model_config.feature_configs):
      raise ValueError(
          'RTL Layer does not currently support dominance constraints.')
    # Check that there are no per-feature lattice regularizers.
    for feature_config in model_config.feature_configs:
      for regularizer_config in feature_config.regularizer_configs or []:
        if not regularizer_config.name.startswith(
            _INPUT_CALIB_REGULARIZER_PREFIX):
          raise ValueError(
              'RTL Layer does not currently support per-feature lattice '
              'regularizers.')
  elif isinstance(model_config.lattices, list):
    # Make sure there are more than one lattice. If not, tell user to use
    # CalibratedLattice instead.
    if len(model_config.lattices) < 2:
      raise ValueError(
          'CalibratedLatticeEnsemble must have >= 2 lattices. For single '
          'lattice models, use CalibratedLattice instead.')
    for lattice in model_config.lattices:
      if (not np.iterable(lattice) or
          any(not isinstance(x, str) for x in lattice)):
        raise ValueError(
            'Lattices are not fully specified for ensemble config.')
  else:
    raise ValueError(
        'Lattices are not fully specified for ensemble config. Lattices must '
        'be set to \'rtl_layer\' or be fully specified as a list of lists of '
        'feature names.')


def _verify_kronecker_factored_config(model_config):
  """Verifies that a kronecker_factored model_config is properly specified.

  Args:
    model_config: Model configuration object describing model architecture.
      Should be one of the model configs in `tfl.configs`.

  Raises:
    ValueError: If there are lattice regularizers.
    ValueError: If there are per-feature lattice regularizers.
    ValueError: If there are unimodality constraints.
    ValueError: If there are trust constraints.
    ValueError: If there are dominance constraints.
  """
  for regularizer_config in model_config.regularizer_configs or []:
    if not regularizer_config.name.startswith(_INPUT_CALIB_REGULARIZER_PREFIX):
      raise ValueError(
          'KroneckerFactoredLattice layer does not currently support '
          'lattice regularizers.')
  for feature_config in model_config.feature_configs:
    for regularizer_config in feature_config.regularizer_configs or []:
      if not regularizer_config.name.startswith(
          _INPUT_CALIB_REGULARIZER_PREFIX):
        raise ValueError(
            'KroneckerFactoredLattice layer does not currently support '
            'per-feature lattice regularizers.')
  # Check that all lattices sizes for all features are the same.
  if any(feature_config.lattice_size !=
         model_config.feature_configs[0].lattice_size
         for feature_config in model_config.feature_configs):
    raise ValueError('KroneckerFactoredLattice layer must have the same '
                     'lattice size for all features.')
  # Check that there are only monotonicity and bound constraints.
  if any(
      feature_config.unimodality != 'none' and feature_config.unimodality != 0
      for feature_config in model_config.feature_configs):
    raise ValueError(
        'KroneckerFactoredLattice layer does not currently support unimodality '
        'constraints.')
  if any(feature_config.reflects_trust_in is not None
         for feature_config in model_config.feature_configs):
    raise ValueError(
        'KroneckerFactoredLattice layer does not currently support trust '
        'constraints.')
  if any(feature_config.dominates is not None
         for feature_config in model_config.feature_configs):
    raise ValueError(
        'KroneckerFactoredLattice layer does not currently support dominance '
        'constraints.')


def _verify_aggregate_function_config(model_config):
  """Verifies that an aggregate function model_config is properly specified.

  Args:
    model_config: Model configuration object describing model architecture.
      Should be one of the model configs in `tfl.configs`.

  Raises:
    ValueError: If `middle_dimension < 1`.
    ValueError: If `model_config.middle_monotonicity` is not None and
      `model_config.middle_calibration` is not True.
  """
  if model_config.middle_dimension < 1:
    raise ValueError('Middle dimension must be at least 1: {}'.format(
        model_config.middle_dimension))
  if (model_config.middle_monotonicity is not None and
      not model_config.middle_calibration):
    raise ValueError(
        'middle_calibration must be true when middle_monotonicity is '
        'specified.')


def _verify_feature_config(feature_config):
  """Verifies that feature_config is properly specified.

  Args:
    feature_config: Feature configuration object describing an input feature to
      a model. Should be an instance of `tfl.configs.FeatureConfig`.

  Raises:
    ValueError: If `feature_config.pwl_calibration_input_keypoints` is not
      iterable or contains non-{int/float} values for a numerical feature.
    ValueError: If `feature_config.monotonicity` is not an iterable for a
      categorical feature.
    ValueError: If any element in `feature_config.monotonicity` is not an
      iterable for a categorical feature.
    ValueError: If any value in any element in `feature_config.monotonicity` is
      not an int for a categorical feature.
    ValueError: If any value in any element in `feature_config.monotonicity` is
      not in the range `[0, feature_config.num_buckets]` for a categorical
      feature.
  """
  if not feature_config.num_buckets:
    # Validate PWL Calibration configuration.
    if (not np.iterable(feature_config.pwl_calibration_input_keypoints) or
        any(not isinstance(x, (int, float))
            for x in feature_config.pwl_calibration_input_keypoints)):
      raise ValueError('Input keypoints are invalid for feature {}: {}'.format(
          feature_config.name, feature_config.pwl_calibration_input_keypoints))
  elif feature_config.monotonicity and feature_config.monotonicity != 'none':
    # Validate Categorical Calibration configuration.
    if not np.iterable(feature_config.monotonicity):
      raise ValueError('Monotonicity is not a list for feature {}: {}'.format(
          feature_config.name, feature_config.monotonicity))
    for i, t in enumerate(feature_config.monotonicity):
      if not np.iterable(t):
        raise ValueError(
            'Element {} is not a list/tuple for feature {} monotonicty: {}'
            .format(i, feature_config.name, t))
      for j, val in enumerate(t):
        if not isinstance(val, int):
          raise ValueError(
              'Element {} for list/tuple {} for feature {} monotonicity is '
              'not an index: {}'.format(j, i, feature_config.name, val))
        if val < 0 or val >= feature_config.num_buckets:
          raise ValueError(
              'Element {} for list/tuple {} for feature {} monotonicity is '
              'an invalid index not in range [0, num_buckets - 1]: {}'.format(
                  j, i, feature_config.name, val))


def verify_config(model_config):
  """Verifies that the model_config and feature_configs are properly specified.

  Args:
    model_config: Model configuration object describing model architecture.
      Should be one of the model configs in `tfl.configs`.

  Raises:
    ValueError: If `model_config.feature_configs` is None.
    ValueError: If `model_config.output_initialization` is not iterable or
      contains non-{int/float} values.

  """
  if model_config.feature_configs is None:
    raise ValueError('Feature configs must be fully specified.')
  if isinstance(model_config, configs.CalibratedLatticeEnsembleConfig):
    _verify_ensemble_config(model_config)
  if ((isinstance(model_config, configs.CalibratedLatticeEnsembleConfig) or
       isinstance(model_config, configs.CalibratedLatticeConfig)) and
      model_config.parameterization == 'kronecker_factored'):
    _verify_kronecker_factored_config(model_config)
  if isinstance(model_config, configs.AggregateFunctionConfig):
    _verify_aggregate_function_config(model_config)
  for feature_config in model_config.feature_configs:
    _verify_feature_config(feature_config)
  if (not np.iterable(model_config.output_initialization) or
      any(not isinstance(x, (int, float))
          for x in model_config.output_initialization)):
    raise ValueError('Output initilization is invalid: {}'.format(
        model_config.output_initialization))
