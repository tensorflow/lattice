# Copyright 2017 The TensorFlow Lattice Authors.
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
# ==============================================================================
"""CalibratedRtl canned estimators."""
import copy
import os
import random

# Dependency imports

import six

from tensorflow_lattice.python.estimators import calibrated as calibrated_lib
from tensorflow_lattice.python.estimators import hparams as tfl_hparams
from tensorflow_lattice.python.lib import lattice_layers
from tensorflow_lattice.python.lib import regularizers

from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables

_EPSILON = 1e-7

_RTL_STRUCTURE_FILE = 'rtl_structure.csv'


class _CalibratedRtl(calibrated_lib.Calibrated):
  """Base class for CalibratedRtl{Classifier|Regressor}."""

  def __init__(self,
               n_classes,
               feature_columns=None,
               model_dir=None,
               quantiles_dir=None,
               keypoints_initializers_fn=None,
               lattice_initializers_fn=None,
               optimizer=None,
               config=None,
               hparams=None,
               head=None,
               weight_column=None):
    """Construct CalibrateRtlClassifier/Regressor."""
    if not hparams:
      hparams = tfl_hparams.CalibratedRtlHParams([])
    self.check_hparams(hparams)
    hparams = self._adjust_calibration_params(hparams)

    self.lattice_initializers_fn_ = lattice_initializers_fn

    super(_CalibratedRtl,
          self).__init__(n_classes, feature_columns, model_dir, quantiles_dir,
                         keypoints_initializers_fn, optimizer, config, hparams,
                         head, weight_column, 'rtl')
    self._structure_file = os.path.join(self._model_dir, _RTL_STRUCTURE_FILE)

  def _check_per_feature_param_configuration(
      self, adjusted, monotonicity, lattice_size, calibration_output_min,
      calibration_output_max, calibration_bound, missing_input_value,
      missing_vertex):
    """Check parameter configuration and returns the error messages."""
    error_messages = []
    if monotonicity not in {-1, 0, +1}:
      error_messages.append('monotonicity should be an integer {-1, 0, +1} '
                            'but is %s' % monotonicity)

    if lattice_size < 2:
      error_messages.append('lattice_size should be greater than equal to 2'
                            'but is %d' % (lattice_size))

    if not calibration_bound:
      error_messages.append(
          'A lattice expects an bounded input from a calibration layer, but '
          'calibration_bound is set to be False')

    if not adjusted:
      if calibration_output_min is not None:
        error_messages.append(
            'calibration_output_min=%d should not be set, it is adjusted '
            'automatically to match the lattice_size' % calibration_output_min)
      if calibration_output_max is not None:
        error_messages.append(
            'calibration_output_max=%d should not be set, it is adjusted '
            'automatically to match the lattice_size' % calibration_output_max)

    if missing_input_value is None and missing_vertex:
      error_messages.append(
          'missing_vertex is True, however missing_input_value not set')

    return error_messages

  def _check_not_allowed_feature_params(self, hparams):
    """Check hparams contains feature-level value that are not allowed.

    Certain values cannot be feature-level hyperparameters. This function checks
    whether any of feature sets hparams that are not allowed to be feature-level
    hyperparameter, and returns non-empty error messages if there is an error.

    Args:
      hparams: (CalibratedRtlHparams) hyperparameters needs to be checked.
    Returns:
      error_messages: (list of strings) error messages.
    """
    not_allowed_feature_params = map(
        'lattice_{}'.format,
        regularizers.LATTICE_MULTI_DIMENSIONAL_REGULARIZERS)
    error_messages = []
    for param in not_allowed_feature_params:
      for feature_name in hparams.get_feature_names():
        if hparams.is_feature_set_param(feature_name, param):
          error_messages.append('feature %s sets %s, which is not allowed.' %
                                (feature_name, param))
    return error_messages

  def check_hparams(self, hparams, adjusted=False):
    """Check pre-conditions of hparams.

    Args:
      hparams: (tfl_hparams.CalibratedRtlHParams) Hyperparameter to be
      examined.
      adjusted: if these are the parameters already adjusted. For example,
        calibrator_output_min and max should be adjusted so that the output is
        in [0, lattice_size - 1] (or [0, lattice_size] if missing_vertex
        == True) and calibrator bound should set to be true, etc.
        If adjust is True, we will check that all the parameter values is valid,
        otherwise, some checks will be skipped.
    Raises:
      ValueError: If the hyperparameter configuration is invalid, for example
      calibration_monotonic is None, but lattice_monotonic is True, then raise
      the error with a root cause.
    """
    error_messages = self._check_not_allowed_feature_params(hparams)

    # Check lattice_rank and num_lattices.
    lattice_rank = hparams.get_param('lattice_rank')
    num_lattices = hparams.get_param('num_lattices')
    if lattice_rank is None or num_lattices is None:
      error_messages.append('lattice_rank and num_lattices should be provided')

    # Check global params.
    feature_names = hparams.get_feature_names()
    packed_feature_values = hparams.get_global_and_feature_params([
        'monotonicity', 'lattice_size', 'calibration_output_min',
        'calibration_output_max', 'calibration_bound', 'missing_input_value',
        'missing_vertex'
    ], feature_names)
    default_feature_values, per_feature_values = packed_feature_values
    param_error_messages = self._check_per_feature_param_configuration(
        adjusted, *default_feature_values)
    if param_error_messages:
      error_messages.append('Error message for default feature param:')
      error_messages += param_error_messages

    # Check per feature params. hparams.get_feature_names()  will only return
    # feature names that sets per feature parameters.
    for feature_idx in range(len(per_feature_values)):
      param_error_messages = self._check_per_feature_param_configuration(
          adjusted, *per_feature_values[feature_idx])
      if param_error_messages:
        error_messages.append(
            'Error message for %s feature param:' % feature_names[feature_idx])
        error_messages += param_error_messages

    if error_messages:
      raise ValueError(
          'Hyperparameter configuration cannot be used in the calibrated '
          'rtl estimator. Error messages report the issue per feature, but'
          ' the parameter may be inherited from global parameter.\nDetailed '
          'error messsages\n%s' % '\n'.join(error_messages))

  def _adjust_calibration_params(self, hparams):
    """Adjust the calibration parameters to match the input siz of lattices."""
    hparams = copy.deepcopy(hparams)
    feature_names = hparams.get_feature_names()
    packed_feature_values = hparams.get_global_and_feature_params(
        ['lattice_size', 'missing_input_value', 'missing_vertex'],
        feature_names)
    default_feature_values, per_feature_values = packed_feature_values
    final_lattice_size, missing_output_value = self._calibration_params(
        *default_feature_values)
    lattice_size = default_feature_values[0]
    hparams.set_param('calibration_output_min', 0)
    hparams.set_param('calibration_output_max', lattice_size - 1)
    hparams.set_param('final_lattice_size', final_lattice_size)
    hparams.set_param('missing_output_value', missing_output_value)

    if len(per_feature_values) != len(feature_names):
      raise ValueError(
          'length of per_feature_value (%d) != length of feature_names (%d)' %
          (len(per_feature_values), len(feature_names)))
    for (per_feature_value, feature_name) in zip(per_feature_values,
                                                 feature_names):
      final_lattice_size, missing_output_value = self._calibration_params(
          *per_feature_value)
      lattice_size = per_feature_value[0]
      hparams.set_feature_param(feature_name, 'calibration_output_min', 0)
      hparams.set_feature_param(feature_name, 'calibration_output_max',
                                lattice_size - 1)
      hparams.set_feature_param(feature_name, 'final_lattice_size',
                                final_lattice_size)
      hparams.set_feature_param(feature_name, 'missing_output_value',
                                missing_output_value)
    return hparams

  def _calibration_params(self, lattice_size, missing_input_value,
                          missing_vertex):
    """Returns final_lattice_size and missing_output_value."""
    if missing_input_value is None or not missing_vertex:
      return lattice_size, None

    # Last vertex of the lattice is reserved for missing values.
    return lattice_size + 1, lattice_size

  def _load_structure(self):
    """Load rtl structure from model_dir."""
    if not file_io.file_exists(self._structure_file):
      raise ValueError(
          'Structure file does not exists in %s!' % self._structure_file)
    structure_csv_string = file_io.read_file_to_string(self._structure_file)
    structure_csvs = structure_csv_string.split('\n')
    structure = []
    for structure_csv in structure_csvs:
      structure.append([int(idx) for idx in structure_csv.split(',')])
    return structure

  def _save_structure(self, structure):
    """Save rtl structure to model_dir."""
    structure_csvs = []
    for lattice in structure:
      structure_csvs.append(','.join([str(idx) for idx in lattice]))
    structure_csv_string = '\n'.join(structure_csvs)
    file_io.write_string_to_file(self._structure_file, structure_csv_string)


  def _create_structure(self, input_dim, num_lattices, lattice_rank, rtl_seed):
    """Create and save rtl structure to model_dir."""
    rtl_random = random.Random(rtl_seed)
    structure = []
    for _ in range(num_lattices):
      structure.append(
          rtl_random.sample(six.moves.xrange(input_dim), lattice_rank))
    return structure

  def calibration_structure_builder(self, columns_to_tensors, hparams):
    """Returns the calibration structure of the model. See base class."""
    return None

  def prediction_builder_from_calibrated(
      self, mode, per_dimension_feature_names, hparams, calibrated):
    """Construct the prediciton."""

    self.check_hparams(hparams, adjusted=True)
    lattice_sizes = [
        hparams.get_feature_param(f, 'final_lattice_size')
        for f in per_dimension_feature_names
    ]
    lattice_monotonic = [(hparams.get_feature_param(f, 'monotonicity') != 0)
                         for f in per_dimension_feature_names]
    num_lattices = hparams.get_param('num_lattices')
    lattice_rank = hparams.get_param('lattice_rank')
    rtl_seed = hparams.get_param('rtl_seed')
    interpolation_type = hparams.get_param('interpolation_type')
    # Create and save structure if it does not exists.
    if not file_io.file_exists(self._structure_file):
      structure = self._create_structure(
          len(lattice_sizes), num_lattices, lattice_rank, rtl_seed)
      self._save_structure(structure)
    structure = self._load_structure()
    # Check structure is what we expect.
    if len(structure) != num_lattices:
      raise ValueError(
          'Expect %d number of lattices, but found %d number of lattices in '
          'structure: %s' % (num_lattices, len(structure), str(structure)))
    for each_lattice in structure:
      if len(each_lattice) != lattice_rank:
        raise ValueError('Expect %d lattice rank, but found %d in structure: %s'
                         % (lattice_rank, len(each_lattice), str(structure)))

    # Setup the regularization.
    regularizer_amounts = {}
    for reg_name in regularizers.LATTICE_MULTI_DIMENSIONAL_REGULARIZERS:
      regularizer_amounts[reg_name] = hparams.get_param(
          'lattice_{}'.format(reg_name))
    for reg_name in regularizers.LATTICE_ONE_DIMENSIONAL_REGULARIZERS:
      regularizer_amounts[reg_name] = [
          hparams.get_feature_param(feature_name, 'lattice_{}'.format(reg_name))
          for feature_name in per_dimension_feature_names
      ]

    packed_results = lattice_layers.ensemble_lattices_layer(
        calibrated,
        lattice_sizes,
        structure,
        is_monotone=lattice_monotonic,
        interpolation_type=interpolation_type,
        lattice_initializers=self.lattice_initializers_fn_,
        **regularizer_amounts)
    (output_tensors, _, projection_ops, regularization) = packed_results
    # Take an average of output_tensors and add bias.
    output_tensor = array_ops.stack(
        output_tensors, axis=0, name='stacked_output')
    ensemble_output = math_ops.reduce_mean(output_tensor, axis=0)
    ensemble_bias_init = hparams.get_param('ensemble_bias')
    b = variables.Variable([ensemble_bias_init], name='ensemble_bias')
    prediction = ensemble_output + b

    # Returns prediction Tensor, projection ops, and regularization.
    return prediction, projection_ops, regularization


def calibrated_rtl_classifier(feature_columns=None,
                              model_dir=None,
                              quantiles_dir=None,
                              keypoints_initializers_fn=None,
                              optimizer=None,
                              config=None,
                              hparams=None,
                              head=None,
                              weight_column=None):
  """Calibrated rtl binary classifier model.



  This model uses a piecewise lattice calibration function on each of the
  inputs (parametrized) and then feeds them to ensemble of random lattices.
  num_lattices and lattice_rank (number of inputs to each lattice) must be
  specified in the hyperparameter. Optionally calibration can be made monotonic.

  It usually requires a preprocessing step on the data, to calculate the
  quantiles of each used feature. This can be done locally or in one worker
  only before training, in a separate invocation of your program (or directly).
  Typically this can be saved (`save_dir` parameter) to the same
  directory where the data is.

  Hyper-parameters are given in the form of the object
  tfl_hparams.CalibrationRtlHParams. lattice_rank and num_lattices must
  be specified; there would be no default value for this. It also takes in
  per-feature parameters.

  Internally values will be converted to tf.float32.

  Example:

  ```python
  def input_fn_train: ...
  def input_fn_eval: ...

  my_feature_columns=[...]

  # Have a separate program flag to generate the quantiles. Need to be run
  # only once.
  if FLAGS.create_quantiles:
    pwl_calibrators_layers.calculate_quantiles_for_keypoints(
      input_fn=input_fn_train,
      feature_columns=my_feature_columns,
      save_dir=FLAGS.data_dir,
      num_quantiles=1000,
      override=True)
    return  # Exit program.

  hparams = hparams.CalibratedRtlHparams(num_lattices=10, lattice_rank=2)
  estimator = calibrated_rtl.calibrated_rtl_classifier(
    feature_columns=feature_columns, hparams=hparams)
  estimator.train(input_fn=input_fn_train)
  estimator.evaluate(input_fn=input_fn_eval)
  estimator.predict(input_fn=input_fn_predict)
  ```

  Args:
    feature_columns: Optional, an iteratable containing all the feature
      columns used by the model. All items in the set should be instances of
      classes derived from `FeatureColumn`. If not given, the model will
      use as features the tensors returned by input_fn.
      Supported types of columns: RealValuedColumn.
    model_dir: Directory to save model parameters, graphs and etc. This can
      also be used to load checkpoints from the directory into an estimator to
      continue training a previously saved model.
    quantiles_dir: location where quantiles for the data was saved. Typically
      the same directory as the training data. These quantiles can be
      generated only once with
      `pwl_calibration_layers.calculate_quantiles_for_keypoints` in a separate
      invocation of your program. If you don't want to use quantiles, you can
      set `keypoints_initializer` instead.
    keypoints_initializers_fn: if you know the distribution of your
      input features you can provide that directly instead of `quantiles_dir`.
      See `pwl_calibrators_layers.uniform_keypoints_for_signal`. It must be
      a closure that returns a pair of tensors with keypoints inputs and
      outputs to use for initialization (must match `num_keypoints` configured
      in `hparams`). Alternatively the closure can return a dict mapping
      feature name to pairs for initialization per feature. If `quantiles_dir`
      and `keypoints_initializers_fn` are set, the later takes precendence,
      and the features for which `keypoints_initializers` are not defined
      fallback to using the quantiles found in `quantiles_dir`. It uses a
      closure instead of the tensors themselves because the graph has to be
      created at the time the model is being build, which happens at a later
      time.
    optimizer: string, `Optimizer` object, or callable that defines the
      optimizer to use for training -- if a callable, it will be called with
      learning_rate=hparams.learning_rate.
    config: RunConfig object to configure the runtime settings. Typically set
      to learn_runner.EstimatorConfig().
    hparams: an instance of tfl_hparams.CalibrationRtlHParams. If set to
      None default parameters are used.
    head: a `TensorFlow Estimator Head` which specifies how the loss function,
      final predictions, and so on are generated from model outputs. Defaults
      to using a sigmoid cross entropy head for binary classification and mean
      squared error head for regression.
    weight_column: A string or a `tf.feature_column.numeric_column` defining
      feature column representing weights. It is used to down weight or boost
      examples during training. It will be multiplied by the loss of the
      example.

  Returns:
    A `calibrated_rtl_classifier` estimator.

  Raises:
    ValueError: invalid parameters.
    KeyError: type of feature not supported.
  """
  return _CalibratedRtl(
      n_classes=2,
      feature_columns=feature_columns,
      model_dir=model_dir,
      quantiles_dir=quantiles_dir,
      keypoints_initializers_fn=keypoints_initializers_fn,
      optimizer=optimizer,
      config=config,
      hparams=hparams,
      head=head,
      weight_column=weight_column)


def calibrated_rtl_regressor(feature_columns=None,
                             model_dir=None,
                             quantiles_dir=None,
                             keypoints_initializers_fn=None,
                             optimizer=None,
                             config=None,
                             hparams=None,
                             head=None,
                             weight_column=None):
  """Calibrated rtl regressor model.

  This model uses a piecewise lattice calibration function on each of the
  inputs (parametrized) and then feeds them to ensemble of random lattices.
  num_lattices and lattice_rank (number of inputs to each lattice) must be
  specified in the hyperparameter. Optionally calibration can be made monotonic.

  It usually requires a preprocessing step on the data, to calculate the
  quantiles of each used feature. This can be done locally or in one worker
  only before training, in a separate invocation of your program (or directly).
  Typically this can be saved (`save_dir` parameter) to the same
  directory where the data is.

  Hyper-parameters are given in the form of the object
  tfl_hparams.CalibrationRtlHParams. lattice_rank and num_lattices must
  be specified; there would be no default value for this. It also takes in
  per-feature parameters.

  Internally values will be converted to tf.float32.

  Example:

  ```python
  def input_fn_train: ...
  def input_fn_eval: ...

  my_feature_columns=[...]

  # Have a separate program flag to generate the quantiles. Need to be run
  # only once.
  if FLAGS.create_quantiles:
    pwl_calibrators_layers.calculate_quantiles_for_keypoints(
      input_fn=input_fn_train,
      feature_columns=my_feature_columns,
      save_dir=FLAGS.data_dir,
      num_quantiles=1000,
      override=True)
    return  # Exit program.

  hparams = hparams.CalibratedRtlHparams(num_lattices=10, lattice_rank=2)
  estimator = calibrated_rtl.calibrated_rtl_classifier(
    feature_columns=feature_columns, hparams=hparams)
  estimator.train(input_fn=input_fn_train)
  estimator.evaluate(input_fn=input_fn_eval)
  estimator.predict(input_fn=input_fn_test)
  ```

  Args:
    feature_columns: Optional, an iteratable containing all the feature
      columns used by the model. All items in the set should be instances of
      classes derived from `FeatureColumn`. If not given, the model will
      use as features the tensors returned by input_fn.
      Supported types of columns: RealValuedColumn.
    model_dir: Directory to save model parameters, graphs and etc. This can
      also be used to load checkpoints from the directory into an estimator to
      continue training a previously saved model.
    quantiles_dir: location where quantiles for the data was saved. Typically
      the same directory as the training data. These quantiles can be
      generated only once with
      `pwl_calibration_layers.calculate_quantiles_for_keypoints` in a separate
      invocation of your program. If you don't want to use quantiles, you can
      set `keypoints_initializer` instead.
    keypoints_initializers_fn: if you know the distribution of your
      input features you can provide that directly instead of `quantiles_dir`.
      See `pwl_calibrators_layers.uniform_keypoints_for_signal`. It must be
      a closure that returns a pair of tensors with keypoints inputs and
      outputs to use for initialization (must match `num_keypoints` configured
      in `hparams`). Alternatively the closure can return a dict mapping
      feature name to pairs for initialization per feature. If `quantiles_dir`
      and `keypoints_initializers_fn` are set, the later takes precendence,
      and the features for which `keypoints_initializers` are not defined
      fallback to using the quantiles found in `quantiles_dir`. It uses a
      closure instead of the tensors themselves because the graph has to be
      created at the time the model is being build, which happens at a later
      time.
    optimizer: string, `Optimizer` object, or callable that defines the
      optimizer to use for training -- if a callable, it will be called with
      learning_rate=hparams.learning_rate.
    config: RunConfig object to configure the runtime settings. Typically set
      to learn_runner.EstimatorConfig().
    hparams: an instance of tfl_hparams.CalibrationRtlHParams. If set to
      None default parameters are used.
    head: a `TensorFlow Estimator Head` which specifies how the loss function,
      final predictions, and so on are generated from model outputs. Defaults
      to using a sigmoid cross entropy head for binary classification and mean
      squared error head for regression.
    weight_column: A string or a `tf.feature_column.numeric_column` defining
      feature column representing weights. It is used to down weight or boost
      examples during training. It will be multiplied by the loss of the
      example.

  Returns:
    A `calibrated_rtl_regressor` estimator.

  Raises:
    ValueError: invalid parameters.
    KeyError: type of feature not supported.
  """
  return _CalibratedRtl(
      n_classes=0,
      feature_columns=feature_columns,
      model_dir=model_dir,
      quantiles_dir=quantiles_dir,
      keypoints_initializers_fn=keypoints_initializers_fn,
      optimizer=optimizer,
      config=config,
      hparams=hparams,
      head=head,
      weight_column=weight_column)
