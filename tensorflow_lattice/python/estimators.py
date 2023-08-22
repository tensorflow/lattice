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
"""TF Lattice canned estimators implement typical monotonic model architectures.

You can use TFL canned estimators to easily construct commonly used monotonic
model architectures. To construct a TFL canned estimator, construct a model
configuration from `tfl.configs` and pass it to the canned estimator
constructor. To use automated quantile calculation, canned estimators also
require passing a *feature_analysis_input_fn* which is similar to the one used
for training, but with a single epoch or a subset of the data. To create a
Crystals ensemble model using `tfl.configs.CalibratedLatticeEnsembleConfig`, you
will also need to provide a *prefitting_input_fn* to the estimator constructor.

```python
feature_columns = ...
model_config = tfl.configs.CalibratedLatticeConfig(...)
feature_analysis_input_fn = create_input_fn(num_epochs=1, ...)
train_input_fn = create_input_fn(num_epochs=100, ...)
estimator = tfl.estimators.CannedClassifier(
    feature_columns=feature_columns,
    model_config=model_config,
    feature_analysis_input_fn=feature_analysis_input_fn)
estimator.train(input_fn=train_input_fn)
```

Supported models are defined in `tfl.configs`. Each model architecture can be
used for:

*   **Classification** using `tfl.estimators.CannedClassifier` with standard
    classification head (softmax cross-entropy loss).

*   **Regression** using `tfl.estimators.CannedRegressor` with standard
    regression head (squared loss).

*   **Custom head** using `tfl.estimators.CannedEstimator` with any custom head
    and loss.

This module also provides `tfl.estimators.get_model_graph` as a mechanism to
extract abstract model graphs and layer parameters from saved models. The
resulting graph (not a TF graph) can be used by the `tfl.visualization` module
for plotting and other visualization and analysis.

```python
model_graph = estimators.get_model_graph(saved_model_path)
visualization.plot_feature_calibrator(model_graph, "feature_name")
visualization.plot_all_calibrators(model_graph)
visualization.draw_model_graph(model_graph)
```

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import os
import re
import time

from . import categorical_calibration_layer
from . import configs
from . import kronecker_factored_lattice_layer as kfll
from . import lattice_layer
from . import linear_layer
from . import model_info
from . import premade
from . import premade_lib
from . import pwl_calibration_layer
from . import rtl_layer

from absl import logging
import numpy as np
import six
import tensorflow as tf
from tensorflow import estimator as tf_estimator

from tensorflow.python.feature_column import feature_column as fc  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.feature_column import feature_column_v2 as fc2  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.training import training_util  # pylint: disable=g-direct-tensorflow-import
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.head import binary_class_head
from tensorflow_estimator.python.estimator.head import multi_class_head
from tensorflow_estimator.python.estimator.head import regression_head

# TODO: support multi dim inputs.
# TODO: support multi dim output.
# TODO: add linear layer regularizers.
# TODO: add examples in docs.
# TODO: make _REPEATED_PAIR_DISCOUNT_IN_CRYSTALS_SCORE config param

# Feed and fetch names for the model.
FEATURES_SCOPE = 'features'
OUTPUT_NAME = 'output'

# File to store and load feature keypoints.
_KEYPOINTS_FILE = 'keypoints.json'

# File to store and load lattice ensemble structure.
_ENSEMBLE_STRUCTURE_FILE = 'ensemble_structure.json'

# Name for label keypoints in keypoints file.
_LABEL_FEATURE_NAME = '__label__'

# Pooling interval and maximum wait time for workers waiting for files.
_MAX_WAIT_TIME = 2400
_POLL_INTERVAL_SECS = 10


class WaitTimeOutError(Exception):
  """Timeout error when waiting for a file."""
  pass


def _poll_for_file(filename):
  """Waits and polls for a file until it exists."""
  start = time.time()
  while not tf.io.gfile.exists(filename):
    time.sleep(_POLL_INTERVAL_SECS)
    if time.time() - start > _MAX_WAIT_TIME:
      raise WaitTimeOutError('Waiting for file {} timed-out'.format(filename))


def transform_features(features, feature_columns=None):
  """Parses the input features using the given feature columns.

  This function can be used to parse input features when constructing a custom
  estimator. When using this function, you will not need to wrap categorical
  features with dense feature embeddings, and the resulting tensors will not be
  concatenated, making it easier to use the features in the calibration layers.

  Args:
    features: A dict from feature names to tensors.
    feature_columns: A list of FeatureColumn objects to be used for parsing. If
      not provided, the input features are assumed to be already parsed.

  Returns:
    collections.OrderedDict mapping feature names to parsed tensors.
  """
  with tf.name_scope('transform'):
    if feature_columns:
      parsed_features = collections.OrderedDict()
      for feature_column in feature_columns:
        # pylint: disable=protected-access
        if (isinstance(feature_column, fc._DenseColumn) or
            isinstance(feature_column, fc2.DenseColumn)):
          parsed_features[
              feature_column.name] = feature_column._transform_feature(features)
        elif (isinstance(feature_column, fc._CategoricalColumn) or
              isinstance(feature_column, fc2.CategoricalColumn)):
          if feature_column.num_oov_buckets:
            # If oov buckets are used, missing values are assigned to the last
            # oov bucket.
            default_value = feature_column.num_buckets - 1
          else:
            default_value = feature_column.default_value
          parsed_features[feature_column.name] = tf.reshape(
              tf.sparse.to_dense(
                  sp_input=feature_column._transform_feature(features),
                  default_value=default_value),
              shape=[-1, 1])
        else:
          raise ValueError(
              'Unsupported feature_column: {}'.format(feature_column))
        # pylint: enable=protected-access
    else:
      parsed_features = collections.OrderedDict(features)

    for name, tensor in parsed_features.items():
      if len(tensor.shape) == 1:
        parsed_features[name] = tf.expand_dims(tensor, 1)
      elif len(tensor.shape) > 2 or tensor.shape[1] != 1:
        raise ValueError('Only 1-d inputs are supported: {}'.format(tensor))

  with tf.name_scope(FEATURES_SCOPE):
    for name, tensor in parsed_features.items():
      parsed_features[name] = tf.identity(parsed_features[name], name=name)

  return parsed_features


def _materialize_locally(tensors, max_elements=1e6):
  """Materialize the given tensors locally, during initialization.

  Assumes non-distributed environment (uses SingularMonitoredSession).

  Args:
    tensors: A dict of name to feed tensors to be materialized.
    max_elements: Data is read and accmulated from tensors until end-of-input is
      reached or when we have at least max_elements collected.

  Returns:
    Materialized tensors as dict.
  """
  # tf.compat.v1.train.SingularMonitoredSession silently catches
  # tf.errors.OutOfRangeError, and we want to expose it to detect end of the
  # data from the given feed tensors.
  with tf.compat.v1.train.SingularMonitoredSession() as sess:
    splits = []
    count = 0
    try:
      while count < max_elements:
        materialized_tensors = sess.run(tensors)
        values = list(materialized_tensors.values())
        if not values:
          break
        count += len(values[0])
        splits.append(materialized_tensors)
    except (tf.errors.OutOfRangeError, StopIteration):
      pass
    concatenated_tensors = {}
    for k in tensors:
      k_tensors = [split[k] for split in splits if split[k].size > 0]
      # If k_tensors is empty, the np.concat call below would raise
      # an exception below. So we test this condition here to provide
      # a better error message.
      if not k_tensors:
        raise ValueError('Did not find any values for key: {}'.format(k))
      concatenated_tensors[k] = np.concatenate(k_tensors)
    return concatenated_tensors


def _finalize_keypoints(model_config, config, feature_columns,
                        feature_analysis_input_fn,
                        feature_analysis_weight_column,
                        feature_analysis_weight_reduction, logits_output):
  """Calculates and sets keypoints for input and output calibration.

  Input and label keypoints are calculated, stored in a file and also set in the
  model_config to be used for model construction.

  Args:
    model_config: Model config to be updated.
    config: A `tf.RunConfig` to indicate if worker is chief.
    feature_columns: A list of FeatureColumn's to use for feature parsing.
    feature_analysis_input_fn: An input_fn used to collect feature statistics.
    feature_analysis_weight_column: None or weight column to use.
    feature_analysis_weight_reduction: Reduction applied to weights for repeated
      values. Must be either 'mean' or 'sum'.
    logits_output: A boolean indicating if model outputs logits.

  Raises:
    ValueError: If keypoints mode is invalid.
  """
  if not feature_analysis_input_fn:
    return

  keypoints_filename = os.path.join(config.model_dir, _KEYPOINTS_FILE)
  if ((config is None or config.is_chief) and
      not tf.io.gfile.exists(keypoints_filename)):
    # As the chief worker, calculate and store the keypoints.
    with tf.Graph().as_default():
      ds = feature_analysis_input_fn()
      if isinstance(ds, tf.data.Dataset):
        ds = tf.compat.v1.data.make_one_shot_iterator(ds).get_next()
      features, labels = ds

      # Add weights and labels to the feature list to be materialized.
      if feature_analysis_weight_column is not None:
        if isinstance(feature_analysis_weight_column, str):
          feature_analysis_weight_column = tf.feature_column.numeric_column(
              feature_analysis_weight_column)
        feature_columns = feature_columns + [feature_analysis_weight_column]

      values = transform_features(features, feature_columns)
      values[_LABEL_FEATURE_NAME] = labels
      values = _materialize_locally(values)
      for feature_name in values:
        values[feature_name] = values[feature_name].flatten()

      # Pop out label and weight values.
      labels = values.pop(_LABEL_FEATURE_NAME)
      if feature_analysis_weight_column is not None:
        weights = values.pop(feature_analysis_weight_column.name)
      else:
        weights = None

    keypoints = premade_lib.compute_feature_keypoints(
        feature_configs=model_config.feature_configs,
        features=values,
        weights=weights,
        weight_reduction=feature_analysis_weight_reduction)
    keypoints[_LABEL_FEATURE_NAME] = premade_lib.compute_label_keypoints(
        model_config=model_config,
        labels=labels,
        logits_output=logits_output,
        weights=weights,
        weight_reduction=feature_analysis_weight_reduction)

    # Convert to python float type for serialization.
    keypoints = {
        k: [float(v) for v in vs] for k, vs in six.iteritems(keypoints)
    }

    # Save keypoints to file as the chief worker.
    tmp_keypoints_filename = keypoints_filename + 'tmp'
    with tf.io.gfile.GFile(tmp_keypoints_filename, 'w') as keypoints_file:
      keypoints_file.write(json.dumps(keypoints, indent=2))
    tf.io.gfile.rename(tmp_keypoints_filename, keypoints_filename)
  else:
    # Non-chief workers read the keypoints from file.
    _poll_for_file(keypoints_filename)
    with tf.io.gfile.GFile(keypoints_filename) as keypoints_file:
      keypoints = json.loads(keypoints_file.read())

  # Set the keypoint values in the model config.
  label_keypoints = keypoints.pop(_LABEL_FEATURE_NAME)
  premade_lib.set_feature_keypoints(
      feature_configs=model_config.feature_configs,
      feature_keypoints=keypoints,
      add_missing_feature_configs=True)
  premade_lib.set_label_keypoints(
      model_config=model_config, label_keypoints=label_keypoints)


def _fix_ensemble_for_2d_constraints(model_config, feature_names):
  """Fixes 2d constraint violations by adding missing features to some lattices.

  Some 2d shape constraints require lattices from ensemble to either contain
  both constrained features or none of them, e.g. trapezoid trust constraint
  requires a lattice that has the "conditional" feature to include the "main"
  feature.

  Args:
    model_config: Model config to be updated.
    feature_names: List of feature names.
  """
  must_include_features = collections.defaultdict(set)
  for feature_name in feature_names:
    feature_config = model_config.feature_config_by_name(feature_name)
    for trust_config in feature_config.reflects_trust_in or []:
      if trust_config.trust_type == 'trapezoid':
        must_include_features[feature_name].add(trust_config.feature_name)
    for dominance_config in feature_config.dominates or []:
      must_include_features[dominance_config.feature_name].add(feature_name)

  fixed_lattices = []
  for idx, lattice in enumerate(model_config.lattices):
    fixed_lattice = set()
    for feature_name in lattice:
      fixed_lattice.add(feature_name)
      fixed_lattice.update(must_include_features[feature_name])
    assert len(lattice) <= len(fixed_lattice)
    fixed_lattices.append(list(fixed_lattice))
    if len(lattice) < len(fixed_lattice):
      logging.info(
          'Fixed 2d constraint violations in lattices[%d]. Lattice rank '
          'increased from %d to %d.', idx, len(lattice), len(fixed_lattice))

  model_config.lattices = fixed_lattices


def _set_crystals_lattice_ensemble(model_config, feature_names, label_dimension,
                                   feature_columns, head, prefitting_input_fn,
                                   prefitting_optimizer, prefitting_steps,
                                   config, dtype):
  """Sets the lattice ensemble in model_config using the crystals algorithm."""
  if prefitting_input_fn is None:
    raise ValueError('prefitting_input_fn must be set for crystals models')

  # Get prefitting model config.
  prefitting_model_config = premade_lib.construct_prefitting_model_config(
      model_config, feature_names)

  def prefitting_model_fn(features, labels, mode, config):
    return _calibrated_lattice_ensemble_model_fn(
        features=features,
        labels=labels,
        label_dimension=label_dimension,
        feature_columns=feature_columns,
        mode=mode,
        head=head,
        model_config=prefitting_model_config,
        optimizer=prefitting_optimizer,
        config=config,
        dtype=dtype)

  config = tf_estimator.RunConfig(
      keep_checkpoint_max=1,
      save_summary_steps=0,
      save_checkpoints_steps=10000000,
      tf_random_seed=config.tf_random_seed if config is not None else 42)
  logging.info('Creating the prefitting estimator.')
  prefitting_estimator = tf_estimator.Estimator(
      model_fn=prefitting_model_fn, config=config)
  logging.info('Training the prefitting estimator.')
  prefitting_estimator.train(
      input_fn=prefitting_input_fn, steps=prefitting_steps)
  premade_lib.set_crystals_lattice_ensemble(
      model_config=model_config,
      prefitting_model_config=prefitting_model_config,
      prefitting_model=prefitting_estimator,
      feature_names=feature_names)
  logging.info('Finished training the prefitting estimator.')

  # Cleanup model_dir since we might be reusing it for the main estimator.
  # Note that other workers are blocked until model structure file is
  # generated by the chief worker, so modifying files here should be safe.
  remove_list = [
      os.path.join(prefitting_estimator.model_dir, 'graph.pbtxt'),
      os.path.join(prefitting_estimator.model_dir, 'checkpoint'),
  ]
  remove_list.extend(
      tf.io.gfile.glob(prefitting_estimator.latest_checkpoint() + '*'))
  for file_path in remove_list:
    tf.io.gfile.remove(file_path)


def _finalize_model_structure(model_config, label_dimension, feature_columns,
                              head, prefitting_input_fn, prefitting_optimizer,
                              prefitting_steps, model_dir, config,
                              warm_start_from, dtype):
  """Sets up the lattice ensemble in model_config with requested algorithm."""
  if (not isinstance(model_config, configs.CalibratedLatticeEnsembleConfig) or
      isinstance(model_config.lattices, list)):
    return

  # TODO: If warmstarting, look for the previous ensemble file.
  if warm_start_from:
    raise ValueError('Warm starting lattice ensembles without explicitly '
                     'defined lattices is not supported yet.')

  if feature_columns:
    feature_names = [feature_column.name for feature_column in feature_columns]
  else:
    feature_names = [
        feature_config.name for feature_config in model_config.feature_configs
    ]

  if model_config.lattice_rank > len(feature_names):
    raise ValueError(
        'lattice_rank {} cannot be larger than the number of features: {}'
        .format(model_config.lattice_rank, feature_names))

  if model_config.num_lattices * model_config.lattice_rank < len(feature_names):
    raise ValueError(
        'Model with {}x{}d lattices is not large enough for all features: {}'
        .format(model_config.num_lattices, model_config.lattice_rank,
                feature_names))

  ensemble_structure_filename = os.path.join(model_dir,
                                             _ENSEMBLE_STRUCTURE_FILE)
  if ((config is None or config.is_chief) and
      not tf.io.gfile.exists(ensemble_structure_filename)):
    if model_config.lattices not in ['random', 'crystals', 'rtl_layer']:
      raise ValueError('Unsupported ensemble structure: {}'.format(
          model_config.lattices))
    if model_config.lattices == 'random':
      premade_lib.set_random_lattice_ensemble(model_config, feature_names)
    elif model_config.lattices == 'crystals':
      _set_crystals_lattice_ensemble(
          feature_names=feature_names,
          label_dimension=label_dimension,
          feature_columns=feature_columns,
          head=head,
          model_config=model_config,
          prefitting_input_fn=prefitting_input_fn,
          prefitting_optimizer=prefitting_optimizer,
          prefitting_steps=prefitting_steps,
          config=config,
          dtype=dtype)
    if (model_config.fix_ensemble_for_2d_constraints and
        model_config.lattices != 'rtl_layer'):
      # Note that we currently only support monotonicity and bound constraints
      # for RTL.
      _fix_ensemble_for_2d_constraints(model_config, feature_names)

    # Save lattices to file as the chief worker.
    tmp_ensemble_structure_filename = ensemble_structure_filename + 'tmp'
    with tf.io.gfile.GFile(tmp_ensemble_structure_filename,
                           'w') as ensemble_structure_file:
      ensemble_structure_file.write(json.dumps(model_config.lattices, indent=2))
    tf.io.gfile.rename(tmp_ensemble_structure_filename,
                       ensemble_structure_filename)
  else:
    # Non-chief workers read the lattices from file.
    _poll_for_file(ensemble_structure_filename)
    with tf.io.gfile.GFile(
        ensemble_structure_filename) as ensemble_structure_file:
      model_config.lattices = json.loads(ensemble_structure_file.read())

  logging.info('Finalized model structure: %s', str(model_config.lattices))


def _verify_config(model_config, feature_columns):
  """Verifies that the config is setup correctly and ready for model_fn."""
  if feature_columns:
    feature_configs = [
        model_config.feature_config_by_name(feature_column.name)
        for feature_column in feature_columns
    ]
  else:
    feature_configs = model_config.feature_configs or []

  for feature_config in feature_configs:
    if not feature_config.num_buckets:
      if (not np.iterable(feature_config.pwl_calibration_input_keypoints) or
          any(not isinstance(x, float)
              for x in feature_config.pwl_calibration_input_keypoints)):
        raise ValueError(
            'Input keypoints are invalid for feature {}: {}'.format(
                feature_config.name,
                feature_config.pwl_calibration_input_keypoints))

  if (not np.iterable(model_config.output_initialization) or any(
      not isinstance(x, float) for x in model_config.output_initialization)):
    raise ValueError('Output initilization is invalid: {}'.format(
        model_config.output_initialization))


def _update_by_feature_columns(model_config, feature_columns):
  """Updates a model config with the given feature columns."""
  for feature_column in feature_columns or []:
    feature_config = model_config.feature_config_by_name(feature_column.name)
    # pylint: disable=protected-access
    if (isinstance(feature_column, fc._DenseColumn) or
        isinstance(feature_column, fc2.DenseColumn)):
      feature_config.default_value = feature_column.default_value
    elif (isinstance(feature_column, fc._VocabularyListCategoricalColumn) or
          isinstance(feature_column, fc2.VocabularyListCategoricalColumn)):
      feature_config.vocabulary_list = feature_column.vocabulary_list
      feature_config.num_buckets = feature_column.num_buckets
      if feature_column.num_oov_buckets:
        # A positive num_oov_buckets can not be specified with default_value.
        # See tf.feature_column.categorical_column_with_vocabulary_list.
        feature_config.default_value = None
      else:
        # We add a bucket at the end for the default_value, since num_buckets
        # does not include the default value (but includes oov buckets).
        feature_config.default_value = feature_column.default_value
        feature_config.num_buckets += 1
    else:
      raise ValueError('Unsupported feature_column: {}'.format(feature_column))
    # pylint: enable=protected-access

  # Change categorical monotonicities to indices.
  premade_lib.set_categorical_monotonicities(model_config.feature_configs)


def _calibrated_lattice_ensemble_model_fn(features, labels, label_dimension,
                                          feature_columns, mode, head,
                                          model_config, optimizer, config,
                                          dtype):
  """Calibrated Lattice Ensemble Model."""
  del config
  if label_dimension != 1:
    raise ValueError('Only 1-dimensional output is supported.')

  # Get input tensors and corresponding feature configs.
  transformed_features = transform_features(features, feature_columns)
  feature_names = list(transformed_features.keys())
  input_tensors = [
      transformed_features[feature_name] for feature_name in feature_names
  ]
  # Reconstruct feature_config in order of feature_names
  feature_configs = [
      model_config.feature_config_by_name(feature_name)
      for feature_name in feature_names
  ]
  del model_config.feature_configs[:]
  model_config.feature_configs.extend(feature_configs)

  training = (mode == tf_estimator.ModeKeys.TRAIN)
  model = premade.CalibratedLatticeEnsemble(
      model_config=model_config, dtype=dtype)
  logits = tf.identity(
      model(input_tensors, training=training), name=OUTPUT_NAME)

  if training:
    optimizer = optimizers.get_optimizer_instance_v2(optimizer)
    optimizer.iterations = training_util.get_or_create_global_step()
  else:
    optimizer = None

  return head.create_estimator_spec(
      features=features,
      mode=mode,
      labels=labels,
      optimizer=optimizer,
      logits=logits,
      trainable_variables=model.trainable_variables,
      update_ops=model.updates,
      regularization_losses=model.losses or None)


def _calibrated_lattice_model_fn(features, labels, label_dimension,
                                 feature_columns, mode, head, model_config,
                                 optimizer, config, dtype):
  """Calibrated Lattice Model."""
  del config
  if label_dimension != 1:
    raise ValueError('Only 1-dimensional output is supported.')

  # Get input tensors and corresponding feature configs.
  transformed_features = transform_features(features, feature_columns)
  feature_names = list(transformed_features.keys())
  input_tensors = [
      transformed_features[feature_name] for feature_name in feature_names
  ]
  # Reconstruct feature_config in order of feature_names
  feature_configs = [
      model_config.feature_config_by_name(feature_name)
      for feature_name in feature_names
  ]
  del model_config.feature_configs[:]
  model_config.feature_configs.extend(feature_configs)

  training = (mode == tf_estimator.ModeKeys.TRAIN)
  model = premade.CalibratedLattice(model_config=model_config, dtype=dtype)
  logits = tf.identity(
      model(input_tensors, training=training), name=OUTPUT_NAME)

  if training:
    optimizer = optimizers.get_optimizer_instance_v2(optimizer)
    optimizer.iterations = training_util.get_or_create_global_step()
  else:
    optimizer = None

  return head.create_estimator_spec(
      features=features,
      mode=mode,
      labels=labels,
      optimizer=optimizer,
      logits=logits,
      trainable_variables=model.trainable_variables,
      update_ops=model.updates,
      regularization_losses=model.losses or None)


def _calibrated_linear_model_fn(features, labels, label_dimension,
                                feature_columns, mode, head, model_config,
                                optimizer, config, dtype):
  """Calibrated Linear Model."""
  del config
  if label_dimension != 1:
    raise ValueError('Only 1-dimensional output is supported.')

  # Get input tensors and corresponding feature configs.
  transformed_features = transform_features(features, feature_columns)
  feature_names = list(transformed_features.keys())
  input_tensors = [
      transformed_features[feature_name] for feature_name in feature_names
  ]
  # Reconstruct feature_config in order of feature_names
  feature_configs = [
      model_config.feature_config_by_name(feature_name)
      for feature_name in feature_names
  ]
  del model_config.feature_configs[:]
  model_config.feature_configs.extend(feature_configs)

  training = (mode == tf_estimator.ModeKeys.TRAIN)
  model = premade.CalibratedLinear(model_config=model_config, dtype=dtype)
  logits = tf.identity(
      model(input_tensors, training=training), name=OUTPUT_NAME)

  if training:
    optimizer = optimizers.get_optimizer_instance_v2(optimizer)
    optimizer.iterations = training_util.get_or_create_global_step()

  return head.create_estimator_spec(
      features=features,
      mode=mode,
      labels=labels,
      optimizer=optimizer,
      logits=logits,
      trainable_variables=model.trainable_variables,
      update_ops=model.updates,
      regularization_losses=model.losses or None)


def _get_model_fn(label_dimension, feature_columns, head, model_config,
                  optimizer, dtype):
  """Returns the model_fn for the given model_config."""
  if isinstance(model_config, configs.CalibratedLatticeConfig):

    def calibrated_lattice_model_fn(features, labels, mode, config):
      return _calibrated_lattice_model_fn(
          features=features,
          labels=labels,
          label_dimension=label_dimension,
          feature_columns=feature_columns,
          mode=mode,
          head=head,
          model_config=model_config,
          optimizer=optimizer,
          config=config,
          dtype=dtype)

    return calibrated_lattice_model_fn
  elif isinstance(model_config, configs.CalibratedLinearConfig):

    def calibrated_linear_model_fn(features, labels, mode, config):
      return _calibrated_linear_model_fn(
          features=features,
          labels=labels,
          label_dimension=label_dimension,
          feature_columns=feature_columns,
          mode=mode,
          head=head,
          model_config=model_config,
          optimizer=optimizer,
          config=config,
          dtype=dtype)

    return calibrated_linear_model_fn
  if isinstance(model_config, configs.CalibratedLatticeEnsembleConfig):

    def calibrated_lattice_ensemble_model_fn(features, labels, mode, config):
      return _calibrated_lattice_ensemble_model_fn(
          features=features,
          labels=labels,
          label_dimension=label_dimension,
          feature_columns=feature_columns,
          mode=mode,
          head=head,
          model_config=model_config,
          optimizer=optimizer,
          config=config,
          dtype=dtype)

    return calibrated_lattice_ensemble_model_fn
  else:
    raise ValueError('Unsupported model type: {}'.format(type(model_config)))


class CannedEstimator(estimator_lib.EstimatorV2):
  """An estimator for TensorFlow lattice models.

  Creates an estimator with a custom head for the model architecutre specified
  by the `model_config`, which should be one of those defined in `tfl.configs`.
  Calculation of feature quantiles for input keypoint initialization is done
  using `feature_analysis_input_fn`. If this auxiliary input fn is not provided,
  all keypoint values should be explicitly provided via the `model_config`.

  Example:

  ```python
  model_config = tfl.configs.CalibratedLatticeConfig(...)
  feature_analysis_input_fn = create_input_fn(num_epochs=1, ...)
  train_input_fn = create_input_fn(num_epochs=100, ...)
  head = ...
  estimator = tfl.estimators.CannedEstimator(
      feature_columns=feature_columns,
      model_config=model_config,
      feature_analysis_input_fn=feature_analysis_input_fn
      head=head)
  estimator.train(input_fn=train_input_fn)
  ```
  """

  def __init__(self,
               head,
               model_config,
               feature_columns,
               feature_analysis_input_fn=None,
               feature_analysis_weight_column=None,
               feature_analysis_weight_reduction='mean',
               prefitting_input_fn=None,
               model_dir=None,
               label_dimension=1,
               optimizer='Adagrad',
               prefitting_optimizer='Adagrad',
               prefitting_steps=None,
               config=None,
               warm_start_from=None,
               dtype=tf.float32):
    """Initializes a `CannedEstimator` instance.

    Args:
      head: A `_Head` instance constructed with a method such as
        `tf.contrib.estimator.multi_label_head`.
      model_config: Model configuration object describing model architecutre.
        Should be one of the model configs in `tfl.configs`.
      feature_columns: An iterable containing all the feature columns used by
        the model.
      feature_analysis_input_fn: An input_fn used to calculate statistics about
        features and labels in order to setup calibration keypoint and values.
      feature_analysis_weight_column: A string or a `_NumericColumn` created by
        `tf.feature_column.numeric_column` defining feature column representing
        weights used for calculating weighted feature statistics (quantiles).
        Can be the same as `weight_column`.
      feature_analysis_weight_reduction: Reduction used on weights when
        aggregating repeated values during feature analysis. Can be either 'sum'
        or 'mean'.
      prefitting_input_fn: An input_fn used in the pre fitting stage to estimate
        non-linear feature interactions. Required for crystals models.
        Prefitting typically uses the same dataset as the main training, but
        with fewer epochs.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      label_dimension: Number of regression targets per example. This is the
        size of the last dimension of the labels and logits `Tensor` objects
        (typically, these have shape `[batch_size, label_dimension]`).
      optimizer: An instance of `tf.Optimizer` used to train the model. Can also
        be a string (one of 'Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'), or
        callable. Defaults to Adagrad optimizer.
      prefitting_optimizer: An instance of `tf.Optimizer` used to train the
        model during the pre-fitting stage. Can also be a string (one of
        'Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'), or callable. Defaults to
        Adagrad optimizer.
      prefitting_steps: Number of steps for which to pretraing train the model
        during the prefitting stage. If None, train forever or train until
        prefitting_input_fn generates the tf.errors.OutOfRange error or
        StopIteration exception.
      config: `RunConfig` object to configure the runtime settings.
      warm_start_from: A string filepath to a checkpoint to warm-start from, or
        a `WarmStartSettings` object to fully configure warm-starting.  If the
        string filepath is provided instead of a `WarmStartSettings`, then all
        weights are warm-started, and it is assumed that vocabularies and Tensor
        names are unchanged.
      dtype: dtype of layers used in the model.
    """
    config = estimator_lib.maybe_overwrite_model_dir_and_session_config(
        config, model_dir)
    model_dir = config.model_dir

    model_config = copy.deepcopy(model_config)
    _update_by_feature_columns(model_config, feature_columns)

    _finalize_keypoints(
        model_config=model_config,
        config=config,
        feature_columns=feature_columns,
        feature_analysis_input_fn=feature_analysis_input_fn,
        feature_analysis_weight_column=feature_analysis_weight_column,
        feature_analysis_weight_reduction=feature_analysis_weight_reduction,
        logits_output=True)

    _verify_config(model_config, feature_columns)

    _finalize_model_structure(
        label_dimension=label_dimension,
        feature_columns=feature_columns,
        head=head,
        model_config=model_config,
        prefitting_input_fn=prefitting_input_fn,
        prefitting_optimizer=prefitting_optimizer,
        prefitting_steps=prefitting_steps,
        model_dir=model_dir,
        config=config,
        warm_start_from=warm_start_from,
        dtype=dtype)

    model_fn = _get_model_fn(
        label_dimension=label_dimension,
        feature_columns=feature_columns,
        head=head,
        model_config=model_config,
        optimizer=optimizer,
        dtype=dtype)

    super(CannedEstimator, self).__init__(
        model_fn=model_fn,
        model_dir=model_dir,
        config=config,
        warm_start_from=warm_start_from)


class CannedClassifier(estimator_lib.EstimatorV2):
  """Canned classifier for TensorFlow lattice models.

  Creates a classifier for the model architecutre specified by the
  `model_config`, which should be one of those defined in `tfl.configs`.
  Calclulation of feature quantiles for input keypoint initialization is done
  using `feature_analysis_input_fn`. If this auxiliary input fn is not provided,
  all keypoint values should be explicitly provided via the `model_config`.

  Training loss is softmax cross-entropy as defined for the default
  TF classificaiton head.

  Example:

  ```python
  model_config = tfl.configs.CalibratedLatticeConfig(...)
  feature_analysis_input_fn = create_input_fn(num_epochs=1, ...)
  train_input_fn = create_input_fn(num_epochs=100, ...)
  estimator = tfl.estimators.CannedClassifier(
      feature_columns=feature_columns,
      model_config=model_config,
      feature_analysis_input_fn=feature_analysis_input_fn)
  estimator.train(input_fn=train_input_fn)
  ```
  """

  def __init__(self,
               model_config,
               feature_columns,
               feature_analysis_input_fn=None,
               feature_analysis_weight_column=None,
               feature_analysis_weight_reduction='mean',
               prefitting_input_fn=None,
               model_dir=None,
               n_classes=2,
               weight_column=None,
               label_vocabulary=None,
               optimizer='Adagrad',
               prefitting_optimizer='Adagrad',
               prefitting_steps=None,
               config=None,
               warm_start_from=None,
               loss_reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
               loss_fn=None,
               dtype=tf.float32):
    """Initializes a `CannedClassifier` instance.

    Args:
      model_config: Model configuration object describing model architecutre.
        Should be one of the model configs in `tfl.configs`.
      feature_columns: An iterable containing all the feature columns used by
        the model.
      feature_analysis_input_fn: An input_fn used to calculate statistics about
        features and labels in order to setup calibration keypoint and values.
      feature_analysis_weight_column: A string or a `_NumericColumn` created by
        `tf.feature_column.numeric_column` defining feature column representing
        weights used for calculating weighted feature statistics (quantiles).
        Can be the same as `weight_column`.
      feature_analysis_weight_reduction: Reduction used on weights when
        aggregating repeated values during feature analysis. Can be either 'sum'
        or 'mean'.
      prefitting_input_fn: An input_fn used in the pre fitting stage to estimate
        non-linear feature interactions. Required for crystals models.
        Prefitting typically uses the same dataset as the main training, but
        with fewer epochs.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      n_classes: Number of label classes. Defaults to 2, namely binary
        classification. Must be > 1.
      weight_column: A string or a `_NumericColumn` created by
        `tf.feature_column.numeric_column` defining feature column representing
        weights. It is only used by the estimator head to down weight or boost
        examples during training. It will be multiplied by the loss of the
        example. If it is a string, it is used as a key to fetch the weight
        tensor from the `features` dictionary output of the input function. If
        it is a `_NumericColumn`, a raw tensor is fetched by key
        `weight_column.key`, then weight_column.normalizer_fn is applied on it
        to get the weight tensor. Note that in both cases 'weight_column' should
        *not* be a member of the 'feature_columns'  parameter to the constructor
        since these will be used for both serving and training.
      label_vocabulary: A list of strings represents possible label values. If
        given, labels must be string type and have any value in
        `label_vocabulary`. If it is not given, that means labels are already
        encoded as integer or float within [0, 1] for `n_classes=2` and encoded
        as integer values in {0, 1,..., n_classes-1} for `n_classes`>2 . Also
        there will be errors if vocabulary is not provided and labels are
        string.
      optimizer: An instance of `tf.Optimizer` used to train the model. Can also
        be a string (one of 'Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'), or
        callable. Defaults to Adagrad optimizer.
      prefitting_optimizer: An instance of `tf.Optimizer` used to train the
        model during the pre-fitting stage. Can also be a string (one of
        'Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'), or callable. Defaults to
        Adagrad optimizer.
      prefitting_steps: Number of steps for which to pretraing train the model
        during the prefitting stage. If None, train forever or train until
        prefitting_input_fn generates the tf.errors.OutOfRange error or
        StopIteration exception.
      config: `RunConfig` object to configure the runtime settings.
      warm_start_from: A string filepath to a checkpoint to warm-start from, or
        a `WarmStartSettings` object to fully configure warm-starting.  If the
        string filepath is provided instead of a `WarmStartSettings`, then all
        weights are warm-started, and it is assumed that vocabularies and Tensor
        names are unchanged.
      loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how
        to reduce training loss over batch. Defaults to `SUM_OVER_BATCH_SIZE`.
      loss_fn: Optional loss function.
      dtype: dtype of layers used in the model.
    """
    config = estimator_lib.maybe_overwrite_model_dir_and_session_config(
        config, model_dir)
    model_dir = config.model_dir

    if n_classes == 2:
      head = binary_class_head.BinaryClassHead(
          weight_column=weight_column,
          label_vocabulary=label_vocabulary,
          loss_reduction=loss_reduction,
          loss_fn=loss_fn)
    else:
      head = multi_class_head.MultiClassHead(
          n_classes,
          weight_column=weight_column,
          label_vocabulary=label_vocabulary,
          loss_reduction=loss_reduction,
          loss_fn=loss_fn)

    label_dimension = 1 if n_classes == 2 else n_classes

    model_config = copy.deepcopy(model_config)
    _update_by_feature_columns(model_config, feature_columns)

    _finalize_keypoints(
        model_config=model_config,
        config=config,
        feature_columns=feature_columns,
        feature_analysis_input_fn=feature_analysis_input_fn,
        feature_analysis_weight_column=feature_analysis_weight_column,
        feature_analysis_weight_reduction=feature_analysis_weight_reduction,
        logits_output=True)

    _verify_config(model_config, feature_columns)

    _finalize_model_structure(
        label_dimension=label_dimension,
        feature_columns=feature_columns,
        head=head,
        model_config=model_config,
        prefitting_input_fn=prefitting_input_fn,
        prefitting_optimizer=prefitting_optimizer,
        prefitting_steps=prefitting_steps,
        model_dir=model_dir,
        config=config,
        warm_start_from=warm_start_from,
        dtype=dtype)

    model_fn = _get_model_fn(
        label_dimension=label_dimension,
        feature_columns=feature_columns,
        head=head,
        model_config=model_config,
        optimizer=optimizer,
        dtype=dtype)

    super(CannedClassifier, self).__init__(
        model_fn=model_fn,
        model_dir=model_dir,
        config=config,
        warm_start_from=warm_start_from)


class CannedRegressor(estimator_lib.EstimatorV2):
  """A regressor for TensorFlow lattice models.

  Creates a regressor for the model architecutre specified by the
  `model_config`, which should be one of those defined in `tfl.configs`.
  Calclulation of feature quantiles for input keypoint initialization is done
  using `feature_analysis_input_fn`. If this auxiliary input fn is not provided,
  all keypoint values should be explicitly provided via the `model_config`.

  Training loss is squared error as defined for the default TF regression head.

  Example:

  ```python
  model_config = tfl.configs.CalibratedLatticeConfig(...)
  feature_analysis_input_fn = create_input_fn(num_epochs=1, ...)
  train_input_fn = create_input_fn(num_epochs=100, ...)
  estimator = tfl.estimators.CannedRegressor(
      feature_columns=feature_columns,
      model_config=model_config,
      feature_analysis_input_fn=feature_analysis_input_fn)
  estimator.train(input_fn=train_input_fn)
  ```
  """

  def __init__(self,
               model_config,
               feature_columns,
               feature_analysis_input_fn=None,
               feature_analysis_weight_column=None,
               feature_analysis_weight_reduction='mean',
               prefitting_input_fn=None,
               model_dir=None,
               label_dimension=1,
               weight_column=None,
               optimizer='Adagrad',
               prefitting_optimizer='Adagrad',
               prefitting_steps=None,
               config=None,
               warm_start_from=None,
               loss_reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
               loss_fn=None,
               dtype=tf.float32):
    """Initializes a `CannedRegressor` instance.

    Args:
      model_config: Model configuration object describing model architecutre.
        Should be one of the model configs in `tfl.configs`.
      feature_columns: An iterable containing all the feature columns used by
        the model.
      feature_analysis_input_fn: An input_fn used to calculate statistics about
        features and labels in order to setup calibration keypoint and values.
      feature_analysis_weight_column: A string or a `_NumericColumn` created by
        `tf.feature_column.numeric_column` defining feature column representing
        weights used for calculating weighted feature statistics (quantiles).
        Can be the same as `weight_column`.
      feature_analysis_weight_reduction: Reduction used on weights when
        aggregating repeated values during feature analysis. Can be either 'sum'
        or 'mean'.
      prefitting_input_fn: An input_fn used in the pre fitting stage to estimate
        non-linear feature interactions. Required for crystals models.
        Prefitting typically uses the same dataset as the main training, but
        with fewer epochs.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      label_dimension: Number of regression targets per example. This is the
        size of the last dimension of the labels and logits `Tensor` objects
        (typically, these have shape `[batch_size, label_dimension]`).
      weight_column: A string or a `_NumericColumn` created by
        `tf.feature_column.numeric_column` defining feature column representing
        weights. It is only used by the estimator head to down weight or boost
        examples during training. It will be multiplied by the loss of the
        example. If it is a string, it is used as a key to fetch the weight
        tensor from the `features` dictionary output of the input function. If
        it is a `_NumericColumn`, a raw tensor is fetched by key
        `weight_column.key`, then weight_column.normalizer_fn is applied on it
        to get the weight tensor. Note that in both cases 'weight_column' should
        *not* be a member of the 'feature_columns'  parameter to the constructor
        since these will be used for both serving and training.
      optimizer: An instance of `tf.Optimizer` used to train the model. Can also
        be a string (one of 'Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'), or
        callable. Defaults to Adagrad optimizer.
      prefitting_optimizer: An instance of `tf.Optimizer` used to train the
        model during the pre-fitting stage. Can also be a string (one of
        'Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD'), or callable. Defaults to
        Adagrad optimizer.
      prefitting_steps: Number of steps for which to pretraing train the model
        during the prefitting stage. If None, train forever or train until
        prefitting_input_fn generates the tf.errors.OutOfRange error or
        StopIteration exception.
      config: `RunConfig` object to configure the runtime settings.
      warm_start_from: A string filepath to a checkpoint to warm-start from, or
        a `WarmStartSettings` object to fully configure warm-starting.  If the
        string filepath is provided instead of a `WarmStartSettings`, then all
        weights are warm-started, and it is assumed that vocabularies and Tensor
        names are unchanged.
      loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how
        to reduce training loss over batch. Defaults to `SUM_OVER_BATCH_SIZE`.
      loss_fn: Optional loss function. Defaults to `mean_squared_error`.
      dtype: dtype of layers used in the model.
    """
    config = estimator_lib.maybe_overwrite_model_dir_and_session_config(
        config, model_dir)
    model_dir = config.model_dir
    head = regression_head.RegressionHead(
        label_dimension=label_dimension,
        weight_column=weight_column,
        loss_reduction=loss_reduction,
        loss_fn=loss_fn)

    model_config = copy.deepcopy(model_config)
    _update_by_feature_columns(model_config, feature_columns)

    _finalize_keypoints(
        model_config=model_config,
        config=config,
        feature_columns=feature_columns,
        feature_analysis_input_fn=feature_analysis_input_fn,
        feature_analysis_weight_column=feature_analysis_weight_column,
        feature_analysis_weight_reduction=feature_analysis_weight_reduction,
        logits_output=False)

    _verify_config(model_config, feature_columns)

    _finalize_model_structure(
        label_dimension=label_dimension,
        feature_columns=feature_columns,
        head=head,
        model_config=model_config,
        prefitting_input_fn=prefitting_input_fn,
        prefitting_optimizer=prefitting_optimizer,
        prefitting_steps=prefitting_steps,
        model_dir=model_dir,
        config=config,
        warm_start_from=warm_start_from,
        dtype=dtype)

    model_fn = _get_model_fn(
        label_dimension=label_dimension,
        feature_columns=feature_columns,
        head=head,
        model_config=model_config,
        optimizer=optimizer,
        dtype=dtype)

    super(CannedRegressor, self).__init__(
        model_fn=model_fn,
        model_dir=model_dir,
        config=config,
        warm_start_from=warm_start_from)


def _match_op(ops, regex):
  """Returns ops that match given regex along with the matched sections."""
  matches = []
  prog = re.compile(regex)
  for op in ops:
    op_matches = prog.findall(op)
    if op_matches:
      matches.append((op, op_matches[0]))
  return matches


def _create_feature_nodes(sess, ops, graph):
  """Returns a map from feature name to InputFeatureNode."""
  # Extract list of features from the graph.
  # {FEATURES_SCOPE}/{feature_name}
  feature_nodes = {}
  feature_op_re = r'^{}/(.*)'.format(re.escape(FEATURES_SCOPE))
  for (_, feature_name) in _match_op(ops, feature_op_re):
    is_categorical = False
    vocabulary_list = None

    # Check to see if there is a categorical mapping defined for this feature
    # (i.e. lookup table for categorical feature column). Note that there can be
    # at most one such mapping either under transform or transform_1 namespace.
    # transform(_\d)?/{feature_name}_lookup/Const
    category_table_re = r'transform(_\d)?/{}_lookup/Const'.format(
        re.escape(feature_name))
    for (category_table_op, _) in _match_op(ops, category_table_re):
      if is_categorical:
        raise ValueError(
            'Model graph has multiple category mappings for feature {}'.format(
                feature_name))
      is_categorical = True
      vocabulary_list = sess.run(
          graph.get_operation_by_name(category_table_op).outputs[0])
      # Replace byte types with their string values.
      vocabulary_list = [
          str(x.decode()) if isinstance(x, bytes) else str(x)
          for x in vocabulary_list
      ]

    feature_node = model_info.InputFeatureNode(
        name=feature_name,
        is_categorical=is_categorical,
        vocabulary_list=vocabulary_list)
    feature_nodes[feature_name] = feature_node
  return feature_nodes


def _create_categorical_calibration_nodes(sess, ops, graph, feature_nodes):
  """Returns a map from feature_name to list of CategoricalCalibrationNode."""
  categorical_calibration_nodes = collections.defaultdict(list)
  # Get calibrator output values. We need to call the read variable op.
  # {CALIB_LAYER_NAME}_{feature_name}/
  #   {CATEGORICAL_CALIBRATION_KERNEL_NAME}/Read/ReadVariableOp
  kernel_op_re = '^{}_(.*)/{}/Read/ReadVariableOp$'.format(
      premade_lib.CALIB_LAYER_NAME,
      categorical_calibration_layer.CATEGORICAL_CALIBRATION_KERNEL_NAME,
  )
  for kernel_op, feature_name in _match_op(ops, kernel_op_re):
    output_values = sess.run(graph.get_operation_by_name(kernel_op).outputs[0])

    # Get default input value if defined.
    # {CALIB_LAYER_NAME}_{feature_name}/
    #   {DEFAULT_INPUT_VALUE_NAME}
    default_input_value_op = '^{}_{}/{}$'.format(
        premade_lib.CALIB_LAYER_NAME,
        feature_name,
        categorical_calibration_layer.DEFAULT_INPUT_VALUE_NAME,
    )
    if default_input_value_op in ops:
      default_input = sess.run(
          graph.get_operation_by_name(default_input_value_op).outputs[0])
    else:
      default_input = None

    # Create one calibration node per output dimension of the calibrator.
    for calibration_output_idx in range(output_values.shape[1]):
      categorical_calibration_node = model_info.CategoricalCalibrationNode(
          input_node=feature_nodes[feature_name],
          output_values=output_values[:, calibration_output_idx],
          default_input=default_input)
      categorical_calibration_nodes[feature_name].append(
          categorical_calibration_node)
  return categorical_calibration_nodes


def _create_pwl_calibration_nodes(sess, ops, graph, feature_nodes):
  """Returns a map from feature_name to list of PWLCalibrationNode."""
  pwl_calibration_nodes = collections.defaultdict(list)
  # Calculate input keypoints.
  # We extract lengh (deltas between keypoints) and kernel interpolation
  # keypoints (which does not include the last keypoint), and then
  # construct the full keypoints list using both.

  # Lengths (deltas between keypoints).
  # {CALIB_LAYER_NAME}_{feature_name}/{LENGTHS_NAME}
  lengths_op_re = '^{}_(.*)/{}$'.format(
      premade_lib.CALIB_LAYER_NAME,
      pwl_calibration_layer.LENGTHS_NAME,
  )
  for lengths_op, feature_name in _match_op(ops, lengths_op_re):
    # Interpolation keypoints does not inlcude the last input keypoint.
    # {CALIB_LAYER_NAME}_{feature_name}/{INTERPOLATION_KEYPOINTS_NAME}
    keypoints_op = '{}_{}/{}'.format(
        premade_lib.CALIB_LAYER_NAME,
        feature_name,
        pwl_calibration_layer.INTERPOLATION_KEYPOINTS_NAME,
    )

    # Output keypoints. We need to call the varible read op.
    # {CALIB_LAYER_NAME}_{feature_name}/{PWL_CALIBRATION_KERNEL_NAME}
    kernel_op = '{}_{}/{}/Read/ReadVariableOp'.format(
        premade_lib.CALIB_LAYER_NAME,
        feature_name,
        pwl_calibration_layer.PWL_CALIBRATION_KERNEL_NAME,
    )

    (lengths, keypoints, kernel) = sess.run(
        (graph.get_operation_by_name(lengths_op).outputs[0],
         graph.get_operation_by_name(keypoints_op).outputs[0],
         graph.get_operation_by_name(kernel_op).outputs[0]))
    output_keypoints = np.cumsum(kernel, axis=0)

    # For calibrators with fixed input keypoints, the shape of 'keypoints'
    # and 'lengths' is (num_keypoints - 1,). For calibrators with keypoint_type
    # set to learned_interior the shape is (num_units, num_keypoints - 1).
    # Change the shape to (num_units, num_keypoints - 1).
    # If needed, repeat the keypoints for each output unit.
    if keypoints.ndim == 1:
      keypoints = np.expand_dims(keypoints, axis=0)
      lengths = np.expand_dims(lengths, axis=0)
    if keypoints.shape[0] == 1:
      keypoints = np.tile(keypoints, [output_keypoints.shape[1], 1])
      lengths = np.tile(lengths, [output_keypoints.shape[1], 1])

    # Add the last keypoint to the keypoint list. The resulting shape is
    # (num_units, num_keypoints).
    # TODO: handle cyclic PWL layers.
    input_keypoints = np.concatenate(
        [keypoints, keypoints[:, -1:] + lengths[:, -1:]], axis=-1)

    # Get missing/default input value if present:
    # {CALIB_LAYER_NAME}_{feature_name}/{MISSING_INPUT_VALUE_NAME}
    default_input_value_op = '{}_{}/{}'.format(
        premade_lib.CALIB_LAYER_NAME,
        feature_name,
        pwl_calibration_layer.MISSING_INPUT_VALUE_NAME,
    )
    if default_input_value_op in ops:
      default_input = sess.run(
          graph.get_operation_by_name(default_input_value_op).outputs[0])[0]
    else:
      default_input = None

    # Find corresponding default/missing output if present.
    # {CALIB_LAYER_NAME}_{feature_name}/{PWL_CALIBRATION_MISSING_OUTPUT_NAME}
    default_output_op = '{}_{}/{}/Read/ReadVariableOp'.format(
        premade_lib.CALIB_LAYER_NAME,
        feature_name,
        pwl_calibration_layer.PWL_CALIBRATION_MISSING_OUTPUT_NAME,
    )
    if default_output_op in ops:
      default_output = sess.run(
          graph.get_operation_by_name(default_output_op).outputs[0])
    else:
      default_output = None

    # Create one calibration node per output dimension of the calibrator.
    for calibration_output_idx in range(output_keypoints.shape[1]):
      pwl_calibration_node = model_info.PWLCalibrationNode(
          input_node=feature_nodes[feature_name],
          input_keypoints=input_keypoints[calibration_output_idx, :],
          output_keypoints=output_keypoints[:, calibration_output_idx],
          default_input=default_input,
          default_output=(None if default_output is None else
                          default_output[:, calibration_output_idx]))
      pwl_calibration_nodes[feature_name].append(pwl_calibration_node)
  return pwl_calibration_nodes


def _create_submodel_input_map(ops, calibration_nodes_map):
  """Returns a map from submodel_idx to a list of calibration nodes."""
  submodel_input_nodes = collections.defaultdict(list)
  for feature_name, calibration_nodes in calibration_nodes_map.items():
    # Identity passthrough ops that pass this calibration to each submodel.
    # {CALIB_PASSTHROUGH_NAME}_{feature_name}_
    #   {calibration_output_idx}_{submodel_idx}_{submodel_input_idx}
    shared_calib_passthrough_op_re = r'^{}_{}_(\d*)_(\d*)_(\d*)$'.format(
        premade_lib.CALIB_PASSTHROUGH_NAME, feature_name)
    for _, (calibration_output_idx, submodel_idx,
            submodel_input_idx) in _match_op(ops,
                                             shared_calib_passthrough_op_re):
      submodel_input_nodes[submodel_idx].append(
          (submodel_input_idx, calibration_nodes[int(calibration_output_idx)]))
  return submodel_input_nodes


def _create_linear_nodes(sess, ops, graph, submodel_input_nodes):
  """Returns a map from submodel_idx to LinearNode."""
  linear_nodes = {}
  # Linear coefficients.
  # {LINEAR_LAYER_NAME}_{submodel_idx}/{LINEAR_LAYER_KERNEL_NAME}
  linear_kernel_op_re = '^{}_(.*)/{}/Read/ReadVariableOp$'.format(
      premade_lib.LINEAR_LAYER_NAME,
      linear_layer.LINEAR_LAYER_KERNEL_NAME,
  )
  for linear_kernel_op, submodel_idx in _match_op(ops, linear_kernel_op_re):
    coefficients = sess.run(
        graph.get_operation_by_name(linear_kernel_op).outputs[0]).flatten()

    # Bias term.
    # {LINEAR_LAYER_NAME}_{submodel_idx}/{LINEAR_LAYER_BIAS_NAME}
    bias_op = '{}_{}/{}/Read/ReadVariableOp'.format(
        premade_lib.LINEAR_LAYER_NAME,
        submodel_idx,
        linear_layer.LINEAR_LAYER_BIAS_NAME,
    )
    if bias_op in ops:
      bias = sess.run(graph.get_operation_by_name(bias_op).outputs[0])
    else:
      bias = 0.0

    # Sort input nodes by input index.
    input_nodes = [
        node for _, node in sorted(submodel_input_nodes[submodel_idx])
    ]

    linear_node = model_info.LinearNode(
        input_nodes=input_nodes, coefficients=coefficients, bias=bias)
    linear_nodes[submodel_idx] = linear_node
  return linear_nodes


def _create_lattice_nodes(sess, ops, graph, submodel_input_nodes):
  """Returns a map from submodel_idx to LatticeNode."""
  lattice_nodes = {}
  # Lattice weights.
  # {LATTICE_LAYER_NAME}_{submodel_idx}/{LATTICE_KERNEL_NAME}
  lattice_kernel_op_re = '^{}_(.*)/{}/Read/ReadVariableOp$'.format(
      premade_lib.LATTICE_LAYER_NAME,
      lattice_layer.LATTICE_KERNEL_NAME,
  )
  for lattice_kernel_op, submodel_idx in _match_op(ops, lattice_kernel_op_re):
    lattice_kernel = sess.run(
        graph.get_operation_by_name(lattice_kernel_op).outputs[0]).flatten()

    # Lattice sizes.
    # {Lattice_LAYER_NAME}_{submodel_idx}/{LATTICE_SIZES_NAME}
    lattice_sizes_op_name = '{}_{}/{}'.format(premade_lib.LATTICE_LAYER_NAME,
                                              submodel_idx,
                                              lattice_layer.LATTICE_SIZES_NAME)
    lattice_sizes = sess.run(
        graph.get_operation_by_name(
            lattice_sizes_op_name).outputs[0]).flatten()

    # Shape the flat lattice parameters based on the calculated lattice sizes.
    weights = np.reshape(lattice_kernel, lattice_sizes)

    # Sort input nodes by input index.
    input_nodes = [
        node for _, node in sorted(submodel_input_nodes[submodel_idx])
    ]

    lattice_node = model_info.LatticeNode(
        input_nodes=input_nodes, weights=weights)
    lattice_nodes[submodel_idx] = lattice_node
  return lattice_nodes


def _create_kronecker_factored_lattice_nodes(sess, ops, graph,
                                             submodel_input_nodes):
  """Returns a map from submodel_idx to KroneckerFactoredLatticeNode."""
  kfl_nodes = {}
  # KroneckerFactoredLattice kernel weights.
  # {KFL_LAYER_NAME}_{submodel_idx}/{KFL_KERNEL_NAME}
  kfl_kernel_op_re = '^{}_(.*)/{}/Read/ReadVariableOp$'.format(
      premade_lib.KFL_LAYER_NAME,
      kfll.KFL_KERNEL_NAME,
  )
  for kfl_kernel_op, submodel_idx in _match_op(ops, kfl_kernel_op_re):
    kfl_kernel = sess.run(
        graph.get_operation_by_name(kfl_kernel_op).outputs[0]).flatten()

    # KroneckerFactoredLattice scale.
    # {KFL_LAYER_NAME}_{submodel_idx}/{KFL_SCALE_NAME}
    kfl_scale_op_name = '{}_{}/{}/Read/ReadVariableOp'.format(
        premade_lib.KFL_LAYER_NAME, submodel_idx, kfll.KFL_SCALE_NAME)
    kfl_scale = sess.run(
        graph.get_operation_by_name(kfl_scale_op_name).outputs[0]).flatten()

    # KroneckerFactoredLattice bias.
    # {KFL_LAYER_NAME}_{submodel_idx}/{KFL_BIAS_NAME}
    kfl_bias_op_name = '{}_{}/{}/Read/ReadVariableOp'.format(
        premade_lib.KFL_LAYER_NAME, submodel_idx, kfll.KFL_BIAS_NAME)
    kfl_bias = sess.run(
        graph.get_operation_by_name(kfl_bias_op_name).outputs[0]).flatten()

    # Lattice sizes.
    # {KFL_LAYER_NAME}_{submodel_idx}/{LATTICE_SIZES_NAME}
    lattice_sizes_op_name = '{}_{}/{}'.format(premade_lib.KFL_LAYER_NAME,
                                              submodel_idx,
                                              kfll.LATTICE_SIZES_NAME)
    lattice_sizes = sess.run(
        graph.get_operation_by_name(lattice_sizes_op_name).outputs[0])

    # Units.
    # {KFL_LAYER_NAME}_{submodel_idx}/{UNITS_NAME}
    units_op_name = '{}_{}/{}'.format(premade_lib.KFL_LAYER_NAME, submodel_idx,
                                      kfll.UNITS_NAME)
    units = sess.run(graph.get_operation_by_name(units_op_name).outputs[0])

    # Dims.
    # {KFL_LAYER_NAME}_{submodel_idx}/{DIMS_NAME}
    dims_op_name = '{}_{}/{}'.format(premade_lib.KFL_LAYER_NAME, submodel_idx,
                                     kfll.DIMS_NAME)
    dims = sess.run(graph.get_operation_by_name(dims_op_name).outputs[0])

    # Num terms.
    # {KFL_LAYER_NAME}_{submodel_idx}/{NUM_TERMS_NAME}
    num_terms_op_name = '{}_{}/{}'.format(premade_lib.KFL_LAYER_NAME,
                                          submodel_idx, kfll.NUM_TERMS_NAME)
    num_terms = sess.run(
        graph.get_operation_by_name(num_terms_op_name).outputs[0])

    # Shape the flat weights, scale, and bias parameters based on the calculated
    # lattice_sizes, units, dims, and num_terms.
    weights = np.reshape(kfl_kernel,
                         (1, lattice_sizes, units * dims, num_terms))
    scale = np.reshape(kfl_scale, (units, num_terms))
    bias = np.reshape(kfl_bias, (units))

    # Sort input nodes by input index.
    input_nodes = [
        node for _, node in sorted(submodel_input_nodes[submodel_idx])
    ]

    kfl_node = model_info.KroneckerFactoredLatticeNode(
        input_nodes=input_nodes, weights=weights, scale=scale, bias=bias)
    kfl_nodes[submodel_idx] = kfl_node
  return kfl_nodes


def _create_rtl_submodel_kronecker_factored_lattice_nodes(
    sess, ops, graph, flattened_calibration_nodes, submodel_idx, submodel_key):
  """Returns next key and map from key+unit to KroneckerFactoredLatticeNode."""
  submodel_kfl_nodes = {}
  # KroneckerFactoredLattice kernel weights
  # {RTL_LAYER_NAME}_{submodel_idx}/
  # {RTL_KFL_NAME}_{monotonicities}/{KFL_KERNEL_NAME}
  kfl_kernel_op_re = '^{}_{}/{}_(.*)/{}/Read/ReadVariableOp$'.format(
      premade_lib.RTL_LAYER_NAME,
      submodel_idx,
      rtl_layer.RTL_KFL_NAME,
      kfll.KFL_KERNEL_NAME,
  )
  for kfl_kernel_op, monotonicities in _match_op(ops, kfl_kernel_op_re):
    kfl_kernel = sess.run(
        graph.get_operation_by_name(kfl_kernel_op).outputs[0]).flatten()

    # KroneckerFactoredLattice scale.
    # {RTL_LAYER_NAME}_{submodel_idx}/
    # {RTL_KFL_NAME}_{monotonicities}/{KFL_SCALE_NAME}
    kfl_scale_op_name = '{}_{}/{}_{}/{}/Read/ReadVariableOp'.format(
        premade_lib.RTL_LAYER_NAME,
        submodel_idx,
        rtl_layer.RTL_KFL_NAME,
        monotonicities,
        kfll.KFL_SCALE_NAME,
    )
    kfl_scale = sess.run(
        graph.get_operation_by_name(kfl_scale_op_name).outputs[0]).flatten()

    # KroneckerFactoredLattice bias.
    # {RTL_LAYER_NAME}_{submodel_idx}/
    # {RTL_KFL_NAME}_{monotonicities}/{KFL_BIAS_NAME}
    kfl_bias_op_name = '{}_{}/{}_{}/{}/Read/ReadVariableOp'.format(
        premade_lib.RTL_LAYER_NAME,
        submodel_idx,
        rtl_layer.RTL_KFL_NAME,
        monotonicities,
        kfll.KFL_BIAS_NAME,
    )
    kfl_bias = sess.run(
        graph.get_operation_by_name(kfl_bias_op_name).outputs[0]).flatten()

    # Lattice sizes.
    # {RTL_LAYER_NAME}_{submodel_idx}/
    # {RTL_KFL_NAME}_{monotonicities}/{LATTICE_SIZES_NAME}
    lattice_sizes_op_name = '{}_{}/{}_{}/{}'.format(
        premade_lib.RTL_LAYER_NAME,
        submodel_idx,
        rtl_layer.RTL_KFL_NAME,
        monotonicities,
        kfll.LATTICE_SIZES_NAME,
    )
    lattice_sizes = sess.run(
        graph.get_operation_by_name(lattice_sizes_op_name).outputs[0])

    # Dims.
    # {RTL_LAYER_NAME}_{submodel_idx}/
    # {RTL_KFL_NAME}_{monotonicities}/{DIMS_NAME}
    dims_op_name = '{}_{}/{}_{}/{}'.format(
        premade_lib.RTL_LAYER_NAME,
        submodel_idx,
        rtl_layer.RTL_KFL_NAME,
        monotonicities,
        kfll.DIMS_NAME,
    )
    dims = sess.run(graph.get_operation_by_name(dims_op_name).outputs[0])

    # Num terms.
    # {RTL_LAYER_NAME}_{submodel_idx}/
    # {RTL_KFL_NAME}_{monotonicities}/{NUM_TERMS_NAME}
    num_terms_op_name = '{}_{}/{}_{}/{}'.format(
        premade_lib.RTL_LAYER_NAME,
        submodel_idx,
        rtl_layer.RTL_KFL_NAME,
        monotonicities,
        kfll.NUM_TERMS_NAME,
    )
    num_terms = sess.run(
        graph.get_operation_by_name(num_terms_op_name).outputs[0])

    # inputs_for_units
    # {RTL_LAYER_NAME}_{submodel_index}/
    # {INPUTS_FOR_UNITS_PREFIX}_{monotonicities}
    inputs_for_units_op_name = '{}_{}/{}_{}'.format(
        premade_lib.RTL_LAYER_NAME, submodel_idx,
        rtl_layer.INPUTS_FOR_UNITS_PREFIX, monotonicities)
    inputs_for_units = sess.run(
        graph.get_operation_by_name(inputs_for_units_op_name).outputs[0])

    # Make a unique kfl for each unit.
    units = inputs_for_units.shape[0]
    for i in range(units):
      # Shape the flat weights, scale, and bias parameters based on the
      # calculated lattice_sizes, units, dims, and num_terms.
      weights = np.reshape(kfl_kernel,
                           (1, lattice_sizes, units * dims, num_terms))
      scale = np.reshape(kfl_scale, (units, num_terms))
      bias = np.reshape(kfl_bias, (units))

      # Gather input nodes for lattice node.
      indices = inputs_for_units[i]
      input_nodes = [flattened_calibration_nodes[index] for index in indices]

      kfl_node = model_info.KroneckerFactoredLatticeNode(
          input_nodes=input_nodes, weights=weights, scale=scale, bias=bias)
      submodel_kfl_nodes[submodel_key] = kfl_node
      submodel_key += 1
  return submodel_key, submodel_kfl_nodes


def _create_rtl_submodel_lattice_nodes(sess, ops, graph,
                                       flattened_calibration_nodes,
                                       submodel_idx, submodel_key):
  """Returns next key and map from key+unit to LatticeNode."""
  submodel_lattice_nodes = {}
  # Lattice kernel weights.
  # {RTL_LAYER_NAME}_{submodel_idx}/
  # {RTL_LATTICE_NAME}_{monotonicities}/{LATTICE_KERNEL_NAME}
  lattice_kernel_op_re = '^{}_{}/{}_(.*)/{}/Read/ReadVariableOp$'.format(
      premade_lib.RTL_LAYER_NAME,
      submodel_idx,
      rtl_layer.RTL_LATTICE_NAME,
      lattice_layer.LATTICE_KERNEL_NAME,
  )
  for lattice_kernel_op, monotonicities in _match_op(ops, lattice_kernel_op_re):
    lattice_kernel = sess.run(
        graph.get_operation_by_name(lattice_kernel_op).outputs[0])

    # Lattice sizes.
    # {RTL_LAYER_NAME}_{submodel_idx}/
    # {RTL_LATTICE_NAME}_{monotonicities}/{LATTICE_SIZES_NAME}
    lattice_sizes_op_name = '{}_{}/{}_{}/{}'.format(
        premade_lib.RTL_LAYER_NAME, submodel_idx, rtl_layer.RTL_LATTICE_NAME,
        monotonicities, lattice_layer.LATTICE_SIZES_NAME)
    lattice_sizes = sess.run(
        graph.get_operation_by_name(
            lattice_sizes_op_name).outputs[0]).flatten()

    # inputs_for_units
    # {RTL_LAYER_NAME}_{submodel_index}/
    # {INPUTS_FOR_UNITS_PREFIX}_{monotonicities}
    inputs_for_units_op_name = '{}_{}/{}_{}'.format(
        premade_lib.RTL_LAYER_NAME, submodel_idx,
        rtl_layer.INPUTS_FOR_UNITS_PREFIX, monotonicities)
    inputs_for_units = sess.run(
        graph.get_operation_by_name(inputs_for_units_op_name).outputs[0])

    # Make a unique lattice for each unit.
    units = inputs_for_units.shape[0]
    for i in range(units):
      # Shape the flat lattice parameters based on the calculated lattice
      # sizes.
      weights = np.reshape(lattice_kernel[:, i], lattice_sizes)

      # Gather input nodes for lattice node.
      indices = inputs_for_units[i]
      input_nodes = [flattened_calibration_nodes[index] for index in indices]

      lattice_node = model_info.LatticeNode(
          input_nodes=input_nodes, weights=weights)
      submodel_lattice_nodes[submodel_key] = lattice_node
      submodel_key += 1
  return submodel_key, submodel_lattice_nodes


def _create_rtl_lattice_nodes(sess, ops, graph, calibration_nodes_map,
                              kronecker_factored):
  """Returns a map from lattice_submodel_index to lattice type Node."""
  lattice_nodes = {}
  lattice_submodel_index = 0
  # Feature name in concat op.
  # {RTL_INPUT_NAME}_{feature_name}:0
  feature_name_prog = re.compile('^{}_(.*):0$'.format(
      premade_lib.RTL_INPUT_NAME))
  # RTL Layer identified by single concat op per submodel.
  # {RTL_LAYER_NAME}_{submodel_idx}/RTL_CONCAT_NAME
  rtl_layer_concat_op_re = '^{}_(.*)/{}$'.format(premade_lib.RTL_LAYER_NAME,
                                                 rtl_layer.RTL_CONCAT_NAME)
  for concat_op_name, submodel_idx in _match_op(ops, rtl_layer_concat_op_re):
    # First we reconstruct the flattened calibration outputs for this submodel.
    concat_op = graph.get_operation_by_name(concat_op_name)
    input_names = [input_tensor.name for input_tensor in concat_op.inputs]
    names_in_flattened_order = []
    for input_name in input_names:
      match = feature_name_prog.match(input_name)
      if match:
        names_in_flattened_order.append(match.group(1))
    flattened_calibration_nodes = []
    for feature_name in names_in_flattened_order:
      flattened_calibration_nodes.extend(calibration_nodes_map[feature_name])

    if kronecker_factored:
      node_fn = _create_rtl_submodel_kronecker_factored_lattice_nodes
    else:
      node_fn = _create_rtl_submodel_lattice_nodes
    lattice_submodel_index, submodel_lattice_nodes = node_fn(
        sess, ops, graph, flattened_calibration_nodes, submodel_idx,
        lattice_submodel_index)
    lattice_nodes.update(submodel_lattice_nodes)
  return lattice_nodes


def _create_output_combination_node(sess, ops, graph, submodel_output_nodes):
  """Returns None, a LinearNode, or a MeanNode."""
  output_combination_node = None
  # Mean node is only added for ensemble models.
  if len(submodel_output_nodes) > 1:
    input_nodes = [
        submodel_output_nodes[idx]
        for idx in sorted(submodel_output_nodes.keys(), key=int)
    ]

    # Linear coefficients.
    # {LINEAR_LAYER_COMBINATION_NAME}/{LINEAR_LAYER_KERNEL_NAME}
    linear_combination_kernel_op = '{}/{}/Read/ReadVariableOp'.format(
        premade_lib.OUTPUT_LINEAR_COMBINATION_LAYER_NAME,
        linear_layer.LINEAR_LAYER_KERNEL_NAME,
    )
    if linear_combination_kernel_op in ops:
      coefficients = sess.run(
          graph.get_operation_by_name(
              linear_combination_kernel_op).outputs[0]).flatten()

      # Bias term.
      # {OUTPUT_LINEAR_COMBINATION_LAYER_NAME}/{LINEAR_LAYER_BIAS_NAME}
      bias_op = '{}/{}/Read/ReadVariableOp'.format(
          premade_lib.OUTPUT_LINEAR_COMBINATION_LAYER_NAME,
          linear_layer.LINEAR_LAYER_BIAS_NAME,
      )
      if bias_op in ops:
        bias = sess.run(graph.get_operation_by_name(bias_op).outputs[0])
      else:
        bias = 0.0

      linear_combination_node = model_info.LinearNode(
          input_nodes=input_nodes, coefficients=coefficients, bias=bias)
      output_combination_node = linear_combination_node
    else:
      average_node = model_info.MeanNode(input_nodes=input_nodes)
      output_combination_node = average_node
  return output_combination_node


def _create_output_calibration_node(sess, ops, graph, input_node):
  """Returns a PWLCalibrationNode."""
  output_calibration_node = None
  # Lengths (deltas between keypoints).
  # {OUTPUT_CALIB_LAYER_NAME}/{LENGTHS_NAME}
  lengths_op = '{}/{}'.format(
      premade_lib.OUTPUT_CALIB_LAYER_NAME,
      pwl_calibration_layer.LENGTHS_NAME,
  )
  if lengths_op in ops:
    # Interpolation keypoints does not inlcude the last input keypoint.
    # {OUTPUT_CALIB_LAYER_NAME}/{INTERPOLATION_KEYPOINTS_NAME}
    keypoints_op = '{}/{}'.format(
        premade_lib.OUTPUT_CALIB_LAYER_NAME,
        pwl_calibration_layer.INTERPOLATION_KEYPOINTS_NAME,
    )

    # Output keypoints. We need to call the varible read op.
    # {OUTPUT_CALIB_LAYER_NAME}/{PWL_CALIBRATION_KERNEL_NAME}
    kernel_op = '{}/{}/Read/ReadVariableOp'.format(
        premade_lib.OUTPUT_CALIB_LAYER_NAME,
        pwl_calibration_layer.PWL_CALIBRATION_KERNEL_NAME,
    )

    (lengths, keypoints, kernel) = sess.run(
        (graph.get_operation_by_name(lengths_op).outputs[0],
         graph.get_operation_by_name(keypoints_op).outputs[0],
         graph.get_operation_by_name(kernel_op).outputs[0]))
    output_keypoints = np.cumsum(kernel.flatten())

    # Add the last keypoint to the keypoint list.
    input_keypoints = np.append(keypoints, keypoints[-1] + lengths[-1])

    output_calibration_node = model_info.PWLCalibrationNode(
        input_node=input_node,
        input_keypoints=input_keypoints,
        output_keypoints=output_keypoints,
        default_input=None,
        default_output=None)
  return output_calibration_node


# TODO: add support for KFL in RTL Layer
def get_model_graph(saved_model_path, tag='serve'):
  """Returns all layers and parameters used in a saved model as a graph.

  The returned graph is not a TF graph, rather a graph of python object that
  encodes the model structure and includes trained model parameters. The graph
  can be used by the `tfl.visualization` module for plotting and other
  visualization and analysis.

  Example:

  ```python
  model_graph = estimators.get_model_graph(saved_model_path)
  visualization.plot_feature_calibrator(model_graph, "feature_name")
  visualization.plot_all_calibrators(model_graph)
  visualization.draw_model_graph(model_graph)
  ```

  Args:
    saved_model_path: Path to the saved model.
    tag: Saved model tag for loading.

  Returns:
    A `model_info.ModelGraph` object that includes the model graph.
  """
  # List of all the nodes in the model.
  nodes = []

  # Dict from submodel index to the output node of the submodel.
  submodel_output_nodes = {}

  tf.compat.v1.reset_default_graph()
  with tf.compat.v1.Session() as sess:
    tf.compat.v1.saved_model.loader.load(sess, [tag], saved_model_path)
    graph = tf.compat.v1.get_default_graph()
    ops = [op.name for op in graph.get_operations()]

    # Dict from feature name to corresponding InputFeatureNode object.
    feature_nodes = _create_feature_nodes(sess, ops, graph)
    nodes.extend(feature_nodes.values())

    # Categorical Calibration Nodes.
    categorical_calibration_nodes = _create_categorical_calibration_nodes(
        sess, ops, graph, feature_nodes)
    for calibration_nodes in categorical_calibration_nodes.values():
      nodes.extend(calibration_nodes)

    # PWL Calibration Nodes.
    pwl_calibration_nodes = _create_pwl_calibration_nodes(
        sess, ops, graph, feature_nodes)
    for calibration_nodes in pwl_calibration_nodes.values():
      nodes.extend(calibration_nodes)

    # Dict from feature name to list of calibration nodes (Categorical and PWL).
    calibration_nodes_map = {}
    calibration_nodes_map.update(categorical_calibration_nodes)
    calibration_nodes_map.update(pwl_calibration_nodes)
    # Dict from submodel index to a list of calibrated inputs for the submodel.
    submodel_input_nodes = _create_submodel_input_map(ops,
                                                      calibration_nodes_map)

    # Linear nodes
    linear_nodes = _create_linear_nodes(sess, ops, graph, submodel_input_nodes)
    submodel_output_nodes.update(linear_nodes)
    nodes.extend(linear_nodes.values())

    # Ensemble Lattice nodes.
    lattice_nodes = _create_lattice_nodes(sess, ops, graph,
                                          submodel_input_nodes)
    submodel_output_nodes.update(lattice_nodes)
    nodes.extend(lattice_nodes.values())

    # Ensemble Kronecker Factored Lattice nodes.
    kfl_nodes = _create_kronecker_factored_lattice_nodes(
        sess, ops, graph, submodel_input_nodes)
    submodel_output_nodes.update(kfl_nodes)
    nodes.extend(kfl_nodes.values())

    # RTL Lattice nodes.
    rtl_lattice_nodes = _create_rtl_lattice_nodes(
        sess, ops, graph, calibration_nodes_map, kronecker_factored=False)
    submodel_output_nodes.update(rtl_lattice_nodes)
    nodes.extend(rtl_lattice_nodes.values())

    # RTL Kronecker Factored Lattice nodes.
    kfl_rtl_nodes = _create_rtl_lattice_nodes(
        sess, ops, graph, calibration_nodes_map, kronecker_factored=True)
    submodel_output_nodes.update(kfl_rtl_nodes)
    nodes.extend(kfl_rtl_nodes.values())

    # Output combination node.
    model_output_node = _create_output_combination_node(sess, ops, graph,
                                                        submodel_output_nodes)
    if model_output_node:
      nodes.append(model_output_node)
    else:
      model_output_node = list(submodel_output_nodes.values())[0]

    # Output calibration node.
    output_calibration_node = _create_output_calibration_node(
        sess, ops, graph, model_output_node)
    if output_calibration_node:
      nodes.append(output_calibration_node)
      model_output_node = output_calibration_node

  return model_info.ModelGraph(nodes=nodes, output_node=model_output_node)
