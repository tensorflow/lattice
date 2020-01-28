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
import itertools
import json
import os
import re
import time

from . import categorical_calibration_layer
from . import configs
from . import lattice_layer
from . import lattice_lib
from . import linear_layer
from . import model_info
from . import pwl_calibration_layer
from . import pwl_calibration_lib

from absl import logging
import enum
import numpy as np
import six
import tensorflow as tf

from tensorflow.python.feature_column import feature_column as fc  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.feature_column import feature_column_v2 as fc2  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.keras.utils import losses_utils  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.training import training_util  # pylint: disable=g-direct-tensorflow-import
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head

# TODO: support multi dim inputs.
# TODO: support multi dim output.
# TODO: add linear layer regularizers.
# TODO: add examples in docs.
# TODO: make _REPEATED_PAIR_DISCOUNT_IN_CRYSTALS_SCORE config param

# Layer names used for layers in the canned models.
INPUT_LAYER_NAME = 'tfl_input'
CALIB_LAYER_NAME = 'tfl_calib'
LATTICE_LAYER_NAME = 'tfl_lattice'
LINEAR_LAYER_NAME = 'tfl_linear'
OUTPUT_CALIB_LAYER_NAME = 'tfl_output_calib'

# Prefix for passthrough (identity) nodes for shared calibration.
# These nodes pass shared calibrated values to submodels in an ensemble.
CALIB_PASSTHROUGH_NAME = 'tfl_calib_passthrough'

# Feed and fetch names for the model.
FEATURES_SCOPE = 'features'
OUTPUT_NAME = 'output'

# File to store and load feature keypoints.
_KEYPOINTS_FILE = 'keypoints.json'

# File to store and load lattice ensemble structure.
_ENSEMBLE_STRUCTURE_FILE = 'ensemble_structure.json'

# Name for label keypoints in keypoints file.
_LABEL_FEATURE_NAME = '__label__'

# Prefix for defining feature calibrator regularizers.
_INPUT_CALIB_REGULARIZER_PREFIX = 'calib_'

# Prefix for defining output calibrator regularizers.
_OUTPUT_CALIB_REGULARIZER_PREFIX = 'output_calib_'

# Pooling interval and maximum wait time for workers waiting for files.
_MAX_WAIT_TIME = 1200
_POLL_INTERVAL_SECS = 10

# Weight of laplacian in feature importance for the crystal algorithm.
_LAPLACIAN_WEIGHT_IN_IMPORTANCE = 6.0

# Discount amount for repeated co-occurrence of pairs of features in crystals.
_REPEATED_PAIR_DISCOUNT_IN_CRYSTALS_SCORE = 0.5

# Maximum number of swaps for the crystals algorithm.
_MAX_CRYSTALS_SWAPS = 1000


class WaitTimeOutError(Exception):
  """Timeout error when waiting for a file."""
  pass


def _poll_for_file(filename):
  """Waits and polls for a file until it exists."""
  start = time.time()
  while not tf.io.gfile.exists(filename):
    time.sleep(_POLL_INTERVAL_SECS)
    if time.time() - start > _MAX_WAIT_TIME:
      raise WaitTimeOutError('Waiting for file {} timed-out'.filename)


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
          parsed_features[
              feature_column.name] = feature_column._transform_feature(
                  features).values
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
      concatenated_tensors[k] = np.concatenate(
          [split[k] for split in splits if split[k].size > 0])
    return concatenated_tensors


def _finalize_keypoints(model_config, config, feature_columns,
                        feature_analysis_input_fn, logits_output):
  """Calculates and sets keypoints for input and output calibration.

  Input and label keypoints are calculated, stored in a file and also set in the
  model_config to be used for model construction.

  Args:
    model_config: Model config to be updated.
    config: A `tf.RunConfig` to indicate if worker is chief.
    feature_columns: A list of FeatureColumn's to use for feature parsing.
    feature_analysis_input_fn: An input_fn used to collect feature statistics.
    logits_output: A boolean indicating if model outputs logits.

  Raises:
    ValueError: If keypoints mode is invalid.
  """
  if not feature_analysis_input_fn:
    return

  keypoints_filename = os.path.join(config.model_dir, _KEYPOINTS_FILE)
  if ((config is None or config.is_chief) and
      not tf.io.gfile.exists(keypoints_filename)):
    with tf.Graph().as_default():
      features, label = feature_analysis_input_fn()
      features = transform_features(features, feature_columns)
      features[_LABEL_FEATURE_NAME] = label
      features = _materialize_locally(features)

    feature_keypoints = {}
    for feature_name, feature_values in six.iteritems(features):
      feature_values = feature_values.flatten()

      if feature_name == _LABEL_FEATURE_NAME:
        # Default feature_values to [0, ... n_class-1] if string label.
        if label.dtype == tf.string:
          feature_values = np.arange(len(set(feature_values)))
        num_keypoints = model_config.output_calibration_num_keypoint
        keypoints = model_config.output_initialization
        clip_min = model_config.output_min
        clip_max = model_config.output_max
        default_value = None
      else:
        feature_config = model_config.feature_config_by_name(feature_name)
        if feature_config.num_buckets:
          # Skip categorical features.
          continue
        num_keypoints = feature_config.pwl_calibration_num_keypoints
        keypoints = feature_config.pwl_calibration_input_keypoints
        clip_min = feature_config.pwl_calibration_clip_min
        clip_max = feature_config.pwl_calibration_clip_max
        default_value = feature_config.default_value

      # Remove default values before calculating stats.
      feature_values = feature_values[feature_values != default_value]

      if np.isnan(feature_values).any():
        raise ValueError(
            'NaN values were observed for numeric feature `{}`. '
            'Consider replacing the values in transform or input_fn.'.format(
                feature_name))

      # Before calculating keypoints, clip values as requested.
      # Add min and max to the value list to make sure min/max in values match
      # the requested range.
      if clip_min is not None:
        feature_values = np.maximum(feature_values, clip_min)
        feature_values = np.append(feature_values, clip_min)
      if clip_max is not None:
        feature_values = np.minimum(feature_values, clip_max)
        feature_values = np.append(feature_values, clip_max)

      # Remove duplicate values before calculating stats.
      feature_values = np.unique(feature_values)

      if isinstance(keypoints, str):
        if keypoints == 'quantiles':
          if (feature_name != _LABEL_FEATURE_NAME and
              feature_values.size < num_keypoints):
            logging.info(
                'Not enough unique values observed for feature `%s` to '
                'construct %d keypoints for pwl calibration. Using %d unique '
                'values as keypoints.', feature_name, num_keypoints,
                feature_values.size)
            num_keypoints = feature_values.size
          quantiles = np.quantile(
              feature_values,
              np.linspace(0., 1., num_keypoints),
              interpolation='nearest')
          feature_keypoints[feature_name] = [float(x) for x in quantiles]
        elif keypoints == 'uniform':
          linspace = np.linspace(
              np.min(feature_values), np.max(feature_values), num_keypoints)
          feature_keypoints[feature_name] = [float(x) for x in linspace]
        else:
          raise ValueError(
              'Invalid keypoint generation mode: {}'.format(keypoints))
      else:
        # Keypoints are explicitly provided in the config.
        feature_keypoints[feature_name] = [float(x) for x in keypoints]

    # Save keypoints to file as the chief worker.
    tmp_keypoints_filename = keypoints_filename + 'tmp'
    with tf.io.gfile.GFile(tmp_keypoints_filename, 'w') as keypoints_file:
      keypoints_file.write(json.dumps(feature_keypoints, indent=2))
    tf.io.gfile.rename(tmp_keypoints_filename, keypoints_filename)
  else:
    # Non-chief workers read the keypoints from file.
    _poll_for_file(keypoints_filename)
    with tf.io.gfile.GFile(keypoints_filename) as keypoints_file:
      feature_keypoints = json.loads(keypoints_file.read())

  if _LABEL_FEATURE_NAME in feature_keypoints:
    output_init = feature_keypoints.pop(_LABEL_FEATURE_NAME)
    if logits_output and isinstance(model_config.output_initialization, str):
      # If model is expected to produce logits, initialize linearly in the
      # range [-2, 2], ignoring the label distribution.
      model_config.output_initialization = [
          float(x) for x in np.linspace(
              -2, 2, model_config.output_calibration_num_keypoint)
      ]
    else:
      model_config.output_initialization = output_init

  for feature_name, keypoints in feature_keypoints.items():
    model_config.feature_config_by_name(
        feature_name).pwl_calibration_input_keypoints = keypoints


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


def _set_random_lattice_ensemble(model_config, feature_names):
  """Sets random lattice ensemble in the given model_config."""
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


def _get_torsions_and_laplacians(prefitting_model_config, prefitting_estimator,
                                 feature_names):
  """Returns average torsion and laplacian regularizers in prefitted model."""
  num_fatures = len(feature_names)
  laplacians = [[] for _ in range(num_fatures)]
  torsions = [[[] for _ in range(num_fatures)] for _ in range(num_fatures)]
  for (lattice_index, lattice) in enumerate(prefitting_model_config.lattices):
    # Get normalized lattice weights.
    lattice_kernel_variable_name = '{}_{}/{}'.format(
        LATTICE_LAYER_NAME, lattice_index, lattice_layer.LATTICE_KERNEL_NAME)
    weights = prefitting_estimator.get_variable_value(
        lattice_kernel_variable_name)
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


def _set_final_crystal_lattices(model_config, feature_names,
                                prefitting_model_config, prefitting_estimator):
  """Sets the lattice ensemble in model_config based on a prefitted model."""
  torsions, laplacians = _get_torsions_and_laplacians(
      prefitting_model_config=prefitting_model_config,
      prefitting_estimator=prefitting_estimator,
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

  model_config.lattices = [[
      feature_names[features_index] for features_index in lattice
  ] for lattice in lattices]


def _set_crystals_lattice_ensemble(model_config, feature_names, label_dimension,
                                   feature_columns, head, prefitting_input_fn,
                                   prefitting_optimizer, prefitting_steps,
                                   config, dtype):
  """Sets the lattice ensemble in model_config using the crystals algorithm."""
  if prefitting_input_fn is None:
    raise ValueError('prefitting_input_fn must be set for crystals models')

  prefitting_model_config = copy.deepcopy(model_config)
  _set_all_pairs_cover_lattices(
      prefitting_model_config=prefitting_model_config,
      feature_names=feature_names)

  # Trim the model for faster prefitting.
  for feature_config in prefitting_model_config.feature_configs:
    feature_config.lattice_size = 2
    # Unimodality requires lattice_size > 2.
    feature_config.unimodality = 0

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

  config = tf.estimator.RunConfig(
      keep_checkpoint_max=1,
      save_summary_steps=0,
      save_checkpoints_steps=10000000,
      tf_random_seed=config.tf_random_seed if config is not None else 42)
  logging.info('Creating the prefitting estimator.')
  prefitting_estimator = tf.estimator.Estimator(
      model_fn=prefitting_model_fn, config=config)
  logging.info('Training the prefitting estimator.')
  prefitting_estimator.train(
      input_fn=prefitting_input_fn, steps=prefitting_steps)
  _set_final_crystal_lattices(
      feature_names=feature_names,
      model_config=model_config,
      prefitting_model_config=prefitting_model_config,
      prefitting_estimator=prefitting_estimator)
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
    if model_config.lattices == 'random':
      _set_random_lattice_ensemble(
          model_config=model_config, feature_names=feature_names)
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
    else:
      raise ValueError('Unsupported ensemble structure: {}'.format(
          model_config.lattices))
    if model_config.fix_ensemble_for_2d_constraints:
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
    if (feature_config.num_buckets and
        isinstance(feature_config.monotonicity, list)):
      if not feature_config.vocabulary_list:
        raise ValueError('Vocabulary list must be provided to use categorical'
                         'monotonicities.')
      if not all(
          isinstance(m, tuple) and len(m) == 2
          for m in feature_config.monotonicity):
        raise ValueError(
            'Monotonicities should be a list of pairs (tuples): {}'.format(
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


def _input_calibration_regularizers(model_config, feature_config):
  """Returns pwl layer regularizers defined in the model and feature configs."""
  regularizer_configs = []
  regularizer_configs.extend(feature_config.regularizer_configs or [])
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


class _LayerOutputRange(enum.Enum):
  """Enum to indicate the output range based on the input of the next layers."""
  MODEL_OUTPUT = 1
  INPUT_TO_LATTICE = 2
  INPUT_TO_FINAL_CALIBRATION = 3


def _output_range(layer_output_range, model_config, feature_config=None):
  """Returns min/max/init_min/init_max for a given output range."""
  if layer_output_range == _LayerOutputRange.INPUT_TO_LATTICE:
    if feature_config is None:
      raise ValueError('Expecting feature config for lattice inputs.')
    output_init_min = output_min = 0.0
    output_init_max = output_max = feature_config.lattice_size - 1.0
  elif layer_output_range == _LayerOutputRange.MODEL_OUTPUT:
    output_min = model_config.output_min
    output_max = model_config.output_max
    output_init_min = np.min(model_config.output_initialization)
    output_init_max = np.max(model_config.output_initialization)
  elif layer_output_range == _LayerOutputRange.INPUT_TO_FINAL_CALIBRATION:
    output_init_min = output_min = 0.0
    output_init_max = output_max = 1.0
  else:
    raise ValueError('Unsupported layer output range.')
  return output_min, output_max, output_init_min, output_init_max


def _input_layer(feature_configs, dtype):
  """Creates a calibration layer."""
  input_layer = {}
  for feature_config in feature_configs:
    layer_name = '{}_{}'.format(INPUT_LAYER_NAME, feature_config.name)
    if feature_config.num_buckets:
      input_layer[feature_config.name] = tf.keras.Input(
          shape=(1,), dtype=tf.int32, name=layer_name)
    else:
      input_layer[feature_config.name] = tf.keras.Input(
          shape=(1,), dtype=dtype, name=layer_name)
  return input_layer


def _calibration_layers(calibration_input_layer, feature_configs, model_config,
                        layer_output_range, submodels, separate_calibrators,
                        dtype):
  """Creates a calibration layer for `submodels` as list of list of features."""
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

  calibration_output = {}
  for feature_config in feature_configs:
    feature_name = feature_config.name
    units = max(calibration_last_index[feature_name], 1)
    calibration_input = calibration_input_layer[feature_name]
    layer_name = '{}_{}'.format(CALIB_LAYER_NAME, feature_name)

    (output_min, output_max, output_init_min,
     output_init_max) = _output_range(layer_output_range, model_config,
                                      feature_config)

    if feature_config.num_buckets:
      kernel_initializer = tf.compat.v1.random_uniform_initializer(
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
              dtype=dtype,
              name=layer_name)(calibration_input))
    else:
      kernel_regularizer = _input_calibration_regularizers(
          model_config, feature_config)
      monotonicity = feature_config.monotonicity
      if (pwl_calibration_lib.canonicalize_monotonicity(monotonicity) == 0 and
          feature_config.pwl_calibration_always_monotonic):
        monotonicity = 1
      kernel_initializer = pwl_calibration_layer.UniformOutputInitializer(
          output_min=output_init_min,
          output_max=output_init_max,
          monotonicity=monotonicity)
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
              dtype=dtype,
              name=layer_name)(calibration_input))
    if units == 1:
      calibration_output[feature_name] = [calibrated]
    else:
      calibration_output[feature_name] = tf.split(calibrated, units, axis=1)

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


def _linear_layer(linear_input, feature_configs, model_config, weighted_average,
                  submodel_index, dtype):
  """Creates a linear layer initialized to be an average."""
  layer_name = '{}_{}'.format(LINEAR_LAYER_NAME, submodel_index)

  linear_input = tf.keras.layers.Concatenate(axis=1)(linear_input)
  num_input_dims = len(feature_configs)
  kernel_initializer = tf.compat.v1.constant_initializer(
      [1.0 / num_input_dims] * num_input_dims)
  bias_initializer = tf.compat.v1.constant_initializer(0)

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


def _lattice_layer(lattice_input, feature_configs, model_config,
                   layer_output_range, submodel_index, is_inside_ensemble,
                   dtype):
  """Creates a lattice layer."""
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
  lattice_regularizers = _lattice_regularizers(model_config, feature_configs)

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
      kernel_regularizer=lattice_regularizers,
      kernel_initializer=kernel_initializer,
      dtype=dtype,
      name=layer_name)(
          lattice_input)


def _output_calibration_layer(output_calibration_input, model_config, dtype):
  """Creates a monotonic output calibration layer with inputs range [0, 1]."""
  # kernel format: bias followed by diffs between consecutive keypoint outputs.
  kernel_init_values = np.ediff1d(
      model_config.output_initialization,
      to_begin=model_config.output_initialization[0])
  input_keypoints = np.linspace(0.0, 1.0, num=len(kernel_init_values))
  kernel_initializer = tf.compat.v1.constant_initializer(kernel_init_values)
  kernel_regularizer = _output_calibration_regularizers(model_config)
  return pwl_calibration_layer.PWLCalibration(
      input_keypoints=input_keypoints,
      output_min=model_config.output_min,
      output_max=model_config.output_max,
      kernel_initializer=kernel_initializer,
      kernel_regularizer=kernel_regularizer,
      monotonicity=1,
      dtype=dtype,
      name=OUTPUT_CALIB_LAYER_NAME)(
          output_calibration_input)


def _calibrated_lattice_ensemble_model_fn(features, labels, label_dimension,
                                          feature_columns, mode, head,
                                          model_config, optimizer, config,
                                          dtype):
  """Calibrated Lattice Ensemble Model."""
  del config
  if label_dimension != 1:
    ValueError('Only 1-dimensional output is supported.')

  # Get input tensors and corresponding feature configs.
  transformed_features = transform_features(features, feature_columns)
  feature_names = list(transformed_features.keys())
  feature_configs = [
      model_config.feature_config_by_name(feature_name)
      for feature_name in feature_names
  ]
  input_layer = _input_layer(feature_configs=feature_configs, dtype=dtype)

  submodels_inputs = _calibration_layers(
      calibration_input_layer=input_layer,
      feature_configs=feature_configs,
      model_config=model_config,
      layer_output_range=_LayerOutputRange.INPUT_TO_LATTICE,
      submodels=model_config.lattices,
      separate_calibrators=model_config.separate_calibrators,
      dtype=dtype)

  lattice_outputs = []
  for submodel_index, (lattice_feature_names, lattice_input) in enumerate(
      zip(model_config.lattices, submodels_inputs)):
    lattice_feature_configs = [
        model_config.feature_config_by_name(feature_name)
        for feature_name in lattice_feature_names
    ]

    lattice_layer_output_range = (
        _LayerOutputRange.INPUT_TO_FINAL_CALIBRATION
        if model_config.output_calibration else _LayerOutputRange.MODEL_OUTPUT)
    lattice_outputs.append(
        _lattice_layer(
            lattice_input=lattice_input,
            feature_configs=lattice_feature_configs,
            model_config=model_config,
            layer_output_range=lattice_layer_output_range,
            submodel_index=submodel_index,
            is_inside_ensemble=True,
            dtype=dtype))

  averaged_lattice_output = tf.keras.layers.Average()(lattice_outputs)
  if model_config.output_calibration:
    model_output = _output_calibration_layer(
        output_calibration_input=averaged_lattice_output,
        model_config=model_config,
        dtype=dtype)
  else:
    model_output = averaged_lattice_output

  input_tensors = [
      transformed_features[feature_name] for feature_name in feature_names
  ]
  inputs = [input_layer[feature_name] for feature_name in feature_names]
  training = (mode == tf.estimator.ModeKeys.TRAIN)
  model = tf.keras.Model(inputs=inputs, outputs=model_output)
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
    ValueError('Only 1-dimensional output is supported.')

  # Get input tensors and corresponding feature configs.
  transformed_features = transform_features(features, feature_columns)
  feature_names = list(transformed_features.keys())
  feature_configs = [
      model_config.feature_config_by_name(feature_name)
      for feature_name in feature_names
  ]
  input_layer = _input_layer(feature_configs=feature_configs, dtype=dtype)
  submodels_inputs = _calibration_layers(
      calibration_input_layer=input_layer,
      feature_configs=feature_configs,
      model_config=model_config,
      layer_output_range=_LayerOutputRange.INPUT_TO_LATTICE,
      submodels=[[feature_column.name for feature_column in feature_columns]],
      separate_calibrators=False,
      dtype=dtype)

  lattice_layer_output_range = (
      _LayerOutputRange.INPUT_TO_FINAL_CALIBRATION
      if model_config.output_calibration else _LayerOutputRange.MODEL_OUTPUT)
  lattice_output = _lattice_layer(
      lattice_input=submodels_inputs[0],
      feature_configs=feature_configs,
      model_config=model_config,
      layer_output_range=lattice_layer_output_range,
      submodel_index=0,
      is_inside_ensemble=False,
      dtype=dtype)

  if model_config.output_calibration:
    model_output = _output_calibration_layer(
        output_calibration_input=lattice_output,
        model_config=model_config,
        dtype=dtype)
  else:
    model_output = lattice_output

  input_tensors = [
      transformed_features[feature_name] for feature_name in feature_names
  ]
  inputs = [input_layer[feature_name] for feature_name in feature_names]
  training = (mode == tf.estimator.ModeKeys.TRAIN)
  model = tf.keras.Model(inputs=inputs, outputs=model_output)
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


def _calibrated_linear_model_fn(features, labels, label_dimension,
                                feature_columns, mode, head, model_config,
                                optimizer, config, dtype):
  """Calibrated Linear Model."""
  del config
  if label_dimension != 1:
    ValueError('Only 1-dimensional output is supported.')

  # Get input tensors and corresponding feature configs.
  transformed_features = transform_features(features, feature_columns)
  feature_names = list(transformed_features.keys())
  feature_configs = [
      model_config.feature_config_by_name(feature_name)
      for feature_name in feature_names
  ]
  input_layer = _input_layer(feature_configs=feature_configs, dtype=dtype)

  calibration_layer_output_range = (
      _LayerOutputRange.INPUT_TO_FINAL_CALIBRATION
      if model_config.output_calibration else _LayerOutputRange.MODEL_OUTPUT)
  submodels_inputs = _calibration_layers(
      calibration_input_layer=input_layer,
      feature_configs=feature_configs,
      model_config=model_config,
      layer_output_range=calibration_layer_output_range,
      submodels=[[feature_column.name for feature_column in feature_columns]],
      separate_calibrators=False,
      dtype=dtype)

  weighted_average = (
      model_config.output_min is not None or
      model_config.output_max is not None or model_config.output_calibration)
  linear_output = _linear_layer(
      linear_input=submodels_inputs[0],
      feature_configs=feature_configs,
      model_config=model_config,
      weighted_average=weighted_average,
      submodel_index=0,
      dtype=dtype)

  if model_config.output_calibration:
    model_output = _output_calibration_layer(
        output_calibration_input=linear_output,
        model_config=model_config,
        dtype=dtype)
  else:
    model_output = linear_output

  input_tensors = [
      transformed_features[feature_name] for feature_name in feature_names
  ]
  inputs = [input_layer[feature_name] for feature_name in feature_names]
  training = (mode == tf.estimator.ModeKeys.TRAIN)
  model = tf.keras.Model(inputs=inputs, outputs=model_output)
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
               loss_reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
               dtype=tf.float32):
    """Initializes a `CannedClassifier` instance.

    Args:
      model_config: Model configuration object describing model architecutre.
        Should be one of the model configs in `tfl.configs`.
      feature_columns: An iterable containing all the feature columns used by
        the model.
      feature_analysis_input_fn: An input_fn used to calculate statistics about
        features and labels in order to setup calibration keypoint and values.
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
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example. If it is a string, it is
        used as a key to fetch weight tensor from the `features`. If it is a
        `_NumericColumn`, raw tensor is fetched by key `weight_column.key`, then
        weight_column.normalizer_fn is applied on it to get weight tensor.
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
      dtype: dtype of layers used in the model.
    """
    config = estimator_lib.maybe_overwrite_model_dir_and_session_config(
        config, model_dir)
    model_dir = config.model_dir
    head = head_utils.binary_or_multi_class_head(
        n_classes=n_classes,
        weight_column=weight_column,
        label_vocabulary=label_vocabulary,
        loss_reduction=loss_reduction)
    label_dimension = 1 if n_classes == 2 else n_classes

    model_config = copy.deepcopy(model_config)
    _update_by_feature_columns(model_config, feature_columns)

    _finalize_keypoints(
        model_config=model_config,
        config=config,
        feature_columns=feature_columns,
        feature_analysis_input_fn=feature_analysis_input_fn,
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
               prefitting_input_fn=None,
               model_dir=None,
               label_dimension=1,
               weight_column=None,
               optimizer='Adagrad',
               prefitting_optimizer='Adagrad',
               prefitting_steps=None,
               config=None,
               warm_start_from=None,
               loss_reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE,
               dtype=tf.float32):
    """Initializes a `CannedRegressor` instance.

    Args:
      model_config: Model configuration object describing model architecutre.
        Should be one of the model configs in `tfl.configs`.
      feature_columns: An iterable containing all the feature columns used by
        the model.
      feature_analysis_input_fn: An input_fn used to calculate statistics about
        features and labels in order to setup calibration keypoint and values.
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
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example. If it is a string, it is
        used as a key to fetch weight tensor from the `features`. If it is a
        `_NumericColumn`, raw tensor is fetched by key `weight_column.key`, then
        weight_column.normalizer_fn is applied on it to get weight tensor.
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
      dtype: dtype of layers used in the model.
    """
    config = estimator_lib.maybe_overwrite_model_dir_and_session_config(
        config, model_dir)
    model_dir = config.model_dir
    head = regression_head.RegressionHead(
        label_dimension=label_dimension,
        weight_column=weight_column,
        loss_reduction=loss_reduction)

    model_config = copy.deepcopy(model_config)
    _update_by_feature_columns(model_config, feature_columns)

    _finalize_keypoints(
        model_config=model_config,
        config=config,
        feature_columns=feature_columns,
        feature_analysis_input_fn=feature_analysis_input_fn,
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

    super(CannedRegressor, self).__init__(
        model_fn=model_fn,
        model_dir=model_dir,
        config=config,
        warm_start_from=warm_start_from)


def _match_op(ops, regex):
  """Returns ops that match given regex along with the matched sections."""
  matches = []
  for op in ops:
    op_matches = re.findall(regex, op)
    if op_matches:
      matches.append((op, op_matches[0]))
  return matches


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

  # Dict from feature name to corresponding InputFeatureNode object.
  feature_nodes = {}

  # Dict from submodel index to a list of calibrated inputs for the submodel.
  submodel_input_nodes = collections.defaultdict(list)

  # Dict from submodel index to the output node of the submodel.
  submodel_output_nodes = {}

  tf.compat.v1.reset_default_graph()
  with tf.compat.v1.Session() as sess:
    tf.compat.v1.saved_model.loader.load(sess, [tag], saved_model_path)
    g = tf.compat.v1.get_default_graph()
    ops = [op.name for op in g.get_operations()]

    #############################
    # Create input feature nodes.
    #############################

    # Extract list of features from the graph.
    # {FEATURES_SCOPE}/{feature_name}
    feature_op_re = '{}/(.*)'.format(FEATURES_SCOPE)
    for (_, feature_name) in _match_op(ops, feature_op_re):
      category_table_op = 'transform/{}_lookup/Const'.format(feature_name)
      if category_table_op in ops:
        is_categorical = True
        vocabulary_list = sess.run(
            g.get_operation_by_name(category_table_op).outputs[0])
        # Replace byte types with their string values.
        vocabulary_list = [
            str(x.decode()) if isinstance(x, bytes) else str(x)
            for x in vocabulary_list
        ]
      else:
        is_categorical = False
        vocabulary_list = None

      feature_node = model_info.InputFeatureNode(
          name=feature_name,
          is_categorical=is_categorical,
          vocabulary_list=vocabulary_list)
      feature_nodes[feature_name] = feature_node
      nodes.append(feature_node)

    #######################################
    # Create categorical calibration nodes.
    #######################################

    # Get calibrator output values. We need to call the read variable op.
    # {CALIB_LAYER_NAME}_{feature_name}/
    #   {CATEGORICAL_CALIBRATION_KERNEL_NAME}/Read/ReadVariableOp
    kernel_op_re = '^{}_(.*)/{}/Read/ReadVariableOp$'.format(
        CALIB_LAYER_NAME,
        categorical_calibration_layer.CATEGORICAL_CALIBRATION_KERNEL_NAME,
    )
    for kernel_op, feature_name in _match_op(ops, kernel_op_re):
      output_values = sess.run(g.get_operation_by_name(kernel_op).outputs[0])

      # Get default input value if defined.
      # {CALIB_LAYER_NAME}_{feature_name}/
      #   {DEFAULT_INPUT_VALUE_NAME}
      default_input_value_op = '^{}_{}/{}$'.format(
          CALIB_LAYER_NAME,
          feature_name,
          categorical_calibration_layer.DEFAULT_INPUT_VALUE_NAME,
      )
      if default_input_value_op in ops:
        default_input = sess.run(
            g.get_operation_by_name(default_input_value_op).outputs[0])
      else:
        default_input = None

      # Create one calibration node per output dimension of the calibrator.
      categorical_calibration_nodes = []
      for calibration_output_idx in range(output_values.shape[1]):
        categorical_calibration_node = model_info.CategoricalCalibrationNode(
            input_node=feature_nodes[feature_name],
            output_values=output_values[:, calibration_output_idx],
            default_input=default_input)
        categorical_calibration_nodes.append(categorical_calibration_node)
        nodes.append(categorical_calibration_node)

      # Identity passthrough ops that pass this calibration to each submodel.
      # {CALIB_PASSTHROUGH_NAME}_{feature_name}_
      #   {calibration_output_idx}_{submodel_idx}_{submodel_input_idx}
      shared_calib_passthrough_op_re = r'^{}_{}_(\d*)_(\d*)_(\d*)$'.format(
          CALIB_PASSTHROUGH_NAME, feature_name)
      for op, (calibration_output_idx, submodel_idx,
               submodel_input_idx) in _match_op(ops,
                                                shared_calib_passthrough_op_re):
        submodel_input_nodes[submodel_idx].append(
            (submodel_input_idx,
             categorical_calibration_nodes[int(calibration_output_idx)]))

    ###############################
    # Create PWL calibration nodes.
    ###############################

    # Calculate input keypoints.
    # We extract lengh (deltas between keypoints) and kernel interpolation
    # keypoints (which does not include the last keypoint), and then
    # construct the full keypoints list using both.

    # Lengths (deltas between keypoints).
    # {CALIB_LAYER_NAME}_{feature_name}/{LENGTHS_NAME}
    lengths_op_re = '^{}_(.*)/{}$'.format(
        CALIB_LAYER_NAME,
        pwl_calibration_layer.LENGTHS_NAME,
    )
    for lengths_op, feature_name in _match_op(ops, lengths_op_re):
      # Interpolation keypoints does not inlcude the last input keypoint.
      # {CALIB_LAYER_NAME}_{feature_name}/{INTERPOLATION_KEYPOINTS_NAME}
      keypoints_op = '{}_{}/{}'.format(
          CALIB_LAYER_NAME,
          feature_name,
          pwl_calibration_layer.INTERPOLATION_KEYPOINTS_NAME,
      )

      # Output keypoints. We need to call the varible read op.
      # {CALIB_LAYER_NAME}_{feature_name}/{PWL_CALIBRATION_KERNEL_NAME}
      kernel_op = '{}_{}/{}/Read/ReadVariableOp'.format(
          CALIB_LAYER_NAME,
          feature_name,
          pwl_calibration_layer.PWL_CALIBRATION_KERNEL_NAME,
      )

      (lengths, keypoints, kernel) = sess.run(
          (g.get_operation_by_name(lengths_op).outputs[0],
           g.get_operation_by_name(keypoints_op).outputs[0],
           g.get_operation_by_name(kernel_op).outputs[0]))
      output_keypoints = np.cumsum(kernel, axis=0)

      # Add the last keypoint to the keypoint list.
      # TODO: handle cyclic PWL layers.
      input_keypoints = np.append(keypoints, keypoints[-1] + lengths[-1])

      # Get missing/default input value if present:
      # {CALIB_LAYER_NAME}_{feature_name}/{MISSING_INPUT_VALUE_NAME}
      default_input_value_op = '{}_{}/{}'.format(
          CALIB_LAYER_NAME,
          feature_name,
          pwl_calibration_layer.MISSING_INPUT_VALUE_NAME,
      )
      if default_input_value_op in ops:
        default_input = sess.run(
            g.get_operation_by_name(default_input_value_op).outputs[0])[0]
      else:
        default_input = None

      # Find corresponding default/missing output if present.
      # {CALIB_LAYER_NAME}_{feature_name}/{PWL_CALIBRATION_MISSING_OUTPUT_NAME}
      default_output_op = '{}_{}/{}/Read/ReadVariableOp'.format(
          CALIB_LAYER_NAME,
          feature_name,
          pwl_calibration_layer.PWL_CALIBRATION_MISSING_OUTPUT_NAME,
      )
      if default_output_op in ops:
        default_output = sess.run(
            g.get_operation_by_name(default_output_op).outputs[0])
      else:
        default_output = None

      # Create one calibration node per output dimension of the calibrator.
      pwl_calibration_nodes = []
      for calibration_output_idx in range(output_keypoints.shape[1]):
        pwl_calibration_node = model_info.PWLCalibrationNode(
            input_node=feature_nodes[feature_name],
            input_keypoints=input_keypoints,
            output_keypoints=output_keypoints[:, calibration_output_idx],
            default_input=default_input,
            default_output=(None if default_output is None else
                            default_output[:, calibration_output_idx]))
        pwl_calibration_nodes.append(pwl_calibration_node)
        nodes.append(pwl_calibration_node)

      # Identity passthrough ops that pass this calibration to each submodel.
      # {CALIB_PASSTHROUGH_NAME}_{feature_name}_
      #   {calibration_output_idx}_{submodel_idx}_{submodel_input_idx}
      shared_calib_passthrough_op_re = r'^{}_{}_(\d*)_(\d*)_(\d*)$'.format(
          CALIB_PASSTHROUGH_NAME, feature_name)
      for op, (calibration_output_idx, submodel_idx,
               submodel_input_idx) in _match_op(ops,
                                                shared_calib_passthrough_op_re):
        submodel_input_nodes[submodel_idx].append(
            (submodel_input_idx,
             pwl_calibration_nodes[int(calibration_output_idx)]))

    ######################
    # Create linear nodes.
    ######################

    # Linear coefficients.
    # {LINEAR_LAYER_NAME}_{submodel_idx}/{LINEAR_LAYER_KERNEL_NAME}
    linear_kernel_op_re = '^{}_(.*)/{}/Read/ReadVariableOp$'.format(
        LINEAR_LAYER_NAME,
        linear_layer.LINEAR_LAYER_KERNEL_NAME,
    )
    for linear_kernel_op, submodel_idx in _match_op(ops, linear_kernel_op_re):
      coefficients = sess.run(
          g.get_operation_by_name(linear_kernel_op).outputs[0]).flatten()

      # Bias term.
      # {LINEAR_LAYER_NAME}/{LINEAR_LAYER_BIAS_NAME}
      bias_op = '{}/{}/Read/ReadVariableOp'.format(
          LINEAR_LAYER_NAME,
          linear_layer.LINEAR_LAYER_BIAS_NAME,
      )
      if bias_op in ops:
        bias = sess.run(g.get_operation_by_name(bias_op).outputs[0])
      else:
        bias = 0.0

      # Sort input nodes by input index.
      input_nodes = [
          node for _, node in sorted(submodel_input_nodes[submodel_idx])
      ]

      linear_node = model_info.LinearNode(
          input_nodes=input_nodes, coefficients=coefficients, bias=bias)
      submodel_output_nodes[submodel_idx] = linear_node
      nodes.append(linear_node)

    #######################
    # Create lattice nodes.
    #######################

    # Lattice weights.
    # {Lattice_LAYER_NAME}_{submodel_idx}/{LATTICE_KERNEL_NAME}
    lattice_kernel_op_re = '^{}_(.*)/{}/Read/ReadVariableOp$'.format(
        LATTICE_LAYER_NAME,
        lattice_layer.LATTICE_KERNEL_NAME,
    )
    for lattice_kernel_op, submodel_idx in _match_op(ops, lattice_kernel_op_re):
      lattice_kernel = sess.run(
          g.get_operation_by_name(lattice_kernel_op).outputs[0]).flatten()

      # Lattice sizes.
      # {Lattice_LAYER_NAME}_{submodel_idx}/{LATTICE_SIZES_NAME}
      lattice_sizes_op_name = '{}_{}/{}'.format(
          LATTICE_LAYER_NAME, submodel_idx, lattice_layer.LATTICE_SIZES_NAME)
      lattice_sizes = sess.run(
          g.get_operation_by_name(lattice_sizes_op_name).outputs[0]).flatten()

      # Shape the flat lattice parameters based on the calculated lattice sizes.
      weights = np.reshape(lattice_kernel, lattice_sizes)

      # Sort input nodes by input index.
      input_nodes = [
          node for _, node in sorted(submodel_input_nodes[submodel_idx])
      ]

      lattice_node = model_info.LatticeNode(
          input_nodes=input_nodes, weights=weights)
      submodel_output_nodes[submodel_idx] = lattice_node
      nodes.append(lattice_node)

    ###################
    # Create mean node.
    ###################

    # Mean node is only added for ensemble models.
    if len(submodel_output_nodes) > 1:
      input_nodes = [
          submodel_output_nodes[idx]
          for idx in sorted(submodel_output_nodes.keys(), key=int)
      ]
      average_node = model_info.MeanNode(input_nodes=input_nodes)
      nodes.append(average_node)
      model_output_node = average_node
    else:
      model_output_node = list(submodel_output_nodes.values())[0]

    #####################################
    # Create output PWL calibration node.
    #####################################

    # Lengths (deltas between keypoints).
    # {OUTPUT_CALIB_LAYER_NAME}/{LENGTHS_NAME}
    lengths_op = '{}/{}'.format(
        OUTPUT_CALIB_LAYER_NAME,
        pwl_calibration_layer.LENGTHS_NAME,
    )
    if lengths_op in ops:
      # Interpolation keypoints does not inlcude the last input keypoint.
      # {OUTPUT_CALIB_LAYER_NAME}/{INTERPOLATION_KEYPOINTS_NAME}
      keypoints_op = '{}/{}'.format(
          OUTPUT_CALIB_LAYER_NAME,
          pwl_calibration_layer.INTERPOLATION_KEYPOINTS_NAME,
      )

      # Output keypoints. We need to call the varible read op.
      # {OUTPUT_CALIB_LAYER_NAME}/{PWL_CALIBRATION_KERNEL_NAME}
      kernel_op = '{}/{}/Read/ReadVariableOp'.format(
          OUTPUT_CALIB_LAYER_NAME,
          pwl_calibration_layer.PWL_CALIBRATION_KERNEL_NAME,
      )

      (lengths, keypoints, kernel) = sess.run(
          (g.get_operation_by_name(lengths_op).outputs[0],
           g.get_operation_by_name(keypoints_op).outputs[0],
           g.get_operation_by_name(kernel_op).outputs[0]))
      output_keypoints = np.cumsum(kernel.flatten())

      # Add the last keypoint to the keypoint list.
      input_keypoints = np.append(keypoints, keypoints[-1] + lengths[-1])

      output_calibration_node = model_info.PWLCalibrationNode(
          input_node=model_output_node,
          input_keypoints=input_keypoints,
          output_keypoints=output_keypoints,
          default_input=None,
          default_output=None)
      nodes.append(output_calibration_node)
      model_output_node = output_calibration_node

  return model_info.ModelGraph(nodes=nodes, output_node=model_output_node)
