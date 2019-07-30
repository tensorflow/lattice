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
"""Piecewise linear calibration keypoints initialization functions.

Piecewise linear calibration requires initialization of its keypoints inputs
and outputs. If these initialization values are known one can use them directly.

But usually these initialization values are calculated in one of two ways:

1) As a preprocessing step one calculates the quantiles of some input features
(see function save_quantiles_for_keypoints below). Then during training
this quantile is sub-sampled to the number of keypoints, and these are the
initialization values used (see function load_keypoints_from_quantiles below).
Since the quantiles are independent of the number of keypoints, the quantiles
saved once can be used for training of models with different number of
keypoints, so the saved quantiles can be loaded multiple times during
hyperparameter optimization.

2) The user knows the input range and the number of keypoints. Use the function
uniform_keypoints_for_signal below to calculate evenly spread initialization
keypoints inputs based on that.

Notice that in both scenarios it is assumed that the user knows the output range
and the keypoints outputs are initialized linearly along the min and the max
of the output, so the calibration starts as a fully linear model.

Notice that the keypoints initialization values are saved, so they are no longer
needed in production (inference) time.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import os

# Dependency imports
import numpy as np
import six
import tensorflow as tf

from tensorflow_lattice.python.lib import tools
from tensorflow import gfile
from tensorflow.python.lib.io import file_io  # pylint: disable=g-direct-tensorflow-import

_QUANTILES_SUBDIRECTORY = "quantiles"

# The "feature" name for the label. The labels quantiles will be saved
# to a file whose name is based on this name. We assume that there's no
# regular feature with this name.
_LABEL_FEATURE_NAME = "__label__"


def _get_size(o):
  # Returns number of elements in o, for o list, dict or tuple
  if isinstance(o, dict):
    total = 0
    for v in o.values():
      total += _get_size(v)
    return total
  if isinstance(o, list) or isinstance(o, tuple) or isinstance(o, np.ndarray):
    return len(o)
  return 1


def _materialize_locally(tensors, num_steps=1, feed_dict=None, safety_size=1e9):
  """Materialize the given tensors locally, during initialization.

  Assumes non-distributed environment (uses SingularMonitoredSession).

  Args:
    tensors: tensors to be materialized: array or dict.
    num_steps: number of steps to run. Usually it's faster/easier to run in
      one step, a large batch. Set it to 0 or None to run until queue is
      exhausted, when a OutOfRangeError exception is raised -- typically when
      an input_fn is set to run for a fixed num_epochs.
    feed_dict: optional feed_dict.
    safety_size: if num_steps is None and one created input_fn to loop
      indefinitely (num_epochs=None), this could loop consuming memory. This
      is a safety limit on memory to prevent that. Increase this is you actually
      need more than these many elements in your results, or set num_steps.

  Returns:
    Materialized tensors as array or dict, like `tensors` arg.

  Raises:
    ValueError: for negative num_steps.
    tf.errors.OutOfRangeError: if can't read num_steps times.
  """
  if num_steps and num_steps < 0:
    raise ValueError("can not run with num_steps=%s" % num_steps)

  # tf.compat.v1.train.SingularMonitoredSession silently catches
  # tf.errors.OutOfRangeError, and we want to expose it.
  error = None
  with tf.compat.v1.train.SingularMonitoredSession() as sess:
    try:
      splits = []
      if not num_steps:
        # Run until queue exhausted.
        try:
          count = 0
          while True:
            r = sess.run(tensors, feed_dict=feed_dict)
            count += _get_size(r)
            if count > safety_size:
              raise ValueError(
                  "Unbound (num_steps=None) materialization of "
                  "input reached safety size of {}".format(safety_size))
            splits.append(r)
        except tf.errors.OutOfRangeError:
          pass
      else:
        # Run num_steps times.
        splits = [
            sess.run(tensors, feed_dict=feed_dict) for _ in range(num_steps)
        ]
      if isinstance(splits[0], dict):
        materialized = {}
        for k in splits[0].keys():
          materialized[k] = np.concatenate([
              splits[i][k] for i in range(len(splits))
              if splits[i][k].size > 0])
      else:
        materialized = np.concatenate(splits)
    except (tf.errors.OutOfRangeError, StopIteration) as ex:
      error = ex
  if error:
    raise error  # pylint: disable=raising-bad-type
  return materialized


def _path_for_quantile(subdir, feature_name):
  # Change slashes to dashes to make quantile filenames valid.
  # Note that there is a slight chance of name collision here.
  feature_name = str(feature_name).replace("/", "-")
  return os.path.join(subdir, "%s.txt" % feature_name)


def _save_quantiles(subdir, feature_name, quantiles):
  file_io.write_string_to_file(
      _path_for_quantile(subdir, str(feature_name)), str(quantiles))


def _load_quantiles(subdir, feature_name):
  """Returns False if failed to load."""
  serialized = file_io.read_file_to_string(
      _path_for_quantile(subdir, feature_name))
  return ast.literal_eval(serialized)


def uniform_keypoints_for_signal(num_keypoints,
                                 input_min,
                                 input_max,
                                 output_min,
                                 output_max,
                                 dtype=tf.float32):
  """Returns a pair of initialization tensors for calibration keypoints.

  This is used when the input range to be calibrated is known.

  Args:
    num_keypoints: number of keypoints to use for calibrating this signal.
    input_min: Scalar with the minimum value that the uncalibrated input can
      take.
    input_max: Scalar with the maximum value that the uncalibrated input can
      take.
    output_min: Scalar with calibrated value associated with input_min.
      Typically the minimum expected calibrated value, but not necessarily.
      Specially if the calibration is decreasing.
    output_max: Scalar with calibrated scalar value associated with input_max.
    dtype: If any of the scalars are not given as tensors, they are converted to
      tensors with this dtype.

  Returns:
    Two tensors to be used as the keypoints_inputs and keypoints_outputs
    initialization, uniformly distributed over given ranges. Dtype is given
    by input_min, input_max, output_min, output_max.

  Raises:
    ValueError: if underlying types (dtype) don't match.
  """
  input_min = tools.cast_to_scalar_tensor_of_dtype(input_min, dtype)
  input_max = tools.cast_to_scalar_tensor_of_dtype(input_max, dtype)
  output_min = tools.cast_to_scalar_tensor_of_dtype(output_min, dtype)
  output_max = tools.cast_to_scalar_tensor_of_dtype(output_max, dtype)
  types_set = set(
      [input_min.dtype, input_max.dtype, output_min.dtype, output_max.dtype])
  if len(types_set) != 1:
    raise ValueError("different dtypes for parameters: got %s" % types_set)
  return (tf.linspace(input_min, input_max, num_keypoints),
          tf.linspace(output_min, output_max, num_keypoints))


def save_quantiles_for_keypoints(input_fn,
                                 save_dir,
                                 feature_columns=None,
                                 num_steps=1,
                                 override=True,
                                 num_quantiles=1000,
                                 dtype=tf.float32):

  """Calculates and saves quantiles for given features and optionally the label.

  These values can later be retrieved and used by keypoints_from_quantiles()
  below.

  Repeated values are discarded before the quantiles are calculated. That means
  that the quantiles of a very skewed distribution (for instance where 99%
  of the values are 0), will be different. But for the purpose of calibration
  this approach is more useful.

  Nothing is returned, the values are simply saved in the given location.

  This function can be called as a preprocessing step before actual training
  starts. Typically one will run this in a separate process locally, before
  starting training for instance.

  Args:
    input_fn: Similar to input_fn provided to Estimators. Typically one
      doesn't need to go over the full data to get good quantiles. Typically
      some 100 random examples per quantile is good enough for the purpose of
      calibration. If you don't have too much data, just use everything.
      If input_fn returns a label, the label quantiles will be saved into a
      file named _LABEL_FEATURE_NAME in '<save_dir>/quantiles' directory and
      they can be used to initialize the keypoint outputs by passing True to
      the 'use_label_quantiles_for_outputs' in
      load_keypoints_from_quantiles().
    save_dir: Where to save these quantiles. Since when optimizing
      hyperparameters we train various models, we can share the quantiles
      information generated here. So this should be a directory that can be
      accessed by all training sessions. A subdirectory called "quantiles" will
      be created, and inside one file per feature is created: named after the
      feature name, and with the quantiles stored in JSON format.
    feature_columns: If set, quantiles are generated for these feature columns.
      The file name used to save the quantiles uses a hash of the names of the
      feature_columns, so it can support different quantiles sets for different
      parts of the model if needed. If not set quantiles will be generated for
      all features returned by input_fn.
    num_steps: number of steps to take over input_fn to gather enough data to
      create quantiles. Set to 0 or None to run until queue is exhausted,
      like if you used num_epochs in your input_fn.
    override: if False it won't regenerate quantiles for files that are already
      there. This works as long as the features definition/distribution hasn't
      change from one run to another.
    num_quantiles: This value should be larger than the maximum number of
      keypoints that will be considered for calibrating these features. If
      there are not enough quantiles for the keypoints, the system is robust and
      will simply interpolate the missing quantiles. Similarly if there are not
      enough examples to represent the quantiles, it will interpolate the
      quantiles from the examples given.
    dtype: Default dtype to use, in particular for categorical values.

  Returns: Nothing, results are saved to disk.

  Raises:
    tf.errors.OpError: For I/O errors.

  FutureWork:
    * Use Munro-Paterson algorithm to calculate quantiles in a streaming
      fashion. See Squawd library.
    * Add support to weighted examples.
    * Handle cases where there are not enough different values in quantiles.
  """
  subdir = os.path.join(save_dir, _QUANTILES_SUBDIRECTORY)
  file_io.recursive_create_dir(subdir)
  with tf.Graph().as_default():
    tensor_to_feature = _compute_tensor_to_feature_dict(
        input_fn, feature_columns, dtype)
    if override:
      tensor_to_saved_feature = tensor_to_feature
    else:
      tensor_to_saved_feature = {
          name: tensor
          for (name, tensor) in six.iteritems(tensor_to_feature)
          if not gfile.Exists(_path_for_quantile(subdir, name))}
    materialized_tensors = _materialize_locally(
        tensor_to_saved_feature, num_steps)

  percentiles = np.linspace(0., 100., num_quantiles)
  for key, values in six.iteritems(materialized_tensors):
    values = np.unique(values)
    quantiles = np.percentile(values, percentiles, interpolation="nearest")
    quantiles = list(quantiles)
    _save_quantiles(subdir, key, quantiles)


def _compute_tensor_to_feature_dict(input_fn, feature_columns, dtype):
  """Computes a feature_name-to-tensor dict for the given features.

  Args:
    input_fn: See the same argument in 'save_quantiles_for_keypoints'.
    feature_columns: See the same argument in 'save_quantiles_for_keypoints'.
    dtype: See the same argument in 'save_quantiles_for_keypoints'.

  Returns:
    A str->tensor dict mapping each feature name to the tensor containing its
    feature values for the current batch. The dict contains all the features
    returned by input_fn if feature_columns are none, or only those features
    included in 'feature_columns', otherwise. If a non-None label is returned by
    'input_fn', it will also be included in the dict.
  """
  if feature_columns is not None:
    transformed_columns_to_tensors, label = input_fn()
    features_to_tensors = {
        f_col.name: tools.input_from_feature_column(
            transformed_columns_to_tensors, f_col, dtype)
        for f_col in feature_columns
    }
  else:
    features_to_tensors, label = input_fn()
  if label is None:
    return features_to_tensors
  if _LABEL_FEATURE_NAME in features_to_tensors:
    raise ValueError(
        ("Can't save a label as there's already a feature named: '%s'."
         " Try renaming that feature. ") % _LABEL_FEATURE_NAME)
  features_to_tensors[_LABEL_FEATURE_NAME] = label
  return features_to_tensors


def save_quantiles_for_keypoints_once(
    input_fn, save_dir, is_chief, timeout_secs=600, **kwargs):
  """Concurrency-safe version of save_quantiles_for_keypoints.

  If is_chief is True and the quantiles do not already exist in 'save_dir',
  calls save_quantiles_for_keypoints; otherwise waits for up to timeout_secs
  seconds for the quantiles to be created and returns. Raises
  tools.SaveOrWaitTimeOutError if the timeout expires before the quantiles have
  been created.

  In multi-process tensorflow training, one must ensure that
  save_quantiles_for_keypoints is called by a single process before any process
  calls load_keypoints_from_quantiles. This function facilitates this, by making
  the chief worker save the quantiles and all the other processes wait for the
  quantiles to be created. Simply call this function in each process before
  the process calls load_keypoints_from_quantiles.

  Note that for a given 'save_dir', the quantiles will only be created on the
  first execution of the program. Successive executions will not overwrite the
  quantiles. To recreate the quantiles, the save_dir directory must be deleted.

  Args:
    input_fn: Passed to save_quantiles_for_keypoints.
    save_dir: Passed to save_quantiles_for_keypoints.
    is_chief: bool. Whether the caller is the chief.
    timeout_secs: int. The amount of time in seconds to wait for the chief.
    **kwargs: Other keyword arguments to be passed to
      save_quantiles_for_keypoints.
  """
  def write_fn():
    save_quantiles_for_keypoints(input_fn, save_dir, **kwargs)
  tools.save_once_or_wait_for_chief(
      write_fn=write_fn,
      metadata_dir=save_dir,
      is_chief=is_chief,
      timeout_secs=timeout_secs)


def load_keypoints_from_quantiles(feature_names,
                                  save_dir,
                                  num_keypoints,
                                  output_min=None,
                                  output_max=None,
                                  use_label_quantiles_for_outputs=False,
                                  reversed_dict=None,
                                  missing_input_values_dict=None,
                                  dtype=tf.float32):
  """Retrieves keypoints initialization values for selected features.

  It expects that the quantiles have already been calculated and saved in the
  save_dir by the save_quantiles_for_keypoints function. It will raise
  an I/O error if not.

  Args:
    feature_names: List of features names for which to get keypoints
      initialization values.
    save_dir: Directory where the quantiles have been saved to. Same value used
      when save_quantiles_for_keypoints was called.
    num_keypoints: Desired number of keypoints to use for calibration. This can
      either be a scalar to be used for all features, or a dict mapping feature
      name to num_keypoints. Fewer keypoints than requested can end up being
      used when for the given feature there are not enough different values. If
      num_keypoints for a feature is missing, None or 0, no initialization is
      generated.
    output_min: If not None, specifies the initial calibrated value associated
      with the first calibration keypoint. The keypoints outputs in between will
      be linearly interpolated.  It can be given as a scalar, in which case the
      value is used for all features, or a dict mapping feature name to
      output_min.
    output_max: Like output_min, but the calibrated value associated to the last
      keypoint. Scalar or dict.
    use_label_quantiles_for_outputs: Sets the keypoint outputs (calibrated
      values) to the label quantiles. If this parameter is true then output_min
      and output_max must both be None and the label quantiles must have been
      saved in the call to save_quantiles_for_keypoints that generated the
      quantile files (i.e. the input_fn parameter for the latter function must
      have returned a label). If this parameter is False, then neither
      output_min nor output_max may be None.
    reversed_dict: An optional dict. If reversed_dict[feature_name] is True,
      then the initial output keypoints will be in reversed order for that
      feature, i.e., input_min will be mapped to output_max or the last label
      quantile if use_label_quantiles_for_outputs is true, and input_max will be
      mapped to output_min or the first label quantile if
      use_label_quantiles_for_outputs is true. Reversing output keypoints is
      useful for decreasing monotonic calibrators.
    missing_input_values_dict: An optional dict. If provided, it should include
      all features passed via feature_names. If the value of
      missing_input_values[feature_name] is Not none, it is excluded from the
      input keypoint values.
    dtype: Type to be used for calibration.

  Returns:
    Dict of feature name to pair of constant tensors that can be used to
    initialize calibrators keypoints inputs and outputs.

  Raises:
    tf.errors.NotFoundError: if quantiles file not found.


    values in the signal. This would probably be better handled as categorical,
    but still this should handle the case correctly.
  """
  if (output_min is None) != (output_max is None):
    raise ValueError(
        "Either both output_min and output_max should be given or neither.")

  output_labels_given = (output_min is not None)
  if (use_label_quantiles_for_outputs and output_labels_given):
    raise ValueError(
        "If use_label_quantiles_for_outputs is true, then"
        " output_min and output_max cannot be given.")
  if (not use_label_quantiles_for_outputs and not output_labels_given):
    raise ValueError(
        "Either use_label_quantiles_for_outputs should be true or "
        " output_min and output_max must be given.")

  subdir = os.path.join(save_dir, _QUANTILES_SUBDIRECTORY)
  num_keypoints = tools.cast_to_dict(num_keypoints, feature_names,
                                     num_keypoints)
  if use_label_quantiles_for_outputs:
    label_quantiles = _load_quantiles(subdir, _LABEL_FEATURE_NAME)
  else:
    label_quantiles = None
    output_min = tools.cast_to_dict_of_tensor_scalars(output_min, feature_names,
                                                      dtype, "output_min")
    output_max = tools.cast_to_dict_of_tensor_scalars(output_max, feature_names,
                                                      dtype, "output_max")
  keypoints = {}
  for feature_name in feature_names:
    if feature_name not in num_keypoints or not num_keypoints[feature_name]:
      continue
    all_quantiles = _load_quantiles(subdir, feature_name)
    if (missing_input_values_dict is not None and
        feature_name in missing_input_values_dict):
      exclude_val = missing_input_values_dict[feature_name]
      if exclude_val is not None:
        all_quantiles = [q for q in all_quantiles if q != exclude_val]
    quantiles = _resample_quantiles(all_quantiles, num_keypoints[feature_name])
    unique_quantiles = sorted(set(quantiles))
    input_keypoints = tf.constant(
        unique_quantiles, shape=[len(unique_quantiles)], dtype=dtype)
    if use_label_quantiles_for_outputs:
      output_keypoints = tf.constant(
          _resample_quantiles(label_quantiles, len(unique_quantiles)),
          shape=[len(unique_quantiles)],
          dtype=dtype)
    else:
      output_keypoints = tf.linspace(output_min[feature_name],
                                     output_max[feature_name],
                                     len(unique_quantiles))
    if reversed_dict is not None and reversed_dict[feature_name]:
      output_keypoints = tf.reverse(output_keypoints, axis=[0])
    keypoints[feature_name] = (input_keypoints, output_keypoints)
  return keypoints


def _resample_quantiles(quantiles, new_size):
  """Computes new-size-quantiles on the given array of quantiles.

  This is roughly equivalent to computing new-size-quantiles on the
  original data from which 'quantiles' was created.

  Args:
    quantiles: list. The original quantiles.
    new_size: int. The number of quantiles to generate.
  Returns:
    A list of the new quantiles.
  """
  percentiles = np.linspace(0., 100., new_size)
  return np.percentile(quantiles, percentiles, interpolation="nearest")
