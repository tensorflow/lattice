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
"""Piecewise linear calibration layers library for TensorFlow Lattice.

Piecewise linear calibration is a 1D lookup table (LUT), part of TensorFlow
Lattice set of models, and typically used as calibration of input to lattice
models, but can be used in conjunction with other types of models as well.

It also works particularly well with linear models, not breaking independence
of the variables (desirable in some situations).

This modules provides functions used when building models, as opposed to the
basic operators exported by pwl_calibration_ops.py
"""
# Dependency imports

from tensorflow_lattice.python.lib import regularizers
from tensorflow_lattice.python.lib import tools
from tensorflow_lattice.python.ops import pwl_calibration_ops

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope


def one_dimensional_calibration_layer(uncalibrated_tensor,
                                      num_keypoints,
                                      signal_name,
                                      keypoints_initializers=None,
                                      keypoints_initializer_fns=None,
                                      bound=False,
                                      monotonic=None,
                                      missing_input_value=None,
                                      missing_output_value=None,
                                      **regularizer_amounts):
  """Creates a calibration layer for one single continuous signal.

  Returns a calibrated tensor of the uncalibrated continuous signal and a list
  of projections ops.

  Args:
    uncalibrated_tensor: Tensor of shape [batch_size] of one single signal.
    num_keypoints: Number of keypoints to use.
    signal_name: (Required) Used as a suffix to the variable names.
    keypoints_initializers: For evaluation or inference (or when resuming
      training from a checkpoint) the values will be loaded from disk, so they
      don't need to be given -- but in this case num_keypoints need to be
      accurate. Two tensors of shape [num_keypoints]. See
      load_keypoints_from_quantiles or uniform_keypoints_for_signal on how to
      generate these (module keypoints_initialization).
    keypoints_initializer_fns: Like keypoints_initializers but using lambda
      initializers. They should be compatible with tf.get_variable. If this is
      set, then keypoints_initializers must be None.
    bound: boolean whether output of calibration must be bound. Alternatively
      a dict mapping feature name to boundness.
    monotonic: whether calibration has to be kept monotonic: None or 0 means
      no monotonicity. Positive or negative values mean increasing or decreasing
      monotonicity respectively. Alternatively a dict mapping feature name
      to monotonic.
    missing_input_value: If set, and if the input has this value it is assumed
      to be missing and the output will either be calibrated to some value
      between `[calibration_output_min, calibration_output_max]` or set to a
      fixed value set by missing_output_value. Limitation: it only works for
      scalars.
    missing_output_value: Requires missing_input_value also to be set. If set
      if will convert missing input to this value.
    **regularizer_amounts: Keyword args of regularization amounts passed to
      regularizers.calibrator_regularization(). Keyword names should be among
      supported regularizers.CALIBRATOR_REGULARIZERS and values should be
      float.

  Returns:
    A tuple of:
    * calibrated tensor of shape [batchsize]
    * None or projection ops, that must be applied at each
      step (or every so many steps) to project the model to a feasible space:
      used for bounding the outputs or for imposing monotonicity.
    * None of a regularization loss, if regularization is configured.

  Raises:
    ValueError: if dtypes are incompatible.
    ValueError: if keypoints_initializers and keypoints_initializer_fns are both
      set.




  """
  if (keypoints_initializers is not None and
      keypoints_initializer_fns is not None):
    raise ValueError('keypoints_initializers and keypoints_initializer_fns '
                     'cannot both be set.')
  with variable_scope.variable_scope('pwl_calibration'):
    # Sanity checks.
    if uncalibrated_tensor.get_shape().ndims != 1:
      raise ValueError(
          'one_dimensional_calibration_layer can only be used for a single '
          'signal, so uncalibrated shape must be of form (batchsize), got %s' %
          uncalibrated_tensor.get_shape())
    if missing_output_value is not None and missing_input_value is None:
      raise ValueError(
          'missing_output_value can only be set if a misisng_input_value is '
          'also set, missing_input_value=None, missing_output_values=%s' %
          missing_output_value)

    # Create variables: only uses initializer if they are given.
    kp_in_name = signal_name + '_keypoints_inputs'
    kp_out_name = signal_name + '_keypoints_outputs'
    missing_out_calibrated_name = signal_name + '_calibrated_missing_output'

    if keypoints_initializers is not None:
      kp_in, kp_out = keypoints_initializers[0], keypoints_initializers[1]
      if (uncalibrated_tensor.dtype != kp_in.dtype or
          uncalibrated_tensor.dtype != kp_out.dtype):
        raise ValueError(
            'incompatible types for signal \'%s\': uncalibrated=%s, '
            'keypoints_initializers[input=%s, output=%s]' %
            (signal_name, uncalibrated_tensor.dtype, kp_in.dtype, kp_out.dtype))
      tools.assert_shape(kp_in, [num_keypoints],
                         'keypoints_initializers[input]')
      tools.assert_shape(kp_out, [num_keypoints],
                         'keypoints_initializers[output]')
      keypoints_inputs = variable_scope.get_variable(
          kp_in_name, initializer=kp_in)
      keypoints_outputs = variable_scope.get_variable(
          kp_out_name, initializer=kp_out)

      if missing_input_value is not None:
        # Value to be taken by missing features.
        if missing_output_value is not None:
          missing_out_calibrated = constant_op.constant(
              missing_output_value, dtype=uncalibrated_tensor.dtype)
        else:
          # Learned missing value, initialized by the first value of kp_out.
          missing_out_calibrated = variable_scope.get_variable(
              missing_out_calibrated_name, initializer=kp_out[0])
    elif keypoints_initializer_fns is not None:
      kp_in, kp_out = keypoints_initializer_fns[0], keypoints_initializer_fns[1]
      keypoints_inputs = variable_scope.get_variable(
          kp_in_name, shape=[num_keypoints], initializer=kp_in)
      keypoints_outputs = variable_scope.get_variable(
          kp_out_name, shape=[num_keypoints], initializer=kp_out)

      if missing_input_value is not None:
        # Value to be taken by missing features.
        if missing_output_value is not None:
          missing_out_calibrated = constant_op.constant(
              missing_output_value, dtype=uncalibrated_tensor.dtype)
        else:
          # Learned missing value, initialized by the first value of kp_out.
          def first_kp_out(*args, **kwargs):
            return kp_out(*args, **kwargs)[0]

          missing_out_calibrated = variable_scope.get_variable(
              missing_out_calibrated_name, shape=[], initializer=first_kp_out)
    else:
      # When loading a model, no initializer.
      keypoints_inputs = variable_scope.get_variable(
          kp_in_name, shape=[num_keypoints], dtype=uncalibrated_tensor.dtype)
      keypoints_outputs = variable_scope.get_variable(
          kp_out_name, shape=[num_keypoints], dtype=uncalibrated_tensor.dtype)
      if missing_input_value:
        if missing_output_value:
          missing_out_calibrated = constant_op.constant(
              missing_output_value, dtype=uncalibrated_tensor.dtype)
        else:
          missing_out_calibrated = variable_scope.get_variable(
              missing_out_calibrated_name,
              shape=[],
              dtype=uncalibrated_tensor.dtype)

    # Split missing values from normal values.
    # FutureWork: move handling of missing values be moved to C++ land.
    if missing_input_value is not None:
      missing_mask = math_ops.equal(uncalibrated_tensor,
                                    constant_op.constant(missing_input_value))
      mask_indices = math_ops.range(array_ops.shape(uncalibrated_tensor)[0])
      mask_indices = data_flow_ops.dynamic_partition(
          mask_indices, math_ops.cast(missing_mask, dtypes.int32), 2)
      (uncalibrated_tensor, missing_values) = data_flow_ops.dynamic_partition(
          uncalibrated_tensor, math_ops.cast(missing_mask, dtypes.int32), 2)

      # Assign value to missing_values.
      missing_values = array_ops.ones_like(missing_values)
      missing_values *= missing_out_calibrated

    # Dense implementation.
    interpolation = pwl_calibration_ops.pwl_indexing_calibrator(
        uncalibrated_tensor, keypoints_inputs)
    calibrated = math_ops.reduce_sum(interpolation * keypoints_outputs, 1)
    projection_ops = None

    # Re-join missing values.
    if missing_input_value is not None:
      calibrated = data_flow_ops.dynamic_stitch(mask_indices,
                                                [calibrated, missing_values])

    # Boundness.
    projected_keypoints_outputs = None
    if bound:
      bound_min_name = signal_name + '_bound_min'
      bound_max_name = signal_name + '_bound_max'
      # Set bound_min/max from min/max values initialized.
      if keypoints_initializers is not None:
        # Store bound_min and bound_max in variables because their values (from
        # kp_out) are only available during train (when keypoints_initializers
        # is available). During inference the value is not available. Storing
        # them in variables make them available during inference.
        bound_min = variable_scope.get_variable(
            bound_min_name,
            dtype=uncalibrated_tensor.dtype,
            initializer=math_ops.reduce_min(kp_out))
        bound_max = variable_scope.get_variable(
            bound_max_name,
            dtype=uncalibrated_tensor.dtype,
            initializer=math_ops.reduce_max(kp_out))
      elif keypoints_initializer_fns is not None:
        # Store bound_min and bound_max in variables because their values (from
        # kp_out) are only available during train (when keypoints_initializers
        # is available). During inference the value is not available. Storing
        # them in variables make them available during inference.
        def min_kp_out(*args, **kwargs):
          return math_ops.reduce_min(kp_out(*args, **kwargs))

        def max_kp_out(*args, **kwargs):
          return math_ops.reduce_max(kp_out(*args, **kwargs))

        bound_min = variable_scope.get_variable(
            bound_min_name,
            dtype=uncalibrated_tensor.dtype,
            shape=[],
            initializer=min_kp_out)
        bound_max = variable_scope.get_variable(
            bound_max_name,
            dtype=uncalibrated_tensor.dtype,
            shape=[],
            initializer=max_kp_out)
      else:
        # No need to initialize, since presumably their values will be read
        # from some checkpoint.
        bound_min = variable_scope.get_variable(
            bound_min_name, dtype=uncalibrated_tensor.dtype, shape=[])
        bound_max = variable_scope.get_variable(
            bound_max_name, dtype=uncalibrated_tensor.dtype, shape=[])
      projected_keypoints_outputs = math_ops.minimum(
          math_ops.maximum(keypoints_outputs, bound_min), bound_max)

    # Monotonicity.
    if monotonic:
      # First a soft-enforcement: might not break indirect constraints.
      if projected_keypoints_outputs is None:
        projected_keypoints_outputs = keypoints_outputs
      projected_keypoints_outputs = pwl_calibration_ops.monotonic_projection(
          increasing=bool(monotonic > 0),
          values=projected_keypoints_outputs,
          name='project_calibration_to_monotonic')

    # Make assing_add op to projected output.
    if projected_keypoints_outputs is not None:
      constrained_diff = projected_keypoints_outputs - keypoints_outputs
      projection_ops = state_ops.assign_add(
          keypoints_outputs,
          constrained_diff,
          use_locking=None,
          name='project_feasible')
      if (bound and missing_input_value is not None and
          missing_output_value is None):
        # Include op bounding calibrated missing value.
        projected_missing_out_calibrated = math_ops.minimum(
            math_ops.maximum(missing_out_calibrated, bound_min), bound_max)
        projected_missing_out_calibrated_diff = (
            projected_missing_out_calibrated - missing_out_calibrated)
        projected_missing_out_calibrated_op = state_ops.assign_add(
            missing_out_calibrated,
            projected_missing_out_calibrated_diff,
            use_locking=None,
            name='project_missing_calibration_to_bounds')
        projection_ops = control_flow_ops.group(
            projection_ops, projected_missing_out_calibrated_op)

    # Regularization
    regularization = regularizers.calibrator_regularization(
        keypoints_outputs,
        name=signal_name + '_calibrator_regularization',
        **regularizer_amounts)
  return calibrated, projection_ops, regularization


def input_calibration_layer(columns_to_tensors,
                            num_keypoints,
                            feature_columns=None,
                            keypoints_initializers=None,
                            keypoints_initializer_fns=None,
                            bound=False,
                            monotonic=None,
                            missing_input_values=None,
                            missing_output_values=None,
                            dtype=dtypes.float32,
                            **regularizer_amounts):
  """Creates a calibration layer for the given input and feature_columns.

  Returns a tensor with the calibrated values of the given features, a list
  of the names of the features in the order they feature in the returned, and
  a list of projection ops, that must be applied at each step (or every so many
  steps) to project the model to a feasible space: used for bounding the outputs
  or for imposing monotonic -- the list will be empty if bound and
  monotonic are not set.

  Args:
    columns_to_tensors: A mapping from feature name to tensors. 'string' key
      means a base feature (not-transformed). If feature_columns is not set
      these are the features calibrated. Otherwise the transformed
      feature_columns are the ones calibrated.
    num_keypoints: Number of keypoints to use. Either a single int, or a dict
      mapping feature names to num_keypoints. If a value of the dict is 0 or
      None the correspondent feature won't be calibrated.
    feature_columns: Optional. If set to a set of FeatureColumns, these will
      be the features used and calibrated.
    keypoints_initializers: For evaluation or inference (or when resuming
      training from a checkpoint) the values will be loaded from disk, so they
      don't need to be given (leave it as None).
      Either a tuple of two tensors of shape [num_keypoints], or a dict mapping
      feature names to pair of tensors of shape [num_keypoints[feature_name]].
      See load_keypoints_from_quantiles or uniform_keypoints_for_signal on how
      to generate these (module keypoints_initialization).
    keypoints_initializer_fns: Like keypoints_initializers but using lambda
      initializers. They should be compatible with tf.get_variable. If this is
      set, then keypoints_initializers must be None.
    bound: boolean whether output of calibration must be bound. Alternatively
      a dict mapping feature name to boundness.
    monotonic: whether calibration has to be kept monotonic: None or 0 means
      no monotonic. Positive or negative values mean increasing or decreasing
      monotonic respectively. Alternatively a dict mapping feature name
      to monotonic.
    missing_input_values: If set, and if the input has this value it is assumed
      to be missing and the output will either be calibrated to some value
      between `[calibration_output_min, calibration_output_max]` or set to a
      fixed value set by missing_output_value. Limitation: it only works for
      scalars. Either one value for all inputs, or a dict mapping feature name
      to missing_input_value for the respective feature.
    missing_output_values: Requires missing_input_value also to be set. If set
      if will convert missing input to this value. Either one value for all
      inputs, or a dict mapping feature name to missing_input_value for the
      respective feature.
    dtype: If any of the scalars are not given as tensors, they are converted
      to tensors with this dtype.
    **regularizer_amounts: Keyword args of regularization amounts passed to
      regularizers.calibrator_regularization(). Keyword names should be among
      supported regularizers.CALIBRATOR_REGULARIZERS and values should be
      either float or {feature_name: float}. If float, then same value is
      applied to all features.

  Returns:
    A tuple of:
    * calibrated tensor of shape [batch_size, sum(features dimensions)].
    * list of the feature names in the order they feature in the calibrated
      tensor. A name may appear more than once if the feature is
      multi-dimension (for instance a multi-dimension embedding)
    * list of projection ops, that must be applied at each step (or every so
      many steps) to project the model to a feasible space: used for bounding
      the outputs or for imposing monotonicity. Empty if none are requested.
    * None or tensor with regularization loss.

  Raises:
    ValueError: if dtypes are incompatible.


  """
  with ops.name_scope('input_calibration_layer'):
    feature_names = tools.get_sorted_feature_names(columns_to_tensors,
                                                   feature_columns)
    num_keypoints = tools.cast_to_dict(num_keypoints, feature_names,
                                       'num_keypoints')
    bound = tools.cast_to_dict(bound, feature_names, 'bound')
    monotonic = tools.cast_to_dict(monotonic, feature_names, 'monotonic')
    keypoints_initializers = tools.cast_to_dict(
        keypoints_initializers, feature_names, 'keypoints_initializers')
    keypoints_initializer_fns = tools.cast_to_dict(
        keypoints_initializer_fns, feature_names, 'keypoints_initializer_fns')
    missing_input_values = tools.cast_to_dict(
        missing_input_values, feature_names, 'missing_input_values')
    missing_output_values = tools.cast_to_dict(
        missing_output_values, feature_names, 'missing_output_values')
    regularizer_amounts = {
        regularizer_name: tools.cast_to_dict(
            regularizer_amounts[regularizer_name], feature_names,
            regularizer_name) for regularizer_name in regularizer_amounts
    }

    per_dimension_feature_names = []

    # Get uncalibrated tensors, either from columns_to_tensors, or using
    # feature_columns.
    if feature_columns is None:
      uncalibrated_features = [
          columns_to_tensors[name] for name in feature_names
      ]
    else:
      transformed_columns_to_tensors = columns_to_tensors.copy()
      dict_feature_columns = {f_col.name: f_col for f_col in feature_columns}
      uncalibrated_features = [
          tools.input_from_feature_column(transformed_columns_to_tensors,
                                          dict_feature_columns[name], dtype)
          for name in feature_names
      ]

    projection_ops = []
    calibrated_splits = []
    total_regularization = None
    for feature_idx in range(len(feature_names)):
      name = feature_names[feature_idx]
      uncalibrated_feature = uncalibrated_features[feature_idx]
      if uncalibrated_feature.shape.ndims == 1:
        feature_dim = 1
        uncalibrated_splits = [uncalibrated_feature]
      elif uncalibrated_feature.shape.ndims == 2:
        feature_dim = uncalibrated_feature.shape.dims[1].value
        uncalibrated_splits = array_ops.unstack(uncalibrated_feature, axis=1)
      else:
        raise ValueError(
            'feature {}: it has rank {}, but only ranks 1 or 2 are '
            'supported; feature shape={}'.format(
                name, uncalibrated_feature.shape.ndims,
                uncalibrated_feature.shape))
      missing_input_value = missing_input_values[name]
      missing_output_value = missing_output_values[name]
      feature_regularizer_amounts = {
          regularizer_name: regularizer_amounts[regularizer_name][name]
          for regularizer_name in regularizer_amounts
      }

      # FutureWork: make the interpolation ops handle multi-dimension values,
      #   so this step is not needed.
      for dim_idx in range(feature_dim):
        per_dimension_feature_names += [name]
        split_name = name
        if feature_dim > 1:
          split_name = '{}_dim_{}'.format(name, dim_idx)
        uncalibrated = uncalibrated_splits[dim_idx]
        if not num_keypoints[name]:
          # No calibration for this feature:
          calibrated_splits += [uncalibrated]
          if (missing_input_value is not None or
              missing_output_value is not None):
            raise ValueError(
                'feature %s: cannot handle missing values if feature is not '
                'calibrated, missing_input_value=%s, missing_output_value=%s' %
                (name, missing_input_value, missing_output_value))
        else:
          calibrated, projection, reg = one_dimensional_calibration_layer(
              uncalibrated,
              num_keypoints[name],
              signal_name=split_name,
              keypoints_initializers=keypoints_initializers[name],
              keypoints_initializer_fns=keypoints_initializer_fns[name],
              bound=bound[name],
              monotonic=monotonic[name],
              missing_input_value=missing_input_value,
              missing_output_value=missing_output_value,
              **feature_regularizer_amounts)
          calibrated_splits += [calibrated]
          if projection is not None:
            projection_ops += [projection]
          total_regularization = tools.add_if_not_none(total_regularization,
                                                       reg)

    all_calibrated = array_ops.stack(
        calibrated_splits, axis=1, name='stack_calibrated')
    return (all_calibrated, per_dimension_feature_names, projection_ops,
            total_regularization)


def calibration_layer(uncalibrated_tensor,
                      num_keypoints,
                      keypoints_initializers=None,
                      keypoints_initializer_fns=None,
                      bound=False,
                      monotonic=None,
                      missing_input_values=None,
                      missing_output_values=None,
                      name=None,
                      **regularizer_amounts):
  """Creates a calibration layer for uncalibrated values.

  Returns a calibrated tensor of the same shape as the uncalibrated continuous
  signals passed in, and a list of projection ops, that must be applied at
  each step (or every so many steps) to project the model to a feasible space:
  used for bounding the outputs or for imposing monotonicity -- the list will be
  empty if bound and monotonic are not set.

  Args:
    uncalibrated_tensor: Tensor of shape [batch_size, ...] with uncalibrated
      values.
    num_keypoints: Number of keypoints to use. Either a scalar value that
      will be used for every uncalibrated signal, or a list of n values,
      per uncalibrated signal -- uncalibrated is first flattened (
      see tf.contrib.layers.flatten) to [batch_size, n], and there should
      be one value in the list per n. If a value of the list is 0 or None
      the correspondent signal won't be calibrated.
    keypoints_initializers: For evaluation or inference (or when resuming
      training from a checkpoint) the values will be loaded from disk, so they
      don't need to be given (leave it as None).
      Otherwise provide either a tuple of two tensors of shape [num_keypoints],
      or a list of n pairs of tensors, each of shape [num_keypoints]. In this
      list there should be one pair per uncalibrated signal, just like
      num_keypoints above. Notice that num_keypoints can be different per
      signal.
    keypoints_initializer_fns: Like keypoints_initializers but using lambda
      initializers. They should be compatible with tf.get_variable. If this is
      set, then keypoints_initializers must be None.
    bound: boolean whether output of calibration must be bound. Alternatively
      a list of n booleans, one per uncalibrated value, like num_keypoints
      above.
    monotonic: whether calibration is monotonic: None or 0 means no
      monotonicity. Positive or negative values mean increasing or decreasing
      monotonicity respectively. Alternatively a list of n monotonic values,
      one per uncalibrated value, like num_keypoints above.
    missing_input_values: If set, and if the input has this value it is assumed
      to be missing and the output will either be calibrated to some value
      between `[calibration_output_min, calibration_output_max]` or set to a
      fixed value set by missing_output_value. Limitation: it only works for
      scalars. Either one value for all inputs, or a list with one value per
      uncalibrated value.
    missing_output_values: Requires missing_input_value also to be set. If set
      if will convert missing input to this value. Either one value for all
      outputs, or a list with one value per uncalibrated value.
    name: Name scope for operations.
    **regularizer_amounts: Keyword args of regularization amounts passed to
      regularizers.calibrator_regularization(). Keyword names should be among
      supported regularizers.CALIBRATOR_REGULARIZERS and values should be
      either float or list of floats. If float, then same value is applied to
      all input signals.

  Returns:
    A tuple of:
    * calibrated tensor of shape [batch_size, ...], the same shape as
      uncalibrated.
    * list of projection ops, that must be applied at each step (or every so
      many steps) to project the model to a feasible space: used for bounding
      the outputs or for imposing monotonicity. Empty if none are requested.
    * None or tensor with regularization loss.

  Raises:
    ValueError: If dimensions don't match.
  """
  with ops.name_scope(name or 'calibration_layer'):
    # Flattening uncalibrated tensor [batch_Size, k1, k2, ..., kn] to
    # [batch_size, k1 * k2 * ... * kn].
    uncalibrated_shape = uncalibrated_tensor.get_shape().as_list()
    n = 1
    for non_batch_dim in uncalibrated_shape[1:]:
      n *= non_batch_dim
    flat_uncalibrated = array_ops.reshape(
        uncalibrated_tensor, shape=[-1, n], name='flat_uncalibrated')

    num_keypoints = tools.cast_to_list(num_keypoints, n, 'num_keypoints')
    keypoints_initializers = tools.cast_to_list(keypoints_initializers, n,
                                                'keypoints_initializers')
    keypoints_initializer_fns = tools.cast_to_list(keypoints_initializer_fns, n,
                                                   'keypoints_initializer_fns')
    bound = tools.cast_to_list(bound, n, 'bound')
    monotonic = tools.cast_to_list(monotonic, n, 'monotonic')
    missing_input_values = tools.cast_to_list(missing_input_values, n,
                                              'missing_input_values')
    missing_output_values = tools.cast_to_list(missing_output_values, n,
                                               'missing_output_values')
    regularizer_amounts = {
        regularizer_name: tools.cast_to_list(
            regularizer_amounts[regularizer_name], n, regularizer_name)
        for regularizer_name in regularizer_amounts
    }

    signal_names = ['signal_%d' % ii for ii in range(n)]

    uncalibrated_splits = array_ops.unstack(flat_uncalibrated, axis=1)
    calibrated_splits = []
    projection_ops = []
    total_regularization = None
    for ii in range(n):
      if not num_keypoints[ii]:
        # No calibration for this signal.
        calibrated_splits += [uncalibrated_splits[ii]]
      else:
        signal_regularizer_amounts = {
            regularizer_name: regularizer_amounts[regularizer_name][ii]
            for regularizer_name in regularizer_amounts
        }
        calibrated, projection, reg = one_dimensional_calibration_layer(
            uncalibrated_splits[ii],
            num_keypoints[ii],
            signal_name=signal_names[ii],
            keypoints_initializers=keypoints_initializers[ii],
            keypoints_initializer_fns=keypoints_initializer_fns[ii],
            bound=bound[ii],
            monotonic=monotonic[ii],
            missing_input_value=missing_input_values[ii],
            missing_output_value=missing_output_values[ii],
            **signal_regularizer_amounts)
        calibrated_splits += [calibrated]
        if projection is not None:
          projection_ops += [projection]
        total_regularization = tools.add_if_not_none(total_regularization, reg)
    flat_calibrated = array_ops.stack(
        calibrated_splits, axis=1, name='stack_calibrated')
    reshaped_calibrated = array_ops.reshape(
        flat_calibrated,
        shape=array_ops.shape(uncalibrated_tensor),
        name='reshape_calibrated')
    return reshaped_calibrated, projection_ops, total_regularization
