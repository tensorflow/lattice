# Copyright 2020 Google LLC
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
"""Piecewise linear calibration layer.

Sonnet (v2) implementation of tensorflow lattice pwl calibration module. Module
takes single or multi-dimensional input and transforms it using piecewise linear
functions following monotonicity, convexity/concavity and bounds constraints if
specified.
"""

# TODO: Add built-in regularizers like laplacian, hessian, etc.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import pwl_calibration_lib
from . import utils

from absl import logging
import numpy as np
import sonnet as snt
import tensorflow as tf

INTERPOLATION_KEYPOINTS_NAME = "interpolation_keypoints"
LENGTHS_NAME = "lengths"
MISSING_INPUT_VALUE_NAME = "missing_input_value"
PWL_CALIBRATION_KERNEL_NAME = "pwl_calibration_kernel"
PWL_CALIBRATION_MISSING_OUTPUT_NAME = "pwl_calibration_missing_output"


class PWLCalibration(snt.Module):
  # pyformat: disable
  """Piecewise linear calibration layer.

  Module takes input of shape `(batch_size, units)` or `(batch_size, 1)` and
  transforms it using `units` number of piecewise linear functions following
  monotonicity, convexity and bounds constraints if specified. If multi
  dimensional input is provides, each output will be for the corresponding
  input, otherwise all PWL functions will act on the same input. All units share
  the same configuration, but each has their separate set of trained
  parameters.

  Input shape:
  Single input should be a rank-2 tensor with shape: `(batch_size, units)` or
  `(batch_size, 1)`. The input can also be a list of two tensors of the same
  shape where the first tensor is the regular input tensor and the second is the
  `is_missing` tensor. In the `is_missing` tensor, 1.0 represents missing input
  and 0.0 represents available input.

  Output shape:
  Rank-2 tensor with shape: `(batch_size, units)`.

  Attributes:
    - All `__init__` arguments.
    kernel: TF variable which stores weights of piecewise linear function.
    missing_output: TF variable which stores output learned for missing input.
      Or TF Constant which stores `missing_output_value` if one is provided.
      Available only if `impute_missing` is True.

  Example:

  ```python
  calibrator = tfl.sonnet_modules.PWLCalibration(
      # Key-points of piecewise-linear function.
      input_keypoints=np.linspace(1., 4., num=4),
      # Output can be bounded, e.g. when this layer feeds into a lattice.
      output_min=0.0,
      output_max=2.0,
      # You can specify monotonicity and other shape constraints for the layer.
      monotonicity='increasing',
  )
  ```
  """
  # pyformat: enable

  def __init__(self,
               input_keypoints,
               units=1,
               output_min=None,
               output_max=None,
               clamp_min=False,
               clamp_max=False,
               monotonicity="none",
               convexity="none",
               is_cyclic=False,
               kernel_init="equal_heights",
               impute_missing=False,
               missing_input_value=None,
               missing_output_value=None,
               num_projection_iterations=8,
               **kwargs):
    # pyformat: disable
    """Initializes an instance of `PWLCalibration`.

    Args:
      input_keypoints: Ordered list of keypoints of piecewise linear function.
        Can be anything accepted by tf.convert_to_tensor().
      units: Output dimension of the layer. See class comments for details.
      output_min: Minimum output of calibrator.
      output_max: Maximum output of calibrator.
      clamp_min: For monotonic calibrators ensures that output_min is reached.
      clamp_max: For monotonic calibrators ensures that output_max is reached.
      monotonicity: Constraints piecewise linear function to be monotonic using
        'increasing' or 1 to indicate increasing monotonicity, 'decreasing' or
        -1 to indicate decreasing monotonicity and 'none' or 0 to indicate no
        monotonicity constraints.
      convexity: Constraints piecewise linear function to be convex or concave.
        Convexity is indicated by 'convex' or 1, concavity is indicated by
        'concave' or -1, 'none' or 0 indicates no convexity/concavity
        constraints.
        Concavity together with increasing monotonicity as well as convexity
        together with decreasing monotonicity results in diminishing return
        constraints.
        Consider increasing the value of `num_projection_iterations` if
        convexity is specified, especially with larger number of keypoints.
      is_cyclic: Whether the output for last keypoint should be identical to
        output for first keypoint. This is useful for features such as
        "time of day" or "degree of turn". If inputs are discrete and exactly
        match keypoints then is_cyclic will have an effect only if TFL
        regularizers are being used.
      kernel_init: None or one of:
        - String `"equal_heights"`: For pieces of pwl function to have equal
          heights.
        - String `"equal_slopes"`: For pieces of pwl function to have equal
          slopes.
        - Any Sonnet initializer object. If you are passing such object make
          sure that you know how this module uses the variables.
      impute_missing: Whether to learn an output for cases where input data is
        missing. If set to True, either `missing_input_value` should be
        initialized, or the `call()` method should get pair of tensors. See
        class input shape description for more details.
      missing_input_value: If set, all inputs which are equal to this value will
        be considered as missing. Can not be set if `impute_missing` is False.
      missing_output_value: If set, instead of learning output for missing
        inputs, simply maps them into this value. Can not be set if
        `impute_missing` is False.
      num_projection_iterations: Number of iterations of the Dykstra's
        projection algorithm. Constraints are strictly satisfied at the end of
        each update, but the update will be closer to a true L2 projection with
        higher number of iterations. See
        `tfl.pwl_calibration_lib.project_all_constraints` for more details.
      **kwargs: Other args passed to `snt.Module` initializer.

    Raises:
      ValueError: If layer hyperparameters are invalid.
    """
    # pyformat: enable
    super(PWLCalibration, self).__init__(**kwargs)

    pwl_calibration_lib.verify_hyperparameters(
        input_keypoints=input_keypoints,
        output_min=output_min,
        output_max=output_max,
        monotonicity=monotonicity,
        convexity=convexity,
        is_cyclic=is_cyclic)
    if missing_input_value is not None and not impute_missing:
      raise ValueError("'missing_input_value' is specified, but "
                       "'impute_missing' is set to False. "
                       "'missing_input_value': " + str(missing_input_value))
    if missing_output_value is not None and not impute_missing:
      raise ValueError("'missing_output_value' is specified, but "
                       "'impute_missing' is set to False. "
                       "'missing_output_value': " + str(missing_output_value))
    if input_keypoints is None:
      raise ValueError("'input_keypoints' can't be None")
    if monotonicity is None:
      raise ValueError("'monotonicity' can't be None. Did you mean '0'?")

    self.input_keypoints = input_keypoints
    self.units = units
    self.output_min = output_min
    self.output_max = output_max
    self.clamp_min = clamp_min
    self.clamp_max = clamp_max
    (self._output_init_min, self._output_init_max, self._output_min_constraints,
     self._output_max_constraints
    ) = pwl_calibration_lib.convert_all_constraints(self.output_min,
                                                    self.output_max,
                                                    self.clamp_min,
                                                    self.clamp_max)

    self.monotonicity = monotonicity
    self.convexity = convexity
    self.is_cyclic = is_cyclic

    if kernel_init == "equal_heights":
      self.kernel_init = _UniformOutputInitializer(
          output_min=self._output_init_min,
          output_max=self._output_init_max,
          monotonicity=self.monotonicity)
    elif kernel_init == "equal_slopes":
      self.kernel_init = _UniformOutputInitializer(
          output_min=self._output_init_min,
          output_max=self._output_init_max,
          monotonicity=self.monotonicity,
          keypoints=self.input_keypoints)

    self.impute_missing = impute_missing
    self.missing_input_value = missing_input_value
    self.missing_output_value = missing_output_value
    self.num_projection_iterations = num_projection_iterations

  @snt.once
  def _create_parameters_once(self, inputs):
    """Creates the variables that will be reused on each call of the module."""
    self.dtype = tf.convert_to_tensor(self.input_keypoints).dtype
    input_keypoints = np.array(self.input_keypoints)
    # Don't need last keypoint for interpolation because we need only beginnings
    # of intervals.
    self._interpolation_keypoints = tf.constant(
        input_keypoints[:-1],
        dtype=self.dtype,
        name=INTERPOLATION_KEYPOINTS_NAME)
    self._lengths = tf.constant(
        input_keypoints[1:] - input_keypoints[:-1],
        dtype=self.dtype,
        name=LENGTHS_NAME)

    constraints = _PWLCalibrationConstraints(
        monotonicity=self.monotonicity,
        convexity=self.convexity,
        lengths=self._lengths,
        output_min=self.output_min,
        output_max=self.output_max,
        output_min_constraints=self._output_min_constraints,
        output_max_constraints=self._output_max_constraints,
        num_projection_iterations=self.num_projection_iterations)

    # If 'is_cyclic' is specified - last weight will be computed from previous
    # weights in order to connect last keypoint with first.
    num_weights = input_keypoints.size - self.is_cyclic

    # PWL calibration layer kernel is units-column matrix. First row of matrix
    # represents bias. All remaining represent delta in y-value compare to
    # previous point. Aka heights of segments.
    self.kernel = tf.Variable(
        initial_value=self.kernel_init([num_weights, self.units],
                                       dtype=self.dtype),
        name=PWL_CALIBRATION_KERNEL_NAME,
        constraint=constraints)

    if self.impute_missing:
      if self.missing_input_value is not None:
        self._missing_input_value_tensor = tf.constant(
            self.missing_input_value,
            dtype=self.dtype,
            name=MISSING_INPUT_VALUE_NAME)
      else:
        self._missing_input_value_tensor = None

      if self.missing_output_value is not None:
        self.missing_output = tf.constant(
            self.missing_output_value, shape=[1, self.units], dtype=self.dtype)
      else:
        missing_init = (self._output_init_min + self._output_init_max) / 2.0
        missing_constraints = _NaiveBoundsConstraints(
            lower_bound=self.output_min, upper_bound=self.output_max)
        initializer = snt.initializers.Constant(missing_init)
        self.missing_output = tf.Variable(
            initial_value=initializer([1, self.units], self.dtype),
            name=PWL_CALIBRATION_MISSING_OUTPUT_NAME,
            constraint=missing_constraints)

  def __call__(self, inputs):
    """Standard Sonnet __call__() method..

    Args:
      inputs: Either input tensor or list of 2 elements: input tensor and
        `is_missing` tensor.

    Returns:
      Calibrated input tensor.

    Raises:
      ValueError: If `is_missing` tensor specified incorrectly.
    """
    self._create_parameters_once(inputs)
    is_missing = None
    if isinstance(inputs, list):
      # Only 2 element lists are allowed. When such list is given - second
      # element represents 'is_missing' tensor encoded as float value.
      if not self.impute_missing:
        raise ValueError("Multiple inputs for PWLCalibration module assume "
                         "regular input tensor and 'is_missing' tensor, but "
                         "this instance of a layer is not configured to handle "
                         "missing value. See 'impute_missing' parameter.")
      if len(inputs) > 2:
        raise ValueError("Multiple inputs for PWLCalibration module assume "
                         "normal input tensor and 'is_missing' tensor, but more"
                         " than 2 tensors given. 'inputs': " + str(inputs))
      if len(inputs) == 2:
        inputs, is_missing = inputs
        if is_missing.shape.as_list() != inputs.shape.as_list():
          raise ValueError(
              "is_missing shape %s does not match inputs shape %s for "
              "PWLCalibration module" %
              (str(is_missing.shape), str(inputs.shape)))
      else:
        [inputs] = inputs
    if len(inputs.shape) != 2 or (inputs.shape[1] != self.units and
                                  inputs.shape[1] != 1):
      raise ValueError("Shape of input tensor for PWLCalibration module must"
                       " be [-1, units] or [-1, 1]. It is: " +
                       str(inputs.shape))

    if self._interpolation_keypoints.dtype != inputs.dtype:
      raise ValueError("dtype(%s) of input to PWLCalibration module does not "
                       "correspond to dtype(%s) of keypoints. You can enforce "
                       "dtype of keypoints by passing keypoints "
                       "in such format which by default will be converted into "
                       "the desired one." %
                       (inputs.dtype, self._interpolation_keypoints.dtype))
    # Here is calibration. Everything else is handling of missing.
    if inputs.shape[1] > 1:
      # Add dimension to multi dim input to get shape [batch_size, units, 1].
      # Interpolation will have shape [batch_size, units, weights].
      inputs_to_calibration = tf.expand_dims(inputs, -1)
    else:
      inputs_to_calibration = inputs
    interpolation_weights = pwl_calibration_lib.compute_interpolation_weights(
        inputs_to_calibration, self._interpolation_keypoints, self._lengths)
    if self.is_cyclic:
      # Need to add such last height to make all heights to sum up to 0.0 in
      # order to make calibrator cyclic.
      bias_and_heights = tf.concat(
          [self.kernel, -tf.reduce_sum(self.kernel[1:], axis=0, keepdims=True)],
          axis=0)
    else:
      bias_and_heights = self.kernel

    # bias_and_heights has shape [weight, units].
    if inputs.shape[1] > 1:
      # Multi dim input has interpolation shape [batch_size, units, weights].
      result = tf.reduce_sum(
          interpolation_weights * tf.transpose(bias_and_heights), axis=-1)
    else:
      # Single dim input has interpolation shape [batch_size, weights].
      result = tf.matmul(interpolation_weights, bias_and_heights)

    if self.impute_missing:
      if is_missing is None:
        if self.missing_input_value is None:
          raise ValueError("PWLCalibration layer is configured to impute "
                           "missing but no 'missing_input_value' specified and "
                           "'is_missing' tensor is not given.")
        assert self._missing_input_value_tensor is not None
        is_missing = tf.cast(
            tf.equal(inputs, self._missing_input_value_tensor),
            dtype=self.dtype)
      result = is_missing * self.missing_output + (1.0 - is_missing) * result
    return result


class _UniformOutputInitializer(snt.initializers.Initializer):
  # pyformat: disable
  """Initializes PWL calibration layer to represent linear function.

  PWL calibration layer weights are one-d tensor. First element of tensor
  represents bias. All remaining represent delta in y-value compare to previous
  point. Aka heights of segments.

  Attributes:
    - All `__init__` arguments.
  """
  # pyformat: enable

  def __init__(self, output_min, output_max, monotonicity, keypoints=None):
    # pyformat: disable
    """Initializes an instance of `_UniformOutputInitializer`.

    Args:
      output_min: Minimum value of PWL calibration output after initialization.
      output_max: Maximum value of PWL calibration output after initialization.
      monotonicity:
        - if 'none' or 'increasing', the returned function will go from
          `(input_min, output_min)` to `(input_max, output_max)`.
        - if 'decreasing', the returned function will go from
          `(input_min, output_max)` to `(input_max, output_min)`.
      keypoints:
        - if not provided (None or []), all pieces of returned function
          will have equal heights (i.e. `y[i+1] - y[i]` is constant).
        - if provided, all pieces of returned function will have equal slopes
          (i.e. `(y[i+1] - y[i]) / (x[i+1] - x[i])` is constant).
    """
    # pyformat: enable
    pwl_calibration_lib.verify_hyperparameters(
        input_keypoints=keypoints,
        output_min=output_min,
        output_max=output_max,
        monotonicity=monotonicity)
    self.output_min = output_min
    self.output_max = output_max
    self.monotonicity = monotonicity
    self.keypoints = keypoints

  def __call__(self, shape, dtype):
    """Returns weights of PWL calibration layer.

    Args:
      shape: Must be a collection of the form `(k, units)` where `k >= 2`.
      dtype: Standard Sonnet initializer param.

    Returns:
      Weights of PWL calibration layer.

    Raises:
      ValueError: If requested shape is invalid for PWL calibration layer
        weights.
    """
    return pwl_calibration_lib.linear_initializer(
        shape=shape,
        output_min=self.output_min,
        output_max=self.output_max,
        monotonicity=utils.canonicalize_monotonicity(self.monotonicity),
        keypoints=self.keypoints,
        dtype=dtype)


class _PWLCalibrationConstraints(object):
  # pyformat: disable
  """Monotonicity and bounds constraints for PWL calibration layer.

  Applies an approximate L2 projection to the weights of a PWLCalibration layer
  such that the result satisfies the specified constraints.

  Attributes:
    - All `__init__` arguments.
  """
  # pyformat: enable

  def __init__(
      self,
      monotonicity="none",
      convexity="none",
      lengths=None,
      output_min=None,
      output_max=None,
      output_min_constraints=pwl_calibration_lib.BoundConstraintsType.NONE,
      output_max_constraints=pwl_calibration_lib.BoundConstraintsType.NONE,
      num_projection_iterations=8):
    """Initializes an instance of `PWLCalibration`.

    Args:
      monotonicity: Same meaning as corresponding parameter of `PWLCalibration`.
      convexity: Same meaning as corresponding parameter of `PWLCalibration`.
      lengths: Lengths of pieces of piecewise linear function. Needed only if
        convexity is specified.
      output_min: Minimum possible output of pwl function.
      output_max: Maximum possible output of pwl function.
      output_min_constraints: A `tfl.pwl_calibration_lib.BoundConstraintsType`
        describing the constraints on the layer's minimum value.
      output_max_constraints: A `tfl.pwl_calibration_lib.BoundConstraintsType`
        describing the constraints on the layer's maximum value.
      num_projection_iterations: Same meaning as corresponding parameter of
        `PWLCalibration`.
    """
    pwl_calibration_lib.verify_hyperparameters(
        output_min=output_min,
        output_max=output_max,
        monotonicity=monotonicity,
        convexity=convexity,
        lengths=lengths)
    self.monotonicity = monotonicity
    self.convexity = convexity
    self.lengths = lengths
    self.output_min = output_min
    self.output_max = output_max
    self.output_min_constraints = output_min_constraints
    self.output_max_constraints = output_max_constraints
    self.num_projection_iterations = num_projection_iterations

    canonical_convexity = utils.canonicalize_convexity(self.convexity)
    canonical_monotonicity = utils.canonicalize_monotonicity(self.monotonicity)
    if (canonical_convexity != 0 and canonical_monotonicity == 0 and
        (output_min_constraints != pwl_calibration_lib.BoundConstraintsType.NONE
         or output_max_constraints !=
         pwl_calibration_lib.BoundConstraintsType.NONE)):
      logging.warning("Convexity constraints are specified with bounds "
                      "constraints, but without monotonicity. Such combination "
                      "might lead to convexity being slightly violated. "
                      "Consider increasing num_projection_iterations to "
                      "reduce violation.")

  def __call__(self, w):
    """Applies constraints to w."""
    return pwl_calibration_lib.project_all_constraints(
        weights=w,
        monotonicity=utils.canonicalize_monotonicity(self.monotonicity),
        output_min=self.output_min,
        output_max=self.output_max,
        output_min_constraints=self.output_min_constraints,
        output_max_constraints=self.output_max_constraints,
        convexity=utils.canonicalize_convexity(self.convexity),
        lengths=self.lengths,
        num_projection_iterations=self.num_projection_iterations)


class _NaiveBoundsConstraints(object):
  # pyformat: disable
  """Naively clips all elements of tensor to be within bounds.

  This constraint is used only for the weight tensor for missing output value.

  Attributes:
    - All `__init__` arguments.
  """
  # pyformat: enable

  def __init__(self, lower_bound=None, upper_bound=None):
    """Initializes an instance of `_NaiveBoundsConstraints`.

    Args:
      lower_bound: Lower bound to clip variable values to.
      upper_bound: Upper bound to clip variable values to.
    """
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound

  def __call__(self, w):
    """Applies constraints to w."""
    if self.lower_bound is not None:
      w = tf.maximum(w, self.lower_bound)
    if self.upper_bound is not None:
      w = tf.minimum(w, self.upper_bound)
    return w
