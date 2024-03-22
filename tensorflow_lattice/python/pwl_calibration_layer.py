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
"""Piecewise linear calibration layer.

Keras implementation of tensorflow lattice pwl calibration layer. Layer takes
single or multi-dimensional input and transforms it using piecewise linear
functions following monotonicity, convexity/concavity and bounds constraints if
specified.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import numpy as np
import six
import tensorflow as tf
# pylint: disable=g-import-not-at-top
# Use Keras 2.
version_fn = getattr(tf.keras, "version", None)
if version_fn and version_fn().startswith("3."):
  import tf_keras as keras
else:
  keras = tf.keras

from . import pwl_calibration_lib
from . import utils

INTERPOLATION_KEYPOINTS_NAME = "interpolation_keypoints"
LENGTHS_NAME = "lengths"
MISSING_INPUT_VALUE_NAME = "missing_input_value"
PWL_CALIBRATION_KERNEL_NAME = "pwl_calibration_kernel"
PWL_CALIBRATION_MISSING_OUTPUT_NAME = "pwl_calibration_missing_output"
INTERPOLATION_LOGITS_NAME = "interpolation_logits"


class PWLCalibration(keras.layers.Layer):
  # pyformat: disable
  """Piecewise linear calibration layer.

  Layer takes input of shape `(batch_size, units)` or `(batch_size, 1)` and
  transforms it using `units` number of piecewise linear functions following
  monotonicity, convexity and bounds constraints if specified. If multi
  dimensional input is provides, each output will be for the corresponding
  input, otherwise all PWL functions will act on the same input. All units share
  the same layer configuration, but each has their separate set of trained
  parameters.

  See `tfl.layers.ParallelCombination` layer for using PWLCalibration layer
  within Sequential Keras models.

  Input shape:
  Single input should be a rank-2 tensor with shape: `(batch_size, units)` or
  `(batch_size, 1)`. The input can also be a list of two tensors of the same
  shape where the first tensor is the regular input tensor and the second is the
  `is_missing` tensor. In the `is_missing` tensor, 1.0 represents missing input
  and 0.0 represents available input.

  Output shape:
  If units > 1 and split_outputs is True, a length `units` list of Rank-2
    tensors with shape `(batch_size, 1)`. Otherwise, a Rank-2 tensor with shape:
    `(batch_size, units)`

  Attributes:
    - All `__init__` arguments.
    kernel: TF variable which stores weights of piecewise linear function.
    missing_output: TF variable which stores output learned for missing input.
      Or TF Constant which stores `missing_output_value` if one is provided.
      Available only if `impute_missing` is True.

  Example:

  ```python
  calibrator = tfl.layers.PWLCalibration(
      # Key-points of piecewise-linear function.
      input_keypoints=np.linspace(1., 4., num=4),
      # Output can be bounded, e.g. when this layer feeds into a lattice.
      output_min=0.0,
      output_max=2.0,
      # You can specify monotonicity and other shape constraints for the layer.
      monotonicity='increasing',
      # You can specify TFL regularizers as tuple ('regularizer name', l1, l2).
      # You can also pass any keras Regularizer object.
      kernel_regularizer=('hessian', 0.0, 1e-4),
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
               kernel_initializer="equal_heights",
               kernel_regularizer=None,
               impute_missing=False,
               missing_input_value=None,
               missing_output_value=None,
               num_projection_iterations=8,
               split_outputs=False,
               input_keypoints_type="fixed",
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
      kernel_initializer: None or one of:
        - String `"equal_heights"`: For pieces of pwl function to have equal
          heights.
        - String `"equal_slopes"`: For pieces of pwl function to have equal
          slopes.
        - Any Keras initializer object. If you are passing such object make sure
          that you know how layer stores its data.
      kernel_regularizer: None or single element or list of following:
        - Tuple `("laplacian", l1, l2)` where `l1` and `l2` are floats which
          represent corresponding regularization amount for Laplacian
          regularizer. It penalizes the first derivative to make the function
          more constant. See `tfl.pwl_calibration.LaplacianRegularizer` for more
          details.
        - Tuple `("hessian", l1, l2)` where `l1` and `l2` are floats which
          represent corresponding regularization amount for Hessian regularizer.
          It penalizes the second derivative to make the function more linear.
          See `tfl.pwl_calibration.HessianRegularizer` for more details.
        - Tuple `("wrinkle", l1, l2)` where `l1` and `l2` are floats which
          represent corresponding regularization amount for wrinkle regularizer.
          It penalizes the third derivative to make the function more smooth.
          See 'tfl.pwl_calibration.WrinkleRegularizer` for more details.
        - Any Keras regularizer object.
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
      split_outputs: Whether to split the output tensor into a list of
        outputs for each unit. Ignored if units < 2.
      input_keypoints_type: One of "fixed" or "learned_interior". If
        "learned_interior", keypoints are initialized to the values in
        `input_keypoints` but then allowed to vary during training, with the
        exception of the first and last keypoint location which are fixed.
        Convexity can only be imposed with "fixed".
      **kwargs: Other args passed to `keras.layers.Layer` initializer.

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
        is_cyclic=is_cyclic,
        input_keypoints_type=input_keypoints_type)
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
    if convexity not in ("none",
                         0) and input_keypoints_type == "learned_interior":
      raise ValueError("Cannot set input_keypoints_type to 'learned_interior'"
                       " and impose convexity constraints.")

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

    if kernel_initializer == "equal_heights":
      self.kernel_initializer = UniformOutputInitializer(
          output_min=self._output_init_min,
          output_max=self._output_init_max,
          monotonicity=self.monotonicity)
    elif kernel_initializer == "equal_slopes":
      self.kernel_initializer = UniformOutputInitializer(
          output_min=self._output_init_min,
          output_max=self._output_init_max,
          monotonicity=self.monotonicity,
          keypoints=self.input_keypoints)
    else:
      # Keras deserialization logic must have explicit acceess to all custom
      # classes. This is standard way to provide such access.
      with keras.utils.custom_object_scope({
          "UniformOutputInitializer": UniformOutputInitializer,
      }):
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

    self.kernel_regularizer = []
    if kernel_regularizer:
      if (callable(kernel_regularizer) or
          (isinstance(kernel_regularizer, tuple) and
           isinstance(kernel_regularizer[0], six.string_types))):
        kernel_regularizer = [kernel_regularizer]

      for reg in kernel_regularizer:
        if isinstance(reg, tuple):
          (name, l1, l2) = reg
          if name.lower() == "laplacian":
            self.kernel_regularizer.append(
                LaplacianRegularizer(l1=l1, l2=l2, is_cyclic=self.is_cyclic))
          elif name.lower() == "hessian":
            self.kernel_regularizer.append(
                HessianRegularizer(l1=l1, l2=l2, is_cyclic=self.is_cyclic))
          elif name.lower() == "wrinkle":
            self.kernel_regularizer.append(
                WrinkleRegularizer(l1=l1, l2=l2, is_cyclic=self.is_cyclic))
          else:
            raise ValueError("Unknown custom lattice regularizer: %s" % reg)
        else:
          # This is needed for Keras deserialization logic to be aware of our
          # custom objects.
          with keras.utils.custom_object_scope({
              "LaplacianRegularizer": LaplacianRegularizer,
              "HessianRegularizer": HessianRegularizer,
              "WrinkleRegularizer": WrinkleRegularizer,
          }):
            self.kernel_regularizer.append(keras.regularizers.get(reg))

    self.impute_missing = impute_missing
    self.missing_input_value = missing_input_value
    self.missing_output_value = missing_output_value
    self.num_projection_iterations = num_projection_iterations
    self.split_outputs = split_outputs
    self.input_keypoints_type = input_keypoints_type

  def build(self, input_shape):
    """Standard Keras build() method."""
    input_keypoints = np.array(self.input_keypoints)
    # Don't need last keypoint for interpolation because we need only beginnings
    # of intervals.
    if self.input_keypoints_type == "fixed":
      self._interpolation_keypoints = tf.constant(
          input_keypoints[:-1],
          dtype=self.dtype,
          name=INTERPOLATION_KEYPOINTS_NAME)
      self._lengths = tf.constant(
          input_keypoints[1:] - input_keypoints[:-1],
          dtype=self.dtype,
          name=LENGTHS_NAME)
    else:
      self._keypoint_min = input_keypoints[0]
      self._keypoint_range = input_keypoints[-1] - input_keypoints[0]
      # Logits are initialized such that they will recover the scaled keypoint
      # gaps in input_keypoints.
      initial_logits = np.log(
          (input_keypoints[1:] - input_keypoints[:-1]) / self._keypoint_range)
      tiled_logits = np.tile(initial_logits, self.units)
      self.interpolation_logits = self.add_weight(
          INTERPOLATION_LOGITS_NAME,
          shape=[self.units, len(input_keypoints) - 1],
          initializer=tf.constant_initializer(tiled_logits),
          dtype=self.dtype)

    constraints = PWLCalibrationConstraints(
        monotonicity=self.monotonicity,
        convexity=self.convexity,
        lengths=self._lengths if self.input_keypoints_type == "fixed" else None,
        output_min=self.output_min,
        output_max=self.output_max,
        output_min_constraints=self._output_min_constraints,
        output_max_constraints=self._output_max_constraints,
        num_projection_iterations=self.num_projection_iterations)

    if not self.kernel_regularizer:
      kernel_reg = None
    elif len(self.kernel_regularizer) == 1:
      kernel_reg = self.kernel_regularizer[0]
    else:
      # Keras interface assumes only one regularizer, so summ all regularization
      # losses which we have.
      kernel_reg = lambda x: tf.add_n([r(x) for r in self.kernel_regularizer])

    # If 'is_cyclic' is specified - last weight will be computed from previous
    # weights in order to connect last keypoint with first.
    num_weights = input_keypoints.size - self.is_cyclic

    # PWL calibration layer kernel is units-column matrix. First row of matrix
    # represents bias. All remaining represent delta in y-value compare to
    # previous point. Aka heights of segments.
    self.kernel = self.add_weight(
        PWL_CALIBRATION_KERNEL_NAME,
        shape=[num_weights, self.units],
        initializer=self.kernel_initializer,
        regularizer=kernel_reg,
        constraint=constraints,
        dtype=self.dtype)

    if self.kernel_regularizer and not tf.executing_eagerly():
      # Keras has its own mechanism to handle regularization losses which
      # does not use GraphKeys, but we want to also add losses to graph keys so
      # they are easily accessable when layer is being used outside of Keras.
      # Adding losses to GraphKeys will not interfer with Keras.
      for reg in self.kernel_regularizer:
        tf.compat.v1.add_to_collection(
            tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES, reg(self.kernel))

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
        missing_constraints = NaiveBoundsConstraints(
            lower_bound=self.output_min, upper_bound=self.output_max)
        self.missing_output = self.add_weight(
            PWL_CALIBRATION_MISSING_OUTPUT_NAME,
            shape=[1, self.units],
            initializer=keras.initializers.Constant(value=missing_init),
            constraint=missing_constraints,
            dtype=self.dtype)

    super(PWLCalibration, self).build(input_shape)

  def call(self, inputs):
    """Standard Keras call() method..

    Args:
      inputs: Either input tensor or list of 2 elements: input tensor and
        `is_missing` tensor.

    Returns:
      Calibrated input tensor.

    Raises:
      ValueError: If `is_missing` tensor specified incorrectly.
    """
    is_missing = None
    if isinstance(inputs, list):
      # Only 2 element lists are allowed. When such list is given - second
      # element represents 'is_missing' tensor encoded as float value.
      if not self.impute_missing:
        raise ValueError("Multiple inputs for PWLCalibration layer assume "
                         "regular input tensor and 'is_missing' tensor, but "
                         "this instance of a layer is not configured to handle "
                         "missing value. See 'impute_missing' parameter.")
      if len(inputs) > 2:
        raise ValueError("Multiple inputs for PWLCalibration layer assume "
                         "normal input tensor and 'is_missing' tensor, but more"
                         " than 2 tensors given. 'inputs': " + str(inputs))
      if len(inputs) == 2:
        inputs, is_missing = inputs
        if is_missing.shape.as_list() != inputs.shape.as_list():
          raise ValueError(
              "is_missing shape %s does not match inputs shape %s for "
              "PWLCalibration layer" %
              (str(is_missing.shape), str(inputs.shape)))
      else:
        [inputs] = inputs
    if len(inputs.shape) != 2 or (inputs.shape[1] != self.units and
                                  inputs.shape[1] != 1):
      raise ValueError("Shape of input tensor for PWLCalibration layer must be "
                       "[-1, units] or [-1, 1]. It is: " + str(inputs.shape))

    if self.input_keypoints_type == "fixed":
      keypoints_dtype = self._interpolation_keypoints.dtype
    else:
      keypoints_dtype = self.interpolation_logits.dtype
    if inputs.dtype != keypoints_dtype:
      raise ValueError("dtype(%s) of input to PWLCalibration layer does not "
                       "correspond to dtype(%s) of keypoints. You can enforce "
                       "dtype of keypoints by explicitly providing 'dtype' "
                       "parameter to layer constructor or by passing keypoints "
                       "in such format which by default will be converted into "
                       "desired one." % (inputs.dtype, keypoints_dtype))

    # Here is calibration. Everything else is handling of missing.
    if inputs.shape[1] > 1 or (self.input_keypoints_type == "learned_interior"
                               and self.units > 1):
      # Interpolation will have shape [batch_size, units, weights] in these
      # cases. To prepare for that, we add a dimension to the input here to get
      # shape [batch_size, units, 1] or [batch_size, 1, 1] if 1d input.
      inputs_to_calibration = tf.expand_dims(inputs, -1)
    else:
      inputs_to_calibration = inputs
    if self.input_keypoints_type == "learned_interior":
      self._lengths = tf.multiply(
          tf.nn.softmax(self.interpolation_logits, axis=1),
          self._keypoint_range,
          name=LENGTHS_NAME)
      self._interpolation_keypoints = tf.add(
          tf.cumsum(self._lengths, axis=1, exclusive=True),
          self._keypoint_min,
          name=INTERPOLATION_KEYPOINTS_NAME)
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
    if len(interpolation_weights.shape) > 2:
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

    if self.units > 1 and self.split_outputs:
      result = tf.split(result, self.units, axis=1)

    return result

  def compute_output_shape(self, input_shape):
    """Standard Keras compute_output_shape() method."""
    del input_shape
    if self.units > 1 and self.split_outputs:
      return [(None, 1)] * self.units
    else:
      return (None, self.units)

  def get_config(self):
    """Standard Keras config for serialization."""
    config = {
        "input_keypoints": self.input_keypoints,
        "units": self.units,
        "output_min": self.output_min,
        "output_max": self.output_max,
        "clamp_min": self.clamp_min,
        "clamp_max": self.clamp_max,
        "monotonicity": self.monotonicity,
        "convexity": self.convexity,
        "is_cyclic": self.is_cyclic,
        "kernel_initializer":
            keras.initializers.serialize(
                self.kernel_initializer, use_legacy_format=True),
        "kernel_regularizer":
            [keras.regularizers.serialize(r, use_legacy_format=True)
             for r in self.kernel_regularizer],
        "impute_missing": self.impute_missing,
        "missing_input_value": self.missing_input_value,
        "num_projection_iterations": self.num_projection_iterations,
        "split_outputs": self.split_outputs,
        "input_keypoints_type": self.input_keypoints_type,
    }  # pyformat: disable
    config.update(super(PWLCalibration, self).get_config())
    return config

  def assert_constraints(self, eps=1e-6):
    """Asserts that layer weights satisfy all constraints.

    In graph mode builds and returns list of assertion ops. Note that ops will
    be created at the moment when this function is being called.
    In eager mode directly executes assertions.

    Args:
      eps: Allowed constraints violation.

    Returns:
      List of assertion ops in graph mode or immediately asserts in eager mode.
    """
    # Assert by computing outputs for keypoints and testing them against
    # constraints.
    test_inputs = tf.constant(
        value=self.input_keypoints,
        dtype=self.dtype,
        shape=[len(self.input_keypoints), 1])
    outputs = self.call(test_inputs)

    asserts = pwl_calibration_lib.assert_constraints(
        outputs=outputs,
        monotonicity=utils.canonicalize_monotonicity(self.monotonicity),
        output_min=self.output_min,
        output_max=self.output_max,
        clamp_min=self.clamp_min,
        clamp_max=self.clamp_max,
        debug_tensors=["weights:", self.kernel],
        eps=eps)

    if self.impute_missing and self.missing_output_value is None:
      asserts.append(
          pwl_calibration_lib.assert_constraints(
              outputs=self.missing_output,
              monotonicity=0,
              output_min=self.output_min,
              output_max=self.output_max,
              clamp_min=False,
              clamp_max=False,
              debug_tensors=["Imputed missing value:", self.missing_output],
              eps=eps))
    return asserts

  def keypoints_outputs(self):
    """Returns tensor of keypoint outputs of shape [num_weights, num_units]."""
    kp_outputs = tf.cumsum(self.kernel)
    if self.is_cyclic:
      kp_outputs = tf.concat([kp_outputs, kp_outputs[0:1]], axis=0)
    return kp_outputs

  def keypoints_inputs(self):
    """Returns tensor of keypoint inputs of shape [num_weights, num_units]."""
    # We don't store the last keypoint in self._interpolation_keypoints since
    # it is not needed for training or evaluation, but we re-add it here to
    # align with the keypoints_outputs function.
    if self.input_keypoints_type == "fixed":
      all_keypoints = tf.concat([
          self._interpolation_keypoints,
          self._interpolation_keypoints[-1:] + self._lengths[-1:]
      ],
                                axis=0)
      return tf.stack([all_keypoints] * self.units, axis=1)
    else:
      lengths = tf.nn.softmax(
          self.interpolation_logits, axis=-1) * self._keypoint_range
      interpolation_keypoints = tf.cumsum(
          lengths, axis=-1, exclusive=True) + self._keypoint_min
      all_keypoints = tf.concat([
          interpolation_keypoints,
          interpolation_keypoints[:, -1:] + lengths[:, -1:]
      ],
                                axis=1)
      return tf.transpose(all_keypoints)


class UniformOutputInitializer(keras.initializers.Initializer):
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
    """Initializes an instance of `UniformOutputInitializer`.

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
    """  # pyformat: enable
    pwl_calibration_lib.verify_hyperparameters(
        input_keypoints=keypoints,
        output_min=output_min,
        output_max=output_max,
        monotonicity=monotonicity)
    self.output_min = output_min
    self.output_max = output_max
    self.monotonicity = monotonicity
    self.keypoints = keypoints

  def __call__(self, shape, dtype=None, partition_info=None):
    """Returns weights of PWL calibration layer.

    Args:
      shape: Must be a collection of the form `(k, units)` where `k >= 2`.
      dtype: Standard Keras initializer param.
      partition_info: Standard Keras initializer param.

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

  def get_config(self):
    """Standard Keras config for serialization."""
    return {
        "output_min": self.output_min,
        "output_max": self.output_max,
        "monotonicity": self.monotonicity,
        "keypoints": self.keypoints,
    }  # pyformat: disable


class PWLCalibrationConstraints(keras.constraints.Constraint):
  # pyformat: disable
  """Monotonicity and bounds constraints for PWL calibration layer.

  Applies an approximate L2 projection to the weights of a PWLCalibration layer
  such that the result satisfies the specified constraints.

  Attributes:
    - All `__init__` arguments.
  """  # pyformat: enable

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

  def get_config(self):
    """Standard Keras config for serialization."""
    return {
        "monotonicity": self.monotonicity,
        "output_min": self.output_min,
        "output_max": self.output_max,
        "output_min_constraints": self.output_min_constraints,
        "output_max_constraints": self.output_max_constraints,
        "convexity": self.convexity,
        "lengths": self.lengths,
        "num_projection_iterations": self.num_projection_iterations,
    }  # pyformat: disable


class NaiveBoundsConstraints(keras.constraints.Constraint):
  # pyformat: disable
  """Naively clips all elements of tensor to be within bounds.

  This constraint is used only for the weight tensor for missing output value.

  Attributes:
    - All `__init__` arguments.
  """  # pyformat: enable

  def __init__(self, lower_bound=None, upper_bound=None):
    """Initializes an instance of `NaiveBoundsConstraints`.

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

  def get_config(self):
    """Standard Keras config for serialization."""
    return {
        "lower_bound": self.lower_bound,
        "upper_bound": self.upper_bound
    }  # pyformat: disable


class LaplacianRegularizer(keras.regularizers.Regularizer):
  # pyformat: disable
  """Laplacian regularizer for PWL calibration layer.

  Calibrator Laplacian regularization penalizes the change in the calibration
  output. It is defined to be:

  `l1 * ||delta||_1 + l2 * ||delta||_2^2`

  where `delta` is:

  `output_keypoints[1:end] - output_keypoints[0:end-1]`.

  Attributes:
    - All `__init__` arguments.
  """  # pyformat: enable

  def __init__(self, l1=0.0, l2=0.0, is_cyclic=False):
    """Initializes an instance of `LaplacianRegularizer`.

    Args:
      l1: l1 regularization amount as float.
      l2: l2 regularization amount as float.
      is_cyclic: Whether the first and last keypoints should take the same
        output value.
    """
    self.l1 = l1
    self.l2 = l2
    self.is_cyclic = is_cyclic

  def __call__(self, x):
    """Returns regularization loss.

    Args:
      x: Tensor of shape: `(k, units)` which represents weights of PWL
        calibration layer. First row of weights is bias term. All remaining
        represent delta in y-value compare to previous point (segment heights).
    """
    if not self.l1 and not self.l2:
      return tf.constant(0.0, dtype=x.dtype, shape=())
    heights = x[1:]
    if self.is_cyclic:
      # Need to add such last height to make all heights to sum up to 0.0 in
      # order to make calibrator cyclic.
      heights = tf.concat(
          [heights, -tf.reduce_sum(heights, axis=0, keepdims=True)], axis=0)

    losses = []
    if self.l1:
      losses.append(self.l1 * tf.reduce_sum(tf.abs(heights)))
    if self.l2:
      losses.append(self.l2 * tf.reduce_sum(tf.square(heights)))

    result = losses[0]
    if len(losses) == 2:
      result += losses[1]
    return result

  def get_config(self):
    """Standard Keras config for serialization."""
    return {
        "l1": self.l1,
        "l2": self.l2,
        "is_cyclic": self.is_cyclic,
    }  # pyformat: disable


class HessianRegularizer(keras.regularizers.Regularizer):
  # pyformat: disable
  """Hessian regularizer for PWL calibration layer.

  Calibrator hessian regularizer penalizes the change in slopes of linear
  pieces. It is define to be:

  `l1 * ||nonlinearity||_1 + l2 * ||nonlinearity||_2^2`

  where `nonlinearity` is:

  `2 * output_keypoints[1:end-1] - output_keypoints[0:end-2]
     - output_keypoints[2:end]`.

  This regularizer is zero when the output_keypoints form a linear function of
  the index (and not necessarily linear in input values, e.g. when using
  non-uniform input keypoints).

  Attributes:
    - All `__init__` arguments.
  """  # pyformat: enable

  def __init__(self, l1=0.0, l2=0.0, is_cyclic=False):
    """Initializes an instance of `HessianRegularizer`.

    Args:
      l1: l1 regularization amount as float.
      l2: l2 regularization amount as float.
      is_cyclic: Whether the first and last keypoints should take the same
        output value.
    """
    self.l1 = l1
    self.l2 = l2
    self.is_cyclic = is_cyclic

  def __call__(self, x):
    """Returns regularization loss.

    Args:
      x: Tensor of shape: `(k, units)` which represents weights of PWL
        calibration layer. First row of weights is bias term. All remaining
        represent delta in y-value compare to previous point (segment heights).
    """
    if not self.l1 and not self.l2:
      return tf.constant(0.0, dtype=x.dtype, shape=())

    if self.is_cyclic:
      heights = x[1:]
      heights = tf.concat(
          [
              heights,
              -tf.reduce_sum(heights, axis=0, keepdims=True),
              heights[0:1],
          ],
          axis=0,
      )
      nonlinearity = heights[1:] - heights[:-1]
    else:
      nonlinearity = x[2:] - x[1:-1]

    losses = []
    if self.l1:
      losses.append(self.l1 * tf.reduce_sum(tf.abs(nonlinearity)))
    if self.l2:
      losses.append(self.l2 * tf.reduce_sum(tf.square(nonlinearity)))

    result = losses[0]
    if len(losses) == 2:
      result += losses[1]
    return result

  def get_config(self):
    """Standard Keras config for serialization."""
    return {
        "l1": self.l1,
        "l2": self.l2,
        "is_cyclic": self.is_cyclic,
    }  # pyformat: disable


class WrinkleRegularizer(keras.regularizers.Regularizer):
  # pyformat: disable
  """Wrinkle regularizer for PWL calibration layer.

  Calibrator wrinkle regularization penalizes the change in the second
  derivative. It is defined to be:

  `l1 * ||third_derivative||_1 + l2 * ||third_derivative||_2^2`

  where `third_derivative` is:

  `3 * output_keypoints[1:end-2] - 3 * output_keypoints[2:end-1]
   - output_keypoints[0:end-3] + output_keypoints[3:end]`.

  This regularizer is zero when the output_keypoints form a 2nd order polynomial
  of the index (and not necessarily in input values, e.g. when using
  non-uniform input keypoints).

  Attributes:
    - All `__init__` arguments.
  """  # pyformat: enable

  def __init__(self, l1=0.0, l2=0.0, is_cyclic=False):
    """Initializes an instance of `WrinkleRegularizer`.

    Args:
      l1: l1 regularization amount as float.
      l2: l2 regularization amount as float.
      is_cyclic: Whether the first and last keypoints should take the same
        output value.
    """
    self.l1 = l1
    self.l2 = l2
    self.is_cyclic = is_cyclic

  def __call__(self, x):
    """Returns regularization loss.

    Args:
      x: Tensor of shape: `(k, units)` which represents weights of PWL
        calibration layer. First row of weights is bias term. All remaining
        represent delta in y-value compare to previous point (segment heights).
    """
    if not self.l1 and not self.l2:
      return tf.constant(0.0, dtype=x.dtype, shape=())
    if x.shape[0] < 3:
      return tf.constant(0.0, dtype=x.dtype, shape=())

    if self.is_cyclic:
      heights = x[1:]
      heights = tf.concat(
          [
              heights,
              -tf.reduce_sum(heights, axis=0, keepdims=True),
              heights[0:1],
              heights[1:2],
          ],
          axis=0,
      )
      nonlinearity = heights[1:] - heights[:-1]
    else:
      nonlinearity = x[2:] - x[1:-1]
    wrinkleness = nonlinearity[1:] - nonlinearity[0:-1]

    losses = []
    if self.l1:
      losses.append(self.l1 * tf.reduce_sum(tf.abs(wrinkleness)))
    if self.l2:
      losses.append(self.l2 * tf.reduce_sum(tf.square(wrinkleness)))

    result = losses[0]
    if len(losses) == 2:
      result += losses[1]
    return result

  def get_config(self):
    """Standard Keras config for serialization."""
    return {
        "l1": self.l1,
        "l2": self.l2,
        "is_cyclic": self.is_cyclic,
    }  # pyformat: disable
