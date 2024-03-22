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
"""Categorical calibration layer with monotonicity and bound constraints.

Keras implementation of tensorflow lattice categorical calibration layer. This
layer takes single or multi-dimensional input and transforms it using lookup
tables satisfying monotonicity and bounds constraints if specified.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
# pylint: disable=g-import-not-at-top
# Use Keras 2.
version_fn = getattr(tf.keras, "version", None)
if version_fn and version_fn().startswith("3."):
  import tf_keras as keras
else:
  keras = tf.keras
from . import categorical_calibration_lib

DEFAULT_INPUT_VALUE_NAME = "default_input_value"
CATEGORICAL_CALIBRATION_KERNEL_NAME = "categorical_calibration_kernel"

# TODO: implement variation/variance regularizer.


class CategoricalCalibration(keras.layers.Layer):
  # pyformat: disable
  """Categorical calibration layer with monotonicity and bound constraints.

  This layer takes input of shape `(batch_size, units)` or `(batch_size, 1)` and
  transforms it using `units` number of lookup tables satisfying monotonicity
  and bounds constraints if specified. If multi dimensional input is provided,
  each output will be for the corresponding input, otherwise all calibration
  functions will act on the same input. All units share the same layer
  configuration, but each one has their separate set of trained parameters.

  Input shape:
  Rank-2 tensor with shape:  `(batch_size, units)` or `(batch_size, 1)`.

  Output shape:
  If units > 1 and split_outputs is True, a length `units` list of Rank-2
    tensors with shape `(batch_size, 1)`. Otherwise, a Rank-2 tensor with shape:
    `(batch_size, units)`

  Attributes:
    - All `__init__` args.
    kernel: TF variable of shape `(batch_size, units)` which stores the lookup
    table.

  Example:

  ```python
  calibrator = tfl.layers.CategoricalCalibration(
      # Number of categories.
      num_buckets=3,
      # Output can be bounded.
      output_min=0.0,
      output_max=1.0,
      # For categorical calibration layer monotonicity is specified for pairs of
      # indices of categories. Output for first category in pair will be less
      # than or equal to output for second category.
      monotonicities=[(0, 1), (0, 2)])
  ```

  Usage with functional models:

  ```python
  input_feature = keras.layers.Input(shape=[1])
  calibrated_feature = tfl.layers.CategoricalCalibration(
      num_buckets=3,
      output_min=0.0,
      output_max=1.0,
      monotonicities=[(0, 1), (0, 2)],
  )(feature)
  ...
  model = keras.models.Model(
      inputs=[input_feature, ...],
      outputs=...)
  ```
  """
  # pyformat: enable

  def __init__(self,
               num_buckets,
               units=1,
               output_min=None,
               output_max=None,
               monotonicities=None,
               kernel_initializer="uniform",
               kernel_regularizer=None,
               default_input_value=None,
               split_outputs=False,
               **kwargs):
    # pyformat: disable
    """Initializes a `CategoricalCalibration` instance.

    Args:
      num_buckets: Number of categories.
      units: Output dimension of the layer. See class comments for details.
      output_min: Minimum output of calibrator.
      output_max: Maximum output of calibrator.
      monotonicities: List of pairs with `(i, j)` indices indicating `output(i)`
        should be less than or equal to `output(j)`.
      kernel_initializer: None or one of:
        - `'uniform'`: If `output_min` and `output_max` are provided initial
          values will be uniformly sampled from `[output_min, output_max]`
          range.
        - `'constant'`: If `output_min` and `output_max` are provided all output
          values will be initlized to the constant
          `(output_min + output_max) / 2`.
        - Any Keras initializer object.
      kernel_regularizer: None or single element or list of any Keras
        regularizer objects.
      default_input_value: If set, all inputs which are equal to this value will
        be treated as default and mapped to the last bucket.
      split_outputs: Whether to split the output tensor into a list of
        outputs for each unit. Ignored if units < 2.
      **kwargs: Other args passed to `keras.layers.Layer` initializer.

    Raises:
      ValueError: If layer hyperparameters are invalid.
    """
    # pyformat: enable
    dtype = kwargs.pop("dtype", tf.float32)  # output dtype
    super(CategoricalCalibration, self).__init__(dtype=dtype, **kwargs)

    categorical_calibration_lib.verify_hyperparameters(
        num_buckets=num_buckets,
        output_min=output_min,
        output_max=output_max,
        monotonicities=monotonicities)
    self.num_buckets = num_buckets
    self.units = units
    self.output_min = output_min
    self.output_max = output_max
    self.monotonicities = monotonicities
    if output_min is not None and output_max is not None:
      if kernel_initializer == "constant":
        kernel_initializer = keras.initializers.Constant(
            (output_min + output_max) / 2)
      elif kernel_initializer == "uniform":
        kernel_initializer = keras.initializers.RandomUniform(
            output_min, output_max)
    self.kernel_initializer = keras.initializers.get(kernel_initializer)
    self.kernel_regularizer = []
    if kernel_regularizer:
      if callable(kernel_regularizer):
        kernel_regularizer = [kernel_regularizer]
      for reg in kernel_regularizer:
        self.kernel_regularizer.append(keras.regularizers.get(reg))
    self.default_input_value = default_input_value
    self.split_outputs = split_outputs

  def build(self, input_shape):
    """Standard Keras build() method."""
    if (self.output_min is not None or self.output_max is not None or
        self.monotonicities):
      constraints = CategoricalCalibrationConstraints(
          output_min=self.output_min,
          output_max=self.output_max,
          monotonicities=self.monotonicities)
    else:
      constraints = None

    if not self.kernel_regularizer:
      kernel_reg = None
    elif len(self.kernel_regularizer) == 1:
      kernel_reg = self.kernel_regularizer[0]
    else:
      # Keras interface assumes only one regularizer, so summ all regularization
      # losses which we have.
      kernel_reg = lambda x: tf.add_n([r(x) for r in self.kernel_regularizer])

    # categorical calibration layer kernel is units-column matrix with value of
    # output(i) = self.kernel[i]. Default value converted to the last index.
    self.kernel = self.add_weight(
        CATEGORICAL_CALIBRATION_KERNEL_NAME,
        shape=[self.num_buckets, self.units],
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

    super(CategoricalCalibration, self).build(input_shape)

  def call(self, inputs):
    """Standard Keras call() method."""
    if inputs.dtype not in [tf.uint8, tf.int32, tf.int64]:
      inputs = tf.cast(inputs, dtype=tf.int32)

    if self.default_input_value is not None:
      default_input_value_tensor = tf.constant(
          int(self.default_input_value),
          dtype=inputs.dtype,
          name=DEFAULT_INPUT_VALUE_NAME)
      replacement = tf.zeros_like(inputs) + (self.num_buckets - 1)
      inputs = tf.where(
          tf.equal(inputs, default_input_value_tensor), replacement, inputs)

    # We can't use tf.gather_nd(self.kernel, inputs) as it doesn't support
    # constraints (constraint functions are not supported for IndexedSlices).
    # Instead we use matrix multiplication by one-hot encoding of the index.
    if self.units == 1:
      # This can be slightly faster as it uses matmul.
      return tf.matmul(
          tf.one_hot(tf.squeeze(inputs, axis=[-1]), depth=self.num_buckets),
          self.kernel)
    result = tf.reduce_sum(
        tf.one_hot(inputs, axis=1, depth=self.num_buckets) * self.kernel,
        axis=1)

    if self.split_outputs:
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
        "num_buckets": self.num_buckets,
        "units": self.units,
        "output_min": self.output_min,
        "output_max": self.output_max,
        "monotonicities": self.monotonicities,
        "kernel_initializer":
            keras.initializers.serialize(
                self.kernel_initializer, use_legacy_format=True),
        "kernel_regularizer":
            [keras.regularizers.serialize(r, use_legacy_format=True)
             for r in self.kernel_regularizer],
        "default_input_value": self.default_input_value,
        "split_outputs": self.split_outputs,
    }  # pyformat: disable
    config.update(super(CategoricalCalibration, self).get_config())
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
    return categorical_calibration_lib.assert_constraints(
        weights=self.kernel,
        output_min=self.output_min,
        output_max=self.output_max,
        monotonicities=self.monotonicities,
        eps=eps)


class CategoricalCalibrationConstraints(keras.constraints.Constraint):
  # pyformat: disable
  """Monotonicity and bounds constraints for categorical calibration layer.

  Updates the weights of CategoricalCalibration layer to satify bound and
  monotonicity constraints. The update is an approximate L2 projection into the
  constrained parameter space.

  Attributes:
    - All `__init__` arguments.
  """
  # pyformat: enable

  def __init__(self, output_min=None, output_max=None, monotonicities=None):
    """Initializes an instance of `CategoricalCalibrationConstraints`.

    Args:
      output_min: Minimum possible output of categorical function.
      output_max: Maximum possible output of categorical function.
      monotonicities: Monotonicities of CategoricalCalibration layer.
    """
    categorical_calibration_lib.verify_hyperparameters(
        output_min=output_min,
        output_max=output_max,
        monotonicities=monotonicities)
    self.monotonicities = monotonicities
    self.output_min = output_min
    self.output_max = output_max

  def __call__(self, w):
    """Applies constraints to w."""
    return categorical_calibration_lib.project(
        weights=w,
        output_min=self.output_min,
        output_max=self.output_max,
        monotonicities=self.monotonicities)

  def get_config(self):
    """Standard Keras config for serialization."""
    return {
        "output_min": self.output_min,
        "output_max": self.output_max,
        "monotonicities": self.monotonicities,
    }  # pyformat: disable
