# Copyright 2021 Google LLC
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
"""Projection free Cumulative Distribution Function layer.

Keras implementation of TensorFlow Lattice CDF layer. Layer takes single or
multi-dimensional input and transforms it using a set of step functions. The
layer is naturally monotonic and bounded to the range [0, 1].
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
from . import utils


class CDF(keras.layers.Layer):
  # pyformat: disable
  """Cumulative Distribution Function (CDF) layer.

  Layer takes input of shape `(batch_size, input_dim)` or `(batch_size, 1)` and
  transforms it using `input_dim` number of cumulative distribution functions,
  which are naturally monotonic and bounded to the range [0, 1]. If multi
  dimensional input is provided, each output will be for the corresponding
  input, otherwise all CDF functions will act on the same input. All units share
  the same layer configuration, but each has their separate set of trained
  parameters. The smoothness of the cumulative distribution functions depends on
  the number of keypoints (i.e. step functions), the activation, and input
  scaling.

  Input shape:
  Single input should be a rank-2 tensor with shape: `(batch_size, input_dim)`
  or `(batch_size, 1)`.

  Output shape:
  Rank-2 tensor with shape `(batch, input_dim / factor, units)` if
  `reduction=='none'`. Otherwise a rank-2 tensor with shape
  `(batch_size, units)`.

  Attributes:
    - All `__init__` arguments.
    kernel: TF variable which stores weights of each cdf function.
    input_scaling: A constant if `input_scaling_type` is `'fixed'`, and a TF
      variable if set to `'learned'`.

  Example:

  ```python
  cdf = tfl.layers.CDF(
    num_keypoints=10,
    units=10,
    # You can specify the type of activation to use for the step functions.
    activation="sigmoid",
    # You can specifyc the type of reduction to use across the input dimension.
    reduction="mean",
    # The input scaling type determines whether or not to use a fixed value or
    # to learn the value during training.
    input_scaling_type="fixed",
    # You can make the layer less connected by increasing the pruning factor,
    # which must be a divisor of both the input dimension and units.
    sparsity_factor=1,
  )
  ```
  """

  def __init__(self,
               num_keypoints,
               units=1,
               activation="relu6",
               reduction="mean",
               input_scaling_init=None,
               input_scaling_type="fixed",
               input_scaling_monotonicity="increasing",
               sparsity_factor=1,
               kernel_initializer="random_uniform",
               **kwargs):
    # pyformat: disable
    """Initializes an instance of `Lattice`.

    Args:
      num_keypoints: The number of keypoints (i.e. step functions) to use for
        each of `units` CDF functions.
      units: The output dimension of the layer.
      activation: The activation function to use for the step functions. One of:
        - `'relu6'`: The `tf.nn.relu6` function.
        - `'sigmoid'`: The `tf.nn.sigmoid` function.
      reduction: The reduction used for each of the `units` CDF functions to
        combine the CDF function output for each input dimension. One of:
        - `'mean'`: The `tf.reduce_mean` function.
        - `'geometric_mean'`: The n'th root of the product of each of the n
          input dimensions.
        - `'none'`: No input reduction.
      input_scaling_init: The value used to initialize the input scaling.
        Defaults to `num_keypoints` if set to `None`.
      input_scaling_type: The type of input scaling to use. One of:
        - `'fixed'`: input scaling will be a constant with value
          `input_scaling_init`. This will be the value used for all input
          dimensions.
        - `'learned_shared'`: input scaling will be a weight learned during
          training initialized with value `input_scaling_init`. This will be the
          value used for all input dimensions.
        - `'learned_per_input'`: input scaling will be a weight learned during
          training initialized with value `input_scaling_init`. A separate value
          will be learned for each input dimension.
      input_scaling_monotonicity: One of:
        - `'increasing'` or `1`: input scaling will be constrained to be
          non-negative such that the output of the layer is monotonic in each
          dimension.
        - `'none'` or `0`: input scaling will not be constrained and the output
          of the layer will no be guaranteed to be monotonic.
      sparsity_factor: The factor by which to prune the connectivity of the
        layer. If set to `1` there will be no pruning and the layer will be
        fully connected. If set to `>1` the layer will be partially connected
        where the number of connections will be reduced by this factor. Must be
        a divisor of both the `input_dim` and `units`.
      kernel_initializer: None or one of:
        - `'random_uniform'`: initializes parameters as uniform
          random functions in the range [0, 1].
        - Any Keras initializer object.
      **kwargs: Any additional `keras.layers.Layer` arguments.
    """
    # pyformat: enable
    super(CDF, self).__init__(**kwargs)
    self.num_keypoints = num_keypoints
    self.units = units
    self.activation = activation
    self.reduction = reduction
    if input_scaling_init is None:
      self.input_scaling_init = float(num_keypoints)
    else:
      self.input_scaling_init = float(input_scaling_init)
    self.input_scaling_type = input_scaling_type
    self.input_scaling_monotonicity = utils.canonicalize_monotonicity(
        input_scaling_monotonicity)
    self.sparsity_factor = sparsity_factor

    self.kernel_initializer = create_kernel_initializer(
        kernel_initializer_id=kernel_initializer)

  def build(self, input_shape):
    """Standard Keras build() method."""
    input_dim = int(input_shape[-1])
    if input_dim % self.sparsity_factor != 0:
      raise ValueError(
          "sparsity_factor ({}) must be a divisor of input_dim ({})".format(
              self.sparsity_factor, input_dim))
    if self.units % self.sparsity_factor != 0:
      raise ValueError(
          "sparsity_factor ({}) must be a divisor of units ({})".format(
              self.sparsity_factor, self.units))

    # Each keypoint represents a step function defined by the activation
    # function specified. For an activation like relu6, this represents the
    # the hinge point.
    self.kernel = self.add_weight(
        "kernel",
        initializer=self.kernel_initializer,
        shape=[
            1, input_dim, self.num_keypoints,
            int(self.units // self.sparsity_factor)
        ])

    # Input scaling ultimately represents the slope of the step function used.
    # If the type is "learned_*" then input scaling will be a variable weight
    # that is constrained depending on the monotonicity specified.
    if self.input_scaling_type == "fixed":
      self.input_scaling = tf.constant(self.input_scaling_init)
    elif self.input_scaling_type == "learned_shared":
      self.input_scaling = self.add_weight(
          "input_scaling",
          initializer=keras.initializers.Constant(self.input_scaling_init),
          constraint=keras.constraints.NonNeg()
          if self.input_scaling_monotonicity else None,
          shape=[1])
    elif self.input_scaling_type == "learned_per_input":
      self.input_scaling = self.add_weight(
          "input_scaling",
          initializer=keras.initializers.Constant(self.input_scaling_init),
          constraint=keras.constraints.NonNeg()
          if self.input_scaling_monotonicity else None,
          shape=[1, input_dim, 1, 1])
    else:
      raise ValueError("Invalid input_scaling_type: {}".format(
          self.input_scaling_type))

  def call(self, inputs):
    """Standard Keras call() method."""
    input_dim = int(inputs.shape[-1])
    # We add new axes to enable broadcasting.
    x = inputs[..., tf.newaxis, tf.newaxis]

    # Shape: (batch, input_dim, 1, 1)
    #    --> (batch, input_dim, num_keypoints, units / factor)
    #    --> (batch, input_dim, units / factor)
    if self.activation == "relu6":
      cdfs = tf.reduce_mean(
          tf.nn.relu6(self.input_scaling * (x - self.kernel)), axis=2) / 6
    elif self.activation == "sigmoid":
      cdfs = tf.reduce_mean(
          tf.nn.sigmoid(self.input_scaling * (x - self.kernel)), axis=2)
    else:
      raise ValueError("Invalid activation: {}".format(self.activation))

    result = cdfs

    if self.sparsity_factor != 1:
      # Shape: (batch, input_dim, units / factor)
      #    --> (batch, input_dim / factor, units)
      result = tf.reshape(
          result, [-1, int(input_dim // self.sparsity_factor), self.units])

    # Shape: (batch, input_dim / factor, units)
    #.   --> (batch, units)
    if self.reduction == "mean":
      result = tf.reduce_mean(result, axis=1)
    elif self.reduction == "geometric_mean":
      num_terms = input_dim // self.sparsity_factor
      result = tf.math.exp(
          tf.reduce_sum(tf.math.log(result + 1e-3), axis=1) / num_terms)
      # we use the log form above so that we can add the epsilon term
      # tf.pow(tf.reduce_prod(cdfs, axis=1), 1. / num_terms)
    elif self.reduction != "none":
      raise ValueError("Invalid reduction: {}".format(self.reduction))

    return result

  def get_config(self):
    """Standard Keras get_config() method."""
    config = {
        "num_keypoints":
            self.num_keypoints,
        "units":
            self.units,
        "activation":
            self.activation,
        "reduction":
            self.reduction,
        "input_scaling_init":
            self.input_scaling_init,
        "input_scaling_type":
            self.input_scaling_type,
        "input_scaling_monotonicity":
            self.input_scaling_monotonicity,
        "sparsity_factor":
            self.sparsity_factor,
        "kernel_initializer":
            keras.initializers.serialize(
                self.kernel_initializer, use_legacy_format=True),
    }
    config.update(super(CDF, self).get_config())
    return config


def create_kernel_initializer(kernel_initializer_id):
  """Returns a kernel Keras initializer object from its id.

  This function is used to convert the 'kernel_initializer' parameter in the
  constructor of `tfl.layers.CDF` into the corresponding initializer object.

  Args:
    kernel_initializer_id: See the documentation of the 'kernel_initializer'
      parameter in the constructor of `tfl.layers.CDF`.

  Returns:
    The Keras initializer object for the `tfl.layers.CDF` kernel variable.
  """
  if kernel_initializer_id in ["random_uniform", "RandomUniform"]:
    return keras.initializers.RandomUniform(0.0, 1.0)
  else:
    return keras.initializers.get(kernel_initializer_id)
