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
"""Layer which represents linear function. See class level comment.

This layer applies a linear transformation to the input tensor with an optional
bias term. It supports monotonicity, monotonic dominance and fixed-norm
constraints.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import linear_lib
import tensorflow as tf
from tensorflow import keras

LINEAR_LAYER_KERNEL_NAME = "linear_layer_kernel"
LINEAR_LAYER_BIAS_NAME = "linear_layer_bias"


class Linear(keras.layers.Layer):
  # pyformat: disable
  """Layer which represents linear function.

  Monotonicity can be specified for any input dimension in which case learned
  weight for that dimension is guaranteed to be either non negative for
  increasing or non positive for decreasing monotonicity.

  Monotonic dominance can be specified for any pair of dimensions referred to as
  *dominant* and *weak* dimensions such that  the effect (slope) in the
  direction of the *dominant* dimension to be greater than that of the *weak*
  dimension for any point. Both dominant and weak dimensions must be increasing.

  Weights can be constrained to have a fixed norm.

  Input shape:
  Rank-2 tensor with shape: (batch_size, num_input_dims)

  Output shape:
  Rank-2 tensor with shape: (batch_size, 1)

  Attributes:
    - All `__init__ `arguments.
    kernel: layer's kernel.
    bias: layer's bias. Only available if `use_bias == True`.

  Example:

  ```python
  layer = tfl.linear_layer.Linear(
      num_input_dims=8,
      # Monotonicity constraints can be defined per dimension or for all dims.
      monotonicities='increasing',
      use_bias=True,
      # You can force the L1 norm to be 1. Since this is a monotonic layer,
      # the coefficients will sum to 1, making this a "weighted average".
      normalization_order=1)
  ```
  """
  # pyformat: enable

  def __init__(self,
               num_input_dims,
               monotonicities=None,
               monotonic_dominances=None,
               use_bias=True,
               normalization_order=None,
               kernel_initializer="random_uniform",
               bias_initializer="random_uniform",
               kernel_regularizer=None,
               bias_regularizer=None,
               **kwargs):
    """initializes an instance of `Linear`.

    Args:
      num_input_dims: Number of input dimensions.
      monotonicities: None or list or tuple of length 'num_input_dims' of
        {'decreasing', 'none', 'increasing', -1, 0, 1} which specifies if the
        model output should be monotonic in corresponding feature, using
        'increasing' or 1 to indicate increasing monotonicity, 'decreasing' or
        -1 to indicate decreasing monotonicity and 'none' or 0 to indicate no
        monotonicity constraints..
        In case of decreasing monotonicity corresponding weight will be
        constrained to be non positive, in case of increasing non-negative.
        Instead of a list or tuple single value can be specified to indicate the
        monotonicity constraint across all dimensions.
      monotonic_dominances: None or list of two-element tuples. First element is
        the index of the dominant feature. Second element is the index of the
        weak feature.
      use_bias: Whether linear function has bias.
      normalization_order: If specified learned weights will be adjusted to have
        norm 1. Norm will be computed by: `tf.norm(tensor,
          ord=normalization_order)`.
      kernel_initializer: Any keras initializer to be applied to kernel.
      bias_initializer: Any keras initializer to be applied to bias. Only valid
        if `use_bias == True`.
      kernel_regularizer: None or single element or list of any Keras
        regularizer objects.
      bias_regularizer: None or single element or list of any Keras regularizer
        objects.
      **kwargs: Other args passed to `tf.keras.layers.Layer` initializer.

    Raises:
      ValueError: if monotonicity specified incorrectly.
    """
    super(Linear, self).__init__(**kwargs)

    self.num_input_dims = num_input_dims

    if isinstance(monotonicities, list) or isinstance(monotonicities, tuple):
      self.monotonicities = list(monotonicities)
    elif monotonicities is not None:
      self.monotonicities = [monotonicities] * self.num_input_dims
    else:
      self.monotonicities = [0] * self.num_input_dims
    self.monotonic_dominances = monotonic_dominances
    # Verify hyperparameters after converting monotonicities to list because
    # internally everything expects monotonicites to be list or tuple rather
    # than single element.
    linear_lib.verify_hyperparameters(
        num_input_dims=self.num_input_dims, monotonicities=self.monotonicities)

    self.use_bias = use_bias
    self.normalization_order = normalization_order
    self.kernel_initializer = keras.initializers.get(kernel_initializer)
    if use_bias:
      self.bias_initializer = keras.initializers.get(bias_initializer)

    self.kernel_regularizer = []
    if kernel_regularizer:
      if callable(kernel_regularizer):
        kernel_regularizer = [kernel_regularizer]
      for reg in kernel_regularizer:
        self.kernel_regularizer.append(keras.regularizers.get(reg))
    self.bias_regularizer = []
    if bias_regularizer:
      if callable(bias_regularizer):
        bias_regularizer = [bias_regularizer]
      for reg in bias_regularizer:
        self.bias_regularizer.append(keras.regularizers.get(reg))

    self.input_spec = keras.layers.InputSpec(
        dtype=self.dtype, shape=(None, num_input_dims))

  def build(self, input_shape):
    """Standard Keras build() method.

    Args:
      input_shape: Must be: (batch_size, num_input_dims)

    Raises:
      ValueError: If shape is not (batch_size, num_input_dims).
    """
    if len(input_shape) != 2 or input_shape[1] != self.num_input_dims:
      raise ValueError("'input_shape' must be of rank two and number of "
                       "elements of second dimension must be equal to "
                       "'num_input_dims'. 'input_shape': " + str(input_shape) +
                       "'num_input_dims': " + str(self.num_input_dims))

    if (any(self.monotonicities) or self.monotonic_dominances or
        self.normalization_order):
      constraints = LinearConstraints(
          monotonicities=self.monotonicities,
          monotonic_dominances=self.monotonic_dominances,
          normalization_order=self.normalization_order)
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

    self.kernel = self.add_weight(
        LINEAR_LAYER_KERNEL_NAME,
        # 1 column matrix rather than verctor for matrix multiplication.
        shape=[self.num_input_dims, 1],
        initializer=self.kernel_initializer,
        regularizer=kernel_reg,
        constraint=constraints,
        dtype=self.dtype)

    if self.use_bias:
      if not self.bias_regularizer:
        bias_reg = None
      elif len(self.bias_regularizer) == 1:
        bias_reg = self.bias_regularizer[0]
      else:
        bias_reg = lambda x: tf.add_n([r(x) for r in self.bias_regularizer])
      self.bias = self.add_weight(
          LINEAR_LAYER_BIAS_NAME,
          shape=[],
          initializer=self.bias_initializer,
          regularizer=bias_reg,
          constraint=None,
          dtype=self.dtype)

    super(Linear, self).build(input_shape)

  def call(self, inputs):
    """Standard Keras call() method."""
    result = tf.matmul(inputs, self.kernel)
    if self.use_bias:
      result += self.bias
    return result

  def compute_output_shape(self, input_shape):
    """Standard Keras compute_output_shape() method."""
    del input_shape
    return [None, 1]

  def get_config(self):
    """Standard Keras get_config() method."""
    config = {
        "num_input_dims": self.num_input_dims,
        "monotonicities": self.monotonicities,
        "use_bias": self.use_bias,
        "normalization_order": self.normalization_order,
        "monotonic_dominances": self.monotonic_dominances,
        "kernel_initializer":
            keras.initializers.serialize(self.kernel_initializer),
        "kernel_regularizer": [
            keras.regularizers.serialize(r) for r in self.kernel_regularizer
        ],
    }  # pyformat: disable
    if self.use_bias:
      config["bias_initializer"] = keras.initializers.serialize(
          self.bias_initializer)
      config["bias_regularizer"] = [
          keras.regularizers.serialize(r) for r in self.bias_regularizer
      ]

    config.update(super(Linear, self).get_config())
    return config

  # Default eps is bigger than one for other layers because normalization is
  # prone to numerical errors.
  def assert_constraints(self, eps=1e-4):
    """Asserts that weights satisfy all constraints.

    In graph mode builds and returns list of assertion ops.
    In eager mode directly executes assetions.

    Args:
      eps: Allowed constraints violation.

    Returns:
      List of assertion ops in graph mode or immideately asserts in eager mode.
    """
    return linear_lib.assert_constraints(
        weights=self.kernel,
        monotonicities=linear_lib.canonicalize_monotonicities(
            self.monotonicities),
        monotonic_dominances=self.monotonic_dominances,
        normalization_order=self.normalization_order,
        eps=eps)


class LinearConstraints(keras.constraints.Constraint):
  # pyformat: disable
  """Applies monotonicity constraints and normalization to TFL Linear layer.

  Monotonicity is specified per input dimension in which case learned weight for
  those dimensions is guaranteed to be either non negative for increasing or non
  positive for decreasing monotonicity.

  Weights can be constrained to have norm 1.

  Attributes:
    - All `__init__` arguments.
  """
  # pyformat: enable

  def __init__(self, monotonicities, monotonic_dominances=None,
               normalization_order=None):
    """initializes an instance of `LinearConstraints`.

    Args:
      monotonicities: Same meaning as corresponding parameter of `Linear`.
      monotonic_dominances: Same meaning as corresponding parameter of `Linear`.
      normalization_order: Same meaning as corresponding parameter of `Linear`.
    """
    linear_lib.verify_hyperparameters(monotonicities=monotonicities,
                                      monotonic_dominances=monotonic_dominances)
    self.monotonicities = monotonicities
    self.monotonic_dominances = monotonic_dominances
    self.normalization_order = normalization_order

  def __call__(self, w):
    """Applies constraints to w.

    Args:
      w: Tensor which represents weights of TFL linear layer. Must have shape:
        `(len(self.monotonicities), 1)`.

    Raises:
      ValueError: if shape of `w` is not `(len(self.monotonicities), 1)`.

    Returns:
      Tensor `w` with monotonicity constraints and normalization applied to it.
    """
    return linear_lib.project(
        weights=w,
        monotonicities=linear_lib.canonicalize_monotonicities(
            self.monotonicities),
        monotonic_dominances=self.monotonic_dominances,
        normalization_order=self.normalization_order)

  def get_config(self):
    """Standard Keras get_config() method."""
    return {
        "monotonicities": self.monotonicities,
        "monotonic_dominances": self.monotonic_dominances,
        "normalization_order": self.normalization_order
    }  # pyformat: disable
