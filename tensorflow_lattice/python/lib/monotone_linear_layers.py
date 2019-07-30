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
"""Monotonic linear embedding layers library for TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_lattice.python.lib import regularizers
from tensorflow_lattice.python.lib import tools


def monotone_linear_layer(input_tensor,
                          input_dim,
                          output_dim,
                          is_monotone=None,
                          add_bias=True,
                          normalization_order=None,
                          init_weight_mean=2.0,
                          init_weight_stddev=0.5,
                          init_bias=None,
                          l1_reg=None,
                          l2_reg=None):
  """Creates a partially monotonic linear embedding layer.

  Returns an output of partially monotonic linear embedding layer, weights in
  the linear embedding layer, projection ops and regularizers.

    output = input * weight' + bias

  and the kth row is constrained to be non-negative, if is_monotone[k] == True.
  weight is initialized to entrywise Normal random variable (init_weight_mean,
  init_weight_stdev). If init_b is not provided, then the initial bias is
  initialized to -1/2 * init_weight_mean * input_dim. This offset term is used
  to make the initial mean to 0, assuming each input tensor is from the uniform
  distribution [0, 1]:
    E[output] = E[input * weight' + bias] = E[input] * E[weight] + bias
      = 1/2 * init_weight_mean * input_dim + bias
      = 0.

  Args:
    input_tensor: [batch_size, input_dim] tensor.
    input_dim: (int) input dimension.
    output_dim: (int) output dimension.
    is_monotone:  A list of input_dim booleans, a single boolean, or None. If
      None or False, linear layer will not have monotonicity constraints. If
      True, all of inputs are set to be monotonic. In the case of boolean list,
      input_tensor[:, k] is set to be monotonic if is_monotone[k] == True.
    add_bias: (bool) If a bias term should be added.
    normalization_order: If specified, the returned projection will normalize
      the weight vector across each output dimension to have norm 1. The norm
      order can be 1, 2 or np.inf. Norm is lower bounded by 1e-12.
    init_weight_mean: (float) A mean for Normal random weight initializer.
    init_weight_stddev: (float) A standard deviation for Normal random weight
      initializer.
    init_bias: (float) initial bias. If not provided, -1/2 * init_weight_mean *
      input_dim is used.
    l1_reg: (float) amount of l1 regularization.
    l2_reg: (float) amount of l2 regularization.

  Returns:
    A tuple of:
    * output tensor of shape [batch_size, output_dim]
    * weight tensor of shape [output_dim, input_dim]
    * None or projection ops, that must be applied at each
      step (or every so many steps) to project the model to a feasible space:
      used for bounding the outputs or for imposing monotonicity.
    * None or a regularization loss, if regularization is configured.

  Raises:
    ValueError: If is_monotone is not None, but its length != input_dim.
  """
  with tf.compat.v1.variable_scope('monotone_linear'):
    # We use [output_dim, input_dim] convention to use broadcasting in
    # projeciton.
    init_weights = tf.random.normal([output_dim, input_dim],
                                    mean=init_weight_mean,
                                    stddev=init_weight_stddev)
    if init_bias is None:
      init_biases = [-init_weight_mean * 0.5 * input_dim] * output_dim
    else:
      init_biases = [init_bias] * output_dim

    w = tf.compat.v1.get_variable(
        name='weight', initializer=init_weights, dtype=input_tensor.dtype)
    output_tensor = tf.matmul(input_tensor, w, transpose_b=True)
    if add_bias:
      b = tf.compat.v1.get_variable(
          name='bias', initializer=init_biases, dtype=input_tensor.dtype)
      output_tensor = output_tensor + b

    # Constructing a projection op.
    projection = None
    if is_monotone or normalization_order:
      with tf.name_scope('monotonic_projection'):
        diff = None
        if is_monotone:
          if isinstance(is_monotone, list):
            # is_monotone is given as a list. We should only apply positivity
            # constraints to a masked version of the weights.
            if input_dim != len(is_monotone):
              raise ValueError('input_dim (%d) != is_monotone length (%d)' %
                               (input_dim, len(is_monotone)))
            # Construct a multiplicative mask for monotonic dimension
            # selection.
            monotone_mask = tf.constant(
                [1.0 if monotone else 0.0 for monotone in is_monotone],
                dtype=w.dtype)
            # Since input_dim is the last dimension of the weight, we can use
            # broadcasting.
            masked_w = tf.multiply(w, monotone_mask)
          else:
            # is_monotone is set to True.
            masked_w = w

          projected_w = tf.maximum(masked_w, 0.0)
          diff = projected_w - masked_w

        if normalization_order:
          unnormalized_w = w if diff is None else w + diff
          normalized_w = unnormalized_w / tf.maximum(
              tf.norm(
                  unnormalized_w,
                  ord=normalization_order,
                  axis=1,
                  keepdims=True), 1e-12)
          diff = normalized_w - w

        projection = w.assign_add(diff)

    # Constructing a regularization op.
    regularizer = None
    if l1_reg is not None or l2_reg is not None:
      with tf.name_scope('linear_regularization'):
        regularizer = regularizers.linear_regularization(w, l1_reg, l2_reg)

    return (output_tensor, w, projection, regularizer)


def split_monotone_linear_layer(input_tensor,
                                input_dim,
                                monotonic_output_dim,
                                non_monotonic_output_dim,
                                is_monotone=None,
                                init_weight_mean=2.0,
                                init_weight_stddev=0.5,
                                init_bias=None,
                                l1_reg=None,
                                l2_reg=None):
  """Creates a split monotonic linear embedding layer.

  Returns outputs of partially monotonic linear embedding layers, weights in
  the linear embedding layers, projection ops and regularizers. This function
  splits monotonic and non-monotonic input based on is_monotone, and creates
  two separate linear embedding in the following form:

    monotonic_output = monotonic_input * monotonic_weight
                      + non-monotonic_input * nm_weight
                      + bias
    non_monotonic_output = non-monotonic_input * nn_weight + bias

  where monotonic_weight has to be non-negative. All elements in
  monotonic_output should be treated as a monotonic signal, otherwise there
  would be no monotonicity guarantee.
  Weights are initialized as in monotone_linear_layer.

  Args:
    input_tensor: [batch_size, input_dim] tensor.
    input_dim: (int) input dimension.
    monotonic_output_dim: (int) monotonic_output's dimension.
    non_monotonic_output_dim: (int) non_monotonic_output's dimension.
    is_monotone: A list of input_dim booleans, or None. If None, all inputs are
      set to be non-monotonic. In a boolean list case, the input_tensor[:, k]
      is set to be monotonic input if is_monotone[k] == True.
    init_weight_mean: (float) A mean for Normal random weight initializer.
    init_weight_stddev: (float) A standard deviation for Normal random weight
      initializer.
    init_bias: (float) initial bias. If not provided,
      -1/2 * init_weight_mean * input_dim is used.
    l1_reg: (float) amount of l1 regularization.
    l2_reg: (float) amount of l2 regularization.

  Returns:
    A tuple of:
    * monotonic_output tensor of shape [batch_size, monotonic_output_dim]
      or None if monotonic_outpu_dim == 0.
    * monotonic output's weight tensor of shape
      [input_dim, monotonic_output_dim] or None if monotonic_outpu_dim == 0.
    * non_monotonic_output tensor of shape
      [batch_size, non_monotonic_output_dim] or None if
      non_monotonic_output_dim == 0.
    * non_monotonic_output's weight tensor of shape
      [non_monotonic_input_dim, non_monotonic_output_dim] or None if
      non_monotonic_output_dim == 0.
    * None or projection ops, that must be applied at each
      step (or every so many steps) to project the model to a feasible space:
      used for bounding the outputs or for imposing monotonicity.
    * None or a regularization loss, if regularization is configured.

  Raises:
    ValueError: * If is_monotone is not None nor a list.
    * is_monotone is a list but its length != input_dim.
    * All values is_monotone is True, but non_monotonic_output_dim is not 0.
  """
  monotonic_output = None
  m_weight = None
  non_monotonic_output = None
  n_weight = None
  projections = []
  regularization = None
  if monotonic_output_dim > 0:
    with tf.compat.v1.variable_scope('split_monotone'):
      packed_results = monotone_linear_layer(
          input_tensor,
          input_dim=input_dim,
          output_dim=monotonic_output_dim,
          is_monotone=is_monotone,
          init_weight_mean=init_weight_mean,
          init_weight_stddev=init_weight_stddev,
          init_bias=init_bias,
          l1_reg=l1_reg,
          l2_reg=l2_reg)
      (monotonic_output, m_weight, projection, regularizer) = packed_results
      projections.append(projection)
      regularization = tools.add_if_not_none(regularization, regularizer)

  if non_monotonic_output_dim > 0:
    with tf.compat.v1.variable_scope('split_non_monotone'):
      # Construct non_monotone_input_tensor.
      if is_monotone is None:
        non_monotone_input_tensor = input_tensor
      else:
        if not isinstance(is_monotone, list):
          raise ValueError('is_monotone should be None or a list of booleans')
        if len(is_monotone) != input_dim:
          raise ValueError('input_dim (%d) != is_monotone length (%d)' %
                           (input_dim, len(is_monotone)))

        input_columns = tf.unstack(input_tensor, axis=1)
        non_monotone_columns = []
        for (monotone, input_column) in zip(is_monotone, input_columns):
          if not monotone:
            non_monotone_columns.append(input_column)
        if not non_monotone_columns:
          raise ValueError(
              'non_monotonic_output_dim is not None nor zero, but all inputs '
              'are required to be non-monotonic.')
        non_monotone_input_tensor = tf.stack(non_monotone_columns, axis=1)
      # Create a linear embedding.
      packed_results = monotone_linear_layer(
          non_monotone_input_tensor,
          input_dim=len(non_monotone_columns),
          output_dim=non_monotonic_output_dim,
          is_monotone=None,
          init_weight_mean=init_weight_mean,
          init_weight_stddev=init_weight_stddev,
          init_bias=init_bias,
          l1_reg=l1_reg,
          l2_reg=l2_reg)
      (non_monotonic_output, n_weight, _, regularizer) = packed_results
      regularization = tools.add_if_not_none(regularization, regularizer)

  return (monotonic_output, m_weight, non_monotonic_output, n_weight,
          projections, regularization)
