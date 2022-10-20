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
"""Algorithm implementations required for Kronecker-Factored Lattice layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import utils
import numpy as np
import tensorflow as tf


def custom_reduce_prod(t, axis):
  """tf.reduce_prod(t, axis) with faster custom gradient.

  Shows comparable speed on CPU, up to 2x speed up on GPU, and 7x on TPU.

  Args:
    t: The tensor to reduce.
    axis: The dimension to reduce.

  Returns:
    prod(t) and grad(prod(t))
  """

  @tf.custom_gradient
  def fn(t):
    # Can safely use the built in forward op.
    fwd = tf.reduce_prod(t, axis=axis)

    def grad_fn(dy):
      """Computes the gradient function.

      Args:
        dy: The gradient flowing into the output of this function.

      Returns:
        The gradient flowing out through the input of this function.
      """
      is_zero = tf.cast(tf.equal(t, 0), tf.float32)
      num_zeros = tf.reduce_sum(is_zero, axis=axis)

      # If the product contains no zero elements, then simply divide the
      # product by each element to determine the partial gradients.
      grad0 = tf.math.divide_no_nan(tf.expand_dims(fwd, axis=axis), t)

      # If the product contained one zero element, then compute the gradient
      # for that zero element. The gradients for other elements should be
      # zero.
      prod = tf.reduce_prod(t + is_zero, axis=axis)
      grad1 = tf.cast(tf.equal(num_zeros, 1), tf.float32) * prod
      grad1 = tf.expand_dims(grad1, axis=axis) * is_zero

      return tf.expand_dims(dy, axis=axis) * (grad0 + grad1)

    return fwd, grad_fn

  return fn(t)


def evaluate_with_hypercube_interpolation(inputs, scale, bias, kernel, units,
                                          num_terms, lattice_sizes,
                                          clip_inputs):
  """Evaluates a Kronecker-Factored Lattice using hypercube interpolation.

  Kronecker-Factored Lattice function is the product of the piece-wise linear
  interpolation weights for each dimension of the input.

  Args:
    inputs: Tensor representing points to apply lattice interpolation to. If
      units = 1, tensor should be of shape: `(batch_size, ..., dims)` or list of
        `dims` tensors of same shape `(batch_size, ..., 1)`. If units > 1,
        tensor
      should be of shape: `(batch_size, ..., units, dims)` or list of `dims`
        tensors of same shape `(batch_size, ..., units, 1)`. A typical shape is
        `(batch_size, dims)`.
    scale: Kronecker-Factored Lattice scale of shape `(units, num_terms)`.
    bias: Kronecker-Factored Lattice bias of shape `(units)`.
    kernel: Kronecker-Factored Lattice kernel of shape
      `(1, lattice_sizes, units * dims, num_terms)`.
    units: Output dimension of the Kronecker-Factored Lattice.
    num_terms: Number of independently trained submodels per unit, the outputs
      of which are averaged to get the final output.
    lattice_sizes: Number of vertices per dimension.
    clip_inputs: If inputs should be clipped to the input range of the
      Kronecker-Factored Lattice.

  Returns:
    Tensor of shape: `(batch_size, ..., units)`.
  """
  # Convert list of tensors to single tensor object.
  if isinstance(inputs, list):
    inputs = tf.concat(inputs, axis=-1)
  if clip_inputs:
    inputs = tf.clip_by_value(inputs, 0.0, lattice_sizes - 1.0)

  inputs_shape = inputs.get_shape().as_list()
  dims = inputs_shape[-1]
  # Compute total dimension size before units excluding batch to squeeze into
  # one axis.
  idx = -1 if units == 1 else -2
  rows = int(np.prod(inputs_shape[1:idx]))
  inputs = tf.reshape(inputs, [-1, rows, units * dims])

  # interpolation_weights.shape: (batch, rows, lattice_sizes, units * dims).
  # interpolation_weights[m,n,i,j] should be the interpolation weight of the
  # (m,n,j) input in the i'th vertex, i.e. 0 if dist(input[m,n,j], i) >= 1,
  # otherwise 1 - dist(input[m,n,j], i), where `dist(...)` denotes the Euclidean
  # distance between scalars.
  if lattice_sizes == 2:
    interpolation_weights = tf.stack([1 - inputs, inputs], axis=-2)
  else:
    vertices = tf.constant(
        list(range(lattice_sizes)),
        shape=(lattice_sizes, 1),
        dtype=inputs.dtype)
    interpolation_weights = vertices - tf.expand_dims(inputs, axis=-2)
    interpolation_weights = 1 - tf.minimum(tf.abs(interpolation_weights), 1)

  # dotprod.shape: (batch, rows, 1, units * dims * num_terms)
  dotprod = tf.nn.depthwise_conv2d(
      interpolation_weights, kernel, [1, 1, 1, 1], padding="VALID")
  dotprod = tf.reshape(dotprod, [-1, rows, units, dims, num_terms])

  prod = custom_reduce_prod(dotprod, axis=-2)

  results = scale * prod
  # Average across terms for each unit.
  results = tf.reduce_mean(results, axis=-1)
  results = results + bias

  # results.shape: (batch, rows, units)
  results_shape = [-1] + inputs_shape[1:-1]
  if units == 1:
    results_shape.append(1)
  results = tf.reshape(results, results_shape)
  return results


def default_init_params(output_min, output_max):
  """Returns default initialization bounds depending on layer output bounds.

  Args:
    output_min: None or minimum layer output.
    output_max: None or maximum layer output.
  """
  if output_min is None and output_max is None:
    return 0.5, 1.5
  else:
    return 0.0, 1.0


def kfl_random_monotonic_initializer(shape,
                                     scale,
                                     monotonicities,
                                     init_min=0.5,
                                     init_max=1.5,
                                     dtype=tf.float32,
                                     seed=None):
  """Returns a uniformly random sampled monotonic weight tensor.

  - The uniform random monotonic function will initilaize the lattice parameters
    uniformly at random and make it such that the parameters are monotonically
    increasing for each input.
  - The random parameters will be sampled from `[init_min, init_max]`

  Args:
    shape: Shape of weights to initialize. Must be: `(1, lattice_sizes, units *
      dims, num_terms)`.
    scale: Scale variable of shape: `(units, num_terms)`.
    monotonicities: None or list or tuple of length dims of elements of {0,1}
      which represents monotonicity constraints per dimension. 1 stands for
      increasing (non-decreasing in fact), 0 for no monotonicity constraints.
    init_min: The lower bound on the range of initialized weights.
    init_max: The upper bound on the range of initialized weights.
    dtype: dtype
    seed: A Python integer. Used to create a random seed for the distribution.

  Returns:
    Kronecker-Factored Lattice weights tensor of shape:
    `(1, lattice_sizes, units * dims, num_terms)`.
  """
  # Sample from the uniform distribution.
  weights = tf.random.uniform(
      shape, minval=init_min, maxval=init_max, dtype=dtype, seed=seed)
  if utils.count_non_zeros(monotonicities) > 0:
    # To sort, we must first reshape and unstack our weights.
    dims = len(monotonicities)
    _, lattice_sizes, units_times_dims, num_terms = shape
    if units_times_dims % dims != 0:
      raise ValueError(
          "len(monotonicities) is {}, which does not evenly divide shape[2]."
          "len(monotonicities) should be equal to `dims`, and shape[2] "
          "should be equal to units * dims.".format(dims))
    units = units_times_dims // dims
    weights = tf.reshape(weights, [-1, lattice_sizes, units, dims, num_terms])
    # Make all dimensions monotonically increasing with respect to the sign of
    # scale.
    direction = tf.expand_dims(tf.sign(scale), axis=1)
    # Now we can unstack each dimension.
    weights = tf.unstack(direction * weights, axis=3)
    monotonic_weights = [
        tf.sort(weight, axis=1) if monotonicity else weight
        for monotonicity, weight in zip(monotonicities, weights)
    ]
    # Restack, reshape, and return weights
    weights = tf.stack(monotonic_weights, axis=3)
    weights = tf.reshape(direction * weights, shape)
  return weights


def scale_initializer(units, num_terms, output_min, output_max):
  """Initializes scale depending on output_min and output_max.

  If both output_min and output_max are set, scale is initialized to half their
  difference, alternating signs for each term. If only output_min is set, scale
  is initialized to 1 for each term. If only output_max is set, scale is
  initialized to -1 for each term. Otherwise scale is initialized to alternate
  between 1 and -1 for each term.

  Args:
    units: Output dimension of the layer. Each unit's scale will be initialized
      identically.
    num_terms: Number of independently trained submodels per unit, the outputs
      of which are averaged to get the final output.
    output_min: None or minimum layer output.
    output_max: None or maximum layer output.

  Returns:
    Kronecker-Factored Lattice scale of shape: `(units, num_terms)`.
  """
  if output_min is not None and output_max is None:
    return np.ones([units, num_terms])
  if output_min is None and output_max is not None:
    return -np.ones([units, num_terms])
  # Both or neither bounds are set, so we alternate sign.
  signs = (np.arange(num_terms) % -2) * 2 + 1
  scale = np.tile(signs, [units, 1])
  if output_min is not None and output_max is not None:
    scale = scale * ((output_max - output_min) / 2.0)
  return scale


def bias_initializer(units, output_min, output_max, dtype=tf.float32):
  """Initializes bias depending on output_min and output_max.

  If both output_min and output_max are set, bias is initialized to their
  average. If only output_min is set, bias is initialized to output_min. If only
  output_max is set, bias is initialized to output_max. Otherwise bias is
  initialized to zeros.

  Args:
    units: Output dimension of the layer. Each of units bias will be initialized
      identically.
    output_min: None or minimum layer output.
    output_max: None or maximum layer output.
    dtype: dtype

  Returns:
    Kronecker-Factored Lattice bias of shape: `(units)`.
  """
  if output_min is not None and output_max is not None:
    return tf.constant(
        (output_min + output_max) / 2.0, shape=[units], dtype=dtype)
  elif output_min is not None:
    return tf.constant(output_min, shape=[units], dtype=dtype)
  elif output_max is not None:
    # In this case, weights will be nonnegative and scale will be nonpositive so
    # we add output_max to interpolation output to achieve proper bound.
    return tf.constant(output_max, shape=[units], dtype=dtype)
  else:
    return tf.zeros(shape=[units], dtype=dtype)


def _approximately_project_monotonicity(weights, units, scale, monotonicities):
  """Approximately projects to strictly meet monotonicity constraints.

  For more details, see _approximately_project_monotonicity in lattice_lib.py.

  Args:
    weights: Tensor with weights of shape `(1, lattice_sizes, units * dims,
      num_terms)`.
    units: Number of units per input dimension.
    scale: Scale variable of shape: `(units, num_terms)`.
    monotonicities: List or tuple of length dims of elements of {0,1} which
      represents monotonicity constraints per dimension. 1 stands for increasing
      (non-decreasing in fact), 0 for no monotonicity constraints.

  Returns:
    Tensor with projected weights matching shape of input weights.
  """
  # Recall that w.shape is (1, lattice_sizes, units * dims, num_terms).
  weights_shape = weights.get_shape().as_list()
  _, lattice_sizes, units_times_dims, num_terms = weights_shape
  assert units_times_dims % units == 0
  dims = units_times_dims // units
  weights = tf.reshape(weights, [-1, lattice_sizes, units, dims, num_terms])

  # Extract the sign of scale to determine the projection direction.
  direction = tf.expand_dims(tf.sign(scale), axis=1)

  # TODO: optimize for case where all dims are monotonic and we won't
  # need to unstack.
  # Unstack our weights such that we have the weight for each dimension. We
  # multiply by direction such that we always project the weights to be
  # increasing.
  weights = tf.unstack(direction * weights, axis=3)
  projected = []
  for weight, monotonicity in zip(weights, monotonicities):
    if monotonicity:
      # First we go forward to find the maximum projection.
      max_projection = tf.unstack(weight, axis=1)
      for i in range(1, len(max_projection)):
        max_projection[i] = tf.maximum(max_projection[i], max_projection[i - 1])
      # Find the halfway projection to find the minimum projection.
      half_projection = (weight + tf.stack(max_projection, axis=1)) / 2.0
      # Now we go backwards to find the minimum projection.
      min_projection = tf.unstack(half_projection, axis=1)
      for i in range(len(min_projection) - 2, -1, -1):
        min_projection[i] = tf.minimum(min_projection[i], min_projection[i + 1])
      # Restack our weight from the minimum projection.
      weight = tf.stack(min_projection, axis=1)
    # Add our projected weight to our running list.
    projected.append(weight)
  # Restack our final projected weights. We multiply by direction such that if
  # direction is negative we end up with decreasing weights.
  weights = direction * tf.stack(projected, axis=3)

  # Reshape projected weights into original shape and return them.
  weights = tf.reshape(weights, weights_shape)
  return weights


def _approximately_project_bounds(weights, units, output_min, output_max):
  """Approximately projects to strictly meet bound constraints.

  For more details, see _approximately_project_bounds in lattice_lib.py.

  Args:
    weights: Tensor with weights of shape `(1, lattice_sizes, units * dims,
      num_terms)`.
    units: Number of units per input dimension.
    output_min: None or minimum layer output.
    output_max: None or maximum layer output.

  Returns:
    Tensor with projected weights matching shape of input weights.
  """
  if output_min is None and output_max is None:
    return weights

  # We project by the dims'th root projection factor of the weights, ultimately
  # projecting each term into the range [-1,1], but only if both output_min and
  # output_max are specified. Otherwise, we restrict the weights to be
  # nonnegative and the interpolation will do a final shift to respect the
  # one-sided bound.
  if output_min is not None and output_max is not None:
    # Recall that w.shape is (1, lattice_sizes, units * dims, num_terms).
    weights_shape = weights.get_shape().as_list()
    _, lattice_sizes, units_times_dims, num_terms = weights_shape
    assert units_times_dims % units == 0
    dims = units_times_dims // units
    weights = tf.reshape(weights, [-1, lattice_sizes, units, dims, num_terms])
    max_keypoint_values = tf.reduce_max(tf.abs(weights), axis=1, keepdims=True)
    max_output_value = tf.reduce_prod(
        max_keypoint_values, axis=3, keepdims=True)
    full_projection_factor = tf.maximum(max_output_value, 1.0)
    individual_projection_factor = tf.pow(full_projection_factor, 1.0 / dims)
    weights = weights / individual_projection_factor
    # We must reshape to get our final projected weights.
    weights = tf.reshape(weights, weights_shape)
  else:
    weights = tf.maximum(weights, 0)

  return weights


# Note: this function must not depend on the result of projecting scale.
# Currently this function depends on the sign of scale, but the scale projection
# will not flip the sign of scale (only make it 0 in the worse case), which will
# not cause any issues.
def finalize_weight_constraints(weights, units, scale, monotonicities,
                                output_min, output_max):
  """Approximately projects weights to strictly satisfy all constraints.

  This projeciton guarantees that constraints are strictly met, but it is not
  an exact projection w.r.t. the L2 norm. The computational cost is
  `O(num_monotonic_dims * num_lattice_weights)`.

  See helper functions `_approximately_project_*` for details of the individual
  projection algorithms for each set of constraints.

  Args:
    weights: Kronecker-Factored Lattice weights tensor of shape: `(1,
      lattice_sizes, units * dims, num_terms)`.
    units: Number of units per input dimension.
    scale: Scale variable of shape: `(units, num_terms)`.
    monotonicities: List or tuple of length dims of elements of {0,1} which
      represents monotonicity constraints per dimension. 1 stands for increasing
      (non-decreasing in fact), 0 for no monotonicity constraints.
    output_min: None or minimum layer output.
    output_max: None or maximum layer output.

  Returns:
    Projected weights tensor of same shape as `weights`.
  """
  if utils.count_non_zeros(monotonicities) > 0:
    # TODO: in the case of only one monotonic dimension, we only have to
    # constrain the non-monotonic dimensions to be positive.
    # There must be monotonicity constraints, so we need all nonnegative
    # weights.
    weights = tf.maximum(weights, 0)
    weights = _approximately_project_monotonicity(
        weights=weights,
        units=units,
        scale=scale,
        monotonicities=monotonicities)

  if output_min is not None or output_max is not None:
    weights = _approximately_project_bounds(
        weights=weights,
        units=units,
        output_min=output_min,
        output_max=output_max)

  return weights


# Note: we cannot rely on the weights projection occuring always before or
# always after the scale projection, so this function must not result in a
# projection that would ultimately change the results of the weights projection.
# Currently the weights projection depends on the sign of scale, so this
# function does not change the sign (only makes scale 0 in the worst case),
# which will not cause any issues.
def finalize_scale_constraints(scale, output_min, output_max):
  """Clips scale to strictly satisfy all constraints.

  Args:
    scale: Scale variable of shape: `(units, num_terms)`.
    output_min: None or minimum layer output.
    output_max: None or maximum layer output.

  Returns:
    Clipped scale tensor of same shape as `scale`.
  """
  if output_min is not None and output_max is not None:
    bound = (output_max - output_min) / 2.0
    scale = tf.clip_by_value(scale, clip_value_min=-bound, clip_value_max=bound)
  elif output_min is not None:
    # In this case, we need scale to be nonnegative to properly shift by bias
    # and satisfy the one-sided max bound.
    scale = tf.maximum(scale, 0)
  elif output_max is not None:
    # In this case, we need scale to be nonpositive to properly mirror and shift
    # by bias and satisfy the one-sided min bound.
    scale = tf.minimum(scale, 0)
  return scale


def verify_hyperparameters(lattice_sizes=None,
                           units=None,
                           num_terms=None,
                           input_shape=None,
                           monotonicities=None,
                           output_min=None,
                           output_max=None):
  """Verifies that all given hyperparameters are consistent.

  This function does not inspect weights themselves. Only their shape. Use
  `assert_constraints()` to assert actual weights against constraints.

  See `tfl.layers.KroneckerFactoredLattice` class level comment for detailed
  description of arguments.

  Args:
    lattice_sizes: Lattice size to check against.
    units: Units hyperparameter of `KroneckerFactoredLattice` layer.
    num_terms: Number of independently trained submodels hyperparameter of
      `KroneckerFactoredLattice` layer.
    input_shape: Shape of layer input. Useful only if `units` and/or
      `monotonicities` is set.
    monotonicities: Monotonicities hyperparameter of `KroneckerFactoredLattice`
      layer. Useful only if `input_shape` is set.
    output_min: Minimum output of `KroneckerFactoredLattice` layer.
    output_max: Maximum output of `KroneckerFactoredLattice` layer.

  Raises:
    ValueError: If lattice_sizes < 2.
    ValueError: If units < 1.
    ValueError: If num_terms < 1.
    ValueError: If len(monotonicities) does not match number of inputs.
  """
  if lattice_sizes and lattice_sizes < 2:
    raise ValueError("Lattice size must be at least 2. Given: %s" %
                     lattice_sizes)

  if units and units < 1:
    raise ValueError("Units must be at least 1. Given: %s" % units)

  if num_terms and num_terms < 1:
    raise ValueError("Number of terms must be at least 1. Given: %s" %
                     num_terms)

  # input_shape: (batch, ..., units, dims)
  if input_shape:
    # It also raises errors if monotonicities is specified incorrectly.
    monotonicities = utils.canonicalize_monotonicities(
        monotonicities, allow_decreasing=False)
    # Extract shape to check units and dims to check monotonicity
    if isinstance(input_shape, list):
      dims = len(input_shape)
      # Check monotonicity.
      if monotonicities and len(monotonicities) != dims:
        raise ValueError("If input is provided as list of tensors, their number"
                         " must match monotonicities. 'input_list': %s, "
                         "'monotonicities': %s" % (input_shape, monotonicities))
      shape = input_shape[0]
    else:
      dims = input_shape.as_list()[-1]
      # Check monotonicity.
      if monotonicities and len(monotonicities) != dims:
        raise ValueError("Last dimension of input shape must have same number "
                         "of elements as 'monotonicities'. 'input shape': %s, "
                         "'monotonicities': %s" % (input_shape, monotonicities))
      shape = input_shape
    if units and units > 1 and (len(shape) < 3 or shape[-2] != units):
      raise ValueError("If 'units' > 1 then input shape of "
                       "KroneckerFactoredLattice layer must have rank at least "
                       "3 where the second from the last dimension is equal to "
                       "'units'. 'units': %s, 'input_shape: %s" %
                       (units, input_shape))

  if output_min is not None and output_max is not None:
    if output_min >= output_max:
      raise ValueError("'output_min' must be strictly less than 'output_max'. "
                       "'output_min': %f, 'output_max': %f" %
                       (output_min, output_max))


def _assert_monotonicity_constraints(weights, units, scale, monotonicities,
                                     eps):
  """Asserts that weights satisfy monotonicity constraints.

  Args:
    weights: `KroneckerFactoredLattice` weights tensor of shape: `(1,
      lattice_sizes, units * dims, num_terms)`.
    units: Number of units per input dimension.
    scale: Scale variable of shape: `(units, num_terms)`.
    monotonicities: Monotonicity constraints.
    eps: Allowed constraints violation.

  Returns:
    List of monotonicity assertion ops in graph mode or directly executes
    assertions in eager mode and returns a list of NoneType elements.
  """
  monotonicity_asserts = []

  # Recall that w.shape is (1, lattice_sizes, units * dims, num_terms).
  weights_shape = weights.get_shape().as_list()
  _, lattice_sizes, units_times_dims, num_terms = weights_shape
  assert units_times_dims % units == 0
  dims = units_times_dims // units
  weights = tf.reshape(weights, [-1, lattice_sizes, units, dims, num_terms])

  # Extract the sign of scale to determine the assertion direction.
  direction = tf.expand_dims(tf.sign(scale), axis=1)

  # Unstack our weights given our extracted sign.
  weights = tf.unstack(direction * weights, axis=3)
  for i, (weight, monotonicity) in enumerate(zip(weights, monotonicities)):
    if monotonicity:
      keypoints = tf.unstack(weight, axis=1)
      for j in range(1, len(keypoints)):
        diff = tf.reduce_min(keypoints[j] - keypoints[j - 1])
        monotonicity_asserts.append(
            tf.Assert(
                diff >= -eps,
                data=[
                    "Monotonicity violation", "Feature index:", i,
                    "Min monotonicity diff:", diff, "Upper layer number:", j,
                    "Epsilon:", eps, "Keypoints:", keypoints[j],
                    keypoints[j - 1]
                ]))

  return monotonicity_asserts


def _assert_bound_constraints(weights, units, scale, output_min, output_max,
                              eps):
  """Asserts that weights satisfy monotonicity constraints.

  Args:
    weights: `KroneckerFactoredLattice` weights tensor of shape: `(1,
      lattice_sizes, units * dims, num_terms)`.
    units: Number of units per input dimension.
    scale: Scale variable of shape: `(units, num_terms)`.
    output_min: None or minimum layer output.
    output_max: None or maximum layer output.
    eps: Allowed constraints violation.

  Returns:
    List of monotonicity assertion ops in graph mode or directly executes
    assertions in eager mode and returns a list of NoneType elements.
  """
  bound_asserts = []

  # Recall that w.shape is (1, lattice_sizes, units * dims, num_terms).
  weights_shape = weights.get_shape().as_list()
  _, lattice_sizes, units_times_dims, num_terms = weights_shape
  assert units_times_dims % units == 0
  dims = units_times_dims // units
  weights = tf.reshape(weights, [-1, lattice_sizes, units, dims, num_terms])

  # If both bounds are specified, we must also have that the maximum output be
  # between -1 and 1.
  if output_min is not None and output_max is not None:
    for term, term_weights in enumerate(tf.unstack(weights, axis=4)):
      max_keypoint_values = tf.reduce_max(
          tf.abs(term_weights), axis=1, keepdims=True)
      max_output_values = tf.reduce_prod(
          max_keypoint_values, axis=3, keepdims=True)
      for unit, unit_max_output_value in enumerate(
          tf.unstack(max_output_values, axis=2)):
        diff = tf.squeeze(1 - unit_max_output_value)
        bound_asserts.append(
            tf.Assert(
                diff >= -eps,
                data=[
                    "Bound violation (max output greater than 1)", "Diff", diff,
                    "Epsilon", eps, "Maximum output value",
                    unit_max_output_value, "Term index", term, "Unit", unit,
                    "Weights", weights
                ]))
  else:
    # If only one bound is specified, we must have that all of our weights are
    # nonnegative at this point. There can be no allowed epsilon error here
    # because of the effect of a negative value.
    total_negative_weights = tf.reduce_sum(tf.cast(weights < 0, tf.int32))
    bound_asserts.append(
        tf.Assert(
            total_negative_weights <= 0,
            data=[
                "Bound violation (negative weights)",
                "Number of negative weights", total_negative_weights, "Weights",
                weights
            ]))

  # If both bounds are specified, scale must be between
  # -(output_max-output_min)/2 and (output_max-output_min)/2. If only output_min
  # is specified, then scale must be nonnegative. If only output_max is
  # specified, then scale must be nonpositive.
  if output_min is not None and output_max is not None:
    bound = (output_max - output_min) / 2.0
    below_bound_scales = tf.reduce_sum(tf.cast(scale < -bound, tf.int32))
    above_bound_scale = tf.reduce_sum(tf.cast(scale > bound, tf.int32))
    bound_asserts.append(
        tf.Assert(
            below_bound_scales + above_bound_scale <= 0,
            data=[
                "Bound violation (scale out of bounds)", "Bound", bound,
                "Scale", scale
            ]))
  elif output_min is not None:
    negative_scales = tf.reduce_sum(tf.cast(scale < 0, tf.int32))
    bound_asserts.append(
        tf.Assert(
            negative_scales <= 0,
            data=[
                "Bound violation (only output_min specified with negative "
                "scale values)", "Scale", scale
            ]))
  elif output_max is not None:
    positive_scales = tf.reduce_sum(tf.cast(scale > 0, tf.int32))
    bound_asserts.append(
        tf.Assert(
            positive_scales <= 0,
            data=[
                "Bound violation (only output_max specified with positive "
                "scale values)", "Scale", scale
            ]))

  return bound_asserts


def assert_constraints(weights,
                       units,
                       scale,
                       monotonicities,
                       output_min,
                       output_max,
                       eps=1e-6):
  """Asserts that weights satisfy constraints.

  Args:
    weights: `KroneckerFactoredLattice` weights tensor of shape: `(1,
      lattice_sizes, units * dims, num_terms)`.
    units: Number of units per input dimension.
    scale: Scale variable of shape: `(units, num_terms)`.
    monotonicities: Monotonicity constraints.
    output_min: None or minimum layer output.
    output_max: None or maximum layer output.
    eps: Allowed constraints violation.

  Returns:
    List of assertion ops in graph mode or directly executes assertions in eager
    mode.
  """
  asserts = []

  if monotonicities:
    monotonicity_asserts = _assert_monotonicity_constraints(
        weights=weights,
        units=units,
        scale=scale,
        monotonicities=monotonicities,
        eps=eps)
    asserts.extend(monotonicity_asserts)

  if output_min is not None or output_max is not None:
    bound_asserts = _assert_bound_constraints(
        weights=weights,
        units=units,
        scale=scale,
        output_min=output_min,
        output_max=output_max,
        eps=eps)
    asserts.extend(bound_asserts)

  return asserts
