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
    kernel: Kronecker-Factored Lattice kernel of shape `(1, lattice_sizes, units
      * dims, num_terms)`.
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
    inputs = tf.stack(inputs, axis=-1)
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

  prod = tf.reduce_prod(dotprod, axis=-2)

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


def random_monotonic_initializer(shape,
                                 monotonicities,
                                 dtype=tf.float32,
                                 seed=None):
  """Returns a uniformly random sampled monotonic weight tensor.

  - The uniform random monotonic function will initilaize the lattice parameters
    uniformly at random and make it such that the parameters are monotonically
    increasing for each input.
  - The random parameters will be sampled from `[0, 1]`

  Args:
    shape: Shape of weights to initialize. Must be: `(1, lattice_sizes, units *
      dims, num_terms)`.
    monotonicities: None or list or tuple of length dims of elements of {0,1}
      which represents monotonicity constraints per dimension. 1 stands for
      increasing (non-decreasing in fact), 0 for no monotonicity constraints.
    dtype: dtype
    seed: A Python integer. Used to create a random seed for the distribution.

  Returns:
    Kronecker-Factored Lattice weights tensor of shape:
      `(1, lattice_sizes, units * dims, num_terms)`.
  """
  # Sample from the uniform distribution.
  weights = tf.random.uniform(shape, dtype=dtype, seed=seed)
  if utils.count_non_zeros(monotonicities) > 0:
    # To sort, we must first reshape and unstack our weights.
    dims = len(monotonicities)
    _, lattice_sizes, units_times_dims, num_terms = shape
    if units_times_dims % dims != 0:
      raise ValueError(
          "len(monotonicities) is {}, which does not evenly divide shape[2]"
          "len(monotonicities) should be equal to `dims`, and shape[2] "
          "should be equal to units * dims.".format(dims))
    units = units_times_dims // dims
    weights = tf.reshape(weights, [-1, lattice_sizes, units, dims, num_terms])
    # Now we can unstack each dimension.
    weights = tf.unstack(weights, axis=3)
    monotonic_weights = [
        tf.sort(weight) if monotonicity else weight
        for monotonicity, weight in zip(monotonicities, weights)
    ]
    # Restack, reshape, and return weights
    weights = tf.stack(monotonic_weights, axis=3)
    weights = tf.reshape(weights, shape)
  return weights


def scale_initializer(units, num_terms):
  """Initializes scale to alternate between 1 and -1 for each term.

  Args:
    units: Output dimension of the layer. Each of units scale will be
      initialized identically.
    num_terms: Number of independently trained submodels per unit, the outputs
      of which are averaged to get the final output.

  Returns:
    Kronecker-Factored Lattice scale of shape: `(units, num_terms)`.
  """
  signs = (np.arange(num_terms) % -2) * 2 + 1
  return np.tile(signs, [units, 1])


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


def finalize_constraints(weights, units, scale, monotonicities):
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

  Returns:
    Projected weights tensor of same shape as `weights`.
  """
  if utils.count_non_zeros(monotonicities) == 0:
    return weights

  # TODO: in the case of only one monotonic dimension, we only have to
  # constrain the non-monotonic dimensions to be positive.
  # There must be monotonicity constraints, so we need all positive weights.
  weights = tf.maximum(weights, 0)

  # Project monotonicity constraints.
  weights = _approximately_project_monotonicity(weights, units, scale,
                                                monotonicities)

  return weights


def verify_hyperparameters(lattice_sizes=None,
                           units=None,
                           num_terms=None,
                           input_shape=None,
                           monotonicities=None):
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


def assert_constraints(weights, units, scale, monotonicities, eps=1e-6):
  """Asserts that weights satisfy constraints.

  Args:
    weights: `KroneckerFactoredLattice` weights tensor of shape: `(1,
      lattice_sizes, units * dims, num_terms)`.
    units: Number of units per input dimension.
    scale: Scale variable of shape: `(units, num_terms)`.
    monotonicities: Monotonicity constraints.
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

  return asserts
