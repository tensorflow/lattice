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
"""Implementation of algorithms required for Lattice layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import itertools
import math

from . import utils
from absl import logging
import numpy as np
import six
import tensorflow as tf


def evaluate_with_simplex_interpolation(inputs, kernel, units, lattice_sizes,
                                        clip_inputs):
  """Evaluates a lattice using simplex interpolation.

  Within each cell of the lattice, we partition the hypercube into d! simplices,
  where each simplex has d+1 vertices. Each simplex (relative to the lower
  corner of the hypercube) includes the all-zeros vertex, a vertex with a
  single one, a vertex with two ones, ... and the all-ones vertex.
  For example, for a three-dimensional unit hypercube the 3! = 6 simplices are:

  [0,0,0], [0,0,1], [0,1,1], [1,1,1]
  [0,0,0], [0,0,1], [1,0,1], [1,1,1]
  [0,0,0], [0,1,0], [0,1,1], [1,1,1]
  [0,0,0], [0,1,0], [1,1,0], [1,1,1]
  [0,0,0], [1,0,0], [1,1,0], [1,1,1]
  [0,0,0], [1,0,0], [1,0,1], [1,1,1]

  A point x in the hypercube is contained in the simplex corresponding to the
  order of x's components. For example, x = [0.4,0.2,0.8] is contained in the
  simplex specified by [2,0,1] (second in the above list). The weight associated
  with each vertex in the simplex is the difference between the decreasingly
  sorted cooredinates of the input. For details, see e.g. "Dissection of the
  hypercube into simplices", D.G. Mead, Proceedings of the AMS, 76:2, Sep. 1979.

  Args:
    inputs: Tensor of shape: `(batch_size, ..., len(lattice_sizes))` or list of
      `len(lattice_sizes)` tensors of same shape `(batch_size, ..., 1)` which
      represents points to apply lattice interpolation to. A typical shape is
      `(batch_size, len(lattice_sizes))`.
    kernel: Lattice kernel of shape (num_params_per_lattice, units).
    units: Output dimension of the lattice.
    lattice_sizes: List or tuple of integers which represents lattice sizes of
      layer for which interpolation is being computed.
    clip_inputs: Whether inputs should be clipped to the input range of the
      lattice.

  Returns:
    Tensor of shape: `(batch_size, ..., units)`.
  """
  if isinstance(inputs, list):
    inputs = tf.concat(inputs, axis=-1)

  if clip_inputs:
    inputs = _clip_onto_lattice_range(
        inputs=inputs, lattice_sizes=lattice_sizes)

  lattice_rank = len(lattice_sizes)
  input_dim = len(inputs.shape)
  all_size_2 = all(size == 2 for size in lattice_sizes)

  # Strides are the changes in the global index (index into the flattened
  # parameters) when moving across each dimension.
  # E.g. for 2x2x2, strides are [4, 2, 1].
  strides = tf.constant(
      np.cumprod([1] + lattice_sizes[::-1][:-1])[::-1], tf.int32)

  if not all_size_2:
    # Find offset (into flattened parameters) for the lower corner of the
    # hypercube where input lands in.
    lower_corner_coordinates = tf.cast(inputs, tf.int32)
    # Avoid the corner case of landing on the outermost edge.
    lower_corner_coordinates = tf.minimum(lower_corner_coordinates,
                                          np.array(lattice_sizes) - 2)

    # Multiplying coordinates by strides and summing up gives out the index into
    # the flattened parameter tensor.
    # Note: Alternative method using tf.tensordot + tf.expand_dims is slower.
    lower_corner_offset = tf.reduce_sum(
        lower_corner_coordinates * strides, axis=-1, keepdims=True)

    # Continue simplex interpolation with the residuals
    inputs = inputs - tf.cast(lower_corner_coordinates, inputs.dtype)

  # Get sorted values and indicies.
  # TODO: investigate if there is a way to avoid sorting twice.
  sorted_indices = tf.argsort(inputs, direction="DESCENDING")
  sorted_inputs = tf.sort(inputs, direction="DESCENDING")

  # Simplex interpolation weights are the deltas between residuals.
  no_padding_dims = [[0, 0]] * (input_dim - 1)
  sorted_inputs_padded_left = tf.pad(
      sorted_inputs, no_padding_dims + [[1, 0]], constant_values=1.)
  sorted_inputs_padded_right = tf.pad(
      sorted_inputs, no_padding_dims + [[0, 1]], constant_values=0.)
  weights = sorted_inputs_padded_left - sorted_inputs_padded_right

  # Calculate cumsum over the strides of sorted dimensions to get index of
  # simplex vertices into the flattened lattice parameters.
  sorted_strides = tf.gather(strides, sorted_indices)
  if all_size_2:
    # Lower corner offset is 0 for 2^d lattices.
    corner_offset_and_sorted_strides = tf.pad(sorted_strides,
                                              no_padding_dims + [[1, 0]])
  else:
    corner_offset_and_sorted_strides = tf.concat(
        [lower_corner_offset, sorted_strides], axis=-1)
  indices = tf.cumsum(corner_offset_and_sorted_strides, axis=-1)

  # Get parameters values of simplex indicies.
  if units == 1:
    gathered_params = tf.gather(tf.reshape(kernel, [-1]), indices)
  else:
    # We now have two tensors 'indices' and 'weights' of shape (batch, units).
    # The kernel is of shape (num_params_per_lattice, units).
    # In order to use tf.gather, we need to convert 'indices' so that they are
    # indices into the flattened parameter tensor.
    # Note: Alternative method that uses a transpose on the parameters instead
    # of a multiply on the indices is slower with typical batch sizes.
    unit_offset = tf.constant([[i] * (lattice_rank + 1) for i in range(units)])
    flat_indices = indices * units + unit_offset
    gathered_params = tf.gather(tf.reshape(kernel, [-1]), flat_indices)

  # Dot product with interpolation weights.
  # Note: Alternative method using tf.einsum is slightly slower on CPU.
  return tf.reduce_sum(
      tf.multiply(gathered_params, weights), axis=-1, keepdims=(units == 1))


def evaluate_with_hypercube_interpolation(inputs, kernel, units, lattice_sizes,
                                          clip_inputs):
  """Evaluates a lattice using hypercube interpolation.

  Lattice function is multi-linearly interpolated between the 2^d vertices of a
  hypercube. This interpolation method is typically slower than simplex
  interpolation, since each value is interpolated from 2^d hypercube corners,
  rather than d+1 simplex corners. For details, see e.g. "Dissection of the
  hypercube into simplices", D.G. Mead, Proceedings of the AMS, 76:2, Sep. 1979.

  Args:
    inputs: Tensor representing points to apply lattice interpolation to. If
      units = 1, tensor should be of shape: `(batch_size, ...,
        len(lattice_sizes))` or list of `len(lattice_sizes)` tensors of same
        shape `(batch_size, ..., 1)`.
      If units > 1, tensor should be of shape: `(batch_size, ..., units,
        len(lattice_sizes))` or list of `len(lattice_sizes)` tensors of same
        shape `(batch_size, ..., units, 1)`. A typical shape is `(batch_size,
        len(lattice_sizes))`.
    kernel: Lattice kernel of shape (num_params_per_lattice, units).
    units: Output dimension of the lattice.
    lattice_sizes: List or tuple of integers which represents lattice sizes of
      layer for which interpolation is being computed.
    clip_inputs: Whether inputs should be clipped to the input range of the
      lattice.

  Returns:
    Tensor of shape: `(batch_size, ..., units)`.
  """
  interpolation_weights = compute_interpolation_weights(
      inputs=inputs, lattice_sizes=lattice_sizes, clip_inputs=clip_inputs)

  if units == 1:
    # Weights shape: (batch-size, ..., prod(lattice_sizes))
    # Kernel shape:  (prod(lattice_sizes), 1)
    return tf.matmul(interpolation_weights, kernel)
  else:
    # Weights shape: (batch-size, ..., units, prod(lattice_sizes))
    # Kernel shape:  (prod(lattice_sizes), units)
    return tf.reduce_sum(interpolation_weights * tf.transpose(kernel), axis=-1)


# TODO: Rename and update usage.
def compute_interpolation_weights(inputs, lattice_sizes, clip_inputs=True):
  """Computes weights for hypercube lattice interpolation.

  Running time: `O(batch_size * prod(lattice_sizes))`

  If `clip_inputs == True`, inputs outside of the range defined by
  `lattice_sizes` will be clipped into the lattice input range. If not, the
  corresponding weights will linearly approach 0.0 with input moving away from
  the valid input range.

  Args:
    inputs: Tensor of shape: `(batch_size, ..., len(lattice_sizes))` or list of
      `len(lattice_sizes)` tensors of same shape `(batch_size, ..., 1)` which
      represents points to apply lattice interpolation to. A typical shape is
      `(batch_size, len(lattice_sizes))`.
    lattice_sizes: List or tuple of integers which represents lattice sizes of
      layer for which interpolation is being computed.
    clip_inputs: Whether inputs should be clipped to the input range of the
      lattice.

  Raises:
    ValueError: If last dimension of `inputs` does not match `lattice_sizes`.

  Returns:
    Interpolation weights tensor of shape:
    `(batch_size, ..., prod(lattice_sizes))`.
  """
  if isinstance(inputs, list):
    input_shape = [tensor.shape for tensor in inputs]
    input_dtype = inputs[0].dtype
  else:
    input_shape = inputs.shape
    input_dtype = inputs.dtype
  verify_hyperparameters(lattice_sizes=lattice_sizes, input_shape=input_shape)

  # Special case: 2^d lattice with input passed in as a single tensor
  if all(size == 2 for size in lattice_sizes) and not isinstance(inputs, list):
    w = tf.stack([(1.0 - inputs), inputs], axis=-1)
    if clip_inputs:
      w = tf.clip_by_value(w, clip_value_min=0, clip_value_max=1)
    one_d_interpolation_weights = tf.unstack(w, axis=-2)
    return batch_outer_operation(one_d_interpolation_weights, operation="auto")

  if clip_inputs:
    inputs = _clip_onto_lattice_range(
        inputs=inputs, lattice_sizes=lattice_sizes)

  # Create interpolation keypoints in advance in order to reuse them for all
  # dimensions of same size.
  dim_keypoints = {}
  for dim_size in set(lattice_sizes):
    dim_keypoints[dim_size] = tf.constant([i for i in range(dim_size)],
                                          dtype=input_dtype)

  # Bucketize in order to share interpolation ops across consequtive dims of
  # same size.
  bucketized_inputs = _bucketize_consequtive_equal_dims(
      inputs=inputs, lattice_sizes=lattice_sizes)

  one_d_interpolation_weights = []
  for tensor, bucket_size, dim_size in bucketized_inputs:
    if bucket_size > 1:
      # Within bucket all dims have same lattice sizes so instead of splitting
      # before interpolation we split after interpolation.
      # Expand dims in order to make interpolation through broadcasting work.
      tensor = tf.expand_dims(tensor, axis=-1)

    # Broadcasting subtraction op.
    distance = tf.abs(tensor - dim_keypoints[dim_size])
    # Following ops will do following:
    # 1) if distance >= 1.0 then set interpolation weight to 0.0.
    # 2) if distance < 1.0 then set interpolation weight to 1.0 - distance.
    weights = 1.0 - tf.minimum(distance, 1.0)

    if bucket_size == 1:
      one_d_interpolation_weights.append(weights)
    else:
      one_d_interpolation_weights.extend(tf.unstack(weights, axis=-2))

  return batch_outer_operation(one_d_interpolation_weights, operation="auto")


def batch_outer_operation(list_of_tensors, operation="auto"):
  """Computes outer operation of last dimensions of each of given tensors.

  Args:
    list_of_tensors: List of tensors of same shape `(batch_size, ..., k[i])`
      where everything expect `k_i` matches.
    operation: - binary TF operation which supports broadcasting to be applied.
      - string "auto" in order to apply tf.multiply for first several tensors
      and tf.matmul for remaining.

  Returns:
    Tensor of shape: `(batch_size, ..., mul_i(k[i]))`.
  """
  # Alternative implementation using tf.einsum creates fewer graph nodes.
  # This is slightly slower on CPU as of 2020/5, but the timing results might
  # change with different setup/platform/hardware.
  # Create a formula for outer product. e.g. '...a,...b,...c->...abc'
  # if operation == "auto":
  #   n = len(list_of_tensors)
  #   chars = string.ascii_lowercase[:n]
  #   eqn = ",".join(["..." + c for c in chars]) + "->..." + "".join(chars)
  #   result = tf.einsum(eqn, *list_of_tensors)
  #   result_shape = [-1] + [int(size) for size in result.shape[1:]]
  #   output_shape = result_shape[:-n] + [np.prod(result_shape[-n:])]
  #   return tf.reshape(result, shape=output_shape)

  if len(list_of_tensors) == 1:
    return list_of_tensors[0]

  # Dimensions of size '1' at position -1 of first tensor and -2 of second
  # tensor will result in outer operation due to broadcasting.
  result = tf.expand_dims(list_of_tensors[0], axis=-1)

  for i, tensor in enumerate(list_of_tensors[1:]):
    if operation == "auto":
      # Threshold 6 determined empirically for 2^d lattices.
      op = tf.multiply if i < 6 else tf.matmul
    else:
      op = operation

    result = op(result, tf.expand_dims(tensor, axis=-2))

    # For TF1 compatibility convert shape to integers allowing first dimension
    # to be undefined.
    #
    # If we want to support arbitrary number of undefined dimensions we must
    # compute new_shape using tf ops. It is undesireble because we want to
    # minimize graph size.
    shape = [-1] + [int(size) for size in result.shape[1:]]

    # Merge last 2 dimensions which we just multiplied.
    new_shape = shape[:-2] + [shape[-2] * shape[-1]]

    # Since we are doing reshape anyway append 1 to prepare 'result' for
    # following outer operation.
    if i < len(list_of_tensors) - 2:
      new_shape.append(1)

    result = tf.reshape(result, shape=new_shape)
  return result


def _clip_onto_lattice_range(inputs, lattice_sizes):
  """Clips inputs onto valid input range for given lattice_sizes.

  Args:
    inputs: `inputs` argument of `compute_interpolation_weights`.
    lattice_sizes: list or tuple of integers which represents lattice sizes to
      clip onto.

  Returns:
    Clipped `inputs`.
  """
  if not isinstance(inputs, list):
    upper_bounds = [dim_size - 1.0 for dim_size in lattice_sizes]
    return tf.clip_by_value(
        inputs,
        clip_value_min=tf.zeros(shape=len(lattice_sizes), dtype=inputs.dtype),
        clip_value_max=tf.constant(upper_bounds, dtype=inputs.dtype))
  else:
    # Share bound constant across dimensions of same size.
    dim_upper_bounds = {}
    for dim_size in set(lattice_sizes):
      dim_upper_bounds[dim_size] = tf.constant(
          dim_size - 1.0, dtype=inputs[0].dtype)
    dim_lower_bound = tf.zeros(shape=[], dtype=inputs[0].dtype)

    clipped_inputs = []
    for one_d_input, dim_size in zip(inputs, lattice_sizes):
      clipped_inputs.append(
          tf.clip_by_value(
              one_d_input,
              clip_value_min=dim_lower_bound,
              clip_value_max=dim_upper_bounds[dim_size]))
    return clipped_inputs


def _bucketize_consequtive_equal_dims(inputs, lattice_sizes):
  """Groups consequite dimensions of same size together.

  For example `lattice_sizes == [2, 2, 2, 5, 5, 2]` produce 3 buckets:
  - bucket of size 3 which corresponds to first group of dimensions of size 2.
  - bucket of size 2 which corresponds to group of dimensions of size 5.
  - bucket of size 1 which corresponds to last dimension of size 2.
  If `inputs` is a single tensor then it will be split accordig to buckets.

  If `inputs` is a list of tensor then all buckets will be of size 1 regardless
  of lattice sizes in order to avoid merging tensors. In this case function acts
  merely as a convenience helper to unify output format.

  Args:
    inputs: `inputs` argument of `compute_interpolation_weights`.
    lattice_sizes: list or tuple of integers which represents lattice sizes.

  Returns:
    Iterable of tuples: `(tensor, bucket_size, bucket_dim_size)` where
    `tensor.shape[-1] == bucket_size` and `bucket_dim_size` is a lattice size
    which corresponds to bucket.
  """
  if not isinstance(inputs, list):
    bucket_sizes = []
    bucket_dim_sizes = []
    current_size = 1
    for i in range(1, len(lattice_sizes)):
      if lattice_sizes[i] != lattice_sizes[i - 1]:
        bucket_sizes.append(current_size)
        bucket_dim_sizes.append(lattice_sizes[i - 1])
        current_size = 1
      else:
        current_size += 1
    bucket_sizes.append(current_size)
    bucket_dim_sizes.append(lattice_sizes[-1])
    inputs = tf.split(inputs, num_or_size_splits=bucket_sizes, axis=-1)
  else:
    # TODO: run benchmark and figure out whether it make sense to merge
    # indiviaul tensors here.
    bucket_sizes = [1] * len(lattice_sizes)
    bucket_dim_sizes = lattice_sizes
  return zip(inputs, bucket_sizes, bucket_dim_sizes)


def default_init_params(output_min, output_max):
  """Returns reasonable default parameters if not defined explicitly.

  Args:
    output_min: None or minimum layer output.
    output_max: None or maximum layer output.
  """
  if output_min is not None:
    init_min = output_min
  elif output_max is not None:
    init_min = min(0.0, output_max)
  else:
    init_min = 0.0

  if output_max is not None:
    init_max = output_max
  elif output_min is not None:
    init_max = max(1.0, output_min)
  else:
    init_max = 1.0

  # Return our min and max.
  return init_min, init_max


def linear_initializer(lattice_sizes,
                       output_min,
                       output_max,
                       monotonicities=None,
                       unimodalities=None,
                       units=1,
                       dtype=tf.float32):
  """Returns a lattice layer weight tensor that represents a linear function.

  - The linear function will have positive coefficients for monotonic dimensions
    and 0 otherwise. If all dimensions are unconstrained, all coefficients will
    be positive.
  - Linear coefficients are set such that the minimum/maximum output of the
    lattice matches the given output_min/output_max.
  - Each monotonic dimension contributes with same weight regardless of number
    of vertices per dimension.
  - No dimension can be both monotonic and unimodal.
  - Unimodal dimensions contribute with same weight as monotonic dimensions.
  - Unimodal dimensions linearly decrease for first `(dim_size + 1) // 2`
    vertices and then linearly increase for following vertices.

  Args:
    lattice_sizes: List or tuple of integers which represents lattice sizes.
    output_min: Minimum output of lattice layer after initialization.
    output_max: Maximum output of lattice layer after initialization.
    monotonicities: None or list or tuple of same length as lattice_sizes of {0,
      1} which represents monotonicity constraints per dimension. 1 stands for
      increasing (non-decreasing in fact), 0 for no monotonicity constraints.
    unimodalities: None or list or tuple of same length as lattice_sizes of {-1,
      0, 1} which represents unimodality constraints per dimension. 1 indicates
      that function first decreases then increases, -1 indicates that function
      first increases then decreases, 0 indicates no unimodality constraints.
    units: Output dimension of the layer. Each of units lattices will be
      initialized identically.
    dtype: dtype.

  Returns:
    Lattice weights tensor of shape: `(prod(lattice_sizes), units)`.
  """
  verify_hyperparameters(
      lattice_sizes=lattice_sizes,
      monotonicities=monotonicities,
      unimodalities=unimodalities)
  if monotonicities is None:
    monotonicities = [0] * len(lattice_sizes)
  if unimodalities is None:
    unimodalities = [0] * len(lattice_sizes)

  num_constraint_dims = utils.count_non_zeros(monotonicities, unimodalities)
  if num_constraint_dims == 0:
    monotonicities = [1] * len(lattice_sizes)
    num_constraint_dims = len(lattice_sizes)

  dim_range = float(output_max - output_min) / num_constraint_dims
  one_d_weights = []

  for monotonicity, unimodality, dim_size in zip(monotonicities, unimodalities,
                                                 lattice_sizes):
    if monotonicity != 0:
      one_d = _linspace(start=0.0, stop=dim_range, num=dim_size)
    elif unimodality != 0:
      decreasing = _linspace(start=dim_range, stop=0.0, num=(dim_size + 1) // 2)
      increasing = _linspace(start=0.0, stop=dim_range, num=(dim_size + 1) // 2)
      # For odd size dimensions we want just 1 extreme point. For even sized we
      # want 2.
      if unimodality == 1:
        one_d = decreasing + increasing[dim_size % 2:]
      else:
        one_d = increasing + decreasing[dim_size % 2:]
    else:
      one_d = [0.0] * dim_size
    # Insert batch dim of size 1 at the beginning for batch_outer_operation.
    one_d_weights.append(tf.constant(one_d, dtype=dtype, shape=[1, dim_size]))

  # Use same implementation of outer operation as interpolation logic in order
  # to guarantee same weights order.
  weights = batch_outer_operation(one_d_weights, operation=tf.add)
  weights = tf.reshape(weights + output_min, shape=[-1, 1])
  if units > 1:
    weights = tf.tile(weights, multiples=[1, units])
  return weights


def _linspace(start, stop, num):
  """Returns `num` uniformly spaced floats between `start` and `stop`."""
  if num == 1:
    return [start]
  return [start + (stop - start) * i / (num - 1.0) for i in range(num)]


def random_monotonic_initializer(lattice_sizes,
                                 output_min,
                                 output_max,
                                 units=1,
                                 dtype=tf.float32):
  """Returns a uniformly random sampled monotonic lattice layer weight tensor.

  - The uniform random monotonic function will initilaize the lattice parameters
    uniformly at random and make it such that the parameters are monotonically
    increasing for each input.
  - The random parameters will be sampled from `[output_min, output_max]`

  Args:
    lattice_sizes: List or tuple of integers which represents lattice sizes.
    output_min: Minimum output of lattice layer after initialization.
    output_max: Maximum output of lattice layer after initialization.
    units: Output dimension of the layer. Each of units lattices will be
      initialized identically.
    dtype: dtype.

  Returns:
    Lattice weights tensor of shape: `(prod(lattice_sizes), units)`.
  """
  # First we verify parameters
  verify_hyperparameters(lattice_sizes=lattice_sizes)

  dimension = len(lattice_sizes)
  # Pre-compute the bases of the global index for each dimension.
  index_bases = [1] * dimension
  for i in range(0, dimension - 1)[::-1]:
    index_bases[i] = index_bases[i + 1] * lattice_sizes[i + 1]
  total_lattice_size = np.prod(lattice_sizes)
  # Create parameter indices to later gather parameter values in the proper
  # ordering.
  lattice_parameter_indices = [0] * total_lattice_size

  # Starting from the all-0 vertex, expand new vertices by getting the vertices
  # that are children of the vertices expanded in the last iteration in terms of
  # monotonic dependencies. Create constant tensor representing order of init
  # mapping each index to its corresponding random parameter value.
  parameter_index = 1
  # Vertices expanded in the last iteration.
  last_vertices = [0]
  while last_vertices:
    new_vertices_set = set()
    for index in last_vertices:
      remaining_index = index
      # For each dimension, if the vertex is not at the end of that dimension,
      # we can create a child of the current vertex by increasing the value
      # of the vertex in that dimension by one.
      for i in range(dimension):
        index_base = index_bases[i]
        # The value of the vertex index in the i'th dimension
        index_dim = remaining_index // index_base
        if index_dim < lattice_sizes[i] - 1:
          new_index = index + index_base
          if new_index not in new_vertices_set:
            new_vertices_set.add(new_index)
        remaining_index = remaining_index % index_base
    # Randomly sort the vertices expanded in the current iteration. Note that
    # there can be no monotonic dependency between vertices expanded in the same
    # iteration because their sum of all dimensions are the same (we increase
    # them one-by-one in each iteration).
    new_vertices = list(new_vertices_set)
    np.random.shuffle(new_vertices)
    # Assign parameter values
    for vertex in new_vertices:
      lattice_parameter_indices[vertex] = parameter_index
      parameter_index += 1
    last_vertices = new_vertices

  # Convert lattice_parameter_indices into a tensor.
  lattice_parameter_indices = tf.constant(lattice_parameter_indices)
  # Uniformly generate the random parameter values.
  parameter_values = tf.random.uniform(
      shape=[total_lattice_size],
      minval=output_min,
      maxval=output_max,
      dtype=dtype)
  parameter_values = tf.sort(parameter_values)
  # Convert lattice_parameter_indices to weights tensor and tile if necessary.
  weights = tf.gather(parameter_values, lattice_parameter_indices)
  weights = tf.reshape(weights, shape=[-1, 1])
  if units > 1:
    weights = tf.tile(weights, multiples=[1, units])
  return weights


# TODO: Add final projection for unimodality constraints.
def _approximately_project_monotonicity(weights, lattice_sizes, monotonicities):
  """Approximately projects to strictly meet monotonicity constraints.

  Algorithm details:

  Definition:
  A[i] refer to i-th coordinate of vertex A.
  For 2 vertices A and B:
    "A <p B": if A[i] <= B[i] for all monotonic dimensions i. (aka dominated by
      Pareto)

  In order for lattice to be monotonic it is sufficient that either:
    1) for any vertex V: weight[V] >= weight[X] for any vertex X that: X <p V.
  or
    2) for any vertex V: weight[V] <= weight[X] for any vertex X that: V <p X.

  For example consider lattice:

  ```
  0---1---2---3
  |   |   |   |
  4---5---6---7
  |   |   |   |
  8---9---10--11
  ```

  For examle for vertex 6 it's sufficient that:

  weight[6] >= max(weight[4, 5, 8, 9, 10])
  Or:
  weight[6] <= min(weight[2, 3, 7])

  Given the above definition, we can use either of the following update rules to
  approximately project into the feasible space:
  max_proj[V] = max(weight[X]) for any X that: X <p V.
  min_proj[V] = min(weight[X]) for any X that: V <p X.

  It's clear though that these algorithms either only increase weights or only
  decrease weights. We know that true projection algorithm increases some
  weights and decreases others. To get closer to a true projection, we modify
  and use both update rules as follows:

  1) half_proj[V] = weight[V] + (max_proj[V] - weight[V]) / 2
     ... move half way up towards max_proj.
  2) min_max_proj[V] = min_proj[half_proj[V]]
     ... move remained way down towards min_proj.

  Differs from _project_partial_monotonicity in that this algorithm guarantees a
  global satisfying solution for all monotonicity constraints.

  Args:
    weights: Tensor with weights whose shape matches lattice_sizes.
    lattice_sizes: List or tuple of integers which represents lattice sizes.
      which correspond to weights.
    monotonicities: List or tuple of same length as lattice_sizes of {0, 1}
      which represents monotonicity constraints per dimension. 1 stands for
      increasing (non-decreasing in fact), 0 for no monotonicity constraints.

  Returns:
    Tensor with projected weights matching shape of input weights.
  """

  # To compute max_proj[V] for all V altogether compute cumulative maximum
  # along every monotonic dimension in arbitrary order.
  max_projection = weights
  for dim in range(len(lattice_sizes)):
    if monotonicities[dim] == 0:
      continue
    layers = tf.unstack(max_projection, axis=dim)
    for i in range(1, len(layers)):
      # Computing cummulative maximum.
      layers[i] = tf.maximum(layers[i], layers[i - 1])
    max_projection = tf.stack(layers, axis=dim)

  half_projection = (weights + max_projection) / 2.0

  min_projection = half_projection
  for dim in range(len(lattice_sizes)):
    if monotonicities[dim] == 0:
      continue
    layers = tf.unstack(min_projection, axis=dim)
    for i in range(len(layers) - 2, -1, -1):
      # Compute cumulitive minimum in reversed order compare to cumulative
      # maximum above.
      layers[i] = tf.minimum(layers[i], layers[i + 1])
    min_projection = tf.stack(layers, axis=dim)

  return min_projection


def _approximately_project_edgeworth(weights, lattice_sizes, units,
                                     edgeworth_trusts):
  """Approximately projects to strictly meet all edgeworth trust constraints.

  Note that this function will not introduce violations to any
  previously-satisfied monotonicity constraints.

  Algorithm details:

  For a constraint on main dimension i and conditional dimension j, consider
  some slice of weights that is fixed along all other dimensions, leaving a grid

  ```
  0---1---2---3
  |   |   |   |
  4---5---6---7
  |   |   |   |
  8---9---10--11
  ```

  You can think of all the other dimensions as other such grids stacked behind
  this one, e.g. weight[8] and the points behind it are all such points with
  index 0 in the i'th and j'th dimensions, and weight[6] and the points behind
  it are all such points with index 2 in the i'th dimension and index 1 in the
  j'th.

  To enforce this edgeworth trust constraint without messing up monotonicity or
  other trust constraints, the key idea is that we will always translate all
  points 'behind' a point on this grid together. This ensures that no other
  trust constraints will be violated, since all other weight differences
  constrained by trust constraints will occur 'behind' a single such point
  (no conditional feature can also be a main feature).

  With that in mind, we project to edgeworth trust on this grid while
  maintaining monotonicity by working up and right and always increasing the
  top-right point in each four-point square. Here, we would first find how much
  we need to increase weight[5] by to maintain edgeworth trust on {4,5,8,9}. To
  follow the principle above, we then consider all such squares 'behind'
  {4,5,8,9} and find the biggest such difference. weight[5] and all points
  behind will be increased by that amount, and then we continue until fixing the
  top-right grid, {2,3,6,7}.

  If the trust constraint is in the opposite direction, i.e. cond_direction =
  -1, repeat all of the above except that we start in the top-right {2,3,6,7}
  grid and always lower the bottom-left point (weight[6] to start) until we
  reach the bottom-left {4,5,8,9} grid.

  Differs from _project_partial_edgeworth in that this algorithm guarantees a
  global satisfying solution for all edgeworth trust constraints.

  Args:
    weights: Tensor with weights whose shape matches lattice_sizes
      plus units if units > 1.
    lattice_sizes: List or tuple of integers which represents lattice sizes.
      which correspond to weights.
    units: Output dimension of the lattice.
    edgeworth_trusts: None or iterable of three-element tuples. First element is
      the index of the main (monotonic) feature. Second element is the index of
      the conditional feature. Third element is the direction of trust: 1 if
        higher values of the conditional feature should increase trust in the
        main feature and -1 otherwise.

  Returns:
    Tensor with projected weights matching shape of input weights.
  """

  # Project onto trust constraints by cumulatively fixing violations.
  trust_projection = weights
  for main_dim, cond_dim, cond_direction in edgeworth_trusts or []:
    layers = _unstack_nd(trust_projection, [main_dim, cond_dim])
    # Unlike other trust projections, cannot simply reverse layers beforehand
    # based on cond_direction; asymmetry would break algorithm.
    dims = len(layers[0][0].shape)
    axis = (tf.constant(list(range(dims - 1)), dtype=tf.int32) if units > 1
            else None)
    if cond_direction > 0:
      for i in range(0, lattice_sizes[main_dim] - 1):
        for j in range(0, lattice_sizes[cond_dim] - 1):
          difference_in_slopes = ((layers[i + 1][j] - layers[i][j]) -
                                  (layers[i + 1][j + 1] - layers[i][j + 1]))
          # Move all weights by the value of the biggest violation to both
          # satisfy this constraint and not hurt others. See function comments
          # for more details.
          max_violation = tf.maximum(
              tf.reduce_max(difference_in_slopes, axis=axis), 0)
          layers[i + 1][j + 1] += max_violation
    else:
      for i in range(lattice_sizes[main_dim] - 2, -1, -1):
        for j in range(lattice_sizes[cond_dim] - 2, -1, -1):
          difference_in_slopes = ((layers[i + 1][j + 1] - layers[i][j + 1]) -
                                  (layers[i + 1][j] - layers[i][j]))
          max_violation = tf.maximum(
              tf.reduce_max(difference_in_slopes, axis=axis), 0)
          layers[i][j] -= max_violation
    trust_projection = _stack_nd(layers, [main_dim, cond_dim])

  return trust_projection


# TODO: It is likely that this algorithm will work for all trapezoid
# trust constraints without needing the reduce_max, as long as there are no
# edgeworth constraints. If true, consider using that approach when possible.
def _approximately_project_trapezoid(weights, lattice_sizes, units,
                                     trapezoid_trusts, edgeworth_trusts):
  """Approximately projects to strictly meet all trapezoid trust constraints.

  Note that this function will not introduce violations to any
  previously-satisfied monotonicity or edgeworth constraints.

  Algorithm details:

  For a constraint on main dimension i and conditional dimension j, consider
  some slice of weights that is fixed along all other dimensions, leaving a grid

  ```
  0---1---2---3
  |   |   |   |
  4---5---6---7
  |   |   |   |
  8---9---10--11
  ```

  You can think of all the other dimensions as other such grids stacked behind
  this one, e.g. weight[8] and the points behind it are all such points with
  index 0 in the i'th and j'th dimensions, and weight[6] and the points behind
  it are all such points with index 2 in the i'th dimension and index 1 in the
  j'th.

  We project to trapezoid trust on this grid by working up both edges of
  the lattice and only ever decreasing weights on the low main_feature side and
  increasing weights on the high main_feature side. In the above example, we
  would first consider the pair {8, 4} and update weight 4 to be min(8, 4),
  before then looking at {4, 0} and updating 0 to be min(4, 0). Similarly set
  weight 7 to be max(7, 11) and then weight 3 to max(3, 7). Flip the orders if
  cond_direction is -1: work down instead of up.

  Unlike in the edgeworth trust case, we do not necessarily look 'behind' the
  page and update all points behind a given grid point by the maximum violation
  at each step. It turns out that while this does have the nice property of
  maintaining almost all types of edgeworth constraints, for the same reason
  that the edgeworth algorithm does (co-movement of weights involved in other
  constraints), it can actually break other trapezoid constraints, namely those
  which share the same conditional feature.

  There is one exception, which is the matching edgeworth trust constraint. In
  this case, the trapezoid updates only touch one corner of each edgeworth
  constraint and so can violate them. The solution is to update by the max of
  all violations behind the page and all violations encountered below in the
  grid.

  If you separately update each grid by the violations in that grid, this update
  procedure turns out to respect all trapezoid constraints. The rationale is a
  bit more subtle than in the edgeworth case. The basic idea is that since each
  trapezoid and monotonicity constraint operates on two weights that are next to
  each other (i.e. differ only in the index of one dimension), we can create
  a 'square' of points in which one edge goes across the constraint we want to
  maintain and the perpendicular edges go across the constraint we are updating.

  For example, consider the 4 weights

  ```
  A -- B
  |    |
  C -- D
  ```

  A/B and C/D differ in the same one index (the constraint we hope to maintain)
  while A/C and B/D differ across the conditional index of the trapezoid
  constraint we are updating. Say we are focused on whether we maintain A'<=B'
  (A' is A after imposing trapezoid trust) and we are operating on the 'min main
  feature' side of the lattice so that any updates that occur will lower
  weights. If B'=B after trapezoid trust, things are easy because A'<=A by 'min
  main feature' and A<=B by the preexisting constraint. If not, and B'<B, we
  start with A'<=C' by trapezoid trust and C'<=C by 'min main feature'. By
  the preexisting constraints, C<=D, and by the trapezoid trust update procedure
  and the fact that B has changed, it must be that B'=D.

  Unfortunately, this algorithm will break edgeworth constraints.

  The solution we take is to update independently for each grid whenever we have
  only trapezoid constraints and to update with the max across all other
  dimensions (and potentially below, in the case of matching constraints)
  when there are both types of constraints, recognizing that in this second case
  we may not achieve guarantees for trapezoid constraints which share a
  conditional feature.

  Differs from _project_partial_trapezoid in that this algorithm guarantees a
  global satisfying solution for all trapezoid trust constraints.

  Args:
    weights: Tensor with weights whose shape matches lattice_sizes plus units
      if units > 1.
    lattice_sizes: List or tuple of integers which represents lattice sizes.
      which correspond to weights.
    units: Output dimension of the lattice.
    trapezoid_trusts: None or iterable of three-element tuples. First element is
      the index of the main (monotonic) feature. Second element is the index of
      the conditional feature. Third element is the direction of trust set to 1
      if higher values of the conditional feature should increase trust in the
      main feature and -1 otherwise.
    edgeworth_trusts: None or iterable of three-element tuples. First element is
      the index of the main (monotonic) feature. Second element is the index of
      the conditional feature. Third element is the direction of trust set to 1
      if higher values of the conditional feature should increase trust in the
      main feature and -1 otherwise.

  Returns:
    Tensor with projected weights matching shape of input weights.
  """

  any_edgeworth = bool(edgeworth_trusts)

  # Project onto trust constraints by cumulatively fixing violations.
  for main_dim, cond_dim, cond_direction in trapezoid_trusts or []:
    layers = _unstack_nd(weights, [main_dim, cond_dim])
    max_main_dim = lattice_sizes[main_dim] - 1
    same_edgeworth = (main_dim, cond_dim,
                      cond_direction) in set(edgeworth_trusts or [])
    if cond_direction < 0:
      layers = _reverse_second_list_dimension(layers)
    lhs_update, rhs_update = 0, 0
    for j in range(0, lattice_sizes[cond_dim] - 1):
      lhs_difference = layers[0][j + 1] - layers[0][j]
      lhs_update = _trapezoid_violation_update(lhs_difference, units,
                                               any_edgeworth, same_edgeworth,
                                               lhs_update)
      layers[0][j + 1] -= lhs_update
      rhs_difference = layers[max_main_dim][j] - layers[max_main_dim][j + 1]
      rhs_update = _trapezoid_violation_update(rhs_difference, units,
                                               any_edgeworth, same_edgeworth,
                                               rhs_update)
      layers[max_main_dim][j + 1] += rhs_update
    if cond_direction < 0:
      layers = _reverse_second_list_dimension(layers)
    weights = _stack_nd(layers, [main_dim, cond_dim])

  return weights


def _trapezoid_violation_update(differences, units, any_edgeworth,
                                same_edgeworth, prior_update):
  """Calculates update amount based on violations for trapezoid projection.

  Note that the shape of the returned tensor is different based on the value
  of the any_edgeworth boolean feature. A single-valued tensor is
  returned when it is true, representing the amount by which all relevant
  weights will be updated. A tensor matching the shape of differences is
  returned when it is false, representing the individual updates to be applied
  to each relevant weight.

  Args:
    differences: Tensor containing amounts by which constraints are satisfied or
      violated.
    units: Output dimension of the lattice.
    any_edgeworth: Boolean for whether any edgeworth trust constraints are set
      for this lattice layer.
    same_edgeworth: Boolean for whether there is a matching edgeworth constraint
      for the trapezoid constraint being updated.
    prior_update: Tensor containing previous trapezoid constraint update.

  Returns:
    Tensor either matching the shape of the input differences tensor or
    consisting of a single element.

  """
  dims = len(differences.shape) - 1
  axis = tf.constant(list(range(dims)), dtype=tf.int32) if units > 1 else None
  if any_edgeworth and same_edgeworth:
    return tf.maximum(tf.maximum(
        tf.reduce_max(differences, axis=axis), 0), prior_update)
  elif any_edgeworth:
    return tf.maximum(tf.reduce_max(differences, axis=axis), 0)
  else:
    return tf.maximum(differences, 0)


def _approximately_project_bounds(weights, units, output_min, output_max):
  """Approximately projects to strictly meet min/max constraints.

  Note that this function will not introduce violations to any
  previously-satisfied monotonicity or trust constraints.

  Algorithm details:

  The idea of the min/max projection is to evenly scale (squish) the weights
  to fit within the desired range. This ensures that the weight differences-of-
  differences encountered in the trust constraints will not be affected.

  For example, given min_weight < output_min < 0 < output_max < max_weight, we
  will translate all weights such that min_weight = 0, then scale the weights
  by the difference in ratios between max_weight - min_weight and output_max -
  output_min, and then translate back so that min_weight = output_min and
  max_weight = output_max.

  Args:
    weights: Tensor with weights whose shape matches `lattice_sizes` plus units
      if units > 1.
    units: Output dimension of the lattice.
    output_min: None or minimum possible output.
    output_max: None or maximum possible output.

  Returns:
    Tensor with projected weights matching shape of input weights.
  """

  # Project into [output_min, output_max] by translating and scaling output if
  # necessary.
  dims = len(weights.shape) - 1
  axis = tf.constant(list(range(dims)), dtype=tf.int32) if units > 1 else None
  final_projection = weights
  if output_max is None and output_min is not None:
    final_projection += tf.maximum(
        output_min - tf.reduce_min(final_projection, axis=axis), 0)
  elif output_max is not None and output_min is None:
    final_projection -= tf.maximum(
        tf.reduce_max(final_projection, axis=axis) - output_max, 0)
  elif output_max is not None and output_min is not None:
    max_violation = tf.maximum(
        tf.reduce_max(final_projection, axis=axis) - output_max, 0)
    min_violation = tf.maximum(
        output_min - tf.reduce_min(final_projection, axis=axis), 0)
    final_projection += (min_violation - output_min)
    final_projection *= ((output_max - output_min) /
                         ((output_max + max_violation) -
                          (output_min - min_violation)))
    final_projection += output_min
  return final_projection


def finalize_constraints(weights,
                         lattice_sizes,
                         monotonicities,
                         edgeworth_trusts=None,
                         trapezoid_trusts=None,
                         output_min=None,
                         output_max=None):
  """Approximately projects lattice weights to strictly satisfy all constraints.

  This projeciton guarantees that constraints are strictly met, but it is not
  an exact projection w.r.t. the L2 norm. The computationally cost is
  `O((num_monotonic_dims + num_trust_constraints) * num_lattice_weights)`.

  See helper functions `_approximately_project_*` for details of the individual
  projection algorithms for each set of constraints. They are designed to be
  applied sequentially: monotonicity, then edgeworth, trapezoid, and bounds if
  necessary. This is because the projection algorithms are guaranteed to not
  violate *previous* constraints, though they may lead to violations of *later*
  constraints.

  Args:
    weights: Lattice weights tensor of shape: `(prod(lattice_sizes), units)`.
    lattice_sizes: List or tuple of integers which represents lattice sizes.
      which correspond to weights.
    monotonicities: List or tuple of same length as lattice_sizes of {0, 1}
      which represents monotonicity constraints per dimension. 1 stands for
      increasing (non-decreasing in fact), 0 for no monotonicity constraints.
    edgeworth_trusts: None or iterable of three-element tuples. First element is
      the index of the main (monotonic) feature. Second element is the index of
      the conditional feature. Third element is the direction of trust set to 1
      if higher values of the conditional feature should increase trust in the
      main feature and -1 otherwise.
    trapezoid_trusts: None or iterable of three-element tuples. First element is
      the index of the main (monotonic) feature. Second element is the index of
      the conditional feature. Third element is the direction of trust set to 1
      if higher values of the conditional feature should increase trust in the
      main feature and -1 otherwise.
    output_min: None or minimum possible output.
    output_max: None or maximum possible output.

  Returns:
    Projected weights tensor of same shape as `weights`.
  """
  if utils.count_non_zeros(monotonicities) == 0:
    return weights
  units = weights.shape[1]
  if units > 1:
    lattice_sizes = lattice_sizes + [int(units)]
    if monotonicities:
      monotonicities = monotonicities + [0]

  weights = tf.reshape(weights, shape=lattice_sizes)

  weights = _approximately_project_monotonicity(weights, lattice_sizes,
                                                monotonicities)
  if edgeworth_trusts or trapezoid_trusts:
    weights = _approximately_project_edgeworth(weights, lattice_sizes, units,
                                               edgeworth_trusts)
    weights = _approximately_project_trapezoid(weights, lattice_sizes, units,
                                               trapezoid_trusts,
                                               edgeworth_trusts)
    # Simple capping, applied in a later step, adds less distortion than this
    # scaling projection; however, it could violate trust constraints.
    weights = _approximately_project_bounds(weights, units, output_min,
                                            output_max)
  return tf.reshape(weights, shape=[-1, units])


# TODO: approach used to implement regluarizers is likely to be more
# efficient than one used here. Especially on TPU. Investigate it.
def _project_partial_monotonicity(weights, lattice_sizes, monotonicities,
                                  unimodalities, dimension, constraint_group):
  """Applies exact monotonicity projection to a subset of a single dimension.

  Algorithm details:

  In order to project into k constrained dimensions we split all constraints
  into 2k sets in such way that within each sets all constraints are
  independent. These 2k sets are chosen in such way that for each constrained
  dimension we have 2 sets of constraints: even and odd constraints according to
  index of smallest vertex in constraint. We apply Dykstra's algorithm to these
  sets handling each individual constraint within each set independently.

  This function in particular, then, operates on one of these independent sets,
  as defined by a specific dimension and constraint group: 0 for the even
  constraints and 1 for the odd constraints.

  Note that in case of just 2 lattice vertices per dimension odd set for that
  dimension will be empty.

  * k constrained dimensions projection:
  If we know how to project into single constrained dimension then we can use
  Dykstra algorithm to project into union of all k constrained dimensions.

  * Single constrained dimension projection:
  For single dimension projection we have multiple independent 1-d sequences of
  constrained weights of same length.
  For example 2 x 6 lattice with monotonicity along 2-nd dimension:

  ```
  0--<--1--<--2--<--3--<--4--<--5
  |     |     |     |     |     |
  6--<--7--<--8--<--9--<--10-<--11
  ```

  we have 2 independent rows of constraints. It's clear that both rows can be
  projected independently.

  To project 1 row, we can again apply Dykstra's algorithm splitting all
  constraints into two sets: constraints with odd indices and constraints with
  even indices. For example for first row:
  - even constraints set: {0 < 1, 2 < 3, 4 < 5}
  - odd constraints set:  {1 < 2, 3 < 4}

  Within each set no constraints interact with each other so we can project
  every individual constraint independently.

  * Individual constraint projection:
  Constraint weight[0] <= weight[1]:
  - weight[0] = min(weight[0], (weight[0] + weight[1]) / 2)
  - weight[1] = max(weight[1], (weight[0] + weight[1]) / 2)

  Differs from _approximately_project_monotonicity in that this algorithm
  - Only operates on a single dimension.
  - Does not guarantee an satisfying solution to the full monotonicity
    constraint.
  - Exactly projects (in L2 terms) on the subset of constraints it does
    operate on.

  Args:
    weights: Tensor with weights of lattice layer, with shape lattice_sizes.
    lattice_sizes: List or tuple of integers which represents lattice sizes.
      which correspond to weights.
    monotonicities: None or list or tuple of same length as lattice_sizes of {0,
      1} which represents monotonicity constraints per dimension. 1 stands for
      increasing (non-decreasing in fact), 0 for no monotonicity constraints.
    unimodalities: None or list or tuple of same length as lattice_sizes of {-1,
      0, 1} which represents unimodality constraints per dimension. 1 indicates
      that function first decreases then increases, -1 indicates that function
      first increases then decreases, 0 indicates no unimodality constraints.
    dimension: Index of feature to which we are applying constraints.
    constraint_group: 0 or 1 as defined above, representing whether we are
      operating on 'even' or 'odd' constraints.

  Returns:
    Tensor with projected weights matching shape of input weights.

  Raises:
    ValueError: If provided dimension has no monotonicity or unimodality
      constraint associated with it.
  """

  if monotonicities[dimension] == 0 and unimodalities[dimension] == 0:
    raise ValueError(
        "Trying to project monotonicity and unimodality onto unconstrained "
        "dimension: %d." % dimension)

  layers = tf.unstack(weights, axis=dimension)
  for i in range(constraint_group, lattice_sizes[dimension] - 1, 2):
    # Project individual independent constraints.
    average = (layers[i] + layers[i + 1]) / 2.0

    if monotonicities[dimension] == 1:
      layers[i] = tf.minimum(layers[i], average)
      layers[i + 1] = tf.maximum(layers[i + 1], average)

    if unimodalities[dimension] != 0:
      is_first_part = (i < lattice_sizes[dimension] // 2)
      if ((unimodalities[dimension] == -1 and is_first_part) or
          (unimodalities[dimension] == 1 and not is_first_part)):
        layers[i] = tf.minimum(layers[i], average)
        layers[i + 1] = tf.maximum(layers[i + 1], average)
      else:
        layers[i] = tf.maximum(layers[i], average)
        layers[i + 1] = tf.minimum(layers[i + 1], average)

  return tf.stack(layers, axis=dimension)


def _project_partial_edgeworth(weights, lattice_sizes, edgeworth_trust,
                               constraint_group):
  """Applies exact edgeworth trust projection to a subset of one constraint.

  Algorithm details:

  For the Edgeworth trust projection, we follow a similar approach to the
  monotonicity projection by splitting up the constraints into independent sets.
  Here, each trust constraint touches every lattice vertex, but can be broken up
  into 4 independent sets of constraints, based on whether the constraint's
  smaller indices along the main and conditional dimensions are even or odd.
  That leaves us with 4t sets of constraints if we have t trust constraints,
  which we can sequentially project onto with the Dykstra's algorithm.

  This function applies to a single set of independent constraints within a
  single trust constraint. The constraint group can take the value (0,0), (0,1),
  (1,0), or (1,1) corresponding to even (0) or odd (1) for the main and
  conditional dimensions, respectively.

  * k trust constraints projection:
  If we know how to project into single trust constraint then we can use
  Dykstra algorithm to project into union of all k trust constraints.

  * Single trust constraint projection:
  Edgeworth constraints require the difference in weights across the main
  feature to be larger when the conditional feature is higher. We can think of
  this as separate constraints applied to each 'square' of weights {(i,j,...),
  (i+1,j,...), (i,j+1,...), (i+1,j+1,...), where i and j denote the index
  dimensions of the main and conditional features and the ellipses represent
  a fixed value of the other feature dimensions. It is immediately clear that
  we can apply the constraint at the same time for different values of the
  other dimensions. Considering then a fixed slice, and a grid

  ```
  0---1---2---3
  |   |   |   |
  4---5---6---7
  |   |   |   |
  8---9---10--11
  |   |   |   |
  12--13--14--15
  ```

  we get our four independent sets by considering non-overlapping squares of
  constraints. In particular, we define the sets by the combination of even &
  odd starting indices in each dimension. So if we start our indexing at the
  top-left, the even/even set would be the four squares {0,1,4,5}, {2,3,6,7},
  {8,9,12,13}, and {10,11,14,15}, the even/odd set would be {4,5,8,9} and
  {6,7,10,11} and so on.

  * Individual weight projection:
  Within each square the projection moves each of the four weights by the
  constraint violation / 4, if necessary, increasing the gap between high-trust
  weights across the main feature and decreasing the gap between low-trust
  weights across the main feature.

  Differs from _approximately_project_edgeworth in that this algorithm
  - Only operates on the constraints for a single (main_dim, cond_dim) pair.
  - Does not guarantee a satisfying solution to the full trust constraint.
  - Exactly projects (in L2 terms) on the subset of constraints it does
    operate on.

  Args:
    weights: Tensor with weights of lattice layer, with shape lattice_sizes.
    lattice_sizes: List or tuple of integers which represents lattice sizes.
      which correspond to weights.
    edgeworth_trust: Three-element tuple representing a single trust constraint.
      First element is the index of the main (monotonic) feature. Second element
      is the index of the conditional feature. Third element is the direction of
      trust set to 1 if higher values of the conditional feature increase trust
      and -1 otherwise.
    constraint_group: Two-element tuple of 0s and 1s as defined above,
      representing the combination of 'even' and 'odd' constraints we are
      projecting on.

  Returns:
    Tensor with projected weights matching shape of input weights.
  """

  main_dim, cond_dim, cond_direction = edgeworth_trust
  layers = _unstack_nd(weights, [main_dim, cond_dim])

  if cond_direction < 0:
    layers = _reverse_second_list_dimension(layers)
  for i in range(constraint_group[0], lattice_sizes[main_dim] - 1, 2):
    for j in range(constraint_group[1], lattice_sizes[cond_dim] - 1, 2):
      difference_in_slopes = ((layers[i + 1][j] - layers[i][j]) -
                              (layers[i + 1][j + 1] - layers[i][j + 1]))
      correction = tf.maximum(difference_in_slopes / 4, 0)
      layers[i][j] += correction
      layers[i][j + 1] -= correction
      layers[i + 1][j] -= correction
      layers[i + 1][j + 1] += correction
  if cond_direction < 0:
    layers = _reverse_second_list_dimension(layers)

  return _stack_nd(layers, [main_dim, cond_dim])


def _project_partial_trapezoid(weights, lattice_sizes, trapezoid_trust,
                               constraint_group):
  """Applies exact trapezoid trust projection to a subset of one constraint.

  Algorithm details:

  For the trapezoid trust projection, each trust constraint touches every
  lattice vertex, but can be broken up into 2 independent sets of constraints,
  based on whether the constraint's smaller index along the conditional
  dimension is even or odd. That leaves us with 2t sets of constraints if we
  have t trust constraints, which we can sequentially project onto with the
  Dykstra algorithm.

  This function applies to a single set of independent constraints within a
  single trust constraint. The constraint group can take the value 0 or 1,
  corresponding to even (0) or odd (1) for conditional dimension index.

  * k trust constraints projection:
  If we know how to project into single trust constraint then we can use
  Dykstra algorithm to project into union of all k trust constraints.

  * Single trust constraint projection:
  Trapezoid constraints require the range of possible model outputs across the
  main feature to be larger when the conditional feature demonstrates higher
  trust in the main feature. That is, they constrain the 'extreme' (minimum and
  maximum) weights in the main feature dimension but not any of the weights in
  the middle if the lattice size is larger than 2. We therefore have one set of
  constraints along the conditional dimension when the main feature is at its
  minimum and one when the main feature is at its maximum. For example, consider
  the grid

  ```
  0---1---2---3
  |   |   |   |
  4---5---6---7
  |   |   |   |
  8---9---10--11
  |   |   |   |
  12--13--14--15
  ```

  If the main feature is on the x-axis and the conditional feature is on the y-
  axis in this grid, our constraints operate on {0,4,8,12} and {3,7,11,15}. In
  fact, those constraints are simply monotonicity constraints in opposite
  directions. If the cond_direction = 1, we are monotonically decreasing between
  12 and 0 (0 < 4 < 8 < 12) and monotonically increasing between 15 and 3
  (3 > 7 > 11 > 15). Note that these imply that [0,3] is a superset of [4,7] and
  so on down to the smallest subset [12,15]. Our two independent sets of these
  constraints match those for monotonicity based on even and odd indices. For
  example, [8 < 12], [4 < 0], [11 > 15], and [3 > 7] can be projected onto at
  once, while [4 < 8] and [7 > 11] are in the other group. All constraint
  directions are flipped if cond_direction = -1.

  * Individual weight projection:
  For each pair of constraints, we project as in monotonicity: each weight moves
  halfway towards each other if the constraint is being violated, and stays the
  same otherwise.

  Differs from _approximately_project_trapezoid in that this algorithm
  - Only operates on the constraints for a single (main_dim, cond_dim) pair.
  - Does not guarantee a satisfying solution to the full trust constraint.
  - Exactly projects (in L2 terms) on the subset of constraints it does
    operate on.

  Args:
    weights: Tensor with weights of lattice layer, with shape lattice_sizes.
    lattice_sizes: List or tuple of integers which represents lattice sizes.
      which correspond to weights.
    trapezoid_trust: Three-element tuple representing a single trust constraint.
      First element is the index of the main (monotonic) feature. Second element
      is the index of the conditional feature. Third element is the direction of
      trust set to 1 if higher values of the conditional feature increase trust
      and -1 otherwise.
    constraint_group: 0 or 1 as defined above, representing whether we are
      acting on even or odd indices

  Returns:
    Tensor with projected weights matching shape of input weights.
  """

  main_dim, cond_dim, cond_direction = trapezoid_trust
  layers = _unstack_nd(weights, [main_dim, cond_dim])

  max_main_dim = lattice_sizes[main_dim] - 1
  if cond_direction < 0:
    layers = _reverse_second_list_dimension(layers)
  for j in range(constraint_group, lattice_sizes[cond_dim] - 1, 2):
    lhs_difference = layers[0][j + 1] - layers[0][j]
    lhs_correction = tf.maximum(lhs_difference / 2, 0)
    layers[0][j] += lhs_correction
    layers[0][j + 1] -= lhs_correction

    rhs_difference = layers[max_main_dim][j] - layers[max_main_dim][j + 1]
    rhs_correction = tf.maximum(rhs_difference / 2, 0)
    layers[max_main_dim][j] -= rhs_correction
    layers[max_main_dim][j + 1] += rhs_correction
  if cond_direction < 0:
    layers = _reverse_second_list_dimension(layers)

  return _stack_nd(layers, [main_dim, cond_dim])


def _project_partial_monotonic_dominance(weights, lattice_sizes,
                                         monotonic_dominance, constraint_group):
  r"""Applies exact monotonic dominance projection to given constraint group.

  Algorithm details:

  For the monotonic dominance projection, we follow a similar approach to the
  monotonicity projection by splitting up the constraints into independent sets.
  Here, each dominance constraint can be broken up into 8 independent sets of
  constraints, based on (1) whether the constraint's smaller indices along the
  dominant and weak dimensions are even or odd and (2) two triplets of vertices
  to consider for each square in the grid shown below.

  That leaves us with 8k sets of constraints if we have k dominance constraints,
  which we can sequentially project onto with the Dykstra algorithm.

  This function applies to a single set of independent constraints within a
  single dominance constraint group. The constraint group can take the value
  {0,1} x {0,1} x {0,1}. Even (0) or odd (1) of the first two elements
  correspond to the dominant and weak features and the third element determines
  which of the two triplets within a square to consider.

  * k monotonic dominance constraints projection:
  If we know how to project into single monotonic dominance constraint then we
  can use Dykstra algorithm to project into union of all k dominance
  constraints.

  * Single monotonic dominance constraint projection
  Monotonic dominance constraints require the effect (slope) in the direction
  of the dominant dimension to be greater than that of the weak dimension for
  any point in the lattice. We can think of this as separate constraints applied
  to each 'triangle' of weights represented as either {(i,j,...), (i+1,j,...),
  (i+1,j+1,...)} or {(i,j,...), (i,j+1,...), (i+1,j+1,...)} where i and j denote
  the index dimensions of the dominant and weak features and the ellipses
  represent a fixed value of the other feature dimensions. Considering then a
  fixed slice, and a grid

  ```
  0---1---2---3
  | \ | \ | \ |
  4---5---6---7
  | \ | \ | \ |
  8---9---10--11
  | \ | \ | \ |
  12--13--14--15
  ```

  where the dominant feature is on the x-axis and the weak feature is on the
  y-axis, we get our 8 independent sets of non-overlapping triangular triplets
  of vertices. For example, one set consists of {(0,1,4), (8,9,12), (2,3,6),
  (10,11,14)}.

  * Individual weight projection
  Within each triangular triplet, the projection moves the weight of the right
  angled vertex, either top-right or bottom-left, by 2 * violation / 3 and the
  other two vertices by violation / 3 to satisfy the constraint while minimizing
  the L2 distance from the initial point.

  Args:
    weights: tensor with weights of lattice layer, with shape lattice_sizes.
    lattice_sizes: list or tuple of integers which represents lattice sizes
      which correspond to weights.
    monotonic_dominance: two-element tuple representing a single monotonic
      dominance constraint. First element is the index of the dominant feature.
      Second element is the index of the weak feature.
    constraint_group: three-element tuple as defined above, representing 'even'
      or 'odd' indices and which of the two triangles we are acting on.

  Returns:
    Tensor with projected weights matching shape of input weights.
  """

  dominant_dim, weak_dim = monotonic_dominance
  layers = _unstack_nd(weights, [dominant_dim, weak_dim])
  for i in range(constraint_group[0], lattice_sizes[dominant_dim] - 1, 2):
    for j in range(constraint_group[1], lattice_sizes[weak_dim] - 1, 2):
      midpoint = (layers[i][j] + layers[i + 1][j + 1]) / 2
      if constraint_group[2] == 1:
        difference = midpoint - layers[i + 1][j]
        correction = tf.maximum(difference / 3, 0)
        layers[i + 1][j] += 2 * correction
      else:
        difference = midpoint - layers[i][j + 1]
        correction = tf.minimum(difference / 3, 0)
        layers[i][j + 1] += 2 * correction
      layers[i][j] -= correction
      layers[i + 1][j + 1] -= correction

  return _stack_nd(layers, [dominant_dim, weak_dim])


def _project_partial_range_dominance(weights, lattice_sizes, range_dominance,
                                     constraint_group):
  r"""Applies exact range dominance projection to given constraint group.

  Algorithm details:

  For the range dominance projection, each range dominance constraint can be
  broken up into M x N independent constraints where M and N are the lattice
  sizes of the dominant and weak dimensions. In other words, there are vertex
  number of constraints to project onto. This leaves us with M x N x k
  constraints if we have k range dominance constraints, which we can
  sequentially project onto with the Dykstra algorithm.

  This function applies to a single independent constraint within a single range
  dominance constraint as specificed by the given constraint group.

  * k range dominance constraints projection:
  If we know how to project into single range dominance constraint then we can
  use Dykstra algorithm to project into union of all k dominance constraints.

  * Single range dominance constraint projection:
  Range dominance constraints require the range of possible outputs to be
  greater if one varies the dominant dimension than if one varies the weak
  dimension for any point. Considering then a fixed slice, and a grid

  ```
  0---1---2---3
  |   |   |   |
  4---5---6---7
  |   |   |   |
  8---9---10--11
  |   |   |   |
  12--13--14--15
  ```

  where the dominant dimension is on the x-axis and the weak dimension is on the
  y-axis, we get, for each vertex defined by the x and y coordinates, a
  constraint where the range for direction in x-axis is required to be greater
  than the range for direction in y-axis. For example, vertex 1 requires its
  dominant range defined by vertices 0 and 3 to be greater than its weak range
  defined by vertices 1 and 13.

  * Individual weight projection:
  The projection moves the weights of all four vertices defining the dominant
  and weak ranges by the constraint violation / 4 such that the dominant range
  grows and the weak range shrinks. The only exception is the four corner
  vertices, i.e. vertices 0, 3, 12, 15. In this case, there are three
  participating vertices and since one of the vertices is shared by the two
  conflicting ranges, we only move the weights of the other two vertices. This
  means, for vertex 0, we move the weight of vertex 3 up halfway, the weight of
  vertex 12 down halfway and leave the weight of vertex 0 unchanged.

  Args:
    weights: tensor with weights of lattice layer, with shape lattice_sizes.
    lattice_sizes: list or tuple of integers which represents lattice sizes
      which correspond to weights.
    range_dominance: two-element tuple representing a single range dominance
      constraint. First element is the index of the dominant feature. Second
      element is the index of the weak feature.
    constraint_group: two-element tuple as defined above, representing the
      location of a vertex we are acting on.

  Returns:
    Tensor with projected weights matching shape of input weights.
  """

  dom_dim, weak_dim = range_dominance
  dom_dim_size = lattice_sizes[dom_dim]
  weak_dim_size = lattice_sizes[weak_dim]
  i, j = constraint_group
  layers = _unstack_nd(weights, [dom_dim, weak_dim])
  difference = ((layers[i][weak_dim_size - 1] - layers[i][0]) -
                (layers[dom_dim_size - 1][j] - layers[0][j]))
  if (i == 0 or i == dom_dim_size - 1) and (j == 0 or j == weak_dim_size - 1):
    correction = tf.maximum(difference / 2, 0)
    if i == 0:
      layers[dom_dim_size - 1][j] += correction
    else:
      layers[0][j] -= correction
    if j == 0:
      layers[i][weak_dim_size - 1] -= correction
    else:
      layers[i][0] += correction
  else:
    correction = tf.maximum(difference / 4, 0)
    layers[i][weak_dim_size - 1] -= correction
    layers[i][0] += correction
    layers[dom_dim_size - 1][j] += correction
    layers[0][j] -= correction

  return _stack_nd(layers, [dom_dim, weak_dim])


def _project_partial_joint_monotonicity(weights, lattice_sizes,
                                        joint_monotonicity, constraint_group):
  """Applies exact joint monotonicity projection to given constraint group.

  Algorithm details:

  For the joint monotonicity projection, we follow a similar approach to the
  per-dimension monotonicity projection by splitting up the constraints into
  independent sets. Here, each joint monotonicity constraint can be broken up
  into 8 independent sets of constraints, based on (1) whether the constraint's
  smaller indices along the two given dimensions are even or odd and (2) two
  triplets of vertices to consider for each square in the grid shown below.

  That leaves us with 8k sets of constraints if we have k joint monotonocity
  constraints, which we can sequentially project onto with the Dykstra
  algorithm.

  This function applies to a single set of independent constraints within a
  single joint monotonicity constraint. The constraint group can take the value
  {0,1} x {0,1} x {0,1}. Even (0) or odd (1) of the first two elements
  correspond to the two features that are jointly monotonic and the third
  element determines which of the two triplets within in a square to consider.

  * k joint monotonicity constraints projection:
  If we know how to project into single joint monotonicity constraint then we
  can use Dykstra algorithm to project into union of all k joint monotonicity
  constraints.

  * Single joint monotonicity constraint projection
  Joint monotonicity constraints require the function to be monotonic along a
  diagonal direction of a two-feature subspace, ceteris paribus all other
  features. The sum of the partial derivatives on the constraint features needs
  to be non-negative. We can think of this as separate constraints applied to
  each 'triangle' of weights represented as either {(i,j,...), (i+1,j,...),
  (i,j+1,...)} or {(i+1,j+1,...), (i+1,1,...), (i,j+1,...)} where i  and j
  denote the index dimensions of the two features and the ellipses represent a
  fixed value of the other feature dimensions. Considering then a fixed slice,
  and a grid

  ```
  0---1---2---3
  | / | / | / |
  4---5---6---7
  | / | / | / |
  8---9---10--11
  | / | / | / |
  12--13--14--15
  ```

  we get our 8 independent sets of non-overlapping triangular triplets of
  vertices. For example, one set consists of {(0,1,4}, (8,9,12), (2,3,6),
  (10,11,14)}.

  * Individual weight projection
  Within each triangular triplet, the projection moves the weight of the right
  angled vertex, either top-left or bottom-right, by 2 * violation / 3 and the
  other two vertices by violation / 3 to satisfy the constraint while minimizing
  the L2 distance from the initial point.

  Args:
    weights: tensor with weights of lattice layer, with shape lattice_sizes.
    lattice_sizes: list or tuple of integers which represents lattice sizes
      which correspond to weights.
    joint_monotonicity: two-element tuple representing a single joint
      monotonicity constraint. The two elements are the index of the two
      constrained features.
    constraint_group: three-element tuple as defined above, representing the
      combination of 'even' and 'odd' constraints we are projecting on.

  Returns:
    Tensor with projected weights matching shape of input weights.
  """

  dim1, dim2 = joint_monotonicity
  layers = _unstack_nd(weights, [dim1, dim2])
  for i in range(constraint_group[0], lattice_sizes[dim1] - 1, 2):
    for j in range(constraint_group[1], lattice_sizes[dim2] - 1, 2):
      midpoint = (layers[i + 1][j] + layers[i][j + 1]) / 2
      if constraint_group[2] == 1:
        difference = midpoint - layers[i + 1][j + 1]
        correction = tf.maximum(difference / 3, 0)
        layers[i + 1][j + 1] += 2 * correction
      else:
        difference = midpoint - layers[i][j]
        correction = tf.minimum(difference / 3, 0)
        layers[i][j] += 2 * correction
      layers[i + 1][j] -= correction
      layers[i][j + 1] -= correction

  return _stack_nd(layers, [dim1, dim2])


def _project_partial_joint_unimodality(weights, lattice_sizes,
                                       joint_unimodalities, vertex, offsets):
  """Applies exact joint unimodality projection to given constraint group.

  Constraint group is represented by vertex and offsets. Vertex means vertex
  of lattice for which directional derivatives are being computed. Offsets is a
  list of {-1, 1} which represent which of hypercubes adjacent to vertex is
  being processed.
  Each pair of vertex and offsets results in linear equation involving
  len(vertex) + 1 of constrained vertices. We project onto this equation being
  positive or negative depending on whether we need peak or valley.

  Args:
    weights: tensor with weights of lattice layer, with shape lattice_sizes.
    lattice_sizes: list or tuple of integers which represents lattice sizes
      which correspond to weights.
    joint_unimodalities: tuple representing single joint unimodality constraint.
      Elements are the indices of constrained features followed by 'valley' or
      'peak'.
    vertex: len(joint_unimodalities)-1 dimensional lattice vertex from
      dimensions specified by joint_unimodalities.
    offsets: list of {-1, 1} which represents which of hypercubes adjacent to
      vertex is being processed.

  Returns:
    None or tensor with projected weights matching shape of input weights. In
    case of None pair: (vertex, offset) resulted into constraint group for
    which no update to weights is needed.
  """
  # This functoin builds hyperplane equation and then calls
  # _project_onto_hyperplane() to project.
  dimensions = joint_unimodalities[0]
  if len(vertex) != len(dimensions):
    raise ValueError("%s %s" % (vertex, joint_unimodalities))

  upper_bound = [lattice_sizes[dim] for dim in dimensions]
  center = [size // 2 for size in upper_bound]

  if all(v == c for v, c in zip(vertex, center)):
    return None

  equation = []
  all_vertices = []

  for dim, offset in enumerate(offsets):
    dim_weight = vertex[dim] - center[dim]
    if dim_weight == 0:
      continue

    neighbour = list(vertex)
    neighbour[dim] += offset
    if neighbour[dim] < 0 or neighbour[dim] >= upper_bound[dim]:
      return None

    all_vertices.append(neighbour)
    equation.append(dim_weight * offset)

  if not all_vertices:
    return None

  # Add 'vertex' iteself with corresponding weights.
  all_vertices.append(list(vertex))
  equation.append(-sum(equation))

  return _project_onto_hyperplane(
      weights=weights,
      joint_unimodalities=joint_unimodalities,
      hyperplane=equation,
      vertices=all_vertices)


def _project_onto_hyperplane(weights, joint_unimodalities, hyperplane,
                             vertices):
  """Projects onto hyperplane.

  Args:
    weights: tensor with weights of lattice layer, with shape lattice_sizes.
    joint_unimodalities: tuple representing a single joint unimodality
      constraint. Elements are the index of constrained features, followed by
      'valley' or 'peak'.
    hyperplane: list of coefficients of hyperplane onto which we project.
    vertices: list of len(joint_unimodalities)-1 dimensional points of lenght
      len(hyperplane) which correspond to coefficients of hyperplane. This
      points will be used to extract elements from 'weights' which are related
      to given hyperplane.

  Returns:
    Tensor with projected weights matching shape of input weights.
  """
  hyperplane = tf.constant(hyperplane, dtype=weights.dtype)

  # TODO: unstacking entire set of weights for the purpuse of projection
  # onto single hyperplane is very inefficient for high number of jointly
  # unimodal dims. Consider other options. So far I see 4 candidates:
  # 1) Find a way to efficiently combine independent hyperplanes so we can
  #    project onto several hyperplane at once. This would be correct projection
  #    with respect to L2 norm, but headroom for this approach is limited
  #    because for example for 4 constrained dims of size 3 (3^4) we have 81
  #    different varialbes and 5 variables per equation. This gives us upper
  #    bound of 81/5 = 16 times speed up. In reality it will probably be around
  #    5-10 times.
  # 2) Use tf.gather_nd() to gather affected weights instead of stacking and
  #    unstacking. Hard to estimare how much of an improvement it will be.
  # 3) Project onto all hyperplanes in a single step. This will violate Dykstra
  #    algorithm, so projection will not be into nearest point because according
  #    to Dykstra we need to porject into all dependent hyperplanes
  #    consequently. But regardless it could work well enough in practice and
  #    hopefully will be fast enough.
  # 4) Come up with better option.
  dimensions, direction = joint_unimodalities
  layers = _unstack_nd(weights, dims=dimensions)
  affected_weights = [
      _get_element(lists=layers, indices=position) for position in vertices
  ]

  affected_weights = tf.stack(affected_weights, axis=-1)
  violation = tf.reduce_sum(affected_weights * hyperplane, axis=-1)
  if direction == "valley":
    violation = tf.minimum(violation, 0.0)
  else:
    violation = tf.maximum(violation, 0.0)

  correction_factor = violation / tf.reduce_sum(hyperplane * hyperplane)
  correction = tf.expand_dims(correction_factor, axis=-1) * hyperplane
  projection = affected_weights - correction

  affected_weights = tf.unstack(projection, axis=-1)
  for tensor, position in zip(affected_weights, vertices):
    _set_element(lists=layers, indices=position, value=tensor)

  return _stack_nd(layers, dims=dimensions)


# TODO: Test whether adding min/max capping to dykstra projection would
# improve performance.
def project_by_dykstra(weights,
                       lattice_sizes,
                       monotonicities=None,
                       unimodalities=None,
                       edgeworth_trusts=None,
                       trapezoid_trusts=None,
                       monotonic_dominances=None,
                       range_dominances=None,
                       joint_monotonicities=None,
                       joint_unimodalities=None,
                       num_iterations=1):
  """Applies dykstra's projection algorithm for monotonicity/trust constraints.

  - Returns honest projection with respect to L2 norm if num_iterations is inf.
  - Monotonicity will be violated by some small eps(num_iterations).
  - Complexity: O(num_iterations * (num_monotonic_dims + num_trust_constraints)
    * num_lattice_weights)

  Dykstra's alternating projections algorithm projects into intersection of
  several convex sets. For algorithm description itself use Google or Wiki:
  https://en.wikipedia.org/wiki/Dykstra%27s_projection_algorithm

  Here, each monotonicity constraint is split up into 2 independent convex sets
  each trust constraint is split up into 4 independent convex sets. These sets
  are then projected onto exactly (in L2 space). For more details, see the
  _project_partial_* functions.

  Args:
    weights: `Lattice` weights tensor of shape: `(prod(lattice_sizes), units)`.
    lattice_sizes: list or tuple of integers which represents lattice sizes.
      which correspond to weights.
    monotonicities: None or list or tuple of same length as lattice_sizes of {0,
      1} which represents monotonicity constraints per dimension. 1 stands for
      increasing (non-decreasing in fact), 0 for no monotonicity constraints.
    unimodalities: None or list or tuple of same length as lattice_sizes of {-1,
      0, 1} which represents unimodality constraints per dimension. 1 indicates
      that function first decreases then increases, -1 indicates that function
      first increases then decreases, 0 indicates no unimodality constraints.
    edgeworth_trusts: None or iterable of three-element tuples. First element is
      the index of the main (monotonic) feature. Second element is the index of
      the conditional feature. Third element is the direction of trust: 1 if
        higher values of the conditional feature should increase trust in the
        main feature and -1 otherwise.
    trapezoid_trusts: None or iterable of three-element tuples. First element is
      the index of the main (monotonic) feature. Second element is the index of
      the conditional feature. Third element is the direction of trust: 1 if
        higher values of the conditional feature should increase trust in the
        main feature and -1 otherwise.
    monotonic_dominances: None or iterable of two-element tuples. First element
      is the index of the dominant feature. Second element is the index of the
      weak feature.
    range_dominances: None or iterable of two-element tuples. First element is
      the index of the dominant feature. Second element is the index of the weak
      feature.
    joint_monotonicities: None or iterable of two-element tuples. Each tuple
      represents a pair of feature indices that require joint monotoniticity.
    joint_unimodalities: None or tuple or iterable of tuples. Each tuple
      represents indices of single group of jointly unimodal features followed
      by 'valley' or 'peak'.
    num_iterations: number of iterations of Dykstra's algorithm.

  Returns:
    Projected weights tensor of same shape as `weights`.
  """
  if num_iterations == 0:
    return weights
  if (utils.count_non_zeros(monotonicities, unimodalities) == 0 and
      not joint_monotonicities and not joint_unimodalities and
      not range_dominances):
    return weights

  units = weights.shape[1]
  if monotonicities is None:
    monotonicities = [0] * len(lattice_sizes)
  if unimodalities is None:
    unimodalities = [0] * len(lattice_sizes)
  if edgeworth_trusts is None:
    edgeworth_trusts = []
  if trapezoid_trusts is None:
    trapezoid_trusts = []
  if monotonic_dominances is None:
    monotonic_dominances = []
  if range_dominances is None:
    range_dominances = []
  if joint_monotonicities is None:
    joint_monotonicities = []
  if joint_unimodalities is None:
    joint_unimodalities = []
  if units > 1:
    lattice_sizes = lattice_sizes + [int(units)]
    monotonicities = monotonicities + [0]
    unimodalities = unimodalities + [0]

  weights = tf.reshape(weights, lattice_sizes)

  def body(iteration, weights, last_change):
    """Body of the tf.while_loop for Dykstra's projection algorithm.

    This implements Dykstra's projection algorithm and requires rolling back
    the last projection change.

    Args:
      iteration: Iteration counter tensor.
      weights: Tensor with project weights at each iteraiton.
      last_change: Dict that stores the last change in the weights after
        projecting onto the each subset of constraints.

    Returns:
      The tuple (iteration, weights, last_change) at the end of each iteration.
    """
    last_change = copy.copy(last_change)
    for dim in range(len(lattice_sizes)):
      if monotonicities[dim] == 0 and unimodalities[dim] == 0:
        continue

      for constraint_group in [0, 1]:
        # Iterate over 2 sets of constraints per dimension: even and odd.
        # Odd set exists only when there are more than 2 lattice vertices.
        if constraint_group + 1 >= lattice_sizes[dim]:
          continue

        # Rolling back last projection into current set as required by Dykstra's
        # algorithm.
        rolled_back_weights = weights - last_change[("MONOTONICITY", dim,
                                                     constraint_group)]
        weights = _project_partial_monotonicity(rolled_back_weights,
                                                lattice_sizes, monotonicities,
                                                unimodalities, dim,
                                                constraint_group)
        last_change[("MONOTONICITY", dim,
                     constraint_group)] = weights - rolled_back_weights

    for constraint in edgeworth_trusts:
      main_dim, cond_dim, _ = constraint
      for constraint_group in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        if (constraint_group[0] >= lattice_sizes[main_dim] - 1 or
            constraint_group[1] >= lattice_sizes[cond_dim] - 1):
          continue

        rolled_back_weights = (
            weights - last_change[("EDGEWORTH", constraint, constraint_group)])
        weights = _project_partial_edgeworth(rolled_back_weights, lattice_sizes,
                                             constraint, constraint_group)
        last_change[("EDGEWORTH", constraint,
                     constraint_group)] = weights - rolled_back_weights

    for constraint in trapezoid_trusts:
      _, cond_dim, _ = constraint
      for constraint_group in [0, 1]:
        if constraint_group >= lattice_sizes[cond_dim] - 1:
          continue

        rolled_back_weights = (
            weights - last_change[("TRAPEZOID", constraint, constraint_group)])
        weights = _project_partial_trapezoid(rolled_back_weights, lattice_sizes,
                                             constraint, constraint_group)
        last_change[("TRAPEZOID", constraint,
                     constraint_group)] = weights - rolled_back_weights

    for constraint in monotonic_dominances:
      dominant_dim, weak_dim = constraint
      for constraint_group in itertools.product([0, 1], [0, 1], [0, 1]):
        if (constraint_group[0] >= lattice_sizes[dominant_dim] - 1 or
            constraint_group[1] >= lattice_sizes[weak_dim] - 1):
          continue

        rolled_back_weights = weights - last_change[
            ("MONOTONIC_DOMINANCE", constraint, constraint_group)]
        weights = _project_partial_monotonic_dominance(rolled_back_weights,
                                                       lattice_sizes,
                                                       constraint,
                                                       constraint_group)
        last_change[("MONOTONIC_DOMINANCE", constraint,
                     constraint_group)] = weights - rolled_back_weights

    for constraint in range_dominances:
      dominant_dim, weak_dim = constraint
      dom_dim_idx = range(lattice_sizes[dominant_dim])
      weak_dim_idx = range(lattice_sizes[weak_dim])
      for constraint_group in itertools.product(dom_dim_idx, weak_dim_idx):
        rolled_back_weights = weights - last_change[
            ("RANGE_DOMINANCE", constraint, constraint_group)]
        weights = _project_partial_range_dominance(rolled_back_weights,
                                                   lattice_sizes, constraint,
                                                   constraint_group)
        last_change[("RANGE_DOMINANCE", constraint,
                     constraint_group)] = weights - rolled_back_weights

    for constraint in joint_monotonicities:
      dim1, dim2 = constraint
      for constraint_group in itertools.product([0, 1], [0, 1], [0, 1]):
        if (constraint_group[0] >= lattice_sizes[dim1] - 1 or
            constraint_group[1] >= lattice_sizes[dim2] - 1):
          continue

        rolled_back_weights = weights - last_change[
            ("JOINT_MONOTONICITY", constraint, constraint_group)]
        weights = _project_partial_joint_monotonicity(rolled_back_weights,
                                                      lattice_sizes, constraint,
                                                      constraint_group)
        last_change[("JOINT_MONOTONICITY", constraint,
                     constraint_group)] = weights - rolled_back_weights

    for constraint in joint_unimodalities:
      dimensions = tuple(constraint[0])
      lattice_ranges = [range(lattice_sizes[dim]) for dim in dimensions]
      for vertex in itertools.product(*lattice_ranges):
        for offsets in itertools.product([-1, 1], repeat=len(dimensions)):
          # For this projection constraint group is represented by pair: vertex,
          # offsets.
          projection_key = ("JOINT_UNIMODALITY", dimensions, vertex, offsets)
          if projection_key in last_change:
            rolled_back_weights = weights - last_change[projection_key]
          else:
            rolled_back_weights = weights
          projected_weights = _project_partial_joint_unimodality(
              weights=rolled_back_weights,
              lattice_sizes=lattice_sizes,
              joint_unimodalities=constraint,
              vertex=vertex,
              offsets=offsets)
          if projected_weights is not None:
            weights = projected_weights
            last_change[projection_key] = weights - rolled_back_weights
    return iteration + 1, weights, last_change

  def cond(iteration, weights, last_change):
    del weights, last_change
    return tf.less(iteration, num_iterations)

  # Run the body of the loop once to find required last_change keys. The set of
  # keys in the input and output of the body of tf.while_loop must be the same.
  # The resulting ops are discarded and will not be part of the TF graph.
  zeros = tf.zeros(shape=lattice_sizes, dtype=weights.dtype)
  last_change = collections.defaultdict(lambda: zeros)
  (_, _, last_change) = body(0, weights, last_change)

  # Apply Dykstra's algorithm with tf.while_loop.
  iteration = tf.constant(0)
  last_change = {k: zeros for k in last_change}
  (_, weights, _) = tf.while_loop(cond, body, (iteration, weights, last_change))
  return tf.reshape(weights, shape=[-1, units])


def laplacian_regularizer(weights, lattice_sizes, l1=0.0, l2=0.0):
  """Returns Laplacian regularization loss for `Lattice` layer.

  Laplacian regularizer penalizes the difference between adjacent vertices in
  multi-cell lattice (see
  [publication](http://jmlr.org/papers/v17/15-243.html)).

  Consider a 3 x 2 lattice with weights `w`:

  ```
  w[3]-----w[4]-----w[5]
    |        |        |
    |        |        |
  w[0]-----w[1]-----w[2]
  ```

  where the number at each node represents the weight index.
  In this case, the laplacian regularizer is defined as:

  ```
  l1[0] * (|w[1] - w[0]| + |w[2] - w[1]| +
           |w[4] - w[3]| + |w[5] - w[4]|) +
  l1[1] * (|w[3] - w[0]| + |w[4] - w[1]| + |w[5] - w[2]|) +

  l2[0] * ((w[1] - w[0])^2 + (w[2] - w[1])^2 +
           (w[4] - w[3])^2 + (w[5] - w[4])^2) +
  l2[1] * ((w[3] - w[0])^2 + (w[4] - w[1])^2 + (w[5] - w[2])^2)
  ```

  Arguments:
    weights: `Lattice` weights tensor of shape: `(prod(lattice_sizes), units)`.
    lattice_sizes: List or tuple of integers which represents lattice sizes.
    l1: l1 regularization amount. Either single float or list or tuple of floats
      to specify different regularization amount per dimension.
    l2: l2 regularization amount. Either single float or list or tuple of floats
      to specify different regularization amount per dimension.

  Returns:
    Laplacian regularization loss.
  """
  if not l1 and not l2:
    return 0.0

  rank = len(lattice_sizes)
  # If regularization amount is given as single float assume same amount for
  # every dimension.
  if l1 and not isinstance(l1, (list, tuple)):
    l1 = [l1] * rank
  if l2 and not isinstance(l2, (list, tuple)):
    l2 = [l2] * rank

  if weights.shape[1] > 1:
    lattice_sizes = lattice_sizes + [int(weights.shape[1])]
    rank += 1
    if l1:
      l1 = l1 + [0.0]
    if l2:
      l2 = l2 + [0.0]
  weights = tf.reshape(weights, shape=lattice_sizes)

  result = tf.constant(0.0, shape=[], dtype=weights.dtype)
  for dim in range(rank):
    if (not l1 or not l1[dim]) and (not l2 or not l2[dim]):
      continue
    if dim > 0:
      # Transpose so current dimension becomes first one in order to simplify
      # indexing and be able to merge all other dimensions into 1 for better TPU
      # performance.
      permut = [p for p in range(rank)]
      permut[0], permut[dim] = permut[dim], permut[0]
      slices = tf.transpose(weights, perm=permut)
    else:
      slices = weights
    slices = tf.reshape(slices, shape=[lattice_sizes[dim], -1])

    diff = slices[1:] - slices[0:-1]
    if l1:
      result += tf.reduce_sum(tf.abs(diff)) * l1[dim]
    if l2:
      result += tf.reduce_sum(tf.square(diff)) * l2[dim]
  return result


def torsion_regularizer(weights, lattice_sizes, l1=0.0, l2=0.0):
  """Returns Torsion regularization loss for `Lattice` layer.

  Lattice torsion regularizer penalizes how much the lattice function twists
  from side-to-side (see
  [publication](http://jmlr.org/papers/v17/15-243.html)).

  Consider a 3 x 2 lattice with weights `w`:

  ```
  w[3]-----w[4]-----w[5]
    |        |        |
    |        |        |
  w[0]-----w[1]-----w[2]
  ```

  In this case, the torsion regularizer is defined as:

  ```
  l1 * (|w[4] + w[0] - w[3] - w[1]| + |w[5] + w[1] - w[4] - w[2]|) +
  l2 * ((w[4] + w[0] - w[3] - w[1])^2 + (w[5] + w[1] - w[4] - w[2])^2)
  ```

  Arguments:
    weights: `Lattice` weights tensor of shape: `(prod(lattice_sizes), units)`.
    lattice_sizes: List or tuple of integers which represents lattice sizes.
    l1: l1 regularization amount. Either single float or list or tuple of floats
      to specify different regularization amount per dimension.
    l2: l2 regularization amount. Either single float or list or tuple of floats
      to specify different regularization amount per dimension. The amount for
      the interaction term between i and j is the corresponding product of each
      per feature amount.

  Returns:
    Laplacian regularization loss.
  """
  rank = len(lattice_sizes)
  if rank == 1 or (not l1 and not l2):
    return 0.0

  # If regularization amount is given as single float assume same amount for
  # every dimension.
  if l1 and not isinstance(l1, (list, tuple)):
    l1 = [math.sqrt(l1)] * rank
  if l2 and not isinstance(l2, (list, tuple)):
    l2 = [math.sqrt(l2)] * rank

  if weights.shape[1] > 1:
    lattice_sizes = lattice_sizes + [int(weights.shape[1])]
    rank += 1
    if l1:
      l1 = l1 + [0.0]
    if l2:
      l2 = l2 + [0.0]
  weights = tf.reshape(weights, shape=lattice_sizes)

  result = tf.constant(0.0, shape=[], dtype=weights.dtype)
  for i in range(rank - 1):
    for j in range(i + 1, rank):
      if ((not l1 or not l1[i] or not l1[j]) and
          (not l2 or not l2[i] or not l2[j])):
        continue
      if j == 1:
        planes = weights
      else:
        # Transpose so dimensions i and j become first in order to simplify
        # indexing and be able to merge all other dimensions into 1 for better
        # TPU performance.
        permut = [p for p in range(rank)]
        permut[0], permut[i] = permut[i], permut[0]
        permut[1], permut[j] = permut[j], permut[1]
        planes = tf.transpose(weights, perm=permut)
      planes = tf.reshape(
          planes, shape=[lattice_sizes[i], lattice_sizes[j], -1])

      a00 = planes[0:-1, 0:-1]
      a01 = planes[0:-1, 1:]
      a10 = planes[1:, 0:-1]
      a11 = planes[1:, 1:]
      torsion = a00 + a11 - a01 - a10

      if l1:
        result += tf.reduce_sum(tf.abs(torsion)) * l1[i] * l1[j]
      if l2:
        result += tf.reduce_sum(tf.square(torsion)) * l2[i] * l2[j]
  return result


def _verify_dominances_hyperparameters(dominances, dominance_type,
                                       monotonicities, num_input_dims):
  """Verifies that dominances hyperparameters are consistent.

  Args:
    dominances: Dominances hyperparameters of `Lattice` layer.
    dominance_type: Type of dominance constraints which is either 'monotonic' or
      'range'.
    monotonicities: Monotonicities hyperparameter of `Lattice` layer.
    num_input_dims: Number of input dimensions.

  Raises:
    ValueError: If something is inconsistent.
  """
  assert dominance_type in ("monotonic", "range")
  dim_pairs = set()
  for constraint in dominances:
    if len(constraint) != 2:
      raise ValueError("%s dominance constraints must consist of 2 elements. "
                       "Seeing constraint tuple %s" %
                       (dominance_type.capitalize(), constraint))
    dominant_dim, weak_dim = constraint
    if (dominant_dim >= num_input_dims or weak_dim >= num_input_dims or
        dominant_dim < 0 or weak_dim < 0):
      raise ValueError("Dimensions constrained by %s dominance constraints are "
                       "not within the range of the lattice. 'dims': %s, %s, "
                       "num_dims: %s" %
                       (dominance_type, dominant_dim, weak_dim, num_input_dims))
    if not isinstance(dominant_dim, int) or not isinstance(weak_dim, int):
      raise ValueError("%s dominance constraint dimensions must be integers. "
                       "Seeing dominant_dim %s and weak_dim %s" %
                       (dominance_type.capitalize(), dominant_dim, weak_dim))
    for dim in [dominant_dim, weak_dim]:
      if monotonicities[dim] != 1:
        raise ValueError("%s dominance constraint's dimensions must be "
                         "monotonic. Dimension %d is not monotonic." %
                         (dominance_type.capitalize(), dim))
    # TODO: Determine partial ordering of features by dominance and
    # detect any inconsistencies.
    if (weak_dim, dominant_dim) in dim_pairs:
      raise ValueError("Cannot have two %s dominance constraints on the same "
                       "pair of features conflicting. Features: %d, %d" %
                       (dominance_type, dominant_dim, weak_dim))
    dim_pairs.add((dominant_dim, weak_dim))


def verify_hyperparameters(lattice_sizes,
                           units=None,
                           weights_shape=None,
                           input_shape=None,
                           monotonicities=None,
                           unimodalities=None,
                           edgeworth_trusts=None,
                           trapezoid_trusts=None,
                           monotonic_dominances=None,
                           range_dominances=None,
                           joint_monotonicities=None,
                           joint_unimodalities=None,
                           output_min=None,
                           output_max=None,
                           regularization_amount=None,
                           regularization_info="",
                           interpolation="hypercube"):
  """Verifies that all given hyperparameters are consistent.

  This function does not inspect weights themselves. Only their shape. Use
  `assert_constraints()` to assert actual weights against constraints.

  See `tfl.layers.Lattice` class level comment for detailed description of
  arguments.

  Args:
    lattice_sizes: Lattice sizes to check againts.
    units: Units hyperparameter of `Lattice` layer.
    weights_shape: Shape of tensor which represents `Lattice` layer weights.
    input_shape: Shape of layer input. Useful only if `units` is set.
    monotonicities: Monotonicities hyperparameter of `Lattice` layer.
    unimodalities: Unimodalities hyperparameter of `Lattice` layer.
    edgeworth_trusts: Edgeworth_trusts hyperparameter of `Lattice` layer.
    trapezoid_trusts: Trapezoid_trusts hyperparameter of `Lattice` layer.
    monotonic_dominances: Monotonic dominances hyperparameter of `Lattice`
      layer.
    range_dominances: Range dominances hyperparameter of `Lattice` layer.
    joint_monotonicities: Joint monotonicities hyperparameter of `Lattice`
      layer.
    joint_unimodalities: Joint unimodalities hyperparameter of `Lattice` layer.
    output_min: Minimum output of `Lattice` layer.
    output_max: Maximum output of `Lattice` layer.
    regularization_amount: Regularization amount for regularizers.
    regularization_info: String which describes `regularization_amount`.
    interpolation: One of 'simplex' or 'hypercube' interpolation.

  Raises:
    ValueError: If something is inconsistent.
  """
  for size in lattice_sizes:
    if size < 2:
      raise ValueError("All lattice sizes must be at least 2. Given: %s" %
                       lattice_sizes)

  # It also raises errors if monotonicities specified incorrectly.
  monotonicities = utils.canonicalize_monotonicities(
      monotonicities, allow_decreasing=False)
  if monotonicities is not None:
    if len(monotonicities) != len(lattice_sizes):
      raise ValueError("If provided 'monotonicities' should have same number "
                       "of elements as 'lattice_sizes'. 'monotonicities': %s,"
                       "'lattice_sizes: %s" % (monotonicities, lattice_sizes))

  unimodalities = utils.canonicalize_unimodalities(unimodalities)
  if unimodalities is not None:
    if len(unimodalities) != len(lattice_sizes):
      raise ValueError("If provided 'unimodalities' should have same number "
                       "of elements as 'lattice_sizes'. 'unimodalities': %s, "
                       "'lattice_sizes: %s" % (unimodalities, lattice_sizes))
    for unimodality, dim_size in zip(unimodalities, lattice_sizes):
      if unimodality != 0 and dim_size < 3:
        raise ValueError("Unimodal dimensions must have lattice size at "
                         "least 3. unimodalities: %s, lattice_sizes: %s" %
                         (unimodalities, lattice_sizes))

  if monotonicities is not None and unimodalities is not None:
    for i, (monotonicity,
            unimodality) in enumerate(zip(monotonicities, unimodalities)):
      if monotonicity != 0 and unimodality != 0:
        raise ValueError("Both monotonicity and unimodality can not be set "
                         "simultaniously for same dimension. Dimension: %d, "
                         "'monotonicities': %s, 'unimodalities': %s" %
                         (i, monotonicities, unimodalities))

  all_trusts = utils.canonicalize_trust((edgeworth_trusts or []) +
                                        (trapezoid_trusts or [])) or []
  main_dims, cond_dims, trapezoid_cond_dims = set(), set(), set()
  dim_pairs_direction = {}
  for i, constraint in enumerate(all_trusts):
    main_dim, cond_dim, cond_direction = constraint
    if (main_dim >= len(lattice_sizes) or cond_dim >= len(lattice_sizes) or
        main_dim < 0 or cond_dim < 0):
      raise ValueError("Dimensions constrained by trust constraints "
                       "are not within the range of the lattice. "
                       "'trust_dims': %s, %s, num_dims: %s" %
                       (main_dim, cond_dim, len(lattice_sizes)))
    if not isinstance(main_dim, int) or not isinstance(cond_dim, int):
      raise ValueError("Trust constraint dimensions must be integers. Seeing "
                       "main_dim %s and cond_dim %s" % (main_dim, cond_dim))
    if monotonicities[main_dim] != 1:
      raise ValueError("Trust constraint's main feature must be "
                       "monotonic. Dimension %s is not monotonic." % (main_dim))
    if (main_dim, cond_dim) in dim_pairs_direction and dim_pairs_direction[
        (main_dim, cond_dim)] != cond_direction:
      raise ValueError("Cannot have two trust constraints on the same pair of "
                       "features in opposite directions. Features: %d, %d" %
                       (main_dim, cond_dim))
    # Only apply this check to trapezoid constraints when there are also
    # edgeworth constraints.
    if edgeworth_trusts and i >= len(edgeworth_trusts):
      if cond_dim in trapezoid_cond_dims:
        logging.warning(
            "Conditional dimension %d is being used in multiple trapezoid "
            "trust constraints. Because of this and the presence of edgeworth "
            "constraints, there may be slight trust violations of one or more "
            "of these constraints at the end of training. Consider increasing "
            "num_projection_iterations to reduce violation.", cond_dim)
      trapezoid_cond_dims.add(cond_dim)
    main_dims.add(main_dim)
    cond_dims.add(cond_dim)
    dim_pairs_direction[(main_dim, cond_dim)] = cond_direction
  main_and_cond = main_dims.intersection(cond_dims)
  if main_and_cond:
    raise ValueError("A feature cannot be both a main feature and a "
                     "conditional feature in trust constraints. "
                     "Seeing dimension %d in both" % (main_and_cond.pop()))

  if monotonic_dominances is not None:
    _verify_dominances_hyperparameters(monotonic_dominances, "monotonic",
                                       monotonicities, len(lattice_sizes))
  if range_dominances is not None:
    _verify_dominances_hyperparameters(range_dominances, "range",
                                       monotonicities, len(lattice_sizes))

  if joint_monotonicities is not None:
    for i, constraint in enumerate(joint_monotonicities):
      if len(constraint) != 2:
        raise ValueError("Joint monotonicities constraints must consist of 2 "
                         "elements. Seeing constraint tuple %s" % (constraint,))
      dim1, dim2 = constraint
      if (dim1 >= len(lattice_sizes) or dim2 >= len(lattice_sizes) or
          dim1 < 0 or dim2 < 0):
        raise ValueError("Dimensions constrained by joint monotonicity "
                         "constraints are not within the range of the lattice. "
                         "'dims': %s, %s, num_dims: %s" %
                         (dim1, dim2, len(lattice_sizes)))
      if not isinstance(dim1, int) or not isinstance(dim2, int):
        raise ValueError("Joint monotonicity constraint dimensions must be "
                         "integers. Seeing dimensions %s, %s" % (dim1, dim2))

  if joint_unimodalities is not None:
    for single_constraint in joint_unimodalities:
      dimensions, direction = single_constraint
      if (not isinstance(direction, six.string_types) or
          (direction.lower() != "valley" and direction.lower() != "peak")):
        raise ValueError("Joint unimodality tuple must end with string 'valley'"
                         " or 'peak' which represents unimodality direction. "
                         "Given: %s" % (single_constraint,))
      for dim in dimensions:
        if dim < 0 or dim >= len(lattice_sizes):
          raise ValueError("Dimension constrained by joint unimodality is not "
                           "within the range of the lattice. Joint unimodality "
                           "dimension: %s, total number of dimensions: "
                           "%s" % (dim, len(lattice_sizes)))
        if not isinstance(dim, int):
          raise ValueError("Joint unimodality constraint dimensions must be "
                           "integer. Seeing: %s" % dim)
        if lattice_sizes[dim] < 3:
          raise ValueError("Dimensions constrained for joint unimodality must "
                           "have lattice size at least 3. "
                           "Dim: %s has size: %s" % (dim, lattice_sizes[dim]))
        if monotonicities and monotonicities[dim] != 0:
          raise ValueError("Dimension %d constrained for joint_unimodalities "
                           "can not also by monotonic." % dim)
      dims_set = set(dimensions)
      if len(dims_set) != len(dimensions):
        raise ValueError("All dimensions within single joint unimodality "
                         "constraint must be distinct. "
                         "Given: %s" % single_constraint)

  if weights_shape is not None:
    if len(weights_shape) != 2:
      raise ValueError("Weights must have shape of rank-2. "
                       "Given: %s" % weights_shape)
    expected_num_weights = 1
    for dim_size in lattice_sizes:
      expected_num_weights *= dim_size
    if weights_shape[0] != expected_num_weights:
      raise ValueError("Number of elements in weights does not correspond to "
                       "lattice sizes. Weights shape: %s, lattice sizes: %s, "
                       "Number of elements defined by lattice sizes: %d" %
                       (weights_shape, lattice_sizes, expected_num_weights))

  if input_shape is not None:
    if isinstance(input_shape[-1], int):
      if input_shape[-1] != len(lattice_sizes):
        raise ValueError("Last dimension of input shape must have same number "
                         "of elements as 'lattice_sizes'. 'input shape': %s, "
                         "'lattice_sizes': %s" % (input_shape, lattice_sizes))
      shape = input_shape
    else:
      if len(input_shape) != len(lattice_sizes):
        raise ValueError("If lattice input is provided as list of tensors their"
                         " number must match lattice_sizes. 'input list': %s, "
                         "'lattice_sizes': %s" % (input_shape, lattice_sizes))
      shape = input_shape[0]
    if units is not None:  # FYI: It is inside "if input_shape is not None:"
      if units > 1 and (len(shape) < 3 or shape[-2] != units):
        raise ValueError("If 'units' > 1 then input shape of Lattice layer must"
                         " have rank at least 3 where second from last "
                         "dimension is equal to 'units'. 'units': %s, "
                         "input_shape: %s" % (units, input_shape))

  if output_min is not None and output_max is not None:
    if output_min >= output_max:
      raise ValueError("'output_min' must be not greater than 'output_max'. "
                       "'output_min': %f, 'output_max': %f" %
                       (output_min, output_max))

  if regularization_amount and isinstance(regularization_amount, (list, tuple)):
    if len(regularization_amount) != len(lattice_sizes):
      raise ValueError(
          "If %s losses are given per dimension their number must "
          "match number of dimensions defined by lattice sizes. "
          "l1: %s, lattice sizes: %s" %
          (regularization_info, regularization_amount, lattice_sizes))

  if interpolation not in ["hypercube", "simplex"]:
    raise ValueError("Lattice interpolation type should be either 'simplex' "
                     "or 'hypercube': %s" % interpolation)


# TODO: investigate whether eps should be bigger.
def assert_constraints(weights,
                       lattice_sizes,
                       monotonicities,
                       edgeworth_trusts,
                       trapezoid_trusts,
                       monotonic_dominances,
                       range_dominances,
                       joint_monotonicities,
                       joint_unimodalities,
                       output_min=None,
                       output_max=None,
                       eps=1e-6):
  """Asserts that weights satisfy constraints.

  Args:
    weights: `Lattice` weights tensor of shape: `(prod(lattice_sizes), units)`.
    lattice_sizes: List or tuple of integers which represents lattice sizes.
    monotonicities: Monotonicity constraints.
    edgeworth_trusts: Edgeworth trust constraints.
    trapezoid_trusts: Trapezoid trust constraints.
    monotonic_dominances: Monotonic dominance constraints.
    range_dominances: Range dominance constraints.
    joint_monotonicities: Joint monotonicity constraints.
    joint_unimodalities: Joint unimodality constraints.
    output_min: None or lower bound constraints.
    output_max: None or upper bound constraints.
    eps: Allowed constraints violation.

  Returns:
    List of assertion ops in graph mode or directly executes assertions in eager
    mode.
  """
  # TODO: actually assert them.
  del joint_unimodalities

  if weights.shape[1] > 1:
    lattice_sizes = lattice_sizes + [int(weights.shape[1])]
    if monotonicities:
      monotonicities = monotonicities + [0]
  weights = tf.reshape(weights, shape=lattice_sizes)
  asserts = []

  for i in range(len(monotonicities or [])):
    if monotonicities[i] != 1:
      continue
    weights_layers = tf.unstack(weights, axis=i)

    for j in range(1, len(weights_layers)):
      diff = tf.reduce_min(weights_layers[j] - weights_layers[j - 1])
      asserts.append(
          tf.Assert(
              diff >= -eps,
              data=[
                  "Monotonicity violation", "Feature index:", i,
                  "Min monotonicity diff:", diff, "Upper layer number:", j,
                  "Epsilon:", eps, "Layers:", weights_layers[j],
                  weights_layers[j - 1]
              ]))

  for main_dim, cond_dim, cond_direction in edgeworth_trusts or []:
    weights_layers = _unstack_nd(weights, [main_dim, cond_dim])
    for i in range(lattice_sizes[main_dim] - 1):
      for j in range(lattice_sizes[cond_dim] - 1):
        diff = tf.reduce_min(
            cond_direction *
            ((weights_layers[i + 1][j + 1] - weights_layers[i][j + 1]) -
             (weights_layers[i + 1][j] - weights_layers[i][j])))
        asserts.append(
            tf.Assert(
                diff >= -eps,
                data=[
                    "Edgeworth trust violation", "Feature indices:", main_dim,
                    ",", cond_dim, "Min trust diff:", diff, "Epsilon:", eps,
                    "Layers:", weights_layers[i + 1][j + 1],
                    weights_layers[i][j + 1], weights_layers[i + 1][j],
                    weights_layers[i][j]
                ]))

  for main_dim, cond_dim, cond_direction in trapezoid_trusts or []:
    weights_layers = _unstack_nd(weights, [main_dim, cond_dim])
    max_main_dim = lattice_sizes[main_dim] - 1
    for j in range(lattice_sizes[cond_dim] - 1):
      lhs_diff = tf.reduce_min(
          cond_direction * (weights_layers[0][j] - weights_layers[0][j + 1]))
      asserts.append(
          tf.Assert(
              lhs_diff >= -eps,
              data=[
                  "Trapezoid trust violation", "Feature indices:", main_dim,
                  ",", cond_dim, "Min trust diff:", lhs_diff, "Epsilon:", eps,
                  "Layers:", weights_layers[0][j], weights_layers[0][j + 1]
              ]))
      rhs_diff = tf.reduce_min(cond_direction *
                               (weights_layers[max_main_dim][j + 1] -
                                weights_layers[max_main_dim][j]))
      asserts.append(
          tf.Assert(
              rhs_diff >= -eps,
              data=[
                  "Trapezoid trust violation", "Feature indices:", main_dim,
                  ",", cond_dim, "Min trust diff:", rhs_diff, "Epsilon:", eps,
                  "Layers:", weights_layers[max_main_dim][j + 1],
                  weights_layers[max_main_dim][j]
              ]))

  for dominant_dim, weak_dim in monotonic_dominances or []:
    weights_layers = _unstack_nd(weights, [dominant_dim, weak_dim])
    for i in range(lattice_sizes[dominant_dim] - 1):
      for j in range(lattice_sizes[weak_dim] - 1):
        midpoint = (weights_layers[i + 1][j + 1] + weights_layers[i][j]) / 2
        dominant_diff = tf.reduce_min(weights_layers[i + 1][j] - midpoint)
        asserts.append(
            tf.Assert(
                dominant_diff >= -eps,
                data=[
                    "Dominance violation", "Feature indices:", dominant_dim,
                    ",", weak_dim, "Min dominance diff:", dominant_diff,
                    "Epsilon:", eps, "Layers:", weights_layers[i][j],
                    weights_layers[i + 1][j], weights_layers[i + 1][j + 1]
                ]))
        weak_diff = tf.reduce_min(midpoint - weights_layers[i][j + 1])
        asserts.append(
            tf.Assert(
                weak_diff >= -eps,
                data=[
                    "Dominance violation", "Feature indices:", dominant_dim,
                    ",", weak_dim, "Min dominance diff:", weak_diff, "Epsilon:",
                    eps, "Layers:", weights_layers[i][j],
                    weights_layers[i + 1][j], weights_layers[i + 1][j + 1]
                ]))

  for dominant_dim, weak_dim in range_dominances or []:
    weights_layers = _unstack_nd(weights, [dominant_dim, weak_dim])
    dom_dim_size = lattice_sizes[dominant_dim]
    weak_dim_size = lattice_sizes[weak_dim]
    for i in range(dom_dim_size):
      for j in range(weak_dim_size):
        diff = tf.reduce_min(
            (weights_layers[dom_dim_size - 1][j] - weights_layers[0][j]) -
            (weights_layers[i][weak_dim_size - 1] - weights_layers[i][0]))
        asserts.append(
            tf.Assert(
                diff >= -eps,
                data=[
                    "Range dominance violation", "Feature indices:",
                    dominant_dim, ",", weak_dim, "Min dominance diff:", diff,
                    "Epsilon:", eps, "Layers:",
                    weights_layers[dom_dim_size - 1][j], weights_layers[0][j],
                    weights_layers[i][weak_dim_size - 1], weights_layers[i][0]
                ]))

  for dim1, dim2 in joint_monotonicities or []:
    weights_layers = _unstack_nd(weights, [dim1, dim2])
    for i in range(lattice_sizes[dim1] - 1):
      for j in range(lattice_sizes[dim2] - 1):
        midpoint = (weights_layers[i + 1][j] + weights_layers[i][j + 1]) / 2
        lower_triangle_diff = tf.reduce_min(weights_layers[i + 1][j + 1] -
                                            midpoint)
        asserts.append(
            tf.Assert(
                lower_triangle_diff >= -eps,
                data=[
                    "Joint monotonicity violation", "Feature indices:", dim1,
                    ",", dim2, "Min lower triangle diff:", lower_triangle_diff,
                    "Epsilon:", eps, "Layers:", weights_layers[i + 1][j + 1],
                    weights_layers[i + 1][j], weights_layers[i][j + 1]
                ]))
        upper_triangle_diff = tf.reduce_min(midpoint - weights_layers[i][j])
        asserts.append(
            tf.Assert(
                upper_triangle_diff >= -eps,
                data=[
                    "Joint monotonicity violation", "Feature indices:", dim1,
                    ",", dim2, "Min upper triangle diff:", upper_triangle_diff,
                    "Epsilon:", eps, "Layers:", weights_layers[i][j],
                    weights_layers[i + 1][j], weights_layers[i][j + 1]
                ]))

  if output_min is not None:
    min_weight = tf.reduce_min(weights)
    asserts.append(
        tf.Assert(
            min_weight >= output_min - eps,
            data=[
                "Lower bound violation.", "output_min:", output_min,
                "Smallest weight:", min_weight, "Epsilon:", eps, "Weights:",
                weights
            ]))

  if output_max is not None:
    max_weight = tf.reduce_max(weights)
    asserts.append(
        tf.Assert(
            max_weight <= output_max + eps,
            data=[
                "Upper bound violation.", "output_max:", output_max,
                "Largest weight:", max_weight, "Epsilon:", eps, "Weights:",
                weights
            ]))
  return asserts


def _unstack_nested_lists(tensor_or_list, axis):
  """Unstacks tensors stored within nested list."""
  if isinstance(tensor_or_list, list):
    return [_unstack_nested_lists(item, axis) for item in tensor_or_list]
  else:
    return tf.unstack(tensor_or_list, axis=axis)


def _unstack_nd(tensor, dims):
  """Returns nested lists of tensors resulting from n unstack operations."""
  dims = list(dims)
  # Following unstack operations will remove some dims. It will result in dims
  # higher than removed dims shifting left. So update passed in dims to reflect
  # shift resulted from tf.unstack() in advance.
  for i in range(len(dims) - 1, 0, -1):
    dims[i] -= sum([dims[i] > previous_dims for previous_dims in dims[:i]])

  result = tensor
  for dim in dims:
    result = _unstack_nested_lists(result, axis=dim)
  return result


def _stack_nested_lists(tensor_or_list, axis):
  """Stacks tensors stored within nested list.."""
  if isinstance(tensor_or_list[0], list):
    return [_stack_nested_lists(item, axis) for item in tensor_or_list]
  else:
    return tf.stack(tensor_or_list, axis=axis)


def _stack_nd(tensor, dims):
  """Returns tensor that re-stacks tensor layers formed from unstacking."""
  dims = list(dims)
  # Following stack operations will add some dims. It will result in dims higher
  # than removed dims shifting right. So update passed in dims to reflect
  # shift resulted from tf.stack() in advance.
  for i in range(len(dims) - 1, 0, -1):
    dims[i] -= sum([dims[i] > previous_dims for previous_dims in dims[:i]])

  result = tensor
  for dim in reversed(dims):
    result = _stack_nested_lists(result, axis=dim)
  return result


def _get_element(lists, indices):
  """Gets element from nested lists of arbitrary depth."""
  result = lists
  for i in indices:
    result = result[i]
  return result


def _set_element(lists, indices, value):
  """Sets element into nested lists of arbitrary depth."""
  result = lists
  for i in indices[:-1]:
    result = result[i]
  result[indices[-1]] = value


def _reverse_second_list_dimension(layers):
  """Reverses each list within a list of lists, but not the outer list."""
  return [layer[::-1] for layer in layers]
