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
"""Lattice layers library for TensorFlow Lattice.

Lattice is an interpolated lookup table (LUT), part of TensorFlow Lattice
models.

This modules provides functions used when building models, as opposed to the
basic operators exported by lattice_ops.py
"""
import functools

from tensorflow_lattice.python.lib import regularizers
from tensorflow_lattice.python.lib import tools
from tensorflow_lattice.python.ops import lattice_ops
from tensorflow_lattice.python.ops.gen_monotone_lattice import monotone_lattice

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope

_VALID_INTERPOLATION_TYPES = ['hypercube', 'simplex']


def lattice_param_as_linear(lattice_sizes, output_dim, linear_weights=1.0):
  """Returns lattice parameter that represents a normalized linear function.

  For simplicity, let's assume output_dim == 1 (when output_dim > 1 you get
  output_dim lattices one for each linear function). This function returns a
  lattice parameter so that

    lattice_param' * phi(x) = 1 / len(lattice_sizes) *
      (sum_k x[k] * linear_weights[k]/(lattice_sizes[k] - 1) + bias)

  where phi(x) is the lattice interpolation weight and
  bias = -sum_k linear_weights[k] / 2.

  The normalization in the weights and the bias term make the output lie in the
  range [-0.5, 0.5], when every member of linear_weights is 1.0.
  In addition, the bias term makes the expected value zero when x[k] is from the
  uniform distribution over [0, lattice_sizes[k] - 1].

  The returned lattice_param can be used to initialize a lattice layer as a
  linear function.

  Args:
    lattice_sizes: (list of ints) A list of lattice sizes of each dimension.
    output_dim: (int) number of outputs.
    linear_weights: (float, list of floats, list of list of floats) linear
      function's weight terms. linear_weights[k][n] == kth output's nth weight.
      If float, then all the weights uses one value as
      [[linear_weights] * len(lattice_sizes)] * output_dim.
      If list of floats, then the len(linear_weights) == len(lattice_sizes) is
      expected, and the weights are [linear_weights] * output_dim, i.e., all
      output_dimension will get same linear_weights.
  Returns:
    List of list of floats with size (output_dim, number_of_lattice_param).
  Raises:
    ValueError: * Any element in lattice_sizes is less than 2.
      * lattice_sizes is empty.
      * If linear_weights is not supported type, or shape of linear_weights are
        not the desired values .
  """
  if not lattice_sizes:
    raise ValueError('lattice_sizes should not be empty')
  for lattice_size in lattice_sizes:
    if lattice_size < 2:
      raise ValueError('All elements in lattice_sizes are expected to greater '
                       'than equal to 2, but got %s' % lattice_sizes)

  lattice_rank = len(lattice_sizes)
  linear_weight_matrix = None
  if isinstance(linear_weights, float):
    linear_weight_matrix = [[linear_weights] * lattice_rank] * output_dim
  elif isinstance(linear_weights, list):
    # Branching using the first element in linear_weights. linear_weights[0]
    # should exist, since lattice_sizes is not empty.
    if isinstance(linear_weights[0], float):
      if len(linear_weights) != lattice_rank:
        raise ValueError(
            'A number of elements in linear_weights (%d) != lattice rank (%d)' %
            (len(linear_weights), lattice_rank))
      # Repeating same weights for all output_dim.
      linear_weight_matrix = [linear_weights] * output_dim
    elif isinstance(linear_weights[0], list):
      # 2d matrix case.
      if len(linear_weights) != output_dim:
        raise ValueError(
            'A number of lists in linear_weights (%d) != output_dim (%d)' %
            (len(linear_weights), output_dim))
      for linear_weight in linear_weights:
        if len(linear_weight) != lattice_rank:
          raise ValueError(
              'linear_weights contain more than one list whose length != '
              'lattice rank(%d)' % lattice_rank)
      linear_weight_matrix = linear_weights
    else:
      raise ValueError(
          'Only list of float or list of list of floats are supported')
  else:
    raise ValueError(
        'Only float or list of float or list of list of floats are supported.')

  # Create lattice structure to enumerate (index, lattice_dim) pairs.
  lattice_structure = tools.LatticeStructure(lattice_sizes)

  # Normalize linear_weight_matrix.
  lattice_parameters = []
  for linear_weight_per_output in linear_weight_matrix:
    sum_of_weights = 0.0
    for weight in linear_weight_per_output:
      sum_of_weights += weight
    sum_of_weights /= (2.0 * lattice_rank)
    lattice_parameter = [-sum_of_weights] * lattice_structure.num_vertices
    for (idx, vertex) in tools.lattice_indices_generator(lattice_structure):
      for dim in range(lattice_rank):
        lattice_parameter[idx] += (
            linear_weight_per_output[dim] * float(vertex[dim]) / float(
                lattice_rank * (lattice_sizes[dim] - 1)))
    lattice_parameters.append(lattice_parameter)

  return lattice_parameters


def lattice_layer(input_tensor,
                  lattice_sizes,
                  is_monotone=None,
                  output_min=None,
                  output_max=None,
                  output_dim=1,
                  interpolation_type='hypercube',
                  lattice_initializer=None,
                  **regularizer_amounts):
  """Creates a lattice layer.

  Returns an output of lattice, lattice parameters, and projection ops.

  Args:
    input_tensor: [batch_size, input_dim] tensor.
    lattice_sizes: A list of lattice sizes of each dimension.
    is_monotone: A list of input_dim booleans, boolean or None. If None or
      False, lattice will not have monotonicity constraints. If
      is_monotone[k] == True, then the lattice output has the non-decreasing
      monotonicity with respect to input_tensor[?, k] (the kth coordinate). If
      True, all the input coordinate will have the non-decreasing monotonicity.
    output_min: Optional output lower bound.
    output_max: Optional output upper bound.
    output_dim: Number of outputs.
    interpolation_type: 'hypercube' or 'simplex'.
    lattice_initializer: (Optional) Initializer for lattice parameter vectors,
      a 2D tensor [output_dim, parameter_dim] (where parameter_dim ==
      lattice_sizes[0] * ... * lattice_sizes[input_dim - 1]). If None,
      lattice_param_as_linear initializer will be used with
      linear_weights=[1] * len(lattice_sizes).
    **regularizer_amounts: Keyword args of regularization amounts passed to
      regularizers.lattice_regularization(). Keyword names should be among
      regularizers.LATTICE_ONE_DIMENSIONAL_REGULARIZERS or
      regularizers.LATTICE_MULTI_DIMENSIONAL_REGULARIZERS. For
      multi-dimensional regularizers the value should be float. For
      one-dimensional regularizers the values should be float or list of floats.
      If a single float value is provided, then all dimensions will get the same
      value.

  Returns:
    A tuple of:
    * output tensor of shape [batch_size, output_dim]
    * parameter tensor of shape [output_dim, parameter_dim]
    * None or projection ops, that must be applied at each
      step (or every so many steps) to project the model to a feasible space:
      used for bounding the outputs or for imposing monotonicity.
    * None or a regularization loss, if regularization is configured.

  Raises:
    ValueError: for invalid parameters.
  """
  if interpolation_type not in _VALID_INTERPOLATION_TYPES:
    raise ValueError('interpolation_type should be one of {}'.format(
        _VALID_INTERPOLATION_TYPES))

  if lattice_initializer is None:
    linear_weights = [1.0] * len(lattice_sizes)
    lattice_initializer = lattice_param_as_linear(
        lattice_sizes, output_dim, linear_weights=linear_weights)

  parameter_tensor = variable_scope.get_variable(
      interpolation_type + '_lattice_parameters',
      initializer=lattice_initializer)

  output_tensor = lattice_ops.lattice(
      input_tensor,
      parameter_tensor,
      lattice_sizes,
      interpolation_type=interpolation_type)

  with ops.name_scope('lattice_monotonic_projection'):
    if is_monotone or output_min or output_max:
      projected_parameter_tensor = parameter_tensor
      if is_monotone:
        is_monotone = tools.cast_to_list(is_monotone, len(lattice_sizes),
                                         'is_monotone')
        projected_parameter_tensor = monotone_lattice(
            projected_parameter_tensor,
            lattice_sizes=lattice_sizes,
            is_monotone=is_monotone)

      if output_min:
        projected_parameter_tensor = math_ops.maximum(
            projected_parameter_tensor, output_min)

      if output_min:
        projected_parameter_tensor = math_ops.minimum(
            projected_parameter_tensor, output_max)

      delta = projected_parameter_tensor - parameter_tensor
      projection_ops = [parameter_tensor.assign_add(delta)]
    else:
      projection_ops = None

  with ops.name_scope('lattice_regularization'):
    reg = regularizers.lattice_regularization(parameter_tensor, lattice_sizes,
                                              **regularizer_amounts)

  return (output_tensor, parameter_tensor, projection_ops, reg)


def ensemble_lattices_layer(input_tensor,
                            lattice_sizes,
                            structure_indices,
                            is_monotone=None,
                            output_dim=1,
                            interpolation_type='hypercube',
                            lattice_initializers=None,
                            **regularizer_amounts):
  """Creates a ensemble of lattices layer.

  Returns a list of output of lattices, lattice parameters, and projection ops.

  Args:
    input_tensor: [batch_size, input_dim] tensor.
    lattice_sizes: A list of lattice sizes of each dimension.
    structure_indices: A list of list of ints. structure_indices[k] is a list
    of indices that belongs to kth lattices.
    is_monotone: A list of input_dim booleans, boolean or None. If None or
      False, lattice will not have monotonicity constraints. If
      is_monotone[k] == True, then the lattice output has the non-decreasing
      monotonicity with respect to input_tensor[?, k] (the kth coordinate). If
      True, all the input coordinate will have the non-decreasing monotonicity.
    output_dim: Number of outputs.
    interpolation_type: 'hypercube' or 'simplex'.
    lattice_initializers: (Optional) A list of initializer for each lattice
      parameter vectors. lattice_initializer[k] is a 2D tensor
      [output_dim, parameter_dim[k]], where parameter_dim[k] is the number of
      parameter in the kth lattice. If None, lattice_param_as_linear initializer
      will be used with
      linear_weights=[1 if monotone else 0 for monotone in is_monotone].
    **regularizer_amounts: Keyword args of regularization amounts passed to
      regularizers.lattice_regularization(). Keyword names should be among
      regularizers.LATTICE_ONE_DIMENSIONAL_REGULARIZERS or
      regularizers.LATTICE_MULTI_DIMENSIONAL_REGULARIZERS. For
      multi-dimensional regularizers the value should be float. For
      one-dimensional regularizers the values should be float or list of floats.
      If a single float value is provided, then all dimensions will get the same
      value.

  Returns:
    A tuple of:
    * a list of output tensors, [batch_size, output_dim], with length
      len(structure_indices), i.e., one for each lattice.
    * a list of parameter tensors shape [output_dim, parameter_dim]
    * None or projection ops, that must be applied at each
      step (or every so many steps) to project the model to a feasible space:
      used for bounding the outputs or for imposing monotonicity.
    * None or a regularization loss, if regularization is configured.
  """
  num_lattices = len(structure_indices)
  lattice_initializers = tools.cast_to_list(lattice_initializers, num_lattices,
                                            'lattice initializers')
  one_dimensional_regularizers = \
    regularizers.LATTICE_ONE_DIMENSIONAL_REGULARIZERS
  for regularizer_name in regularizer_amounts:
    if regularizer_name in one_dimensional_regularizers:
      regularizer_amounts[regularizer_name] = tools.cast_to_list(
          regularizer_amounts[regularizer_name], len(lattice_sizes),
          regularizer_name)

  # input_slices[k] = input_tensor[:, k].
  input_slices = array_ops.unstack(input_tensor, axis=1)

  output_tensors = []
  param_tensors = []
  projections = []
  regularization = None
  if is_monotone:
    is_monotone = tools.cast_to_list(is_monotone, len(lattice_sizes),
                                     'is_monotone')
  # Now iterate through structure_indices to construct lattices.
  get_indices = lambda indices, iterable: [iterable[index] for index in indices]
  for (cnt, structure) in enumerate(structure_indices):
    with variable_scope.variable_scope('lattice_%d' % cnt):
      sub = functools.partial(get_indices, structure)
      sub_lattice_sizes = sub(lattice_sizes)
      sub_is_monotone = None
      if is_monotone:
        sub_is_monotone = sub(is_monotone)

      sub_input_tensor_list = sub(input_slices)
      sub_input_tensor = array_ops.stack(sub_input_tensor_list, axis=1)

      sub_regularizer_amounts = {}
      for regularizer_name in regularizer_amounts:
        if regularizer_name in one_dimensional_regularizers:
          sub_regularizer_amounts[regularizer_name] = sub(
              regularizer_amounts[regularizer_name])
        else:
          sub_regularizer_amounts[regularizer_name] = regularizer_amounts[
              regularizer_name]

      packed_results = lattice_layer(
          sub_input_tensor,
          sub_lattice_sizes,
          sub_is_monotone,
          output_dim=output_dim,
          interpolation_type=interpolation_type,
          lattice_initializer=lattice_initializers[cnt],
          **sub_regularizer_amounts)
      (sub_output, sub_param, sub_proj, sub_reg) = packed_results

      output_tensors.append(sub_output)
      param_tensors.append(sub_param)
      if sub_proj:
        projections += sub_proj
      regularization = tools.add_if_not_none(regularization, sub_reg)

  return (output_tensors, param_tensors, projections, regularization)
