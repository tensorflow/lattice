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
"""Lattice interpolation and gradient ops."""
# pylint: disable=unused-import
from tensorflow_lattice.python.ops.gen_lattice_interpolation import hypercube_gradient
from tensorflow_lattice.python.ops.gen_lattice_interpolation import hypercube_interpolation
from tensorflow_lattice.python.ops.gen_lattice_interpolation import simplex_gradient
from tensorflow_lattice.python.ops.gen_lattice_interpolation import simplex_interpolation
# pylint: enable=unused-import

from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import resource_loader

_lattice_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('../../cc/ops/_lattice_ops.so'))


@ops.RegisterGradient('HypercubeInterpolation')
def _hypercube_gradient(op, grad_wrt_weight):
  """Register gradient for HypercubeInterpolationOp."""
  grad_wrt_input = hypercube_gradient(
      input=op.inputs[0],
      weight=op.outputs[0],
      grad_wrt_weight=grad_wrt_weight,
      lattice_sizes=op.get_attr('lattice_sizes'))
  return [grad_wrt_input]


@ops.RegisterGradient('SimplexInterpolation')
def _simplex_gradient(op, grad_wrt_weight):
  """Register gradient for SimplexInterpolationOp."""
  grad_wrt_input = simplex_gradient(
      input=op.inputs[0],
      weight=op.outputs[0],
      grad_wrt_weight=grad_wrt_weight,
      lattice_sizes=op.get_attr('lattice_sizes'))
  return [grad_wrt_input]


def lattice(input_tensor,
            parameter_tensor,
            lattice_sizes,
            interpolation_type='hypercube'):
  """Returns an interpolated look-up table (lattice) op.

  Args:
    input_tensor: [batch_size, input_dim] tensor.
    parameter_tensor: [output_dim, param_dim] tensor, where param_dim ==
      lattice_sizes[0] * ... * lattice_sizes[input_dim - 1].
    lattice_sizes: A list of lattice sizes of each dimension.
    interpolation_type: 'hypercube' or 'simplex'.

  Returns:
    output_tensor: [batch_size, num_outputs] tensor that contains the output of
    hypercube lattice.

  Raises:
    ValueError: If interpolation_type is not 'hypercube' nor 'simplex'.


  """
  if interpolation_type not in ['hypercube', 'simplex']:
    raise ValueError("interpolation_type should be 'hypercube' or 'simplex'")

  if interpolation_type == 'hypercube':
    interpolation_weights = hypercube_interpolation(
        input_tensor, lattice_sizes=lattice_sizes)
  else:
    interpolation_weights = simplex_interpolation(
        input_tensor, lattice_sizes=lattice_sizes)

  # Now the dimension is [batch_size, num_outputs].

  output_tensor = math_ops.matmul(
      interpolation_weights, parameter_tensor, transpose_b=True)

  return output_tensor
