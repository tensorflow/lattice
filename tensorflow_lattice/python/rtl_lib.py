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
"""Implementation of algorithms required for RTL layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six


def verify_hyperparameters(lattice_size,
                           input_shape=None,
                           output_min=None,
                           output_max=None,
                           interpolation="hypercube",
                           parameterization="all_vertices",
                           kernel_initializer=None,
                           kernel_regularizer=None):
  """Verifies that all given hyperparameters are consistent.

  See `tfl.layers.RTL` class level comment for detailed description of
  arguments.

  Args:
    lattice_size: Lattice size to check againts.
    input_shape: Shape of layer input.
    output_min: Minimum output of `RTL` layer.
    output_max: Maximum output of `RTL` layer.
    interpolation: One of 'simplex' or 'hypercube' interpolation.
    parameterization: One of 'all_vertices' or 'kronecker_factored'
      parameterizations.
    kernel_initializer: Initizlier to check against.
    kernel_regularizer: Regularizers to check against.

  Raises:
    ValueError: If lattice_size < 2.
    KeyError: If input_shape is a dict with incorrect keys.
    ValueError: If output_min >= output_max.
    ValueError: If interpolation is not one of 'simplex' or 'hypercube'.
    ValueError: If parameterization is 'kronecker_factored' and
      kernel_initializer is 'linear_initializer'.
    ValueError: If parameterization is 'kronecker_factored' and
      kernel_regularizer is not None.
    ValueError: If kernel_regularizer contains a tuple with len != 3.
    ValueError: If kernel_regularizer contains a tuple with non-float l1 value.
    ValueError: If kernel_regularizer contains a tuple with non-flaot l2 value.

  """
  if lattice_size < 2:
    raise ValueError(
        "Lattice size must be at least 2. Given: {}".format(lattice_size))

  if input_shape:
    if isinstance(input_shape, dict):
      for key in input_shape:
        if key not in ["unconstrained", "increasing"]:
          raise KeyError("Input shape keys should be either 'unconstrained' "
                         "or 'increasing', but seeing: {}".format(key))

  if output_min is not None and output_max is not None:
    if output_min >= output_max:
      raise ValueError("'output_min' must be not greater than 'output_max'. "
                       "'output_min': %f, 'output_max': %f" %
                       (output_min, output_max))

  if interpolation not in ["hypercube", "simplex"]:
    raise ValueError("RTL interpolation type should be either 'simplex' "
                     "or 'hypercube': %s" % interpolation)

  if (parameterization == "kronecker_factored" and
      kernel_initializer == "linear_initializer"):
    raise ValueError("'kronecker_factored' parameterization does not currently "
                     "support linear iniitalization. 'parameterization': %s, "
                     "'kernel_initializer': %s" %
                     (parameterization, kernel_initializer))

  if (parameterization == "kronecker_factored" and
      kernel_regularizer is not None):
    raise ValueError("'kronecker_factored' parameterization does not currently "
                     "support regularization. 'parameterization': %s, "
                     "'kernel_regularizer': %s" %
                     (parameterization, kernel_regularizer))

  if kernel_regularizer:
    if isinstance(kernel_regularizer, list):
      regularizers = kernel_regularizer
      if isinstance(kernel_regularizer[0], six.string_types):
        regularizers = [kernel_regularizer]
      for regularizer in regularizers:
        if len(regularizer) != 3:
          raise ValueError("Regularizer tuples/lists must have three elements "
                           "(type, l1, and l2). Given: {}".format(regularizer))
        _, l1, l2 = regularizer
        if not isinstance(l1, float):
          raise ValueError(
              "Regularizer l1 must be a single float. Given: {}".format(
                  type(l1)))
        if not isinstance(l2, float):
          raise ValueError(
              "Regularizer l2 must be a single float. Given: {}".format(
                  type(l2)))
