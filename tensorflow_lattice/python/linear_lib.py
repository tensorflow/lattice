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
"""Implementation of algorithms required for Linear layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import internal_utils
from . import utils
import six
import tensorflow as tf

_NORMALIZATION_EPS = 1e-8


def project(weights,
            monotonicities,
            monotonic_dominances=None,
            range_dominances=None,
            input_min=None,
            input_max=None,
            normalization_order=None):
  """Applies constraints to weights.

  Args:
    weights: Tensor which represents weights of TFL linear layer. Must have
      shape [len(monotonicities), units].
    monotonicities: List or tuple of same length as number of elements in
      'weights' of {-1, 0, 1} which represent monotonicity constraints per
      dimension. -1 stands for decreasing, 0 for no constraints, 1 for
      increasing.
    monotonic_dominances: List of two-element tuples. First element is the index
      of the dominant feature. Second element is the index of the weak feature.
    range_dominances: List of two-element tuples. First element is the index of
      the dominant feature. Second element is the index of the weak feature.
    input_min: List or tuple of length same length as number of elements in
      'weights' of either None or float to compute input range for range
      dominance projection.
    input_max: List or tuple of length same length as number of elements in
      'weights' of either None or float to compute input range for range
      dominance projection.
    normalization_order: If specified weights will be adjusted to have norm 1.
      Norm will be computed by: `tf.norm(tensor, ord=normalization_order)`.

  Raises:
    ValueError: If shape of weights is not `(len(monotonicities), units)`.

  Returns:
    'weights' with monotonicity constraints and normalization applied to it.
  """
  verify_hyperparameters(
      weights_shape=weights.shape,
      monotonicities=monotonicities,
      monotonic_dominances=monotonic_dominances,
      range_dominances=range_dominances,
      input_min=input_min,
      input_max=input_max)
  if any(monotonicities):
    if 1 in monotonicities:
      inverted_increasing_mask = tf.constant(
          value=[0.0 if m == 1 else 1.0 for m in monotonicities],
          dtype=weights.dtype,
          shape=(weights.shape[0], 1))
      # Multiplying by this mask will keep non monotonic dims same and will
      # set monotonic dims to 0.0. Later by taking maximum with this product
      # we'll essentially take maximumum of monotonic dims with 0.0.
      weights = tf.maximum(weights, weights * inverted_increasing_mask)

    if -1 in monotonicities:
      inverted_decreasing_mask = tf.constant(
          value=[0.0 if m == -1 else 1.0 for m in monotonicities],
          dtype=weights.dtype,
          shape=(weights.shape[0], 1))
      weights = tf.minimum(weights, weights * inverted_decreasing_mask)

  if monotonic_dominances:
    monotonic_dominances = [(j, i) for i, j in monotonic_dominances]
    weights = internal_utils.approximately_project_categorical_partial_monotonicities(
        weights, monotonic_dominances)

  if range_dominances:
    range_dominances = [(j, i) for i, j in range_dominances]
    scalings = [-1.0 if m == -1 else 1.0 for m in monotonicities]
    for dim, (lower, upper) in enumerate(zip(input_min, input_max)):
      if lower is not None and upper is not None:
        scalings[dim] *= upper - lower
    scalings = tf.constant(
        scalings, dtype=weights.dtype, shape=(weights.shape[0], 1))
    weights *= scalings
    weights = internal_utils.approximately_project_categorical_partial_monotonicities(
        weights, range_dominances)
    weights /= scalings

  if normalization_order:
    norm = tf.norm(weights, axis=0, ord=normalization_order)
    norm = tf.where(norm < _NORMALIZATION_EPS, 1.0, norm)
    weights = weights / norm

  return weights


def assert_constraints(weights,
                       monotonicities,
                       monotonic_dominances,
                       range_dominances,
                       input_min,
                       input_max,
                       normalization_order,
                       eps=1e-4):
  """Asserts that weights satisfy constraints.

  Args:
    weights: Weights of Linear layer.
    monotonicities: List or tuple of same length as number of elements in
      'weights' of {-1, 0, 1} which represent monotonicity constraints per
      dimension. -1 stands for decreasing, 0 for no constraints, 1 for
      increasing.
    monotonic_dominances: List of two-element tuple. First element is the index
      of the dominant feature. Second element is the index of the weak feature.
    range_dominances: List of two-element tuples. First element is the index of
      the dominant feature. Second element is the index of the weak feature.
    input_min: List or tuple of length same length as number of elements in
      'weights' of either None or float which specifies the minimum value to
      clip by.
    input_max: List or tuple of length same length as number of elements in
      'weights' of either None or float which specifies the maximum value to
      clip by.
    normalization_order: Whether weights have to have norm 1. Norm will be
      computed by: `tf.norm(tensor, ord=normalization_order)`.
    eps: Allowed constraints violation.

  Returns:
    List of assetion ops in graph mode or directly executes assertions in eager
    mode.
  """
  asserts = []
  if any(monotonicities):
    # Create constant specifying shape explicitly because otherwise due to
    # weights shape ending with dimesion of size 1 broadcasting will hurt us.
    monotonicities_constant = tf.constant(
        monotonicities, shape=(weights.shape[0], 1), dtype=weights.dtype)
    diff = tf.reduce_min(weights * monotonicities_constant)
    asserts.append(
        tf.Assert(
            diff >= -eps,
            data=[
                "Monotonicity violation", "Monotonicities:", monotonicities,
                "Min monotonicity diff:", diff, "Epsilon:", eps, "Weights:",
                weights
            ],
            summarize=weights.shape[0]))

  for dominant_dim, weak_dim in monotonic_dominances or []:
    diff = tf.reduce_min(weights[dominant_dim] - weights[weak_dim])
    asserts.append(
        tf.Assert(
            diff >= -eps,
            data=[
                "Monotonic dominance violation", "Dominant dim:", dominant_dim,
                "Weak dim:", weak_dim, "Epsilon:", eps, "Weights:", weights
            ],
            summarize=weights.shape[0]))

  if range_dominances:
    scalings = [-1.0 if m == -1 else 1.0 for m in monotonicities]
    for dim, (lower, upper) in enumerate(zip(input_min, input_max)):
      if lower is not None and upper is not None:
        scalings[dim] *= upper - lower
    for dominant_dim, weak_dim in range_dominances:
      diff = tf.reduce_min(scalings[dominant_dim] * weights[dominant_dim] -
                           scalings[weak_dim] * weights[weak_dim])
      asserts.append(
          tf.Assert(
              diff >= -eps,
              data=[
                  "Range dominance violation", "Dominant dim:", dominant_dim,
                  "Weak dim:", weak_dim, "Epsilon:", eps, "Weights:", weights,
                  "Scalings:", scalings
              ],
              summarize=weights.shape[0]))

  if normalization_order:
    norm = tf.norm(weights, axis=0, ord=normalization_order)
    asserts.append(
        # Norm can be either 0.0 or 1.0, because if all weights are close to 0.0
        # we can't scale them to get norm 1.0.
        tf.Assert(
            tf.logical_or(
                tf.abs(norm - 1.0) < eps,
                tf.abs(norm) < _NORMALIZATION_EPS),
            data=[
                "Normalization order violation", "Norm:", norm, "Epsilon:", eps,
                "Weights:", weights
            ],
            summarize=weights.shape[0]))
  return asserts


def verify_hyperparameters(num_input_dims=None,
                           units=None,
                           input_shape=None,
                           monotonicities=None,
                           monotonic_dominances=None,
                           range_dominances=None,
                           input_min=None,
                           input_max=None,
                           weights_shape=None):
  """Verifies that all given hyperparameters are consistent.

  This function does not inspect weights themselves. Only their shape. Use
  `assert_constraints()` to assert actual weights against constraints.

  Unlike linear layer itself this function requires monotonicites to be
  specified via list or tuple rather than via single element because that's how
  monotonicites are stored internaly.

  See `tfl.layers.Linear` Layer class level comment for detailed description of
  arguments.

  Args:
    num_input_dims: None or number of input dimensions.
    units: Units hyperparameter of Linear layer.
    input_shape: Shape of layer input.
    monotonicities: List or tuple of same length as number of elements in
      `weights` of {-1, 0, 1} which represent monotonicity constraints per
      dimension. -1 stands for decreasing, 0 for no constraints, 1 for
      increasing.
    monotonic_dominances: List of two-element tuples. First element is the index
      of the dominant feature. Second element is the index of the weak feature.
    range_dominances: List of two-element tuples. First element is the index of
      the dominant feature. Second element is the index of the weak feature.
    input_min: List or tuple of length same length as number of elements in
      'weights' of either None or float which specifies the minimum value to
      clip by.
    input_max: List or tuple of length same length as number of elements in
      'weights' of either None or float which specifies the maximum value to
      clip by.
    weights_shape: None or shape of tensor which represents weights of Linear
      layer.

  Raises:
    ValueError: If something is inconsistent.
  """
  # It also raises errors if monotonicities specified incorrectly.
  monotonicities = utils.canonicalize_monotonicities(monotonicities)
  input_min = utils.canonicalize_input_bounds(input_min)
  input_max = utils.canonicalize_input_bounds(input_max)

  if monotonicities is not None and num_input_dims is not None:
    if len(monotonicities) != num_input_dims:
      raise ValueError("Number of elements in 'monotonicities' must be equal to"
                       " num_input_dims. monotoniticites: %s, "
                       "len(monotonicities): %d, num_input_dims: %d" %
                       (monotonicities, len(monotonicities), num_input_dims))

  if weights_shape is not None:
    if len(weights_shape) != 2:
      raise ValueError("Expect weights to be a rank 2 tensor. Weights shape: "
                       "%s" % (weights_shape,))
    if monotonicities is not None and weights_shape[0] != len(monotonicities):
      raise ValueError("Number of elements in 'monotonicities' does not "
                       "correspond to number of weights. Weights shape: %s, "
                       "monotonicities: %s" % (weights_shape, monotonicities))
    if input_min is not None and weights_shape[0] != len(input_min):
      raise ValueError(
          "Number of elements in 'input_min' does not correspond "
          "to number of weights. Weights shape: %s, input_min: %s" %
          (weights_shape, input_min))
    if input_max is not None and weights_shape[0] != len(input_max):
      raise ValueError(
          "Number of elements in 'input_max' does not correspond "
          "to number of weights. Weights shape: %s, input_max: %s" %
          (weights_shape, input_max))

  if input_shape is not None:
    assert units is not None and num_input_dims is not None
    if (units > 1 and
        (len(input_shape) != 3 or input_shape[1] != units or
         input_shape[2] != num_input_dims)):
      raise ValueError("'input_shape' must be of rank three and number of "
                       "elements of second and third dimensions must be "
                       "equal to 'units' and 'num_input_dims' respectively. "
                       "'input_shape': " + str(input_shape) + "'units': " +
                       str(units) + "'num_input_dims': " + str(num_input_dims))
    elif (units == 1 and
          (len(input_shape) != 2 or input_shape[1] != num_input_dims)):
      raise ValueError("'input_shape' must be of rank two and number of "
                       "elements of second dimension must be equal to "
                       "'num_input_dims'. 'input_shape': " + str(input_shape) +
                       "'num_input_dims': " + str(num_input_dims))

  for dim, (lower, upper) in enumerate(zip(input_min or [], input_max or [])):
    if lower is not None and upper is not None and lower > upper:
      raise ValueError("Cannot have 'input_min' greater than 'input_max'."
                       "Dimension: %d, input_min[%d]: %f, input_max[%d]: %f" %
                       (dim, dim, input_min[dim], dim, input_max[dim]))

  if monotonic_dominances is not None:
    assert monotonicities is not None
    num_input_dims = len(monotonicities)
    dim_pairs = set()
    for constraint in monotonic_dominances:
      if len(constraint) != 2:
        raise ValueError("Monotonic dominance constraints must consist of 2 "
                         "elements. Seeing constraint tuple %s" % (constraint,))
      dominant_dim, weak_dim = constraint
      if (dominant_dim >= num_input_dims or weak_dim >= num_input_dims or
          dominant_dim < 0 or weak_dim < 0):
        raise ValueError("Dimensions constrained by monotonic dominance "
                         "constraints are not within the input dimensions. "
                         "'dims': %s, %s, num_dims: %s" %
                         (dominant_dim, weak_dim, num_input_dims))
      if not isinstance(dominant_dim, int) or not isinstance(weak_dim, int):
        raise ValueError("Monotonic dominance constraint dimensions must be "
                         "integers. Seeing dominant_dim %s and weak_dim %s" %
                         (dominant_dim, weak_dim))
      for dim in [dominant_dim, weak_dim]:
        if monotonicities[dim] != 1:
          raise ValueError("Monotonic dominance constraint's dimensions must "
                           "be monotonic. Dimension %d is not monotonic." %
                           (dim))
      if (weak_dim, dominant_dim) in dim_pairs:
        raise ValueError("Cannot have two monotonic dominance constraints on "
                         "the same pair of features conflicting. Features: %d, "
                         "%d" % (dominant_dim, weak_dim))
      dim_pairs.add((dominant_dim, weak_dim))

  if range_dominances is not None:
    assert monotonicities is not None
    num_input_dims = len(monotonicities)
    dim_pairs = set()
    for constraint in range_dominances:
      if len(constraint) != 2:
        raise ValueError("Range dominance constraints must consist of 2 "
                         "elements. Seeing constraint tuple %s" % (constraint,))
      dominant_dim, weak_dim = constraint
      if (dominant_dim >= num_input_dims or weak_dim >= num_input_dims or
          dominant_dim < 0 or weak_dim < 0):
        raise ValueError("Dimensions constrained by range dominance "
                         "constraints are not within the input dimensions. "
                         "'dims': %s, %s, num_dims: %s" %
                         (dominant_dim, weak_dim, num_input_dims))
      if not isinstance(dominant_dim, int) or not isinstance(weak_dim, int):
        raise ValueError("Range dominance constraint dimensions must be "
                         "integers. Seeing dominant_dim %s and weak_dim %s" %
                         (dominant_dim, weak_dim))
      if (monotonicities[dominant_dim] != monotonicities[weak_dim] or
          monotonicities[dominant_dim] == 0):
        raise ValueError("Range dominance constraint's dimensions must have "
                         "the same direction of monotonicity. Dimension %d is "
                         "%d. Dimension %d is %d." %
                         (dominant_dim, monotonicities[dominant_dim], weak_dim,
                          monotonicities[weak_dim]))
      for dim in [dominant_dim, weak_dim]:
        if input_min is None or input_min[dim] is None:
          raise ValueError("Range dominance constraint's dimensions must "
                           "have `input_min` set. Dimension %d is not set." %
                           (dim))
        if input_max is None or input_max[dim] is None:
          raise ValueError("Range dominance constraint's dimensions must "
                           "have `input_max` set. Dimension %d is not set." %
                           (dim))
      if (weak_dim, dominant_dim) in dim_pairs:
        raise ValueError("Cannot have two range dominance constraints on the "
                         "same pair of features conflicting. Features: %d, %d" %
                         (dominant_dim, weak_dim))
      dim_pairs.add((dominant_dim, weak_dim))

  if range_dominances is not None and monotonic_dominances is not None:
    monotonic_dominance_dims = set()
    for dims in monotonic_dominances:
      for dim in dims:
        monotonic_dominance_dims.add(dim)
    for dims in range_dominances:
      for dim in dims:
        if dim in monotonic_dominance_dims:
          raise ValueError("Cannot have both monotonic and range dominance "
                           "constraints specified on the same dimension. "
                           "Dimension %d is set by both." % (dim))
