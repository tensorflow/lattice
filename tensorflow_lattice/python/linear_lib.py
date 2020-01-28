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

from . import utils
import six
import tensorflow as tf

_NORMALIZATION_EPS = 1e-8


def project(weights, monotonicities, monotonic_dominances=None,
            normalization_order=None):
  """Applies constraints to weights.

  Args:
    weights: Tensor which represents weights of TFL linear layer. Must have
      shape [len(monotonicities), 1].
    monotonicities: List or tuple of same length as number of elements in
      'weights' of {-1, 0, 1} which represent monotonicity constraints per
      dimension. -1 stands for decreasing, 0 for no constraints, 1 for
      increasing.
    monotonic_dominances: List of two-element tuples. First element is the index
      of the dominant feature. Second element is the index of the weak feature.
    normalization_order: If specified weights will be adjusted to have norm 1.
      Norm will be computed by: `tf.norm(tensor, ord=normalization_order)`.

  Raises:
    ValueError: If shape of weights is not `(len(monotonicities), 1)`.

  Returns:
    'weights' with monotonicity constraints and normalization applied to it.
  """
  verify_hyperparameters(weights_shape=weights.shape,
                         monotonicities=monotonicities,
                         monotonic_dominances=monotonic_dominances)
  if any(monotonicities):
    if 1 in monotonicities:
      inverted_increasing_mask = tf.constant(
          value=[0.0 if m == 1 else 1.0 for m in monotonicities],
          dtype=weights.dtype,
          shape=weights.shape)
      # Multiplying by this mask will keep non monotonic dims same and will
      # set monotonic dims to 0.0. Later by taking maximum with this product
      # we'll essentially take maximumum of monotonic dims with 0.0.
      weights = tf.maximum(weights, weights * inverted_increasing_mask)

    if -1 in monotonicities:
      inverted_decreasing_mask = tf.constant(
          value=[0.0 if m == -1 else 1.0 for m in monotonicities],
          dtype=weights.dtype,
          shape=weights.shape)
      weights = tf.minimum(weights, weights * inverted_decreasing_mask)

  if monotonic_dominances:
    monotonic_dominances = [(j, i) for i, j in monotonic_dominances]
    weights = utils.approximately_project_categorical_partial_monotonicities(
        weights, monotonic_dominances)

  if normalization_order:
    norm = tf.norm(weights, ord=normalization_order)
    weights = tf.cond(norm < _NORMALIZATION_EPS,
                      true_fn=lambda: weights,
                      false_fn=lambda: weights / norm)

  return weights


def assert_constraints(weights, monotonicities, monotonic_dominances,
                       normalization_order, eps=1e-4):
  """Asserts that weights satisfy constraints.

  Args:
    weights: Weights of Linear layer.
    monotonicities: List or tuple of same length as number of elements in
      'weights' of {-1, 0, 1} which represent monotonicity constraints per
      dimension. -1 stands for decreasing, 0 for no constraints, 1 for
      increasing.
    monotonic_dominances: List of two-element tuple. First element is the index
      of the dominant feature. Second element is the index of the weak feature.
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
    monotonicities_constant = tf.constant(monotonicities,
                                          shape=weights.shape,
                                          dtype=weights.dtype)
    diff = tf.reduce_min(weights * monotonicities_constant)
    asserts.append(
        tf.Assert(diff >= -eps,
                  data=["Monotonicity violation",
                        "Monotonicities:", monotonicities,
                        "Min monotonicity diff:", diff,
                        "Epsilon:", eps,
                        "Weights:", weights],
                  summarize=weights.shape[0]))

  for dominant_dim, weak_dim in monotonic_dominances or []:
    diff = tf.reduce_min(weights[dominant_dim] - weights[weak_dim])
    asserts.append(
        tf.Assert(diff >= -eps,
                  data=["Monotonic dominance violation",
                        "Dominant dim:", dominant_dim,
                        "Weak dim:", weak_dim,
                        "Epsilon:", eps,
                        "Weights:", weights],
                  summarize=weights.shape[0]))

  if normalization_order:
    norm = tf.norm(weights, ord=normalization_order)
    asserts.append(
        # Norm can be either 0.0 or 1.0, because if all weights are close to 0.0
        # we can't scale them to get norm 1.0.
        tf.Assert(tf.logical_or(tf.abs(norm - 1.0) < eps,
                                tf.abs(norm) < _NORMALIZATION_EPS),
                  data=["Normalization order violation",
                        "Norm:", norm,
                        "Epsilon:", eps,
                        "Weights:", weights],
                  summarize=weights.shape[0]))
  return asserts


def verify_hyperparameters(num_input_dims=None,
                           monotonicities=None,
                           monotonic_dominances=None,
                           weights_shape=None):
  """Verifies that all given hyperparameters are consistent.

  This function does not inspect weights themselves. Only their shape. Use
  `assert_constraints()` to assert actual weights against constraints.

  Unlike linear layer itself this function requires monotonicites to be
  specified via list or tuple rather than via single element because that's how
  monotonicites are stored internaly.

  See `tfl.linear_layer.Linear` Layer class level comment for detailed
  description of arguments.

  Args:
    num_input_dims: None or number of input dimensions.
    monotonicities: List or tuple of same length as number of elements in
      `weights` of {-1, 0, 1} which represent monotonicity constraints per
      dimension. -1 stands for decreasing, 0 for no constraints, 1 for
      increasing.
    monotonic_dominances: List of two-element tuples. First element is the index
      of the dominant feature. Second element is the index of the weak feature.
    weights_shape: None or shape of tensor which represents weights of Linear
      layer.

  Raises:
    ValueError: If something is inconsistent.
  """
  # It also raises errors if monotonicities specified incorrectly.
  monotonicities = canonicalize_monotonicities(monotonicities)

  if monotonicities is not None and num_input_dims is not None:
    if len(monotonicities) != num_input_dims:
      raise ValueError("Number of elements in 'monotonicities' must be equal to"
                       " num_input_dims. monotoniticites: %s, "
                       "len(monotonicities): %d, num_input_dims: %d"
                       % (monotonicities, len(monotonicities), num_input_dims))

  if weights_shape is not None and monotonicities is not None:
    if (len(weights_shape) != 2 or weights_shape[0] != len(monotonicities)
        or weights_shape[1] != 1):
      raise ValueError("Number of elements in 'monotonicities' does not "
                       "correspond to number of weights. Weights shape: %s, "
                       "monotonicities: %s" % (weights_shape, monotonicities))

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
          raise ValueError("Monotonic dominance constraint's features must be "
                           "monotonic. Dimension %d is not monotonic." % (dim))
      if (weak_dim, dominant_dim) in dim_pairs:
        raise ValueError("Cannot have two dominance constraints on the same "
                         "pair of features conflicting. Features: %d, %d" %
                         (dominant_dim, weak_dim))
      dim_pairs.add((dominant_dim, weak_dim))


def canonicalize_monotonicities(monotonicities):
  """Converts string constants representing monotonicities into integers.

  Args:
    monotonicities: monotonicities hyperparameter of `Lattice` layer.

  Raises:
    ValueError if one of monotonicities is invalid.

  Returns:
    monotonicities represented as 0 or 1.
  """
  if monotonicities:
    canonicalized = []
    for item in monotonicities:
      if item in [-1, 0, 1]:
        canonicalized.append(item)
      elif isinstance(item, six.string_types) and item.lower() == "decreasing":
        canonicalized.append(-1)
      elif isinstance(item, six.string_types) and item.lower() == "none":
        canonicalized.append(0)
      elif isinstance(item, six.string_types) and item.lower() == "increasing":
        canonicalized.append(1)
      else:
        raise ValueError("'monotonicities' elements must be from: [-1, 0, 1, "
                         "'decreasing', 'none', 'increasing']. "
                         "Given: %s" % monotonicities)
    return canonicalized
  return None
