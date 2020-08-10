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
"""Helpers shared by multiple modules in TFL."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six


# TODO: update library not to explicitly check if None so we can return
# an empty list instead of None for these canonicalization methods.
def canonicalize_convexity(convexity):
  """Converts string constants representing convexity into integers.

  Args:
    convexity: The convexity hyperparameter of `tfl.layers.PWLCalibration`
      layer.

  Returns:
    convexity represented as -1, 0, 1, or None.

  Raises:
    ValueError: If convexity is not in the set
      {-1, 0, 1, 'concave', 'none', 'convex'}.
  """
  if convexity is None:
    return None

  if convexity in [-1, 0, 1]:
    return convexity
  elif isinstance(convexity, six.string_types):
    if convexity.lower() == "concave":
      return -1
    if convexity.lower() == "none":
      return 0
    if convexity.lower() == "convex":
      return 1
  raise ValueError("'convexity' must be from: [-1, 0, 1, 'concave', "
                   "'none', 'convex']. Given: {}".format(convexity))


def canonicalize_input_bounds(input_bounds):
  """Converts string constant 'none' representing unspecified bound into None.

  Args:
    input_bounds: The input_min or input_max hyperparameter of
      `tfl.layers.Linear` layer.

  Returns:
    A list of [val, val, ...] where val can be a float or None, or the value
    None if input_bounds is None.

  Raises:
    ValueError: If one of elements in input_bounds is not a float, None or
      'none'.
  """
  if input_bounds:
    canonicalized = []
    for item in input_bounds:
      if isinstance(item, float) or item is None:
        canonicalized.append(item)
      elif isinstance(item, six.string_types) and item.lower() == "none":
        canonicalized.append(None)
      else:
        raise ValueError("Both 'input_min' and 'input_max' elements must be "
                         "either int, float, None, or 'none'. Given: {}".format(
                             input_bounds))
    return canonicalized
  return None


def canonicalize_monotonicity(monotonicity, allow_decreasing=True):
  """Converts string constants representing monotonicity into integers.

  Args:
    monotonicity: The monotonicities hyperparameter of a `tfl.layers` Layer
      (e.g. `tfl.layers.PWLCalibration`).
    allow_decreasing: If decreasing monotonicity is considered a valid
      monotonicity.

  Returns:
    monotonicity represented as -1, 0, 1, or None.

  Raises:
    ValueError: If monotonicity is not in the set
      {-1, 0, 1, 'decreasing', 'none', 'increasing'} and allow_decreasing is
      True.
    ValueError: If monotonicity is not in the set {0, 1, 'none', 'increasing'}
      and allow_decreasing is False.
  """
  if monotonicity is None:
    return None

  if monotonicity in [-1, 0, 1]:
    if not allow_decreasing and monotonicity == -1:
      raise ValueError(
          "'monotonicities' must be from: [0, 1, 'none', 'increasing']. "
          "Given: {}".format(monotonicity))
    return monotonicity
  elif isinstance(monotonicity, six.string_types):
    if monotonicity.lower() == "decreasing":
      if not allow_decreasing:
        raise ValueError(
            "'monotonicities' must be from: [0, 1, 'none', 'increasing']. "
            "Given: {}".format(monotonicity))
      return -1
    if monotonicity.lower() == "none":
      return 0
    if monotonicity.lower() == "increasing":
      return 1
  raise ValueError("'monotonicities' must be from: [-1, 0, 1, 'decreasing', "
                   "'none', 'increasing']. Given: {}".format(monotonicity))


def canonicalize_monotonicities(monotonicities, allow_decreasing=True):
  """Converts string constants representing monotonicities into integers.

  Args:
    monotonicities: monotonicities hyperparameter of a `tfl.layers` Layer (e.g.
      `tfl.layers.Lattice`).
    allow_decreasing: If decreasing monotonicity is considered a valid
      monotonicity.

  Returns:
    A list of monotonicities represented as -1, 0, 1, or the value None
    if monotonicities is None.

  Raises:
    ValueError: If one of monotonicities is not in the set
      {-1, 0, 1, 'decreasing', 'none', 'increasing'} and allow_decreasing is
      True.
    ValueError: If one of monotonicities is not in the set
      {0, 1, 'none', 'increasing'} and allow_decreasing is False.
  """
  if monotonicities:
    return [
        canonicalize_monotonicity(
            monotonicity, allow_decreasing=allow_decreasing)
        for monotonicity in monotonicities
    ]
  return None


def canonicalize_trust(trusts):
  """Converts string constants representing trust direction into integers.

  Args:
    trusts: edgeworth_trusts or trapezoid_trusts hyperparameter of
      `tfl.layers.Lattice` layer.

  Returns:
    A list of trust constraint tuples of the form
    (feature_a, feature_b, direction) where direction can be -1 or 1, or the
    value None if trusts is None.

  Raises:
    ValueError: If one of trust constraints does not have 3 elements.
    ValueError: If one of trust constraints' direction is not in the set
      {-1, 1, 'negative', 'positive'}.
  """
  if trusts:
    canonicalized = []
    for trust in trusts:
      if len(trust) != 3:
        raise ValueError("Trust constraints must consist of 3 elements. Seeing "
                         "constraint tuple {}".format(trust))
      feature_a, feature_b, direction = trust
      if direction in [-1, 1]:
        canonicalized.append(trust)
      elif (isinstance(direction, six.string_types) and
            direction.lower() == "negative"):
        canonicalized.append((feature_a, feature_b, -1))
      elif (isinstance(direction, six.string_types) and
            direction.lower() == "positive"):
        canonicalized.append((feature_a, feature_b, 1))
      else:
        raise ValueError("trust constraint direction must be from: [-1, 1, "
                         "'negative', 'positive']. Given: {}".format(direction))
    return canonicalized
  return None


def canonicalize_unimodalities(unimodalities):
  """Converts string constants representing unimodalities into integers.

  Args:
    unimodalities: unimodalities hyperparameter of `tfl.layers.Lattice` layer.

  Returns:
    A list of unimodalities represented as -1, 0, 1, or the value None if
    unimodalities is None.

  Raises:
    ValueError: If one of unimodalities is not in the set
      {-1, 0, 1, 'peak', 'none', 'valley'}.
  """
  if not unimodalities:
    return None
  canonicalized = []
  for unimodality in unimodalities:
    if unimodality in [-1, 0, 1]:
      canonicalized.append(unimodality)
    elif isinstance(unimodality,
                    six.string_types) and unimodality.lower() == "peak":
      canonicalized.append(-1)
    elif isinstance(unimodality,
                    six.string_types) and unimodality.lower() == "none":
      canonicalized.append(0)
    elif isinstance(unimodality,
                    six.string_types) and unimodality.lower() == "valley":
      canonicalized.append(1)
    else:
      raise ValueError(
          "'unimodalities' elements must be from: [-1, 0, 1, 'peak', 'none', "
          "'valley']. Given: {}".format(unimodalities))
  return canonicalized


def count_non_zeros(*iterables):
  """Returns total number of non 0 elements in given iterables.

  Args:
    *iterables: Any number of the value None or iterables of numeric values.
  """
  result = 0
  for iterable in iterables:
    if iterable is not None:
      result += sum(1 for element in iterable if element != 0)
  return result
