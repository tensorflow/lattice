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
"""Internal helpers shared by multiple modules in TFL.

Note that this module is not expected to be used by TFL users, and that it is
not exposed in the TFL package.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf


def _topological_sort(key_less_than_values):
  """Topological sort for monotonicities.

  Args:
    key_less_than_values: A defaultdict from index to a list of indices, such
      that for j in key_less_than_values[i] we must have output(i) <= output(j).

  Returns:
    A topologically sorted list of indices.

  Raises:
    ValueError: If monotonicities are circular.
  """
  all_values = set()
  for values in key_less_than_values.values():
    all_values.update(values)

  q = [k for k in key_less_than_values if k not in all_values]
  if not q:
    raise ValueError(
        "Circular monotonicity constraints: {}".format(key_less_than_values))

  result = []
  seen = set()
  while q:
    v = q[-1]
    seen.add(v)
    expand = [x for x in key_less_than_values[v] if x not in seen]
    if not expand:
      result = [v] + result
      q.pop()
    else:
      q.append(expand[0])

  return result


def _min_projection(weights, sorted_indices, key_less_than_values, step):
  """Returns an approximate partial min projection with the given step_size.

  Args:
    weights: A list of tensors of shape `(units,)` to be approximatly projected
      based on the monotonicity constraints.
    sorted_indices: Topologically sorted list of indices based on the
      monotonicity constraints.
    key_less_than_values: A defaultdict from index to a list of indices, such
      that for `j` in `key_less_than_values[i]` we must have `weight[i] <=
      weight[j]`.
    step: A value defining if we should apply a full projection (`step == 1`) or
      a partial projection (`step < 1`).

  Returns:
    Projected list of tensors.
  """
  projected_weights = list(weights)  # copy
  for i in sorted_indices[::-1]:
    if key_less_than_values[i]:
      min_projection = projected_weights[i]
      for j in key_less_than_values[i]:
        min_projection = tf.minimum(min_projection, projected_weights[j])
      if step == 1:
        projected_weights[i] = min_projection
      else:
        projected_weights[i] = (
            step * min_projection + (1 - step) * projected_weights[i])
  return projected_weights


def _max_projection(weights, sorted_indices, key_greater_than_values, step):
  """Returns an approximate partial max projection with the given step_size.

  Args:
    weights: A list of tensors of shape `(units,)` to be approximatly projected
      based on the monotonicity constraints.
    sorted_indices: Topologically sorted list of indices based on the
      monotonicity constraints.
    key_greater_than_values: A defaultdict from index to a list of indices,
      indicating that for index `j` in `key_greater_than_values[i]` we must have
      `weight[i] >= weight[j]`.
    step: A value defining if we should apply a full projection (`step == 1`) or
      a partial projection (`step < 1`).

  Returns:
    Projected list of tensors.
  """
  projected_weights = list(weights)  # copy
  for i in sorted_indices:
    if key_greater_than_values[i]:
      max_projection = projected_weights[i]
      for j in key_greater_than_values[i]:
        max_projection = tf.maximum(max_projection, projected_weights[j])
      if step == 1:
        projected_weights[i] = max_projection
      else:
        projected_weights[i] = (
            step * max_projection + (1 - step) * projected_weights[i])
  return projected_weights


def approximately_project_categorical_partial_monotonicities(
    weights, monotonicities):
  """Returns an approximation L2 projection for categorical monotonicities.

  Categorical monotonocities are monotonicity constraints applied to the real
  values that are mapped from categorical inputs. Each monotonicity constraint
  is specified by a pair of categorical input indices. The projection is also
  used to constrain pairs of coefficients in linear models.

  Args:
    weights: Tensor of weights to be approximately projected based on the
      monotonicity constraints.
    monotonicities: List of pairs of indices `(i, j)`, indicating constraint
      `weights[i] <= weights[j]`.
  """
  key_less_than_values = collections.defaultdict(list)
  key_greater_than_values = collections.defaultdict(list)
  for i, j in monotonicities:
    key_less_than_values[i].append(j)
    key_greater_than_values[j].append(i)

  sorted_indices = _topological_sort(key_less_than_values)

  projected_weights = tf.unstack(weights)

  # A 0.5 min projection followed by a full max projection.
  projected_weights_min_max = _min_projection(projected_weights, sorted_indices,
                                              key_less_than_values, 0.5)
  projected_weights_min_max = _max_projection(projected_weights_min_max,
                                              sorted_indices,
                                              key_greater_than_values, 1)
  projected_weights_min_max = tf.stack(projected_weights_min_max)

  # A 0.5 max projection followed by a full min projection.
  projected_weights_max_min = _max_projection(projected_weights, sorted_indices,
                                              key_greater_than_values, 0.5)
  projected_weights_max_min = _min_projection(projected_weights_max_min,
                                              sorted_indices,
                                              key_less_than_values, 1)
  projected_weights_max_min = tf.stack(projected_weights_max_min)

  # Take the average of the two results to avoid sliding to one direction.
  projected_weights = (projected_weights_min_max +
                       projected_weights_max_min) / 2
  return projected_weights
