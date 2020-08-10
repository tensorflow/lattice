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
"""Helpers and computations of categorical calibration layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import internal_utils
import tensorflow as tf


def project(weights, output_min, output_max, monotonicities):
  """Monotonicity/bounds constraints implementation for categorical calibration.

  Returns the approximate L2 projection of the CategoricalCalibration weights
  into the constrained parameter space.

  Args:
    weights: Tensor which represents weights of Categorical calibration layer.
    output_min: Lower bound constraint on weights.
    output_max: Upper bound constraint on weights.
    monotonicities: List of pair of indices `(i, j)`, indicating constraint
      `weight[i] <= weight[j]`.

  Returns:
    Projected `weights` tensor.

  Raises:
    ValueError: If monotonicities are not of the correct format or are circular.
  """
  num_buckets = weights.shape[0]
  verify_hyperparameters(
      num_buckets=num_buckets,
      output_min=output_min,
      output_max=output_max,
      monotonicities=monotonicities)

  projected_weights = weights

  if monotonicities:
    projected_weights = (
        internal_utils.approximately_project_categorical_partial_monotonicities(
            projected_weights, monotonicities))

  if output_min is not None:
    projected_weights = tf.maximum(projected_weights, output_min)
  if output_max is not None:
    projected_weights = tf.minimum(projected_weights, output_max)
  return projected_weights


def assert_constraints(weights,
                       output_min,
                       output_max,
                       monotonicities,
                       debug_tensors=None,
                       eps=1e-6):
  """Asserts that `weights` satisfiy constraints.

  Args:
    weights: Tensor which represents weights of Categorical calibration layer.
    output_min: Lower bound constraint on weights.
    output_max: Upper bound constraint on weights.
    monotonicities: List of pair of indices `(i, j)`, indicating constraint
      `weight[i] <= weight[j]`.
    debug_tensors: None or list of anything convertible to tensor (for example
      tensors or strings) which will be printed in case of constraints
      violation.
    eps: Allowed constraints violation.

  Returns:
    List of assertion ops in graph mode or immideately asserts in eager mode.
  """
  num_buckets = weights.shape[0]
  verify_hyperparameters(
      num_buckets=num_buckets,
      output_min=output_min,
      output_max=output_max,
      monotonicities=monotonicities)

  info = ["Outputs: ", weights, "Epsilon: ", eps]
  if debug_tensors:
    info += debug_tensors
  asserts = []

  if output_min is not None:
    min_output = tf.reduce_min(weights)
    asserts.append(
        tf.Assert(
            min_output >= output_min - eps,
            data=["Lower bound violation.", "output_min:", output_min] + info,
            summarize=num_buckets))

  if output_max is not None:
    max_output = tf.reduce_max(weights)
    asserts.append(
        tf.Assert(
            max_output <= output_max + eps,
            data=["Upper bound violation.", "output_max:", output_max] + info,
            summarize=num_buckets))

  if monotonicities:
    left = tf.gather_nd(weights, [[i] for (i, j) in monotonicities])
    right = tf.gather_nd(weights, [[j] for (i, j) in monotonicities])
    asserts.append(
        tf.Assert(
            tf.reduce_min(left - right) < eps,
            data=["Monotonicity violation.", "monotonicities:", monotonicities]
            + info,
            summarize=num_buckets))

  return asserts


def verify_hyperparameters(num_buckets=None,
                           output_min=None,
                           output_max=None,
                           monotonicities=None):
  """Verifies that all given hyperparameters are consistent.

  See `tfl.layers.CategoricalCalibration` class level comment for detailes.

  Args:
    num_buckets: `num_buckets` of CategoricalCalibration layer.
    output_min: `smallest output` of CategoricalCalibration layer.
    output_max: `largest output` of CategoricalCalibration layer.
    monotonicities: `monotonicities` of CategoricalCalibration layer.

  Raises:
    ValueError: If parameters are incorrect or inconsistent.
  """
  if output_min is not None and output_max is not None:
    if output_max < output_min:
      raise ValueError(
          "If specified output_max must be greater than output_min. "
          "They are: ({}, {})".format(output_min, output_max))

  if monotonicities:
    if (not isinstance(monotonicities, list) or not all(
        isinstance(m, (list, tuple)) and len(m) == 2 for m in monotonicities)):
      raise ValueError(
          "Monotonicities should be a list of pairs (list/tuples).")
    for (i, j) in monotonicities:
      if (i < 0 or j < 0 or (num_buckets is not None and
                             (i >= num_buckets or j >= num_buckets))):
        raise ValueError(
            "Monotonicities should be pairs of be indices in range "
            "[0, num_buckets). They are: {}".format(monotonicities))
