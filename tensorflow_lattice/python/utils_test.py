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
"""Tests for Tensorflow Lattice utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_lattice.python import utils


class UtilsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters((-1, -1), (0, 0), (1, 1), ("concave", -1),
                            ("none", 0), ("convex", 1))
  def testCanonicalizeConvexity(self, convexity,
                                expected_canonicalized_convexity):
    canonicalized_convexity = utils.canonicalize_convexity(convexity)
    self.assertEqual(canonicalized_convexity, expected_canonicalized_convexity)

  @parameterized.parameters((-2), (0.5), (3), ("invalid_convexity"),
                            ("concaves"), ("nonw"), ("conve"))
  def testInvalidConvexity(self, invalid_convexity):
    error_message = (
        "'convexity' must be from: [-1, 0, 1, 'concave', 'none', 'convex']. "
        "Given: {}").format(invalid_convexity)
    with self.assertRaisesWithLiteralMatch(ValueError, error_message):
      utils.canonicalize_convexity(invalid_convexity)

  # Note: must use mapping format because otherwise input parameter list is
  # considered multiple parameters (not just a single list parameter).
  @parameterized.parameters(
      {
          "input_bounds": [0.0, -3.0],
          "expected_canonicalized_input_bounds": [0.0, -3.0]
      }, {
          "input_bounds": [float("-inf"), 0.12345],
          "expected_canonicalized_input_bounds": [float("-inf"), 0.12345]
      }, {
          "input_bounds": ["none", None],
          "expected_canonicalized_input_bounds": [None, None]
      })
  def testCanonicalizeInputBounds(self, input_bounds,
                                  expected_canonicalized_input_bounds):
    canonicalized_input_bounds = utils.canonicalize_input_bounds(input_bounds)
    self.assertAllEqual(canonicalized_input_bounds,
                        expected_canonicalized_input_bounds)

  @parameterized.parameters({"invalid_input_bounds": [0, 1.0, 2.0]},
                            {"invalid_input_bounds": [None, "nonw"]})
  def testInvalidInputBounds(self, invalid_input_bounds):
    error_message = (
        "Both 'input_min' and 'input_max' elements must be either int, float, "
        "None, or 'none'. Given: {}").format(invalid_input_bounds)
    with self.assertRaisesWithLiteralMatch(ValueError, error_message):
      utils.canonicalize_input_bounds(invalid_input_bounds)

  @parameterized.parameters((-1, -1), (0, 0), (1, 1), ("decreasing", -1),
                            ("none", 0), ("increasing", 1))
  def testCanonicalizeMonotonicity(self, monotonicity,
                                   expected_canonicalized_monotonicity):
    canonicalized_monotonicity = utils.canonicalize_monotonicity(monotonicity)
    self.assertEqual(canonicalized_monotonicity,
                     expected_canonicalized_monotonicity)

  @parameterized.parameters((-2), (0.5), (3), ("invalid_monotonicity"),
                            ("decrease"), ("increase"))
  def testInvalidMonotonicity(self, invalid_monotonicity):
    error_message = (
        "'monotonicities' must be from: [-1, 0, 1, 'decreasing', 'none', "
        "'increasing']. Given: {}").format(invalid_monotonicity)
    with self.assertRaisesWithLiteralMatch(ValueError, error_message):
      utils.canonicalize_monotonicity(invalid_monotonicity)

  @parameterized.parameters(("decreasing"), (-1))
  def testInvalidDecreasingMonotonicity(self, invalid_monotonicity):
    error_message = (
        "'monotonicities' must be from: [0, 1, 'none', 'increasing']. "
        "Given: {}").format(invalid_monotonicity)
    with self.assertRaisesWithLiteralMatch(ValueError, error_message):
      utils.canonicalize_monotonicity(
          invalid_monotonicity, allow_decreasing=False)

  # Note: since canonicalize_monotonicities calls canonicalize_monotonicity,
  # the above test for invalidity is sufficient.
  @parameterized.parameters(([-1, 0, 1], [-1, 0, 1]),
                            (["decreasing", "none", "increasing"], [-1, 0, 1]),
                            (["decreasing", -1], [-1, -1]),
                            (["none", 0], [0, 0]), (["increasing", 1], [1, 1]))
  def testCanonicalizeMonotonicities(self, monotonicities,
                                     expected_canonicalized_monotonicities):
    canonicalized_monotonicities = utils.canonicalize_monotonicities(
        monotonicities)
    self.assertAllEqual(canonicalized_monotonicities,
                        expected_canonicalized_monotonicities)

  @parameterized.parameters(([("a", "b", -1), ("b", "c", 1)], [("a", "b", -1),
                                                               ("b", "c", 1)]),
                            ([("a", "b", "negative"),
                              ("b", "c", "positive")], [("a", "b", -1),
                                                        ("b", "c", 1)]))
  def testCanonicalizeTrust(self, trusts, expected_canonicalized_trusts):
    canonicalized_trusts = utils.canonicalize_trust(trusts)
    self.assertAllEqual(canonicalized_trusts, expected_canonicalized_trusts)

  # Note 1: this test assumes the first trust in the list has the incorrect
  # direction. A list with a single trust tuple is sufficient.
  # Note 2: must use mapping format because otherwise input parameter list is
  # considered multiple parameters (not just a single list parameter).
  @parameterized.parameters({"invalid_trusts": [("a", "b", 0)]},
                            {"invalid_trusts": [("a", "b", "negativ")]})
  def testInvalidTrustDirection(self, invalid_trusts):
    error_message = (
        "trust constraint direction must be from: [-1, 1, 'negative', "
        "'positive']. Given: {}").format(invalid_trusts[0][2])
    with self.assertRaisesWithLiteralMatch(ValueError, error_message):
      utils.canonicalize_trust(invalid_trusts)

  # Note 1: this test assumes the first trust in the list has the incorrect
  # size. A list with a single trust tuple is sufficient.
  # Note 2: must use mapping format because otherwise input parameter list is
  # considered multiple parameters (not just a single list parameter).
  @parameterized.parameters({"invalid_trusts": [("a", 1)]},
                            {"invalid_trusts": [("a", "b", -1, 1)]})
  def testInvalidTrustLength(self, invalid_trusts):
    error_message = (
        "Trust constraints must consist of 3 elements. Seeing constraint "
        "tuple {}").format(invalid_trusts[0])
    with self.assertRaisesWithLiteralMatch(ValueError, error_message):
      utils.canonicalize_trust(invalid_trusts)

  @parameterized.parameters(([0, 1, 1, 0], [1, 0], 3),
                            ([0, 0, 0], [0, 0, 0], 0),
                            ([-1, 0, 0, 1], [0, 0], 2),
                            (None, [1, 1, 1, 1, 1], 5))
  def testCountNonZeros(self, monotonicities, unimodalities,
                        expected_non_zeros):
    non_zeros = utils.count_non_zeros(monotonicities, unimodalities)
    self.assertEqual(non_zeros, expected_non_zeros)

  @parameterized.parameters(
      ([-1, 0, 1], [-1, 0, 1]), (["peak", "none", "valley"], [-1, 0, 1]),
      (["peak", -1], [-1, -1]), (["none", 0], [0, 0]), (["valley", 1], [1, 1]))
  def testCanonicalizeUnimodalities(self, unimodalities,
                                    expected_canonicalized_unimodalities):
    canonicalized_unimodalities = utils.canonicalize_unimodalities(
        unimodalities)
    self.assertAllEqual(canonicalized_unimodalities,
                        expected_canonicalized_unimodalities)

  # Note: must use mapping format because otherwise input parameter list is
  # considered multiple parameters (not just a single list parameter).
  @parameterized.parameters({"invalid_unimodalities": ["vally", 0]},
                            {"invalid_unimodalities": [-1, 0, 2]})
  def testInvalidUnimoadlities(self, invalid_unimodalities):
    error_message = (
        "'unimodalities' elements must be from: [-1, 0, 1, 'peak', 'none', "
        "'valley']. Given: {}").format(invalid_unimodalities)
    with self.assertRaisesWithLiteralMatch(ValueError, error_message):
      utils.canonicalize_unimodalities(invalid_unimodalities)


if __name__ == "__main__":
  tf.test.main()
