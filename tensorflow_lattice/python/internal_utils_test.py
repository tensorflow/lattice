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
import numpy as np
import tensorflow as tf
from tensorflow_lattice.python import internal_utils


class InternalUtilsTest(parameterized.TestCase, tf.test.TestCase):

  def _ResetAllBackends(self):
    tf.compat.v1.reset_default_graph()

  @parameterized.parameters(
      ([3., 4.], [(0, 1)], [3., 4.]), ([4., 3.], [(0, 1)], [3.5, 3.5]),
      ([1., 0.], [(0, 1)], [0.5, 0.5]), ([-1., 0.], [(1, 0)], [-0.5, -0.5]),
      ([4., 3., 2., 1., 0.], [(0, 1), (1, 2), (2, 3),
                              (3, 4)], [2., 2., 2., 2., 2.]))
  def testApproximatelyProjectCategoricalPartialMonotonicities(
      self, weights, monotonicities, expected_projected_weights):
    self._ResetAllBackends()
    weights = tf.Variable(weights)
    projected_weights = (
        internal_utils.approximately_project_categorical_partial_monotonicities(
            weights, monotonicities))
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(
        self.evaluate(projected_weights), np.array(expected_projected_weights))


if __name__ == '__main__':
  tf.test.main()
