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
"""Tests for Tensorflow Lattice premade."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_lattice.python import aggregation_layer
# pylint: disable=g-import-not-at-top
# Use Keras 2.
version_fn = getattr(tf.keras, 'version', None)
if version_fn and version_fn().startswith('3.'):
  import tf_keras as keras
else:
  keras = tf.keras


test_input = [
    tf.ragged.constant([[1, 2], [1, 2, 3], [3]]),
    tf.ragged.constant([[4, 5], [4, 4, 4], [6]]),
    tf.ragged.constant([[1, 6], [5, 5, 5], [9]])
]

expected_output = tf.constant([32, 40, 162])


class AggregationTest(tf.test.TestCase):

  def testAggregationLayer(self):
    # First we test our assertion that the model must be a keras.Model
    with self.assertRaisesRegex(ValueError,
                                'Model must be a keras.Model instance.'):
      aggregation_layer.Aggregation(None)
    # Now let's make sure our layer aggregates properly.
    inputs = [keras.Input(shape=()) for _ in range(len(test_input))]
    output = keras.layers.multiply(inputs)
    model = keras.Model(inputs=inputs, outputs=output)
    agg_layer = aggregation_layer.Aggregation(model)
    self.assertAllEqual(agg_layer(test_input), expected_output)


if __name__ == '__main__':
  tf.test.main()
