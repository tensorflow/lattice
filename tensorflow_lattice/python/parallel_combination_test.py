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
"""Tests for Lattice Layer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_lattice.python import lattice_layer as ll
from tensorflow_lattice.python import parallel_combination_layer as pcl
# pylint: disable=g-import-not-at-top
# Use Keras 2.
version_fn = getattr(tf.keras, "version", None)
if version_fn and version_fn().startswith("3."):
  import tf_keras as keras
else:
  keras = tf.keras


class ParallelCombinationTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(ParallelCombinationTest, self).setUp()
    self.disable_all = False
    keras.utils.set_random_seed(42)

  def testParallelCombinationSingleInput(self):
    if self.disable_all:
      return
    all_calibrators = pcl.ParallelCombination()
    for i in range(3):
      # Its not typical to use 1-d Lattice layer for calibration, but lets do it
      # to avoid redundant dependency on PWLCalibration layer.
      calibrator = ll.Lattice(
          lattice_sizes=[2], output_min=0.0, output_max=i + 1.0)
      all_calibrators.append(calibrator)

    # Given output range specified below linear initializer will have lattice to
    # simply sum up inputs.
    simple_sum = ll.Lattice(
        lattice_sizes=[5] * 3,
        kernel_initializer="linear_initializer",
        output_min=0.0,
        output_max=12.0,
        name="SummingLattice")
    model = keras.models.Sequential()
    model.add(all_calibrators)
    model.add(simple_sum)

    test_inputs = np.asarray([
        [0.0, 0.0, 0.0],
        [0.1, 0.2, 0.3],
        [1.0, 1.0, 1.0],
    ])
    predictions = model.predict(test_inputs)
    print("predictions")
    print(predictions)
    self.assertTrue(np.allclose(predictions, np.asarray([[0.0], [1.4], [6.0]])))

  def testParallelCombinationMultipleInputs(self):
    if self.disable_all:
      return
    input_layers = [keras.layers.Input(shape=[1]) for _ in range(3)]
    all_calibrators = pcl.ParallelCombination(single_output=False)
    for i in range(3):
      # Its not typical to use 1-d Lattice layer for calibration, but lets do it
      # to avoid redundant dependency on PWLCalibration layer.
      calibrator = ll.Lattice(
          lattice_sizes=[2], output_min=0.0, output_max=i + 1.0)
      all_calibrators.append(calibrator)

    # Given output range specified below linear initializer will have lattice to
    # simply sum up inputs.
    simple_sum = ll.Lattice(
        lattice_sizes=[5] * 3,
        kernel_initializer="linear_initializer",
        output_min=0.0,
        output_max=12.0,
        name="SummingLattice",
        trainable=False)

    output = simple_sum(all_calibrators(input_layers))
    model = keras.models.Model(inputs=input_layers, outputs=output)

    test_inputs = [
        np.asarray([[0.0], [0.1], [1.0]]),
        np.asarray([[0.0], [0.2], [1.0]]),
        np.asarray([[0.0], [0.3], [1.0]]),
    ]
    predictions = model.predict(test_inputs)
    print("predictions")
    print(predictions)
    self.assertTrue(np.allclose(predictions, np.asarray([[0.0], [1.4], [6.0]])))

  def testParallelCombinationClone(self):
    if self.disable_all:
      return
    input_layers = [keras.layers.Input(shape=[1]) for _ in range(3)]
    all_calibrators = pcl.ParallelCombination(single_output=False)
    for i in range(3):
      # Its not typical to use 1-d Lattice layer for calibration, but lets do it
      # to avoid redundant dependency on PWLCalibration layer.
      calibrator = ll.Lattice(
          lattice_sizes=[2], output_min=0.0, output_max=i + 1.0)
      all_calibrators.append(calibrator)

    # Given output range specified below linear initializer will have lattice to
    # simply sum up inputs.
    simple_sum = ll.Lattice(
        lattice_sizes=[5] * 3,
        kernel_initializer="linear_initializer",
        output_min=0.0,
        output_max=12.0,
        name="SummingLattice",
        trainable=False)

    output = simple_sum(all_calibrators(input_layers))
    model = keras.models.Model(inputs=input_layers, outputs=output)
    clone = keras.models.clone_model(model)

    test_inputs = [
        np.asarray([[0.0], [0.1], [1.0]]),
        np.asarray([[0.0], [0.2], [1.0]]),
        np.asarray([[0.0], [0.3], [1.0]]),
    ]
    predictions = clone.predict(test_inputs)
    print("predictions")
    print(predictions)
    self.assertTrue(np.allclose(predictions, np.asarray([[0.0], [1.4], [6.0]])))

    with tempfile.NamedTemporaryFile(suffix=".h5") as f:
      model.save(f.name)
      loaded_model = keras.models.load_model(
          f.name,
          custom_objects={
              "ParallelCombination": pcl.ParallelCombination,
              "Lattice": ll.Lattice,
          },
      )
      predictions = loaded_model.predict(test_inputs)
      self.assertTrue(
          np.allclose(predictions, np.asarray([[0.0], [1.4], [6.0]])))


if __name__ == "__main__":
  tf.test.main()
