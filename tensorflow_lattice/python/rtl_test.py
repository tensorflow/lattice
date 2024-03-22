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
from tensorflow_lattice.python import linear_layer
from tensorflow_lattice.python import pwl_calibration_layer
from tensorflow_lattice.python import rtl_layer
# pylint: disable=g-import-not-at-top
# Use Keras 2.
version_fn = getattr(tf.keras, "version", None)
if version_fn and version_fn().startswith("3."):
  import tf_keras as keras
else:
  keras = tf.keras


class RTLTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(RTLTest, self).setUp()
    self.disable_all = False
    keras.utils.set_random_seed(42)

  def testRTLInputShapes(self):
    if self.disable_all:
      return
    data_size = 100

    # Dense input format.
    a = np.random.random_sample(size=(data_size, 10))
    b = np.random.random_sample(size=(data_size, 20))
    target_ab = (
        np.max(a, axis=1, keepdims=True) + np.min(b, axis=1, keepdims=True))

    input_a = keras.layers.Input(shape=(10,))
    input_b = keras.layers.Input(shape=(20,))

    rtl_0 = rtl_layer.RTL(num_lattices=6, lattice_rank=5)
    rtl_outputs = rtl_0({"unconstrained": input_a, "increasing": input_b})
    outputs = keras.layers.Dense(1)(rtl_outputs)
    model = keras.Model(inputs=[input_a, input_b], outputs=outputs)
    model.compile(loss="mse")
    model.fit([a, b], target_ab)
    model.predict([a, b])

    # Inputs to be calibrated.
    c = np.random.random_sample(size=(data_size, 1))
    d = np.random.random_sample(size=(data_size, 1))
    e = np.random.random_sample(size=(data_size, 1))
    f = np.random.random_sample(size=(data_size, 1))
    target_cdef = np.sin(np.pi * c) * np.cos(np.pi * d) - e * f

    input_c = keras.layers.Input(shape=(1,))
    input_d = keras.layers.Input(shape=(1,))
    input_e = keras.layers.Input(shape=(1,))
    input_f = keras.layers.Input(shape=(1,))

    input_keypoints = np.linspace(0.0, 1.0, 10)
    calib_c = pwl_calibration_layer.PWLCalibration(
        units=2,
        input_keypoints=input_keypoints,
        output_min=0.0,
        output_max=1.0)(
            input_c)
    calib_d = pwl_calibration_layer.PWLCalibration(
        units=3,
        input_keypoints=input_keypoints,
        output_min=0.0,
        output_max=1.0)(
            input_d)
    calib_e = pwl_calibration_layer.PWLCalibration(
        units=4,
        input_keypoints=input_keypoints,
        output_min=0.0,
        output_max=1.0,
        monotonicity="decreasing")(
            input_e)
    calib_f = pwl_calibration_layer.PWLCalibration(
        units=5,
        input_keypoints=input_keypoints,
        output_min=0.0,
        output_max=1.0,
        monotonicity="decreasing")(
            input_f)

    rtl_0 = rtl_layer.RTL(num_lattices=10, lattice_rank=3)
    rtl_0_outputs = rtl_0({
        "unconstrained": [calib_c, calib_d],
        "increasing": [calib_e, calib_f]
    })
    outputs = linear_layer.Linear(
        num_input_dims=10, monotonicities=[1] * 10)(
            rtl_0_outputs)
    model = keras.Model(
        inputs=[input_c, input_d, input_e, input_f], outputs=outputs
    )
    model.compile(loss="mse")
    model.fit([c, d, e, f], target_cdef)
    model.predict([c, d, e, f])

    # Two layer RTL model.
    rtl_0 = rtl_layer.RTL(
        num_lattices=10,
        lattice_rank=3,
        output_min=0.0,
        output_max=1.0,
        separate_outputs=True)
    rtl_0_outputs = rtl_0({
        "unconstrained": [calib_c, calib_d],
        "increasing": [calib_e, calib_f]
    })
    rtl_1 = rtl_layer.RTL(num_lattices=3, lattice_rank=4)
    rtl_1_outputs = rtl_1(rtl_0_outputs)
    outputs = linear_layer.Linear(
        num_input_dims=3, monotonicities=[1] * 3)(
            rtl_1_outputs)
    model = keras.Model(
        inputs=[input_c, input_d, input_e, input_f], outputs=outputs
    )
    model.compile(loss="mse")
    model.fit([c, d, e, f], target_cdef)
    model.predict([c, d, e, f])

  def testRTLOutputShape(self):
    if self.disable_all:
      return

    # Multiple Outputs Per Lattice
    input_shape, output_shape = (30,), (None, 6)
    input_a = keras.layers.Input(shape=input_shape)
    rtl_0 = rtl_layer.RTL(num_lattices=6, lattice_rank=5)
    output = rtl_0(input_a)
    self.assertAllEqual(output_shape, rtl_0.compute_output_shape(input_a.shape))
    self.assertAllEqual(output_shape, output.shape)

    # Average Outputs
    output_shape = (None, 1)
    rtl_1 = rtl_layer.RTL(num_lattices=6, lattice_rank=5, average_outputs=True)
    output = rtl_1(input_a)
    self.assertAllEqual(output_shape, rtl_1.compute_output_shape(input_a.shape))
    self.assertAllEqual(output_shape, output.shape)

  def testRTLSaveLoad(self):
    if self.disable_all:
      return

    input_c = keras.layers.Input(shape=(1,))
    input_d = keras.layers.Input(shape=(1,))
    input_e = keras.layers.Input(shape=(1,))
    input_f = keras.layers.Input(shape=(1,))

    input_keypoints = np.linspace(0.0, 1.0, 10)
    calib_c = pwl_calibration_layer.PWLCalibration(
        units=2,
        input_keypoints=input_keypoints,
        output_min=0.0,
        output_max=1.0)(
            input_c)
    calib_d = pwl_calibration_layer.PWLCalibration(
        units=3,
        input_keypoints=input_keypoints,
        output_min=0.0,
        output_max=1.0)(
            input_d)
    calib_e = pwl_calibration_layer.PWLCalibration(
        units=4,
        input_keypoints=input_keypoints,
        output_min=0.0,
        output_max=1.0,
        monotonicity="decreasing")(
            input_e)
    calib_f = pwl_calibration_layer.PWLCalibration(
        units=5,
        input_keypoints=input_keypoints,
        output_min=0.0,
        output_max=1.0,
        monotonicity="decreasing")(
            input_f)

    rtl_0 = rtl_layer.RTL(
        num_lattices=10,
        lattice_rank=3,
        output_min=0.0,
        output_max=1.0,
        separate_outputs=True)
    rtl_0_outputs = rtl_0({
        "unconstrained": [calib_c, calib_d],
        "increasing": [calib_e, calib_f]
    })
    rtl_1 = rtl_layer.RTL(num_lattices=3, lattice_rank=4)
    rtl_1_outputs = rtl_1(rtl_0_outputs)
    outputs = linear_layer.Linear(
        num_input_dims=3, monotonicities=[1] * 3)(
            rtl_1_outputs)
    model = keras.Model(
        inputs=[input_c, input_d, input_e, input_f], outputs=outputs
    )
    model.compile(loss="mse")
    model.use_legacy_config = True

    with tempfile.NamedTemporaryFile(suffix=".h5") as f:
      model.save(f.name)
      _ = keras.models.load_model(
          f.name,
          custom_objects={
              "RTL": rtl_layer.RTL,
              "PWLCalibration": pwl_calibration_layer.PWLCalibration,
              "Linear": linear_layer.Linear,
          },
      )


if __name__ == "__main__":
  tf.test.main()
