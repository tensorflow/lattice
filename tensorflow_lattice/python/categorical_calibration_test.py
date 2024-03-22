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
"""Tests for categorical calibration layer.

This test should be run with "-c opt" since otherwise it's slow.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_lattice.python import categorical_calibration_layer as categorical_calibraion
from tensorflow_lattice.python import parallel_combination_layer as parallel_combination
from tensorflow_lattice.python import test_utils
# pylint: disable=g-import-not-at-top
# Use Keras 2.
version_fn = getattr(tf.keras, "version", None)
if version_fn and version_fn().startswith("3."):
  import tf_keras as keras
else:
  keras = tf.keras


class CategoricalCalibrationLayerTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(CategoricalCalibrationLayerTest, self).setUp()
    self._disable_all = False
    self._loss_eps = 1e-2
    self._loss_diff_eps = 1e-4
    keras.utils.set_random_seed(42)

  def _ResetAllBackends(self):
    keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

  def _ScatterXUniformly(self, units, num_points, num_buckets,
                         missing_probability, default_input_value):
    """Randomly uniformly scatters points across input space."""
    data = []
    for unit_idx in range(units):
      if missing_probability > 0.0:
        missing_points = int(num_points * missing_probability)
      else:
        missing_points = 0

      x = ([default_input_value for _ in range(missing_points)] +
           [i % num_buckets for i in range(num_points - missing_points)])
      np.random.seed(unit_idx)
      np.random.shuffle(x)
      if data:
        data = [values + (value,) for values, value in zip(data, x)]
      else:
        data = [(value,) for value in x]

    return [np.asarray(v, dtype=np.int32) for v in data]

  def _SetDefaults(self, config):
    config.setdefault("units", 1)
    config.setdefault("use_multi_calibration_layer", False)
    config.setdefault("one_d_input", False)
    config.setdefault("output_min", None)
    config.setdefault("output_max", None)
    config.setdefault("default_input_value", None)
    config.setdefault("monotonicities", None)
    config.setdefault("missing_probability", 0.0)
    config.setdefault("constraint_assertion_eps", 1e-6)
    config.setdefault("kernel_regularizer", None)
    config.setdefault("model_dir", "/tmp/test_pwl_model_dir/")
    return config

  def _TrainModel(self, config):
    """Trains model and returns loss.

    Args:
      config: Layer config internal for this test which specifies params of
        piecewise linear layer to train.

    Returns:
      Training loss.
    """
    logging.info("Testing config:")
    logging.info(config)
    config = self._SetDefaults(config)

    self._ResetAllBackends()

    if config["default_input_value"] is not None:
      # default_input_value is mapped to the last bucket, hence x_generator
      # needs to generate in [0, ..., num_buckets-2] range.
      num_random_buckets = config["num_buckets"] - 1
    else:
      num_random_buckets = config["num_buckets"]

    # The input to the model can either be single or multi dimensional.
    input_units = 1 if config["one_d_input"] else config["units"]

    training_inputs = config["x_generator"](
        units=input_units,
        num_points=config["num_training_records"],
        num_buckets=num_random_buckets,
        missing_probability=config["missing_probability"],
        default_input_value=config["default_input_value"])
    training_labels = [config["y_function"](x) for x in training_inputs]

    # Either create multiple CategoricalCalibration layers and combine using a
    # ParallelCombination layer, or create a single CategoricalCalibration with
    # multiple output dimensions.
    if config["use_multi_calibration_layer"]:
      num_calibration_layers = config["units"]
      categorical_calibraion_units = 1
    else:
      num_calibration_layers = 1
      categorical_calibraion_units = config["units"]

    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=[input_units], dtype=tf.int32))
    calibration_layers = []
    for _ in range(num_calibration_layers):
      calibration_layers.append(
          categorical_calibraion.CategoricalCalibration(
              units=categorical_calibraion_units,
              kernel_initializer="constant",
              num_buckets=config["num_buckets"],
              output_min=config["output_min"],
              output_max=config["output_max"],
              monotonicities=config["monotonicities"],
              kernel_regularizer=config["kernel_regularizer"],
              default_input_value=config["default_input_value"]))
    if len(calibration_layers) == 1:
      model.add(calibration_layers[0])
    else:
      model.add(parallel_combination.ParallelCombination(calibration_layers))
    if config["units"] > 1:
      model.add(
          keras.layers.Lambda(
              lambda x: tf.reduce_mean(x, axis=1, keepdims=True)))
    model.compile(
        loss=keras.losses.mean_squared_error,
        optimizer=config["optimizer"](learning_rate=config["learning_rate"]))

    training_data = (training_inputs, training_labels)

    loss = test_utils.run_training_loop(
        config=config,
        training_data=training_data,
        keras_model=model,
        input_dtype=np.int32)

    assetion_ops = []
    for calibration_layer in calibration_layers:
      assetion_ops.extend(
          calibration_layer.assert_constraints(
              eps=config["constraint_assertion_eps"]))
    if not tf.executing_eagerly() and assetion_ops:
      tf.compat.v1.keras.backend.get_session().run(assetion_ops)

    return loss

  @parameterized.parameters((np.mean,), (lambda x: -np.mean(x),))
  def testUnconstrainedNoMissingValue(self, y_function):
    if self._disable_all:
      return
    config = {
        "num_training_records": 200,
        "num_training_epoch": 500,
        "optimizer": keras.optimizers.Adam,
        "learning_rate": 0.15,
        "x_generator": self._ScatterXUniformly,
        "y_function": y_function,
        "num_buckets": 10,
        "output_min": None,
        "output_max": None,
        "monotonicities": None,
    }
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.0, delta=self._loss_eps)
    config["units"] = 3
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.0, delta=self._loss_eps)
    config["one_d_input"] = True
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.0, delta=self._loss_eps)

  @parameterized.parameters((np.mean,), (lambda x: -np.mean(x),))
  def testUnconstrainedWithMissingValue(self, y_function):
    if self._disable_all:
      return
    config = {
        "num_training_records": 200,
        "num_training_epoch": 500,
        "optimizer": keras.optimizers.Adam,
        "learning_rate": 0.15,
        "x_generator": self._ScatterXUniformly,
        "y_function": y_function,
        "num_buckets": 10,
        "output_min": None,
        "output_max": None,
        "monotonicities": None,
        "default_input_value": -1,
        "missing_probability": 0.1,
    }
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.0, delta=self._loss_eps)
    config["units"] = 3
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.0, delta=self._loss_eps)
    config["one_d_input"] = True
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.0, delta=self._loss_eps)

  @parameterized.parameters(
      (0.0, 9.0, None, 0.0),
      (1.0, 8.0, None, 0.2),
      (1.0, 8.0, [(6, 5)], 0.25),
      (1.0, 8.0, [(6, 5), (5, 4)], 0.4),
      (1.0, 8.0, [(6, 5), (7, 5)], 0.4),
      (1.0, 8.0, [(6, 5), (5, 4), (4, 3)], 0.7),
      (1.0, 8.0, [(7, 6), (6, 5), (4, 3), (3, 2)], 0.6),
      (1.0, 8.0, [(7, 6), (6, 5), (5, 4), (4, 3), (3, 2)], 1.95),
  )
  def testConstraints(self, output_min, output_max, monotonicities,
                      expected_loss):
    if self._disable_all:
      return
    config = {
        "num_training_records": 1000,
        "num_training_epoch": 1000,
        "optimizer": keras.optimizers.Adam,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": np.mean,
        "num_buckets": 10,
        "output_min": output_min,
        "output_max": output_max,
        "monotonicities": monotonicities,
    }

    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)

    # Same input with multiple calibration units, should give out the same loss.
    config["one_d_input"] = True
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)

    # With independently sampled unit-dim inputs loss is caled by 1/units.
    config["one_d_input"] = False
    loss = self._TrainModel(config)
    self.assertAlmostEqual(
        loss,
        expected_loss / config["units"],
        delta=self._loss_eps * config["units"])

    # Using separate calibration layers should give out the same loss.
    config["use_multi_calibration_layer"] = True
    loss_multi_calib = self._TrainModel(config)
    self.assertAlmostEqual(loss, loss_multi_calib, delta=self._loss_diff_eps)

  def testCircularMonotonicites(self):
    if self._disable_all:
      return
    config = {
        "num_training_records": 200,
        "num_training_epoch": 500,
        "optimizer": keras.optimizers.Adam,
        "learning_rate": 0.15,
        "x_generator": self._ScatterXUniformly,
        "y_function": float,
        "num_buckets": 5,
        "monotonicities": [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
    }

    with self.assertRaises(ValueError):
      self._TrainModel(config)

  @parameterized.parameters(
      # Standard Keras regularizer:
      (
          keras.regularizers.l1_l2(l1=0.01, l2=0.001),),
      # Tuple of regularizers:
      (
          (keras.regularizers.l1_l2(
              l1=0.01, l2=0.0), keras.regularizers.l1_l2(l1=0.0, l2=0.001)),),
  )
  def testRegularizers(self, regularizer):
    if self._disable_all:
      return
    config = {
        "num_training_records": 20,
        "num_training_epoch": 0,
        "optimizer": keras.optimizers.Adam,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": lambda _: 2.0,
        "kernel_regularizer": regularizer,
        "num_buckets": 3,
        "output_min": 0.0,
        "output_max": 4.0,
    }
    loss = self._TrainModel(config)
    # This loss is pure regularization loss because initializer matches target
    # function and there was 0 training epochs.
    self.assertAlmostEqual(loss, 0.072, delta=self._loss_eps)

  def testOutputShape(self):
    if self._disable_all:
      return

    # Not Splitting
    units = 10
    input_shape, output_shape = (units,), (None, units)
    input_a = keras.layers.Input(shape=input_shape)
    cat_cal_0 = categorical_calibraion.CategoricalCalibration(
        num_buckets=3, units=units)
    output = cat_cal_0(input_a)
    self.assertAllEqual(output_shape,
                        cat_cal_0.compute_output_shape(input_a.shape))
    self.assertAllEqual(output_shape, output.shape)

    # Splitting
    output_shape = [(None, 1)] * units
    cat_cal_1 = categorical_calibraion.CategoricalCalibration(
        num_buckets=3, units=units, split_outputs=True)
    output = cat_cal_1(input_a)
    self.assertAllEqual(output_shape,
                        cat_cal_1.compute_output_shape(input_a.shape))
    self.assertAllEqual(output_shape, [o.shape for o in output])


if __name__ == "__main__":
  tf.test.main()
