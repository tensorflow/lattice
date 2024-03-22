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
"""Tests for PWL calibration layer.

This test should be run with "-c opt" since otherwise it's slow.
Also, to only run a subset of the tests (useful when developing a new test or
set of tests), change the initialization of the _disable_all boolean to 'True'
in the SetUp method, and comment out the check for this boolean in those tests
that you want to run.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_lattice.python import parallel_combination_layer as parallel_combination
from tensorflow_lattice.python import pwl_calibration_layer as keras_layer
from tensorflow_lattice.python import test_utils
from tensorflow_lattice.python import utils
# pylint: disable=g-import-not-at-top
# Use Keras 2.
version_fn = getattr(tf.keras, "version", None)
if version_fn and version_fn().startswith("3."):
  import tf_keras as keras
else:
  keras = tf.keras


class CalibrateWithSeparateMissing(keras.layers.Layer):
  """Create separate is_missing tensor.

  Splits input tensor into list: [input_tensor, is_missing_tensor] and passes
  this list as input to given calibration layer.
  """

  def __init__(self, calibration_layer, missing_input_value):
    super(CalibrateWithSeparateMissing, self).__init__()
    self.calibration_layer = calibration_layer
    self.missing_input_value = missing_input_value

  def call(self, x):
    is_missing = tf.cast(
        tf.equal(x, self.missing_input_value), dtype=tf.float32)
    return self.calibration_layer([x, is_missing])


class PwlCalibrationLayerTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(PwlCalibrationLayerTest, self).setUp()
    self._disable_all = False
    self._loss_eps = 0.0001
    self._small_eps = 1e-6
    keras.utils.set_random_seed(42)

  def _ResetAllBackends(self):
    keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

  def _ScatterXUniformly(self, units, num_points, input_min, input_max,
                         missing_probability, missing_input_value):
    """Randomly uniformly scatters points across input space."""
    np.random.seed(41)
    x = [
        input_min + np.random.random(units) * (input_max - input_min)
        for _ in range(num_points)
    ]
    if missing_probability > 0.0:
      is_missings = np.random.random([num_points, units]) < missing_probability
      x = [
          is_missing * missing_input_value + (1. - is_missing) * point
          for point, is_missing in zip(x, is_missings)
      ]
    x.sort(key=np.sum)
    return x

  def _ScatterXUniformlyIncludeBounds(self, units, **kwargs):
    """Same as _ScatterXUniformly() but includes bounds."""
    x = self._ScatterXUniformly(units, **kwargs)
    x[0] = np.array([kwargs["input_min"]] * units)
    x[-1] = np.array([kwargs["input_max"]] * units)
    return x

  def _SmallWaves(self, x):
    return np.mean(
        np.power(x, 3) + 0.1 * np.sin(x * math.pi * 8), keepdims=True)

  def _SmallWavesPlusOne(self, x):
    return self._SmallWaves(x) + 1.0

  def _WavyParabola(self, x):
    return np.mean(
        np.power(x, 2) + 0.1 * np.sin(x * math.pi * 8) - 0.5, keepdims=True)

  def _SinCycle(self, x):
    # Almost entire cycle of sin.
    return np.mean(np.sin(x / 26.0 * (2.0 * math.pi)), keepdims=True)

  def _GenPWLFunction(self, input_keypoints, pwl_weights):
    """Returns python function equivalent to PWL calibration layer.

    Output of returned function is equivalent ot output of PWL calibration layer
    with keypoints being 'input_keypoints' and learned weights being
    'pwl_weights'.

    Args:
      input_keypoints: list of keypoints of PWL calibration layer.
      pwl_weights: list of weights of PWL calibration layer.
    """

    def Pwl(x):
      result = pwl_weights[0]
      for begin, end, weight in zip(input_keypoints[0:-1], input_keypoints[1:],
                                    pwl_weights[1:]):
        result += weight * np.maximum(
            np.minimum((x - begin) / (end - begin), 1.0), 0.0)
      return np.mean(result, keepdims=True)

    return Pwl

  def _SetDefaults(self, config):
    config.setdefault("units", 1)
    config.setdefault("use_multi_calibration_layer", False)
    config.setdefault("one_d_input", False)
    config.setdefault("use_separate_missing", False)
    config.setdefault("output_min", None)
    config.setdefault("output_max", None)
    config.setdefault("missing_input_value", None)
    config.setdefault("missing_output_value", None)
    config.setdefault("monotonicity", 0)
    config.setdefault("convexity", 0)
    config.setdefault("is_cyclic", False)
    config.setdefault("clamp_min", False)
    config.setdefault("clamp_max", False)
    config.setdefault("initializer", "equal_heights")
    config.setdefault("kernel_regularizer", None)
    config.setdefault("impute_missing", False)
    config.setdefault("missing_probability", 0.0)
    config.setdefault("num_projection_iterations", 8)
    config.setdefault("constraint_assertion_eps", 1e-6)
    config.setdefault("model_dir", "/tmp/test_pwl_model_dir/")
    config.setdefault("dtype", tf.float32)
    config.setdefault("input_keypoints_type", "fixed")

    if "input_keypoints" not in config:
      # If "input_keypoints" are provided - other params referred by code below
      # might be not available, so we make sure it exists before executing
      # this code.
      config.setdefault(
          "input_keypoints",
          np.linspace(
              start=config["input_min"],
              stop=config["input_max"],
              num=config["num_keypoints"]))
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

    # The input to the model can either be single or multi dimensional.
    input_units = 1 if config["one_d_input"] else config["units"]

    training_inputs = config["x_generator"](
        units=input_units,
        num_points=config["num_training_records"],
        input_min=config["input_keypoints"][0],
        input_max=config["input_keypoints"][-1],
        missing_probability=config["missing_probability"],
        missing_input_value=config["missing_input_value"])
    training_labels = [config["y_function"](x) for x in training_inputs]

    # Either create multiple PWLCalibration layers and combine using a
    # ParallelCombination layer, or create a single PWLCalibration with multiple
    # output dimensions.
    if config["use_multi_calibration_layer"]:
      num_calibration_layers = config["units"]
      pwl_calibration_units = 1
    else:
      num_calibration_layers = 1
      pwl_calibration_units = config["units"]

    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=[input_units], dtype=tf.float32))
    calibration_layers = []
    for _ in range(num_calibration_layers):
      calibration_layers.append(
          keras_layer.PWLCalibration(
              units=pwl_calibration_units,
              dtype=tf.float32,
              input_keypoints=config["input_keypoints"],
              output_min=config["output_min"],
              output_max=config["output_max"],
              clamp_min=config["clamp_min"],
              clamp_max=config["clamp_max"],
              monotonicity=config["monotonicity"],
              convexity=config["convexity"],
              is_cyclic=config["is_cyclic"],
              kernel_initializer=config["initializer"],
              kernel_regularizer=config["kernel_regularizer"],
              impute_missing=config["impute_missing"],
              missing_output_value=config["missing_output_value"],
              missing_input_value=config["missing_input_value"],
              num_projection_iterations=config["num_projection_iterations"],
              input_keypoints_type=config["input_keypoints_type"]))
    if len(calibration_layers) == 1:
      if config["use_separate_missing"]:
        model.add(
            CalibrateWithSeparateMissing(
                calibration_layer=calibration_layers[0],
                missing_input_value=config["missing_input_value"]))
      else:
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
        config=config, training_data=training_data, keras_model=model
    )

    assetion_ops = []
    for calibration_layer in calibration_layers:
      assetion_ops.extend(
          calibration_layer.assert_constraints(
              eps=config["constraint_assertion_eps"]))
    if not tf.executing_eagerly() and assetion_ops:
      tf.compat.v1.keras.backend.get_session().run(assetion_ops)

    return loss

  def _InverseAndTrain(self, config):
    """Changes monotonicity directions to opposite and trains model."""
    inversed_config = dict(config)
    inversed_config["y_function"] = lambda x: -config["y_function"](x)

    inversed_config["output_max"] = config["output_min"]
    if inversed_config["output_max"] is not None:
      inversed_config["output_max"] = inversed_config["output_max"] * -1.0

    inversed_config["output_min"] = config["output_max"]
    if inversed_config["output_min"] is not None:
      inversed_config["output_min"] = inversed_config["output_min"] * -1.0

    inversed_config["clamp_min"] = config["clamp_max"]
    inversed_config["clamp_max"] = config["clamp_min"]
    inversed_config["monotonicity"] = -utils.canonicalize_monotonicity(
        config["monotonicity"])
    inversed_config["convexity"] = -utils.canonicalize_convexity(
        config["convexity"])
    inversed_loss = self._TrainModel(inversed_config)
    return inversed_loss

  def _CreateTrainingData(self, config):
    training_inputs = config["x_generator"](
        units=config["units"],
        num_points=config["num_training_records"],
        input_min=config["input_keypoints"][0],
        input_max=config["input_keypoints"][-1],
        missing_probability=config["missing_probability"],
        missing_input_value=config["missing_input_value"])
    training_labels = [config["y_function"](x) for x in training_inputs]
    training_inputs = tf.convert_to_tensor(training_inputs, dtype=tf.float32)
    training_labels = tf.convert_to_tensor(training_labels, dtype=tf.float32)
    return (training_inputs, training_labels)

  def _CreateKerasLayer(self, config):
    missing_input_value = config["missing_input_value"]
    if config["use_separate_missing"]:
      # We use 'config["missing_input_value"]' to create the is_missing tensor,
      # and we want the model to use the is_missing tensor so we don't pass
      # a missing_input_value to the model.
      missing_input_value = None
    return keras_layer.PWLCalibration(
        input_keypoints=config["input_keypoints"],
        units=config["units"],
        output_min=config["output_min"],
        output_max=config["output_max"],
        clamp_min=config["clamp_min"],
        clamp_max=config["clamp_max"],
        monotonicity=config["monotonicity"],
        convexity=config["convexity"],
        is_cyclic=config["is_cyclic"],
        kernel_initializer=config["initializer"],
        kernel_regularizer=config["kernel_regularizer"],
        impute_missing=config["impute_missing"],
        missing_output_value=config["missing_output_value"],
        missing_input_value=missing_input_value,
        num_projection_iterations=config["num_projection_iterations"],
        dtype=config["dtype"])

  @parameterized.parameters(
      (1, False, 0.001022, "fixed"),
      (3, False, 0.000543, "fixed"),
      (3, True, 0.000987, "fixed"),
      (1, False, 0.000393, "learned_interior"),
      (3, False, 0.000427, "learned_interior"),
      (3, True, 0.000577, "learned_interior"),
  )
  def testUnconstrainedNoMissingValue(self, units, one_d_input, expected_loss,
                                      input_keypoints_type):
    if self._disable_all:
      return
    config = {
        "units": units,
        "one_d_input": one_d_input,
        "num_training_records": 100,
        "num_training_epoch": 2000,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 0.15,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SmallWaves,
        "monotonicity": 0,
        "num_keypoints": 21,
        "input_min": -1.0,
        "input_max": 1.0,
        "output_min": None,
        "output_max": None,
        "input_keypoints_type": input_keypoints_type,
    }
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)
    if units > 1 and not one_d_input:
      config["use_multi_calibration_layer"] = True
      loss = self._TrainModel(config)
      self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)

  @parameterized.parameters(
      (1, None, 0.000858),
      (1, 0.5, 0.637769),
      (3, None, 0.000471),
      (3, 0.5, 0.190513),
  )
  def testUnconstrainedWithMissingValue(self, units, missing_output_value,
                                        expected_loss):
    if self._disable_all:
      return
    config = {
        "units": units,
        "num_training_records": 100,
        "num_training_epoch": 2000,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 0.15,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SmallWaves,
        "monotonicity": 0,
        "num_keypoints": 21,
        "input_min": -1.0,
        "input_max": 1.0,
        "output_min": None,
        "output_max": None,
        "impute_missing": True,
        "missing_input_value": -1.2,
        "missing_output_value": missing_output_value,
        "missing_probability": 0.1,
    }
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)
    config["use_separate_missing"] = True
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)
    if units > 1:
      config["use_multi_calibration_layer"] = True
      loss = self._TrainModel(config)
      self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)

  @parameterized.parameters(
      (1, -1.5, 1.5, keras.optimizers.SGD, 2100, 0.002957),
      (1, -1.5, 1.5, keras.optimizers.Adagrad, 2100, 0.002798),
      # TODO: Something really weird is going on here with Adam
      # optimizer in case when num_training_epoch is exactly 2010.
      # Test verifies result with 2100 epochs which behaves as expected.
      (1, -1.5, 1.5, keras.optimizers.Adam, 2100, 0.000769),
      (1, -0.5, 0.5, keras.optimizers.SGD, 200, 0.011483),
      (1, -0.5, 0.5, keras.optimizers.Adagrad, 200, 0.011645),
      (1, -0.5, 0.5, keras.optimizers.Adam, 200, 0.011116),
      (3, -1.5, 1.5, keras.optimizers.Adagrad, 2100, 0.001759),
      (3, -0.5, 0.5, keras.optimizers.Adagrad, 200, 0.005986),
  )
  def testNonMonotonicFunction(self, units, output_min, output_max, optimizer,
                               num_training_epoch, expected_loss):
    if self._disable_all:
      return
    config = {
        "units": units,
        "num_training_records": 100,
        "num_training_epoch": 2100,
        "optimizer": keras.optimizers.SGD,
        "learning_rate": 0.015,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SmallWaves,
        "monotonicity": 0,
        "num_keypoints": 21,
        "input_min": -1.0,
        "input_max": 1.0,
        "output_min": -1.5,
        "output_max": 1.5,
        "clamp_min": False,
        "clamp_max": False,
    }
    config["output_min"] = output_min
    config["output_max"] = output_max
    config["optimizer"] = optimizer
    config["num_training_epoch"] = num_training_epoch
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)
    if units > 1:
      config["use_multi_calibration_layer"] = True
      loss = self._TrainModel(config)
      self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)

  @parameterized.parameters(
      (1, -1.5, 0.287357),
      (1, 1.5, 0.287357),
      (3, -1.5, 0.122801),
      (3, 1.5, 0.106150),
  )
  # Since function is symmetric result should be same for both values above.
  def testBoundsForMissing(self, units, missing_input_value, expected_loss):
    if self._disable_all:
      return
    config = {
        "units": units,
        "num_training_records": 100,
        "num_training_epoch": 200,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 0.15,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SmallWaves,
        "monotonicity": 1,
        "num_keypoints": 21,
        "input_min": -1.0,
        "input_max": 1.0,
        "output_min": -2.0,
        "output_max": 2.0,
        "clamp_min": False,
        "clamp_max": True,
        "impute_missing": True,
        "missing_probability": 0.1,
    }
    config["missing_input_value"] = missing_input_value
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)
    if units > 1:
      config["use_multi_calibration_layer"] = True
      loss = self._TrainModel(config)
      self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)

  @parameterized.parameters(
      (1, None, None, 0.002505),
      (1, None, 1.21, 0.008076),
      (1, None, 1.6, 0.000251),
      (1, None, 2.0, 0.001107),
      (1, 0.5, None, 0.000790),
      (1, 0.5, 1.21, 0.008353),
      (1, 0.5, 1.6, 0.000685),
      (1, 0.5, 2.0, 0.000694),
      (1, 0.9, None, 0.000143),
      (1, 0.9, 1.21, 0.008108),
      (1, 0.9, 1.6, 0.000125),
      (1, 0.9, 2.0, 0.000120),
      (1, 1.2, None, 0.025762),
      (1, 1.2, 1.21, 0.026069),
      (1, 1.2, 1.6, 0.025240),
      (1, 1.2, 2.0, 0.024802),
      (3, None, None, 0.003268),
      (3, None, 1.21, 0.003901),
      (3, None, 1.6, 0.000897),
      (3, None, 2.0, 0.002608),
      (3, 0.5, None, 0.000945),
      (3, 0.5, 1.21, 0.004830),
      (3, 0.5, 1.6, 0.000945),
      (3, 0.5, 2.0, 0.000923),
      (3, 0.9, None, 0.000318),
      (3, 0.9, 1.21, 0.004215),
      (3, 0.9, 1.6, 0.000335),
      (3, 0.9, 2.0, 0.000297),
      (3, 1.2, None, 0.011354),
      (3, 1.2, 1.21, 0.011354),
      (3, 1.2, 1.6, 0.011354),
      (3, 1.2, 2.0, 0.011354),
  )
  def testAllBoundsWithoutMonotonicityConstraints(self, units, output_min,
                                                  output_max, expected_loss):
    if self._disable_all:
      return
    config = {
        "units": units,
        "num_training_records": 100,
        "num_training_epoch": 200,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 0.15,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SmallWavesPlusOne,
        "monotonicity": 0,
        "num_keypoints": 21,
        "input_min": 0.1,
        "input_max": 0.8,
        "clamp_min": False,
        "clamp_max": False,
    }
    config["output_min"] = output_min
    config["output_max"] = output_max
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)
    if units > 1:
      config["use_multi_calibration_layer"] = True
      loss = self._TrainModel(config)
      self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)

  @parameterized.parameters(
      (1, False, keras.optimizers.SGD, 0.004715),
      (1, False, keras.optimizers.Adagrad, 0.003820),
      (1, False, keras.optimizers.Adam, 0.002797),
      (1, True, keras.optimizers.SGD, 0.004427),
      (1, True, keras.optimizers.Adagrad, 0.004084),
      # Adam is doing terrible when required to stretch monotonic function
      # even if bounds are proper.
      (1, True, keras.optimizers.Adam, 0.065664),
      (3, False, keras.optimizers.Adagrad, 0.002371),
      (3, True, keras.optimizers.Adagrad, 0.002670),
  )
  def testMonotonicProperBounds(self, units, is_clamped, optimizer,
                                expected_loss):
    if self._disable_all:
      return
    config = {
        "units": units,
        "num_training_records": 100,
        "num_training_epoch": 400,
        "optimizer": optimizer,
        "learning_rate": 0.015,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SmallWaves,
        "monotonicity": "increasing",
        "num_keypoints": 21,
        "input_min": -1.0,
        "input_max": 1.0,
        "output_min": -1.0,
        "output_max": 1.0,
        "clamp_min": is_clamped,
        "clamp_max": is_clamped,
    }
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)
    if units > 1:
      config["use_multi_calibration_layer"] = True
      loss = self._TrainModel(config)
      self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)

  @parameterized.parameters(
      (1, False, keras.optimizers.SGD, 0.15, 0.009563),
      (1, False, keras.optimizers.Adagrad, 0.015, 0.011117),
      (1, False, keras.optimizers.Adam, 0.015, 0.015356),
      (1, True, keras.optimizers.SGD, 0.15, 0.009563),
      (1, True, keras.optimizers.Adagrad, 0.015, 0.011117),
      # Adam squeezes monotonic function just slightly worse than adagrad.
      (1, True, keras.optimizers.Adam, 0.015, 0.015189),
      (3, False, keras.optimizers.Adagrad, 0.015, 0.006057),
      (3, True, keras.optimizers.Adagrad, 0.015, 0.006049),
  )
  def testMonotonicNarrowBounds(self, units, is_clamped, optimizer,
                                learning_rate, expected_loss):
    if self._disable_all:
      return
    config = {
        "units": units,
        "num_training_records": 100,
        "num_training_epoch": 200,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SmallWaves,
        "monotonicity": 1,
        "num_keypoints": 21,
        "input_min": -1.0,
        "input_max": 1.0,
        "output_min": -0.5,
        "output_max": 0.5,
        "clamp_min": is_clamped,
        "clamp_max": is_clamped,
    }
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)
    if units > 1:
      config["use_multi_calibration_layer"] = True
      loss = self._TrainModel(config)
      self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)

  @parameterized.parameters(
      (1, False, keras.optimizers.SGD, 0.005920),
      (1, False, keras.optimizers.Adagrad, 0.006080),
      (1, False, keras.optimizers.Adam, 0.002914),
      (1, True, keras.optimizers.SGD, 0.013836),
      (1, True, keras.optimizers.Adagrad, 0.066928),
      # Adam is doing terrible when required to stretch monotonic function.
      (1, True, keras.optimizers.Adam, 0.230402),
      (3, False, keras.optimizers.Adagrad, 0.004891),
      (3, True, keras.optimizers.Adagrad, 0.021490),
  )
  def testMonotonicWideBounds(self, units, is_clamped, optimizer,
                              expected_loss):
    if self._disable_all:
      return
    config = {
        "units": units,
        "num_training_records": 100,
        "num_training_epoch": 400,
        "optimizer": optimizer,
        "learning_rate": 0.015,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SmallWaves,
        "monotonicity": 1,
        "num_keypoints": 21,
        "input_min": -1.0,
        "input_max": 1.0,
        "output_min": -1.5,
        "output_max": 1.5,
        "clamp_min": is_clamped,
        "clamp_max": is_clamped,
    }
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)
    if units > 1:
      config["use_multi_calibration_layer"] = True
      loss = self._TrainModel(config)
      self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)

  @parameterized.parameters(
      (1, None, None, False, False, 0.003744),
      (1, None, None, False, True, 0.003744),
      (1, None, 1.6, True, False, 0.001456),
      (1, None, 1.6, True, True, 0.001465),
      (1, None, 2.0, False, False, 0.001712),
      (1, None, 2.0, False, True, 0.01623),
      (1, None, 2.0, True, False, 0.001712),
      (1, None, 2.0, True, True, 0.01623),
      (1, 0.5, None, False, False, 0.002031),
      (1, 0.5, None, False, True, 0.002031),
      (1, 0.5, None, True, False, 0.003621),
      (1, 0.5, None, True, True, 0.003621),
      (1, None, None, True, False, 0.003744),
      (1, 0.5, 1.21, False, False, 0.007572),
      (1, 0.5, 1.21, False, True, 0.007572),
      (1, 0.5, 1.21, True, False, 0.009876),
      (1, 0.5, 1.21, True, True, 0.009876),
      (1, 0.5, 1.6, False, False, 0.001916),
      (1, 0.5, 1.6, False, True, 0.001737),
      (1, 0.5, 1.6, True, False, 0.003103),
      (1, 0.5, 1.6, True, True, 0.002692),
      (1, 0.5, 2.0, False, False, 0.001873),
      (1, 0.5, 2.0, False, True, 0.003333),
      (1, None, None, True, True, 0.003744),
      (1, 0.5, 2.0, True, False, 0.003315),
      (1, 0.5, 2.0, True, True, 0.004289),
      (1, 0.9, None, False, False, 0.00151),
      (1, 0.9, None, False, True, 0.00151),
      (1, 0.9, None, True, False, 0.001552),
      (1, 0.9, None, True, True, 0.001552),
      (1, 0.9, 1.21, False, False, 0.005387),
      (1, 0.9, 1.21, False, True, 0.005387),
      (1, 0.9, 1.21, True, False, 0.005427),
      (1, 0.9, 1.21, True, True, 0.005427),
      (1, None, 1.21, False, False, 0.005366),
      (1, 0.9, 1.6, False, False, 0.0015),
      (1, 0.9, 1.6, False, True, 0.001454),
      (1, 0.9, 1.6, True, False, 0.001546),
      (1, 0.9, 1.6, True, True, 0.001514),
      (1, 0.9, 2.0, False, False, 0.001501),
      (1, 0.9, 2.0, False, True, 0.003067),
      (1, 0.9, 2.0, True, False, 0.001547),
      (1, 0.9, 2.0, True, True, 0.00312),
      (1, 1.2, None, False, False, 0.021835),
      (1, 1.2, None, False, True, 0.021835),
      (1, None, 1.21, False, True, 0.005366),
      (1, 1.2, None, True, False, 0.021835),
      (1, 1.2, None, True, True, 0.021835),
      (1, 1.2, 1.21, False, False, 0.025733),
      (1, 1.2, 1.21, False, True, 0.025733),
      (1, 1.2, 1.21, True, False, 0.025733),
      (1, 1.2, 1.21, True, True, 0.025733),
      (1, 1.2, 1.6, False, False, 0.021834),
      (1, 1.2, 1.6, False, True, 0.021967),
      (1, 1.2, 1.6, True, False, 0.021834),
      (1, 1.2, 1.6, True, True, 0.021967),
      (1, None, 1.21, True, False, 0.005366),
      (1, 1.2, 2.0, False, False, 0.021834),
      (1, 1.2, 2.0, False, True, 0.023642),
      (1, 1.2, 2.0, True, False, 0.021834),
      (1, 1.2, 2.0, True, True, 0.023642),
      (1, None, 1.21, True, True, 0.005366),
      (1, None, 1.6, False, False, 0.001456),
      (1, None, 1.6, False, True, 0.001465),
      (3, None, None, False, False, 0.003969),
      (3, None, None, False, True, 0.003969),
      (3, 0.5, None, True, False, 0.003125),
      (3, 0.5, None, True, True, 0.003125),
      (3, None, None, True, False, 0.003969),
      (3, 0.5, 1.21, False, False, 0.003676),
      (3, 0.5, 1.21, False, True, 0.003676),
      (3, 0.5, 1.21, True, False, 0.006550),
      (3, 0.5, 1.21, True, True, 0.006550),
      (3, 0.5, 1.6, False, False, 0.001246),
      (3, 0.5, 1.6, False, True, 0.001000),
      (3, 0.5, 1.6, True, False, 0.002775),
      (3, None, 1.6, True, False, 0.000662),
      (3, 0.5, 1.6, True, True, 0.002720),
      (3, 0.5, 2.0, False, False, 0.001272),
      (3, 0.5, 2.0, False, True, 0.001779),
      (3, None, None, True, True, 0.003969),
      (3, 0.5, 2.0, True, False, 0.002852),
      (3, 0.5, 2.0, True, True, 0.003496),
      (3, 0.9, None, False, False, 0.000597),
      (3, 0.9, None, False, True, 0.000597),
      (3, 0.9, None, True, False, 0.000678),
      (3, 0.9, None, True, True, 0.000678),
      (3, None, 1.6, True, True, 0.000640),
      (3, 0.9, 1.21, False, False, 0.002630),
      (3, 0.9, 1.21, False, True, 0.002630),
      (3, 0.9, 1.21, True, False, 0.002906),
      (3, 0.9, 1.21, True, True, 0.002906),
      (3, None, 1.21, False, False, 0.002565),
      (3, 0.9, 1.6, False, False, 0.000575),
      (3, 0.9, 1.6, False, True, 0.000520),
      (3, 0.9, 1.6, True, False, 0.000648),
      (3, 0.9, 1.6, True, True, 0.000606),
      (3, 0.9, 2.0, False, False, 0.000556),
      (3, None, 2.0, False, False, 0.000901),
      (3, 0.9, 2.0, False, True, 0.001230),
      (3, 0.9, 2.0, True, False, 0.000636),
      (3, 0.9, 2.0, True, True, 0.001314),
      (3, 1.2, None, False, False, 0.010638),
      (3, 1.2, None, False, True, 0.010638),
      (3, None, 1.21, False, True, 0.002565),
      (3, 1.2, None, True, False, 0.010638),
      (3, 1.2, None, True, True, 0.010638),
      (3, 1.2, 1.21, False, False, 0.011300),
      (3, 1.2, 1.21, False, True, 0.011309),
      (3, None, 2.0, False, True, 0.003166),
      (3, 1.2, 1.21, True, False, 0.011300),
      (3, 1.2, 1.21, True, True, 0.011309),
      (3, 1.2, 1.6, False, False, 0.010631),
      (3, 1.2, 1.6, False, True, 0.012681),
      (3, 1.2, 1.6, True, False, 0.010631),
      (3, 1.2, 1.6, True, True, 0.012681),
      (3, None, 1.21, True, False, 0.002565),
      (3, 1.2, 2.0, False, False, 0.010627),
      (3, 1.2, 2.0, False, True, 0.016435),
      (3, 1.2, 2.0, True, False, 0.010627),
      (3, None, 2.0, True, False, 0.000901),
      (3, 1.2, 2.0, True, True, 0.016435),
      (3, None, 1.21, True, True, 0.002565),
      (3, None, 1.6, False, False, 0.000662),
      (3, None, 1.6, False, True, 0.000640),
      (3, None, 2.0, True, True, 0.003166),
      (3, 0.5, None, False, False, 0.001334),
      (3, 0.5, None, False, True, 0.001334),
  )
  def testAllBoundsAndMonotonicityDirection(self, units, output_min, output_max,
                                            clamp_min, clamp_max,
                                            expected_loss):
    if self._disable_all:
      return
    config = {
        "units": units,
        "num_training_records": 100,
        "num_training_epoch": 200,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 0.15,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SmallWavesPlusOne,
        "monotonicity": 1,
        "num_keypoints": 21,
        "input_min": 0.1,
        "input_max": 0.8,
        "output_min": output_min,
        "output_max": output_max,
        "clamp_min": clamp_min,
        "clamp_max": clamp_max,
    }
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)
    self.assertAlmostEqual(
        loss, self._InverseAndTrain(config), delta=self._small_eps)
    if units > 1:
      config["use_multi_calibration_layer"] = True
      loss = self._TrainModel(config)
      self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)
      self.assertAlmostEqual(
          loss, self._InverseAndTrain(config), delta=self._small_eps)

  @parameterized.parameters(
      (1, 1, 0.018919),
      (1, -1, 0.019434),
      (3, "convex", 0.008592),
      (3, "concave", 0.01134),
  )
  def testConvexitySimple(self, units, convexity, expected_loss):
    # No constraints other than convexity.
    if self._disable_all:
      return
    config = {
        "units": units,
        "num_training_records": 100,
        "num_training_epoch": 120,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SmallWaves,
        "monotonicity": "none",
        "convexity": convexity,
        "num_keypoints": 21,
        "input_min": -1.0,
        "input_max": 1.0,
        "output_min": None,
        "output_max": None,
        "num_projection_iterations": 18,
    }
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)
    if units > 1:
      config["use_multi_calibration_layer"] = True
      loss = self._TrainModel(config)
      self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)

  @parameterized.parameters(
      (1, 1, 0.006286),
      (1, -1, 0.078076),
      (3, 1, 0.002941),
      (3, -1, 0.032497),
  )
  def testConvexityNonUniformKeypoints(self, units, convexity, expected_loss):
    # No constraints other than convexity.
    if self._disable_all:
      return

    config = {
        "units": units,
        "num_training_records": 100,
        "num_training_epoch": 200,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._WavyParabola,
        "monotonicity": 0,
        "convexity": convexity,
        "input_keypoints": [-1.0, -0.9, -0.3, -0.2, 0.0, 0.3, 0.31, 0.35, 1.0],
        "output_min": None,
        "output_max": None,
    }
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)
    if units > 1:
      config["use_multi_calibration_layer"] = True
      loss = self._TrainModel(config)
      self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)

  @parameterized.parameters(
      (1, 2, 0.033706),
      (1, 3, 0.006485),
      (1, 4, 0.005128),
      (1, 5, 0.004878),
      (1, 6, 0.005083),
      (1, 7, 0.004860),
      (3, 2, 0.013585),
      (3, 3, 0.003311),
      (3, 4, 0.002633),
      (3, 5, 0.001909),
      (3, 6, 0.001822),
      (3, 7, 0.001599),
  )
  def testConvexityDifferentNumKeypoints(self, units, num_keypoints,
                                         expected_loss):
    # No constraints other than convexity.
    if self._disable_all:
      return

    config = {
        "units": units,
        "num_training_records": 100,
        "num_training_epoch": 120,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 0.3,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._WavyParabola,
        "monotonicity": 0,
        "convexity": 1,
        "num_keypoints": num_keypoints,
        "input_min": -0.8,
        "input_max": 0.8,
        "output_min": None,
        "output_max": None,
    }
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)
    if units > 1:
      config["use_multi_calibration_layer"] = True
      loss = self._TrainModel(config)
      self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)

  @parameterized.parameters(
      (1, "increasing", None, 0.055837),
      (1, "decreasing", None, 0.046657),
      (1, "none", 0.0, 0.027777),
      (1, "increasing", 0.0, 0.065516),
      (1, "decreasing", 0.0, 0.057453),
      (3, "increasing", None, 0.022467),
      (3, "decreasing", None, 0.019012),
      (3, "none", 0.0, 0.014693),
      (3, "increasing", 0.0, 0.026284),
      (3, "decreasing", 0.0, 0.025498),
  )
  def testConvexityWithMonotonicityAndBounds(self, units, monotonicity,
                                             output_max, expected_loss):
    if self._disable_all:
      return

    config = {
        "units": units,
        "num_training_records": 100,
        "num_training_epoch": 120,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 0.5,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._WavyParabola,
        "monotonicity": monotonicity,
        "convexity": 1,
        "num_keypoints": 21,
        "input_min": -1.0,
        "input_max": 1.0,
        "output_min": None,
        "output_max": output_max,
        "num_projection_iterations": 8,
    }
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)
    self.assertAlmostEqual(
        loss, self._InverseAndTrain(config), delta=self._small_eps)
    if units > 1:
      config["use_multi_calibration_layer"] = True
      loss = self._TrainModel(config)
      self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)
      self.assertAlmostEqual(
          loss, self._InverseAndTrain(config), delta=self._small_eps)

  @parameterized.parameters(
      ([-1.0, -0.8, 0.0, 0.2, 0.8, 1.0],),
      (np.array([-1.0, -0.8, 0.0, 0.2, 0.8, 1.0]),),
  )
  def testInputKeypoints(self, keypoints):
    if self._disable_all:
      return
    config = {
        "num_training_records": 100,
        "num_training_epoch": 200,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 0.15,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SmallWaves,
        "monotonicity": 0,
        "input_keypoints": keypoints,
        "output_min": None,
        "output_max": None,
    }
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.009650, delta=self._loss_eps)

  @parameterized.parameters(
      (1, None, 600, 0.002058),
      (1, ("laplacian", 0.01, 0.0), 420, 0.040492),
      (1, ("hessian", 0.01, 0.01), 300, 0.040932),
      (1, ("wrinkle", 0.01, 0.01), 300, 0.027430),
      (3, None, 600, 0.002150),
      (3, ("laplacian", 0.01, 0.0), 420, 0.096667),
      (3, ("hessian", 0.01, 0.01), 300, 0.092306),
      (3, ("wrinkle", 0.01, 0.01), 300, 0.064053),
  )
  def testIsCyclic(self, units, regularizer, num_training_epoch, expected_loss):
    if self._disable_all:
      return
    config = {
        "units": units,
        "num_training_records": 100,
        "num_training_epoch": num_training_epoch,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 0.15,
        "x_generator": self._ScatterXUniformlyIncludeBounds,
        "y_function": self._SinCycle,
        "monotonicity": 0,
        "input_min": 0.0,
        "input_max": 24.0,
        "num_keypoints": 10,
        "is_cyclic": True,
        "kernel_regularizer": regularizer,
        "output_min": None,
        "output_max": None,
    }
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)
    if units > 1:
      config["use_multi_calibration_layer"] = True
      loss = self._TrainModel(config)
      self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)

  @parameterized.parameters(
      (1, "equal_heights", 0.332572),
      (1, "equal_slopes", 0.476452),
      (3, "equal_heights", 0.271896),
      (3, "equal_slopes", 0.356754),
  )
  def testInitializer(self, units, initializer, expected_loss):
    if self._disable_all:
      return
    config = {
        "units": units,
        "num_training_records": 100,
        # 0 training epochs to see pure output of initializer.
        "num_training_epoch": 0,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 0.15,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SmallWaves,
        "monotonicity": 0,
        "input_keypoints": [-1.0, -0.8, 0.0, 0.2, 0.8, 1.0],
        "output_min": -1.0,
        "output_max": 2.0,
        "initializer": initializer,
    }
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)
    if units > 1:
      config["use_multi_calibration_layer"] = True
      loss = self._TrainModel(config)
      self.assertAlmostEqual(loss, expected_loss, delta=self._loss_eps)

  # TODO: this test is only using the first piece of the PWL.
  @parameterized.parameters(
      (1, ("laplacian", 0.01, 0.001), 0.091, 0.089631),
      (1, ("Hessian", 0.01, 0.001), 0.035, 0.033504),
      (1, ("wrinkle", 0.01, 0.001), 0.011, 0.007018),
      # Standard Keras regularizer:
      (1, keras.regularizers.l1_l2(l1=0.01, l2=0.001), 0.091, 0.089906),
      # List of regularizers:
      (1, [("Hessian", 0.01, 0.001),
           keras.regularizers.l1_l2(l1=0.01, l2=0.001)], 0.126, 0.122192),
      (3, ("laplacian", 0.01, 0.001), 0.273, 0.263244),
      (3, ("Hessian", 0.01, 0.001), 0.105, 0.097368),
      (3, ("wrinkle", 0.01, 0.001), 0.033, 0.013650),
      # Standard Keras regularizer:
      (3, keras.regularizers.l1_l2(l1=0.01, l2=0.001), 0.273, 0.265924),
      # List of regularizers:
      (3, [("Hessian", 0.01, 0.001),
           keras.regularizers.l1_l2(l1=0.01, l2=0.001)], 0.378, 0.354917),
  )
  def testRegularizers(self, units, regularizer, pure_reg_loss, training_loss):
    if self._disable_all:
      return
    keypoints = [0.0, 1.0, 2.0, 3.0]
    pwl_weights = [0.0, 1.0, 2.0, 4.0]
    multi_pwl_weights = [[w] * units for w in pwl_weights]
    # Keypoint outputs which correspond to weights: [0.0, 1.0, 3.0, 7.0]
    config = {
        "units": units,
        "num_training_records": 100,
        "num_training_epoch": 0,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 0.15,
        "x_generator": self._ScatterXUniformly,
        "input_keypoints": keypoints,
        "y_function": self._GenPWLFunction(keypoints, multi_pwl_weights),
        # Initializer exactly matches target function.
        "initializer":
            lambda shape, dtype: tf.constant(multi_pwl_weights, shape=shape),
        "kernel_regularizer": regularizer,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    # This loss is pure regularization loss because initializer matches target
    # function and there was 0 training epochs.
    self.assertAlmostEqual(loss, pure_reg_loss, delta=self._loss_eps)

    config["num_training_epoch"] = 20
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, training_loss, delta=self._loss_eps)
    if units > 1:
      config["use_multi_calibration_layer"] = True
      config["initializer"] = (
          lambda shape, dtype: tf.constant(pwl_weights, shape=shape))
      loss = self._TrainModel(config)
      self.assertAlmostEqual(loss, training_loss, delta=self._loss_eps)

  def testAssertMonotonicity(self):
    if self._disable_all:
      return
    decreasing_initializer = keras_layer.UniformOutputInitializer(
        output_min=0.0, output_max=1.0, monotonicity=-1)
    # Specify decreasing initializer and do 0 training iterations so no
    # projections are being executed.
    config = {
        "num_training_records": 100,
        "num_training_epoch": 0,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 0.15,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SmallWaves,
        "monotonicity": 0,
        "num_keypoints": 21,
        "input_min": 0.0,
        "input_max": 1.0,
        "output_min": 0.0,
        "output_max": 1.0,
        "initializer": decreasing_initializer,
    }
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.347888, delta=self._loss_eps)

    # We have decreasing initializer so with 0 trainig steps monotonicity is
    # violated.
    with self.assertRaises(tf.errors.InvalidArgumentError):
      config["monotonicity"] = 1
      loss = self._TrainModel(config)

    # Now set upper bound bigger than necessary. Everything should be fine...
    config["monotonicity"] = 0
    config["output_max"] = 1.5
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.347888, delta=self._loss_eps)

    # ... until we require to clamp max.
    with self.assertRaises(tf.errors.InvalidArgumentError):
      config["clamp_max"] = True
      loss = self._TrainModel(config)

  def testOutputShape(self):
    if self._disable_all:
      return

    # Not Splitting
    units = 10
    input_keypoints = [1, 2, 3, 4, 5]
    input_shape, output_shape = (units,), (None, units)
    input_a = keras.layers.Input(shape=input_shape)
    pwl_0 = keras_layer.PWLCalibration(
        input_keypoints=input_keypoints, units=units)
    output = pwl_0(input_a)
    self.assertAllEqual(output_shape, pwl_0.compute_output_shape(input_a.shape))
    self.assertAllEqual(output_shape, output.shape)

    # Splitting
    output_shape = [(None, 1)] * units
    pwl_1 = keras_layer.PWLCalibration(
        input_keypoints=input_keypoints, units=units, split_outputs=True)
    output = pwl_1(input_a)
    self.assertAllEqual(output_shape, pwl_1.compute_output_shape(input_a.shape))
    self.assertAllEqual(output_shape, [o.shape for o in output])

  @parameterized.parameters(("fixed", 1, 1), ("fixed", 1, 2), ("fixed", 2, 2),
                            ("learned_interior", 1, 1),
                            ("learned_interior", 1, 2),
                            ("learned_interior", 2, 2))
  def testKeypointsInputs(self, input_keypoints_type, input_dims, output_units):
    if self._disable_all:
      return

    input_keypoints = [0, 0.5, 1]
    expected_function_output = np.array([[0.0] * output_units,
                                         [0.5] * output_units,
                                         [1.0] * output_units])

    # Check after layer build
    pwl = keras_layer.PWLCalibration(
        input_keypoints=input_keypoints,
        units=output_units,
        input_keypoints_type=input_keypoints_type)
    pwl.build(input_shape=[10, input_dims])
    self.assertAllEqual(expected_function_output, pwl.keypoints_inputs())

    # Check after Keras model compile
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=[input_dims], dtype=tf.float32))
    model.add(pwl)
    model.compile(loss=keras.losses.mean_squared_error)
    self.assertAllEqual(expected_function_output, pwl.keypoints_inputs())

    # Check after Keras model fit; look for change in learned case.
    train_x = np.random.uniform(size=(10, input_dims))
    train_y = train_x[:, 0]**2
    model.fit(train_x, train_y, batch_size=len(train_x), epochs=5, verbose=0)
    if input_keypoints_type == "fixed":
      self.assertAllEqual(expected_function_output, pwl.keypoints_inputs())
    else:
      self.assertNotAllEqual(expected_function_output, pwl.keypoints_inputs())


if __name__ == "__main__":
  tf.test.main()
