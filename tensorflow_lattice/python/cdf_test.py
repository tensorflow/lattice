# Copyright 2021 Google LLC
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
"""Tests for cdf."""

import math

from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_lattice.python import cdf_layer
from tensorflow_lattice.python import test_utils
# pylint: disable=g-import-not-at-top
# Use Keras 2.
version_fn = getattr(tf.keras, "version", None)
if version_fn and version_fn().startswith("3."):
  import tf_keras as keras
else:
  keras = tf.keras


class CdfLayerTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(CdfLayerTest, self).setUp()
    self.disable_all = False
    self.loss_eps = 0.001
    self.small_eps = 1e-6
    keras.utils.set_random_seed(42)

  def _ResetAllBackends(self):
    keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

  def _SetDefaults(self, config):
    config.setdefault("input_dims", 1)
    config.setdefault("num_keypoints", 10)
    config.setdefault("units", 1)
    config.setdefault("activation", "relu6")
    config.setdefault("reduction", "mean")
    config.setdefault("input_scaling_init", None)
    config.setdefault("input_scaling_type", "fixed")
    config.setdefault("sparsity_factor", 1)
    config.setdefault("kernel_initializer", "random_uniform")

    return config

  def _ScatterXUniformly(self, num_points, input_dims):
    """Deterministically generates num_point random points within CDF."""
    np.random.seed(42)
    x = []
    for _ in range(num_points):
      point = [np.random.random() for _ in range(input_dims)]
      x.append(np.asarray(point))
    if input_dims == 1:
      x.sort()
    return x

  def _ScatterXUniformlyExtendedRange(self, num_points, input_dims):
    """Extends every dimension by 1.0 on both sides and generates points."""
    np.random.seed(42)
    x = []
    for _ in range(num_points):
      point = [np.random.random() * 2 for _ in range(input_dims)]
      x.append(np.asarray(point))
    if input_dims == 1:
      x.sort()
    return x

  def _TwoDMeshGrid(self, num_points, input_dims):
    """Mesh grid for visualisation of 3-d surfaces via pyplot."""
    if input_dims != 2:
      raise ValueError("2-d mesh grid is possible only for 2-d inputs. Input"
                       " dimension given: %s" % input_dims)
    return test_utils.two_dim_mesh_grid(
        num_points=num_points, x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)

  def _TwoDMeshGridExtendedRange(self, num_points, input_dims):
    """Mesh grid extended by 1.0 on every side."""
    if input_dims != 2:
      raise ValueError(
          "2-d mesh grid is possible only for 2-d lattice. Lattice")
    return test_utils.two_dim_mesh_grid(
        num_points=num_points, x_min=-1.0, y_min=-1.0, x_max=2.0, y_max=2.0)

  def _Sin(self, x):
    return math.sin(x[0])

  def _SinPlusX(self, x):
    return math.sin(x[0]) + x[0] / 3.0

  def _SinPlusXNd(self, x):
    return np.sum([math.sin(y) + y / 5.0 for y in x])

  def _SinOfSum(self, x):
    return math.sin(sum(x))

  def _Square(self, x):
    return x[0]**2

  def _ScaledSum(self, x):
    result = 0.0
    for y in x:
      result += y / len(x)
    return result

  def _GetTrainingInputsAndLabels(self, config):
    """Generates training inputs and labels.

    Args:
      config: Dictionary with config for this unit test.

    Returns:
      Tuple `(training_inputs, training_labels)` where
        `training_inputs` and `training_labels` are data for training.
    """
    raw_training_inputs = config["x_generator"](
        num_points=config["num_training_records"],
        input_dims=config["input_dims"])

    if isinstance(raw_training_inputs, tuple):
      # This means that raw inputs are 2-d mesh grid. Convert them into list of
      # 2-d points.
      training_inputs = list(np.dstack(raw_training_inputs).reshape((-1, 2)))
    else:
      training_inputs = raw_training_inputs

    training_labels = [config["y_function"](x) for x in training_inputs]
    return training_inputs, training_labels

  def _TrainModel(self, config):
    logging.info("Testing config:")
    logging.info(config)
    config = self._SetDefaults(config)
    self._ResetAllBackends()

    training_inputs, training_labels = (
        self._GetTrainingInputsAndLabels(config))

    keras_layer = cdf_layer.CDF(
        num_keypoints=config["num_keypoints"],
        units=config["units"],
        activation=config["activation"],
        reduction=config["reduction"],
        input_scaling_init=config["input_scaling_init"],
        input_scaling_type=config["input_scaling_type"],
        sparsity_factor=config["sparsity_factor"],
        kernel_initializer=config["kernel_initializer"],
        input_shape=(config["input_dims"],),
        dtype=tf.float32)
    model = keras.models.Sequential()
    model.add(keras_layer)

    # When we have multi-unit output, we average across the output units for
    # testing.
    if config["units"] > 1:
      model.add(
          keras.layers.Lambda(
              lambda x: tf.reduce_mean(x, axis=-1, keepdims=True)))

    optimizer = config["optimizer"](learning_rate=config["learning_rate"])
    model.compile(loss="mse", optimizer=optimizer)

    training_data = (training_inputs, training_labels)
    loss = test_utils.run_training_loop(
        config=config, training_data=training_data, keras_model=model
    )

    if tf.executing_eagerly():
      tf.print("final weights: ", keras_layer.kernel)

    return loss

  @parameterized.parameters(
      ("relu6", "mean", "fixed", 0.002203),
      ("relu6", "mean", "learned_shared", 0.002216),
      ("relu6", "mean", "learned_per_input", 0.002216),
      ("relu6", "geometric_mean", "fixed", 0.002176),
      ("relu6", "geometric_mean", "learned_shared", 0.002191),
      ("relu6", "geometric_mean", "learned_per_input", 0.002191),
      ("sigmoid", "mean", "fixed", 0.002451),
      ("sigmoid", "mean", "learned_shared", 0.002443),
      ("sigmoid", "mean", "learned_per_input", 0.002443),
      ("sigmoid", "geometric_mean", "fixed", 0.002419),
      ("sigmoid", "geometric_mean", "learned_shared", 0.002411),
      ("sigmoid", "geometric_mean", "learned_per_input", 0.002411),
  )
  def test1Dim(self, activation, reduction, input_scaling_type, expected_loss):
    if self.disable_all:
      return
    config = {
        "input_dims": 1,
        "activation": activation,
        "reduction": reduction,
        "input_scaling_type": input_scaling_type,
        "num_training_records": 128,
        "num_training_epoch": 50,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SinPlusX,
    }
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self.loss_eps)

  @parameterized.parameters(
      ("relu6", "mean", "fixed", 0.171249),
      ("relu6", "mean", "learned_shared", 0.170965),
      ("relu6", "mean", "learned_per_input", 0.171091),
      ("relu6", "geometric_mean", "fixed", 0.172444),
      ("relu6", "geometric_mean", "learned_shared", 0.172357),
      ("relu6", "geometric_mean", "learned_per_input", 0.172390),
      ("sigmoid", "mean", "fixed", 0.172810),
      ("sigmoid", "mean", "learned_shared", 0.172517),
      ("sigmoid", "mean", "learned_per_input", 0.172653),
      ("sigmoid", "geometric_mean", "fixed", 0.174273),
      ("sigmoid", "geometric_mean", "learned_shared", 0.174110),
      ("sigmoid", "geometric_mean", "learned_per_input", 0.174110),
  )
  def test2Dim(self, activation, reduction, input_scaling_type, expected_loss):
    if self.disable_all:
      return
    config = {
        "input_dims": 2,
        "activation": activation,
        "reduction": reduction,
        "input_scaling_type": input_scaling_type,
        "num_training_records": 900,
        "num_training_epoch": 100,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._TwoDMeshGrid,
        "y_function": self._SinPlusXNd,
    }
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self.loss_eps)

  @parameterized.parameters(
      ("relu6", "mean", "fixed", 0.000156),
      ("relu6", "mean", "learned_shared", 0.000144),
      ("relu6", "mean", "learned_per_input", 0.000154),
      ("relu6", "geometric_mean", "fixed", 0.000988),
      ("relu6", "geometric_mean", "learned_shared", 0.000942),
      ("relu6", "geometric_mean", "learned_per_input", 0.000977),
      ("sigmoid", "mean", "fixed", 0.000078),
      ("sigmoid", "mean", "learned_shared", 0.000078),
      ("sigmoid", "mean", "learned_per_input", 0.0),
      ("sigmoid", "geometric_mean", "fixed", 0.000793),
      ("sigmoid", "geometric_mean", "learned_shared", 0.000794),
      ("sigmoid", "geometric_mean", "learned_per_input", 0.000793),
  )
  def test5DimScaledSum(self, activation, reduction, input_scaling_type,
                        expected_loss):
    if self.disable_all:
      return
    config = {
        "input_dims": 5,
        "activation": activation,
        "reduction": reduction,
        "input_scaling_type": input_scaling_type,
        "num_training_records": 200,
        "num_training_epoch": 200,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._ScaledSum,
    }
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self.loss_eps)

  @parameterized.parameters(
      ("relu6", "mean", "fixed", 0.213702),
      ("relu6", "mean", "learned_shared", 0.213702),
      ("relu6", "mean", "learned_per_input", 0.213702),
      ("relu6", "geometric_mean", "fixed", 0.215817),
      ("relu6", "geometric_mean", "learned_shared", 0.215806),
      ("relu6", "geometric_mean", "learned_per_input", 0.215816),
      ("sigmoid", "mean", "fixed", 0.205054),
      ("sigmoid", "mean", "learned_shared", 0.204950),
      ("sigmoid", "mean", "learned_per_input", 0.205030),
      ("sigmoid", "geometric_mean", "fixed", 0.204511),
      ("sigmoid", "geometric_mean", "learned_shared", 0.204406),
      ("sigmoid", "geometric_mean", "learned_per_input", 0.204488),
  )
  def test5DimSinOfSum(self, activation, reduction, input_scaling_type,
                       expected_loss):
    if self.disable_all:
      return
    config = {
        "input_dims": 5,
        "activation": activation,
        "reduction": reduction,
        "input_scaling_type": input_scaling_type,
        "num_training_records": 200,
        "num_training_epoch": 200,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SinOfSum,
    }
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self.loss_eps)

  @parameterized.parameters(
      ("relu6", "mean", "fixed", 0.000424),
      ("relu6", "mean", "learned_shared", 0.000424),
      ("relu6", "mean", "learned_per_input", 0.000424),
      ("relu6", "geometric_mean", "fixed", 0.000439),
      ("relu6", "geometric_mean", "learned_shared", 0.000439),
      ("relu6", "geometric_mean", "learned_per_input", 0.000439),
      ("sigmoid", "mean", "fixed", 0.000444),
      ("sigmoid", "mean", "learned_shared", 0.000444),
      ("sigmoid", "mean", "learned_per_input", 0.000444),
      ("sigmoid", "geometric_mean", "fixed", 0.000459),
      ("sigmoid", "geometric_mean", "learned_shared", 0.000459),
      ("sigmoid", "geometric_mean", "learned_per_input", 0.000459),
  )
  def test1DimInputOutOfBounds(self, activation, reduction, input_scaling_type,
                               expected_loss):
    if self.disable_all:
      return
    config = {
        "input_dims": 1,
        "activation": activation,
        "reduction": reduction,
        "input_scaling_type": input_scaling_type,
        "num_training_records": 100,
        "num_training_epoch": 20,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformlyExtendedRange,
        "y_function": self._Sin,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self.loss_eps)

  @parameterized.parameters(
      ("relu6", "mean", "fixed", 0.339018),
      ("relu6", "mean", "learned_shared", 0.338988),
      ("relu6", "mean", "learned_per_input", 0.339002),
      ("relu6", "geometric_mean", "fixed", 0.370072),
      ("relu6", "geometric_mean", "learned_shared", 0.370105),
      ("relu6", "geometric_mean", "learned_per_input", 0.370144),
      ("sigmoid", "mean", "fixed", 0.340095),
      ("sigmoid", "mean", "learned_shared", 0.340094),
      ("sigmoid", "mean", "learned_per_input", 0.340094),
      ("sigmoid", "geometric_mean", "fixed", 0.368851),
      ("sigmoid", "geometric_mean", "learned_shared", 0.368849),
      ("sigmoid", "geometric_mean", "learned_per_input", 0.368850),
  )
  def test2DimInputOutOfBounds(self, activation, reduction, input_scaling_type,
                               expected_loss):
    if self.disable_all:
      return
    config = {
        "input_dims": 2,
        "activation": activation,
        "reduction": reduction,
        "input_scaling_type": input_scaling_type,
        "num_training_records": 100,
        "num_training_epoch": 20,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._TwoDMeshGridExtendedRange,
        "y_function": self._SinOfSum,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self.loss_eps)

  @parameterized.parameters(
      (6, 6, "relu6", "mean", "fixed", 3, 0.070477),
      (8, 8, "relu6", "mean", "learned_shared", 4, 0.076625),
      (8, 8, "relu6", "mean", "learned_per_input", 4, 0.076696),
      (3, 3, "relu6", "geometric_mean", "fixed", 3, 0.031802),
      (4, 4, "relu6", "geometric_mean", "learned_shared", 2, 0.049083),
      (5, 5, "relu6", "geometric_mean", "learned_per_input", 2.5, 0.059841),
      (6, 6, "sigmoid", "mean", "fixed", 3, 0.075446),
      (8, 8, "sigmoid", "mean", "learned_shared", 4, 0.087095),
      (8, 8, "sigmoid", "mean", "learned_per_input", 4, 0.087091),
      (3, 3, "sigmoid", "geometric_mean", "fixed", 3, 0.033214),
      (4, 4, "sigmoid", "geometric_mean", "learned_shared", 2, 0.044370),
      (5, 5, "sigmoid", "geometric_mean", "learned_per_input", 2.5, 0.056680),
  )
  def testMultiUnitOutputSparsity(self, input_dims, units, activation,
                                  reduction, input_scaling_type,
                                  sparsity_factor, expected_loss):
    if self.disable_all:
      return
    # Set the random seed for the initializer for consistent results.
    kernel_initializer = keras.initializers.RandomUniform(0.0, 1.0, seed=42)
    config = {
        "input_dims": input_dims,
        "units": units,
        "activation": activation,
        "reduction": reduction,
        "input_scaling_type": input_scaling_type,
        "sparsity_factor": sparsity_factor,
        "kernel_initializer": kernel_initializer,
        "num_training_records": 100,
        "num_training_epoch": 20,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._Square,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self.loss_eps)

  @parameterized.parameters(
      ("relu6", "mean", 4, "fixed", 0.181436),
      ("relu6", "mean", 6, "learned_shared", 0.173429),
      ("relu6", "mean", 6, "learned_per_input", 0.174332),
      ("relu6", "geometric_mean", 8, "fixed", 0.173544),
      ("relu6", "geometric_mean", 15, "learned_shared", 0.172116),
      ("relu6", "geometric_mean", 15, "learned_per_input", 0.172146),
      ("sigmoid", "mean", 4, "fixed", 0.194161),
      ("sigmoid", "mean", 6, "learned_shared", 0.177846),
      ("sigmoid", "mean", 6, "learned_per_input", 0.179537),
      ("sigmoid", "geometric_mean", 8, "fixed", 0.176535),
      ("sigmoid", "geometric_mean", 15, "learned_shared", 0.172762),
      ("sigmoid", "geometric_mean", 15, "learned_per_input", 0.172728),
  )
  def testInputScalingInit(self, activation, reduction, input_scaling_init,
                           input_scaling_type, expected_loss):
    if self.disable_all:
      return
    config = {
        "input_dims": 2,
        "activation": activation,
        "reduction": reduction,
        "input_scaling_init": input_scaling_init,
        "input_scaling_type": input_scaling_type,
        "num_training_records": 900,
        "num_training_epoch": 100,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._TwoDMeshGrid,
        "y_function": self._SinPlusXNd,
    }
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self.loss_eps)

  @parameterized.parameters(
      (2, 10, 5, "relu6", "mean", "fixed", 30),
      (2, 10, 5, "relu6", "mean", "learned_shared", 35),
      (2, 10, 5, "relu6", "mean", "learned_per_input", 35),
      (2, 10, 5, "relu6", "geometric_mean", "fixed", 36),
      (2, 10, 5, "relu6", "geometric_mean", "learned_shared", 41),
      (2, 10, 5, "relu6", "geometric_mean", "learned_per_input", 41),
      (4, 20, 10, "relu6", "mean", "fixed", 30),
      (4, 20, 10, "relu6", "mean", "learned_shared", 35),
      (4, 20, 10, "relu6", "mean", "learned_per_input", 35),
      (4, 20, 10, "relu6", "geometric_mean", "fixed", 36),
      (4, 20, 10, "relu6", "geometric_mean", "learned_shared", 41),
      (4, 20, 10, "relu6", "geometric_mean", "learned_per_input", 41),
      (2, 10, 5, "sigmoid", "mean", "fixed", 28),
      (2, 10, 5, "sigmoid", "mean", "learned_shared", 33),
      (2, 10, 5, "sigmoid", "mean", "learned_per_input", 33),
      (2, 10, 5, "sigmoid", "geometric_mean", "fixed", 34),
      (2, 10, 5, "sigmoid", "geometric_mean", "learned_shared", 39),
      (2, 10, 5, "sigmoid", "geometric_mean", "learned_per_input", 39),
      (4, 20, 10, "sigmoid", "mean", "fixed", 28),
      (4, 20, 10, "sigmoid", "mean", "learned_shared", 33),
      (4, 20, 10, "sigmoid", "mean", "learned_per_input", 33),
      (4, 20, 10, "sigmoid", "geometric_mean", "fixed", 34),
      (4, 20, 10, "sigmoid", "geometric_mean", "learned_shared", 39),
      (4, 20, 10, "sigmoid", "geometric_mean", "learned_per_input", 39),
  )
  def testGraphSize(self, input_dims, num_keypoints, units, activation,
                    reduction, input_scaling_type, expected_graph_size):
    # If this test failed then you modified core lattice interpolation logic in
    # a way which increases number of ops in the graph. Or maybe Keras team
    # changed something under the hood. Please ensure that this increase is
    # unavoidable and try to minimize it.
    if self.disable_all:
      return
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    layer = cdf_layer.CDF(
        num_keypoints=num_keypoints,
        units=units,
        activation=activation,
        reduction=reduction,
        input_scaling_type=input_scaling_type)

    input_tensor = tf.ones(shape=(1, input_dims))
    layer(input_tensor)
    graph_size = len(tf.compat.v1.get_default_graph().as_graph_def().node)

    self.assertLessEqual(graph_size, expected_graph_size)


if __name__ == "__main__":
  tf.test.main()
