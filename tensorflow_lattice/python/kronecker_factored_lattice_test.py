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
"""Tests for KroneckerFactoredLattice Layer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tempfile
from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_lattice.python import kronecker_factored_lattice_layer as kfll
from tensorflow_lattice.python import kronecker_factored_lattice_lib as kfl_lib
from tensorflow_lattice.python import test_utils
# pylint: disable=g-import-not-at-top
# Use Keras 2.
version_fn = getattr(tf.keras, "version", None)
if version_fn and version_fn().startswith("3."):
  import tf_keras as keras
else:
  keras = tf.keras


class KroneckerFactoredLatticeTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(KroneckerFactoredLatticeTest, self).setUp()
    self.disable_all = False
    self.disable_ensembles = False
    self.loss_eps = 0.001
    self.small_eps = 1e-6
    self.seed = 42
    keras.utils.set_random_seed(42)

  def _ResetAllBackends(self):
    keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

  def _ScatterXUniformly(self, num_points, lattice_sizes, input_dims):
    """Deterministically generates num_point random points within lattice."""
    x = []
    for _ in range(num_points):
      point = [
          np.random.random() * (lattice_sizes - 1.0) for _ in range(input_dims)
      ]
      x.append(np.asarray(point))
    if input_dims == 1:
      x.sort()
    return x

  def _ScatterXUniformlyExtendedRange(self, num_points, lattice_sizes,
                                      input_dims):
    """Extends every dimension by 1.0 on both sides and generates points."""
    x = []
    for _ in range(num_points):
      point = [
          np.random.random() * (lattice_sizes + 1.0) - 1.0
          for _ in range(input_dims)
      ]
      x.append(np.asarray(point))
    if input_dims == 1:
      x.sort()
    return x

  def _SameValueForAllDims(self, num_points, lattice_sizes, input_dims):
    """Generates random point with same value for every dimension."""
    x = []
    for _ in range(num_points):
      rand = np.random.random() * (lattice_sizes - 1.0)
      point = [rand] * input_dims
      x.append(np.asarray(point))
    if input_dims == 1:
      x.sort()
    return x

  def _TwoDMeshGrid(self, num_points, lattice_sizes, input_dims):
    """Mesh grid for visualisation of 3-d surfaces via pyplot."""
    if input_dims != 2:
      raise ValueError("2-d mesh grid is possible only for 2-d lattice. Lattice"
                       " dimension given: %s" % input_dims)
    return test_utils.two_dim_mesh_grid(
        num_points=num_points,
        x_min=0.0,
        y_min=0.0,
        x_max=lattice_sizes - 1.0,
        y_max=lattice_sizes - 1.0)

  def _TwoDMeshGridExtendedRange(self, num_points, lattice_sizes, input_dims):
    """Mesh grid extended by 1.0 on every side."""
    if input_dims != 2:
      raise ValueError("2-d mesh grid is possible only for 2-d lattice. Lattice"
                       " dimension given: %s" % input_dims)
    return test_utils.two_dim_mesh_grid(
        num_points=num_points,
        x_min=-1.0,
        y_min=-1.0,
        x_max=lattice_sizes,
        y_max=lattice_sizes)

  def _Sin(self, x):
    return math.sin(x[0])

  def _SinPlusX(self, x):
    return math.sin(x[0]) + x[0] / 3.0

  def _SinPlusLargeX(self, x):
    return math.sin(x[0]) + x[0]

  def _SinPlusXNd(self, x):
    return np.sum([math.sin(y) + y / 5.0 for y in x])

  def _SinOfSum(self, x):
    return math.sin(sum(x))

  def _Max(self, x):
    return np.amax(x)

  def _ScaledSum(self, x):
    result = 0.0
    for y in x:
      result += y / len(x)
    return result

  def _GetNonMonotonicInitializer(self, weights):
    """Tiles given weights along 'units' dimension."""
    dims = len(weights)

    def Initializer(shape, dtype):
      _, lattice_sizes, num_inputs, num_terms = shape
      units = num_inputs // dims
      # Create expanded weights, tile, reshape, return.
      return tf.reshape(
          tf.tile(
              tf.constant(
                  weights,
                  shape=[1, lattice_sizes, 1, dims, num_terms],
                  dtype=dtype),
              multiples=[1, 1, units, 1, 1]), shape)

    return Initializer

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
        lattice_sizes=config["lattice_sizes"],
        input_dims=config["input_dims"])

    if isinstance(raw_training_inputs, tuple):
      # This means that raw inputs are 2-d mesh grid. Convert them into list of
      # 2-d points.
      training_inputs = list(np.dstack(raw_training_inputs).reshape((-1, 2)))
    else:
      training_inputs = raw_training_inputs

    training_labels = [config["y_function"](x) for x in training_inputs]
    return training_inputs, training_labels

  def _SetDefaults(self, config):
    config.setdefault("units", 1)
    config.setdefault("num_terms", 2)
    config.setdefault("monotonicities", None)
    config.setdefault("output_min", None)
    config.setdefault("output_max", None)
    config.setdefault("signal_name", "TEST")
    config.setdefault("target_monotonicity_diff", 0.0)
    config.setdefault("lattice_index", 0)
    config.setdefault("scale_initializer", "scale_initializer")

    return config

  def _TestEnsemble(self, config):
    """Verifies that 'units > 1' lattice produces same output as 'units==1'."""
    # Note that the initialization of the lattice must be the same across the
    # units dimension (otherwise the loss will be different).
    # We fix the random seed to make sure we get similar initialization.
    if self.disable_ensembles:
      return
    config = dict(config)
    config["num_training_epoch"] = 3
    config["kernel_initializer"] = "constant"
    losses = []
    for units, lattice_index in [(1, 0), (3, 0), (3, 2)]:
      config["units"] = units
      config["lattice_index"] = lattice_index
      keras.utils.set_random_seed(42)
      losses.append(self._TrainModel(config))
    self.assertAlmostEqual(min(losses), max(losses), delta=self.loss_eps)

  def _TrainModel(self, config):
    logging.info("Testing config:")
    logging.info(config)
    config = self._SetDefaults(config)
    self._ResetAllBackends()

    training_inputs, training_labels = (
        self._GetTrainingInputsAndLabels(config))

    units = config["units"]
    input_dims = config["input_dims"]
    lattice_sizes = config["lattice_sizes"]
    if units > 1:
      # In order to test multi 'units' lattice replecate inputs 'units' times
      # and later use just one out of 'units' outputs in order to ensure that
      # multi 'units' lattice trains exactly similar to single 'units' one.
      training_inputs = [
          np.tile(np.expand_dims(x, axis=0), reps=[units, 1])
          for x in training_inputs
      ]
      input_shape = (units, input_dims)
    else:
      input_shape = (input_dims,)

    keras_layer = kfll.KroneckerFactoredLattice(
        lattice_sizes=lattice_sizes,
        units=units,
        num_terms=config["num_terms"],
        monotonicities=config["monotonicities"],
        output_min=config["output_min"],
        output_max=config["output_max"],
        kernel_initializer=config["kernel_initializer"],
        scale_initializer=config["scale_initializer"],
        input_shape=input_shape,
        dtype=tf.float32)
    model = keras.models.Sequential()
    model.add(keras_layer)

    # When we use multi-unit lattices, we only extract a single lattice for
    # testing.
    if units > 1:
      lattice_index = config["lattice_index"]
      model.add(
          keras.layers.Lambda(lambda x: x[:, lattice_index:lattice_index + 1]))

    optimizer = config["optimizer"](learning_rate=config["learning_rate"])
    model.compile(loss=keras.losses.mean_squared_error, optimizer=optimizer)

    training_data = (training_inputs, training_labels)
    loss = test_utils.run_training_loop(
        config=config, training_data=training_data, keras_model=model
    )

    if tf.executing_eagerly():
      tf.print("final weights: ", keras_layer.kernel)
      tf.print("final scale: ", keras_layer.scale)
      tf.print("final bias: ", keras_layer.bias)
    assetion_ops = keras_layer.assert_constraints(
        eps=-config["target_monotonicity_diff"])
    if not tf.executing_eagerly() and assetion_ops:
      tf.compat.v1.keras.backend.get_session().run(assetion_ops)

    return loss

  def testMonotonicityOneD(self):
    if self.disable_all:
      return
    monotonicities = [1]
    kernel_initializer = kfll.KFLRandomMonotonicInitializer(
        monotonicities=monotonicities, seed=self.seed)
    config = {
        "lattice_sizes": 20,
        "input_dims": 1,
        "num_training_records": 128,
        "num_training_epoch": 50,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SinPlusX,
        "monotonicities": monotonicities,
        "kernel_initializer": kernel_initializer,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.114794, delta=self.loss_eps)
    self._TestEnsemble(config)

    monotonicities = ["increasing"]
    kernel_initializer = kfll.KFLRandomMonotonicInitializer(
        monotonicities=monotonicities, seed=self.seed)
    config = {
        "lattice_sizes": 20,
        "input_dims": 1,
        "num_training_records": 100,
        "num_training_epoch": 50,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": lambda x: -self._SinPlusX(x),
        "monotonicities": monotonicities,
        "kernel_initializer": kernel_initializer,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 3.011028, delta=self.loss_eps)
    self._TestEnsemble(config)

    monotonicities = [1]
    kernel_initializer = kfll.KFLRandomMonotonicInitializer(
        monotonicities=monotonicities, seed=self.seed)
    config = {
        "lattice_sizes": 5,
        "input_dims": 1,
        "num_terms": 1,
        "num_training_records": 100,
        "num_training_epoch": 200,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SinPlusLargeX,
        "monotonicities": monotonicities,
        "kernel_initializer": kernel_initializer,
        # Target function is strictly increasing.
        "target_monotonicity_diff": 0.01,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.000832, delta=self.loss_eps)

  def testMonotonicityTwoD(self):
    if self.disable_all:
      return
    monotonicities = [1, 1]
    kernel_initializer = kfll.KFLRandomMonotonicInitializer(
        monotonicities=monotonicities, seed=self.seed)
    config = {
        "lattice_sizes": 21,
        "input_dims": 2,
        "num_training_records": 900,
        "num_training_epoch": 100,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._TwoDMeshGrid,
        "y_function": self._SinPlusXNd,
        "monotonicities": monotonicities,
        "kernel_initializer": kernel_initializer,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.407444, delta=self.loss_eps)
    self._TestEnsemble(config)

    monotonicities = ["none", "increasing"]
    kernel_initializer = kfll.KFLRandomMonotonicInitializer(
        monotonicities=monotonicities, seed=self.seed)
    config = {
        "lattice_sizes": 21,
        "input_dims": 2,
        "num_training_records": 900,
        "num_training_epoch": 100,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._TwoDMeshGrid,
        "y_function": self._SinPlusXNd,
        "monotonicities": monotonicities,
        "kernel_initializer": kernel_initializer,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.354508, delta=self.loss_eps)
    self._TestEnsemble(config)

    monotonicities = [1, 0]
    kernel_initializer = kfll.KFLRandomMonotonicInitializer(
        monotonicities=monotonicities, seed=self.seed)
    config = {
        "lattice_sizes": 21,
        "input_dims": 2,
        "num_training_records": 900,
        "num_training_epoch": 100,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 1.5,
        "x_generator": self._TwoDMeshGrid,
        "y_function": self._SinPlusXNd,
        "monotonicities": monotonicities,
        "kernel_initializer": kernel_initializer,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.365634, delta=self.loss_eps)
    self._TestEnsemble(config)

    monotonicities = [1, 1]
    kernel_initializer = kfll.KFLRandomMonotonicInitializer(
        monotonicities=monotonicities, seed=self.seed)
    config = {
        "lattice_sizes": 2,
        "input_dims": 2,
        "num_training_records": 100,
        "num_training_epoch": 20,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._TwoDMeshGrid,
        "y_function": lambda x: -self._ScaledSum(x),
        "monotonicities": monotonicities,
        "kernel_initializer": kernel_initializer,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.054951, delta=self.loss_eps)
    self._TestEnsemble(config)

  def testMonotonicity5d(self):
    if self.disable_all:
      return
    config = {
        "lattice_sizes": 2,
        "input_dims": 5,
        "num_training_records": 100,
        "num_training_epoch": 200,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._ScaledSum,
        "monotonicities": [1, 1, 1, 1, 1],
        "kernel_initializer": keras.initializers.Constant(value=0.5),
        # Function is strictly increasing everywhere, so request monotonicity
        # diff to be strictly positive.
        "target_monotonicity_diff": 0.08,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.000524, delta=self.loss_eps)

    monotonicities = [1, 1, 1, 1, 1]
    kernel_initializer = kfll.KFLRandomMonotonicInitializer(
        monotonicities=monotonicities, seed=self.seed)
    config = {
        "lattice_sizes": 2,
        "input_dims": 5,
        "num_training_records": 100,
        "num_training_epoch": 40,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": lambda x: -self._ScaledSum(x),
        "monotonicities": monotonicities,
        "kernel_initializer": kernel_initializer,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.016635, delta=self.loss_eps)
    self._TestEnsemble(config)

    monotonicities = [1, "increasing", 1, 1]
    kernel_initializer = kfll.KFLRandomMonotonicInitializer(
        monotonicities=monotonicities, seed=self.seed)
    config = {
        "lattice_sizes": 3,
        "input_dims": 4,
        "num_training_records": 100,
        "num_training_epoch": 100,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SinOfSum,
        "monotonicities": monotonicities,
        "kernel_initializer": kernel_initializer,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.398279, delta=self.loss_eps)
    self._TestEnsemble(config)

  @parameterized.parameters(
      ([0, 1, 1],),
      ([1, 0, 1],),
      ([1, 1, 0],),
  )
  def testMonotonicityEquivalence(self, monotonicities):
    if self.disable_all:
      return
    config = {
        "lattice_sizes": 3,
        "input_dims": 3,
        "monotonicities": monotonicities,
        "num_training_records": 100,
        "num_training_epoch": 50,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 10.0,
        "x_generator": self._SameValueForAllDims,
        "y_function": self._SinOfSum,
        "kernel_initializer": "zeros",
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.522080, delta=self.loss_eps)
    self._TestEnsemble(config)

  def testMonotonicity10dAlmostMonotone(self):
    if self.disable_all:
      return
    num_weights = 1024
    weights = [1.0 * i / num_weights for i in range(num_weights)]
    for _ in range(10):
      i = int(np.random.random() * num_weights)
      weights[i] = 0.0

    config = {
        "lattice_sizes": 2,
        "input_dims": 10,
        "num_terms": 128,
        "num_training_records": 1000,
        "num_training_epoch": 100,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": test_utils.get_hypercube_interpolation_fn(weights),
        "monotonicities": [1] * 10,
        "kernel_initializer": "zeros",
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.027578, delta=self.loss_eps)

    config["monotonicities"] = [0, 1, 0, 1, 1, 0, 1, 1, 1, 0]
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.027578, delta=self.loss_eps)
    self._TestEnsemble(config)

  def testMonotonicity10dSinOfSum(self):
    if self.disable_all:
      return
    monotonicities = [1] * 10
    kernel_initializer = kfll.KFLRandomMonotonicInitializer(
        monotonicities=monotonicities, seed=self.seed)
    config = {
        "lattice_sizes": 2,
        "input_dims": 10,
        "num_training_records": 1000,
        "num_training_epoch": 100,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SinOfSum,
        "monotonicities": [1] * 10,
        "kernel_initializer": kernel_initializer,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.158914, delta=self.loss_eps)

    monotonicities = [0, 1, 0, 1, 1, 0, 1, 1, 1, 0]
    kernel_initializer = kfll.KFLRandomMonotonicInitializer(
        monotonicities=monotonicities, seed=self.seed)
    config["monotonicities"] = monotonicities
    config["kernel_initializer"] = kernel_initializer
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.196240, delta=self.loss_eps)

  @parameterized.parameters(
      # Custom TFL initializer:
      ("kfl_random_monotonic_initializer", 2.024714),
      # Standard Keras initializer:
      (keras.initializers.Constant(value=1.5), 2.140740),
      # Standard Keras initializer specified as string constant:
      ("zeros", 2.140740),
  )
  def testInitializerType(self, initializer, expected_loss):
    if self.disable_all:
      return
    if initializer == "kfl_random_monotonic_initializer":
      initializer = kfll.KFLRandomMonotonicInitializer(
          monotonicities=None, seed=self.seed)
    config = {
        "lattice_sizes": 3,
        "input_dims": 2,
        "num_training_records": 100,
        "num_training_epoch": 0,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._TwoDMeshGrid,
        "y_function": self._Max,
        "kernel_initializer": initializer,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self.loss_eps)
    self._TestEnsemble(config)

  def testAssertMonotonicity(self):
    if self.disable_all:
      return
    # Specify non monotonic initializer and do 0 training iterations so no
    # projections are being executed.
    config = {
        "lattice_sizes": 2,
        "input_dims": 2,
        "num_training_records": 100,
        "num_training_epoch": 0,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 0.15,
        "x_generator": self._TwoDMeshGrid,
        "y_function": self._ScaledSum,
        "monotonicities": [0, 0],
        "kernel_initializer": self._GetNonMonotonicInitializer(
            weights=[
                [[4.0, 3.0], [4.0, 3.0]],
                [[2.0, 1.0], [2.0, 1.0]]
            ])
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 4.458333, delta=self.loss_eps)

    for monotonicity in [[0, 1], [1, 0], [1, 1]]:
      for units in [1, 3]:
        config["monotonicities"] = monotonicity
        config["units"] = units
        with self.assertRaises(tf.errors.InvalidArgumentError):
          self._TrainModel(config)

  @parameterized.parameters(
      (
          -1,
          1,
          kfll.KFLRandomMonotonicInitializer(
              monotonicities=None, init_min=-10, init_max=10
          ),
          "scale_initializer",
      ),
      (
          None,
          1,
          kfll.KFLRandomMonotonicInitializer(
              monotonicities=None, init_min=-10, init_max=10
          ),
          "scale_initializer",
      ),
      (
          -1,
          None,
          kfll.KFLRandomMonotonicInitializer(
              monotonicities=None, init_min=-10, init_max=10
          ),
          "scale_initializer",
      ),
      (
          -1,
          1,
          "kfl_random_monotonic_initializer",
          keras.initializers.Constant(value=-100),
      ),
      (
          None,
          1,
          "kfl_random_monotonic_initializer",
          keras.initializers.Constant(value=100),
      ),
      (
          -1,
          None,
          "kfl_random_monotonic_initializer",
          keras.initializers.Constant(value=-100),
      ),
  )
  def testAssertBounds(self, output_min, output_max, kernel_initializer,
                       scale_initializer):
    if self.disable_all:
      return
    # Specify random initializer that ensures initial output can be out of
    # bounds and do 0 training iterations so no projections are executed.
    config = {
        "lattice_sizes": 2,
        "input_dims": 2,
        "num_training_records": 100,
        "num_training_epoch": 0,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 0.15,
        "x_generator": self._TwoDMeshGrid,
        "y_function": self._ScaledSum,
        "monotonicities": [0, 0],
        "output_min": output_min,
        "output_max": output_max,
        "kernel_initializer": kernel_initializer,
        "scale_initializer": scale_initializer,
    }
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self._TrainModel(config)

  @parameterized.parameters(
      (2, 1, -3, -1, 4.82327),
      (2, 2, 0, 1, 0.163572),
      (1, 2, -5, 5, 0.011432),
      (1, 10, -1, 1, 0.6307245),
      (1, 3, None, None, 0.012286),
      (1, 2, None, 5, 0.011590),
      (3, 3, 0, None, 0.011679),
      (4, 2, None, -2, 9.9507179),
  )
  def testOutputBounds(self, units, input_dims, output_min, output_max,
                       expected_loss):
    if self.disable_all:
      return
    monotonicities = [1] * input_dims
    kernel_initializer = kfll.KFLRandomMonotonicInitializer(
        monotonicities=monotonicities, seed=self.seed)
    config = {
        "lattice_sizes": 4,
        "units": units,
        "input_dims": input_dims,
        "num_training_records": 900,
        "num_training_epoch": 100,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SinPlusX,
        "monotonicities": monotonicities,
        "output_min": output_min,
        "output_max": output_max,
        "kernel_initializer": kernel_initializer,
        # This is the epsilon error allowed when asserting constraints,
        # including bounds. We include this to ensure that the bound constraint
        # assertions do not fail due to numerical errors.
        "target_monotonicity_diff": -1e-6,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self.loss_eps)
    self._TestEnsemble(config)

  @parameterized.parameters(
      (2, 1, 3, 2, -1, 1),
      (2, 2, 2, 1, None, 1),
      (2, 1, 3, 4, -1, None),
      (3, 3, 4, 2, -50, 2),
      (4, 4, 2, 4, -1.5, 2.3),
      (2, 2, 2, 2, None, None),
  )
  # Note: dims must be at least 1
  def testConstraints(self, lattice_sizes, units, dims, num_terms, output_min,
                      output_max):
    if self.disable_all:
      return
    # Run our test for 100 iterations to minimize the chance we pass by chance.
    for _ in range(100):
      # Create 100 random inputs that are frozen in all but the increasing
      # dimension, which increases uniformly from 0 to lattice_sizes-1.
      batch_size = 100
      random_vals = [
          np.random.uniform(0, lattice_sizes - 1) for _ in range(dims - 1)
      ]
      increasing_dim = np.random.randint(0, dims)
      step_size = (lattice_sizes - 1) / batch_size
      values = [
          np.roll([0.0 + (i * step_size)] + random_vals, increasing_dim)
          for i in range(batch_size)
      ]
      if units > 1:
        values = [[value] * units for value in values]
        shape = [batch_size, units, dims]
      else:
        shape = [batch_size, dims]
      inputs = tf.constant(values, dtype=tf.float32, shape=shape)

      # Create our weights, constraint them, and evaluate our function on our
      # constructed inputs.
      init_min = -1.5 if output_min is None else output_min
      init_max = 1.5 if output_max is None else output_max

      # Offset the initiailization bounds to increase likelihood of breaking
      # constraints.
      offset = 100
      kernel = tf.random.uniform([1, lattice_sizes, units * dims, num_terms],
                                 minval=init_min - offset,
                                 maxval=init_max + offset)
      scale = tf.random.uniform([units, num_terms],
                                minval=init_min,
                                maxval=init_max)
      bias = kfl_lib.bias_initializer(
          units, output_min, output_max, dtype=tf.float32)

      scale_constraint = kfll.ScaleConstraints(output_min, output_max)
      constrained_scale = scale_constraint(scale)

      monotonicities = [np.random.randint(0, 2) for _ in range(dims)]
      monotonicities[increasing_dim] = 1
      kernel_constraint = kfll.KroneckerFactoredLatticeConstraints(
          units, constrained_scale, monotonicities, output_min, output_max)
      constrained_kernel = kernel_constraint(kernel)

      outputs = kfl_lib.evaluate_with_hypercube_interpolation(
          inputs=inputs,
          scale=constrained_scale,
          bias=bias,
          kernel=constrained_kernel,
          units=units,
          num_terms=num_terms,
          lattice_sizes=lattice_sizes,
          clip_inputs=True)

      # Check that outputs are inside our bounds
      min_check = float("-inf") if output_min is None else output_min
      self.assertEqual(tf.reduce_sum(tf.cast(outputs < min_check, tf.int32)), 0)
      max_check = float("+inf") if output_max is None else output_max
      self.assertEqual(tf.reduce_sum(tf.cast(outputs > max_check, tf.int32)), 0)
      # Check that we satisfy monotonicity constraints. Note that by
      # construction the outputs should already be in sorted order.
      sorted_outputs = tf.sort(outputs, axis=0)
      # We use close equality instead of strict equality because of numerical
      # errors that result in nearly identical arrays failing a strict check
      # after sorting.
      self.assertAllClose(outputs, sorted_outputs, rtol=1e-6, atol=1e-6)

  def testInputOutOfBounds(self):
    if self.disable_all:
      return
    config = {
        "lattice_sizes": 6,
        "input_dims": 1,
        "num_training_records": 100,
        "num_training_epoch": 20,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformlyExtendedRange,
        "y_function": self._Sin,
        "kernel_initializer": keras.initializers.Zeros(),
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.028617, delta=self.loss_eps)
    self._TestEnsemble(config)

    kernel_initializer = kfll.KFLRandomMonotonicInitializer(
        monotonicities=None, seed=self.seed)
    config = {
        "lattice_sizes": 2,
        "input_dims": 2,
        "num_training_records": 100,
        "num_training_epoch": 20,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._TwoDMeshGridExtendedRange,
        "y_function": self._SinOfSum,
        "kernel_initializer": kernel_initializer,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.323999, delta=self.loss_eps)
    self._TestEnsemble(config)

  def testHighDimensionsStressTest(self):
    if self.disable_all:
      return
    monotonicities = [0] * 16
    monotonicities[3], monotonicities[4], monotonicities[10] = (1, 1, 1)
    kernel_initializer = kfll.KFLRandomMonotonicInitializer(
        monotonicities=monotonicities, seed=self.seed)
    config = {
        "lattice_sizes": 2,
        "input_dims": 16,
        "num_terms": 128,
        "units": 2,
        "monotonicities": monotonicities,
        "num_training_records": 100,
        "num_training_epoch": 3,
        "optimizer": keras.optimizers.legacy.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SinOfSum,
        "kernel_initializer": kernel_initializer,
        "target_monotonicity_diff": -1e-5,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.30680, delta=self.loss_eps)

  @parameterized.parameters(
      (2, 5, 2, 57),
      (2, 6, 4, 57),
      (2, 9, 2, 57),
      (3, 5, 4, 63),
      (3, 9, 2, 63),
  )
  def testGraphSize(self, lattice_sizes, input_dims, num_terms,
                    expected_graph_size):
    # If this test failed then you modified core lattice interpolation logic in
    # a way which increases number of ops in the graph. Or maybe Keras team
    # changed something under the hood. Please ensure that this increase is
    # unavoidable and try to minimize it.
    if self.disable_all:
      return
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    layer = kfll.KroneckerFactoredLattice(
        lattice_sizes=lattice_sizes, num_terms=num_terms)
    input_tensor = tf.ones(shape=(1, input_dims))
    layer(input_tensor)
    graph_size = len(tf.compat.v1.get_default_graph().as_graph_def().node)

    self.assertLessEqual(graph_size, expected_graph_size)

  @parameterized.parameters(
      ("random_uniform", keras.initializers.RandomUniform),
      ("kfl_random_monotonic_initializer", kfll.KFLRandomMonotonicInitializer),
  )
  def testCreateKernelInitializer(self, kernel_initializer_id, expected_type):
    self.assertEqual(
        expected_type,
        type(
            kfll.create_kernel_initializer(
                kernel_initializer_id,
                monotonicities=None,
                output_min=None,
                output_max=None)))

  # We test that the scale variable attribute of our KroneckerFactoredLattice
  # is the same object as the scale contained in the constraint on the kernel,
  # both before and after save/load. We test this because we must make sure that
  # any updates to the scale variable (before/after save/load) are consistent
  # across all uses of the object.
  def testSavingLoadingScale(self):
    # Create simple x --> x^2 dataset.
    train_data = [[[float(x)], float(x)**2] for x in range(100)]
    train_x, train_y = zip(*train_data)
    train_x, train_y = np.array(train_x), np.array(train_y)
    # Construct simple single lattice model. Must have monotonicities specified
    # or constraint will be None.
    keras_layer = kfll.KroneckerFactoredLattice(
        lattice_sizes=2, monotonicities=[1])
    model = keras.models.Sequential()
    model.add(keras_layer)
    # Compile and fit the model.
    model.compile(
        loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.1))
    model.fit(train_x, train_y)
    # Extract scale from layer and constraint before save.
    layer_scale = keras_layer.scale
    constraint_scale = keras_layer.kernel.constraint.scale
    self.assertIs(layer_scale, constraint_scale)
    # Save and load the model.
    with tempfile.NamedTemporaryFile(suffix=".h5") as f:
      keras.models.save_model(model, f.name)
      loaded_model = keras.models.load_model(
          f.name,
          custom_objects={
              "KroneckerFactoredLattice":
                  kfll.KroneckerFactoredLattice,
              "KroneckerFactoredLatticeConstraint":
                  kfll.KroneckerFactoredLatticeConstraints,
              "KFLRandomMonotonicInitializer":
                  kfll.KFLRandomMonotonicInitializer,
              "ScaleInitializer":
                  kfll.ScaleInitializer,
              "ScaleConstraints":
                  kfll.ScaleConstraints,
              "BiasInitializer":
                  kfll.BiasInitializer,
          })
    # Extract loaded layer.
    loaded_keras_layer = loaded_model.layers[0]
    # Extract scale from layer and constraint after load.
    loaded_layer_scale = loaded_keras_layer.scale
    loaded_constraint_scale = loaded_keras_layer.kernel.constraint.scale
    self.assertIs(loaded_layer_scale, loaded_constraint_scale)

  @parameterized.parameters(
      (1, 3, 1),
      (1, 3, 2),
      (3, 7, 3),
  )
  def testOutputShapeForDifferentInputTypes(self, batch_size, dims, units):
    expected_output_shape = (batch_size, units)
    # Create KFL Layer instance.
    kfl_layer = kfll.KroneckerFactoredLattice(lattice_sizes=2, units=units)
    # Input (batch_size, dims) or (batch_size, units, dims)
    if units == 1:
      example = [float(i) for i in range(dims)]
      examples = [example for _ in range(batch_size)]
    else:
      example = [[float(i) for i in range(dims)] for _ in range(units)]
      examples = [example for _ in range(batch_size)]
    inputs = tf.constant(examples)
    outputs = kfl_layer(inputs)
    self.assertEqual(outputs.shape, expected_output_shape)
    # Input length-dims list of (batch_size, 1) or (batch_size, units, 1)
    example = tf.constant(
        [[float(i) if units == 1 else [float(i)]
          for i in range(units)]
         for _ in range(batch_size)])
    list_inputs = [example for _ in range(dims)]
    list_outputs = kfl_layer(list_inputs)
    self.assertEqual(list_outputs.shape, expected_output_shape)


if __name__ == "__main__":
  tf.test.main()
