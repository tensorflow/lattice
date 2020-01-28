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

"""Tests for Lattice Layer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow_lattice.python import lattice_layer as ll
from tensorflow_lattice.python import test_utils


class LatticeTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(LatticeTest, self).setUp()
    self.disable_all = False
    self.disable_ensembles = False
    self.loss_eps = 0.0001
    self.small_eps = 1e-6

  def _ResetAllBackends(self):
    keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

  def _ScatterXUniformly(self, num_points, lattice_sizes):
    """Deterministically generates num_point random points within lattice."""
    np.random.seed(41)
    x = []
    for _ in range(num_points):
      point = [
          np.random.random() * (num_vertices - 1.0)
          for num_vertices in lattice_sizes
      ]
      x.append(np.asarray(point))
    if len(lattice_sizes) == 1:
      x.sort()
    return x

  def _ScatterXUniformlyExtendedRange(self, num_points, lattice_sizes):
    """Extends every dimension by 1.0 on both sides and generates points."""
    np.random.seed(41)
    x = []
    for _ in range(num_points):
      point = [
          np.random.random() * (num_vertices + 1.0) - 1.0
          for num_vertices in lattice_sizes
      ]
      x.append(np.asarray(point))
    if len(lattice_sizes) == 1:
      x.sort()
    return x

  def _SameValueForAllDims(self, num_points, lattice_sizes):
    """Generates random point with same value for every dimension."""
    if lattice_sizes.count(lattice_sizes[0]) != len(lattice_sizes):
      raise ValueError("All dimensions must be of same size. "
                       "They are: {}".format(lattice_sizes))
    np.random.seed(41)
    x = []
    for _ in range(num_points):
      rand = np.random.random() * (lattice_sizes[0] - 1.0)
      point = [rand] * len(lattice_sizes)
      x.append(np.asarray(point))
    if len(lattice_sizes) == 1:
      x.sort()
    return x

  def _TwoDMeshGrid(self, num_points, lattice_sizes):
    """Mesh grid for visualisation of 3-d surfaces via pyplot."""
    if len(lattice_sizes) != 2:
      raise ValueError("2-d mesh grid is possible only for 2-d lattice. Lattice"
                       " sizes given: %s" % lattice_sizes)
    return test_utils.two_dim_mesh_grid(
        num_points=num_points,
        x_min=0.0,
        y_min=0.0,
        x_max=lattice_sizes[0] - 1.0,
        y_max=lattice_sizes[1] - 1.0)

  def _TwoDMeshGridExtendedRange(self, num_points, lattice_sizes):
    """Mesh grid extended by 1.0 on every side."""
    if len(lattice_sizes) != 2:
      raise ValueError("2-d mesh grid is possible only for 2-d lattice. Lattice"
                       " sizes given: %s" % lattice_sizes)
    return test_utils.two_dim_mesh_grid(
        num_points=num_points,
        x_min=-1.0,
        y_min=-1.0,
        x_max=lattice_sizes[0],
        y_max=lattice_sizes[1])

  def _Sin(self, x):
    return math.sin(x[0])

  def _SinPlusX(self, x):
    return math.sin(x[0]) + x[0] / 3.0

  def _SinPlusLargeX(self, x):
    return math.sin(x[0]) + x[0]

  def _SinPlusXNd(self, x):
    res = 0.0
    for y in x:
      res = res + math.sin(y) + y / 5.0
    return res

  def _SinOfSum(self, x):
    return math.sin(sum(x))

  def _Square(self, x):
    return x[0]**2

  def _Max(self, x):
    return np.amax(x)

  def _WeightedSum(self, x):
    result = 0.0
    for i in range(len(x)):
      result += (i + 1.0) * x[i]
    return result

  def _MixedSignWeightedSum(self, x):
    result = 0.0
    for i in range(len(x)):
      sign = (i % 2) * -2 + 1
      result += sign * (i + 1.0) * x[i]
    return result

  def _PseudoLinear(self, x):
    result = 0.0
    for i in range(len(x)):
      result += 2 * x[i]
      for j in range(len(x)):
        if i != j:
          result += x[i] * x[j]
    return result

  def _ScaledSum(self, x):
    result = 0.0
    for y in x:
      result += y / len(x)
    return result

  def _GetMultiOutputInitializer(self, weights):
    """Tiles given weights along 'units' dimension."""
    def Initializer(shape, dtype):
      return tf.tile(tf.constant(weights, shape=[len(weights), 1], dtype=dtype),
                     multiples=[1, shape[1]])
    return Initializer

  def _GetTrainingInputsAndLabels(self, config):
    """Generates training inputs and labels.

    Args:
      config: Dictionary with config for this unit test.

    Returns:
      Tuple `(training_inputs, training_labels, raw_training_inputs)` where
        `training_inputs` and `training_labels` are data for training and
        `raw_training_inputs` are representation of training_inputs for
        visualisation.
    """
    raw_training_inputs = config["x_generator"](
        num_points=config["num_training_records"],
        lattice_sizes=config["lattice_sizes"])

    if isinstance(raw_training_inputs, tuple):
      # This means that raw inputs are 2-d mesh grid. Convert them into list of
      # 2-d points.
      training_inputs = list(np.dstack(raw_training_inputs).reshape((-1, 2)))
    else:
      training_inputs = raw_training_inputs

    training_labels = [config["y_function"](x) for x in training_inputs]
    return training_inputs, training_labels, raw_training_inputs

  def _SetDefaults(self, config):
    config.setdefault("monotonicities", None)
    config.setdefault("unimodalities", None)
    config.setdefault("edgeworth_trusts", None)
    config.setdefault("trapezoid_trusts", None)
    config.setdefault("monotonic_dominances", None)
    config.setdefault("joint_monotonicities", None)
    config.setdefault("output_min", None)
    config.setdefault("output_max", None)
    config.setdefault("signal_name", "TEST")
    config.setdefault("kernel_initializer", "linear_initializer")
    config.setdefault("num_projection_iterations", 10)
    config.setdefault("monotonic_at_every_step", True)
    config.setdefault("target_monotonicity_diff", 0.0)
    config.setdefault("kernel_regularizer", None)
    config.setdefault("units", 1)
    config.setdefault("lattice_index", 0)

    return config

  def _TestEnsemble(self, config):
    """Verifies that 'units > 1' lattice produces same output as 'units==1'."""
    if self.disable_ensembles:
      return
    config = dict(config)
    config["num_training_epoch"] = 3
    losses = []
    for units, lattice_index in [(1, 0), (3, 0), (3, 2)]:
      config["units"] = units
      config["lattice_index"] = lattice_index
      losses.append(self._TrainModel(config))
    self.assertAlmostEqual(min(losses), max(losses), delta=self.loss_eps)

  def _TrainModel(self, config, plot_path=None):
    logging.info("Testing config:")
    logging.info(config)
    config = self._SetDefaults(config)
    self._ResetAllBackends()

    training_inputs, training_labels, raw_training_inputs = (
        self._GetTrainingInputsAndLabels(config))

    units = config["units"]
    lattice_sizes = config["lattice_sizes"]
    if units > 1:
      # In order to test multi 'units' lattice replecate inputs 'units' times
      # and later use just one out of 'units' outputs in order to ensure that
      # multi 'units' lattice trains exactly similar to single 'units' one.
      training_inputs = [
          np.tile(np.expand_dims(x, axis=0), reps=[units, 1])
          for x in training_inputs
      ]
      input_shape = (units, len(lattice_sizes))
    else:
      input_shape = (len(lattice_sizes),)

    keras_layer = ll.Lattice(
        lattice_sizes=lattice_sizes,
        units=units,
        monotonicities=config["monotonicities"],
        unimodalities=config["unimodalities"],
        edgeworth_trusts=config["edgeworth_trusts"],
        trapezoid_trusts=config["trapezoid_trusts"],
        monotonic_dominances=config["monotonic_dominances"],
        joint_monotonicities=config["joint_monotonicities"],
        output_min=config["output_min"],
        output_max=config["output_max"],
        num_projection_iterations=config["num_projection_iterations"],
        monotonic_at_every_step=config["monotonic_at_every_step"],
        kernel_initializer=config["kernel_initializer"],
        kernel_regularizer=config["kernel_regularizer"],
        input_shape=input_shape,
        dtype=tf.float32)
    model = keras.models.Sequential()
    model.add(keras_layer)

    if units > 1:
      lattice_index = config["lattice_index"]
      model.add(keras.layers.Lambda(
          lambda x: x[:, lattice_index:lattice_index + 1]))

    optimizer = config["optimizer"](learning_rate=config["learning_rate"])
    model.compile(loss=keras.losses.mean_squared_error, optimizer=optimizer)

    training_data = (training_inputs, training_labels, raw_training_inputs)
    loss = test_utils.run_training_loop(
        config=config,
        training_data=training_data,
        keras_model=model,
        plot_path=plot_path)

    if tf.executing_eagerly():
      tf.print("final weights: ", keras_layer.kernel)
    assetion_ops = keras_layer.assert_constraints(
        eps=-config["target_monotonicity_diff"])
    if not tf.executing_eagerly() and assetion_ops:
      tf.compat.v1.keras.backend.get_session().run(assetion_ops)

    return loss

  def testMonotonicityOneD(self):
    if self.disable_all:
      return
    config = {
        "lattice_sizes": [20],
        "num_training_records": 128,
        "num_training_epoch": 50,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SinPlusX,
        "monotonicities": [1],
        "output_min": 0.0,
        "output_max": 7.0,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.110467, delta=self.loss_eps)
    self._TestEnsemble(config)

    config = {
        "lattice_sizes": [20],
        "num_training_records": 100,
        "num_training_epoch": 50,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": lambda x: -self._SinPlusX(x),
        "monotonicities": ["increasing"],
        "output_min": -7.0,
        "output_max": 0.0,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 2.889168, delta=self.loss_eps)
    self._TestEnsemble(config)

    config = {
        "lattice_sizes": [5],
        "num_training_records": 100,
        "num_training_epoch": 50,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SinPlusLargeX,
        "monotonicities": [1],
        "output_min": 0.0,
        "output_max": 6.0,
        # Target function is strictly increasing.
        "target_monotonicity_diff": 0.02,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.000781, delta=self.loss_eps)

  def testMonotonicityTwoD(self):
    if self.disable_all:
      return
    config = {
        "lattice_sizes": [21, 6],
        "num_training_records": 900,
        "num_training_epoch": 100,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._TwoDMeshGrid,
        "y_function": self._SinPlusXNd,
        "monotonicities": [1, 1],
        "output_min": 0.0,
        "output_max": 7.0,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.443284, delta=self.loss_eps)
    self._TestEnsemble(config)

    config = {
        "lattice_sizes": [6, 21],
        "num_training_records": 900,
        "num_training_epoch": 100,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._TwoDMeshGrid,
        "y_function": self._SinPlusXNd,
        "monotonicities": [1, 1],
        "output_min": 0.0,
        "output_max": 7.0,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.443284, delta=self.loss_eps)
    self._TestEnsemble(config)

    config = {
        "lattice_sizes": [6, 21],
        "num_training_records": 900,
        "num_training_epoch": 100,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._TwoDMeshGrid,
        "y_function": self._SinPlusXNd,
        "monotonicities": ["none", "increasing"],
        "output_min": 0.0,
        "output_max": 7.0,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.202527, delta=self.loss_eps)
    self._TestEnsemble(config)

    config = {
        "lattice_sizes": [6, 21],
        "num_training_records": 900,
        "num_training_epoch": 100,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.5,
        "x_generator": self._TwoDMeshGrid,
        "y_function": self._SinPlusXNd,
        "monotonicities": [1, 0],
        "output_min": 0.0,
        "output_max": 7.0,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.244739, delta=self.loss_eps)
    self._TestEnsemble(config)

    config = {
        "lattice_sizes": [2, 2],
        "num_training_records": 100,
        "num_training_epoch": 20,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._TwoDMeshGrid,
        "y_function": lambda x: -self._ScaledSum(x),
        "monotonicities": [1, 1],
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.051462, delta=self.loss_eps)
    self._TestEnsemble(config)

  def testMonotonicity5d(self):
    if self.disable_all:
      return
    config = {
        "lattice_sizes": [2, 2, 2, 2, 2],
        "num_training_records": 100,
        "num_training_epoch": 200,
        "optimizer": tf.keras.optimizers.Adagrad,
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
    self.assertAlmostEqual(loss, 0.000002, delta=self.loss_eps)

    config = {
        "lattice_sizes": [2, 2, 2, 2, 2],
        "num_training_records": 100,
        "num_training_epoch": 40,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": lambda x: -self._ScaledSum(x),
        "monotonicities": [1, 1, 1, 1, 1],
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.014971, delta=self.loss_eps)
    self._TestEnsemble(config)

    config = {
        "lattice_sizes": [3, 3, 3, 3],
        "num_training_records": 100,
        "num_training_epoch": 100,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SinOfSum,
        "monotonicities": [1, "increasing", 1, 1],
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.358079, delta=self.loss_eps)
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
        "lattice_sizes": [3, 3, 3],
        "monotonicities": monotonicities,
        "num_training_records": 100,
        "num_training_epoch": 50,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 10.0,
        "x_generator": self._SameValueForAllDims,
        "y_function": self._SinOfSum,
        "kernel_initializer": "zeros",
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.000286, delta=self.loss_eps)
    self._TestEnsemble(config)

  def testMonotonicity10dAlmostMonotone(self):
    if self.disable_all:
      return
    np.random.seed(4411)
    num_weights = 1024
    weights = [1.0 * i / num_weights for i in range(num_weights)]
    for _ in range(10):
      i = int(np.random.random() * num_weights)
      weights[i] = 0.0

    config = {
        "lattice_sizes": [2] * 10,
        "num_training_records": 1000,
        "num_training_epoch": 100,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 100.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": test_utils.get_hypercube_interpolation_fn(weights),
        "monotonicities": [1] * 10,
        "kernel_initializer": "zeros",
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.000027, delta=self.loss_eps)

    config["monotonicities"] = [0, 1, 0, 1, 1, 0, 1, 1, 1, 0]
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.000019, delta=self.loss_eps)
    self._TestEnsemble(config)

  def testMonotonicity10dSinOfSum(self):
    if self.disable_all:
      return
    config = {
        "lattice_sizes": [2] * 10,
        "num_training_records": 1000,
        "num_training_epoch": 100,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 100.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SinOfSum,
        "monotonicities": [1] * 10,
        "output_min": -1.0,
        "output_max": 1.0,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.089950, delta=self.loss_eps)

    config["monotonicities"] = [0, 1, 0, 1, 1, 0, 1, 1, 1, 0]
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.078830, delta=self.loss_eps)

    config["monotonicities"] = [0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.052190, delta=self.loss_eps)
    self._TestEnsemble(config)

  @parameterized.parameters(
      ([(0, 1, 1)], [], 0.025785),
      (None, [(0, 1, 1)], 0.042566),
      ([(0, 1, "positive")], [(0, 1, "positive")], 0.042566),
  )
  def testSimpleTrustTwoD(self, edgeworth_trusts, trapezoid_trusts,
                          expected_loss):
    if self.disable_all:
      return
    config = {
        "lattice_sizes": [2, 2],
        "num_training_records": 100,
        "num_training_epoch": 20,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._TwoDMeshGrid,
        "y_function": self._Max,
        "monotonicities": [1, 0],
        "edgeworth_trusts": edgeworth_trusts,
        "trapezoid_trusts": trapezoid_trusts,
        "output_min": 0.0,
        "output_max": 1.0,
        # Leave margin of error (floating point) for trust projection.
        "target_monotonicity_diff": -1e-6,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self.loss_eps)
    self._TestEnsemble(config)

  @parameterized.parameters(
      ([(1, 0, -1)], None, 3.23711),
      (None, [(1, 0, -1)], 6.663453),
      ([(1, 0, "negative")], [(1, 0, "negative")], 9.846122),
  )
  def testDenseTrustTwoD(self, edgeworth_trusts, trapezoid_trusts,
                         expected_loss):
    if self.disable_all:
      return
    config = {
        "lattice_sizes": [4, 3],
        "num_training_records": 150,
        "num_training_epoch": 20,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 10.0,
        "x_generator": self._TwoDMeshGrid,
        "y_function": self._PseudoLinear,
        "monotonicities": [0, 1],
        "edgeworth_trusts": edgeworth_trusts,
        "trapezoid_trusts": trapezoid_trusts,
        "output_min": 0.0,
        "output_max": 22.0,
        # Leave margin of error (floating point) for trust projection.
        "target_monotonicity_diff": -1e-5,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self.loss_eps)
    if not edgeworth_trusts or not trapezoid_trusts:
      self._TestEnsemble(config)

  @parameterized.parameters(
      ([(0, 1, 1)], None, 0.010525),
      (None, [(0, 1, 1)], 0.013343),
      ([(0, 1, 1)], [(0, 1, 1)], 0.013343),
  )
  def testSimpleTrust4D(self, edgeworth_trusts, trapezoid_trusts,
                        expected_loss):
    if self.disable_all:
      return
    config = {
        "lattice_sizes": [2, 2, 2, 2],
        "num_training_records": 100,
        "num_training_epoch": 20,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._Max,
        "monotonicities": [1, 0, 1, 1],
        "edgeworth_trusts": edgeworth_trusts,
        "trapezoid_trusts": trapezoid_trusts,
        "output_min": 0.0,
        "output_max": 1.0,
        # Leave margin of error (floating point) for trust projection.
        "target_monotonicity_diff": -1e-6,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self.loss_eps)
    self._TestEnsemble(config)

  @parameterized.parameters(
      ([(0, 1, 1), (3, 1, -1), (3, 2, 1)], None, 0.334325),
      (None, [(0, 1, 1), (3, 1, -1), (3, 2, 1)], 0.387444),
      ([(0, 1, 1), (3, 1, -1)], [(3, 1, -1), (3, 2, 1)], 0.381514),
  )
  def testMultiDenseTrust4D(self, edgeworth_trusts, trapezoid_trusts,
                            expected_loss):
    if self.disable_all:
      return
    config = {
        "lattice_sizes": [3, 3, 3, 3],
        "num_training_records": 1000,
        "num_training_epoch": 20,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SinOfSum,
        "monotonicities": [1, 0, 0, 1],
        "edgeworth_trusts": edgeworth_trusts,
        "trapezoid_trusts": trapezoid_trusts,
        "output_min": -0.5,
        "output_max": 0.9,
        # Leave margin of error (floating point) for trust projection.
        "target_monotonicity_diff": -1e-6,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self.loss_eps)
    if not edgeworth_trusts or not trapezoid_trusts:
      self._TestEnsemble(config)

  @parameterized.parameters(
      ([(0, 1, 1)],),
      ([(1, 2, 1)],),
      ([(2, 0, 1)],),
  )
  def testEdgeworthTrustEquivalence(self, edgeworth_trusts):
    if self.disable_all:
      return
    config = {
        "lattice_sizes": [3, 3, 3],
        "monotonicities": [1, 1, 1],
        "edgeworth_trusts": edgeworth_trusts,
        "num_training_records": 100,
        "num_training_epoch": 50,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 10.0,
        "x_generator": self._SameValueForAllDims,
        "y_function": self._PseudoLinear,
        "kernel_initializer": "zeros",
        # Leave margin of error (floating point) for trust projection.
        "target_monotonicity_diff": -1e-6,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.006912, delta=self.loss_eps)
    self._TestEnsemble(config)

  @parameterized.parameters(
      (None, 0.00000),
      ([(1, 0)], 0.00000),
      ([(0, 1)], 0.05092),
  )
  def testSimpleMonotonicDominance2D(self, monotonic_dominances, expected_loss):
    if self.disable_all:
      return
    config = {
        "lattice_sizes": [2, 2],
        "num_training_records": 100,
        "num_training_epoch": 20,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._TwoDMeshGrid,
        "y_function": self._WeightedSum,
        "monotonicities": [1, 1],
        "monotonic_dominances": monotonic_dominances,
        "output_min": 0.0,
        "output_max": 3.0,
        # Leave margin of error (floating point) for dominance projection.
        "target_monotonicity_diff": -1e-6,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self.loss_eps)
    self._TestEnsemble(config)

  @parameterized.parameters(
      (None, 0.00113),
      ([(1, 0)], 0.00113),
      ([(0, 1)], 0.81520),
  )
  def testDenseMonotonicDominance2D(self, monotonic_dominances, expected_loss):
    if self.disable_all:
      return
    config = {
        "lattice_sizes": [5, 5],
        "num_training_records": 100,
        "num_training_epoch": 20,
        "num_projection_iterations": 20,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._TwoDMeshGrid,
        "y_function": self._WeightedSum,
        "monotonicities": [1, 1],
        "monotonic_dominances": monotonic_dominances,
        "output_min": 0.0,
        "output_max": 12.0,
        # Leave margin of error (floating point) for dominance projection.
        "target_monotonicity_diff": -1e-2,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self.loss_eps)
    self._TestEnsemble(config)

  @parameterized.parameters(
      ([(1, 0), (2, 1)], 2.52985),
      ([(0, 1), (1, 2)], 6.16700),
  )
  def testDenseMonotonicDominance5D(self, monotonic_dominances, expected_loss):
    if self.disable_all:
      return
    config = {
        "lattice_sizes": [5, 5, 5, 5, 5],
        "num_training_records": 100,
        "num_training_epoch": 300,
        "num_projection_iterations": 20,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._WeightedSum,
        "monotonicities": [1, 1, 1, 1, 1],
        "monotonic_dominances": monotonic_dominances,
        "output_min": 0.0,
        "output_max": 60.0,
        # Leave margin of error (floating point) for dominance projection.
        "target_monotonicity_diff": -1e-1,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self.loss_eps)
    self._TestEnsemble(config)

  @parameterized.parameters(
      (None, 0.00000),
      ([(0, 1)], 0.05092),
      ([(1, 0)], 0.05092),
  )
  def testSimpleJointMonotonicity2D(self, joint_monotonicities, expected_loss):
    if self.disable_all:
      return
    config = {
        "lattice_sizes": [2, 2],
        "num_training_records": 100,
        "num_training_epoch": 20,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._TwoDMeshGrid,
        "y_function": self._MixedSignWeightedSum,
        "monotonicities": [0, 0],
        "joint_monotonicities": joint_monotonicities,
        "output_min": -2.0,
        "output_max": 1.0,
        # Leave margin of error (floating point) for dominance projection.
        "target_monotonicity_diff": -1e-6,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self.loss_eps)
    self._TestEnsemble(config)

  @parameterized.parameters(
      (None, 0.16301),
      ([(0, 1)], 0.86386),
      ([(1, 0)], 0.86413),
  )
  def testDenseJointMonotonicity2D(self, joint_monotonicities, expected_loss):
    if self.disable_all:
      return
    config = {
        "lattice_sizes": [5, 5],
        "num_training_records": 100,
        "num_training_epoch": 40,
        "num_projection_iterations": 40,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._TwoDMeshGrid,
        "y_function": self._MixedSignWeightedSum,
        "monotonicities": [0, 0],
        "joint_monotonicities": joint_monotonicities,
        "output_min": -8.0,
        "output_max": 4.0,
        # Leave margin of error (floating point) for dominance projection.
        "target_monotonicity_diff": -1e-2,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self.loss_eps)
    self._TestEnsemble(config)

  @parameterized.parameters(
      ([(0, 1)], 36.75898),
  )
  def testDenseJointMonotonicity5D(self, joint_monotonicities, expected_loss):
    if self.disable_all:
      return
    config = {
        "lattice_sizes": [5, 5, 5, 5, 5],
        "num_training_records": 100,
        "num_training_epoch": 100,
        "num_projection_iterations": 40,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._MixedSignWeightedSum,
        "monotonicities": [0, 0, 0, 0, 0],
        "joint_monotonicities": joint_monotonicities,
        "output_min": -24.0,
        "output_max": 36.0,
        # Leave margin of error (floating point) for dominance projection.
        "target_monotonicity_diff": -1e-1,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self.loss_eps)
    self._TestEnsemble(config)

  @parameterized.parameters(
      # Custom TFL initializer:
      ("linear_initializer", 0.126068),
      # Standard Keras initializer:
      (keras.initializers.Constant(value=1.5), 0.430379),
      # Standard Keras initializer specified as string constant:
      ("zeros", 1.488072),
  )
  def testInitializerType(self, initializer, expected_loss):
    if self.disable_all:
      return
    config = {
        "lattice_sizes": [2, 3],
        "num_training_records": 98,
        "num_training_epoch": 0,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._TwoDMeshGrid,
        "y_function": self._Max,
        "output_min": 0.0,
        "output_max": 2.0,
        "kernel_initializer": initializer,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self.loss_eps)
    self._TestEnsemble(config)

  def _MergeDicts(self, x, y):
    z = dict(x)
    z.update(y)
    return z

  def testLinearMonotonicInitializer(self):
    if self.disable_all:
      return
    # Test initializer by training linear function using 0 iteration and verify
    # that loss is 0.
    config = {
        "num_training_records": 96,
        "num_training_epoch": 0,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._TwoDMeshGrid,
    }  # pyformat: disable

    init_config = {
        "lattice_sizes": [3, 4],
        "monotonicities": [0, 0],
        "output_min": -1.0,
        "output_max": 2.0,
    }
    config["kernel_initializer"] = "LinearInitializer"
    config["y_function"] = test_utils.get_linear_lattice_interpolation_fn(
        **init_config)
    total_config = self._MergeDicts(config, init_config)
    loss = self._TrainModel(total_config)
    self.assertAlmostEqual(loss, 0.0, delta=self.small_eps)
    self._TestEnsemble(total_config)

    # Change generator since we need more than 2 dimensions from now on.
    config["x_generator"] = self._ScatterXUniformly

    init_config = {
        "lattice_sizes": [2, 3, 4, 5],
        "monotonicities": [1, 1, 0, 1],
        "output_min": 12.0,
        "output_max": 22.0,
    }
    config["kernel_initializer"] = ll.LinearInitializer(**init_config)
    config["y_function"] = test_utils.get_linear_lattice_interpolation_fn(
        **init_config)
    total_config = self._MergeDicts(config, init_config)
    loss = self._TrainModel(total_config)
    self.assertAlmostEqual(loss, 0.0, delta=self.small_eps)
    self._TestEnsemble(total_config)

    init_config = {
        "lattice_sizes": [2, 3, 4, 5],
        "monotonicities": [0, 1, 0, 1],
        "output_min": -10,
        "output_max": -5,
    }
    config["kernel_initializer"] = ll.LinearInitializer(**init_config)
    config["y_function"] = test_utils.get_linear_lattice_interpolation_fn(
        **init_config)
    total_config = self._MergeDicts(config, init_config)
    loss = self._TrainModel(total_config)
    self.assertAlmostEqual(loss, 0.0, delta=self.small_eps)
    self._TestEnsemble(total_config)

    # Try to fit some other function and see loss >0 to ensure that this test
    # does not always returns 0.
    config["y_function"] = self._SinOfSum
    total_config = self._MergeDicts(config, init_config)
    loss = self._TrainModel(total_config)
    self.assertGreater(loss, 0.1)
    self._TestEnsemble(total_config)

    init_config = {
        "lattice_sizes": [2, 3, 4, 5],
        "monotonicities": [0, 0, 0, 0],
        "output_min": 1.0,
        "output_max": 3.0,
    }
    config["kernel_initializer"] = "linear_initializer"
    config["y_function"] = test_utils.get_linear_lattice_interpolation_fn(
        **init_config)
    total_config = self._MergeDicts(config, init_config)
    loss = self._TrainModel(total_config)
    self.assertAlmostEqual(loss, 0.0, delta=self.small_eps)
    self._TestEnsemble(total_config)

  def testUnimodalInitializer(self):
    if self.disable_all:
      return
    config = {
        "lattice_sizes": [3, 4],
        "unimodalities": [1, 1],
        "kernel_initializer": "linear_initializer",
        "num_training_records": 96,
        "num_training_epoch": 0,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._TwoDMeshGrid,
        "y_function": self._Max,
        "output_min": 0.0,
        "output_max": 2.0,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 1.292362, delta=self.loss_eps)
    self._TestEnsemble(config)

    config["unimodalities"] = ["valley", "none"]
    config["monotonicities"] = ["none", "increasing"]
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.794330, delta=self.loss_eps)
    self._TestEnsemble(config)

  def testAssertMonotonicity(self):
    if self.disable_all:
      return
    # Specify non monotonic initializer and do 0 training iterations so no
    # projections are being executed.
    config = {
        "lattice_sizes": [2, 2],
        "num_training_records": 100,
        "num_training_epoch": 0,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 0.15,
        "x_generator": self._TwoDMeshGrid,
        "y_function": self._ScaledSum,
        "monotonicities": [0, 0],
        "kernel_initializer": self._GetMultiOutputInitializer(
            weights=[4.0, 3.0, 2.0, 1.0])
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 4.865740, delta=self.loss_eps)

    for monotonicity in [[0, 1], [1, 0], [1, 1]]:
      for units in [1, 3]:
        config["monotonicities"] = monotonicity
        config["units"] = units
        with self.assertRaises(tf.errors.InvalidArgumentError):
          self._TrainModel(config)

  def testBounds(self):
    if self.disable_all:
      return
    config = {
        "lattice_sizes": [20],
        "num_training_records": 100,
        "num_training_epoch": 40,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._Sin,
        "output_min": -0.6,
        "output_max": 0.4,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.109398, delta=self.loss_eps)
    self._TestEnsemble(config)

    config = {
        "lattice_sizes": [11, 4],
        "num_training_records": 270,
        "num_training_epoch": 40,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._TwoDMeshGrid,
        "y_function": self._SinPlusXNd,
        "monotonicities": [1, 1],
        "output_min": 1.0,
        "output_max": 2.5,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.380813, delta=self.loss_eps)
    self._TestEnsemble(config)

    config = {
        "lattice_sizes": [2] * 5,
        "num_training_records": 100,
        "num_training_epoch": 40,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SinOfSum,
        "monotonicities": [1, 1, 0, 1, 0],
        "output_min": 0.3,
        "output_max": 0.7,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.145910, delta=self.loss_eps)
    self._TestEnsemble(config)

  def testInputOutOfBounds(self):
    if self.disable_all:
      return
    config = {
        "lattice_sizes": [6],
        "num_training_records": 100,
        "num_training_epoch": 20,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformlyExtendedRange,
        "y_function": self._Sin,
        "kernel_initializer": keras.initializers.Zeros(),
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.018727, delta=self.loss_eps)
    self._TestEnsemble(config)

    config = {
        "lattice_sizes": [2, 2],
        "num_training_records": 100,
        "num_training_epoch": 20,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._TwoDMeshGridExtendedRange,
        "y_function": self._SinOfSum,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.130813, delta=self.loss_eps)
    self._TestEnsemble(config)

  @parameterized.parameters(
      # Laplacian with l1 and l2:
      (("laplacian", 0.005, 0.01), 0.03, 0.021399),
      # Different regularization amount for every dimension:
      (("laplacian", [0.005, 0.01], [0.01, 0.02]), 0.045, 0.027941),
      # Torsion with l1 and l2:
      (("torsion", 0.1, 0.01), 0.11, 0.06738),
      # Different regularization amount for every dimension:
      (("torsion", [2.0, 0.05], [0.1, 0.1]), 0.11, 0.06738),
      # List of regularizers:
      ([("torsion", 0.1, 0.0), ("Torsion", 0.0, 0.01)], 0.11, 0.06738),
      # Standard Keras regularizer:
      (keras.regularizers.l1_l2(l1=0.01, l2=0.1), 0.33, 0.214418),
  )
  def testRegularizers2d(self, regularizer, pure_reg_loss, training_loss):
    if self.disable_all:
      return
    weights = [0.0, 1.0, 1.0, 1.0]
    config = {
        "lattice_sizes": [2, 2],
        "num_training_records": 100,
        "num_training_epoch": 0,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._TwoDMeshGrid,
        "y_function": test_utils.get_hypercube_interpolation_fn(
            coefficients=weights),
        "kernel_initializer": self._GetMultiOutputInitializer(weights=weights),
        "kernel_regularizer": regularizer,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    # This loss is pure regularization loss because initializer matches target
    # function and there was 0 training epochs.
    self.assertAlmostEqual(loss, pure_reg_loss, delta=self.loss_eps)

    multioutput_config = dict(config)
    units = 3
    multioutput_config["units"] = units
    loss = self._TrainModel(multioutput_config)
    self.assertAlmostEqual(loss, pure_reg_loss * units, delta=self.loss_eps)

    config["num_training_epoch"] = 20
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, training_loss, delta=self.loss_eps)

  @parameterized.parameters(
      (("torsion", 0.001, 0.0001), 0.147405),
      (("laplacian", 0.001, 0.0001), 0.193870),
  )
  def testRegularizersLargeLattice(self, regularizer, expected_loss):
    if self.disable_all:
      return
    config = {
        "lattice_sizes": [3, 4, 3, 4],
        "num_training_records": 100,
        "num_training_epoch": 100,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SinOfSum,
        "kernel_regularizer": regularizer,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self.loss_eps)

  def testHighDimensionsStressTest(self):
    if self.disable_all:
      return
    lattice_sizes = [3, 3] + [2] * 14
    monotonicities = [0] * 16
    monotonicities[3], monotonicities[4], monotonicities[10] = (1, 1, 1)
    unimodalities = [0] * 16
    unimodalities[1] = 1
    config = {
        "lattice_sizes": lattice_sizes,
        "units": 2,
        "monotonicities": monotonicities,
        "unimodalities": unimodalities,
        "edgeworth_trusts": [(3, 2, 1)],
        "output_min": 0.0,
        "output_max": 1.0,
        "num_training_records": 100,
        "num_training_epoch": 3,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1000.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._SinOfSum,
        "kernel_regularizer": [("torsion", 1e-6, 1e-6),
                               ("laplacian", 1e-5, 1e-5)],
        "target_monotonicity_diff": -1e-5,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    # TODO: this test behaves differently in graph and eager mode.
    # Figure out why.
    self.assertAlmostEqual(loss, 3.689727, delta=0.1)

  @parameterized.parameters(
      ([0], [0], 0.026734),
      ([1], ["none"], 0.195275),
      ([1], None, 0.195275),
      ([0], ["valley"], 0.045627),
      (None, [1], 0.045627),
  )
  def testUnimodalityOneD(self, monotonicities, unimodalities, expected_loss):
    if self.disable_all:
      return

    def WShaped1dFunction(x):
      d = min(abs(x[0] - 3.0), abs(x[0] - 7.0))
      return d * d / 4.0

    config = {
        "lattice_sizes": [11],
        "num_training_records": 128,
        "num_training_epoch": 200,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 0.15,
        "x_generator": self._ScatterXUniformly,
        "y_function": WShaped1dFunction,
        "monotonicities": monotonicities,
        "unimodalities": unimodalities,
        "kernel_initializer": "linear_initializer",
        "output_min": 0.0,
        "output_max": 4.0,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self.loss_eps)
    self._TestEnsemble(config)

  @parameterized.parameters(
      ([0, 0], [0, 0], 0.003822),
      ([1, 1], [0, 0], 0.313155),
      ([0, 0], [1, 1], 0.003073),
      ([1, 0], [0, 1], 0.162484),
      ([0, 0], [1, 0], 0.004883),
  )
  def testUnimodalityTwoD(self, monotonicities, unimodalities, expected_loss):
    if self.disable_all:
      return

    def WShaped2dFunction(x):
      distance = lambda x1, y1, x2, y2: ((x2 - x1)**2 + (y2 - y1)**2)**0.5
      d = distance(x[0], x[1], 5.0, 5.0)
      return (d - 2.0)**2 / 8.0

    config = {
        "lattice_sizes": [11, 11],
        "num_training_records": 900,
        "num_training_epoch": 50,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.5,
        "x_generator": self._TwoDMeshGrid,
        "y_function": WShaped2dFunction,
        "monotonicities": monotonicities,
        "unimodalities": unimodalities,
        "kernel_initializer": "linear_initializer",
        "output_min": 0.0,
        "output_max": 4.0,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self.loss_eps)
    self._TestEnsemble(config)

  def testUnconstrained(self):
    if self.disable_all:
      return
    config = {
        "lattice_sizes": [20],
        "num_training_records": 100,
        "num_training_epoch": 200,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 0.15,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._Sin,
        "kernel_initializer": keras.initializers.Zeros,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.000917, delta=self.loss_eps)
    self._TestEnsemble(config)

    config = {
        "lattice_sizes": [2],
        "num_training_records": 100,
        "num_training_epoch": 50,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 0.15,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._Square,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.004277, delta=self.loss_eps)
    self._TestEnsemble(config)

    config = {
        "lattice_sizes": [2, 2],
        "num_training_records": 100,
        "num_training_epoch": 200,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 0.15,
        "x_generator": self._ScatterXUniformly,
        "y_function": test_utils.get_hypercube_interpolation_fn(
            coefficients=[0.0, 1.0, 1.0, 1.0]),
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.000003, delta=self.loss_eps)
    self._TestEnsemble(config)

    config = {
        "lattice_sizes": [2] * 3,
        "num_training_records": 100,
        "num_training_epoch": 200,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 0.15,
        "x_generator": self._ScatterXUniformly,
        "y_function": test_utils.get_hypercube_interpolation_fn(
            coefficients=[i / 2.0**3 for i in range(2**3)])
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.000001, delta=self.loss_eps)
    self._TestEnsemble(config)

    config = {
        "lattice_sizes": [2] * 5,
        "num_training_records": 100,
        "num_training_epoch": 100,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 1.5,
        "x_generator": self._ScatterXUniformly,
        "y_function": test_utils.get_hypercube_interpolation_fn(
            coefficients=[i / 2.0**5 for i in range(2**5)])
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.000008, delta=self.loss_eps)
    self._TestEnsemble(config)

    config = {
        "lattice_sizes": [2, 2],
        "num_training_records": 100,
        "num_training_epoch": 100,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 0.15,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._Max,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.003599, delta=self.loss_eps)
    self._TestEnsemble(config)

    config = {
        "lattice_sizes": [2] * 6,
        "num_training_records": 100,
        "num_training_epoch": 300,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 30.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._PseudoLinear,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.000118, delta=self.loss_eps)
    self._TestEnsemble(config)

    config = {
        "lattice_sizes": [2, 3, 4],
        "num_training_records": 100,
        "num_training_epoch": 200,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 10.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._PseudoLinear,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.00002, delta=self.loss_eps)
    self._TestEnsemble(config)

    config = {
        "lattice_sizes": [4, 5],
        "num_training_records": 100,
        "num_training_epoch": 100,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 10.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._WeightedSum,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.0, delta=self.loss_eps)
    self._TestEnsemble(config)

    config = {
        "lattice_sizes": [2, 3, 4, 5],
        "num_training_records": 100,
        "num_training_epoch": 200,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 30.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._Max,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.000891, delta=self.loss_eps)
    self._TestEnsemble(config)

    config = {
        "lattice_sizes": [2, 3, 4, 5],
        "num_training_records": 100,
        "num_training_epoch": 200,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 30.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._WeightedSum,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.004216, delta=self.loss_eps)
    self._TestEnsemble(config)

  @parameterized.parameters(
      ([2, 3, 4], 6.429155),
      ([2, 3, 3], 13.390955),
      ([2, 2, 3], 22.205267),
      ([2, 2, 3, 3], 5.049051),
      ([2, 2, 3, 2, 2], 5.3823),
      ([2, 2, 3, 3, 2, 2], 67.775276),
      ([2, 2, 2, 3, 3, 3], 156.755035),
      ([3, 2, 2, 3, 3, 2], 104.419373),
  )
  def testEqaulySizedDimsOptimization(self, lattice_sizes, expected_loss):
    if self.disable_all:
      return
    config = {
        "lattice_sizes": lattice_sizes,
        "num_training_records": 100,
        "num_training_epoch": 1,
        "optimizer": tf.keras.optimizers.Adagrad,
        "learning_rate": 10.0,
        "x_generator": self._ScatterXUniformly,
        "y_function": self._WeightedSum,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=self.loss_eps)

  @parameterized.parameters(
      ([2, 2, 2, 2, 2, 2], 92),
      ([2, 2, 3, 2, 3, 2], 117),
      ([2, 2, 2, 2, 3, 3], 102),
      ([2, 2, 2, 2, 2, 2, 2, 2, 2], 125),
      ([2, 2, 2, 2, 2, 2, 3, 3, 3], 135),
  )
  def testGraphSize(self, lattice_sizes, expected_graph_size):
    # If this test failed then you modified core lattice interpolation logic in
    # a way which increases number of ops in the graph. Or maybe Keras team
    # changed something under the hood. Please ensure that this increase is
    # unavoidable and try to minimize it.
    if self.disable_all:
      return
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    layer = ll.Lattice(lattice_sizes=lattice_sizes)
    input_tensor = tf.ones(shape=(1, len(lattice_sizes)))
    layer(input_tensor)
    graph_size = len(tf.compat.v1.get_default_graph().as_graph_def().node)

    self.assertLessEqual(graph_size, expected_graph_size)


if __name__ == "__main__":
  tf.test.main()
