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

"""Tests for Tensorflow Lattice linear layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_lattice.python import linear_layer as linl
from tensorflow_lattice.python import test_utils
from tensorflow_lattice.python import utils
# pylint: disable=g-import-not-at-top
# Use Keras 2.
version_fn = getattr(tf.keras, "version", None)
if version_fn and version_fn().startswith("3."):
  import tf_keras as keras
else:
  keras = tf.keras

_DISABLE_ALL = False
_LOSS_EPS = 0.0001
_SMALL_EPS = 1e-6


class LinearTest(parameterized.TestCase, tf.test.TestCase):
  """Tests for TFL linear layer."""

  def setUp(self):
    super(LinearTest, self).setUp()
    keras.utils.set_random_seed(42)

  def _ResetAllBackends(self):
    keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

  def _ScaterXUniformly(self, num_points, num_dims, input_min, input_max):
    """Generates num_points num_dims-dimensional points within given range."""
    np.random.seed(41)
    x = []
    for _ in range(num_points):
      point = [
          np.random.random() * (input_max - input_min) + input_min
          for _ in range(num_dims)
      ]
      x.append(np.asarray(point))
    if num_dims == 1:
      x.sort()
    return x

  def _TwoDMeshGrid(self, num_points, num_dims, input_min, input_max):
    """Mesh grid for visualisation of 3-d surfaces via pyplot."""
    if num_dims != 2:
      raise ValueError("2-d mesh grid can be created only for 2-d data. Given: "
                       "%d." % num_dims)
    return test_utils.two_dim_mesh_grid(
        num_points=num_points,
        x_min=input_min,
        y_min=input_min,
        x_max=input_max,
        y_max=input_max)

  def _GenLinearFunction(self, weights, bias=0.0, noise=None):
    """Returns python function which computes linear function."""

    def Linear(x):
      if len(x) != len(weights):
        raise ValueError("X and weights have different number of elements. "
                         "X: " + str(x) + "; weights: " + str(weights))
      result = bias
      if noise:
        result += noise(x)
      for (i, y) in enumerate(x):
        result += weights[i] * y
      return result

    return Linear

  def _SinPlusXPlusD(self, x):
    return math.sin(x[0]) + x[0] / 3.0 + 3.0

  def _SetDefaults(self, config):
    config.setdefault("monotonicities", None)
    config.setdefault("monotonic_dominances", None)
    config.setdefault("range_dominances", None)
    config.setdefault("clip_min", None)
    config.setdefault("clip_max", None)
    config.setdefault("use_bias", False)
    config.setdefault("normalization_order", None)
    config.setdefault("kernel_init_constant", 0.0)
    config.setdefault("bias_init_constant", 0.0)
    config.setdefault("kernel_regularizer", None)
    config.setdefault("bias_regularizer", None)
    config.setdefault("allowed_constraints_violation", 1e-6)
    config.setdefault("units", 1)
    config.setdefault("unit_index", 0)
    return config

  def _GetTrainingInputsAndLabels(self, config):
    """Generates training inputs and labels.

    Args:
      config: Dict with config for this unit test.

    Returns:
      Tuple `(training_inputs, training_labels, raw_training_inputs)` where
        `training_inputs` and `training_labels` are data for training.
    """
    raw_training_inputs = config["x_generator"](
        num_points=config["num_training_records"],
        num_dims=config["num_input_dims"],
        input_min=config["input_min"],
        input_max=config["input_max"])

    if isinstance(raw_training_inputs, tuple):
      # This means that raw inputs are 2-d mesh grid. Convert them into list of
      # 2-d points.
      training_inputs = list(np.dstack(raw_training_inputs).reshape((-1, 2)))
    else:
      training_inputs = raw_training_inputs

    training_labels = [config["y_function"](x) for x in training_inputs]
    return training_inputs, training_labels

  def _TrainModel(self, config):
    """Trains model and returns loss.

    Args:
      config: Layer config internal for this test which specifies params of
        linear layer to train.

    Returns:
      Training loss.
    """
    logging.info("Testing config:")
    logging.info(config)
    config = self._SetDefaults(config)

    self._ResetAllBackends()

    training_inputs, training_labels = (
        self._GetTrainingInputsAndLabels(config))
    units = config["units"]
    num_input_dims = config["num_input_dims"]
    if units > 1:
      # In order to test multi 'units' linear, replicate inputs 'units' times
      # and later use just one out of 'units' outputs in order to ensure that
      # multi 'units' linear trains exactly similar to single 'units' one.
      training_inputs = [
          np.tile(np.expand_dims(x, axis=0), reps=[units, 1])
          for x in training_inputs
      ]
      input_shape = (units, num_input_dims)
    else:
      input_shape = (num_input_dims,)

    linear_layer = linl.Linear(
        input_shape=input_shape,
        num_input_dims=config["num_input_dims"],
        units=units,
        monotonicities=config["monotonicities"],
        monotonic_dominances=config["monotonic_dominances"],
        range_dominances=config["range_dominances"],
        input_min=config["clip_min"],
        input_max=config["clip_max"],
        use_bias=config["use_bias"],
        normalization_order=config["normalization_order"],
        kernel_initializer=keras.initializers.Constant(
            config["kernel_init_constant"]),
        bias_initializer=keras.initializers.Constant(
            config["bias_init_constant"]),
        kernel_regularizer=config["kernel_regularizer"],
        bias_regularizer=config["bias_regularizer"],
        dtype=tf.float32)
    model = keras.models.Sequential()
    model.add(linear_layer)
    # When we use multi-unit linear, we only extract a single unit for testing.
    if units > 1:
      unit_index = config["unit_index"]
      model.add(
          keras.layers.Lambda(lambda x: x[:, unit_index:unit_index + 1]))
    optimizer = config["optimizer"](learning_rate=config["learning_rate"])
    model.compile(loss=keras.losses.mean_squared_error, optimizer=optimizer)

    training_data = (training_inputs, training_labels)

    loss = test_utils.run_training_loop(
        config=config, training_data=training_data, keras_model=model
    )

    assetion_ops = linear_layer.assert_constraints(
        eps=config["allowed_constraints_violation"])
    if not tf.executing_eagerly() and assetion_ops:
      tf.compat.v1.keras.backend.get_session().run(assetion_ops)
    return loss

  def _NegateAndTrain(self, config):
    """Changes monotonicity directions to opposite and trains model."""
    negated_config = dict(config)
    negated_config["y_function"] = lambda x: -config["y_function"](x)
    negated_config["bias_init_constant"] = -config["bias_init_constant"]
    negated_config["kernel_init_constant"] = -config["kernel_init_constant"]

    if isinstance(config["monotonicities"], list):
      negated_config["monotonicities"] = [
          -monotonicity for monotonicity in
          utils.canonicalize_monotonicities(config["monotonicities"])
      ]
    else:
      negated_config["monotonicities"] = -config["monotonicities"]

    negated_loss = self._TrainModel(negated_config)
    return negated_loss

  @parameterized.parameters((False, 1.623906), (True, 0.456815))
  # Expected losses are computed by running this test. Correctness is verified
  # manually by looking at visualisation of learned function vs ground truth.
  def testOneDUnconstrained(self, use_bias, expected_loss):
    if _DISABLE_ALL:
      return
    config = {
        "num_input_dims": 1,
        "use_bias": use_bias,
        "num_training_records": 128,
        "num_training_epoch": 400,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 1.5,
        "x_generator": self._ScaterXUniformly,
        "input_min": 5.0,
        "input_max": 25.0,
        "y_function": self._SinPlusXPlusD,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=_LOSS_EPS)

  @parameterized.parameters((False, 0.881774), (True, 0.441771))
  # Expected losses are computed by running this test. Correctness is verified
  # manually by looking at visualisation of learned function vs ground truth.
  def testTwoDUnconstrained(self, use_bias, expected_loss):
    if _DISABLE_ALL:
      return
    config = {
        "num_input_dims": 2,
        "use_bias": use_bias,
        "num_training_records": 64,
        "num_training_epoch": 160,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 1.5,
        "x_generator": self._TwoDMeshGrid,
        "input_min": 0.0,
        "input_max": 4.0,
        "y_function": self._GenLinearFunction(
            weights=[-1.0, 2.0],
            bias=-2.0,
            noise=lambda x: math.sin(sum(x)) / 1.0),
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=_LOSS_EPS)

  def testInitializers(self):
    # Test initializers by trying to fit linear function using 0 iterations.
    if _DISABLE_ALL:
      return
    config = {
        "num_input_dims": 2,
        "use_bias": True,
        "num_training_records": 64,
        "num_training_epoch": 0,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 1.5,
        "x_generator": self._TwoDMeshGrid,
        "input_min": 0.0,
        "input_max": 4.0,
        "kernel_init_constant": 3.0,
        "bias_init_constant": -2.0,
        "y_function": self._GenLinearFunction(weights=[3.0, 3.0], bias=-2.0)
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.0, delta=_LOSS_EPS)

  def testAssertConstraints(self):
    if _DISABLE_ALL:
      return
    config = {
        "num_input_dims": 4,
        "use_bias": True,
        "num_training_records": 64,
        "num_training_epoch": 0,
        "normalization_order": 1,
        "monotonicities": [1] * 4,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 1.5,
        "x_generator": self._ScaterXUniformly,
        "input_min": 0.0,
        "input_max": 4.0,
        "kernel_init_constant": 0.25,
        "bias_init_constant": -2.0,
        "y_function": self._GenLinearFunction(weights=[0.25] * 4, bias=-2.0)
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.0, delta=_LOSS_EPS)

    with self.assertRaises(tf.errors.InvalidArgumentError):
      config["normalization_order"] = 2
      self._TrainModel(config)

    with self.assertRaises(tf.errors.InvalidArgumentError):
      # Setting valid normalization order back and instead violating
      # monotonicity.
      config["normalization_order"] = 1
      config["monotonicities"] = [1, 1, -1, 0]
      self._TrainModel(config)

  @parameterized.parameters((False, 1.623906), (True, 0.456815))
  # Expected losses are computed by running this test. Correctness is verified
  # manually by looking at visualisation of learned function vs ground truth.
  def testOneDMonotonicities_MonotonicInput(self, use_bias, expected_loss):
    if _DISABLE_ALL:
      return
    config = {
        "num_input_dims": 1,
        "monotonicities": [1],
        "use_bias": use_bias,
        "num_training_records": 128,
        "num_training_epoch": 400,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 1.5,
        "x_generator": self._ScaterXUniformly,
        "input_min": 5.0,
        "input_max": 25.0,
        "y_function": self._SinPlusXPlusD,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=_LOSS_EPS)
    self.assertAlmostEqual(loss, self._NegateAndTrain(config), delta=_SMALL_EPS)

  @parameterized.parameters((False, 62.670425), (True, 3.326165))
  # Expected losses are computed by running this test. Correctness is verified
  # manually by looking at visualisation of learned function vs ground truth.
  def testOneDMonotonicities_AntiMonotonicInput(self, use_bias, expected_loss):
    if _DISABLE_ALL:
      return
    config = {
        "num_input_dims": 1,
        "monotonicities": ["increasing"],
        "use_bias": use_bias,
        "num_training_records": 128,
        "num_training_epoch": 400,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 1.5,
        "x_generator": self._ScaterXUniformly,
        "input_min": 5.0,
        "input_max": 25.0,
        "y_function": lambda x: -self._SinPlusXPlusD(x),
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=_LOSS_EPS)
    self.assertAlmostEqual(loss, self._NegateAndTrain(config), delta=_SMALL_EPS)

  @parameterized.parameters((1, 2.0), (1, -2.0), (2, 2.0), (2, -2.0))
  # Expected loss is computed by running this test. Correctness is verified
  # manually by looking at visualisation of learned function vs ground truth.
  def testOneDNormalizationOrder(self, norm_order, weight):
    if _DISABLE_ALL:
      return
    config = {
        "num_input_dims": 1,
        "monotonicities": [0],
        "normalization_order": norm_order,
        "use_bias": True,
        "num_training_records": 128,
        "num_training_epoch": 20,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 1.5,
        "x_generator": self._ScaterXUniformly,
        "input_min": 0.0,
        "input_max": 5.0,
        "y_function": self._GenLinearFunction(weights=[weight], bias=0.0)
    }  # pyformat: disable
    loss = self._TrainModel(config)
    # For 1-d case normalization order does not change anything.
    self.assertAlmostEqual(loss, 1.727717, delta=_LOSS_EPS)

  def testOneDNormalizationOrderZeroWeights(self):
    if _DISABLE_ALL:
      return
    # Normalization is impossible when all weights are 0.0 so weights should not
    # be affected by it.
    config = {
        "num_input_dims": 1,
        "monotonicities": ["none"],
        "normalization_order": 1,
        "use_bias": True,
        "num_training_records": 128,
        "num_training_epoch": 20,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 1.5,
        "x_generator": self._ScaterXUniformly,
        "input_min": 0.0,
        "input_max": 5.0,
        "y_function": self._GenLinearFunction(weights=[0.0], bias=0.0)
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, 0.0, delta=_LOSS_EPS)

  @parameterized.parameters(
      (0.441771, 0),
      (0.441771, ["none", "none"]),
      (2.61706, 1),
      (2.61706, ["increasing", "increasing"]),
      (2.61706, ["increasing", "none"]),
      (0.441771, ["none", "increasing"])
  )
  # Expected losses are computed by running this test. Correctness is verified
  # manually by looking at visualisation of learned function vs ground truth.
  def testTwoDMonotonicity(self, expected_loss, monotonicities):
    if _DISABLE_ALL:
      return
    config = {
        "num_input_dims": 2,
        "monotonicities": monotonicities,
        "use_bias": True,
        "num_training_records": 64,
        "num_training_epoch": 160,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 1.5,
        "x_generator": self._TwoDMeshGrid,
        "input_min": 0.0,
        "input_max": 4.0,
        "y_function": self._GenLinearFunction(
            weights=[-1.0, 2.0],
            bias=-2.0,
            noise=lambda x: math.sin(sum(x)) / 1.0)
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=_LOSS_EPS)
    self.assertAlmostEqual(loss, self._NegateAndTrain(config), delta=_SMALL_EPS)

    multioutput_config = dict(config)
    units = 3
    multioutput_config["units"] = units
    for unit_index in range(units):
      multioutput_config["unit_index"] = unit_index
      loss = self._TrainModel(multioutput_config)
      self.assertAlmostEqual(loss, expected_loss, delta=_LOSS_EPS)
      self.assertAlmostEqual(
          loss, self._NegateAndTrain(multioutput_config), delta=_SMALL_EPS)

  @parameterized.parameters(
      (1, [0.2, 0.3], 0, 0.250532),  # Testing sum of weights < 1.0.
      (1, [0.2, 0.3], 1, 0.250532),  # Monotonicity does not matter here.
      (2, [0.2, 0.3], 0, 0.753999),
      (1, [1.0, 2.0], 0, 5.688659),  # Testing sum of weights > 1.0.
      (1, [-1.0, 2.0], 0, 4.043515),
      # With negative weights monotonicity matters.
      (1, [-1.0, 2.0], 1, 3.433537))
  # Expected losses are computed by running this test. Correctness is verified
  # manually by looking at visualisation of learned function vs ground truth.
  def testTwoDNormalizationOrder(self, norm_order, weights, monotonicities,
                                 expected_loss):
    if _DISABLE_ALL:
      return
    config = {
        "num_input_dims": 2,
        "normalization_order": norm_order,
        "monotonicities": monotonicities,
        # If normalization order is set then layer will always converges to
        # extremes if there is no bias or other layers. That's why we always
        # use bias for normalization order tests.
        "use_bias": True,
        "num_training_records": 64,
        "num_training_epoch": 160,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 0.15,
        "x_generator": self._TwoDMeshGrid,
        "input_min": 0.0,
        "input_max": 4.0,
        "y_function": self._GenLinearFunction(
            weights=weights, noise=lambda x: math.sin(sum(x)) / 10.0)
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=_LOSS_EPS)

  @parameterized.parameters(
      ([0.5, 0.6, 0.06, 0.07, 0.08], [1, 1, 1, 1, 1], 0.0408642),
      ([0.5, -0.6, 0.06, -0.07, 0.08], [1, 1, 1, 1, 1], 0.561592),
      ([0.5, -0.6, 0.06, -0.07, 0.08], [0, 0, 1, 1, 1], 0.047663))
  # Expected losses are computed by running this test. Correctness is verified
  # manually by looking at visualisation of learned function vs ground truth.
  def testFiveDAllConstraints(self, weights, monotonicities, expected_loss):
    if _DISABLE_ALL:
      return
    config = {
        "num_input_dims": 5,
        "normalization_order": 1,
        "monotonicities": monotonicities,
        "use_bias": True,
        "num_training_records": 640,
        "num_training_epoch": 160,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 1.5,
        "x_generator": self._ScaterXUniformly,
        "input_min": 0.0,
        "kernel_init_constant": 0.7,
        "input_max": 4.0,
        "y_function": self._GenLinearFunction(
            weights=weights, noise=lambda x: math.sin(sum(x)) / 30.0)
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=_LOSS_EPS)

  @parameterized.parameters((0.85766, [(0, 1)]),
                            (1e-13, [(1, 0)]))
  # Expected losses are computed by running this test. Correctness is verified
  # manually by looking at visualisation of learned function vs ground truth.
  def testTwoDMonotonicDominance(self, expected_loss, dominances):
    if _DISABLE_ALL:
      return
    config = {
        "num_input_dims": 2,
        "monotonicities": ["increasing", "increasing"],
        "monotonic_dominances": dominances,
        "num_training_records": 64,
        "num_training_epoch": 160,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 1.5,
        "x_generator": self._TwoDMeshGrid,
        "input_min": 0.0,
        "input_max": 4.0,
        "y_function": self._GenLinearFunction(weights=[1.0, 2.0])
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=_LOSS_EPS)

  @parameterized.parameters(([(0, 1)], [1, 1, 0], [1.0, 2.0, 3.0], 1.8409),
                            ([(0, 1)], [-1, -1, 0], [-1.0, -2.0, -3.0], 1.8409),
                            ([(1, 0)], [1, 1, 0], [1.0, 2.0, 3.0], 0.6567),
                            ([(1, 0)], [-1, -1, 0], [-1.0, -2.0, -3.0], 0.6567))
  # Expected losses are computed by running this test. Correctness is verified
  # manually by looking at visualisation of learned function vs ground truth.
  def testTwoDRangeDominance(self, dominances, monotonicities, weights,
                             expected_loss):
    if _DISABLE_ALL:
      return
    config = {
        "num_input_dims": 3,
        "monotonicities": monotonicities,
        "range_dominances": dominances,
        "clip_min": [0.0, 0.0, "none"],
        "clip_max": (1.0, 4.0, "none"),
        "num_training_records": 64,
        "num_training_epoch": 160,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 1.5,
        "x_generator": self._ScaterXUniformly,
        "input_min": 0.0,
        "input_max": 4.0,
        "y_function": self._GenLinearFunction(weights=weights)
    }  # pyformat: disable
    loss = self._TrainModel(config)
    self.assertAlmostEqual(loss, expected_loss, delta=_LOSS_EPS)

  @parameterized.parameters(
      # Standard Keras regularizer:
      (keras.regularizers.l1_l2(l1=0.01, l2=0.001),),
      # Tuple of regularizers:
      ((keras.regularizers.l1_l2(l1=0.01, l2=0.0),
        keras.regularizers.l1_l2(l1=0.0, l2=0.001)),),
  )
  def testRegularizers(self, regularizer):
    if _DISABLE_ALL:
      return
    config = {
        "num_input_dims": 2,
        "use_bias": True,
        "num_training_records": 64,
        "num_training_epoch": 0,
        "optimizer": keras.optimizers.Adagrad,
        "learning_rate": 1.5,
        "x_generator": self._TwoDMeshGrid,
        "input_min": 0.0,
        "input_max": 4.0,
        "kernel_init_constant": 2.0,
        "bias_init_constant": 3.0,
        "y_function": self._GenLinearFunction(weights=[2.0, 2.0], bias=3.0),
        "kernel_regularizer": regularizer,
        "bias_regularizer": regularizer,
    }  # pyformat: disable
    loss = self._TrainModel(config)
    # This loss is pure regularization loss because initializer matches target
    # function and there was 0 training epochs.
    self.assertAlmostEqual(loss, 0.087, delta=_LOSS_EPS)

if __name__ == "__main__":
  tf.test.main()
