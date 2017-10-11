# Copyright 2017 The TensorFlow Lattice Authors.
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
# ==============================================================================
"""Tests for hypercube interpolation gradient."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensorflow_lattice.python.ops import lattice_ops

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


class LatticeGradientOpTest(test.TestCase):

  def _testGradient(self, x_value_list, x_shape, lattice_sizes, y_shape,
                    is_hypercube):
    """Compute the numerical gradients, and check the error."""
    for x_value in x_value_list:
      with self.test_session(use_gpu=False):
        x = array_ops.placeholder(dtype=dtypes.float32, shape=x_shape, name="x")
        x_init_value = np.asarray(x_value, dtype=np.float32)
        if is_hypercube:
          y = lattice_ops.hypercube_interpolation(
              x, lattice_sizes=lattice_sizes)
        else:
          y = lattice_ops.simplex_interpolation(x, lattice_sizes=lattice_sizes)
        error = gradient_checker.compute_gradient_error(
            x, x_shape, y, y_shape, x_init_value=x_init_value)
      tf_logging.info("x_init_value = %s" % x_init_value)
      tf_logging.info("x error = %f", error)
      self.assertTrue(error < 1e-4)

  def _testGradientWith1DInput(self, is_hypercube):
    x_value_list = [[[-1.0]], [[0.1]], [[0.5]], [[1.001]], [[1.5]], [[2.001]],
                    [[3.0]]]
    x_shape = (1, 1)
    lattice_sizes = [3]
    # interpolation_weight_size = 3.
    y_shape = (1, 3)
    self._testGradient(
        x_value_list,
        x_shape,
        lattice_sizes,
        y_shape,
        is_hypercube=is_hypercube)

  def testHypercubeGradientWith1DInput(self):
    self._testGradientWith1DInput(is_hypercube=True)

  def testSimplexGradientWith1DInput(self):
    self._testGradientWith1DInput(is_hypercube=False)

  def _testGradientWith2DInput(self, is_hypercube):
    x_value_list = [[[-1.0, 1.1]], [[0.1, 0.09]], [[0.5, 2.3]], [[1.001, 0.98]],
                    [[1.5, 0.34]], [[2.001, 10.0]], [[3.0, 0.5]]]
    x_shape = (1, 2)
    lattice_sizes = [3, 2]
    # interpolation_weight_size = 6.
    y_shape = (1, 6)
    self._testGradient(
        x_value_list,
        x_shape,
        lattice_sizes,
        y_shape,
        is_hypercube=is_hypercube)

  def testHypercubeGradientWith2DInput(self):
    self._testGradientWith2DInput(is_hypercube=True)

  def testSimplexGradientWith2DInput(self):
    self._testGradientWith2DInput(is_hypercube=False)

  def _testGradientWith3DInput(self, is_hypercube):
    x_value_list = [[[-1.0, 1.1, 2.11]], [[0.1, 0.099, 0.111]],
                    [[0.5, 2.3, 2.212]], [[1.001, 0.98, 0.123]],
                    [[1.5, 0.34, 0.3312]], [[2.001, 10.0, 9.0]],
                    [[3.0, 0.5, -1.22]]]
    x_shape = (1, 3)
    lattice_sizes = [3, 3, 5]
    # interpolation_weight_size = 45.
    y_shape = (1, 45)
    self._testGradient(x_value_list, x_shape, lattice_sizes, y_shape,
                       is_hypercube)

  def testHypercubeGradientWith3DInput(self):
    self._testGradientWith3DInput(is_hypercube=True)

  def testSimplexGradientWith3DInput(self):
    self._testGradientWith3DInput(is_hypercube=False)

  def testSimplexGradientWith3DBatchInput(self):
    x_value_list = [[[0.5, 0.1, 0.3], [0.11, 0.3, 0.79], [0.33, 0.5, 0.79]]]
    x_shape = (3, 3)
    lattice_sizes = [2, 2, 2]
    # interpolation_weight_size = 8.
    y_shape = (3, 8)
    self._testGradient(
        x_value_list, x_shape, lattice_sizes, y_shape, is_hypercube=False)


class LatticeGradientBoundaryTest(test.TestCase):

  def _testGradient(self, inputs, weights, expected_jacobians_wrt_input,
                    lattice_sizes, is_hypercube):
    """Compute the grad_wrt_input and compare it with expected_grad_wrt_input.

    Args:
      inputs: a 2D array (or numpy array) contains the test inputs. Its shape
        should be num_examples x input_size.
      weights: a 2D array (or numpy array) contains the test weights. Its
        shape should be num_examples x weight_size.
      expected_jacobians_wrt_input: 3D array (or numpy) contains  a transpoed
        jacobian matrix that contains dweight/dinput with shape (num_examples,
        weight_size, input_size).
        In other words, expected_jacobians_wrt_input[num][ii][jj] ==
          dweight[num][jj]/dinput[num][ii], where num means the current example.
      lattice_sizes: A list of lattice_sizes.
      is_hypercube: If true, hypercube gradient is tested, otherwise simplex
        gradient is tested.

    Returns: None

    Raises: Fails if computed jacobian_wrt_inputs != expected_jacobian_wrt_inpu.
    """

    # Number of test examples in inputs.
    num_examples = len(inputs)
    weight_size = len(weights[0])

    # Define the grad_wrt_input_tensor.
    with ops.Graph().as_default():
      input_tensor = constant_op.constant(inputs, dtype=dtypes.float32)
      weight_tensor = constant_op.constant(weights, dtype=dtypes.float32)
      grad_wrt_weight_tensor = array_ops.placeholder(
          dtype=dtypes.float32, shape=(num_examples, weight_size))

      if is_hypercube:
        grad_wrt_input_tensor = lattice_ops.hypercube_gradient(
            input_tensor, weight_tensor, grad_wrt_weight_tensor, lattice_sizes)
      else:
        grad_wrt_input_tensor = lattice_ops.simplex_gradient(
            input_tensor, weight_tensor, grad_wrt_weight_tensor, lattice_sizes)

      # Compute the Jacobian.
      with self.test_session(use_gpu=False):
        tf_logging.info("input = %s " % inputs)
        tf_logging.info("weight = %s " % weights)
        # num_examples x weight_size x input_size tensor.
        jacobians_wrt_input = []
        # Compute dweight[cnt] / dinput.
        for cnt in range(weight_size):
          grad_wrt_weight = [0.] * weight_size
          grad_wrt_weight[cnt] = 1.0
          grad_wrt_weights = [grad_wrt_weight for _ in range(num_examples)]
          tf_logging.info("grad_wrt_weights = %s " % grad_wrt_weights)
          # num_examples x input_size matrix.
          grad_weight_wrt_inputs = grad_wrt_input_tensor.eval(
              feed_dict={grad_wrt_weight_tensor: grad_wrt_weights})
          tf_logging.info("grad_wrt_inputs = %s " % grad_weight_wrt_inputs)
          jacobians_wrt_input.append(grad_weight_wrt_inputs)
      tf_logging.info("jacobian_wrt_inputs = %s " % jacobians_wrt_input)
      tf_logging.info("expected_jacobian_wrt_inputs = %s" %
                      expected_jacobians_wrt_input)
      self.assertAllClose(jacobians_wrt_input, expected_jacobians_wrt_input)

  def _test1DLatticeInputAtBoundary(self, is_hypercube):
    # 1D lattice.
    lattice_sizes = [4]
    # Values at the boundaries.
    inputs = [[-1.0], [0.0], [1.0], [2.0], [3.0], [4.0]]
    # Interpolation weights and grad_wrt_weights.
    weights = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
    # Total 6 test points. So expected_jacobian_wrt_input = 6 x 4 x 1 matrix,
    # where iith row contains dweight[ii]/dinput[0].
    # Jacobain for the input -1.0:
    #  [0, 0, 0, 0].
    # Jacobian for the input, 0.0:
    #  [-1, 1, 0, 0]
    # Jacobian for the input, 1.0:
    #  [0, -1, 1, 0]
    # Jacobian for the input, 2.0:
    #  [0, 0, -1, 1]
    # Jacobian for the input, 3.0:
    #  [0, 0, -1, 1]
    # Jacobian for the input, 4.0:
    #  [0, 0, 0, 0]
    expected_jacobian_wrt_input = [[[0], [-1], [0], [0], [0], [0]],
                                   [[0], [1], [-1], [0], [0], [0]],
                                   [[0], [0], [1], [-1], [-1], [0]],
                                   [[0], [0], [0], [1], [1], [0]]]

    self._testGradient(
        inputs,
        weights,
        expected_jacobian_wrt_input,
        lattice_sizes,
        is_hypercube=is_hypercube)

  def testHypercubeWith1DLatticeInputAtBoundary(self):
    self._test1DLatticeInputAtBoundary(is_hypercube=True)

  def testSimplexWith1DLatticeInputAtBoundary(self):
    self._test1DLatticeInputAtBoundary(is_hypercube=False)


if __name__ == "__main__":
  test.main()
