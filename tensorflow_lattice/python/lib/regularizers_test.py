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
"""Tests for TensorFlow Lattice's keypoints_initialization module."""
# Dependency imports
from tensorflow_lattice.python.lib import regularizers

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class CalibratorLaplacianTestCase(test.TestCase):

  def setUp(self):
    self._num_examples = 4
    self._keypoint_lists = [
        [0.0, 0.1, 1.0],  # for better formatting
        [-1.0, 0.2, 0.3, 0.5],
        [1.11, 2.11, -1.5, -10.232],
        [2.22, -51.1, 321.0, 33.22, -201.0, -50.0]
    ]
    # L1 regularization amount assuming 1.0 weight.
    self._l1_regs = [1.0, 1.4999999999999998, 13.34199999999999, 1098.42]
    # L2 regularization amount assuming 1.0 weight.
    self._l2_regs = [0.8200000000000001, 1.49, 90.28, 301778.78]

    super(CalibratorLaplacianTestCase, self).setUp()

  def _runAndCheckValues(self,
                         output_keypoints,
                         expected_value,
                         l1_reg=None,
                         l2_reg=None):
    output_keypoints_tensor = array_ops.constant(
        output_keypoints, dtype=dtypes.float32)
    reg = regularizers.calibrator_regularization(
        output_keypoints_tensor,
        l1_laplacian_reg=l1_reg,
        l2_laplacian_reg=l2_reg)
    with self.test_session() as sess:
      reg_value = sess.run(reg)
    self.assertAlmostEqual(reg_value, expected_value, delta=1e-1)

  def testL1Regularizer(self):
    """Check l1 regularization amount."""
    l1_reg = 1.0
    for cnt in range(self._num_examples):
      expected_value = l1_reg * self._l1_regs[cnt]
      self._runAndCheckValues(
          self._keypoint_lists[cnt], expected_value, l1_reg=l1_reg)

  def testL2Regularizer(self):
    """Check l2 regularization amount."""
    l2_reg = 1.0
    for cnt in range(self._num_examples):
      expected_value = l2_reg * self._l2_regs[cnt]
      self._runAndCheckValues(
          self._keypoint_lists[cnt], expected_value, l2_reg=l2_reg)

  def testL1AndL2Regularizers(self):
    """Check l1 and l2 regularization amount."""
    l1_reg = 0.5
    l2_reg = 0.5
    for cnt in range(self._num_examples):
      expected_value = l1_reg * self._l1_regs[cnt] + l2_reg * self._l2_regs[cnt]
      self._runAndCheckValues(
          self._keypoint_lists[cnt],
          expected_value,
          l1_reg=l1_reg,
          l2_reg=l2_reg)

  def testRank2TensorExpectsError(self):
    """Pass rank-2 tensor output keypoints and check the error."""
    output_keypoints_tensor = array_ops.placeholder(
        dtype=dtypes.float32, shape=[10, 10])
    with self.assertRaises(ValueError):
      regularizers.calibrator_regularization(output_keypoints_tensor)

  def testUnknownShapeTensorExpectsError(self):
    """Pass rank-1 tensor with unknown shape and check the error."""
    output_keypoints_tensor = array_ops.placeholder(
        dtype=dtypes.float32, shape=[None])
    with self.assertRaises(ValueError):
      regularizers.calibrator_regularization(output_keypoints_tensor)

  def testOneKeypointsExpectsNone(self):
    """Pass a tensor with one keypoints and check None regularizer."""
    output_keypoints_tensor = array_ops.placeholder(
        dtype=dtypes.float32, shape=[1])
    self.assertEqual(
        regularizers.calibrator_regularization(output_keypoints_tensor), None)

  def testNoRegularizerExpectsNone(self):
    """Set no l1_reg and l2_reg and check None regularizer."""
    output_keypoints_tensor = array_ops.placeholder(
        dtype=dtypes.float32, shape=[2])
    self.assertEqual(
        regularizers.calibrator_regularization(output_keypoints_tensor), None)


class CalibratorHessianTestCase(test.TestCase):

  def setUp(self):
    self._num_examples = 4
    self._keypoint_lists = [
        [0.0, 0.1, 1.0],  # for better formatting
        [-1.0, 0.2, 0.3, 0.5],
        [1.11, 2.11, -1.5, -10.232],
        [2.22, -51.1, 321.0, 33.22, -201.0, -50.0]
    ]
    # L1 regularization amount assuming 1.0 weight.
    self._l1_regs = [0.8, 1.2, 9.732, 1524.08]
    # L2 regularization amount assuming 1.0 weight.
    self._l2_regs = [0.64, 1.22, 47.486984, 767686.9128]

    super(CalibratorHessianTestCase, self).setUp()

  def _runAndCheckValues(self,
                         output_keypoints,
                         expected_value,
                         l1_reg=None,
                         l2_reg=None):
    output_keypoints_tensor = array_ops.constant(
        output_keypoints, dtype=dtypes.float32)
    reg = regularizers.calibrator_regularization(
        output_keypoints_tensor, l1_hessian_reg=l1_reg, l2_hessian_reg=l2_reg)
    with self.test_session() as sess:
      reg_value = sess.run(reg)
    self.assertAlmostEqual(reg_value, expected_value, delta=1e-1)

  def testL1Regularizer(self):
    """Check l1 regularization amount."""
    l1_reg = 1.0
    for cnt in range(self._num_examples):
      expected_value = l1_reg * self._l1_regs[cnt]
      self._runAndCheckValues(
          self._keypoint_lists[cnt], expected_value, l1_reg=l1_reg)

  def testL2Regularizer(self):
    """Check l2 regularization amount."""
    l2_reg = 1.0
    for cnt in range(self._num_examples):
      expected_value = l2_reg * self._l2_regs[cnt]
      self._runAndCheckValues(
          self._keypoint_lists[cnt], expected_value, l2_reg=l2_reg)

  def testL1AndL2Regularizers(self):
    """Check l1 and l2 regularization amount."""
    l1_reg = 0.5
    l2_reg = 0.5
    for cnt in range(self._num_examples):
      expected_value = l1_reg * self._l1_regs[cnt] + l2_reg * self._l2_regs[cnt]
      self._runAndCheckValues(
          self._keypoint_lists[cnt],
          expected_value,
          l1_reg=l1_reg,
          l2_reg=l2_reg)

  def testRank2TensorExpectsError(self):
    """Pass rank-2 tensor output keypoints and check the error."""
    output_keypoints_tensor = array_ops.placeholder(
        dtype=dtypes.float32, shape=[10, 10])
    with self.assertRaises(ValueError):
      regularizers.calibrator_regularization(output_keypoints_tensor)

  def testUnknownShapeTensorExpectsError(self):
    """Pass rank-1 tensor with unknown shape and check the error."""
    output_keypoints_tensor = array_ops.placeholder(
        dtype=dtypes.float32, shape=[None])
    with self.assertRaises(ValueError):
      regularizers.calibrator_regularization(output_keypoints_tensor)

  def testTwoKeypointsExpectsNone(self):
    """Pass a tensor with one keypoints and check None regularizer."""
    output_keypoints_tensor = array_ops.placeholder(
        dtype=dtypes.float32, shape=[2])
    self.assertEqual(
        regularizers.calibrator_regularization(output_keypoints_tensor), None)

  def testNoRegularizerExpectsNone(self):
    """Set no l1_reg and l2_reg and check None regularizer."""
    output_keypoints_tensor = array_ops.placeholder(
        dtype=dtypes.float32, shape=[2])
    self.assertEqual(
        regularizers.calibrator_regularization(output_keypoints_tensor), None)


class CalibratorWrinkleTestCase(test.TestCase):

  def setUp(self):
    self._num_examples = 4
    self._keypoint_lists = [
        [0.1, 0.1, 0.1, 0.1],  # constant
        [1.0, 2.0, 3.0, 4.0],  # linear
        [0.0, 1.0, 4.0, 9.0],  # 2nd degree polynomial
        [0.0, 1.0, 4.0, 11.0]
    ]
    # L1 regularization amount assuming 1.0 weight.
    self._l1_regs = [0.0, 0.0, 0.0, 2.0]
    # L2 regularization amount assuming 1.0 weight.
    self._l2_regs = [0.0, 0.0, 0.0, 4.0]

    super(CalibratorWrinkleTestCase, self).setUp()

  def _runAndCheckValues(self,
                         output_keypoints,
                         expected_value,
                         l1_reg=None,
                         l2_reg=None):
    output_keypoints_tensor = array_ops.constant(
        output_keypoints, dtype=dtypes.float32)
    reg = regularizers.calibrator_regularization(
        output_keypoints_tensor, l1_wrinkle_reg=l1_reg, l2_wrinkle_reg=l2_reg)
    with self.test_session() as sess:
      reg_value = sess.run(reg)
    self.assertAlmostEqual(reg_value, expected_value, delta=1e-1)

  def testL1Regularizer(self):
    """Check l1 regularization amount."""
    l1_reg = 1.0
    for cnt in range(self._num_examples):
      expected_value = l1_reg * self._l1_regs[cnt]
      self._runAndCheckValues(
          self._keypoint_lists[cnt], expected_value, l1_reg=l1_reg)

  def testL2Regularizer(self):
    """Check l2 regularization amount."""
    l2_reg = 1.0
    for cnt in range(self._num_examples):
      expected_value = l2_reg * self._l2_regs[cnt]
      self._runAndCheckValues(
          self._keypoint_lists[cnt], expected_value, l2_reg=l2_reg)

  def testL1AndL2Regularizers(self):
    """Check l1 and l2 regularization amount."""
    l1_reg = 0.5
    l2_reg = 0.5
    for cnt in range(self._num_examples):
      expected_value = l1_reg * self._l1_regs[cnt] + l2_reg * self._l2_regs[cnt]
      self._runAndCheckValues(
          self._keypoint_lists[cnt],
          expected_value,
          l1_reg=l1_reg,
          l2_reg=l2_reg)

  def testRank2TensorExpectsError(self):
    """Pass rank-2 tensor output keypoints and check the error."""
    output_keypoints_tensor = array_ops.placeholder(
        dtype=dtypes.float32, shape=[10, 10])
    with self.assertRaises(ValueError):
      regularizers.calibrator_regularization(output_keypoints_tensor)

  def testUnknownShapeTensorExpectsError(self):
    """Pass rank-1 tensor with unknown shape and check the error."""
    output_keypoints_tensor = array_ops.placeholder(
        dtype=dtypes.float32, shape=[None])
    with self.assertRaises(ValueError):
      regularizers.calibrator_regularization(output_keypoints_tensor)

  def testTwoKeypointsExpectsNone(self):
    """Pass a tensor with one keypoints and check None regularizer."""
    output_keypoints_tensor = array_ops.placeholder(
        dtype=dtypes.float32, shape=[2])
    self.assertEqual(
        regularizers.calibrator_regularization(output_keypoints_tensor), None)

  def testNoRegularizerExpectsNone(self):
    """Set no l1_reg and l2_reg and check None regularizer."""
    output_keypoints_tensor = array_ops.placeholder(
        dtype=dtypes.float32, shape=[2])
    self.assertEqual(
        regularizers.calibrator_regularization(output_keypoints_tensor), None)


class CalibratorRegularizersTestCase(test.TestCase):

  def setUp(self):
    self._num_examples = 4
    self._keypoint_lists = [
        [0.0, 0.1, 1.0],  # for better formatting
        [-1.0, 0.2, 0.3, 0.5],
        [1.11, 2.11, -1.5, -10.232],
        [2.22, -51.1, 321.0, 33.22, -201.0, -50.0]
    ]
    # L1 regularization amount assuming 1.0 weight.
    self._l1_regs = [1.1, 2.0, 14.952, 658.54]
    # L2 regularization amount assuming 1.0 weight.
    self._l2_regs = [1.01, 1.38, 112.628024, 149661.7068]
    # L1 laplacian regularization amount assuming 1.0 weight.
    self._l1_laplacian_regs = [
        1.0, 1.4999999999999998, 13.34199999999999, 1098.42
    ]
    # L2 laplacian regularization amount assuming 1.0 weight.
    self._l2_laplacian_regs = [0.8200000000000001, 1.49, 90.28, 301778.78]

    super(CalibratorRegularizersTestCase, self).setUp()

  def _runAndCheckValues(self,
                         output_keypoints,
                         expected_value,
                         l1_reg=None,
                         l2_reg=None,
                         l1_laplacian_reg=None,
                         l2_laplacian_reg=None):
    output_keypoints_tensor = array_ops.constant(
        output_keypoints, dtype=dtypes.float32)
    reg = regularizers.calibrator_regularization(
        output_keypoints_tensor,
        l1_reg=l1_reg,
        l2_reg=l2_reg,
        l1_laplacian_reg=l1_laplacian_reg,
        l2_laplacian_reg=l2_laplacian_reg)
    with self.test_session() as sess:
      reg_value = sess.run(reg)
    self.assertAlmostEqual(reg_value, expected_value, delta=1e-1)

  def testL1Regularizer(self):
    """Check l1 regularization amount."""
    l1_reg = 1.0
    for cnt in range(self._num_examples):
      expected_value = l1_reg * self._l1_regs[cnt]
      self._runAndCheckValues(
          self._keypoint_lists[cnt], expected_value, l1_reg=l1_reg)

  def testL2Regularizer(self):
    """Check l2 regularization amount."""
    l2_reg = 2.0
    for cnt in range(self._num_examples):
      expected_value = l2_reg * self._l2_regs[cnt]
      self._runAndCheckValues(
          self._keypoint_lists[cnt], expected_value, l2_reg=l2_reg)

  def testL1AndL2Regularizers(self):
    """Check l1 and l2 regularization amount."""
    l1_reg = 0.5
    l2_reg = 0.5
    for cnt in range(self._num_examples):
      expected_value = l1_reg * self._l1_regs[cnt] + l2_reg * self._l2_regs[cnt]
      self._runAndCheckValues(
          self._keypoint_lists[cnt],
          expected_value,
          l1_reg=l1_reg,
          l2_reg=l2_reg)

  def testAllRegularizers(self):
    """Check l1, l2 and laplacian regularization amount."""
    l1_reg = 0.5
    l2_reg = 0.5
    l1_laplacian_reg = 0.5
    l2_laplacian_reg = 0.5
    for cnt in range(self._num_examples):
      expected_value = l1_reg * self._l1_regs[cnt]
      expected_value += l2_reg * self._l2_regs[cnt]
      expected_value += l1_laplacian_reg * self._l1_laplacian_regs[cnt]
      expected_value += l2_laplacian_reg * self._l2_laplacian_regs[cnt]
      self._runAndCheckValues(
          self._keypoint_lists[cnt],
          expected_value,
          l1_reg=l1_reg,
          l2_reg=l2_reg,
          l1_laplacian_reg=l1_laplacian_reg,
          l2_laplacian_reg=l2_laplacian_reg)


class LatticeLaplacianTestCase(test.TestCase):

  def _runAndCheckValues(self,
                         lattice_param,
                         lattice_sizes,
                         expected_value,
                         l1_reg=None,
                         l2_reg=None):
    lattice_param_tensor = array_ops.constant(
        lattice_param, dtype=dtypes.float32)
    reg = regularizers.lattice_regularization(
        lattice_param_tensor,
        lattice_sizes,
        l1_laplacian_reg=l1_reg,
        l2_laplacian_reg=l2_reg)
    with self.test_session() as sess:
      reg_value = sess.run(reg)
    self.assertAlmostEqual(reg_value, expected_value, delta=1e-1)

  def testZeroRegularizerValueForVariousLatticeRanks(self):
    """Check zero output value for zero parameters."""
    for lattice_rank in range(2, 10):
      param_dim = 2**lattice_rank
      lattice_param = [[0.0] * param_dim]
      self._runAndCheckValues(
          lattice_param=lattice_param,
          lattice_sizes=[2] * lattice_rank,
          expected_value=0.0,
          l1_reg=1.0,
          l2_reg=1.0)

  def testL1RegularizerWithTwoByTwo(self):
    """Check l1 regularization amount for two by two lattices.

    In 2 x 2 lattice, L1 Laplacian regualrizer has the following form:

      l1_first_reg = (abs(param[1] - param[0]) + abs(param[3] - param[2])
        + abs(param[5] - param[4]) + abs(param[7] - param[6])
        + abs(param[9] - param[8]) + abs(param[11] - param[10]))

      l1_second_reg = (abs(param[2] - param[0]) + abs(param[4] - param[2])
                + abs(param[3] - param[1]) + abs(param[5] - param[3])
                + abs(param[8] - param[6]) + abs(param[10] - param[8])
                + abs(param[9] - param[7]) + abs(param[11] - param[9]))

      l1_reg = l1_reg[0] * l1_first_reg + l1_reg[1] * l1_second_reg,

    where param is the lattice parameter tensor (assuming one output),
    l1_first_reg is the regularization amount along the first dimension,
    l1_second_reg is the regularization amount along the second dimension.
    """
    self._runAndCheckValues(
        lattice_param=[[0.0, 0.0, 1.0, 1.0]],
        lattice_sizes=[2, 2],
        expected_value=2.0,
        l1_reg=1.0)
    self._runAndCheckValues(
        lattice_param=[[0.0, 0.0, 1.0, 1.0]],
        lattice_sizes=[2, 2],
        expected_value=2.0,
        l1_reg=[0.0, 1.0])
    self._runAndCheckValues(
        lattice_param=[[0.0, 0.0, 1.0, 1.0]],
        lattice_sizes=[2, 2],
        expected_value=0.0,
        l1_reg=[1.0, 0.0])
    self._runAndCheckValues(
        lattice_param=[[0.3312, -0.3217, -0.5, 0.1]],
        lattice_sizes=[2, 2],
        expected_value=2.5058,
        l1_reg=1.0)
    self._runAndCheckValues(
        lattice_param=[[0.3312, -0.3217, -0.5, 0.1]],
        lattice_sizes=[2, 2],
        expected_value=1.87935,
        l1_reg=[0.5, 1.0])
    self._runAndCheckValues(
        lattice_param=[[0.0, 0.0, 1.0, 1.0], [0.3312, -0.3217, -0.5, 0.1]],
        lattice_sizes=[2, 2],
        expected_value=4.5058,
        l1_reg=1.0)

  def testL1RegularizerWithTwoByThreeByTwo(self):
    """Check l1 regularization amount for two by three by two lattices.

    In 2 x 3 x 2 lattice, L1 Laplacian regualrizer has the following form:

      l1_first_reg = (abs(param[1] - param[0]) + abs(param[3] - param[2])
        + abs(param[5] - param[4]) + abs(param[7] - param[6])
        + abs(param[9] - param[8]) + abs(param[11] - param[10]))
      l1_second_reg = (abs(param[2] - param[0]) + abs(param[4] - param[2])
        + abs(param[3] - param[1]) + abs(param[5] - param[3])
        + abs(param[8] - param[6]) + abs(param[10] - param[8])
        + abs(param[9] - param[7]) + abs(param[11] - param[9]))
      l1_third_reg = (abs(param[6] - param[0]) + abs(param[7] - param[1])
        + abs(param[8] - param[2]) + abs(param[9] - param[3])
        + abs(param[10] - param[4]) + abs(param[11] - param[5]))

      l1_reg = l1_reg[0] * l1_first_reg + l1_reg[1] * l1_second_reg
        + l1_reg[2] * l1_third_reg,

    where param is the lattice parameter tensor (assuming one output),
    l1_first_reg is the regularization amount along the first dimension,
    l1_second_reg is the regularization amount along the second dimension.
    l1_third_reg is the regularization amount along the third dimension.
    """
    self._runAndCheckValues(
        lattice_param=[[
            1.11, 2.22, -3.22, 0.33, -0.221, 3.123, -0.477, 1.22, 3.221, 11.22,
            22.12, 33.11
        ]],
        lattice_sizes=[2, 3, 2],
        expected_value=28.69,
        l1_reg=[1.0, 0.0, 0.0])
    self._runAndCheckValues(
        lattice_param=[[
            1.11, 2.22, -3.22, 0.33, -0.221, 3.123, -0.477, 1.22, 3.221, 11.22,
            22.12, 33.11
        ]],
        lattice_sizes=[2, 3, 2],
        expected_value=66.499,
        l1_reg=[0.0, 1.0, 0.0])
    self._runAndCheckValues(
        lattice_param=[[
            1.11, 2.22, -3.22, 0.33, -0.221, 3.123, -0.477, 1.22, 3.221, 11.22,
            22.12, 33.11
        ]],
        lattice_sizes=[2, 3, 2],
        expected_value=72.246,
        l1_reg=[0.0, 0.0, 1.0])
    self._runAndCheckValues(
        lattice_param=[[
            1.11, 2.22, -3.22, 0.33, -0.221, 3.123, -0.477, 1.22, 3.221, 11.22,
            22.12, 33.11
        ]],
        lattice_sizes=[2, 3, 2],
        expected_value=167.435,
        l1_reg=1.0)
    self._runAndCheckValues(
        lattice_param=[[
            1.11, 2.22, -3.22, 0.33, -0.221, 3.123, -0.477, 1.22, 3.221, 11.22,
            22.12, 33.11
        ], [
            -2.003, 1.221, 0.321, 0.447, 0.321, 0.446, -0.33192, 0.476, 0.8976,
            -4.123, 0.487, -0.4473
        ]],
        lattice_sizes=[2, 3, 2],
        expected_value=199.30862,
        l1_reg=1.0)

  def testL2RegularizerWithTwoByTwo(self):
    """Check l2 regularization amount.

    Replacing abs to square in the formula in testL1RegulairzerWithTwoByTwo
    gives L2 Laplacian.
    """
    self._runAndCheckValues(
        lattice_param=[[0.0, 0.0, 1.0, 1.0]],
        lattice_sizes=[2, 2],
        expected_value=2.0,
        l2_reg=1.0)
    self._runAndCheckValues(
        lattice_param=[[0.0, 0.0, 1.0, 1.0]],
        lattice_sizes=[2, 2],
        expected_value=2.0,
        l2_reg=[0.0, 1.0])
    self._runAndCheckValues(
        lattice_param=[[0.0, 0.0, 1.0, 1.0]],
        lattice_sizes=[2, 2],
        expected_value=0.0,
        l2_reg=[1.0, 0.0])
    self._runAndCheckValues(
        lattice_param=[[0.3312, -0.3217, -0.5, 0.1]],
        lattice_sizes=[2, 2],
        expected_value=1.65500274,
        l2_reg=1.0)
    self._runAndCheckValues(
        lattice_param=[[0.3312, -0.3217, -0.5, 0.1]],
        lattice_sizes=[2, 2],
        expected_value=1.261863535,
        l2_reg=[0.5, 1.0])
    self._runAndCheckValues(
        lattice_param=[[0.0, 0.0, 1.0, 1.0], [0.3312, -0.3217, -0.5, 0.1]],
        lattice_sizes=[2, 2],
        expected_value=3.65500274,
        l2_reg=1.0)

  def testL2RegularizerWithTwoByThreeByTwo(self):
    """Check l2 regularization amount for two by three by two lattices.

    Replacing abs to square in the formula in
    testL1RegulairzerWithTwoByThreeByTwo gives L2 Laplacian.
    """
    self._runAndCheckValues(
        lattice_param=[[
            1.11, 2.22, -3.22, 0.33, -0.221, 3.123, -0.477, 1.22, 3.221, 11.22,
            22.12, 33.11
        ]],
        lattice_sizes=[2, 3, 2],
        expected_value=2763.733801,
        l2_reg=1.0)
    self._runAndCheckValues(
        lattice_param=[[
            1.11, 2.22, -3.22, 0.33, -0.221, 3.123, -0.477, 1.22, 3.221, 11.22,
            22.12, 33.11
        ]],
        lattice_sizes=[2, 3, 2],
        expected_value=212.660846,
        l2_reg=[1.0, 0.0, 0.0])
    self._runAndCheckValues(
        lattice_param=[[
            1.11, 2.22, -3.22, 0.33, -0.221, 3.123, -0.477, 1.22, 3.221, 11.22,
            22.12, 33.11
        ]],
        lattice_sizes=[2, 3, 2],
        expected_value=989.135355,
        l2_reg=[0.0, 1.0, 0.0])
    self._runAndCheckValues(
        lattice_param=[[
            1.11, 2.22, -3.22, 0.33, -0.221, 3.123, -0.477, 1.22, 3.221, 11.22,
            22.12, 33.11
        ]],
        lattice_sizes=[2, 3, 2],
        expected_value=1561.9376,
        l2_reg=[0.0, 0.0, 1.0])
    self._runAndCheckValues(
        lattice_param=[[
            1.11, 2.22, -3.22, 0.33, -0.221, 3.123, -0.477, 1.22, 3.221, 11.22,
            22.12, 33.11
        ], [
            -2.003, 1.221, 0.321, 0.447, 0.321, 0.446, -0.33192, 0.476, 0.8976,
            -4.123, 0.487, -0.4473
        ]],
        lattice_sizes=[2, 3, 2],
        expected_value=2868.62393167,
        l2_reg=1.0)

  def testL1AndL2Regularizers(self):
    """Check l1 and l2 regularization amount."""
    self._runAndCheckValues(
        lattice_param=[[
            1.11, 2.22, -3.22, 0.33, -0.221, 3.123, -0.477, 1.22, 3.221, 11.22,
            22.12, 33.11
        ]],
        lattice_sizes=[2, 3, 2],
        expected_value=2931.168801,
        l1_reg=1.0,
        l2_reg=1.0)
    self._runAndCheckValues(
        lattice_param=[[
            1.11, 2.22, -3.22, 0.33, -0.221, 3.123, -0.477, 1.22, 3.221, 11.22,
            22.12, 33.11
        ]],
        lattice_sizes=[2, 3, 2],
        expected_value=279.159846,
        l1_reg=[0.0, 1.0, 0.0],
        l2_reg=[1.0, 0.0, 0.0])

  def testNoRegularizerExpectsNone(self):
    """Set no l1_reg and l2_reg and check None regularizer."""
    lattice_param = array_ops.placeholder(dtype=dtypes.float32, shape=[2, 4])
    lattice_sizes = [2, 2]
    self.assertEqual(
        None, regularizers.lattice_regularization(lattice_param, lattice_sizes))

  def testRank1TensorExpectsError(self):
    """Pass rank-1 lattice_param tensor and check the error."""
    lattice_param = array_ops.placeholder(dtype=dtypes.float32, shape=[10])
    lattice_sizes = [2, 5]
    with self.assertRaises(ValueError):
      regularizers.lattice_regularization(
          lattice_param, lattice_sizes, l1_laplacian_reg=1.0)

  def testUnknownShapeTensorExpectsError(self):
    """Pass rank-2 tensor with unknown shape and check the error."""
    lattice_param = array_ops.placeholder(
        dtype=dtypes.float32, shape=[None, None])
    lattice_sizes = [2, 2]
    with self.assertRaises(ValueError):
      regularizers.lattice_regularization(
          lattice_param, lattice_sizes, l1_laplacian_reg=1.0)

  def testWrongL1regularizationsExpectsError(self):
    # 2 x 2 lattice
    lattice_param = array_ops.placeholder(dtype=dtypes.float32, shape=[2, 4])
    lattice_sizes = [2, 2]
    # Set 3 l1_regularizations for 2d lattice.
    l1_reg = [0.0, 1.0, 0.0]
    with self.assertRaises(ValueError):
      regularizers.lattice_regularization(
          lattice_param, lattice_sizes, l1_laplacian_reg=l1_reg)

  def testWrongL2regularizationsExpectsError(self):
    # 2 x 2 lattice
    lattice_param = array_ops.placeholder(dtype=dtypes.float32, shape=[4, 2])
    lattice_sizes = [2, 2]
    # Set 3 l2_regularizations for 2d lattice.
    l2_reg = [0.0, 1.0, 0.0]
    with self.assertRaises(ValueError):
      regularizers.lattice_regularization(
          lattice_param, lattice_sizes, l2_laplacian_reg=l2_reg)


class LatticeTorsionTestCase(test.TestCase):

  def _runAndCheckValues(self,
                         lattice_param,
                         lattice_sizes,
                         expected_value,
                         l1_reg=None,
                         l2_reg=None):
    lattice_param_tensor = array_ops.constant(
        lattice_param, dtype=dtypes.float32)
    reg = regularizers.lattice_regularization(
        lattice_param_tensor,
        lattice_sizes,
        l1_torsion_reg=l1_reg,
        l2_torsion_reg=l2_reg)
    with self.test_session() as sess:
      reg_value = sess.run(reg)
    self.assertAlmostEqual(reg_value, expected_value, delta=1e-1)

  def testZeroRegularizerValueForVariousLatticeRanks(self):
    """Check zero output value for zero parameters."""
    for lattice_rank in range(2, 10):
      param_dim = 2**lattice_rank
      lattice_param = [[0.0] * param_dim]
      self._runAndCheckValues(
          lattice_param=lattice_param,
          lattice_sizes=[2] * lattice_rank,
          expected_value=0.0,
          l1_reg=1.0,
          l2_reg=1.0)

  def testL1RegularizerWithTwoByTwo(self):
    """Check l1 regularization amount for two by two lattices.

    In 2 x 2 lattice, L1 torsion regualrizer has the following form:
      l1_reg * abs(param[0] + param[3] - param[1] - param[2]),
    where param is the lattice parameter tensor (assuming one output),
    """
    self._runAndCheckValues(
        lattice_param=[[0.0, 0.0, 1.0, 1.0]],
        lattice_sizes=[2, 2],
        expected_value=0.0,
        l1_reg=1.0)
    self._runAndCheckValues(
        lattice_param=[[0.0, 1.0, 1.0, 1.0]],
        lattice_sizes=[2, 2],
        expected_value=1.0,
        l1_reg=1.0)
    self._runAndCheckValues(
        lattice_param=[[0.0, 0.0, 0.0, 1.0]],
        lattice_sizes=[2, 2],
        expected_value=1.0,
        l1_reg=1.0)
    self._runAndCheckValues(
        lattice_param=[[0.3312, -0.3217, -0.5, 0.1]],
        lattice_sizes=[2, 2],
        expected_value=1.2529,
        l1_reg=1.0)

  def testL1RegularizerWithTwoByThreeByTwo(self):
    """Check l1 regularization amount for two by three by two lattices.

    In 2 x 3 x 2 lattice, L1 Torsion regualrizer has the following form:
      l1_reg * (abs(param[0] + param[3] - param[1] - param[2])
                + abs(param[2] + param[5] - param[3] - param[4])
                + abs(param[6] + param[9] - param[7] - param[8])
                + abs(param[8] + param[11] - param[9] - param[10])
                + abs(param[0] + param[7] - param[1] - param[6])
                + abs(param[2] + param[9] - param[3] - param[8])
                + abs(param[4] + param[11] - param[5] - param[10])
                + abs(param[0] + param[8] - param[2] - param[6])
                + abs(param[2] + param[10] - param[4] - param[8])
                + abs(param[1] + param[9] - param[3] - param[7])
                + abs(param[3] + param[11] - param[5] - param[9]))

    where param is the lattice parameter tensor (assuming one output),
    """
    self._runAndCheckValues(
        lattice_param=[[
            1.11, 2.22, -3.22, 0.33, -0.221, 3.123, -0.477, 1.22, 3.221, 11.22,
            22.12, 33.11
        ]],
        lattice_sizes=[2, 3, 2],
        expected_value=79.536,
        l1_reg=1.0)
    self._runAndCheckValues(
        lattice_param=[[
            -2.003, 1.221, 0.321, 0.447, 0.321, 0.446, -0.33192, 0.476, 0.8976,
            -4.123, 0.487, -0.4473
        ]],
        lattice_sizes=[2, 3, 2],
        expected_value=30.642580000000002,
        l1_reg=1.0)
    self._runAndCheckValues(
        lattice_param=[[
            1.11, 2.22, -3.22, 0.33, -0.221, 3.123, -0.477, 1.22, 3.221, 11.22,
            22.12, 33.11
        ], [
            -2.003, 1.221, 0.321, 0.447, 0.321, 0.446, -0.33192, 0.476, 0.8976,
            -4.123, 0.487, -0.4473
        ]],
        lattice_sizes=[2, 3, 2],
        expected_value=110.17858000000001,
        l1_reg=1.0)

  def testL2RegularizerWithTwoByTwo(self):
    """Check l2 regularization amount for two by two lattices.

    In 2 x 2 lattice, L2 torsion regualrizer has the following form:
      l2_reg * (param[0] + param[3] - param[1] - param[2]) ** 2,
    where param is the lattice parameter tensor (assuming one output),
    """
    self._runAndCheckValues(
        lattice_param=[[0.0, 0.0, 1.0, 1.0]],
        lattice_sizes=[2, 2],
        expected_value=0.0,
        l2_reg=1.0)
    self._runAndCheckValues(
        lattice_param=[[0.0, 1.0, 1.0, 1.0]],
        lattice_sizes=[2, 2],
        expected_value=1.0,
        l2_reg=1.0)
    self._runAndCheckValues(
        lattice_param=[[0.0, 0.0, 0.0, 1.0]],
        lattice_sizes=[2, 2],
        expected_value=1.0,
        l2_reg=1.0)
    self._runAndCheckValues(
        lattice_param=[[0.3312, -0.3217, -0.5, 0.1]],
        lattice_sizes=[2, 2],
        expected_value=1.5697584099999997,
        l2_reg=1.0)

  def testL2RegularizerWithTwoByThreeByTwo(self):
    """Check l2 regularization amount for two by three by two lattices.

    Replacing abs to square in the formula in
    testL1RegulairzerWithTwoByThreeByTwo gives L2 torsion.
    """
    self._runAndCheckValues(
        lattice_param=[[
            1.11, 2.22, -3.22, 0.33, -0.221, 3.123, -0.477, 1.22, 3.221, 11.22,
            22.12, 33.11
        ]],
        lattice_sizes=[2, 3, 2],
        expected_value=956.5830999999998,
        l2_reg=1.0)
    self._runAndCheckValues(
        lattice_param=[[
            -2.003, 1.221, 0.321, 0.447, 0.321, 0.446, -0.33192, 0.476, 0.8976,
            -4.123, 0.487, -0.4473
        ]],
        lattice_sizes=[2, 3, 2],
        expected_value=123.2293754172,
        l2_reg=1.0)
    self._runAndCheckValues(
        lattice_param=[[
            1.11, 2.22, -3.22, 0.33, -0.221, 3.123, -0.477, 1.22, 3.221, 11.22,
            22.12, 33.11
        ], [
            -2.003, 1.221, 0.321, 0.447, 0.321, 0.446, -0.33192, 0.476, 0.8976,
            -4.123, 0.487, -0.4473
        ]],
        lattice_sizes=[2, 3, 2],
        expected_value=1079.8124754172,
        l2_reg=1.0)

  def testL1andL2Regularizer(self):
    """Check l1 and l2 regularization amount for two by two lattices."""
    self._runAndCheckValues(
        lattice_param=[[0.0, 0.0, 1.0, 1.0]],
        lattice_sizes=[2, 2],
        expected_value=0.0,
        l1_reg=1.0,
        l2_reg=1.0)
    self._runAndCheckValues(
        lattice_param=[[0.0, 1.0, 1.0, 1.0]],
        lattice_sizes=[2, 2],
        expected_value=2.0,
        l1_reg=1.0,
        l2_reg=1.0)
    self._runAndCheckValues(
        lattice_param=[[0.0, 0.0, 0.0, 1.0]],
        lattice_sizes=[2, 2],
        expected_value=2.0,
        l1_reg=1.0,
        l2_reg=1.0)
    self._runAndCheckValues(
        lattice_param=[[0.3312, -0.3217, -0.5, 0.1]],
        lattice_sizes=[2, 2],
        expected_value=2.82265841,
        l1_reg=1.0,
        l2_reg=1.0)

  def testRank1TensorExpectsError(self):
    """Pass rank-1 tensor and check the error."""
    lattice_param = array_ops.placeholder(dtype=dtypes.float32, shape=[10])
    lattice_sizes = [2, 5]
    with self.assertRaises(ValueError):
      regularizers.lattice_regularization(
          lattice_param, lattice_sizes, l1_torsion_reg=1.0)

  def testUnknownShapeTensorExpectsError(self):
    """Pass rank-2 tensor with unknown shape and check the error."""
    lattice_param = array_ops.placeholder(
        dtype=dtypes.float32, shape=[None, None])
    lattice_sizes = [2, 2]
    with self.assertRaises(ValueError):
      regularizers.lattice_regularization(
          lattice_param, lattice_sizes, l1_torsion_reg=1.0)


class LatticeRegularizersTestCase(test.TestCase):

  def setUp(self):
    super(LatticeRegularizersTestCase, self).setUp()
    self._lattice_param = [[0.3312, -0.3217, -0.5, 0.1]]
    self._lattice_sizes = [2, 2]
    # Regularzation amounts for weight = 1.0
    self._l1_laplacian_reg = 2.5058
    self._l2_laplacian_reg = 1.65500274
    self._l1_reg = 1.2529
    self._l2_reg = 0.47318433
    self._l1_torsion_reg = 1.2529
    self._l2_torsion_reg = 1.5697584099999997

  def _runAndCheckValues(self,
                         lattice_param,
                         lattice_sizes,
                         expected_value,
                         l1_reg=None,
                         l2_reg=None,
                         l1_laplacian_reg=None,
                         l2_laplacian_reg=None,
                         l1_torsion_reg=None,
                         l2_torsion_reg=None):
    lattice_param_tensor = array_ops.constant(
        lattice_param, dtype=dtypes.float32)
    reg = regularizers.lattice_regularization(
        lattice_param_tensor,
        lattice_sizes,
        l1_reg=l1_reg,
        l2_reg=l2_reg,
        l1_laplacian_reg=l1_laplacian_reg,
        l2_laplacian_reg=l2_laplacian_reg,
        l1_torsion_reg=l1_torsion_reg,
        l2_torsion_reg=l2_torsion_reg)
    with self.test_session() as sess:
      reg_value = sess.run(reg)
    self.assertAlmostEqual(reg_value, expected_value, delta=1e-1)

  def testL1Regularizer(self):
    """Check l1 regularization amount."""
    self._runAndCheckValues(
        self._lattice_param,
        self._lattice_sizes,
        expected_value=self._l1_reg,
        l1_reg=1.0)

  def testL2Regularizer(self):
    """Check l2 regularization amount."""
    self._runAndCheckValues(
        self._lattice_param,
        self._lattice_sizes,
        expected_value=self._l2_reg,
        l2_reg=1.0)

  def testL1AndL2Regularizers(self):
    """Check l1 and l2 regularization amount."""
    expected_value = 0.5 * self._l1_reg + 0.5 * self._l2_reg
    self._runAndCheckValues(
        self._lattice_param,
        self._lattice_sizes,
        expected_value=expected_value,
        l1_reg=0.5,
        l2_reg=0.5)

  def testAllRegularizers(self):
    """Check l1, l2 and laplacian regularization amount."""
    expected_value = 0.5 * self._l1_reg
    expected_value += 0.5 * self._l2_reg
    expected_value += 0.5 * self._l1_laplacian_reg
    expected_value += 0.5 * self._l2_laplacian_reg
    expected_value += 0.5 * self._l1_torsion_reg
    expected_value += 0.5 * self._l2_torsion_reg
    self._runAndCheckValues(
        self._lattice_param,
        self._lattice_sizes,
        expected_value=expected_value,
        l1_reg=0.5,
        l2_reg=0.5,
        l1_laplacian_reg=0.5,
        l2_laplacian_reg=0.5,
        l1_torsion_reg=0.5,
        l2_torsion_reg=0.5)


class LinearRegularizersTestCase(test.TestCase):

  def setUp(self):
    super(LinearRegularizersTestCase, self).setUp()
    self._linear_param = [[0.3312, -0.3217, -0.5, 0.1]]
    # Regularzation amounts for weight = 1.0
    self._l1_reg = 1.2529
    self._l2_reg = 0.47318433

  def _runAndCheckValues(self,
                         linear_param,
                         expected_value,
                         l1_reg=None,
                         l2_reg=None):
    linear_param_tensor = array_ops.constant(linear_param, dtype=dtypes.float32)
    reg = regularizers.linear_regularization(
        linear_param_tensor, l1_reg=l1_reg, l2_reg=l2_reg)
    with self.test_session() as sess:
      reg_value = sess.run(reg)
    self.assertAlmostEqual(reg_value, expected_value, delta=1e-1)

  def testL1Regularizer(self):
    """Check l1 regularization amount."""
    self._runAndCheckValues(
        self._linear_param, expected_value=self._l1_reg, l1_reg=1.0)

  def testL2Regularizer(self):
    """Check l2 regularization amount."""
    self._runAndCheckValues(
        self._linear_param, expected_value=self._l2_reg, l2_reg=1.0)

  def testL1AndL2Regularizers(self):
    """Check l1 and l2 regularization amount."""
    expected_value = 0.5 * self._l1_reg + 0.5 * self._l2_reg
    self._runAndCheckValues(
        self._linear_param,
        expected_value=expected_value,
        l1_reg=0.5,
        l2_reg=0.5)


if __name__ == '__main__':
  test.main()
