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
"""Tests for TensorFlow Lattice's monotone_linear_layers module."""
# Dependency imports

from tensorflow_lattice.python.lib import monotone_linear_layers

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class MonotoneLinearTestCase(test.TestCase):

  def setUp(self):
    super(MonotoneLinearTestCase, self).setUp()

  def testEvaluationWithZeroBias(self):
    """Create a partial monotone linear layer and check evaluation."""
    input_placeholder = array_ops.placeholder(
        dtype=dtypes.float32, shape=[None, 3])
    input_tensor = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    sum_input_tensor = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
    # Check linearity of the output tensor.
    # f(input_tensor + input_tensor) = 2 * f(input_tensor)
    # since the bias is 0.
    packed_results = monotone_linear_layers.monotone_linear_layer(
        input_placeholder, input_dim=3, output_dim=5, init_bias=0.0)
    (output_tensor, _, _, _) = packed_results
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      # Check linearity of the output tensor.
      # f(input_tensor + input_tensor) = 2 * f(input_tensor)
      # since the bias is 0.
      output_val = sess.run(
          output_tensor, feed_dict={input_placeholder: input_tensor})
      sum_output_val = sess.run(
          output_tensor, feed_dict={input_placeholder: sum_input_tensor})
      expected_sum_output_val = 2 * output_val
    self.assertAllClose(expected_sum_output_val, sum_output_val)

  def testEvaluationWithDefaultBias(self):
    """Create a partial monotone linear layer and check the bias."""
    input_dim = 10
    input_placeholder = array_ops.placeholder(
        dtype=dtypes.float32, shape=[None, input_dim])
    # Monotone linear layers contain random weights and for this input_tensor
    # we expect 0 as an output on "average". In order to control randomness, we
    # set the standard deviation exactly zero.
    input_tensor = [[0.5] * input_dim]
    expected_output_val = [[0.0]]
    packed_results = monotone_linear_layers.monotone_linear_layer(
        input_placeholder,
        input_dim=input_dim,
        output_dim=1,
        init_weight_stddev=0.0)
    (output_tensor, _, _, _) = packed_results
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      # Check linearity of the output tensor.
      # f(input_tensor + input_tensor) = 2 * f(input_tensor)
      # since the bias is 0.
      output_val = sess.run(
          output_tensor, feed_dict={input_placeholder: input_tensor})
    self.assertAllClose(expected_output_val, output_val)

  def testProjection(self):
    """Create a partial monotone linear layer and check the projection."""
    input_dim = 10
    is_monotone = [True, False] * 5
    input_placeholder = array_ops.placeholder(
        dtype=dtypes.float32, shape=[None, input_dim])
    # We set the initial_weight_mean to -10.0. After projection, we expect
    # elements corresponding to monotonic input becomes 0.
    packed_results = monotone_linear_layers.monotone_linear_layer(
        input_placeholder,
        input_dim=input_dim,
        output_dim=2,
        is_monotone=is_monotone,
        init_weight_mean=-10.0,
        init_weight_stddev=0.0)
    (_, weight_tensor, projection_op, _) = packed_results
    # The weight is in shape (output_dim, input_dim).
    expected_pre_projection_weight = [[-10.0] * 10] * 2
    expected_projected_weight = [[0.0, -10.0] * 5] * 2
    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      pre_projection_weight = sess.run(weight_tensor)
      sess.run(projection_op)
      projected_weight = sess.run(weight_tensor)
    self.assertAllClose(expected_pre_projection_weight, pre_projection_weight)
    self.assertAllClose(expected_projected_weight, projected_weight)


class SplitMonotoneLinearTestCase(test.TestCase):

  def setUp(self):
    super(SplitMonotoneLinearTestCase, self).setUp()

  def testEvaluation(self):
    """Create a split monotone linear layer and check the results."""
    batch_size = 5
    input_dim = 10
    monotonic_output_dim = 2
    non_monotonic_output_dim = 3
    # First five is monotonic, and the last five is non-monotonic.
    is_monotone = [True] * 5 + [False] * 5
    input_placeholder = array_ops.placeholder(
        dtype=dtypes.float32, shape=[batch_size, input_dim])
    packed_results = monotone_linear_layers.split_monotone_linear_layer(
        input_placeholder,
        input_dim=input_dim,
        monotonic_output_dim=monotonic_output_dim,
        non_monotonic_output_dim=non_monotonic_output_dim,
        is_monotone=is_monotone)
    (monotonic_output, _, non_monotonic_output, _, _, _) = packed_results

    # Check the shape of outputs.
    self.assertAllEqual(monotonic_output.shape,
                        [batch_size, monotonic_output_dim])
    self.assertAllEqual(non_monotonic_output.shape,
                        [batch_size, non_monotonic_output_dim])

    # Check monotonic inputs are not part of non_monotonic_output.
    # We do this by changing the first half of inputs and check whether it
    # changes the value or not.
    zero_input = [[0.0] * 10] * 5
    identity_in_monotone_inputs = [
        [1.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 5,
        [0.0, 1.0, 0.0, 0.0, 0.0] + [0.0] * 5,
        [0.0, 0.0, 1.0, 0.0, 0.0] + [0.0] * 5,
        [0.0, 0.0, 0.0, 1.0, 0.0] + [0.0] * 5,
        [0.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 5,
    ]

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      non_monotonic_output_at_zero = sess.run(
          non_monotonic_output, feed_dict={input_placeholder: zero_input})
      non_monotonic_output_at_identity = sess.run(
          non_monotonic_output,
          feed_dict={input_placeholder: identity_in_monotone_inputs})

    self.assertAllClose(non_monotonic_output_at_zero,
                        non_monotonic_output_at_identity)

  def testProjection(self):
    """Check projection operator."""
    input_dim = 2
    monotonic_output_dim = 2
    non_monotonic_output_dim = 1
    # First five is monotonic, and the last five is non-monotonic.
    is_monotone = [True, False]
    input_placeholder = array_ops.placeholder(
        dtype=dtypes.float32, shape=[None, input_dim])
    packed_results = monotone_linear_layers.split_monotone_linear_layer(
        input_placeholder,
        input_dim=input_dim,
        monotonic_output_dim=monotonic_output_dim,
        non_monotonic_output_dim=non_monotonic_output_dim,
        is_monotone=is_monotone,
        init_weight_mean=-10.0,
        init_weight_stddev=0.0)
    (_, monotone_weights, _, non_monotone_weights, proj, _) = packed_results

    expected_pre_monotone_weights = [[-10.0, -10.0]] * 2
    expected_pre_non_monotone_weights = [[-10.0]]
    expected_projected_monotone_weights = [[0.0, -10.0]] * 2
    expected_projected_non_monotone_weights = [[-10.0]]

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())
      self.assertAllClose(expected_pre_monotone_weights,
                          monotone_weights.eval())
      self.assertAllClose(expected_pre_non_monotone_weights,
                          non_monotone_weights.eval())
      sess.run(proj)
      self.assertAllClose(expected_projected_monotone_weights,
                          monotone_weights.eval())
      self.assertAllClose(expected_projected_non_monotone_weights,
                          non_monotone_weights.eval())

  def testBooleanIsMonotoneExpectsError(self):
    """Test empty non monotonic output."""
    input_dim = 2
    monotonic_output_dim = 2
    non_monotonic_output_dim = 1
    is_monotone = True
    input_placeholder = array_ops.placeholder(
        dtype=dtypes.float32, shape=[None, input_dim])
    with self.assertRaises(ValueError):
      _ = monotone_linear_layers.split_monotone_linear_layer(
          input_placeholder,
          input_dim=input_dim,
          monotonic_output_dim=monotonic_output_dim,
          non_monotonic_output_dim=non_monotonic_output_dim,
          is_monotone=is_monotone,
          init_weight_mean=-10.0,
          init_weight_stddev=0.0)

  def testZeroNonMonotonicOutputExpectEmptyNonMonotonicOutput(self):
    """Test empty non monotonic output."""
    input_dim = 2
    monotonic_output_dim = 2
    non_monotonic_output_dim = 0
    is_monotone = [True, True]
    input_placeholder = array_ops.placeholder(
        dtype=dtypes.float32, shape=[None, input_dim])
    packed_results = monotone_linear_layers.split_monotone_linear_layer(
        input_placeholder,
        input_dim=input_dim,
        monotonic_output_dim=monotonic_output_dim,
        non_monotonic_output_dim=non_monotonic_output_dim,
        is_monotone=is_monotone,
        init_weight_mean=-10.0,
        init_weight_stddev=0.0)
    (_, _, non_monotonic_outputs, non_monotonic_weights, _, _) = packed_results
    self.assertEqual(non_monotonic_outputs, None)
    self.assertEqual(non_monotonic_weights, None)

  def testNoNonMonotonicInputsWithNonMonotonicOutputExpectFailure(self):
    input_dim = 2
    monotonic_output_dim = 2
    non_monotonic_output_dim = 2
    is_monotone = [True, True]
    input_placeholder = array_ops.placeholder(
        dtype=dtypes.float32, shape=[None, input_dim])
    with self.assertRaises(ValueError):
      _ = monotone_linear_layers.split_monotone_linear_layer(
          input_placeholder,
          input_dim=input_dim,
          monotonic_output_dim=monotonic_output_dim,
          non_monotonic_output_dim=non_monotonic_output_dim,
          is_monotone=is_monotone,
          init_weight_mean=-10.0,
          init_weight_stddev=0.0)


if __name__ == '__main__':
  test.main()
