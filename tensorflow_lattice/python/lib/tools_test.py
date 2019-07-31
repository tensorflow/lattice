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
"""Tests for TensorFlow Lattice's tools module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_lattice.python.lib import test_data
from tensorflow_lattice.python.lib import tools

_NUM_EXAMPLES = 10


class ToolsTestCase(tf.test.TestCase):

  def setUp(self):
    super(ToolsTestCase, self).setUp()
    self._test_data = test_data.TestData(num_examples=_NUM_EXAMPLES)

  def testCastToDict(self):
    names = ['a', 'b', 'c']
    got = tools.cast_to_dict(1.0, names, 'blah')
    self.assertEqual(got['a'], 1.0)
    self.assertEqual(got['b'], 1.0)
    self.assertEqual(got['c'], 1.0)
    self.assertItemsEqual(got.keys(), names)

    got = tools.cast_to_dict({'a': 1.0, 'b': 2.0, 'c': 3.0}, names, 'blah')
    self.assertEqual(got['a'], 1.0)
    self.assertEqual(got['b'], 2.0)
    self.assertEqual(got['c'], 3.0)
    self.assertItemsEqual(got.keys(), names)

    with self.assertRaisesRegexp(
        ValueError,
        'Dict given for blah does not contain definition for feature "c"'):
      got = tools.cast_to_dict({'a': 1.0, 'b': 2.0}, names, 'blah')

    got = tools.cast_to_dict({'a': 1.0, tools.DEFAULT_NAME: 2.0}, names, 'blah')
    self.assertItemsEqual(got.keys(), names)
    self.assertEqual(got['a'], 1.0)
    self.assertEqual(got['b'], 2.0)
    self.assertEqual(got['c'], 2.0)

  def testCastToDictOfTensorScalars(self):
    # Same value for all names.
    names = ['a', 'b', 'c']
    value = np.array(1.0)
    got = tools.cast_to_dict_of_tensor_scalars(value, ['a', 'b', 'c'],
                                               tf.float32, 't1')
    self.assertItemsEqual(got.keys(), names)
    self.assertEqual(got['a'], got['b'])
    self.assertEqual(got['b'], got['c'])
    self.assertIsInstance(got['a'], tf.Tensor)
    self.assertShapeEqual(value, got['a'])

    # Raises for missing names.
    with self.assertRaisesRegexp(
        ValueError,
        'Dict given for t2 does not contain definition for feature "c"'):
      got = tools.cast_to_dict_of_tensor_scalars({
          'a': value,
          'b': value
      }, ['a', 'b', 'c'], tf.float32, 't2')

    # Uses default value
    default_value = np.array(2.0)
    got = tools.cast_to_dict_of_tensor_scalars(
        {
            'a': value,
            'b': value,
            tools.DEFAULT_NAME: default_value,
        }, names, tf.float32, 't2')
    self.assertItemsEqual(got.keys(), names)

  def testInputFromFeatureColumn(self):
    # Tests 1-dimension real valued feature.
    x = np.random.uniform(-1.0, 1.0, size=[self._test_data.num_examples])
    feature_column = tf.feature_column.numeric_column('x')
    # Notice that 1-dimension features [batch_size] are packaged into a 2-dim
    # tensor: [batch_size, 1]
    materialized = self._materialize_feature_column(feature_column, x)
    self.assertEqual(materialized.shape, (self._test_data.num_examples, 1))
    materialized = materialized[:, 0]
    self.assertTrue(
        self._np_array_close(x, materialized),
        'expected:{} != got:{}'.format(x, materialized))

    # Tests that 2-dimensional real valued feature.
    x = np.random.uniform(-1.0, 1.0, size=[self._test_data.num_examples, 2])
    feature_column = tf.feature_column.numeric_column('x', shape=(2,))
    materialized = self._materialize_feature_column(feature_column, x)
    self.assertTrue(
        self._np_array_close(x, materialized),
        'expected:{} != got:{}'.format(x, materialized))

    # Tests that categorical feature is correctly converted.
    x = np.array(['Y', 'N', '?', 'Y', 'Y', 'N'])
    expect = np.array([0., 1., -1., 0., 0., 1.])
    feature_column = tf.feature_column.categorical_column_with_vocabulary_list(
        'x', ['Y', 'N'])
    materialized = self._materialize_feature_column(feature_column, x)[:, 0]
    self.assertTrue(
        self._np_array_close(expect, materialized),
        'expect:{} != got:{}'.format(expect, materialized))

  def testSaveOnceOrWaitForChief(self):
    write_fn = tf.compat.v1.test.mock.Mock()
    tools.save_once_or_wait_for_chief(
        write_fn, self.get_temp_dir(), is_chief=True)
    write_fn.assert_called_once_with()
    write_fn.reset_mock()
    write_fn.assert_not_called()
    tools.save_once_or_wait_for_chief(
        write_fn, self.get_temp_dir(), is_chief=True)
    write_fn.assert_not_called()
    tools.save_once_or_wait_for_chief(
        write_fn, self.get_temp_dir(), is_chief=False)
    write_fn.assert_not_called()

  @tf.compat.v1.test.mock.patch('time.time')
  def testSaveOnceOrWaitForChief_Timeout(self, mock_time):
    write_fn = tf.compat.v1.test.mock.Mock()
    # Return 0 on the first call to 'time.time' and 1000 on the second.
    mock_time.side_effect = [0, 1000]
    self.assertRaises(
        tools.SaveOnceOrWaitTimeOutError,
        tools.save_once_or_wait_for_chief,
        write_fn,
        self.get_temp_dir(),
        is_chief=False,
        timeout_secs=999)
    call = tf.compat.v1.test.mock.call
    self.assertEqual(mock_time.mock_calls, [call(), call()])

  def _np_array_close(self, a, b):
    return np.alltrue(np.isclose(a, b))

  def _materialize_feature_column(self, feature_column, x):
    """Creates input_fn with x then transform and materialize feature_column."""
    input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'x': x},
        y=None,
        batch_size=self._test_data.num_examples,
        num_epochs=1,
        shuffle=False)
    with tf.Graph().as_default():
      features = input_fn()
      input_tensor = tools.input_from_feature_column(features, feature_column)
      materialized = self._materialize_locally(input_tensor)
    return materialized

  def _materialize_locally(self, tensors, feed_dict=None):
    with tf.compat.v1.train.SingularMonitoredSession() as sess:
      materialized = sess.run(tensors, feed_dict=feed_dict)
    return materialized


class LatticeToolsTestCase(tf.test.TestCase):

  def _runIterAndCheck(self, lattice_sizes, expected_vertices):
    # Running iterator, and check the returned vertices with expected_vertices.
    lattice_structure = tools.LatticeStructure(lattice_sizes)
    for (index, vertices) in tools.lattice_indices_generator(lattice_structure):
      self.assertItemsEqual(vertices, expected_vertices[index])

  def testTwoByThreeLatticeIteration(self):
    lattice_sizes = [2, 3]
    expected_vertices = [[0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2]]
    self._runIterAndCheck(lattice_sizes, expected_vertices)

  def testThreeByTwoByTwoIteration(self):
    lattice_sizes = [3, 2, 2]
    expected_vertices = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0],
                         [2, 1, 0], [0, 0, 1], [1, 0, 1], [2, 0, 1], [0, 1, 1],
                         [1, 1, 1], [2, 1, 1]]
    self._runIterAndCheck(lattice_sizes, expected_vertices)

  def testWrongLatticeSizeExpectsError(self):
    with self.assertRaises(ValueError):
      _ = tools.LatticeStructure([1, 1])


class Lattice1DSliceTestCase(tf.test.TestCase):

  def _runAndCheckValues(self, slice_lattice_param_tensor, expected_value):
    with self.session() as sess:
      slice_lattice_param_value = sess.run(slice_lattice_param_tensor)
    self.assertAllClose(slice_lattice_param_value, expected_value)

  def testTwodLatticeSlice(self):
    lattice_sizes = [2, 3]
    # param[0][0] = 0
    # param[1][0] = 1
    # param[0][1] = 2
    # param[1][1] = 3
    # param[0][2] = 4
    # param[1][2] = 5
    lattice_param_tensor = tf.constant([list(range(2 * 3))])
    # param[0][:] = [0, 2, 4]
    param_0_x = tools.lattice_1d_slice(
        lattice_param_tensor, lattice_sizes, lattice_axis=0, begin=0, size=1)
    self._runAndCheckValues(param_0_x, expected_value=[[0, 2, 4]])
    # param[1][:] = [1, 3, 5]
    param_1_x = tools.lattice_1d_slice(
        lattice_param_tensor, lattice_sizes, lattice_axis=0, begin=1, size=1)
    self._runAndCheckValues(param_1_x, expected_value=[[1, 3, 5]])
    # param[:][0] = [0, 1]
    param_x_0 = tools.lattice_1d_slice(
        lattice_param_tensor, lattice_sizes, lattice_axis=1, begin=0, size=1)
    self._runAndCheckValues(param_x_0, expected_value=[[0, 1]])
    # param[:][1] = [2, 3]
    param_x_1 = tools.lattice_1d_slice(
        lattice_param_tensor, lattice_sizes, lattice_axis=1, begin=1, size=1)
    self._runAndCheckValues(param_x_1, expected_value=[[2, 3]])
    # param[:][2] = [4, 5]
    param_x_2 = tools.lattice_1d_slice(
        lattice_param_tensor, lattice_sizes, lattice_axis=1, begin=2, size=1)
    self._runAndCheckValues(param_x_2, expected_value=[[4, 5]])
    # param[:][0:1] = [0, 1, 2, 3]
    param_x_01 = tools.lattice_1d_slice(
        lattice_param_tensor, lattice_sizes, lattice_axis=1, begin=0, size=2)
    self._runAndCheckValues(param_x_01, expected_value=[[0, 1, 2, 3]])
    # param[:][1:2] = [2, 3, 4, 5]
    param_x_12 = tools.lattice_1d_slice(
        lattice_param_tensor, lattice_sizes, lattice_axis=1, begin=1, size=2)
    self._runAndCheckValues(param_x_12, expected_value=[[2, 3, 4, 5]])

  def testTwodMultiOutputLatticeSlice(self):
    lattice_sizes = [2, 2]
    # first_param[0][0] = 0
    # first_param[1][0] = 1
    # first_param[0][1] = 2
    # first_param[1][1] = 3
    # second_param[0][0] = 3
    # second_param[1][0] = 2
    # second_param[0][1] = 1
    # second_param[1][1] = 0
    lattice_param_tensor = tf.constant(
        [list(range(2 * 2)), list(range(2 * 2 - 1, -1, -1))])
    # param[0][:] = [[0, 2], [3, 1]]
    param_0_x = tools.lattice_1d_slice(
        lattice_param_tensor, lattice_sizes, lattice_axis=0, begin=0, size=1)
    self._runAndCheckValues(param_0_x, expected_value=[[0, 2], [3, 1]])
    # param[1][:] = [[1, 3], [2, 0]]
    param_1_x = tools.lattice_1d_slice(
        lattice_param_tensor, lattice_sizes, lattice_axis=0, begin=1, size=1)
    self._runAndCheckValues(param_1_x, expected_value=[[1, 3], [2, 0]])
    # param[:][0] = [[0, 1], [3, 2]]
    param_x_0 = tools.lattice_1d_slice(
        lattice_param_tensor, lattice_sizes, lattice_axis=1, begin=0, size=1)
    self._runAndCheckValues(param_x_0, expected_value=[[0, 1], [3, 2]])
    # param[:][1] = [[2, 3], [1, 0]]
    param_x_1 = tools.lattice_1d_slice(
        lattice_param_tensor, lattice_sizes, lattice_axis=1, begin=1, size=1)
    self._runAndCheckValues(param_x_1, expected_value=[[2, 3], [1, 0]])

  def testThreedLatticeSlice(self):
    lattice_sizes = [2, 3, 2]
    # param[0][0][0] = 0
    # param[1][0][0] = 1
    # param[0][1][0] = 2
    # param[1][1][0] = 3
    # param[0][2][0] = 4
    # param[1][2][0] = 5
    # param[0][0][1] = 6
    # param[1][0][1] = 7
    # param[0][1][1] = 8
    # param[1][1][1] = 9
    # param[0][2][1] = 10
    # param[1][2][1] = 11
    lattice_param_tensor = tf.constant([list(range(2 * 3 * 2))])
    # param[0][:][:] = [0, 2, 4, 6, 8, 10]
    param_0_x_x = tools.lattice_1d_slice(
        lattice_param_tensor, lattice_sizes, lattice_axis=0, begin=0, size=1)
    self._runAndCheckValues(param_0_x_x, expected_value=[[0, 2, 4, 6, 8, 10]])
    # param[1][:][:] = [1, 3, 5, 7, 9, 11]
    param_1_x_x = tools.lattice_1d_slice(
        lattice_param_tensor, lattice_sizes, lattice_axis=0, begin=1, size=1)
    self._runAndCheckValues(param_1_x_x, expected_value=[[1, 3, 5, 7, 9, 11]])
    # param[:][0][:] = [0, 1, 6, 7]
    param_x_0_x = tools.lattice_1d_slice(
        lattice_param_tensor, lattice_sizes, lattice_axis=1, begin=0, size=1)
    self._runAndCheckValues(param_x_0_x, expected_value=[[0, 1, 6, 7]])
    # param[:][1][:] = [2, 3, 8, 9]
    param_x_1_x = tools.lattice_1d_slice(
        lattice_param_tensor, lattice_sizes, lattice_axis=1, begin=1, size=1)
    self._runAndCheckValues(param_x_1_x, expected_value=[[2, 3, 8, 9]])
    # param[:][2][:] = [4, 5, 10, 11]
    param_x_2_x = tools.lattice_1d_slice(
        lattice_param_tensor, lattice_sizes, lattice_axis=1, begin=2, size=1)
    self._runAndCheckValues(param_x_2_x, expected_value=[[4, 5, 10, 11]])
    # param[:][0:1][:] = [0, 1, 2, 3, 6, 7, 8, 9]
    param_x_01_x = tools.lattice_1d_slice(
        lattice_param_tensor, lattice_sizes, lattice_axis=1, begin=0, size=2)
    self._runAndCheckValues(
        param_x_01_x, expected_value=[[0, 1, 2, 3, 6, 7, 8, 9]])
    # param[:][1:2][:] = [2, 3, 4, 5, 8, 9, 10, 11]
    param_x_12_x = tools.lattice_1d_slice(
        lattice_param_tensor, lattice_sizes, lattice_axis=1, begin=1, size=2)
    self._runAndCheckValues(
        param_x_12_x, expected_value=[[2, 3, 4, 5, 8, 9, 10, 11]])
    # param[:][:][0] = [0, 1, 2, 3, 4, 5]
    param_x_x_0 = tools.lattice_1d_slice(
        lattice_param_tensor, lattice_sizes, lattice_axis=2, begin=0, size=1)
    self._runAndCheckValues(param_x_x_0, expected_value=[[0, 1, 2, 3, 4, 5]])
    # param[:][:][1] = [6, 7, 8, 9, 10, 11]
    param_x_x_1 = tools.lattice_1d_slice(
        lattice_param_tensor, lattice_sizes, lattice_axis=2, begin=1, size=1)
    self._runAndCheckValues(param_x_x_1, expected_value=[[6, 7, 8, 9, 10, 11]])

  def testWrongTensorShapeExpectsError(self):
    lattice_param_tensor = tf.compat.v1.placeholder(shape=(2, 2, 2), dtype=tf.float32)
    with self.assertRaises(ValueError):
      _ = tools.lattice_1d_slice(
          lattice_param_tensor,
          lattice_sizes=[2],
          lattice_axis=0,
          begin=0,
          size=1)

  def testOutOfRangeAxisExpectsError(self):
    lattice_param_tensor = tf.compat.v1.placeholder(shape=(2, 4), dtype=tf.float32)
    with self.assertRaises(ValueError):
      _ = tools.lattice_1d_slice(
          lattice_param_tensor,
          lattice_sizes=[2, 2],
          lattice_axis=3,
          begin=0,
          size=1)

  def testBeginSizeOutOfRangeExpectsError(self):
    lattice_param_tensor = tf.compat.v1.placeholder(shape=(2, 4), dtype=tf.float32)
    with self.assertRaises(ValueError):
      _ = tools.lattice_1d_slice(
          lattice_param_tensor,
          lattice_sizes=[2, 2],
          lattice_axis=0,
          begin=1,
          size=2)


if __name__ == '__main__':
  tf.test.main()
