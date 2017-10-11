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
"""Tests for TensorFlow Lattice's pwl_calibration_layers module."""
# Dependency imports

from tensorflow_lattice.python.lib import keypoints_initialization
from tensorflow_lattice.python.lib import pwl_calibration_layers
from tensorflow_lattice.python.lib import tools

from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


_DEFAULT_OUTPUT_MIN = 200
_DEFAULT_OUTPUT_MAX = 300


def _get_variable_by_name(name):
  return ops.get_default_graph().get_tensor_by_name(name)


class PwlCalibratorLayersTestCase(test.TestCase):

  def setUp(self):
    super(PwlCalibratorLayersTestCase, self).setUp()

  def _BuildInputs(self, x0, x1):
    """Returns input_fn, feature_names and feature_columns."""

    def _input_fn():
      return {
          'x0': array_ops.constant(x0, dtype=dtypes.float32),
          'x1': array_ops.constant(x1, dtype=dtypes.float32),
      }

    feature_names = ['x0', 'x1']
    x0_dim = 1 if not isinstance(x0[0], list) else len(x0[0])
    x1_dim = 1 if not isinstance(x1[0], list) else len(x1[0])
    feature_columns = {
        feature_column_lib.numeric_column(key='x0', shape=(x0_dim,)),
        feature_column_lib.numeric_column(key='x1', shape=(x1_dim,)),
    }
    return _input_fn, feature_names, feature_columns

  def _CheckOneDimensionalCalibrationLayer(self, sess, uncalibrated, calibrated,
                                           value, want):
    got = sess.run(calibrated, feed_dict={uncalibrated: value})
    self.assertAllClose(got, want)

  def _UniformKeypoints(self,
                        num_keypoints,
                        output_min=_DEFAULT_OUTPUT_MIN,
                        output_max=_DEFAULT_OUTPUT_MAX):
    return keypoints_initialization.uniform_keypoints_for_signal(
        num_keypoints=num_keypoints,
        input_min=array_ops.constant(0.0, dtype=dtypes.float32),
        input_max=array_ops.constant(1.0, dtype=dtypes.float32),
        output_min=output_min,
        output_max=output_max,
        dtype=dtypes.float32)

  def testOneDimensionalCalibrationLayer(self):
    with ops.Graph().as_default():
      num_keypoints = 10
      keypoints_init = self._UniformKeypoints(num_keypoints)
      uncalibrated = array_ops.placeholder(
          shape=tensor_shape.unknown_shape(ndims=1), dtype=dtypes.float32)
      calibrated, projection, regularization = (
          pwl_calibration_layers.one_dimensional_calibration_layer(
              uncalibrated,
              num_keypoints=num_keypoints,
              signal_name='test_one_dimensional_calibration_layer',
              keypoints_initializers=keypoints_init))
      self.assertEqual(projection, None)
      self.assertEqual(regularization, None)
      with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        self._CheckOneDimensionalCalibrationLayer(sess, uncalibrated,
                                                  calibrated, [0.5], [250.])
        self._CheckOneDimensionalCalibrationLayer(
            sess, uncalibrated, calibrated, [0.2, 0.7], [220., 270.])

  def testOneDimensionalCalibrationLambda(self):
    with ops.Graph().as_default():
      num_keypoints = 10
      def kp_in_fn(*args, **kwargs):
        return math_ops.linspace(0., 1., num_keypoints)
      def kp_out_fn(*args, **kwargs):
        return math_ops.linspace(float(_DEFAULT_OUTPUT_MIN),
                                 float(_DEFAULT_OUTPUT_MAX),
                                 num_keypoints)
      uncalibrated = array_ops.placeholder(
          shape=tensor_shape.unknown_shape(ndims=1), dtype=dtypes.float32)
      calibrated, _, regularization = (
          pwl_calibration_layers.one_dimensional_calibration_layer(
              uncalibrated,
              missing_input_value=0.21,
              num_keypoints=num_keypoints,
              bound=True,
              signal_name='test_one_dimensional_calibration_layer_lambda',
              keypoints_initializer_fns=(kp_in_fn, kp_out_fn)))
      self.assertEqual(regularization, None)
      with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        self._CheckOneDimensionalCalibrationLayer(sess, uncalibrated,
                                                  calibrated, [0.5], [250.])
        self._CheckOneDimensionalCalibrationLayer(
            sess, uncalibrated, calibrated, [0.2, 0.7], [220., 270.])

  def testOneDimensionalCalibrationRaises(self):
    with ops.Graph().as_default():
      num_keypoints = 10
      def kp_in_fn(*args, **kwargs):
        return math_ops.linspace(0., 1., num_keypoints)
      def kp_out_fn(*args, **kwargs):
        return math_ops.linspace(float(_DEFAULT_OUTPUT_MIN),
                                 float(_DEFAULT_OUTPUT_MAX),
                                 num_keypoints)
      keypoints_init = self._UniformKeypoints(num_keypoints)
      uncalibrated = array_ops.placeholder(
          shape=tensor_shape.unknown_shape(ndims=1), dtype=dtypes.float32)
      self.assertRaises(
          ValueError,
          pwl_calibration_layers.one_dimensional_calibration_layer,
          uncalibrated,
          num_keypoints=num_keypoints,
          signal_name='test_one_dimensional_calibration_layer',
          keypoints_initializers=keypoints_init,
          keypoints_initializer_fns=(kp_in_fn, kp_out_fn))

  def testOneDimensionalCalibrationLayerRegularization(self):
    with ops.Graph().as_default():
      num_keypoints = 10
      keypoints_init = self._UniformKeypoints(num_keypoints)
      uncalibrated = array_ops.placeholder(
          shape=tensor_shape.unknown_shape(ndims=1), dtype=dtypes.float32)
      _, _, regularization = (
          pwl_calibration_layers.one_dimensional_calibration_layer(
              uncalibrated,
              num_keypoints=num_keypoints,
              signal_name='test_one_dimensional_calibration_layer',
              l1_reg=1.0,
              l2_reg=1.0,
              l1_laplacian_reg=1.0,
              l2_laplacian_reg=1.0,
              keypoints_initializers=keypoints_init))
      with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        got = sess.run(regularization)
        expected_value = 638896.25
        self.assertAlmostEqual(got, expected_value, delta=1e-1)

  def testInputCalibrationLayer(self):
    x0 = [[0.1], [0.2], [0.3], [0.3], [-1.]]
    x1 = [[0.9], [0.8], [0.7], [-1.], [0.7]]
    input_fn, feature_names, feature_columns = self._BuildInputs(x0, x1)
    num_keypoints = 10

    # Test calibration of two features.
    with ops.Graph().as_default():
      keypoints_init = self._UniformKeypoints(num_keypoints)
      columns_to_tensors = input_fn()
      calibrated, feature_names, projection_ops, regularization = (
          pwl_calibration_layers.input_calibration_layer(
              columns_to_tensors=columns_to_tensors,
              feature_columns=feature_columns,
              num_keypoints=num_keypoints,
              keypoints_initializers=keypoints_init,
              missing_input_values=-1.,
              missing_output_values=7.))
      self.assertEqual(feature_names, ['x0', 'x1'])
      self.assertEqual(projection_ops, [])
      self.assertEqual(regularization, None)
      got = keypoints_initialization._materialize_locally(
          calibrated, num_steps=1)
      self.assertAllClose(got, [[210., 290.], [220., 280.], [230., 270.],
                                [230., 7.], [7., 270.]])

  def testInputCalibrationLayerNonCalibrated(self):
    x0 = [[0.1], [0.2], [0.3], [0.3], [-1.]]
    x1 = [[0.9], [0.8], [0.7], [-1.], [0.7]]
    input_fn, feature_names, feature_columns = self._BuildInputs(x0, x1)
    num_keypoints = 10

    # Test case where one feature is not calibrated.
    with ops.Graph().as_default():
      keypoints_init = self._UniformKeypoints(num_keypoints)
      columns_to_tensors = input_fn()

      calibrated, feature_names, projection_ops, regularization = (
          pwl_calibration_layers.input_calibration_layer(
              columns_to_tensors=columns_to_tensors,
              feature_columns=feature_columns,
              num_keypoints={'x0': num_keypoints,
                             'x1': 0},
              keypoints_initializers=keypoints_init,
              missing_input_values={'x0': -1.,
                                    tools.DEFAULT_NAME: None},
              missing_output_values={'x0': 7.,
                                     tools.DEFAULT_NAME: None}))
      self.assertEqual(projection_ops, [])
      self.assertEqual(feature_names, ['x0', 'x1'])
      self.assertEqual(regularization, None)
      got = keypoints_initialization._materialize_locally(
          calibrated, num_steps=1)
      self.assertAllClose(got, [[210., 0.9], [220., 0.8], [230., 0.7],
                                [230., -1.], [7., 0.7]])

  def testInputCalibrationLayerMultiDimensional(self):
    x0 = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]]
    x1 = [[0.9, 1.2], [0.8, 1.1], [0.7, 0.2]]
    input_fn, feature_names, feature_columns = self._BuildInputs(x0, x1)
    num_keypoints = 10

    # Test case where feature columns are multi-dimensional.
    with ops.Graph().as_default():
      keypoints_init = self._UniformKeypoints(num_keypoints)
      columns_to_tensors = input_fn()
      calibrated, feature_names, projection_ops, regularization = (
          pwl_calibration_layers.input_calibration_layer(
              columns_to_tensors=columns_to_tensors,
              feature_columns=feature_columns,
              num_keypoints={'x0': num_keypoints,
                             'x1': 0},
              keypoints_initializers=keypoints_init))
      self.assertEqual(projection_ops, [])
      self.assertEqual(feature_names, ['x0', 'x0', 'x1', 'x1'])
      self.assertEqual(regularization, None)
      got = keypoints_initialization._materialize_locally(
          calibrated, num_steps=1)
      self.assertAllClose(got, [[210., 290., 0.9, 1.2], [220., 280., 0.8, 1.1],
                                [230., 270., 0.7, 0.2]])

  def testInputCalibrationLayerRegularization(self):
    x0 = [0.1, 0.2, 0.7]
    x1 = [0.9, 0.8, 0.7]
    input_fn, _, feature_columns = self._BuildInputs(x0, x1)
    num_keypoints = 10

    with ops.Graph().as_default():
      keypoints_init = self._UniformKeypoints(num_keypoints)
      columns_to_tensors = input_fn()
      _, _, _, regularization = (pwl_calibration_layers.input_calibration_layer(
          columns_to_tensors=columns_to_tensors,
          feature_columns=feature_columns,
          num_keypoints={'x0': num_keypoints,
                         'x1': num_keypoints},
          l1_reg={'x0': 1.0,
                  'x1': 2.0},
          l2_reg={'x0': 0.5,
                  'x1': None},
          l1_laplacian_reg={'x0': None,
                            'x1': 3.0},
          l2_laplacian_reg={'x0': None,
                            'x1': 5.0},
          keypoints_initializers=keypoints_init))
      with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        got = sess.run(regularization)
        expected_value = 330948.12
        self.assertAlmostEqual(got, expected_value, delta=1e-1)

  def testCalibrationLayer(self):
    with ops.Graph().as_default():
      # Shape: [batch_size=2, 3, 2]
      uncalibrated = array_ops.constant([
          [[0.1, 0.2], [0.9, 0.8], [0.4, 0.6]],
          [[0.2, 0.3], [1.0, 0.9], [0.5, 0.7]],
      ])
      kp_init_0 = self._UniformKeypoints(10)
      kp_init_1 = self._UniformKeypoints(5, 0, 1000)
      num_keypoints = [10, 10, 10, 10, 5, 5]
      kp_init = [
          kp_init_0, kp_init_0, kp_init_0, kp_init_0, kp_init_1, kp_init_1
      ]
      calibrated, projection_ops, regularization = (
          pwl_calibration_layers.calibration_layer(
              uncalibrated, num_keypoints, keypoints_initializers=kp_init,
              name='test'))
      self.assertEqual(projection_ops, [])
      self.assertEqual(regularization, None)
      got = keypoints_initialization._materialize_locally(
          calibrated, num_steps=1)
      want = [
          [[210., 220.], [290., 280.], [400., 600.]],
          [[220., 230.], [300., 290.], [500., 700.]],
      ]
      self.assertAllClose(got, want)

  def testCalibrationLayerRegularization(self):
    with ops.Graph().as_default():
      # Shape: [batch_size=2, 3, 2]
      uncalibrated = array_ops.constant([
          [[0.1, 0.2], [0.9, 0.8], [0.4, 0.6]],
          [[0.2, 0.3], [1.0, 0.9], [0.5, 0.7]],
      ])
      kp_init_0 = self._UniformKeypoints(10)
      kp_init_1 = self._UniformKeypoints(5, 0, 1000)
      num_keypoints = [10, 10, 10, 10, 5, 5]
      kp_init = [
          kp_init_0, kp_init_0, kp_init_0, kp_init_0, kp_init_1, kp_init_1
      ]
      _, _, regularization = (pwl_calibration_layers.calibration_layer(
          uncalibrated,
          num_keypoints,
          keypoints_initializers=kp_init,
          l1_reg=0.1,
          l2_reg=1.0,
          l1_laplacian_reg=[0.3, 0.1, 0.2, 0.3, 0.4, 0.5],
          l2_laplacian_reg=[None, 1.0, None, None, None, None],
          name='test'))

      with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        got = sess.run(regularization)
        expected_value = 6294341.5
        self.assertAlmostEqual(got, expected_value, delta=1e-1)

  def testCalibrationLayerWithUnknownBatchSize(self):
    with ops.Graph().as_default():
      # Shape: [batch_size=2, 3, 2]
      uncalibrated = array_ops.placeholder(dtypes.float32, shape=[None, 3, 2])
      kp_init_0 = self._UniformKeypoints(10)
      kp_init_1 = self._UniformKeypoints(5, 0, 1000)
      num_keypoints = [10, 10, 10, 10, 5, 5]
      kp_init = [
          kp_init_0, kp_init_0, kp_init_0, kp_init_0, kp_init_1, kp_init_1
      ]
      calibrated, projection_ops, regularization = (
          pwl_calibration_layers.calibration_layer(
              uncalibrated, num_keypoints, keypoints_initializers=kp_init,
              name='test'))
      self.assertEqual(projection_ops, [])
      self.assertEqual(regularization, None)
      got = keypoints_initialization._materialize_locally(
          calibrated,
          num_steps=1,
          feed_dict={
              uncalibrated: [
                  [[0.1, 0.2], [0.9, 0.8], [0.4, 0.6]],
                  [[0.2, 0.3], [1.0, 0.9], [0.5, 0.7]],
              ]
          })
      want = [
          [[210., 220.], [290., 280.], [400., 600.]],
          [[220., 230.], [300., 290.], [500., 700.]],
      ]
      self.assertAllClose(got, want)

  def testBoundness(self):
    # Create a bound calibration, then set it outside the bounds and check
    # that it is projected back to the bounds.
    with ops.Graph().as_default():
      num_keypoints = 3
      keypoints_init = keypoints_initialization.uniform_keypoints_for_signal(
          num_keypoints=num_keypoints,
          input_min=array_ops.constant(0.0, dtype=dtypes.float32),
          input_max=array_ops.constant(1.0, dtype=dtypes.float32),
          output_min=0.,
          output_max=1.,
          dtype=dtypes.float32)
      uncalibrated = array_ops.placeholder(
          shape=tensor_shape.unknown_shape(ndims=1), dtype=dtypes.float32)
      with variable_scope.variable_scope('test_boundness'):
        _, projection, regularization = (
            pwl_calibration_layers.one_dimensional_calibration_layer(
                uncalibrated,
                num_keypoints=num_keypoints,
                bound=True,
                signal_name='bounded_x',
                keypoints_initializers=keypoints_init))
      self.assertTrue(projection is not None)
      self.assertEqual(regularization, None)

      with self.test_session() as sess:
        # First initialize keypoints (and all variables)
        sess.run(variables.global_variables_initializer())
        kp_out = _get_variable_by_name(
            'test_boundness/pwl_calibration/bounded_x_keypoints_outputs:0')
        kp_out_values = sess.run(kp_out)
        self.assertAllClose(kp_out_values, [0.0, 0.5, 1.0])

        # Assign values to variable beyond bounds.
        out_of_bounds = [-0.1, 1.2, 0.9]
        sess.run(
            state_ops.assign(kp_out,
                             array_ops.constant(
                                 out_of_bounds, dtype=dtypes.float32)))
        kp_out_values = sess.run(kp_out)
        self.assertAllClose(kp_out_values, out_of_bounds)

        # Execute projection.
        sess.run(projection)
        kp_out_values = sess.run(kp_out)
        self.assertAllClose(kp_out_values, [0.0, 1.0, 0.9])

  def testMonotonicity(self):
    # Create a monotonic calibration, then set it in a non-monotonic way and
    # check that it is projected back to monotonicity.
    with ops.Graph().as_default():
      num_keypoints = 5
      keypoints_init = keypoints_initialization.uniform_keypoints_for_signal(
          num_keypoints=num_keypoints,
          input_min=array_ops.constant(0.0, dtype=dtypes.float32),
          input_max=array_ops.constant(1.0, dtype=dtypes.float32),
          output_min=0.,
          output_max=1.,
          dtype=dtypes.float32)
      uncalibrated = array_ops.placeholder(
          shape=tensor_shape.unknown_shape(ndims=1), dtype=dtypes.float32)
      with variable_scope.variable_scope('test_monotonicity'):
        _, projection, regularization = (
            pwl_calibration_layers.one_dimensional_calibration_layer(
                uncalibrated,
                num_keypoints=num_keypoints,
                monotonic=1,
                signal_name='monotonic_x',
                keypoints_initializers=keypoints_init))
      self.assertTrue(projection is not None)
      self.assertEqual(regularization, None)

      with self.test_session() as sess:
        # First initialize keypoints (and all variables)
        sess.run(variables.global_variables_initializer())
        kp_out = _get_variable_by_name(
            'test_monotonicity/pwl_calibration/monotonic_x_keypoints_outputs:0')
        kp_out_values = sess.run(kp_out)
        self.assertAllClose(kp_out_values, [0.0, 0.25, 0.5, 0.75, 1.0])

        # Assign non_monotonic calibration.
        non_monotonic = [4., 5., 0., 4., -3.]
        sess.run(
            state_ops.assign(kp_out,
                             array_ops.constant(
                                 non_monotonic, dtype=dtypes.float32)))
        kp_out_values = sess.run(kp_out)
        self.assertAllClose(kp_out_values, non_monotonic)

        # Execute projection.
        sess.run(projection)
        kp_out_values = sess.run(kp_out)
        self.assertAllClose(kp_out_values, [2., 2., 2., 2., 2.])

  def testMissingFixedOutput(self):
    with ops.Graph().as_default():
      num_keypoints = 10
      keypoints_init = self._UniformKeypoints(num_keypoints)
      uncalibrated = array_ops.placeholder(
          shape=tensor_shape.unknown_shape(ndims=1), dtype=dtypes.float32)
      calibrated, projection, regularization = (
          pwl_calibration_layers.one_dimensional_calibration_layer(
              uncalibrated,
              num_keypoints=num_keypoints,
              signal_name='test_missing_fixed_output',
              keypoints_initializers=keypoints_init,
              bound=True,
              missing_input_value=-1.,
              missing_output_value=7.))
      self.assertNotEqual(projection, None)
      self.assertEqual(regularization, None)
      with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        # Mix of missing and calibrated:
        self._CheckOneDimensionalCalibrationLayer(
            sess, uncalibrated, calibrated, [0.5, -1.], [250., 7.])
        # Only calibrated:
        self._CheckOneDimensionalCalibrationLayer(
            sess, uncalibrated, calibrated, [0.2, 0.7], [220., 270.])
        # Only missing:
        self._CheckOneDimensionalCalibrationLayer(
            sess, uncalibrated, calibrated, [-1., -1.], [7., 7.])

        # Projection shouldn't affect the missing output value, even though
        # it is outside the bounds.
        sess.run([projection])
        self._CheckOneDimensionalCalibrationLayer(
            sess, uncalibrated, calibrated, [-1., -1.], [7., 7.])

  def testMissingCalibratedOutput(self):
    with ops.Graph().as_default():
      # With calibration:
      num_keypoints = 10
      keypoints_init = self._UniformKeypoints(num_keypoints)
      uncalibrated = array_ops.placeholder(
          shape=tensor_shape.unknown_shape(ndims=1), dtype=dtypes.float32)
      calibrated, projection, regularization = (
          pwl_calibration_layers.one_dimensional_calibration_layer(
              uncalibrated,
              num_keypoints=num_keypoints,
              signal_name='test_missing_calibrated_output',
              keypoints_initializers=keypoints_init,
              bound=True,
              missing_input_value=-1.))
      self.assertNotEqual(projection, None)
      self.assertEqual(regularization, None)
      with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        self._CheckOneDimensionalCalibrationLayer(sess, uncalibrated,
                                                  calibrated, [0.5, -1.],
                                                  [250., _DEFAULT_OUTPUT_MIN])

        # Set out-of-bound value for missing value.
        missing_calibrated_output = _get_variable_by_name(
            'pwl_calibration/'
            'test_missing_calibrated_output_calibrated_missing_output:0')
        sess.run([state_ops.assign(missing_calibrated_output, 700.0)])
        self._CheckOneDimensionalCalibrationLayer(sess, uncalibrated,
                                                  calibrated, [-1.], [700.])

        # Project to bound.
        sess.run(projection)
        self._CheckOneDimensionalCalibrationLayer(
            sess, uncalibrated, calibrated, [-1.], [_DEFAULT_OUTPUT_MAX])

        # Gradient wrt missing_calibrated_output should be 1.0
        d_calibrated_wrt_d_output = gradients.gradients(
            calibrated, missing_calibrated_output)
        got = sess.run(
            d_calibrated_wrt_d_output, feed_dict={uncalibrated: [-1.]})
        self.assertAllClose(got, [1.])


if __name__ == '__main__':
  test.main()
