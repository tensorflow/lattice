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
"""CalibratedLattice provide canned estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow_lattice.python.estimators import calibrated_lattice
from tensorflow_lattice.python.estimators import hparams as tfl_hparams
from tensorflow_lattice.python.lib import keypoints_initialization
from tensorflow_lattice.python.lib import test_data

_NUM_KEYPOINTS = 50


class CalibratedLatticeHParamsTest(tf.test.TestCase):

  def setUp(self):
    super(CalibratedLatticeHParamsTest, self).setUp()
    self.empty_estimator = calibrated_lattice.calibrated_lattice_classifier()
    self.hparams = tfl_hparams.CalibratedLatticeHParams(feature_names=['x'])
    self.hparams.set_param('lattice_size', 2)
    self.hparams.set_param('calibrator_output_min', 0)
    self.hparams.set_param('calibrator_output_max', 1)
    self.hparams.set_param('calibration_bound', True)

  def testWrongLatticeSize(self):
    self.hparams.set_feature_param('x', 'lattice_size', -1)
    self.assertRaisesRegexp(
        ValueError,
        'Hyperparameter configuration cannot be used in the calibrated lattice '
        'estimator.', self.empty_estimator.check_hparams, self.hparams)

  def testWrongCalibrationOutputMin(self):
    self.hparams.set_param('calibration_output_min', 0.0)
    self.hparams.set_feature_param('x', 'calibration_output_min', -1.0)
    self.assertRaisesRegexp(
        ValueError,
        'calibration_output_min=-1 should not be set, it is adjusted '
        'automatically to match the lattice_size',
        self.empty_estimator.check_hparams, self.hparams)

  def testWrongCalibrationOutputMax(self):
    self.hparams.set_param('calibration_output_max', 0.0)
    self.hparams.set_feature_param('x', 'calibration_output_max', 10)
    self.assertRaisesRegexp(
        ValueError,
        'calibration_output_max=10 should not be set, it is adjusted '
        'automatically to match the lattice_size',
        self.empty_estimator.check_hparams, self.hparams)

  def testWrongCalibrationBound(self):
    self.hparams.set_feature_param('x', 'calibration_bound', False)
    self.assertRaisesRegexp(
        ValueError,
        'Hyperparameter configuration cannot be used in the calibrated lattice '
        'estimator.', self.empty_estimator.check_hparams, self.hparams)

  def testWrongLatticeRegularization(self):
    self.hparams.set_feature_param('x', 'lattice_l1_reg', 0.1)
    self.hparams.set_feature_param('x', 'lattice_l2_reg', 0.1)
    self.hparams.set_feature_param('x', 'lattice_l1_torsion_reg', 0.1)
    self.hparams.set_feature_param('x', 'lattice_l1_torsion_reg', 0.1)
    self.assertRaisesRegexp(
        ValueError,
        'Hyperparameter configuration cannot be used in the calibrated lattice '
        'estimator.', self.empty_estimator.check_hparams, self.hparams)


class CalibratedLatticeTest(tf.test.TestCase):

  def setUp(self):
    super(CalibratedLatticeTest, self).setUp()
    self._test_data = test_data.TestData()

  def _CalibratedLatticeRegressor(self,
                                  feature_names,
                                  feature_columns,
                                  weight_column=None,
                                  **hparams_args):

    def init_fn():
      return keypoints_initialization.uniform_keypoints_for_signal(
          _NUM_KEYPOINTS, -1., 1., 0., 1.)

    hparams = tfl_hparams.CalibratedLatticeHParams(
        feature_names, num_keypoints=_NUM_KEYPOINTS, **hparams_args)
    # Turn off monotonic calibrator.
    hparams.set_param('calibration_monotonic', None)
    hparams.set_param('learning_rate', 0.1)
    return calibrated_lattice.calibrated_lattice_regressor(
        feature_columns=feature_columns,
        hparams=hparams,
        weight_column=weight_column,
        keypoints_initializers_fn=init_fn)

  def _CalibratedLatticeClassifier(self, feature_columns, **hparams_args):

    def init_fn():
      return keypoints_initialization.uniform_keypoints_for_signal(
          _NUM_KEYPOINTS, -1., 1., 0., 1.)

    hparams = tfl_hparams.CalibratedLatticeHParams(
        num_keypoints=_NUM_KEYPOINTS, **hparams_args)
    # Turn off monotonic calibrator.
    hparams.set_param('calibration_monotonic', None)
    hparams.set_param('learning_rate', 0.1)

    return calibrated_lattice.calibrated_lattice_classifier(
        feature_columns=feature_columns,
        hparams=hparams,
        keypoints_initializers_fn=init_fn)

  def testCalibratedLatticeRegressorTraining1D(self):
    feature_columns = [
        tf.feature_column.numeric_column('x'),
    ]
    estimator = self._CalibratedLatticeRegressor(['x'], feature_columns)
    estimator.train(input_fn=self._test_data.oned_input_fn())
    results = estimator.evaluate(input_fn=self._test_data.oned_input_fn())
    self.assertLess(results['average_loss'], 1e-3)

  def testCalibratedLatticeRegressorWeightedTraining1D(self):
    feature_columns = [
        tf.feature_column.numeric_column('x'),
    ]
    weight_column = tf.feature_column.numeric_column('zero')
    estimator = self._CalibratedLatticeRegressor(['x'],
                                                 feature_columns,
                                                 weight_column=weight_column)
    estimator.train(input_fn=self._test_data.oned_zero_weight_input_fn())
    results = estimator.evaluate(
        input_fn=self._test_data.oned_zero_weight_input_fn())
    self.assertLess(results['average_loss'], 1e-7)

  def testCalibratedLatticeRegressorTraining2D(self):
    feature_columns = [
        tf.feature_column.numeric_column('x0'),
        tf.feature_column.numeric_column('x1'),
    ]
    estimator = self._CalibratedLatticeRegressor(['x0', 'x1'], feature_columns)
    estimator.train(input_fn=self._test_data.twod_input_fn())
    results = estimator.evaluate(input_fn=self._test_data.twod_input_fn())
    self.assertLess(results['average_loss'], 5e-3)

  def testCalibratedLatticeRegressorTraining2DWithCalibrationRegularizer(self):
    feature_columns = [
        tf.feature_column.numeric_column('x0'),
        tf.feature_column.numeric_column('x1'),
    ]
    estimator = self._CalibratedLatticeRegressor(
        ['x0', 'x1'],
        feature_columns,
        interpolation_type='simplex',
        calibration_l1_reg=1.0,
        calibration_l2_reg=1.0,
        calibration_l1_laplacian_reg=1.0,
        calibration_l2_laplacian_reg=1.0)
    estimator.train(input_fn=self._test_data.twod_input_fn())
    results = estimator.evaluate(input_fn=self._test_data.twod_input_fn())
    # We expect the loss is larger than the loss without regularization.
    self.assertGreater(results['average_loss'], 1e-2)
    self.assertLess(results['average_loss'], 0.1)

  def testCalibratedLatticeRegressorTraining2DWithLatticeRegularizer(self):
    feature_columns = [
        tf.feature_column.numeric_column('x0'),
        tf.feature_column.numeric_column('x1'),
    ]
    estimator = self._CalibratedLatticeRegressor(['x0', 'x1'],
                                                 feature_columns,
                                                 interpolation_type='simplex',
                                                 lattice_l1_reg=1.0,
                                                 lattice_l2_reg=1.0,
                                                 lattice_l1_torsion_reg=1.0,
                                                 lattice_l2_torsion_reg=1.0,
                                                 lattice_l1_laplacian_reg=1.0,
                                                 lattice_l2_laplacian_reg=1.0)
    estimator.train(input_fn=self._test_data.twod_input_fn())
    results = estimator.evaluate(input_fn=self._test_data.twod_input_fn())
    # We expect the loss is larger than the loss without regularization.
    self.assertGreater(results['average_loss'], 1e-2)
    self.assertLess(results['average_loss'], 0.5)

  def testCalibratedLatticeRegressorTraining2DWithPerFeatureRegularizer(self):
    feature_columns = [
        tf.feature_column.numeric_column('x0'),
        tf.feature_column.numeric_column('x1'),
    ]
    estimator = self._CalibratedLatticeRegressor(
        ['x0', 'x1'],
        feature_columns,
        feature__x0__lattice_l1_laplacian_reg=100.0,
        feature__x1__lattice_l2_laplacian_reg=100.0)
    estimator.train(input_fn=self._test_data.twod_input_fn())
    results = estimator.evaluate(input_fn=self._test_data.twod_input_fn())
    # We expect the loss is larger than the loss without regularization.
    self.assertGreater(results['average_loss'], 0.1)
    self.assertLess(results['average_loss'], 0.2)

  def testCalibratedLatticeRegressorTrainingMultiDimensionalFeature(self):
    feature_columns = [
        tf.feature_column.numeric_column('x', shape=(2,)),
    ]
    estimator = self._CalibratedLatticeRegressor(['x'],
                                                 feature_columns,
                                                 interpolation_type='hypercube')
    estimator.train(input_fn=self._test_data.multid_feature_input_fn())
    results = estimator.evaluate(
        input_fn=self._test_data.multid_feature_input_fn())
    self.assertLess(results['average_loss'], 1e-3)

    # Turn-off calibration for feature 'x', it should turn if off for both
    # dimensions, and the results should get much worse.
    estimator = self._CalibratedLatticeRegressor(['x'],
                                                 feature_columns,
                                                 feature__x__num_keypoints=0)
    estimator.train(input_fn=self._test_data.multid_feature_input_fn())
    results = estimator.evaluate(
        input_fn=self._test_data.multid_feature_input_fn())
    self.assertGreater(results['average_loss'], 1e-2)

  def testCalibratedLatticeClassifierTraining(self):
    feature_columns = [
        tf.feature_column.numeric_column('x0'),
        tf.feature_column.numeric_column('x1'),
    ]
    estimator = self._CalibratedLatticeClassifier(feature_columns)
    estimator.train(input_fn=self._test_data.twod_classificer_input_fn())
    results = estimator.evaluate(
        input_fn=self._test_data.twod_classificer_input_fn())
    self.assertGreater(results['auc'], 0.990)

  def testCalibratedLatticeClassifierTrainingWithCalibrationRegularizer(self):
    feature_columns = [
        tf.feature_column.numeric_column('x0'),
        tf.feature_column.numeric_column('x1'),
    ]
    estimator = self._CalibratedLatticeClassifier(
        feature_columns,
        interpolation_type='hypercube',
        calibration_l1_reg=0.3,
        calibration_l2_reg=0.3,
        calibration_l1_laplacian_reg=1.0,
        calibration_l2_laplacian_reg=1.0)
    estimator.train(input_fn=self._test_data.twod_classificer_input_fn())
    results = estimator.evaluate(
        input_fn=self._test_data.twod_classificer_input_fn())
    # We expect AUC is worse than the model without regularization.
    self.assertLess(results['auc'], 0.98)

  def testCalibratedLatticeClassifierTrainingWithLatticeRegularizer(self):
    feature_columns = [
        tf.feature_column.numeric_column('x0'),
        tf.feature_column.numeric_column('x1'),
    ]
    estimator = self._CalibratedLatticeClassifier(
        feature_columns,
        interpolation_type='simplex',
        lattice_l1_reg=5.0,
        lattice_l2_reg=5.0,
        lattice_l1_torsion_reg=5.0,
        lattice_l2_torsion_reg=5.0,
        lattice_l1_laplacian_reg=5.0,
        lattice_l2_laplacian_reg=5.0)
    estimator.train(input_fn=self._test_data.twod_classificer_input_fn())
    results = estimator.evaluate(
        input_fn=self._test_data.twod_classificer_input_fn())
    # We expect AUC is worse than the model without regularization.
    self.assertLess(results['auc'], 0.98)
    self.assertGreater(results['auc'], 0.68)

  def testCalibratedLatticeClassifierTrainingWithPerFeatureRegularizer(self):
    feature_columns = [
        tf.feature_column.numeric_column('x0'),
        tf.feature_column.numeric_column('x1'),
    ]
    estimator = self._CalibratedLatticeClassifier(
        feature_columns,
        feature_names=['x0', 'x1'],
        feature__x0__lattice_l1_laplacian_reg=50.0,
        feature__x1__lattice_l2_laplacian_reg=50.0)
    estimator.train(input_fn=self._test_data.twod_classificer_input_fn())
    results = estimator.evaluate(
        input_fn=self._test_data.twod_classificer_input_fn())
    # We expect AUC is worse than the model without regularization.
    self.assertLess(results['auc'], 0.98)
    self.assertGreater(results['auc'], 0.8)

  def testCalibratedLatticeMonotonicClassifierTraining(self):
    # Construct the following training/testing pair.
    #
    # Training: (x, y)
    # ([0., 0.], 0.0)
    # ([0., 1.], 1.0)
    # ([1., 0.], 1.0)
    # ([1., 1.], 0.0)
    #
    # Test: (x, y)
    # ([0., 0.], 0.0)
    # ([0., 1.], 1.0)
    # ([1., 0.], 1.0)
    # ([1., 1.], 1.0)
    #
    # Note that training example has a noisy sample, ([1., 1.], 0.0), and test
    # examples are generated by the logical-OR function. Therefore by enforcing
    # increasing monotonicity to all features, we should be able to work well
    # in the test examples.
    x0 = np.array([0.0, 0.0, 1.0, 1.0])
    x1 = np.array([0.0, 1.0, 0.0, 1.0])
    x_samples = {'x0': x0, 'x1': x1}
    training_y = np.array([[False], [True], [True], [False]])
    test_y = np.array([[False], [True], [True], [True]])

    train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x=x_samples, y=training_y, batch_size=4, num_epochs=1000, shuffle=False)
    test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x=x_samples, y=test_y, shuffle=False)

    # Define monotonic lattice classifier.
    feature_columns = [
        tf.feature_column.numeric_column('x0'),
        tf.feature_column.numeric_column('x1'),
    ]

    def init_fn():
      return keypoints_initialization.uniform_keypoints_for_signal(
          2, 0., 1., 0., 1.)

    hparams = tfl_hparams.CalibratedLatticeHParams(num_keypoints=2)
    # Monotonic calibrated lattice.

    hparams.set_param('monotonicity', +1)
    hparams.set_param('learning_rate', 0.1)
    hparams.set_param('interpolation_type', 'hypercube')

    estimator = calibrated_lattice.calibrated_lattice_classifier(
        feature_columns=feature_columns,
        hparams=hparams,
        keypoints_initializers_fn=init_fn)

    estimator.train(input_fn=train_input_fn)
    results = estimator.evaluate(input_fn=test_input_fn)
    # We should expect 1.0 accuracy.
    self.assertGreater(results['accuracy'], 0.999)

  def testCalibratedLatticeWithMissingTraining(self):
    # x0 is missing with it's own vertex: so it can take very different values,
    # while x1 is missing and calibrated, in this case to the middle of the
    # lattice.
    x0 = np.array([0., 0., 1., 1., -1., -1., 0., 1.])
    x1 = np.array([0., 1., 0., 1., 0., 1., -1., -1.])
    training_y = np.array([1., 3., 7., 11., 23., 27., 2., 9.])
    x_samples = {'x0': x0, 'x1': x1}

    train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x=x_samples,
        y=training_y,
        batch_size=x0.shape[0],
        num_epochs=2000,
        shuffle=False)
    test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x=x_samples, y=training_y, shuffle=False)
    feature_columns = [
        tf.feature_column.numeric_column('x0'),
        tf.feature_column.numeric_column('x1'),
    ]

    def init_fn():
      return keypoints_initialization.uniform_keypoints_for_signal(
          2, 0., 1., 0., 1.)

    hparams = tfl_hparams.CalibratedLatticeHParams(['x0', 'x1'],
                                                   num_keypoints=2,
                                                   learning_rate=0.1,
                                                   missing_input_value=-1.)
    hparams.set_feature_param('x0', 'missing_vertex', True)

    estimator = calibrated_lattice.calibrated_lattice_regressor(
        feature_columns=feature_columns,
        hparams=hparams,
        keypoints_initializers_fn=init_fn)

    estimator.train(input_fn=train_input_fn)
    results = estimator.evaluate(input_fn=test_input_fn)
    self.assertLess(results['average_loss'], 0.1)


if __name__ == '__main__':
  tf.test.main()
