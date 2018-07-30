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
"""CalibratedRtl provide canned estimators."""
# Dependency imports

import numpy as np

from tensorflow_lattice.python.estimators import hparams as tfl_hparams
from tensorflow_lattice.python.estimators import separately_calibrated_rtl as scrtl
from tensorflow_lattice.python.lib import keypoints_initialization
from tensorflow_lattice.python.lib import test_data

from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.platform import test

_NUM_KEYPOINTS = 50


class CalibratedRtlHParamsTest(test.TestCase):

  def setUp(self):
    self.hparams = tfl_hparams.CalibratedRtlHParams(feature_names=['x'])
    self.hparams.set_param('lattice_size', 2)
    self.hparams.set_param('calibrator_output_min', 0)
    self.hparams.set_param('calibrator_output_max', 1)
    self.hparams.set_param('calibration_bound', True)
    self.hparams.set_param('lattice_rank', 2)
    self.hparams.set_param('num_lattices', 10)
    self.empty_estimator = scrtl.separately_calibrated_rtl_classifier(
        hparams=self.hparams)

  def testWrongLatticeSize(self):
    self.hparams.set_feature_param('x', 'lattice_size', -1)
    self.assertRaisesRegexp(
        ValueError,
        'Hyperparameter configuration cannot be used in the calibrated rtl '
        'estimator.', self.empty_estimator.check_hparams, self.hparams)

  def testWrongCalibrationOutputMin(self):
    self.hparams.set_param('calibration_output_min', 0.0)
    self.hparams.set_feature_param('x', 'calibration_output_min', -1)
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
        'Hyperparameter configuration cannot be used in the calibrated rtl '
        'estimator.', self.empty_estimator.check_hparams, self.hparams)

  def testNoLatticeRank(self):
    self.hparams.set_param('lattice_rank', None)
    self.assertRaisesRegexp(
        ValueError,
        'Hyperparameter configuration cannot be used in the calibrated rtl '
        'estimator.', self.empty_estimator.check_hparams, self.hparams)

  def testNoNumLattices(self):
    self.hparams.set_param('num_lattices', None)
    self.assertRaisesRegexp(
        ValueError,
        'Hyperparameter configuration cannot be used in the calibrated rtl '
        'estimator.', self.empty_estimator.check_hparams, self.hparams)

  def testWrongLatticeRegularization(self):
    self.hparams.set_feature_param('x', 'lattice_l1_reg', 0.1)
    self.hparams.set_feature_param('x', 'lattice_l2_reg', 0.1)
    self.hparams.set_feature_param('x', 'lattice_l1_torsion_reg', 0.1)
    self.hparams.set_feature_param('x', 'lattice_l1_torsion_reg', 0.1)
    self.assertRaisesRegexp(
        ValueError,
        'Hyperparameter configuration cannot be used in the calibrated rtl '
        'estimator.', self.empty_estimator.check_hparams, self.hparams)


class CalibratedRtlTest(test.TestCase):

  def setUp(self):
    super(CalibratedRtlTest, self).setUp()
    self._test_data = test_data.TestData(num_epochs=10)

  def _CalibratedRtlRegressor(self,
                              feature_names,
                              feature_columns,
                              num_lattices=1,
                              lattice_rank=1,
                              num_keypoints=_NUM_KEYPOINTS,
                              weight_column=None,
                              **hparams_args):

    def init_fn():
      return keypoints_initialization.uniform_keypoints_for_signal(
          num_keypoints, -1., 1., 0., 1.)

    hparams = tfl_hparams.CalibratedRtlHParams(
        feature_names,
        num_keypoints=num_keypoints,
        num_lattices=num_lattices,
        lattice_rank=lattice_rank,
        **hparams_args)
    # Turn off monotonic calibrator.
    hparams.set_param('calibration_monotonic', None)
    hparams.set_param('learning_rate', 0.1)

    return scrtl.separately_calibrated_rtl_regressor(
        feature_columns=feature_columns,
        weight_column=weight_column,
        hparams=hparams,
        keypoints_initializers_fn=init_fn)

  def _CalibratedRtlClassifier(self,
                               feature_columns,
                               num_lattices=1,
                               lattice_rank=1,
                               **hparams_args):

    def init_fn():
      return keypoints_initialization.uniform_keypoints_for_signal(
          _NUM_KEYPOINTS, -1., 1., 0., 1.)

    hparams = tfl_hparams.CalibratedRtlHParams(
        num_keypoints=_NUM_KEYPOINTS,
        num_lattices=num_lattices,
        lattice_rank=lattice_rank,
        **hparams_args)
    # Turn off monotonic calibrator.
    hparams.set_param('calibration_monotonic', None)
    hparams.set_param('learning_rate', 0.1)

    return scrtl.separately_calibrated_rtl_classifier(
        feature_columns=feature_columns,
        hparams=hparams,
        keypoints_initializers_fn=init_fn)

  def testCalibratedRtlRegressorTraining1D(self):
    feature_columns = [
        feature_column_lib.numeric_column('x'),
    ]
    estimator = self._CalibratedRtlRegressor(
        ['x'], feature_columns, num_lattices=3, lattice_rank=1)
    estimator.train(input_fn=self._test_data.oned_input_fn())
    results = estimator.evaluate(input_fn=self._test_data.oned_input_fn())
    self.assertLess(results['average_loss'], 1e-2)

  def testSeparatelyCalibratedRtlRegressorWeightedTraining1D(self):
    feature_columns = [feature_column_lib.numeric_column('x')]
    weight_column = feature_column_lib.numeric_column('zero')
    estimator = self._CalibratedRtlRegressor(
        ['x'], feature_columns, num_lattices=2, weight_column=weight_column)
    estimator.train(input_fn=self._test_data.oned_zero_weight_input_fn())
    results = estimator.evaluate(
        input_fn=self._test_data.oned_zero_weight_input_fn())
    # Expects almost zero since the weight values are exactly zero.
    self.assertLess(results['average_loss'], 1e-7)

  def testCalibratedRtlRegressorTraining2D(self):
    feature_columns = [
        feature_column_lib.numeric_column('x0'),
        feature_column_lib.numeric_column('x1'),
    ]
    estimator = self._CalibratedRtlRegressor(
        ['x0', 'x1'], feature_columns, num_lattices=3, lattice_rank=2)
    estimator.train(input_fn=self._test_data.twod_input_fn())
    results = estimator.evaluate(input_fn=self._test_data.twod_input_fn())
    self.assertLess(results['average_loss'], 5e-3)

  def testCalibratedRtlRegressorTraining2DWithCalibrationRegularization(self):
    feature_columns = [
        feature_column_lib.numeric_column('x0'),
        feature_column_lib.numeric_column('x1'),
    ]
    estimator = self._CalibratedRtlRegressor(
        ['x0', 'x1'],
        feature_columns,
        num_lattices=3,
        lattice_rank=2,
        calibration_l1_reg=1e-2,
        calibration_l2_reg=1e-2,
        calibration_l1_laplacian_reg=0.05,
        calibration_l2_laplacian_reg=0.01)
    estimator.train(input_fn=self._test_data.twod_input_fn())
    results = estimator.evaluate(input_fn=self._test_data.twod_input_fn())
    self.assertLess(results['average_loss'], 0.1)

  def testCalibratedLatticeRegressorTraining2DWithLatticeRegularizer(self):
    feature_columns = [
        feature_column_lib.numeric_column('x0'),
        feature_column_lib.numeric_column('x1'),
    ]
    estimator = self._CalibratedRtlRegressor(
        ['x0', 'x1'],
        feature_columns,
        num_lattices=2,
        lattice_rank=2,
        lattice_l1_reg=1.0,
        lattice_l2_reg=1.0,
        lattice_l1_torsion_reg=1.0,
        lattice_l2_torsion_reg=1.0,
        lattice_l1_laplacian_reg=1.0,
        lattice_l2_laplacian_reg=0.1)
    estimator.train(input_fn=self._test_data.twod_input_fn())
    results = estimator.evaluate(input_fn=self._test_data.twod_input_fn())
    # We expect the loss is larger than the loss without regularization.
    self.assertGreater(results['average_loss'], 1e-2)
    self.assertLess(results['average_loss'], 0.5)

  def testCalibratedLatticeRegressorTraining2DWithPerFeatureRegularizer(self):
    feature_columns = [
        feature_column_lib.numeric_column('x0'),
        feature_column_lib.numeric_column('x1'),
    ]
    estimator = self._CalibratedRtlRegressor(
        ['x0', 'x1'],
        feature_columns,
        num_lattices=2,
        lattice_rank=2,
        feature__x0__lattice_l1_laplacian_reg=100.0,
        feature__x1__lattice_l2_laplacian_reg=1.0)
    estimator.train(input_fn=self._test_data.twod_input_fn())
    results = estimator.evaluate(input_fn=self._test_data.twod_input_fn())
    # We expect the loss is larger than the loss without regularization.
    self.assertGreater(results['average_loss'], 0.1)
    self.assertLess(results['average_loss'], 0.2)

  def testCalibratedRtlClassifierTraining(self):
    feature_columns = [
        feature_column_lib.numeric_column('x0'),
        feature_column_lib.numeric_column('x1'),
    ]
    estimator = self._CalibratedRtlClassifier(
        feature_columns, num_lattices=3, lattice_rank=2)
    estimator.train(input_fn=self._test_data.twod_classificer_input_fn())
    results = estimator.evaluate(
        input_fn=self._test_data.twod_classificer_input_fn())
    self.assertGreater(results['auc'], 0.990)

  def testCalibratedRtlClassifierTrainingWithCalibrationRegularizer(self):
    feature_columns = [
        feature_column_lib.numeric_column('x0'),
        feature_column_lib.numeric_column('x1'),
    ]
    estimator = self._CalibratedRtlClassifier(
        feature_columns,
        num_lattices=3,
        lattice_rank=2,
        interpolation_type='simplex',
        calibration_l1_reg=1e-5,
        calibration_l2_reg=1e-5,
        calibration_l1_laplacian_reg=1e-5,
        calibration_l2_laplacian_reg=1e-5)
    estimator.train(input_fn=self._test_data.twod_classificer_input_fn())
    results = estimator.evaluate(
        input_fn=self._test_data.twod_classificer_input_fn())
    self.assertGreater(results['auc'], 0.980)

  def testCalibratedRtlClassifierTrainingWithLatticeRegularizer(self):
    feature_columns = [
        feature_column_lib.numeric_column('x0'),
        feature_column_lib.numeric_column('x1'),
    ]
    estimator = self._CalibratedRtlClassifier(
        feature_columns,
        num_lattices=3,
        lattice_rank=2,
        interpolation_type='hypercube',
        lattice_l1_reg=1.0,
        lattice_l2_reg=1.0,
        lattice_l1_torsion_reg=1.0,
        lattice_l2_torsion_reg=1.0,
        lattice_l1_laplacian_reg=1.0,
        lattice_l2_laplacian_reg=1.0)
    estimator.train(input_fn=self._test_data.twod_classificer_input_fn())
    results = estimator.evaluate(
        input_fn=self._test_data.twod_classificer_input_fn())
    # We expect AUC is worse than the model without regularization.
    self.assertLess(results['auc'], 0.99)
    self.assertGreater(results['auc'], 0.4)

  def testCalibratedRtlClassifierTrainingWithPerFeatureRegularizer(self):
    feature_columns = [
        feature_column_lib.numeric_column('x0'),
        feature_column_lib.numeric_column('x1'),
    ]
    estimator = self._CalibratedRtlClassifier(
        feature_columns,
        feature_names=['x0', 'x1'],
        num_lattices=3,
        lattice_rank=2,
        feature__x0__lattice_l1_laplacian_reg=5.0,
        feature__x1__lattice_l2_laplacian_reg=0.5)
    estimator.train(input_fn=self._test_data.twod_classificer_input_fn())
    results = estimator.evaluate(
        input_fn=self._test_data.twod_classificer_input_fn())
    # We expect AUC is worse than the model without regularization.
    self.assertLess(results['auc'], 0.98)
    self.assertGreater(results['auc'], 0.7)

  def testCalibratedRtlMonotonicClassifierTraining(self):
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

    train_input_fn = numpy_io.numpy_input_fn(
        x=x_samples, y=training_y, batch_size=4, num_epochs=1000, shuffle=False)
    test_input_fn = numpy_io.numpy_input_fn(
        x=x_samples, y=test_y, shuffle=False)

    # Define monotonic lattice classifier.
    feature_columns = [
        feature_column_lib.numeric_column('x0'),
        feature_column_lib.numeric_column('x1'),
    ]

    def init_fn():
      return keypoints_initialization.uniform_keypoints_for_signal(
          2, 0., 1., 0., 1.)

    hparams = tfl_hparams.CalibratedRtlHParams(
        num_keypoints=2, num_lattices=3, lattice_rank=2)
    # Monotonic calibrated lattice.

    hparams.set_param('monotonicity', +1)
    hparams.set_param('learning_rate', 0.1)
    hparams.set_param('interpolation_type', 'hypercube')

    estimator = scrtl.separately_calibrated_rtl_classifier(
        feature_columns=feature_columns,
        hparams=hparams,
        keypoints_initializers_fn=init_fn)

    estimator.train(input_fn=train_input_fn)
    results = estimator.evaluate(input_fn=test_input_fn)
    # We should expect 1.0 accuracy.
    self.assertGreater(results['accuracy'], 0.999)

  def testCalibratedRtlWithMissingTraining(self):
    # x0 is missing with it's own vertex: so it can take very different values,
    # while x1 is missing and calibrated, in this case to the middle of the
    # lattice.
    x0 = np.array([0., 0., 1., 1., -1., -1., 0., 1.])
    x1 = np.array([0., 1., 0., 1., 0., 1., -1., -1.])
    training_y = np.array([1., 3., 7., 11., 23., 27., 2., 9.])
    x_samples = {'x0': x0, 'x1': x1}

    train_input_fn = numpy_io.numpy_input_fn(
        x=x_samples,
        y=training_y,
        batch_size=x0.shape[0],
        num_epochs=2000,
        shuffle=False)
    test_input_fn = numpy_io.numpy_input_fn(
        x=x_samples, y=training_y, shuffle=False)
    feature_columns = [
        feature_column_lib.numeric_column('x0'),
        feature_column_lib.numeric_column('x1'),
    ]

    def init_fn():
      return keypoints_initialization.uniform_keypoints_for_signal(
          2, 0., 1., 0., 1.)

    hparams = tfl_hparams.CalibratedRtlHParams(
        ['x0', 'x1'],
        num_keypoints=2,
        num_lattices=3,
        lattice_rank=2,
        learning_rate=0.1,
        missing_input_value=-1.)
    hparams.set_feature_param('x0', 'missing_vertex', True)

    estimator = scrtl.separately_calibrated_rtl_regressor(
        feature_columns=feature_columns,
        hparams=hparams,
        keypoints_initializers_fn=init_fn)

    estimator.train(input_fn=train_input_fn)
    results = estimator.evaluate(input_fn=test_input_fn)
    self.assertLess(results['average_loss'], 0.1)


if __name__ == '__main__':
  test.main()
