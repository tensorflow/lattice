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
"""CalibratedEtl tests."""
# Dependency imports

import numpy as np

from tensorflow_lattice.python.estimators import calibrated_etl
from tensorflow_lattice.python.estimators import hparams as tfl_hparams
from tensorflow_lattice.python.lib import keypoints_initialization
from tensorflow_lattice.python.lib import test_data

from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.platform import test

_NUM_KEYPOINTS = 50


class CalibratedEtlHParamsTest(test.TestCase):

  def testEmptyMonotonicLatticeRankExpectsError(self):
    hparams = tfl_hparams.CalibratedEtlHParams(feature_names=['x'])
    hparams.set_param('monotonic_num_lattices', 2)
    hparams.set_param('monotonic_lattice_size', 2)
    with self.assertRaisesRegexp(
        ValueError,
        'Hyperparameter configuration cannot be used in the calibrated etl '
        'estimator.'):
      calibrated_etl.calibrated_etl_classifier(hparams=hparams)

  def testEmptyMonotonicLatticeSizeExpectsError(self):
    hparams = tfl_hparams.CalibratedEtlHParams(feature_names=['x'])
    hparams.set_param('monotonic_num_lattices', 2)
    hparams.set_param('monotonic_lattice_rank', 2)
    with self.assertRaisesRegexp(
        ValueError,
        'Hyperparameter configuration cannot be used in the calibrated etl '
        'estimator.'):
      calibrated_etl.calibrated_etl_classifier(hparams=hparams)

  def testEmptyNonMonotonicLatticeRankExpectsError(self):
    hparams = tfl_hparams.CalibratedEtlHParams(feature_names=['x'])
    hparams.set_param('non_monotonic_num_lattices', 2)
    hparams.set_param('non_monotonic_lattice_size', 2)
    with self.assertRaisesRegexp(
        ValueError,
        'Hyperparameter configuration cannot be used in the calibrated etl '
        'estimator.'):
      calibrated_etl.calibrated_etl_classifier(hparams=hparams)

  def testEmptyNonMonotonicLatticeSizeExpectsError(self):
    hparams = tfl_hparams.CalibratedEtlHParams(feature_names=['x'])
    hparams.set_param('non_monotonic_num_lattices', 2)
    hparams.set_param('non_monotonic_lattice_rank', 2)
    with self.assertRaisesRegexp(
        ValueError,
        'Hyperparameter configuration cannot be used in the calibrated etl '
        'estimator.'):
      calibrated_etl.calibrated_etl_classifier(hparams=hparams)

  def testWrongLatticeRegularization(self):
    hparams = tfl_hparams.CalibratedEtlHParams(feature_names=['x'])
    hparams.set_param('non_monotonic_num_lattices', 2)
    hparams.set_param('non_monotonic_lattice_size', 2)
    hparams.set_param('nno_monotonic_lattice_rank', 2)
    hparams.set_feature_param('x', 'lattice_l1_reg', 0.1)
    hparams.set_feature_param('x', 'lattice_l2_reg', 0.1)
    hparams.set_feature_param('x', 'lattice_l1_torsion_reg', 0.1)
    hparams.set_feature_param('x', 'lattice_l1_torsion_reg', 0.1)

    with self.assertRaisesRegexp(
        ValueError,
        'Hyperparameter configuration cannot be used in the calibrated etl '
        'estimator.'):
      calibrated_etl.calibrated_etl_classifier(hparams=hparams)


class CalibratedEtlTest(test.TestCase):

  def setUp(self):
    super(CalibratedEtlTest, self).setUp()
    self._test_data = test_data.TestData()

  def _CalibratedEtlRegressor(self,
                              feature_names,
                              feature_columns,
                              weight_column=None,
                              **hparams_args):

    def init_fn():
      return keypoints_initialization.uniform_keypoints_for_signal(
          _NUM_KEYPOINTS, -1., 1., 0., 1.)

    hparams = tfl_hparams.CalibratedEtlHParams(
        feature_names,
        num_keypoints=_NUM_KEYPOINTS,
        monotonic_num_lattices=1,
        monotonic_lattice_rank=1,
        monotonic_lattice_size=2,
        non_monotonic_num_lattices=1,
        non_monotonic_lattice_rank=1,
        non_monotonic_lattice_size=2,
        **hparams_args)
    # Turn off monotonic calibrator.
    hparams.set_param('calibration_monotonic', None)
    hparams.set_param('learning_rate', 0.1)

    return calibrated_etl.calibrated_etl_regressor(
        feature_columns=feature_columns,
        weight_column=weight_column,
        hparams=hparams,
        keypoints_initializers_fn=init_fn)

  def _CalibratedEtlClassifier(self, feature_columns, **hparams_args):

    def init_fn():
      return keypoints_initialization.uniform_keypoints_for_signal(
          _NUM_KEYPOINTS, -1., 1., 0., 1.)

    hparams = tfl_hparams.CalibratedEtlHParams(
        num_keypoints=_NUM_KEYPOINTS,
        monotonic_num_lattices=1,
        monotonic_lattice_rank=1,
        monotonic_lattice_size=2,
        non_monotonic_num_lattices=1,
        non_monotonic_lattice_rank=1,
        non_monotonic_lattice_size=2,
        **hparams_args)
    # Turn off monotonic calibrator.
    hparams.set_param('calibration_monotonic', None)
    hparams.set_param('learning_rate', 0.1)

    return calibrated_etl.calibrated_etl_classifier(
        feature_columns=feature_columns,
        hparams=hparams,
        keypoints_initializers_fn=init_fn)

  def testCalibratedEtlRegressorTraining1D(self):
    feature_columns = [
        feature_column_lib.numeric_column('x'),
    ]
    estimator = self._CalibratedEtlRegressor(
        ['x'], feature_columns, interpolation_type='simplex')
    estimator.train(input_fn=self._test_data.oned_input_fn())
    # Here we only check the successful evaluation.
    # Checking the actual number, accuracy, etc, makes the test too flaky.
    _ = estimator.evaluate(input_fn=self._test_data.oned_input_fn())

  def testCalibratedEtlRegressorWeightedTraining1D(self):
    feature_columns = [feature_column_lib.numeric_column('x')]
    weight_column = feature_column_lib.numeric_column('zero')
    estimator = self._CalibratedEtlRegressor(
        ['x'], feature_columns, weight_column=weight_column)
    estimator.train(input_fn=self._test_data.oned_zero_weight_input_fn())
    results = estimator.evaluate(
        input_fn=self._test_data.oned_zero_weight_input_fn())
    # Expects almost zero since the weight values are exactly zero.
    self.assertLess(results['average_loss'], 1e-7)

  def testCalibratedEtlRegressorTraining2D(self):
    feature_columns = [
        feature_column_lib.numeric_column('x0'),
        feature_column_lib.numeric_column('x1'),
    ]
    estimator = self._CalibratedEtlRegressor(
        ['x0', 'x1'], feature_columns, interpolation_type='hypercube')
    estimator.train(input_fn=self._test_data.twod_input_fn())
    # Here we only check the successful evaluation.
    # Checking the actual number, accuracy, etc, makes the test too flaky.
    _ = estimator.evaluate(input_fn=self._test_data.twod_input_fn())

  def testCalibratedEtlRegressorTraining2DWithCalbrationRegularization(self):
    feature_columns = [
        feature_column_lib.numeric_column('x0'),
        feature_column_lib.numeric_column('x1'),
    ]
    estimator = self._CalibratedEtlRegressor(
        ['x0', 'x1'],
        feature_columns,
        interpolation_type='simplex',
        calibration_l1_reg=1e-2,
        calibration_l2_reg=1e-2,
        calibration_l1_laplacian_reg=0.05,
        calibration_l2_laplacian_reg=0.01)
    estimator.train(input_fn=self._test_data.twod_input_fn())
    # Here we only check the successful evaluation.
    # Checking the actual number, accuracy, etc, makes the test too flaky.
    _ = estimator.evaluate(input_fn=self._test_data.twod_input_fn())

  def testCalibratedEtlRegressorTraining2DWithLatticeRegularizer(self):
    feature_columns = [
        feature_column_lib.numeric_column('x0'),
        feature_column_lib.numeric_column('x1'),
    ]
    estimator = self._CalibratedEtlRegressor(
        ['x0', 'x1'],
        feature_columns,
        interpolation_type='simplex',
        lattice_l1_reg=1.0,
        lattice_l2_reg=1.0,
        lattice_l1_torsion_reg=1.0,
        lattice_l2_torsion_reg=1.0,
        lattice_l1_laplacian_reg=1.0,
        lattice_l2_laplacian_reg=1.0)
    estimator.train(input_fn=self._test_data.twod_input_fn())
    # Here we only check the successful evaluation.
    # Checking the actual number, accuracy, etc, makes the test too flaky.
    _ = estimator.evaluate(input_fn=self._test_data.twod_input_fn())

  def testCalibratedEtlRegressorTrainingMultiDimensionalFeature(self):
    feature_columns = [
        feature_column_lib.numeric_column('x', shape=(2,)),
    ]
    estimator = self._CalibratedEtlRegressor(['x'], feature_columns)
    estimator.train(input_fn=self._test_data.multid_feature_input_fn())
    results = estimator.evaluate(
        input_fn=self._test_data.multid_feature_input_fn())
    self.assertLess(results['average_loss'], 1e-2)

    # Turn-off calibration for feature 'x', it should turn off for both
    # dimensions, and the results should get much worse.
    estimator = self._CalibratedEtlRegressor(
        ['x'], feature_columns, feature__x__num_keypoints=0)
    estimator.train(input_fn=self._test_data.multid_feature_input_fn())
    # Here we only check the successful evaluation.
    # Checking the actual number, accuracy, etc, makes the test too flaky.
    _ = estimator.evaluate(
        input_fn=self._test_data.multid_feature_input_fn())

  def testCalibratedEtlClassifierTraining(self):
    feature_columns = [
        feature_column_lib.numeric_column('x0'),
        feature_column_lib.numeric_column('x1'),
    ]
    estimator = self._CalibratedEtlClassifier(feature_columns)
    estimator.train(input_fn=self._test_data.twod_classificer_input_fn())
    # Here we only check the successful evaluation.
    # Checking the actual number, accuracy, etc, makes the test too flaky.
    _ = estimator.evaluate(
        input_fn=self._test_data.twod_classificer_input_fn())

  def testCalibratedEtlClassifierTrainingWithCalibrationRegularizer(self):
    feature_columns = [
        feature_column_lib.numeric_column('x0'),
        feature_column_lib.numeric_column('x1'),
    ]
    estimator = self._CalibratedEtlClassifier(
        feature_columns,
        calibration_l1_reg=1e-2,
        calibration_l2_reg=1e-2,
        calibration_l1_laplacian_reg=1e-1,
        calibration_l2_laplacian_reg=1e-1,
        interpolation_type='hypercube')

    estimator.train(input_fn=self._test_data.twod_classificer_input_fn())
    # Here we only check the successful evaluation.
    # Checking the actual number, accuracy, etc, makes the test too flaky.
    _ = estimator.evaluate(
        input_fn=self._test_data.twod_classificer_input_fn())

  def testCalibratedEtlClassifierTrainingWithLatticeRegularizer(self):
    feature_columns = [
        feature_column_lib.numeric_column('x0'),
        feature_column_lib.numeric_column('x1'),
    ]
    estimator = self._CalibratedEtlClassifier(
        feature_columns,
        lattice_l1_reg=1.0,
        lattice_l2_reg=1.0,
        lattice_l1_torsion_reg=1.0,
        lattice_l2_torsion_reg=1.0,
        lattice_l1_laplacian_reg=1.0,
        lattice_l2_laplacian_reg=1.0,
        interpolation_type='hypercube')

    estimator.train(input_fn=self._test_data.twod_classificer_input_fn())
    # Here we only check the successful evaluation.
    # Checking the actual number, accuracy, etc, makes the test too flaky.
    _ = estimator.evaluate(
        input_fn=self._test_data.twod_classificer_input_fn())

  def testCalibratedEtlMonotonicClassifierTraining(self):
    # Construct the following training pair.
    #
    # Training: (x, y)
    # ([0., 0.], 0.0)
    # ([0., 1.], 1.0)
    # ([1., 0.], 1.0)
    # ([1., 1.], 0.0)
    #
    # which is not a monotonic function. Then check the forcing monotonicity
    # resulted in the following monotonicity or not.
    # f(0, 0) <= f(0, 1), f(0, 0) <= f(1, 0), f(0, 1) <= f(1, 1),
    # f(1, 0) < = f(1, 1).
    x0 = np.array([0.0, 0.0, 1.0, 1.0])
    x1 = np.array([0.0, 1.0, 0.0, 1.0])
    x_samples = {'x0': x0, 'x1': x1}
    training_y = np.array([[False], [True], [True], [False]])

    train_input_fn = numpy_io.numpy_input_fn(
        x=x_samples, y=training_y, batch_size=4, num_epochs=1000, shuffle=False)
    test_input_fn = numpy_io.numpy_input_fn(x=x_samples, y=None, shuffle=False)

    # Define monotonic lattice classifier.
    feature_columns = [
        feature_column_lib.numeric_column('x0'),
        feature_column_lib.numeric_column('x1'),
    ]

    def init_fn():
      return keypoints_initialization.uniform_keypoints_for_signal(
          2, 0., 1., 0., 1.)

    hparams = tfl_hparams.CalibratedEtlHParams(
        num_keypoints=2,
        monotonic_num_lattices=2,
        monotonic_lattice_rank=2,
        monotonic_lattice_size=2)
    hparams.set_param('calibration_monotonic', +1)
    hparams.set_param('lattice_monotonic', True)
    hparams.set_param('learning_rate', 0.1)

    estimator = calibrated_etl.calibrated_etl_classifier(
        feature_columns=feature_columns,
        hparams=hparams,
        keypoints_initializers_fn=init_fn)
    estimator.train(input_fn=train_input_fn)
    predictions = [
        results['logits'][0]
        for results in estimator.predict(input_fn=test_input_fn)
    ]

    self.assertEqual(len(predictions), 4)
    # Check monotonicity. Note that projection has its own precision, so we
    # add a small number.
    self.assertLess(predictions[0], predictions[1] + 1e-6)
    self.assertLess(predictions[0], predictions[2] + 1e-6)
    self.assertLess(predictions[1], predictions[3] + 1e-6)
    self.assertLess(predictions[2], predictions[3] + 1e-6)

  def testCalibratedEtlWithMissingTraining(self):
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

    hparams = tfl_hparams.CalibratedEtlHParams(
        ['x0', 'x1'],
        num_keypoints=2,
        non_monotonic_num_lattices=5,
        non_monotonic_lattice_rank=2,
        non_monotonic_lattice_size=2,
        learning_rate=0.1,
        missing_input_value=-1.)

    estimator = calibrated_etl.calibrated_etl_regressor(
        feature_columns=feature_columns,
        hparams=hparams,
        keypoints_initializers_fn=init_fn)

    estimator.train(input_fn=train_input_fn)
    # Here we only check the successful evaluation.
    # Checking the actual number, accuracy, etc, makes the test too flaky.
    _ = estimator.evaluate(input_fn=test_input_fn)


if __name__ == '__main__':
  test.main()
