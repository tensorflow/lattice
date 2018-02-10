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
"""CalibratedLinear provides canned estimators."""
# Dependency imports

from tensorflow_lattice.python.estimators import calibrated_linear
from tensorflow_lattice.python.estimators import hparams as tfl_hparams
from tensorflow_lattice.python.lib import keypoints_initialization
from tensorflow_lattice.python.lib import test_data

from tensorflow.python.estimator.canned import linear as linear_estimator
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

_NUM_KEYPOINTS = 50


class CalibratedLinearTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(CalibratedLinearTest, self).setUp()
    self._test_data = test_data.TestData()

  def _LinearRegressor(self, feature_columns):
    # Can be used for baseline.
    return linear_estimator.LinearRegressor(feature_columns=feature_columns)

  def _CalibratedLinearRegressor(self, feature_names, feature_columns,
                                 **hparams_args):

    def init_fn():
      return keypoints_initialization.uniform_keypoints_for_signal(
          _NUM_KEYPOINTS, -1., 1., -1., 1.)

    hparams = tfl_hparams.CalibratedLinearHParams(
        feature_names, num_keypoints=_NUM_KEYPOINTS, **hparams_args)
    return calibrated_linear.calibrated_linear_regressor(
        feature_columns=feature_columns,
        hparams=hparams,
        keypoints_initializers_fn=init_fn)

  def _CalibratedLinearRegressorWithQuantiles(self, feature_names,
                                              feature_columns, **hparams_args):
    """Model that saves/retrieves quantiles."""

    # Quantiles to be used for x2
    quantiles_dir = self.get_temp_dir()
    keypoints_initialization.save_quantiles_for_keypoints(
        input_fn=self._test_data.threed_input_fn(True),
        save_dir=quantiles_dir,
        feature_columns=feature_columns,
        num_steps=1)

    # Keypoint initialization function for x0 and x1
    def init_fn():
      return keypoints_initialization.uniform_keypoints_for_signal(
          _NUM_KEYPOINTS, -1., 1., -1., 1.)

    hparams = tfl_hparams.CalibratedLinearHParams(
        feature_names, num_keypoints=_NUM_KEYPOINTS, **hparams_args)
    return calibrated_linear.calibrated_linear_regressor(
        feature_columns=feature_columns,
        hparams=hparams,
        keypoints_initializers_fn={'x0': init_fn,
                                   'x1': init_fn},
        quantiles_dir=quantiles_dir  # Used for 'x2'
    )

  def _LinearClassifier(self, feature_columns):
    # Can be used for baseline.
    return linear_estimator.LinearClassifier(
        n_classes=2, feature_columns=feature_columns)

  def _CalibratedLinearClassifier(self, feature_names, feature_columns,
                                  **hparams_args):

    def init_fn():
      return keypoints_initialization.uniform_keypoints_for_signal(
          _NUM_KEYPOINTS, -1., 1., -1., 1.)

    hparams = tfl_hparams.CalibratedLinearHParams(
        feature_names, num_keypoints=_NUM_KEYPOINTS, **hparams_args)
    return calibrated_linear.calibrated_linear_classifier(
        feature_columns=feature_columns,
        hparams=hparams,
        keypoints_initializers_fn=init_fn)

  def testCalibratedLinearRegressorTraining1D(self):
    feature_columns = [
        feature_column_lib.numeric_column('x'),
    ]
    estimator = self._CalibratedLinearRegressor(['x'], feature_columns)
    estimator.train(input_fn=self._test_data.oned_input_fn())
    results = estimator.evaluate(input_fn=self._test_data.oned_input_fn())
    # For the record:
    #   Loss(CalibratedLinear)=~2.5e-5
    #   Loss(LinearRegressor)=~2.5e-2
    self.assertLess(results['average_loss'], 1e-4)

  def testCalibratedLinearMonotonicRegressorTraining1D(self):
    feature_columns = [
        feature_column_lib.numeric_column('x'),
    ]
    estimator = self._CalibratedLinearRegressor(['x'], feature_columns,
            feature__x__monotonicity=+1, feature__x__missing_input_value=-1.0)
    estimator.train(input_fn=self._test_data.oned_input_fn())
    results = estimator.evaluate(input_fn=self._test_data.oned_input_fn())
    self.assertLess(results['average_loss'], 2e-4)

  def testCalibratedLinearRegressorTraining1DWithCalibrationRegularizer(self):
    feature_columns = [
        feature_column_lib.numeric_column('x'),
    ]
    estimator = self._CalibratedLinearRegressor(
        ['x'],
        feature_columns,
        calibration_l1_reg=0.001,
        calibration_l2_reg=0.001,
        calibration_l1_laplacian_reg=0.001,
        calibration_l2_laplacian_reg=0.001)
    estimator.train(input_fn=self._test_data.oned_input_fn())
    results = estimator.evaluate(input_fn=self._test_data.oned_input_fn())
    self.assertLess(results['average_loss'], 1e-2)

  def testCalibratedLinearRegressorTraining2D(self):
    feature_columns = [
        feature_column_lib.numeric_column('x0'),
        feature_column_lib.numeric_column('x1'),
    ]
    estimator = self._CalibratedLinearRegressor(['x0', 'x1'], feature_columns)
    estimator.train(input_fn=self._test_data.twod_input_fn())
    results = estimator.evaluate(input_fn=self._test_data.twod_input_fn())
    # For the record:
    #   Loss(CalibratedLinear)=~6.9e-5
    #   Loss(LinearRegressor)=~3.3e-2
    self.assertLess(results['average_loss'], 1e-4)

  def testCalibratedLinearRegressorTraining3D(self):
    # Tests also categorical features that has a limited number
    # of valid values.
    feature_columns = [
        feature_column_lib.numeric_column('x0'),
        feature_column_lib.numeric_column('x1'),
        feature_column_lib.categorical_column_with_vocabulary_list(
            'x2', ['Y', 'N'])
    ]
    with ops.Graph().as_default():
      estimator = self._CalibratedLinearRegressorWithQuantiles(
          ['x0', 'x1', 'x2'], feature_columns)
    estimator.train(input_fn=self._test_data.threed_input_fn(False, 4))
    results = estimator.evaluate(input_fn=self._test_data.threed_input_fn(
        False, 1))
    # For the record:
    #   average_loss(CalibratedLinear, 4 epochs)=~1e-5
    #   average_loss(LinearRegressor, 100 epochs)=~0.159
    self.assertLess(results['average_loss'], 1e-4)

  def testCalibratedLinearRegressorTrainingMultiDimensionalFeature(self):
    feature_columns = [
        feature_column_lib.numeric_column('x', shape=(2,)),
    ]

    # With calibration.
    estimator = self._CalibratedLinearRegressor(['x'], feature_columns)
    estimator.train(input_fn=self._test_data.multid_feature_input_fn())
    results = estimator.evaluate(
        input_fn=self._test_data.multid_feature_input_fn())
    # For the record:
    #   Loss(CalibratedLinear)=~6.6e-5
    #   Loss(LinearRegressor)=~3.2e-2
    self.assertLess(results['average_loss'], 1e-4)

    # Turn-off calibration for feature 'x', it should turn if off for both
    # dimensions.
    estimator = self._CalibratedLinearRegressor(
        ['x'], feature_columns, feature__x__num_keypoints=0)
    estimator.train(input_fn=self._test_data.multid_feature_input_fn())
    results = estimator.evaluate(
        input_fn=self._test_data.multid_feature_input_fn())
    self.assertGreater(results['average_loss'], 1e-2)

  def testCalibratedLinearClassifierTraining(self):
    feature_columns = [
        feature_column_lib.numeric_column('x0'),
        feature_column_lib.numeric_column('x1'),
    ]
    estimator = self._CalibratedLinearClassifier(['x0', 'x1'], feature_columns)
    estimator.train(input_fn=self._test_data.twod_classificer_input_fn())
    results = estimator.evaluate(
        input_fn=self._test_data.twod_classificer_input_fn())
    # For the record:
    #   auc(CalibratedLinear)=~0.999
    #   auc(LinearClassifier)=~0.481
    self.assertGreater(results['auc'], 0.990)

  def testCalibratedLinearClassifierTrainingWithCalibrationRegularizer(self):
    feature_columns = [
        feature_column_lib.numeric_column('x0'),
        feature_column_lib.numeric_column('x1'),
    ]
    estimator = self._CalibratedLinearClassifier(
        ['x0', 'x1'],
        feature_columns,
        calibration_l1_reg=0.001,
        calibration_l2_reg=0.03,
        calibration_l1_laplacian_reg=0.03,
        calibration_l2_laplacian_reg=0.05)
    estimator.train(input_fn=self._test_data.twod_classificer_input_fn())
    results = estimator.evaluate(
        input_fn=self._test_data.twod_classificer_input_fn())
    self.assertGreater(results['auc'], 0.980)


if __name__ == '__main__':
  test.main()
