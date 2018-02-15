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
"""Optimizers test."""
# Dependency imports

from tensorflow_lattice.python.estimators import calibrated_linear
from tensorflow_lattice.python.estimators import hparams as tfl_hparams
from tensorflow_lattice.python.estimators import optimizers as tfl_optimizers
from tensorflow_lattice.python.lib import keypoints_initialization
from tensorflow_lattice.python.lib import test_data

from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


_NUM_KEYPOINTS = 50


class OptimizersTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(OptimizersTest, self).setUp()
    self._test_data = test_data.TestData()

  def _runCalibratedLinearRegressorTrainingWithOptimizer(self, optimizer_fn):
    def init_fn():
      return keypoints_initialization.uniform_keypoints_for_signal(
          _NUM_KEYPOINTS, -1., 1., -1., 1.)
    feature_names = ['x']
    feature_columns = [feature_column_lib.numeric_column('x')]
    hparams = tfl_hparams.CalibratedLinearHParams(
        feature_names, num_keypoints=_NUM_KEYPOINTS,
        learning_rate=0.1)
    estimator = calibrated_linear.calibrated_linear_regressor(
            feature_columns=feature_columns,
            hparams=hparams,
            keypoints_initializers_fn=init_fn,
            optimizer=optimizer_fn)
    estimator.train(input_fn=self._test_data.oned_input_fn())
    _ = estimator.evaluate(input_fn=self._test_data.oned_input_fn())

  def testCalibratedLinearRegressorWithSgd(self):
    optimizer = tfl_optimizers.gradient_descent_polynomial_decay()
    self._runCalibratedLinearRegressorTrainingWithOptimizer(optimizer)

  def testCalibratedLinearRegressorWithSgd(self):
    optimizer = tfl_optimizers.adagrad_polynomial_decay()
    self._runCalibratedLinearRegressorTrainingWithOptimizer(optimizer)


if __name__ == '__main__':
  test.main()
