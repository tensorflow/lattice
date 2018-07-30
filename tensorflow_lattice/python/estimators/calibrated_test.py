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
"""Calibrated is an abstract base class. This mostly tests dependencies."""
import os
import tempfile

# Dependency imports

from tensorflow_lattice.python.estimators import calibrated as calibrated_lib
from tensorflow_lattice.python.estimators import hparams as lf_hparams

from tensorflow.python.platform import test


class CalibratedFake(calibrated_lib.Calibrated):
  """Fake Calibrated class, only used to instantiate the model."""

  def __init__(self,
               n_classes,
               feature_columns=None,
               model_dir=None,
               quantiles_dir=None,
               keypoints_initializers_fn=None,
               optimizer=None,
               config=None,
               hparams=None):
    super(CalibratedFake, self).__init__(
        n_classes, feature_columns, model_dir, quantiles_dir,
        keypoints_initializers_fn, optimizer, config, hparams, 'Fake')

  def calibration_structure_builder(self, columns_to_tensors, hparams):
    return None

  def prediction_builder_from_calibrated(
      self, mode, per_dimension_feature_names, hparams, calibrated):
    return None


class CalibratedTest(test.TestCase):
  """Constructor tests only, actual test of the code in CalibratedLinearTest."""

  def setUp(self):
    super(CalibratedTest, self).setUp()

  def _testConstructor(self, n_classes):
    hparams = lf_hparams.CalibratedHParams(
        feature_names=['x0', 'x1'],
        num_keypoints=20,
        feature__x0__num_keypoints=0)
    _ = CalibratedFake(n_classes=n_classes, hparams=hparams)

  def testConstructors(self):
    self._testConstructor(n_classes=2)
    self._testConstructor(n_classes=0)

  def testNumKeypointsInitialization(self):
    hparams = lf_hparams.CalibratedHParams(
        feature_names=['x0', 'x1'],
        num_keypoints=20,
        feature__x0__num_keypoints=0)
    _ = CalibratedFake(n_classes=2, hparams=hparams)

    # Test that same number of keypoints initialization is fine.
    self.assertEqual(
        calibrated_lib._update_keypoints(
            feature_name='x0', asked_keypoints=20, kp_init_keypoints=20), 20)

    # Test that fewer number of keypoints initialization is fine.
    self.assertEqual(
        calibrated_lib._update_keypoints(
            feature_name='x0', asked_keypoints=20, kp_init_keypoints=10), 10)

    # Test that no calibration is respected.
    self.assertEqual(
        calibrated_lib._update_keypoints(
            feature_name='x1', asked_keypoints=0, kp_init_keypoints=20), 0)
    self.assertEqual(
        calibrated_lib._update_keypoints(
            feature_name='x0', asked_keypoints=None, kp_init_keypoints=20),
        None)

    # Test that too many keypoints is not ok!
    self.assertRaisesRegexp(
        ValueError,
        r'Calibration initialization returned more keypoints \(20\) than '
        r'requested \(10\) for feature x0', calibrated_lib._update_keypoints,
        'x0', 10, 20)

  def testCreatedDirectory(self):
    # Create and remove temporary directory.
    model_dir = tempfile.mkdtemp()
    os.rmdir(model_dir)
    hparams = lf_hparams.CalibratedHParams(
        feature_names=['x0', 'x1'],
        num_keypoints=20,
        feature__x0__num_keypoints=10)
    CalibratedFake(n_classes=2, hparams=hparams, model_dir=model_dir)
    self.assertTrue(os.path.exists(model_dir))


if __name__ == '__main__':
  test.main()
