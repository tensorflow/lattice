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
"""Tests for hyper-parameters support class for TensorFlow Lattice."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_lattice.python.estimators import hparams


class TensorFlowLatticeHParamsTest(tf.test.TestCase):

  def testPerFeatureHParams(self):
    default_num_keypoints = 10
    feature_x0_num_keypoints = 5
    hp = hparams.PerFeatureHParams(
        ['x0', 'x1'],
        num_keypoints=default_num_keypoints,
        feature__x0__num_keypoints=feature_x0_num_keypoints)
    hp.add_feature(['x2'])
    self.assertEqual(hp.get_feature_names(), ['x0', 'x1', 'x2'])

    # Check missing parameter: not feature specific parameter
    # can be set if the generic one wasn't set first.
    with self.assertRaises(ValueError):
      hp.set_param('feature__x0__foobar', 10)

    # Make sure returns copy of internal list.
    hp.get_feature_names()[0] = 'z'
    self.assertEqual(hp.get_feature_names(), ['x0', 'x1', 'x2'])

    # Check values: both global and for specialized value for x0.
    self.assertEqual(hp.num_keypoints, default_num_keypoints)
    self.assertEqual(
        hp.get_feature_param('x0', 'num_keypoints'), feature_x0_num_keypoints)
    self.assertEqual(
        hp.get_feature_param('x1', 'num_keypoints'), default_num_keypoints)
    self.assertEqual(
        hp.get_feature_param('x2', 'num_keypoints'), default_num_keypoints)

    # Check missing parameter.
    with self.assertRaises(AttributeError):
      _ = hp.foobar

    # Check that missing feature raises exception.
    with self.assertRaisesRegexp(ValueError, 'Unknown feature name "x3".*'):
      hp.get_feature_param('x3', 'num_keypoints')

    # Check that missing parameter returns None.
    self.assertEqual(hp.get_feature_param('x2', 'unknown_parameter'), None)

    # Check is_feature_set_param.
    self.assertEqual(hp.is_feature_set_param('x0', 'num_keypoints'), True)
    self.assertEqual(hp.is_feature_set_param('x1', 'num_keypoints'), False)

    # Check that object can't be created with feature specific parameters set
    # for unknown feature.
    with self.assertRaisesRegexp(
        ValueError, 'Unknown feature "x2" for feature specific parameter '
        '"feature__x2__num_keypoints"'):
      # x2 doesn't exist, this should raise.
      _ = hparams.PerFeatureHParams(
          ['x0', 'x1'],
          num_keypoints=default_num_keypoints,
          feature__x2__num_keypoints=10)

  def testAddFeature(self):
    default_num_keypoints = 10
    feature_x0_num_keypoints = 5
    hp = hparams.PerFeatureHParams(
        [u'x0', 'x1'],
        num_keypoints=default_num_keypoints,
        feature__x0__num_keypoints=feature_x0_num_keypoints)
    # Unicode feature name.
    hp.add_feature([u'x2'])
    self.assertEqual(hp.get_feature_names(), ['x0', 'x1', 'x2'])
    self.assertEqual(
        hp.get_feature_param('x0', 'num_keypoints'), feature_x0_num_keypoints)
    self.assertEqual(
        hp.get_feature_param('x1', 'num_keypoints'), default_num_keypoints)
    self.assertEqual(
        hp.get_feature_param('x2', 'num_keypoints'), default_num_keypoints)
    # Feature name not of expected type string.
    with self.assertRaises(ValueError) as value_error:
      hp.add_feature([1.0])
    self.assertEqual('feature_name should either be a list of strings,'
                     ' or a string, got "[1.0]"',
                     str(value_error.exception))

  def testGlobalPerFeatureHParams(self):
    hp = hparams.PerFeatureHParams(['x0', 'x1'], num_keypoints=2)
    self.assertEqual(hp.get_param('num_keypoints'), 2)
    hp.set_param('num_keypoints', 3)
    self.assertEqual(hp.get_param('num_keypoints'), 3)

  def testParseHParms(self):
    hp_from = hparams.PerFeatureHParams(['x0', 'x1'], num_keypoints=5)
    hp_to = hparams.PerFeatureHParams(['x0', 'x1'], num_keypoints=2)
    hp_to.set_feature_param('x0', 'num_keypoints', 3)
    hp_to.parse_hparams(hp_from)
    self.assertEqual(hp_to.get_feature_param('x0', 'num_keypoints'), 3)
    self.assertEqual(hp_to.get_feature_param('x1', 'num_keypoints'), 5)
    hp_to.parse_hparams(None)

  def testParseString(self):
    hp = hparams.PerFeatureHParams(
        ['x0', 'x1', 'x2'], num_keypoints=2, learning_rate=1.0)
    hp.set_feature_param('x0', 'num_keypoints', 3)

    # Test normal use case.
    hp.parse('num_keypoints=5,learning_rate=0.1,feature__x2__num_keypoints=7')
    self.assertEqual(hp.get_feature_param('x0', 'num_keypoints'), 3)
    self.assertEqual(hp.get_feature_param('x1', 'num_keypoints'), 5)
    self.assertEqual(hp.get_feature_param('x2', 'num_keypoints'), 7)
    self.assertEqual(hp.learning_rate, 0.1)

    # Test that parsing None and empy has no effect.
    hp.parse(None)
    hp.parse('')
    self.assertEqual(hp.get_feature_param('x0', 'num_keypoints'), 3)
    self.assertEqual(hp.get_feature_param('x1', 'num_keypoints'), 5)
    self.assertEqual(hp.get_feature_param('x2', 'num_keypoints'), 7)
    self.assertEqual(hp.learning_rate, 0.1)

    # Test failures.
    with self.assertRaises(ValueError):
      hp.parse('feature__x3__num_keypoints=10')  # Unknwon feature.
    with self.assertRaises(ValueError):
      hp.parse('foobar=10')  # Unknwon parameter.
    with self.assertRaises(ValueError):
      hp.parse('feature__x1__foobar=10')  # Unknwon parameter for feature.
    with self.assertRaises(ValueError):
      hp.parse('num_keypoints=1.1')  # Invalid type to parse.

  def testSetParamType(self):
    hp = hparams.PerFeatureHParams(['x0', 'x1'], foo='abc')
    hp.parse('foo=def')
    self.assertEqual(hp.foo, 'def')
    hp.set_param('foo', 10)
    hp.set_param_type('foo', int)
    with self.assertRaises(ValueError):
      # Should fail, since now foo is of type int.
      hp.parse('foo=def')

  def testConstructorsAllTypes(self):
    _ = hparams.CalibratedHParams(['x0', 'x1'])
    _ = hparams.CalibratedLinearHParams(['x0', 'x1'], learning_rate=0.1)
    _ = hparams.CalibratedLatticeHParams(['x0', 'x1'], learning_rate=0.1)
    _ = hparams.CalibratedRtlHParams(['x0', 'x1'], learning_rate=0.1)
    etl = hparams.CalibratedEtlHParams(['x0', 'x1'], learning_rate=0.1)

    etl.parse('calibration_bound=yes')
    self.assertTrue(etl.calibration_bound)
    etl.parse('calibration_bound=off')
    self.assertFalse(etl.calibration_bound)
    with self.assertRaises(ValueError):
      etl.parse('calibration_bound=foobar')

  def testAddNonExistingPerFeatureParam(self):
    hp = hparams.CalibratedLinearHParams(['x0', 'x1'])
    hp.set_feature_param('x0', 'calibration_l2_laplacian_reg', 0.1)
    self.assertAlmostEqual(
        hp.get_feature_param('x0', 'calibration_l2_laplacian_reg'), 0.1)


if __name__ == '__main__':
  tf.test.main()
