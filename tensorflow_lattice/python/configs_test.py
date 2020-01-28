# Copyright 2019 Google LLC
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

"""Tests for TFL model configuration library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow_lattice.python import configs


class ConfigsTest(tf.test.TestCase):

  def test_updates(self):
    model_config = configs.CalibratedLatticeConfig(
        output_min=0,
        regularizer_configs=[
            configs.RegularizerConfig(name='torsion', l2=2e-3),
        ],
        feature_configs=[
            configs.FeatureConfig(
                name='feature_a',
                pwl_calibration_input_keypoints='quantiles',
                pwl_calibration_num_keypoints=8,
                monotonicity=1,
                pwl_calibration_clip_max=100,
            ),
            configs.FeatureConfig(
                name='feature_b',
                lattice_size=3,
                unimodality='valley',
                pwl_calibration_input_keypoints='uniform',
                pwl_calibration_num_keypoints=5,
                pwl_calibration_clip_min=130,
                pwl_calibration_convexity='convex',
                regularizer_configs=[
                    configs.RegularizerConfig(name='calib_hessian', l2=3e-3),
                ],
            ),
            configs.FeatureConfig(
                name='feature_c',
                pwl_calibration_input_keypoints=[0.0, 0.5, 1.0],
                reflects_trust_in=[
                    configs.TrustConfig(feature_name='feature_a'),
                    configs.TrustConfig(feature_name='feature_b', direction=-1),
                ],
            ),
            configs.FeatureConfig(
                name='feature_d',
                num_buckets=3,
                vocabulary_list=['a', 'b', 'c'],
                default_value=-1,
            ),
        ])

    updates = [
        # Update values can be passed in as numbers.
        ('output_max', 1.0),  # update
        ('regularizer__torsion__l2', 0.004),  # update
        ('regularizer__calib_hessian__l1', 0.005),  # insert
        ('feature__feature_a__lattice_size', 3),  # update
        ('feature__feature_e__lattice_size', 4),  # insert
        # Update values can be strings.
        ('unrelated_hparams_not_affecting_config', 'unrelated'),
        ('feature__feature_a__regularizer__calib_wrinkle__l1', '0.6'),  # insert
        ('feature__feature_b__regularizer__calib_hessian__l1', '0.7'),  # update
        ('yet__another__unrelated_config', '4'),
    ]
    self.assertEqual(configs.apply_updates(model_config, updates), 7)

    model_config.feature_config_by_name('feature_a').monotonicity = 'none'
    model_config.feature_config_by_name('feature_f').num_buckets = 4  # insert

    feature_names = [
        feature_config.name for feature_config in model_config.feature_configs
    ]
    expected_feature_names = [
        'feature_a', 'feature_b', 'feature_c', 'feature_d', 'feature_e',
        'feature_f'
    ]
    self.assertCountEqual(feature_names, expected_feature_names)

    global_regularizer_names = [
        regularizer_config.name
        for regularizer_config in model_config.regularizer_configs
    ]
    expected_global_regularizer_names = ['torsion', 'calib_hessian']
    self.assertCountEqual(global_regularizer_names,
                          expected_global_regularizer_names)

    self.assertEqual(model_config.output_max, 1.0)
    self.assertEqual(
        model_config.feature_config_by_name('feature_a').lattice_size, 3)
    self.assertEqual(
        model_config.feature_config_by_name(
            'feature_b').pwl_calibration_convexity, 'convex')
    self.assertEqual(
        model_config.feature_config_by_name('feature_e').lattice_size, 4)
    self.assertEqual(
        model_config.regularizer_config_by_name('torsion').l2, 0.004)
    self.assertEqual(
        model_config.regularizer_config_by_name('calib_hessian').l1, 0.005)
    self.assertEqual(
        model_config.feature_config_by_name(
            'feature_a').regularizer_config_by_name('calib_wrinkle').l1, 0.6)
    self.assertEqual(
        model_config.feature_config_by_name(
            'feature_b').regularizer_config_by_name('calib_hessian').l1, 0.7)


if __name__ == '__main__':
  tf.test.main()
