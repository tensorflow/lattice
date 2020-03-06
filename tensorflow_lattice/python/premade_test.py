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
"""Tests for Tensorflow Lattice premade."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_lattice.python import configs
from tensorflow_lattice.python import premade_lib

unspecified_feature_configs = [
    configs.FeatureConfig(
        name='numerical_1',
        lattice_size=2,
        pwl_calibration_input_keypoints=np.linspace(0.0, 1.0, num=10),
    ),
    configs.FeatureConfig(
        name='numerical_2',
        lattice_size=2,
        pwl_calibration_input_keypoints=np.linspace(0.0, 1.0, num=10),
    ),
    configs.FeatureConfig(
        name='categorical',
        lattice_size=2,
        num_buckets=2,
        monotonicity=[('0.0', '1.0')],
        vocabulary_list=['0.0', '1.0'],
    ),
]

specified_feature_configs = [
    configs.FeatureConfig(
        name='numerical_1',
        lattice_size=2,
        pwl_calibration_input_keypoints=np.linspace(0.0, 1.0, num=10),
    ),
    configs.FeatureConfig(
        name='numerical_2',
        lattice_size=2,
        pwl_calibration_input_keypoints=np.linspace(0.0, 1.0, num=10),
    ),
    configs.FeatureConfig(
        name='categorical',
        lattice_size=2,
        num_buckets=2,
        monotonicity=[(0, 1)],
    ),
]


class PremadeTest(parameterized.TestCase, tf.test.TestCase):
  """Tests for TFL premade."""

  def testSetRandomLattices(self):
    random_model_config = configs.CalibratedLatticeEnsembleConfig(
        feature_configs=copy.deepcopy(unspecified_feature_configs),
        lattices='random',
        num_lattices=3,
        lattice_rank=2,
        separate_calibrators=True,
        output_initialization=[-1.0, 1.0])

    premade_lib.set_random_lattice_ensemble(random_model_config)
    self.assertLen(random_model_config.lattices, 3)
    self.assertListEqual(
        [2, 2, 2], [len(lattice) for lattice in random_model_config.lattices])

    specified_model_config = configs.CalibratedLatticeEnsembleConfig(
        feature_configs=copy.deepcopy(specified_feature_configs),
        lattices=[['numerical_1', 'categorical'],
                  ['numerical_2', 'categorical']],
        num_lattices=2,
        lattice_rank=2,
        separate_calibrators=True,
        output_initialization=[-1.0, 1.0])

    with self.assertRaisesRegex(
        ValueError, 'model_config.lattices must be set to \'random\'.'):
      premade_lib.set_random_lattice_ensemble(specified_model_config)

  def testSetCategoricalMonotonicities(self):
    set_feature_configs = copy.deepcopy(unspecified_feature_configs)
    premade_lib.set_categorical_monotonicities(set_feature_configs)
    expectation = [(0, 1)]
    self.assertListEqual(expectation, set_feature_configs[2].monotonicity)

  def testVerifyConfig(self):
    unspecified_model_config = configs.CalibratedLatticeEnsembleConfig(
        feature_configs=copy.deepcopy(unspecified_feature_configs),
        lattices='random',
        num_lattices=3,
        lattice_rank=2,
        separate_calibrators=True,
        output_initialization=[-1.0, 1.0])

    with self.assertRaisesRegex(
        ValueError, 'Lattices are not fully specified for ensemble config.'):
      premade_lib.verify_config(unspecified_model_config)
    premade_lib.set_random_lattice_ensemble(unspecified_model_config)
    with self.assertRaisesRegex(
        ValueError,
        'Element 0 for tuple 0 for feature categorical monotonicity is not an '
        'index.'):
      premade_lib.verify_config(unspecified_model_config)
    fixed_feature_configs = copy.deepcopy(unspecified_feature_configs)
    premade_lib.set_categorical_monotonicities(fixed_feature_configs)
    unspecified_model_config.feature_configs = fixed_feature_configs
    premade_lib.verify_config(unspecified_model_config)

    specified_model_config = configs.CalibratedLatticeEnsembleConfig(
        feature_configs=copy.deepcopy(specified_feature_configs),
        lattices=[['numerical_1', 'categorical'],
                  ['numerical_2', 'categorical']],
        num_lattices=2,
        lattice_rank=2,
        separate_calibrators=True,
        output_initialization=[-1.0, 1.0])

    premade_lib.verify_config(specified_model_config)


if __name__ == '__main__':
  tf.test.main()
