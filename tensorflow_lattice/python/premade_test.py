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
import json

import tempfile
from absl import logging
from absl.testing import parameterized
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_lattice.python import configs
from tensorflow_lattice.python import premade
from tensorflow_lattice.python import premade_lib
# pylint: disable=g-import-not-at-top
# Use Keras 2.
version_fn = getattr(tf.keras, 'version', None)
if version_fn and version_fn().startswith('3.'):
  import tf_keras as keras
else:
  keras = tf.keras

fake_data = {
    'train_xs': [np.array([1]), np.array([3]), np.array([0])],
    'train_ys': np.array([1]),
    'eval_xs': [np.array([2]), np.array([30]), np.array([-3])]
}

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

feature_configs = [
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

  def setUp(self):
    super(PremadeTest, self).setUp()
    keras.utils.set_random_seed(42)

    # UCI Statlog (Heart) dataset.
    heart_csv_file = keras.utils.get_file(
        'heart.csv',
        'http://storage.googleapis.com/download.tensorflow.org/data/heart.csv',
    )
    heart_df = pd.read_csv(heart_csv_file)
    thal_vocab_list = ['normal', 'fixed', 'reversible']
    heart_df['thal'] = heart_df['thal'].map(
        {v: i for i, v in enumerate(thal_vocab_list)})
    heart_df = heart_df.astype(float)

    heart_train_size = int(len(heart_df) * 0.8)
    heart_train_dict = dict(heart_df[:heart_train_size])
    heart_test_dict = dict(heart_df[heart_train_size:])

    # Features:
    # - age
    # - sex
    # - cp        chest pain type (4 values)
    # - trestbps  resting blood pressure
    # - chol      serum cholestoral in mg/dl
    # - fbs       fasting blood sugar > 120 mg/dl
    # - restecg   resting electrocardiographic results (values 0,1,2)
    # - thalach   maximum heart rate achieved
    # - exang     exercise induced angina
    # - oldpeak   ST depression induced by exercise relative to rest
    # - slope     the slope of the peak exercise ST segment
    # - ca        number of major vessels (0-3) colored by flourosopy
    # - thal      normal; fixed defect; reversable defect
    self.heart_feature_configs = [
        configs.FeatureConfig(
            name='age',
            lattice_size=3,
            monotonicity='increasing',
            # We must set the keypoints manually.
            pwl_calibration_num_keypoints=5,
            pwl_calibration_input_keypoints='quantiles',
            pwl_calibration_clip_max=100.,
            # Per feature regularization.
            regularizer_configs=[
                configs.RegularizerConfig(name='calib_wrinkle', l2=0.1),
            ],
        ),
        configs.FeatureConfig(
            name='sex',
            num_buckets=2,
        ),
        configs.FeatureConfig(
            name='cp',
            monotonicity='increasing',
            # Keypoints that are uniformly spaced.
            pwl_calibration_num_keypoints=4,
            pwl_calibration_input_keypoints='uniform',
        ),
        configs.FeatureConfig(
            name='chol',
            monotonicity='increasing',
            # Explicit input keypoints initialization.
            pwl_calibration_input_keypoints=[126.0, 210.0, 247.0, 286.0, 564.0],
            # Calibration can be forced to span the full output range
            # by clamping.
            pwl_calibration_clamp_min=True,
            pwl_calibration_clamp_max=True,
            # Per feature regularization.
            regularizer_configs=[
                configs.RegularizerConfig(name='calib_hessian', l2=1e-4),
            ],
        ),
        configs.FeatureConfig(
            name='fbs',
            # Partial monotonicity: output(0) <= output(1)
            monotonicity=[(0, 1)],
            num_buckets=2,
        ),
        configs.FeatureConfig(
            name='trestbps',
            monotonicity='decreasing',
            pwl_calibration_num_keypoints=5,
            pwl_calibration_input_keypoints='quantiles',
        ),
        configs.FeatureConfig(
            name='thalach',
            monotonicity='decreasing',
            pwl_calibration_num_keypoints=5,
            pwl_calibration_input_keypoints='quantiles',
        ),
        configs.FeatureConfig(
            name='restecg',
            # Partial monotonicity:
            # output(0) <= output(1), output(0) <= output(2)
            monotonicity=[(0, 1), (0, 2)],
            num_buckets=3,
        ),
        configs.FeatureConfig(
            name='exang',
            # Partial monotonicity: output(0) <= output(1)
            monotonicity=[(0, 1)],
            num_buckets=2,
        ),
        configs.FeatureConfig(
            name='oldpeak',
            monotonicity='increasing',
            pwl_calibration_num_keypoints=5,
            pwl_calibration_input_keypoints='quantiles',
        ),
        configs.FeatureConfig(
            name='slope',
            # Partial monotonicity:
            # output(0) <= output(1), output(1) <= output(2)
            monotonicity=[(0, 1), (1, 2)],
            num_buckets=3,
        ),
        configs.FeatureConfig(
            name='ca',
            monotonicity='increasing',
            pwl_calibration_num_keypoints=4,
            pwl_calibration_input_keypoints='quantiles',
        ),
        configs.FeatureConfig(
            name='thal',
            # Partial monotonicity:
            # output(normal) <= output(fixed)
            # output(normal) <= output(reversible)
            monotonicity=[('normal', 'fixed'), ('normal', 'reversible')],
            num_buckets=3,
            # We must specify the vocabulary list in order to later set the
            # monotonicities since we used names and not indices.
            vocabulary_list=thal_vocab_list,
        ),
    ]
    premade_lib.set_categorical_monotonicities(self.heart_feature_configs)

    # This ordering of input features should match the feature configs.
    feature_names = [
        feature_config.name for feature_config in self.heart_feature_configs
    ]
    label_name = 'target'
    self.heart_train_x = [
        heart_train_dict[feature_name] for feature_name in feature_names
    ]
    self.heart_test_x = [
        heart_test_dict[feature_name] for feature_name in feature_names
    ]
    self.heart_train_y = heart_train_dict[label_name]
    self.heart_test_y = heart_test_dict[label_name]

    # Construct feature map for keypoint calculation.
    feature_keypoints = premade_lib.compute_feature_keypoints(
        feature_configs=self.heart_feature_configs, features=heart_train_dict)
    premade_lib.set_feature_keypoints(
        feature_configs=self.heart_feature_configs,
        feature_keypoints=feature_keypoints,
        add_missing_feature_configs=False)

  def _ResetAllBackends(self):
    keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

  class Encoder(json.JSONEncoder):

    def default(self, o):
      if isinstance(o, np.int32):
        return int(o)
      if isinstance(o, np.ndarray):
        return o.tolist()
      return json.JSONEncoder.default(self, o)

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
        'Element 0 for list/tuple 0 for feature categorical monotonicity is '
        'not an index: 0.0'):
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

  def testLatticeEnsembleFromConfig(self):
    model_config = configs.CalibratedLatticeEnsembleConfig(
        feature_configs=copy.deepcopy(feature_configs),
        lattices=[['numerical_1', 'categorical'],
                  ['numerical_2', 'categorical']],
        num_lattices=2,
        lattice_rank=2,
        separate_calibrators=True,
        regularizer_configs=[
            configs.RegularizerConfig('calib_hessian', l2=1e-3),
            configs.RegularizerConfig('torsion', l2=1e-4),
        ],
        output_min=-1.0,
        output_max=1.0,
        output_calibration=True,
        output_calibration_num_keypoints=5,
        output_initialization=[-2., -1., 0., 1., 2.])
    model = premade.CalibratedLatticeEnsemble(model_config)
    loaded_model = premade.CalibratedLatticeEnsemble.from_config(
        model.get_config(), custom_objects=premade.get_custom_objects())
    self.assertEqual(
        json.dumps(model.get_config(), sort_keys=True, cls=self.Encoder),
        json.dumps(loaded_model.get_config(), sort_keys=True, cls=self.Encoder))

  def testLatticeFromConfig(self):
    model_config = configs.CalibratedLatticeConfig(
        feature_configs=copy.deepcopy(feature_configs),
        regularizer_configs=[
            configs.RegularizerConfig('calib_wrinkle', l2=1e-3),
            configs.RegularizerConfig('torsion', l2=1e-3),
        ],
        output_min=0.0,
        output_max=1.0,
        output_calibration=True,
        output_calibration_num_keypoints=5,
        output_initialization=[-2., -1., 0., 1., 2.])
    model = premade.CalibratedLattice(model_config)
    loaded_model = premade.CalibratedLattice.from_config(
        model.get_config(), custom_objects=premade.get_custom_objects())
    self.assertEqual(
        json.dumps(model.get_config(), sort_keys=True, cls=self.Encoder),
        json.dumps(loaded_model.get_config(), sort_keys=True, cls=self.Encoder))

  def testLatticeSimplexFromConfig(self):
    model_config = configs.CalibratedLatticeConfig(
        feature_configs=copy.deepcopy(feature_configs),
        regularizer_configs=[
            configs.RegularizerConfig('calib_wrinkle', l2=1e-3),
            configs.RegularizerConfig('torsion', l2=1e-3),
        ],
        output_min=0.0,
        output_max=1.0,
        interpolation='simplex',
        output_calibration=True,
        output_calibration_num_keypoints=5,
        output_initialization=[-2., -1., 0., 1., 2.])
    model = premade.CalibratedLattice(model_config)
    loaded_model = premade.CalibratedLattice.from_config(
        model.get_config(), custom_objects=premade.get_custom_objects())
    self.assertEqual(
        json.dumps(model.get_config(), sort_keys=True, cls=self.Encoder),
        json.dumps(loaded_model.get_config(), sort_keys=True, cls=self.Encoder))

  def testLinearFromConfig(self):
    model_config = configs.CalibratedLinearConfig(
        feature_configs=copy.deepcopy(feature_configs),
        regularizer_configs=[
            configs.RegularizerConfig('calib_hessian', l2=1e-4),
            configs.RegularizerConfig('torsion', l2=1e-3),
        ],
        use_bias=True,
        output_min=0.0,
        output_max=1.0,
        output_calibration=True,
        output_calibration_num_keypoints=5,
        output_initialization=[-2., -1., 0., 1., 2.])
    model = premade.CalibratedLinear(model_config)
    loaded_model = premade.CalibratedLinear.from_config(
        model.get_config(), custom_objects=premade.get_custom_objects())
    self.assertEqual(
        json.dumps(model.get_config(), sort_keys=True, cls=self.Encoder),
        json.dumps(loaded_model.get_config(), sort_keys=True, cls=self.Encoder))

  def testAggregateFromConfig(self):
    model_config = configs.AggregateFunctionConfig(
        feature_configs=feature_configs,
        regularizer_configs=[
            configs.RegularizerConfig('calib_hessian', l2=1e-4),
            configs.RegularizerConfig('torsion', l2=1e-3),
        ],
        middle_calibration=True,
        middle_monotonicity='increasing',
        output_min=0.0,
        output_max=1.0,
        output_calibration=True,
        output_calibration_num_keypoints=5,
        output_initialization=[-2., -1., 0., 1., 2.])
    model = premade.AggregateFunction(model_config)
    loaded_model = premade.AggregateFunction.from_config(
        model.get_config(), custom_objects=premade.get_custom_objects())
    self.assertEqual(
        json.dumps(model.get_config(), sort_keys=True, cls=self.Encoder),
        json.dumps(loaded_model.get_config(), sort_keys=True, cls=self.Encoder))

  @parameterized.parameters(
      ('hypercube', 'all_vertices', 0, 0.85),
      ('simplex', 'all_vertices', 0, 0.89),
      ('hypercube', 'kronecker_factored', 2, 0.82),
      ('hypercube', 'kronecker_factored', 4, 0.82),
  )
  def testCalibratedLatticeEnsembleCrystals(self, interpolation,
                                            parameterization, num_terms,
                                            expected_minimum_auc):
    # Construct model.
    self._ResetAllBackends()
    crystals_feature_configs = copy.deepcopy(self.heart_feature_configs)
    model_config = configs.CalibratedLatticeEnsembleConfig(
        regularizer_configs=[
            configs.RegularizerConfig(name='torsion', l2=1e-4),
            configs.RegularizerConfig(name='output_calib_hessian', l2=1e-4),
        ],
        feature_configs=crystals_feature_configs,
        lattices='crystals',
        num_lattices=6,
        lattice_rank=5,
        interpolation=interpolation,
        parameterization=parameterization,
        num_terms=num_terms,
        separate_calibrators=True,
        output_calibration=False,
        output_initialization=[-2, 2],
    )
    if parameterization == 'kronecker_factored':
      model_config.regularizer_configs = None
      for feature_config in model_config.feature_configs:
        feature_config.lattice_size = 2
        feature_config.unimodality = 'none'
        feature_config.reflects_trust_in = None
        feature_config.dominates = None
        feature_config.regularizer_configs = None
    # Perform prefitting steps.
    prefitting_model_config = premade_lib.construct_prefitting_model_config(
        model_config)
    prefitting_model = premade.CalibratedLatticeEnsemble(
        prefitting_model_config)
    prefitting_model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=keras.optimizers.legacy.Adam(0.01),
    )
    prefitting_model.fit(
        self.heart_train_x,
        self.heart_train_y,
        batch_size=100,
        epochs=50,
        verbose=False)
    premade_lib.set_crystals_lattice_ensemble(model_config,
                                              prefitting_model_config,
                                              prefitting_model)
    # Construct and train final model
    model = premade.CalibratedLatticeEnsemble(model_config)
    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=keras.metrics.AUC(from_logits=True),
        optimizer=keras.optimizers.legacy.Adam(0.01),
    )
    model.fit(
        self.heart_train_x,
        self.heart_train_y,
        batch_size=100,
        epochs=200,
        verbose=False)
    results = model.evaluate(
        self.heart_test_x, self.heart_test_y, verbose=False)
    logging.info('Calibrated lattice ensemble crystals classifier results:')
    logging.info(results)
    self.assertGreater(results[1], expected_minimum_auc)

  @parameterized.parameters(
      ('hypercube', 'all_vertices', 0, 0.85),
      ('simplex', 'all_vertices', 0, 0.88),
      ('hypercube', 'kronecker_factored', 2, 0.82),
      ('hypercube', 'kronecker_factored', 4, 0.82),
  )
  def testCalibratedLatticeEnsembleRTL(self, interpolation, parameterization,
                                       num_terms, expected_minimum_auc):
    # Construct model.
    self._ResetAllBackends()
    rtl_feature_configs = copy.deepcopy(self.heart_feature_configs)
    for feature_config in rtl_feature_configs:
      feature_config.lattice_size = 2
      feature_config.unimodality = 'none'
      feature_config.reflects_trust_in = None
      feature_config.dominates = None
      feature_config.regularizer_configs = None
    model_config = configs.CalibratedLatticeEnsembleConfig(
        regularizer_configs=[
            configs.RegularizerConfig(name='torsion', l2=1e-4),
            configs.RegularizerConfig(name='output_calib_hessian', l2=1e-4),
        ],
        feature_configs=rtl_feature_configs,
        lattices='rtl_layer',
        num_lattices=6,
        lattice_rank=5,
        interpolation=interpolation,
        parameterization=parameterization,
        num_terms=num_terms,
        separate_calibrators=True,
        output_calibration=False,
        output_initialization=[-2, 2],
    )
    # We must remove all regularization if using 'kronecker_factored'.
    if parameterization == 'kronecker_factored':
      model_config.regularizer_configs = None
    # Construct and train final model
    model = premade.CalibratedLatticeEnsemble(model_config)
    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=keras.metrics.AUC(from_logits=True),
        optimizer=keras.optimizers.legacy.Adam(0.01),
    )
    model.fit(
        self.heart_train_x,
        self.heart_train_y,
        batch_size=100,
        epochs=200,
        verbose=False)
    results = model.evaluate(
        self.heart_test_x, self.heart_test_y, verbose=False)
    logging.info('Calibrated lattice ensemble rtl classifier results:')
    logging.info(results)
    self.assertGreater(results[1], expected_minimum_auc)

  @parameterized.parameters(
      ('hypercube', 'all_vertices', 0, 0.81),
      ('simplex', 'all_vertices', 0, 0.81),
      ('hypercube', 'kronecker_factored', 2, 0.77),
      ('hypercube', 'kronecker_factored', 4, 0.77),
  )
  def testCalibratedLattice(self, interpolation, parameterization, num_terms,
                            expected_minimum_auc):
    # Construct model configuration.
    self._ResetAllBackends()
    lattice_feature_configs = copy.deepcopy(self.heart_feature_configs[:5])
    model_config = configs.CalibratedLatticeConfig(
        feature_configs=lattice_feature_configs,
        interpolation=interpolation,
        parameterization=parameterization,
        num_terms=num_terms,
        regularizer_configs=[
            configs.RegularizerConfig(name='torsion', l2=1e-4),
            configs.RegularizerConfig(name='output_calib_hessian', l2=1e-4),
        ],
        output_calibration=False,
        output_initialization=[-2, 2],
    )
    if parameterization == 'kronecker_factored':
      model_config.regularizer_configs = None
      for feature_config in model_config.feature_configs:
        feature_config.lattice_size = 2
        feature_config.unimodality = 'none'
        feature_config.reflects_trust_in = None
        feature_config.dominates = None
        feature_config.regularizer_configs = None
    # Construct and train final model
    model = premade.CalibratedLattice(model_config)
    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=keras.metrics.AUC(from_logits=True),
        optimizer=keras.optimizers.legacy.Adam(0.01),
    )
    model.fit(
        self.heart_train_x[:5],
        self.heart_train_y,
        batch_size=100,
        epochs=200,
        verbose=False)
    results = model.evaluate(
        self.heart_test_x[:5], self.heart_test_y, verbose=False)
    logging.info('Calibrated lattice classifier results:')
    logging.info(results)
    self.assertGreater(results[1], expected_minimum_auc)

  def testLearnedCalibrationInputKeypoints(self):
    # First let's try a CalibratedLatticeEnsemble
    self._ResetAllBackends()
    learned_keypoints_feature_configs = copy.deepcopy(
        self.heart_feature_configs)
    for feature_config in learned_keypoints_feature_configs:
      feature_config.pwl_calibration_input_keypoints_type = 'learned_interior'
    model_config = configs.CalibratedLatticeEnsembleConfig(
        regularizer_configs=[
            configs.RegularizerConfig(name='torsion', l2=1e-4),
            configs.RegularizerConfig(name='output_calib_hessian', l2=1e-4),
        ],
        feature_configs=learned_keypoints_feature_configs,
        lattices='random',
        num_lattices=6,
        lattice_rank=5,
        interpolation='hypercube',
        separate_calibrators=True,
        output_calibration=True,
        output_calibration_num_keypoints=5,
        output_initialization=[-2., -1., 0., 1., 2.],
        output_calibration_input_keypoints_type='learned_interior',
    )
    premade_lib.set_random_lattice_ensemble(model_config)
    # Construct and train final model
    model = premade.CalibratedLatticeEnsemble(model_config)
    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=keras.metrics.AUC(from_logits=True),
        optimizer=keras.optimizers.legacy.Adam(0.01),
    )
    model.fit(
        self.heart_train_x,
        self.heart_train_y,
        batch_size=100,
        epochs=200,
        verbose=False)
    results = model.evaluate(
        self.heart_test_x, self.heart_test_y, verbose=False)
    logging.info('Calibrated random lattice ensemble classifier results:')
    logging.info(results)
    self.assertGreater(results[1], 0.82)

    # Now let's try a CalibratedLattice
    self._ResetAllBackends()
    model_config = configs.CalibratedLatticeConfig(
        feature_configs=learned_keypoints_feature_configs[:5],
        interpolation='hypercube',
        regularizer_configs=[
            configs.RegularizerConfig(name='torsion', l2=1e-4),
            configs.RegularizerConfig(name='output_calib_hessian', l2=1e-4),
        ],
        output_calibration_num_keypoints=5,
        output_initialization=[-2., -1., 0., 1., 2.],
        output_calibration_input_keypoints_type='learned_interior',
    )
    # Construct and train final model
    model = premade.CalibratedLattice(model_config)
    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=keras.metrics.AUC(from_logits=True),
        optimizer=keras.optimizers.legacy.Adam(0.01),
    )
    model.fit(
        self.heart_train_x[:5],
        self.heart_train_y,
        batch_size=100,
        epochs=200,
        verbose=False)
    results = model.evaluate(
        self.heart_test_x[:5], self.heart_test_y, verbose=False)
    logging.info('Calibrated lattice classifier results:')
    logging.info(results)
    self.assertGreater(results[1], 0.79)

  @parameterized.parameters(
      ('all_vertices', 0),
      ('kronecker_factored', 2),
  )
  def testLatticeEnsembleH5FormatSaveLoad(self, parameterization, num_terms):
    model_config = configs.CalibratedLatticeEnsembleConfig(
        feature_configs=copy.deepcopy(feature_configs),
        lattices=[['numerical_1', 'categorical'],
                  ['numerical_2', 'categorical']],
        num_lattices=2,
        lattice_rank=2,
        parameterization=parameterization,
        num_terms=num_terms,
        separate_calibrators=True,
        regularizer_configs=[
            configs.RegularizerConfig('calib_hessian', l2=1e-3),
            configs.RegularizerConfig('torsion', l2=1e-4),
        ],
        output_min=-1.0,
        output_max=1.0,
        output_calibration=True,
        output_calibration_num_keypoints=5,
        output_initialization=[-2., -1., 0., 1., 2.])
    if parameterization == 'kronecker_factored':
      model_config.regularizer_configs = None
      for feature_config in model_config.feature_configs:
        feature_config.lattice_size = 2
        feature_config.unimodality = 'none'
        feature_config.reflects_trust_in = None
        feature_config.dominates = None
        feature_config.regularizer_configs = None
    model = premade.CalibratedLatticeEnsemble(model_config)
    # Compile and fit model.
    model.compile(loss='mse', optimizer=keras.optimizers.legacy.Adam(0.1))
    model.fit(fake_data['train_xs'], fake_data['train_ys'])
    # Save model using H5 format.
    with tempfile.NamedTemporaryFile(suffix='.h5') as f:
      keras.models.save_model(model, f.name)
      loaded_model = keras.models.load_model(
          f.name, custom_objects=premade.get_custom_objects()
      )
      self.assertAllClose(
          model.predict(fake_data['eval_xs']),
          loaded_model.predict(fake_data['eval_xs']))

  @parameterized.parameters(
      ('all_vertices', 0),
      ('kronecker_factored', 2),
  )
  def testLatticeEnsembleRTLH5FormatSaveLoad(self, parameterization, num_terms):
    rtl_feature_configs = copy.deepcopy(feature_configs)
    for feature_config in rtl_feature_configs:
      feature_config.lattice_size = 2
      feature_config.unimodality = 'none'
      feature_config.reflects_trust_in = None
      feature_config.dominates = None
      feature_config.regularizer_configs = None
    model_config = configs.CalibratedLatticeEnsembleConfig(
        feature_configs=copy.deepcopy(rtl_feature_configs),
        lattices='rtl_layer',
        num_lattices=2,
        lattice_rank=2,
        parameterization=parameterization,
        num_terms=num_terms,
        separate_calibrators=True,
        regularizer_configs=[
            configs.RegularizerConfig('calib_hessian', l2=1e-3),
            configs.RegularizerConfig('torsion', l2=1e-4),
        ],
        output_min=-1.0,
        output_max=1.0,
        output_calibration=True,
        output_calibration_num_keypoints=5,
        output_initialization=[-2., -1., 0., 1., 2.])
    if parameterization == 'kronecker_factored':
      model_config.regularizer_configs = None
    model = premade.CalibratedLatticeEnsemble(model_config)
    # Compile and fit model.
    model.compile(loss='mse', optimizer=keras.optimizers.legacy.Adam(0.1))
    model.fit(fake_data['train_xs'], fake_data['train_ys'])
    # Save model using H5 format.
    with tempfile.NamedTemporaryFile(suffix='.h5') as f:
      keras.models.save_model(model, f.name)
      loaded_model = keras.models.load_model(
          f.name, custom_objects=premade.get_custom_objects()
      )
      self.assertAllClose(
          model.predict(fake_data['eval_xs']),
          loaded_model.predict(fake_data['eval_xs']))

  @parameterized.parameters(
      ('all_vertices', 0),
      ('kronecker_factored', 2),
  )
  def testLatticeH5FormatSaveLoad(self, parameterization, num_terms):
    model_config = configs.CalibratedLatticeConfig(
        feature_configs=copy.deepcopy(feature_configs),
        parameterization=parameterization,
        num_terms=num_terms,
        regularizer_configs=[
            configs.RegularizerConfig('calib_wrinkle', l2=1e-3),
            configs.RegularizerConfig('torsion', l2=1e-3),
        ],
        output_min=0.0,
        output_max=1.0,
        output_calibration=True,
        output_calibration_num_keypoints=5,
        output_initialization=[-2., -1., 0., 1., 2.])
    if parameterization == 'kronecker_factored':
      model_config.regularizer_configs = None
      for feature_config in model_config.feature_configs:
        feature_config.lattice_size = 2
        feature_config.unimodality = 'none'
        feature_config.reflects_trust_in = None
        feature_config.dominates = None
        feature_config.regularizer_configs = None
    model = premade.CalibratedLattice(model_config)
    # Compile and fit model.
    model.compile(loss='mse', optimizer=keras.optimizers.legacy.Adam(0.1))
    model.fit(fake_data['train_xs'], fake_data['train_ys'])
    # Save model using H5 format.
    with tempfile.NamedTemporaryFile(suffix='.h5') as f:
      keras.models.save_model(model, f.name)
      loaded_model = keras.models.load_model(
          f.name, custom_objects=premade.get_custom_objects()
      )
      self.assertAllClose(
          model.predict(fake_data['eval_xs']),
          loaded_model.predict(fake_data['eval_xs']))

  def testLinearH5FormatSaveLoad(self):
    model_config = configs.CalibratedLinearConfig(
        feature_configs=copy.deepcopy(feature_configs),
        regularizer_configs=[
            configs.RegularizerConfig('calib_hessian', l2=1e-4),
            configs.RegularizerConfig('torsion', l2=1e-3),
        ],
        use_bias=True,
        output_min=0.0,
        output_max=1.0,
        output_calibration=True,
        output_calibration_num_keypoints=5,
        output_initialization=[-2., -1., 0., 1., 2.])
    model = premade.CalibratedLinear(model_config)
    # Compile and fit model.
    model.compile(loss='mse', optimizer=keras.optimizers.legacy.Adam(0.1))
    model.fit(fake_data['train_xs'], fake_data['train_ys'])
    # Save model using H5 format.
    with tempfile.NamedTemporaryFile(suffix='.h5') as f:
      keras.models.save_model(model, f.name)
      loaded_model = keras.models.load_model(
          f.name, custom_objects=premade.get_custom_objects()
      )
      self.assertAllClose(
          model.predict(fake_data['eval_xs']),
          loaded_model.predict(fake_data['eval_xs']))

  def testAggregateH5FormatSaveLoad(self):
    model_config = configs.AggregateFunctionConfig(
        feature_configs=feature_configs,
        regularizer_configs=[
            configs.RegularizerConfig('calib_hessian', l2=1e-4),
            configs.RegularizerConfig('torsion', l2=1e-3),
        ],
        middle_calibration=True,
        middle_monotonicity='increasing',
        output_min=0.0,
        output_max=1.0,
        output_calibration=True,
        output_calibration_num_keypoints=5,
        output_initialization=[-2., -1., 0., 1., 2.])
    model = premade.AggregateFunction(model_config)
    # Compile and fit model.
    model.compile(loss='mse', optimizer=keras.optimizers.legacy.Adam(0.1))
    model.fit(fake_data['train_xs'], fake_data['train_ys'])
    # Save model using H5 format.
    with tempfile.NamedTemporaryFile(suffix='.h5') as f:
      # Note: because of naming clashes in the optimizer, we cannot include it
      # when saving in HDF5. The keras team has informed us that we should not
      # push to support this since SavedModel format is the new default and no
      # new HDF5 functionality is desired.
      keras.models.save_model(model, f.name, include_optimizer=False)
      loaded_model = keras.models.load_model(
          f.name, custom_objects=premade.get_custom_objects()
      )
      self.assertAllClose(
          model.predict(fake_data['eval_xs']),
          loaded_model.predict(fake_data['eval_xs']))


if __name__ == '__main__':
  tf.test.main()
