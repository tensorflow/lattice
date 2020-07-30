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
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_lattice.python import configs
from tensorflow_lattice.python import premade
from tensorflow_lattice.python import premade_lib


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


class PremadeTest(tf.test.TestCase):
  """Tests for TFL premade."""

  def setUp(self):
    super(PremadeTest, self).setUp()

    # UCI Statlog (Heart) dataset.
    heart_csv_file = tf.keras.utils.get_file(
        'heart.csv', 'http://storage.googleapis.com/applied-dl/heart.csv')
    heart_df = pd.read_csv(heart_csv_file)
    heart_train_size = int(len(heart_df) * 0.8)
    heart_train_dataframe = heart_df[:heart_train_size]
    heart_test_dataframe = heart_df[heart_train_size:]

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
    # - thal      3 = normal; 6 = fixed defect; 7 = reversable defect
    #
    # This ordering of feature names will be the exact same order that we
    # construct our model to expect.
    self.heart_feature_names = [
        'age', 'sex', 'cp', 'chol', 'fbs', 'trestbps', 'thalach', 'restecg',
        'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    feature_name_indices = {
        name: index for index, name in enumerate(self.heart_feature_names)
    }
    # This is the vocab list and mapping we will use for the 'thal' categorical
    # feature.
    thal_vocab_list = ['normal', 'fixed', 'reversible']
    thal_map = {category: i for i, category in enumerate(thal_vocab_list)}

    # Custom function for converting thal categories to buckets
    def convert_thal_features(thal_features):
      # Note that two examples in the test set are already converted.
      return np.array([
          thal_map[feature] if feature in thal_vocab_list else feature
          for feature in thal_features
      ])

    # Custom function for extracting each feature.
    def extract_features(dataframe, label_name='target'):
      features = []
      for feature_name in self.heart_feature_names:
        if feature_name == 'thal':
          features.append(
              convert_thal_features(
                  dataframe[feature_name].values).astype(float))
        else:
          features.append(dataframe[feature_name].values.astype(float))
      labels = dataframe[label_name].values.astype(float)
      return features, labels

    self.heart_train_x, self.heart_train_y = extract_features(
        heart_train_dataframe)
    self.heart_test_x, self.heart_test_y = extract_features(
        heart_test_dataframe)

    # Let's define our label minimum and maximum.
    self.heart_min_label = float(np.min(self.heart_train_y))
    self.heart_max_label = float(np.max(self.heart_train_y))
    # Our lattice models may have predictions above 1.0 due to numerical errors.
    # We can subtract this small epsilon value from our output_max to make sure
    # we do not predict values outside of our label bound.
    self.numerical_error_epsilon = 1e-5

    def compute_quantiles(features,
                          num_keypoints=10,
                          clip_min=None,
                          clip_max=None,
                          missing_value=None):
      # Clip min and max if desired.
      if clip_min is not None:
        features = np.maximum(features, clip_min)
        features = np.append(features, clip_min)
      if clip_max is not None:
        features = np.minimum(features, clip_max)
        features = np.append(features, clip_max)
      # Make features unique.
      unique_features = np.unique(features)
      # Remove missing values if specified.
      if missing_value is not None:
        unique_features = np.delete(unique_features,
                                    np.where(unique_features == missing_value))
      # Compute and return quantiles over unique non-missing feature values.
      return np.quantile(
          unique_features,
          np.linspace(0., 1., num=num_keypoints),
          interpolation='nearest').astype(float)

    self.heart_feature_configs = [
        configs.FeatureConfig(
            name='age',
            lattice_size=3,
            monotonicity='increasing',
            # We must set the keypoints manually.
            pwl_calibration_num_keypoints=5,
            pwl_calibration_input_keypoints=compute_quantiles(
                self.heart_train_x[feature_name_indices['age']],
                num_keypoints=5,
                clip_max=100),
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
            pwl_calibration_input_keypoints=np.linspace(
                np.min(self.heart_train_x[feature_name_indices['cp']]),
                np.max(self.heart_train_x[feature_name_indices['cp']]),
                num=4),
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
            pwl_calibration_input_keypoints=compute_quantiles(
                self.heart_train_x[feature_name_indices['trestbps']],
                num_keypoints=5),
        ),
        configs.FeatureConfig(
            name='thalach',
            monotonicity='decreasing',
            pwl_calibration_num_keypoints=5,
            pwl_calibration_input_keypoints=compute_quantiles(
                self.heart_train_x[feature_name_indices['thalach']],
                num_keypoints=5),
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
            pwl_calibration_input_keypoints=compute_quantiles(
                self.heart_train_x[feature_name_indices['oldpeak']],
                num_keypoints=5),
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
            pwl_calibration_input_keypoints=compute_quantiles(
                self.heart_train_x[feature_name_indices['ca']],
                num_keypoints=4),
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

  def _ResetAllBackends(self):
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

  class Encoder(json.JSONEncoder):

    def default(self, obj):
      if isinstance(obj, np.int32):
        return int(obj)
      if isinstance(obj, np.ndarray):
        return obj.tolist()
      return json.JSONEncoder.default(self, obj)

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
        output_initialization=[-1.0, 1.0])
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
        output_calibration_num_keypoints=6,
        output_initialization=[0.0, 1.0])
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
        output_calibration_num_keypoints=6,
        output_initialization=[0.0, 1.0])
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
        output_calibration_num_keypoints=6,
        output_initialization=[0.0, 1.0])
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
        output_calibration_num_keypoints=8,
        output_initialization=[0.0, 1.0])
    model = premade.AggregateFunction(model_config)
    loaded_model = premade.AggregateFunction.from_config(
        model.get_config(), custom_objects=premade.get_custom_objects())
    self.assertEqual(
        json.dumps(model.get_config(), sort_keys=True, cls=self.Encoder),
        json.dumps(loaded_model.get_config(), sort_keys=True, cls=self.Encoder))

  def testCalibratedLatticeEnsembleCrystals(self):
    # Construct model.
    self._ResetAllBackends()
    model_config = configs.CalibratedLatticeEnsembleConfig(
        regularizer_configs=[
            configs.RegularizerConfig(name='torsion', l2=1e-4),
            configs.RegularizerConfig(name='output_calib_hessian', l2=1e-4),
        ],
        feature_configs=self.heart_feature_configs,
        lattices='crystals',
        num_lattices=6,
        lattice_rank=5,
        separate_calibrators=True,
        output_calibration=False,
        output_min=self.heart_min_label,
        output_max=self.heart_max_label - self.numerical_error_epsilon,
        output_initialization=[self.heart_min_label, self.heart_max_label],
    )
    # Perform prefitting steps.
    prefitting_model_config = premade_lib.construct_prefitting_model_config(
        model_config)
    prefitting_model = premade.CalibratedLatticeEnsemble(
        prefitting_model_config)
    prefitting_model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(0.01))
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
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=tf.keras.metrics.AUC(),
        optimizer=tf.keras.optimizers.Adam(0.01))
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
    self.assertGreater(results[1], 0.85)

  def testCalibratedLatticeEnsembleRTL(self):
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
        separate_calibrators=True,
        output_calibration=False,
        output_min=self.heart_min_label,
        output_max=self.heart_max_label - self.numerical_error_epsilon,
        output_initialization=[self.heart_min_label, self.heart_max_label],
    )
    # Construct and train final model
    model = premade.CalibratedLatticeEnsemble(model_config)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=tf.keras.metrics.AUC(),
        optimizer=tf.keras.optimizers.Adam(0.01))
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
    self.assertGreater(results[1], 0.85)

  def testLatticeEnsembleH5FormatSaveLoad(self):
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
        output_initialization=[-1.0, 1.0])
    model = premade.CalibratedLatticeEnsemble(model_config)
    # Compile and fit model.
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.1))
    model.fit(fake_data['train_xs'], fake_data['train_ys'])
    # Save model using H5 format.
    with tempfile.NamedTemporaryFile(suffix='.h5') as f:
      tf.keras.models.save_model(model, f.name)
      loaded_model = tf.keras.models.load_model(
          f.name, custom_objects=premade.get_custom_objects())
      self.assertAllClose(
          model.predict(fake_data['eval_xs']),
          loaded_model.predict(fake_data['eval_xs']))

  def testLatticeEnsembleRTLH5FormatSaveLoad(self):
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
        separate_calibrators=True,
        regularizer_configs=[
            configs.RegularizerConfig('calib_hessian', l2=1e-3),
            configs.RegularizerConfig('torsion', l2=1e-4),
        ],
        output_min=-1.0,
        output_max=1.0,
        output_calibration=True,
        output_calibration_num_keypoints=5,
        output_initialization=[-1.0, 1.0])
    model = premade.CalibratedLatticeEnsemble(model_config)
    # Compile and fit model.
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.1))
    model.fit(fake_data['train_xs'], fake_data['train_ys'])
    # Save model using H5 format.
    with tempfile.NamedTemporaryFile(suffix='.h5') as f:
      tf.keras.models.save_model(model, f.name)
      loaded_model = tf.keras.models.load_model(
          f.name, custom_objects=premade.get_custom_objects())
      self.assertAllClose(
          model.predict(fake_data['eval_xs']),
          loaded_model.predict(fake_data['eval_xs']))

  def testLatticeH5FormatSaveLoad(self):
    model_config = configs.CalibratedLatticeConfig(
        feature_configs=copy.deepcopy(feature_configs),
        regularizer_configs=[
            configs.RegularizerConfig('calib_wrinkle', l2=1e-3),
            configs.RegularizerConfig('torsion', l2=1e-3),
        ],
        output_min=0.0,
        output_max=1.0,
        output_calibration=True,
        output_calibration_num_keypoints=6,
        output_initialization=[0.0, 1.0])
    model = premade.CalibratedLattice(model_config)
    # Compile and fit model.
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.1))
    model.fit(fake_data['train_xs'], fake_data['train_ys'])
    # Save model using H5 format.
    with tempfile.NamedTemporaryFile(suffix='.h5') as f:
      tf.keras.models.save_model(model, f.name)
      loaded_model = tf.keras.models.load_model(
          f.name, custom_objects=premade.get_custom_objects())
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
        output_calibration_num_keypoints=6,
        output_initialization=[0.0, 1.0])
    model = premade.CalibratedLinear(model_config)
    # Compile and fit model.
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.1))
    model.fit(fake_data['train_xs'], fake_data['train_ys'])
    # Save model using H5 format.
    with tempfile.NamedTemporaryFile(suffix='.h5') as f:
      tf.keras.models.save_model(model, f.name)
      loaded_model = tf.keras.models.load_model(
          f.name, custom_objects=premade.get_custom_objects())
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
        output_calibration_num_keypoints=8,
        output_initialization=[0.0, 1.0])
    model = premade.AggregateFunction(model_config)
    # Compile and fit model.
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.1))
    model.fit(fake_data['train_xs'], fake_data['train_ys'])
    # Save model using H5 format.
    with tempfile.NamedTemporaryFile(suffix='.h5') as f:
      # Note: because of naming clashes in the optimizer, we cannot include it
      # when saving in HDF5. The keras team has informed us that we should not
      # push to support this since SavedModel format is the new default and no
      # new HDF5 functionality is desired.
      tf.keras.models.save_model(model, f.name, include_optimizer=False)
      loaded_model = tf.keras.models.load_model(
          f.name, custom_objects=premade.get_custom_objects())
      self.assertAllClose(
          model.predict(fake_data['eval_xs']),
          loaded_model.predict(fake_data['eval_xs']))


if __name__ == '__main__':
  tf.test.main()
