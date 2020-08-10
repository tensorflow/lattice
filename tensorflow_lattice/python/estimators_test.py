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
"""Tests TFL canned estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from absl import logging
from absl.testing import parameterized
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
import tensorflow as tf
from tensorflow import feature_column as fc
from tensorflow_lattice.python import configs
from tensorflow_lattice.python import estimators
from tensorflow_lattice.python import model_info
from tensorflow_estimator.python.estimator.head import regression_head


class CannedEstimatorsTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(CannedEstimatorsTest, self).setUp()
    self.eps = 0.001

    # UCI Statlog (Heart) dataset.
    heart_csv_file = tf.keras.utils.get_file(
        'heart.csv', 'http://storage.googleapis.com/applied-dl/heart.csv')
    heart_df = pd.read_csv(heart_csv_file)
    heart_target = heart_df.pop('target')
    heart_train_size = int(len(heart_df) * 0.8)
    self.heart_train_x = heart_df[:heart_train_size]
    self.heart_train_y = heart_target[:heart_train_size]
    self.heart_test_x = heart_df[heart_train_size:]
    self.heart_test_y = heart_target[heart_train_size:]

    # Feature columns.
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
    self.heart_feature_columns = [
        fc.numeric_column('age', default_value=-1),
        fc.categorical_column_with_vocabulary_list('sex', [0, 1]),
        fc.numeric_column('cp'),
        fc.numeric_column('trestbps', default_value=-1),
        fc.numeric_column('chol'),
        fc.categorical_column_with_vocabulary_list('fbs', [0, 1]),
        fc.categorical_column_with_vocabulary_list('restecg', [0, 1, 2]),
        fc.numeric_column('thalach'),
        fc.categorical_column_with_vocabulary_list('exang', [0, 1]),
        fc.numeric_column('oldpeak'),
        fc.categorical_column_with_vocabulary_list('slope', [0, 1, 2]),
        fc.numeric_column('ca'),
        fc.categorical_column_with_vocabulary_list(
            'thal', ['normal', 'fixed', 'reversible']),
    ]

    # Feature configs. Each model can pick and choose which features to use.
    self.heart_feature_configs = [
        configs.FeatureConfig(
            name='age',
            lattice_size=3,
            pwl_calibration_num_keypoints=5,
            monotonicity=1,
            pwl_calibration_clip_max=100,
        ),
        configs.FeatureConfig(
            name='cp',
            pwl_calibration_num_keypoints=4,
            pwl_calibration_input_keypoints='uniform',
            monotonicity='increasing',
        ),
        configs.FeatureConfig(
            name='chol',
            pwl_calibration_input_keypoints=[126.0, 210.0, 247.0, 286.0, 564.0],
            monotonicity=1,
            pwl_calibration_clip_min=130,
            pwl_calibration_clamp_min=True,
            pwl_calibration_clamp_max=True,
            regularizer_configs=[
                configs.RegularizerConfig(name='calib_hessian', l2=1e-4),
            ],
        ),
        configs.FeatureConfig(
            name='fbs',
            monotonicity=[(0, 1)],
        ),
        configs.FeatureConfig(
            name='trestbps',
            pwl_calibration_num_keypoints=5,
            monotonicity='decreasing',
        ),
        configs.FeatureConfig(
            name='thalach',
            pwl_calibration_num_keypoints=5,
            monotonicity=-1,
        ),
        configs.FeatureConfig(
            name='restecg',
            monotonicity=[(0, 1), (0, 2)],
        ),
        configs.FeatureConfig(
            name='exang',
            monotonicity=[(0, 1)],
        ),
        configs.FeatureConfig(
            name='oldpeak',
            pwl_calibration_num_keypoints=5,
            monotonicity=1,
        ),
        configs.FeatureConfig(
            name='slope',
            monotonicity=[(0, 1), (1, 2)],
        ),
        configs.FeatureConfig(
            name='ca',
            pwl_calibration_num_keypoints=4,
            monotonicity='increasing',
        ),
        configs.FeatureConfig(
            name='thal',
            monotonicity=[('normal', 'fixed'), ('normal', 'reversible')],
        ),
    ]

    # UCI Boston dataset.
    boston_dataset = load_boston()
    boston_df = pd.DataFrame(
        boston_dataset.data, columns=boston_dataset.feature_names)
    boston_df['CHAS'] = boston_df['CHAS'].astype(np.int32)
    boston_target = pd.Series(boston_dataset.target)
    boston_train_size = int(len(boston_df) * 0.8)
    self.boston_train_x = boston_df[:boston_train_size]
    self.boston_train_y = boston_target[:boston_train_size]
    self.boston_test_x = boston_df[boston_train_size:]
    self.boston_test_y = boston_target[boston_train_size:]

    # Feature columns.
    # - CRIM     per capita crime rate by town
    # - ZN       proportion of residential land zoned for lots over 25,000 sq.ft
    # - INDUS    proportion of non-retail business acres per town
    # - CHAS     Charles River dummy variable (= 1 if tract bounds river)
    # - NOX      nitric oxides concentration (parts per 10 million)
    # - RM       average number of rooms per dwelling
    # - AGE      proportion of owner-occupied units built prior to 1940
    # - DIS      weighted distances to five Boston employment centres
    # - RAD      index of accessibility to radial highways
    # - TAX      full-value property-tax rate per $10,000
    # - PTRATIO  pupil-teacher ratio by town
    # - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    # - LSTAT    % lower status of the population
    # - Target   Median value of owner-occupied homes in $1000's
    self.boston_feature_columns = [
        fc.numeric_column('CRIM'),
        fc.numeric_column('ZN'),
        fc.numeric_column('INDUS'),
        fc.categorical_column_with_vocabulary_list('CHAS', [0, 1]),
        fc.numeric_column('NOX'),
        fc.numeric_column('RM'),
        fc.numeric_column('AGE'),
        fc.numeric_column('DIS'),
        fc.numeric_column('RAD'),
        fc.numeric_column('TAX'),
        fc.numeric_column('PTRATIO'),
        fc.numeric_column('B'),
        fc.numeric_column('LSTAT'),
    ]

    # Feature configs. Each model can pick and choose which features to use.
    self.boston_feature_configs = [
        configs.FeatureConfig(
            name='CRIM',
            lattice_size=3,
            monotonicity=-1,
            pwl_calibration_convexity=1,
        ),
        configs.FeatureConfig(
            name='ZN',
            pwl_calibration_input_keypoints=[0.0, 25.0, 50.0, 75.0, 100.0],
            monotonicity=1,
            reflects_trust_in=[
                configs.TrustConfig(feature_name='RM', trust_type='trapezoid'),
            ],
        ),
        configs.FeatureConfig(
            name='INDUS',
            pwl_calibration_input_keypoints='uniform',
            pwl_calibration_always_monotonic=False,
            reflects_trust_in=[
                configs.TrustConfig(
                    feature_name='RM',
                    trust_type='edgeworth',
                    direction='negative'),
            ],
            regularizer_configs=[
                configs.RegularizerConfig(name='calib_wrinkle', l2=1e-4),
            ],
        ),
        configs.FeatureConfig(name='CHAS',),
        configs.FeatureConfig(name='NOX',),
        configs.FeatureConfig(
            name='RM',
            monotonicity='increasing',
            pwl_calibration_convexity='concave',
        ),
        configs.FeatureConfig(
            name='AGE',
            monotonicity=-1,
        ),
        configs.FeatureConfig(
            name='DIS',
            lattice_size=3,
            unimodality=1,
        ),
        configs.FeatureConfig(name='RAD',),
        configs.FeatureConfig(name='TAX',),
        configs.FeatureConfig(
            name='PTRATIO',
            monotonicity='decreasing',
        ),
        configs.FeatureConfig(name='B',),
        configs.FeatureConfig(
            name='LSTAT',
            monotonicity=-1,
            dominates=[
                configs.DominanceConfig(
                    feature_name='AGE', dominance_type='monotonic'),
            ],
        ),
    ]

  def _ResetAllBackends(self):
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

  def _GetInputFn(self, x, y, num_epochs=1, batch_size=100):
    return tf.compat.v1.estimator.inputs.pandas_input_fn(
        x=x,
        y=y,
        batch_size=batch_size,
        shuffle=False,
        num_epochs=num_epochs,
        num_threads=1)

  def _GetHeartTrainInputFn(self, **kwargs):
    return self._GetInputFn(self.heart_train_x, self.heart_train_y, **kwargs)

  def _GetHeartTestInputFn(self, **kwargs):
    return self._GetInputFn(
        self.heart_test_x, self.heart_test_y, num_epochs=1, **kwargs)

  def _GetBostonTrainInputFn(self, **kwargs):
    return self._GetInputFn(self.boston_train_x, self.boston_train_y, **kwargs)

  def _GetBostonTestInputFn(self, **kwargs):
    return self._GetInputFn(
        self.boston_test_x, self.boston_test_y, num_epochs=1, **kwargs)

  @parameterized.parameters(
      ([
          'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
          'exang', 'oldpeak', 'slope', 'ca', 'thal'
      ], [['sex', 'oldpeak'], ['fbs', 'thalach'], ['thalach', 'thal'],
          ['cp', 'trestbps'], ['age', 'ca', 'chol']
         ], None, None, False, True, 0.8),
      ([
          'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
          'exang', 'oldpeak', 'slope', 'ca', 'thal'
      ], 'random', 6, 5, True, False, 0.85),
      ([
          'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
          'exang', 'oldpeak', 'slope', 'ca', 'thal'
      ], 'crystals', 6, 5, True, False, 0.85),
      ([
          'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
          'exang', 'oldpeak', 'slope', 'ca', 'thal'
      ], 'rtl_layer', 6, 5, True, False, 0.85),
  )
  def testCalibratedLatticeEnsembleClassifier(self, feature_names, lattices,
                                              num_lattices, lattice_rank,
                                              separate_calibrators,
                                              output_calibration, auc):
    self._ResetAllBackends()
    feature_columns = [
        feature_column for feature_column in self.heart_feature_columns
        if feature_column.name in feature_names
    ]
    feature_configs = [
        feature_config for feature_config in self.heart_feature_configs
        if feature_config.name in feature_names
    ]
    if lattices == 'rtl_layer':
      # RTL Layer only supports monotonicity and bound constraints.
      feature_configs = copy.deepcopy(feature_configs)
      for feature_config in feature_configs:
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
        feature_configs=feature_configs,
        lattices=lattices,
        num_lattices=num_lattices,
        lattice_rank=lattice_rank,
        separate_calibrators=separate_calibrators,
        output_calibration=output_calibration,
    )
    estimator = estimators.CannedClassifier(
        feature_columns=feature_columns,
        model_config=model_config,
        feature_analysis_input_fn=self._GetHeartTrainInputFn(num_epochs=1),
        prefitting_input_fn=self._GetHeartTrainInputFn(num_epochs=50),
        optimizer=tf.keras.optimizers.Adam(0.01),
        prefitting_optimizer=tf.keras.optimizers.Adam(0.01))
    estimator.train(input_fn=self._GetHeartTrainInputFn(num_epochs=200))
    results = estimator.evaluate(input_fn=self._GetHeartTestInputFn())
    logging.info('Calibrated lattice ensemble classifier results:')
    logging.info(results)
    self.assertGreater(results['auc'], auc)

  @parameterized.parameters(
      (['age', 'sex', 'fbs', 'restecg', 'ca', 'thal'], False, 0.75),
      (['age', 'cp', 'chol', 'slope', 'ca', 'thal'], False, 0.8),
      (['trestbps', 'thalach', 'exang', 'oldpeak', 'thal'], True, 0.8),
  )
  def testCalibratedLatticeClassifier(self, feature_names, output_calibration,
                                      auc):
    self._ResetAllBackends()
    feature_columns = [
        feature_column for feature_column in self.heart_feature_columns
        if feature_column.name in feature_names
    ]
    feature_configs = [
        feature_config for feature_config in self.heart_feature_configs
        if feature_config.name in feature_names
    ]
    model_config = configs.CalibratedLatticeConfig(
        regularizer_configs=[
            configs.RegularizerConfig(name='torsion', l2=1e-4),
            configs.RegularizerConfig(name='output_calib_hessian', l2=1e-4),
        ],
        output_calibration=output_calibration,
        feature_configs=feature_configs)
    estimator = estimators.CannedClassifier(
        feature_columns=feature_columns,
        model_config=model_config,
        feature_analysis_input_fn=self._GetHeartTrainInputFn(num_epochs=1),
        optimizer=tf.keras.optimizers.Adam(0.01))
    estimator.train(input_fn=self._GetHeartTrainInputFn(num_epochs=200))
    results = estimator.evaluate(input_fn=self._GetHeartTestInputFn())
    logging.info('Calibrated lattice classifier results:')
    logging.info(results)
    self.assertGreater(results['auc'], auc)

  @parameterized.parameters(
      (['age', 'sex', 'fbs', 'restecg', 'ca', 'thal'], False, False, 0.7),
      ([
          'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
          'exang', 'oldpeak', 'slope', 'ca', 'thal'
      ], True, True, 0.8),
  )
  def testCalibratedLinearClassifier(self, feature_names, output_calibration,
                                     use_bias, auc):
    self._ResetAllBackends()
    feature_columns = [
        feature_column for feature_column in self.heart_feature_columns
        if feature_column.name in feature_names
    ]
    feature_configs = [
        feature_config for feature_config in self.heart_feature_configs
        if feature_config.name in feature_names
    ]
    model_config = configs.CalibratedLinearConfig(
        use_bias=use_bias,
        regularizer_configs=[
            configs.RegularizerConfig(name='output_calib_hessian', l2=1e-4),
        ],
        output_calibration=output_calibration,
        feature_configs=feature_configs)
    estimator = estimators.CannedClassifier(
        feature_columns=feature_columns,
        model_config=model_config,
        feature_analysis_input_fn=self._GetHeartTrainInputFn(num_epochs=1),
        optimizer=tf.keras.optimizers.Adam(0.01))
    estimator.train(input_fn=self._GetHeartTrainInputFn(num_epochs=200))
    results = estimator.evaluate(input_fn=self._GetHeartTestInputFn())
    logging.info('Calibrated linear classifier results:')
    logging.info(results)
    self.assertGreater(results['auc'], auc)

  @parameterized.parameters(
      ([
          'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
          'TAX', 'PTRATIO', 'B', 'LSTAT'
      ], [['CRIM', 'ZN', 'RAD', 'DIS'], ['PTRATIO', 'LSTAT', 'ZN', 'RM'],
          ['AGE', 'NOX', 'B'], ['INDUS', 'NOX', 'PTRATIO'], ['TAX', 'CHAS'],
          ['CRIM', 'INDUS', 'AGE', 'RM', 'CHAS']
         ], None, None, False, True, 60.0),
      ([
          'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
          'TAX', 'PTRATIO', 'B', 'LSTAT'
      ], 'random', 6, 5, True, False, 50.0),
      ([
          'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
          'TAX', 'PTRATIO', 'B', 'LSTAT'
      ], 'crystals', 6, 5, True, False, 50.0),
      ([
          'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
          'TAX', 'PTRATIO', 'B', 'LSTAT'
      ], 'rtl_layer', 6, 5, True, False, 50.0),
  )
  def testCalibratedLatticeEnsembleRegressor(self, feature_names, lattices,
                                             num_lattices, lattice_rank,
                                             separate_calibrators,
                                             output_calibration, average_loss):
    self._ResetAllBackends()
    feature_columns = [
        feature_column for feature_column in self.boston_feature_columns
        if feature_column.name in feature_names
    ]
    feature_configs = [
        feature_config for feature_config in self.boston_feature_configs
        if feature_config.name in feature_names
    ]
    if lattices == 'rtl_layer':
      # RTL Layer only supports monotonicity and bound constraints.
      feature_configs = copy.deepcopy(feature_configs)
      for feature_config in feature_configs:
        feature_config.lattice_size = 2
        feature_config.unimodality = 'none'
        feature_config.reflects_trust_in = None
        feature_config.dominates = None
        feature_config.regularizer_configs = None
    model_config = configs.CalibratedLatticeEnsembleConfig(
        regularizer_configs=[
            configs.RegularizerConfig(name='torsion', l2=1e-5),
            configs.RegularizerConfig(name='output_calib_hessian', l2=1e-5),
        ],
        feature_configs=feature_configs,
        lattices=lattices,
        num_lattices=num_lattices,
        lattice_rank=lattice_rank,
        separate_calibrators=separate_calibrators,
        output_calibration=output_calibration,
    )
    estimator = estimators.CannedRegressor(
        feature_columns=feature_columns,
        model_config=model_config,
        feature_analysis_input_fn=self._GetBostonTrainInputFn(num_epochs=1),
        prefitting_input_fn=self._GetBostonTrainInputFn(num_epochs=50),
        optimizer=tf.keras.optimizers.Adam(0.05),
        prefitting_optimizer=tf.keras.optimizers.Adam(0.05))
    estimator.train(input_fn=self._GetBostonTrainInputFn(num_epochs=200))
    results = estimator.evaluate(input_fn=self._GetBostonTestInputFn())
    logging.info('Calibrated lattice ensemble regressor results:')
    logging.info(results)
    self.assertLess(results['average_loss'], average_loss)

  @parameterized.parameters(
      (['CRIM', 'ZN', 'RM', 'DIS', 'PTRATIO', 'LSTAT'], False, 40.0),
      (['CRIM', 'INDUS', 'CHAS', 'NOX', 'AGE', 'RAD', 'TAX', 'B'], True, 40.0),
      (['CRIM', 'INDUS', 'LSTAT', 'NOX', 'AGE', 'RAD', 'TAX', 'B'], True, 40.0),
  )
  def testCalibratedLatticeRegressor(self, feature_names, output_calibration,
                                     average_loss):
    self._ResetAllBackends()
    feature_columns = [
        feature_column for feature_column in self.boston_feature_columns
        if feature_column.name in feature_names
    ]
    feature_configs = [
        feature_config for feature_config in self.boston_feature_configs
        if feature_config.name in feature_names
    ]
    model_config = configs.CalibratedLinearConfig(
        regularizer_configs=[
            configs.RegularizerConfig(name='torsion', l2=1e-4),
            configs.RegularizerConfig(name='output_calib_hessian', l2=1e-4),
        ],
        output_calibration=output_calibration,
        feature_configs=feature_configs)
    estimator = estimators.CannedRegressor(
        feature_columns=feature_columns,
        model_config=model_config,
        feature_analysis_input_fn=self._GetBostonTrainInputFn(num_epochs=1),
        optimizer=tf.keras.optimizers.Adam(0.01))
    estimator.train(input_fn=self._GetBostonTrainInputFn(num_epochs=200))
    results = estimator.evaluate(input_fn=self._GetBostonTestInputFn())
    logging.info('Calibrated lattice regressor results:')
    logging.info(results)
    self.assertLess(results['average_loss'], average_loss)

  @parameterized.parameters(
      (['CRIM', 'ZN', 'RM', 'DIS', 'PTRATIO', 'LSTAT'], False, False, 40.0),
      ([
          'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
          'TAX', 'PTRATIO', 'B', 'LSTAT'
      ], True, True, 40.0),
  )
  def testCalibratedLinearRegressor(self, feature_names, output_calibration,
                                    use_bias, average_loss):
    self._ResetAllBackends()
    feature_columns = [
        feature_column for feature_column in self.boston_feature_columns
        if feature_column.name in feature_names
    ]
    feature_configs = [
        feature_config for feature_config in self.boston_feature_configs
        if feature_config.name in feature_names
    ]
    model_config = configs.CalibratedLinearConfig(
        use_bias=use_bias,
        regularizer_configs=[
            configs.RegularizerConfig(name='output_calib_hessian', l2=1e-4),
        ],
        output_calibration=output_calibration,
        feature_configs=feature_configs)
    estimator = estimators.CannedRegressor(
        feature_columns=feature_columns,
        model_config=model_config,
        feature_analysis_input_fn=self._GetBostonTrainInputFn(num_epochs=1),
        optimizer=tf.keras.optimizers.Adam(0.01))
    estimator.train(input_fn=self._GetBostonTrainInputFn(num_epochs=200))
    results = estimator.evaluate(input_fn=self._GetBostonTestInputFn())
    logging.info('Calibrated linear regressor results:')
    logging.info(results)
    self.assertLess(results['average_loss'], average_loss)

  @parameterized.parameters(
      (['CRIM', 'ZN', 'RM', 'DIS', 'PTRATIO', 'LSTAT'], False, False, 40.0),
      ([
          'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
          'TAX', 'PTRATIO', 'B', 'LSTAT'
      ], True, True, 40.0),
  )
  def testCalibratedLinearEstimator(self, feature_names, output_calibration,
                                    use_bias, average_loss):
    self._ResetAllBackends()
    feature_columns = [
        feature_column for feature_column in self.boston_feature_columns
        if feature_column.name in feature_names
    ]
    feature_configs = [
        feature_config for feature_config in self.boston_feature_configs
        if feature_config.name in feature_names
    ]
    model_config = configs.CalibratedLinearConfig(
        use_bias=use_bias,
        regularizer_configs=[
            configs.RegularizerConfig(name='output_calib_hessian', l2=1e-4),
        ],
        output_calibration=output_calibration,
        feature_configs=feature_configs)
    estimator = estimators.CannedEstimator(
        head=regression_head.RegressionHead(),
        feature_columns=feature_columns,
        model_config=model_config,
        feature_analysis_input_fn=self._GetBostonTrainInputFn(num_epochs=1),
        optimizer=tf.keras.optimizers.Adam(0.01))
    estimator.train(input_fn=self._GetBostonTrainInputFn(num_epochs=200))
    results = estimator.evaluate(input_fn=self._GetBostonTestInputFn())
    logging.info('Calibrated linear regressor results:')
    logging.info(results)
    self.assertLess(results['average_loss'], average_loss)

  @parameterized.parameters(
      ('random', 5, 6, False, True),
      ('random', 4, 5, True, False),
      ('rtl_layer', 5, 6, False, True),
      ('rtl_layer', 4, 5, True, False),
  )
  def testCalibratedLatticeEnsembleModelInfo(self, lattices, num_lattices,
                                             lattice_rank, separate_calibrators,
                                             output_calibration):
    self._ResetAllBackends()
    feature_configs = copy.deepcopy(self.heart_feature_configs)
    if lattices == 'rtl_layer':
      # RTL Layer only supports monotonicity and bound constraints.
      for feature_config in feature_configs:
        feature_config.lattice_size = 2
        feature_config.unimodality = 'none'
        feature_config.reflects_trust_in = None
        feature_config.dominates = None
        feature_config.regularizer_configs = None
    model_config = configs.CalibratedLatticeEnsembleConfig(
        feature_configs=feature_configs,
        lattices=lattices,
        num_lattices=num_lattices,
        lattice_rank=lattice_rank,
        separate_calibrators=separate_calibrators,
        output_calibration=output_calibration,
    )
    estimator = estimators.CannedClassifier(
        feature_columns=self.heart_feature_columns,
        model_config=model_config,
        feature_analysis_input_fn=self._GetHeartTrainInputFn(num_epochs=1),
        prefitting_input_fn=self._GetHeartTrainInputFn(num_epochs=5),
        optimizer=tf.keras.optimizers.Adam(0.01),
        prefitting_optimizer=tf.keras.optimizers.Adam(0.01))
    estimator.train(input_fn=self._GetHeartTrainInputFn(num_epochs=20))

    # Serving input fn is used to create saved models.
    serving_input_fn = (
        tf.estimator.export.build_parsing_serving_input_receiver_fn(
            feature_spec=fc.make_parse_example_spec(self.heart_feature_columns))
    )
    saved_model_path = estimator.export_saved_model(estimator.model_dir,
                                                    serving_input_fn)
    logging.info('Model exported to %s', saved_model_path)
    model = estimators.get_model_graph(saved_model_path)

    expected_num_nodes = (
        len(self.heart_feature_columns) +  # Input features
        num_lattices +  # One lattice per submodel
        1 +  # Averaging submodels
        int(output_calibration))  # Output calibration
    if separate_calibrators:
      expected_num_nodes += num_lattices * lattice_rank
    else:
      expected_num_nodes += len(self.heart_feature_columns)

    self.assertLen(model.nodes, expected_num_nodes)

  @parameterized.parameters(
      (['ZN', 'INDUS', 'RM'], 'random', 3, 1, [['ZN', 'RM'], ['RM'], ['INDUS']
                                              ]),
      (['ZN', 'INDUS', 'RM'], 'crystals', 3, 1, [['RM'], ['INDUS'],
                                                 ['ZN', 'RM']]),
      (['RM', 'LSTAT', 'AGE'], 'crystals', 3, 1, [['LSTAT'], ['LSTAT', 'AGE'],
                                                  ['RM']]),
  )
  def testCalibratedLatticeEnsembleFix2dConstraintViolations(
      self, feature_names, lattices, num_lattices, lattice_rank,
      expected_lattices):
    self._ResetAllBackends()
    feature_columns = [
        feature_column for feature_column in self.boston_feature_columns
        if feature_column.name in feature_names
    ]
    feature_configs = [
        feature_config for feature_config in self.boston_feature_configs
        if feature_config.name in feature_names
    ]

    model_config = configs.CalibratedLatticeEnsembleConfig(
        feature_configs=feature_configs,
        lattices=lattices,
        num_lattices=num_lattices,
        lattice_rank=lattice_rank,
    )
    estimator = estimators.CannedRegressor(
        feature_columns=feature_columns,
        model_config=model_config,
        feature_analysis_input_fn=self._GetBostonTrainInputFn(num_epochs=1),
        prefitting_input_fn=self._GetBostonTrainInputFn(num_epochs=50),
        optimizer=tf.keras.optimizers.Adam(0.05),
        prefitting_optimizer=tf.keras.optimizers.Adam(0.05))
    estimator.train(input_fn=self._GetBostonTrainInputFn(num_epochs=200))

    # Serving input fn is used to create saved models.
    serving_input_fn = (
        tf.estimator.export.build_parsing_serving_input_receiver_fn(
            feature_spec=fc.make_parse_example_spec(feature_columns)))
    saved_model_path = estimator.export_saved_model(estimator.model_dir,
                                                    serving_input_fn)
    logging.info('Model exported to %s', saved_model_path)
    model = estimators.get_model_graph(saved_model_path)
    lattices = []
    for node in model.nodes:
      if isinstance(node, model_info.LatticeNode):
        lattices.append(
            [input_node.input_node.name for input_node in node.input_nodes])

    self.assertLen(lattices, len(expected_lattices))
    for lattice, expected_lattice in zip(lattices, expected_lattices):
      self.assertCountEqual(lattice, expected_lattice)

  @parameterized.parameters(
      ('linear', True),
      ('lattice', False),
  )
  def testCalibratedModelInfo(self, model_type, output_calibration):
    self._ResetAllBackends()
    if model_type == 'linear':
      model_config = configs.CalibratedLinearConfig(
          feature_configs=self.heart_feature_configs,
          output_calibration=output_calibration,
      )
    else:
      model_config = configs.CalibratedLatticeConfig(
          feature_configs=self.heart_feature_configs,
          output_calibration=output_calibration,
      )
    estimator = estimators.CannedClassifier(
        feature_columns=self.heart_feature_columns,
        model_config=model_config,
        feature_analysis_input_fn=self._GetHeartTrainInputFn(num_epochs=1),
        prefitting_input_fn=self._GetHeartTrainInputFn(num_epochs=5),
        optimizer=tf.keras.optimizers.Adam(0.01),
        prefitting_optimizer=tf.keras.optimizers.Adam(0.01))
    estimator.train(input_fn=self._GetHeartTrainInputFn(num_epochs=20))

    # Serving input fn is used to create saved models.
    serving_input_fn = (
        tf.estimator.export.build_parsing_serving_input_receiver_fn(
            feature_spec=fc.make_parse_example_spec(self.heart_feature_columns))
    )
    saved_model_path = estimator.export_saved_model(estimator.model_dir,
                                                    serving_input_fn)
    logging.info('Model exported to %s', saved_model_path)
    model = estimators.get_model_graph(saved_model_path)

    expected_num_nodes = (
        2 * len(self.heart_feature_columns) +  # Input features and calibration
        1 +  # Linear or lattice layer
        int(output_calibration))  # Output calibration

    self.assertLen(model.nodes, expected_num_nodes)


if __name__ == '__main__':
  tf.test.main()
