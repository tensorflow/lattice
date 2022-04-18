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

# Lint as: python3
"""Example usage of TFL canned estimators.

This example trains several TFL canned estimators on the UCI heart dataset.

Example usage:
canned_estimators_uci_heart --config_updates=feature__age__lattice_size=4
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from absl import app
from absl import flags
import pandas as pd
import tensorflow as tf
from tensorflow import estimator as tf_estimator
from tensorflow import feature_column as fc
from tensorflow.compat.v1 import estimator as tf_compat_v1_estimator
from tensorflow_lattice import configs
from tensorflow_lattice import estimators

FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('num_epochs', 50, 'Number of training epoch.')
flags.DEFINE_integer('prefitting_num_epochs', 10, 'Prefitting epochs.')
flags.DEFINE_list(
    'config_updates', '',
    'Comma separated list of updates to model configs in name=value format.'
    'See tfl.configs.apply_updates().')


def main(_):
  # Parse configs updates from command line flags.
  config_updates = []
  for update in FLAGS.config_updates:
    config_updates.extend(re.findall(r'(\S*)\s*=\s*(\S*)', update))

  # UCI Statlog (Heart) dataset.
  csv_file = tf.keras.utils.get_file(
      'heart.csv',
      'http://storage.googleapis.com/download.tensorflow.org/data/heart.csv')
  df = pd.read_csv(csv_file)
  target = df.pop('target')
  train_size = int(len(df) * 0.8)
  train_x = df[:train_size]
  train_y = target[:train_size]
  test_x = df[train_size:]
  test_y = target[train_size:]

  # feature_analysis_input_fn is used to collect statistics about the input
  # features, thus requiring only one loop of the dataset.
  #
  # feature_analysis_input_fn is required if you have at least one FeatureConfig
  # with "pwl_calibration_input_keypoints='quantiles'". Note that 'quantiles' is
  # default keypoints configuration so most likely you'll need it.
  feature_analysis_input_fn = tf_compat_v1_estimator.inputs.pandas_input_fn(
      x=train_x,
      y=train_y,
      shuffle=False,
      batch_size=FLAGS.batch_size,
      num_epochs=1,
      num_threads=1)

  # prefitting_input_fn is used to prefit an initial ensemble that is used to
  # estimate feature interactions. This prefitting step does not need to fully
  # converge and thus requiring fewer epochs than the main training.
  #
  # prefitting_input_fn is only required if your model_config is
  # CalibratedLatticeEnsembleConfig with "lattices='crystals'"
  prefitting_input_fn = tf_compat_v1_estimator.inputs.pandas_input_fn(
      x=train_x,
      y=train_y,
      shuffle=True,
      batch_size=FLAGS.batch_size,
      num_epochs=FLAGS.prefitting_num_epochs,
      num_threads=1)

  train_input_fn = tf_compat_v1_estimator.inputs.pandas_input_fn(
      x=train_x,
      y=train_y,
      shuffle=True,
      batch_size=FLAGS.batch_size,
      num_epochs=FLAGS.num_epochs,
      num_threads=1)

  test_input_fn = tf_compat_v1_estimator.inputs.pandas_input_fn(
      x=test_x,
      y=test_y,
      shuffle=False,
      batch_size=FLAGS.batch_size,
      num_epochs=FLAGS.num_epochs,
      num_threads=1)

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
  feature_columns = [
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

  # Feature configs are used to specify how each feature is calibrated and used.
  feature_configs = [
      configs.FeatureConfig(
          name='age',
          lattice_size=3,
          # By default, input keypoints of pwl are quantiles of the feature.
          pwl_calibration_num_keypoints=5,
          monotonicity='increasing',
          pwl_calibration_clip_max=100,
      ),
      configs.FeatureConfig(
          name='cp',
          pwl_calibration_num_keypoints=4,
          # Keypoints can be uniformly spaced.
          pwl_calibration_input_keypoints='uniform',
          monotonicity='increasing',
      ),
      configs.FeatureConfig(
          name='chol',
          # Explicit input keypoint initialization.
          pwl_calibration_input_keypoints=[126.0, 210.0, 247.0, 286.0, 564.0],
          monotonicity='increasing',
          pwl_calibration_clip_min=130,
          # Calibration can be forced to span the full output range by clamping.
          pwl_calibration_clamp_min=True,
          pwl_calibration_clamp_max=True,
          # Per feature regularization.
          regularizer_configs=[
              configs.RegularizerConfig(name='calib_hessian', l2=1e-4),
          ],
      ),
      configs.FeatureConfig(
          name='fbs',
          # Monotonicity: output for 1 should be larger than output for 0.
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
          monotonicity='decreasing',
      ),
      configs.FeatureConfig(
          name='restecg',
          # Categorical monotonicity can be partial order.
          monotonicity=[(0, 1), (0, 2)],
      ),
      configs.FeatureConfig(
          name='exang',
          monotonicity=[(0, 1)],
      ),
      configs.FeatureConfig(
          name='oldpeak',
          pwl_calibration_num_keypoints=5,
          monotonicity='increasing',
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

  # Serving input fn is used to create saved models.
  serving_input_fn = (
      tf_estimator.export.build_parsing_serving_input_receiver_fn(
          feature_spec=fc.make_parse_example_spec(feature_columns)))

  # Model config defines the model strcutre for the estimator.
  # This is calibrated linear model with outputput calibration: Inputs are
  # calibrated, linearly combined and the output of the linear layer is
  # calibrated again using a PWL function.
  model_config = configs.CalibratedLinearConfig(
      feature_configs=feature_configs,
      use_bias=True,
      output_calibration=True,
      regularizer_configs=[
          # Regularizer for the output calibrator.
          configs.RegularizerConfig(name='output_calib_hessian', l2=1e-4),
      ])
  # Update model configuration.
  # See tfl.configs.apply_updates for details.
  configs.apply_updates(model_config, config_updates)
  estimator = estimators.CannedClassifier(
      feature_columns=feature_columns,
      model_config=model_config,
      feature_analysis_input_fn=feature_analysis_input_fn,
      optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate))
  estimator.train(input_fn=train_input_fn)
  results = estimator.evaluate(input_fn=test_input_fn)
  print('Calibrated linear results: {}'.format(results))
  print('Calibrated linear model exported to {}'.format(
      estimator.export_saved_model(estimator.model_dir, serving_input_fn)))

  # This is calibrated lattice model: Inputs are calibrated, then combined
  # non-linearly using a lattice layer.
  model_config = configs.CalibratedLatticeConfig(
      feature_configs=feature_configs,
      regularizer_configs=[
          # Torsion regularizer applied to the lattice to make it more linear.
          configs.RegularizerConfig(name='torsion', l2=1e-4),
          # Globally defined calibration regularizer is applied to all features.
          configs.RegularizerConfig(name='calib_hessian', l2=1e-4),
      ])
  estimator = estimators.CannedClassifier(
      feature_columns=feature_columns,
      model_config=model_config,
      feature_analysis_input_fn=feature_analysis_input_fn,
      optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate))
  estimator.train(input_fn=train_input_fn)
  results = estimator.evaluate(input_fn=test_input_fn)
  print('Calibrated lattice results: {}'.format(results))
  print('Calibrated lattice model exported to {}'.format(
      estimator.export_saved_model(estimator.model_dir, serving_input_fn)))

  # This is random lattice ensemble model with separate calibration:
  # model output is the average output of separately calibrated lattices.
  model_config = configs.CalibratedLatticeEnsembleConfig(
      feature_configs=feature_configs,
      num_lattices=6,
      lattice_rank=5,
      separate_calibrators=True,
      regularizer_configs=[
          # Torsion regularizer applied to the lattice to make it more linear.
          configs.RegularizerConfig(name='torsion', l2=1e-4),
          # Globally defined calibration regularizer is applied to all features.
          configs.RegularizerConfig(name='calib_hessian', l2=1e-4),
      ])
  configs.apply_updates(model_config, config_updates)
  estimator = estimators.CannedClassifier(
      feature_columns=feature_columns,
      model_config=model_config,
      feature_analysis_input_fn=feature_analysis_input_fn,
      optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate))
  estimator.train(input_fn=train_input_fn)
  results = estimator.evaluate(input_fn=test_input_fn)
  print('Random ensemble results: {}'.format(results))
  print('Random ensemble model exported to {}'.format(
      estimator.export_saved_model(estimator.model_dir, serving_input_fn)))

  # This is Crystals ensemble model with separate calibration: model output is
  # the average output of separately calibrated lattices.
  # Crystals algorithm first trains a prefitting model and uses the interactions
  # between features to form the final lattice ensemble.
  model_config = configs.CalibratedLatticeEnsembleConfig(
      feature_configs=feature_configs,
      # Using Crystals algorithm.
      lattices='crystals',
      num_lattices=6,
      lattice_rank=5,
      separate_calibrators=True,
      regularizer_configs=[
          # Torsion regularizer applied to the lattice to make it more linear.
          configs.RegularizerConfig(name='torsion', l2=1e-4),
          # Globally defined calibration regularizer is applied to all features.
          configs.RegularizerConfig(name='calib_hessian', l2=1e-4),
      ])
  configs.apply_updates(model_config, config_updates)
  estimator = estimators.CannedClassifier(
      feature_columns=feature_columns,
      model_config=model_config,
      feature_analysis_input_fn=feature_analysis_input_fn,
      # prefitting_input_fn is required to train the prefitting model.
      prefitting_input_fn=prefitting_input_fn,
      optimizer=tf.keras.optimizers.Adam(FLAGS.learning_rate))
  estimator.train(input_fn=train_input_fn)
  results = estimator.evaluate(input_fn=test_input_fn)
  print('Crystals ensemble results: {}'.format(results))
  print('Crystals ensemble model exported to {}'.format(
      estimator.export_saved_model(estimator.model_dir, serving_input_fn)))


if __name__ == '__main__':
  app.run(main)
