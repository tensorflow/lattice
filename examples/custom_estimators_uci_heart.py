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
"""Example usage of TFL layers in custom estimators.

This example trains a TFL custom estimators on the UCI heart dataset.

Example usage:
custom_estimators_uci_heart --num_epochs=40
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column as fc
import tensorflow_lattice as tfl
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.head import binary_class_head

FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('num_epochs', 200, 'Number of training epoch.')


def main(_):
  # UCI Statlog (Heart) dataset.
  csv_file = tf.keras.utils.get_file(
      'heart.csv', 'http://storage.googleapis.com/applied-dl/heart.csv')
  df = pd.read_csv(csv_file)
  target = df.pop('target')
  train_size = int(len(df) * 0.8)
  train_x = df[:train_size]
  train_y = target[:train_size]
  test_x = df[train_size:]
  test_y = target[train_size:]

  train_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
      x=train_x,
      y=train_y,
      shuffle=True,
      batch_size=FLAGS.batch_size,
      num_epochs=FLAGS.num_epochs,
      num_threads=1)

  test_input_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(
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
      fc.numeric_column('ca'),
      fc.categorical_column_with_vocabulary_list(
          'thal', ['normal', 'fixed', 'reversible']),
  ]

  def model_fn(features, labels, mode, config):
    """model_fn for the custom estimator."""
    del config
    input_tensors = tfl.estimators.transform_features(features, feature_columns)
    inputs = {
        key: tf.keras.layers.Input(shape=(1,), name=key)
        for key in input_tensors
    }

    lattice_sizes = [3, 2, 2, 2]
    lattice_monotonicities = ['increasing', 'none', 'increasing', 'increasing']
    lattice_input = tf.keras.layers.Concatenate(axis=1)([
        tfl.layers.PWLCalibration(
            input_keypoints=np.linspace(10, 100, num=8, dtype=np.float32),
            # The output range of the calibrator should be the input range of
            # the following lattice dimension.
            output_min=0.0,
            output_max=lattice_sizes[0] - 1.0,
            monotonicity='increasing',
        )(inputs['age']),
        tfl.layers.CategoricalCalibration(
            # Number of categories including any missing/default category.
            num_buckets=2,
            output_min=0.0,
            output_max=lattice_sizes[1] - 1.0,
        )(inputs['sex']),
        tfl.layers.PWLCalibration(
            input_keypoints=[0.0, 1.0, 2.0, 3.0],
            output_min=0.0,
            output_max=lattice_sizes[0] - 1.0,
            # You can specify TFL regularizers as tuple
            # ('regularizer name', l1, l2).
            kernel_regularizer=('hessian', 0.0, 1e-4),
            monotonicity='increasing',
        )(inputs['ca']),
        tfl.layers.CategoricalCalibration(
            num_buckets=3,
            output_min=0.0,
            output_max=lattice_sizes[1] - 1.0,
            # Categorical monotonicity can be partial order.
            # (i, j) indicates that we must have output(i) <= output(i).
            # Make sure to set the lattice monotonicity to 1 for this dimension.
            monotonicities=[(0, 1), (0, 2)],
        )(inputs['thal']),
    ])
    output = tfl.layers.Lattice(
        lattice_sizes=lattice_sizes, monotonicities=lattice_monotonicities)(
            lattice_input)

    training = (mode == tf.estimator.ModeKeys.TRAIN)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    logits = model(input_tensors, training=training)

    if training:
      optimizer = optimizers.get_optimizer_instance_v2('Adam',
                                                       FLAGS.learning_rate)
    else:
      optimizer = None

    head = binary_class_head.BinaryClassHead()
    return head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        optimizer=optimizer,
        logits=logits,
        trainable_variables=model.trainable_variables,
        update_ops=model.updates)

  estimator = tf.estimator.Estimator(model_fn=model_fn)
  estimator.train(input_fn=train_input_fn)
  results = estimator.evaluate(input_fn=test_input_fn)
  print('Results: {}'.format(results))


if __name__ == '__main__':
  app.run(main)
