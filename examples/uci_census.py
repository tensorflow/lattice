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
"""Train and evaluate models on UCI Census data.

This is an example TensorFlow Lattice model training and evaluating program,
using TensorFlow's `tf.estimators` library, a high level abstraction layer
for machine learning models.

TensorFlow Lattice also offers "layer" level components, so one can customize
their own models, but these are not included in this example.

Example run for calibrated linear model:

* Uses bash variables `type` and `attempt` for convenience. You can bump
  `attempt` when trying different hyper-parameters.
* The flag `--create_quantiles` need to be set just the very first time you
  run, since the data quantiles information used for calibration is the same
  for all models.
* Use `--hparams` to set changes to default parameters.
* It will print out evaluation on the training data and evaluation data
  every 1/10th of the training epochs.

```bash
$ type=calibrated_linear ; attempt=1 ;
  python uci_census.py --run=train --model_type=${type}
    --output_dir=${HOME}/experiments/uci_census/${type}_${attempt}
    --quantiles_dir=${HOME}/experiments/uci_census
    --train_epochs=600 --batch_size=1000
    --hparams=learning_rate=1e-3
    --create_quantiles
```

Example run for calibrated RTL model (assumes you already created the
quantiles):

* Notice calibrated RTL models train slower than calibrated linear model, but
should yield slightly better results.

```bash
$ type=calibrated_rtl ; attempt=1 ;
  python uci_census.py --run=train --model_type=${type}
    --output_dir=${HOME}/experiments/uci_census/${type}_${attempt}
    --quantiles_dir=${HOME}/experiments/uci_census
    --train_epochs=600 --batch_size=1000
    --hparams=learning_rate=1e-2
```

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import pandas as pd
import six
import tensorflow as tf
import tensorflow_lattice as tfl

flags = tf.flags
FLAGS = flags.FLAGS

# Run mode of the program.
flags.DEFINE_string(
    "run", "train", "One of 'train', 'evaluate' or 'save', train will "
    "train on training data and also optionally evaluate; evaluate will "
    "evaluate train and test data; save saves the trained model so far "
    "so it can be used by TensorFlow Serving.")

# Dataset.
flags.DEFINE_string("test", "/tmp/uci_census/adult.test", "Path to test file.")
flags.DEFINE_string("train", "/tmp/uci_census/adult.data",
                    "Path to train file.")

# Model flags.
flags.DEFINE_string(
    "output_dir", None,
    "Directory where to store the model. If not set a temporary directory "
    "will be automatically created.")
flags.DEFINE_string(
    "model_type", "calibrated_linear",
    "Types defined in this example: calibrated_linear, calibrated_lattice, "
    " calibrated_rtl, calibrated_etl, calibrated_dnn")
flags.DEFINE_integer("batch_size", 1000,
                     "Number of examples to include in one batch. Increase "
                     "this number to improve parallelism, at cost of memory.")
flags.DEFINE_string("hparams", None,
                    "Model hyperparameters, see hyper-parameters in Tensorflow "
                    "Lattice documentation. Example: --hparams=learning_rate="
                    "0.1,lattice_size=2,num_keypoints=100")

# Calibration quantiles flags.
flags.DEFINE_bool("create_quantiles", False,
                  "Run once to create histogram of features for calibration.")
flags.DEFINE_string(
    "quantiles_dir", None,
    "Directory where to store quantile information, defaults to the model "
    "directory (set by --output-dir) but since quantiles can be reused by "
    "models with different parameters, you may want to have a separate "
    "directory.")

# Training flags.
flags.DEFINE_integer("train_epochs", 10,
                     "How many epochs over data during training.")
flags.DEFINE_bool(
    "train_evaluate_on_train", True,
    "If set, every 1/10th of the train_epochs runs an evaluation on the "
    "full train data.")
flags.DEFINE_bool(
    "train_evaluate_on_test", True,
    "If set, every 1/10th of the train_epochs runs an evaluation on the "
    "full test data.")

# Columns in dataset files.
CSV_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "gender",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income_bracket"
]


def get_test_input_fn(batch_size, num_epochs, shuffle):
  return get_input_fn(FLAGS.test, batch_size, num_epochs, shuffle)


def get_train_input_fn(batch_size, num_epochs, shuffle):
  return get_input_fn(FLAGS.train, batch_size, num_epochs, shuffle)


# Copy of data read from train/test files: keep copy to avoid re-reading
# it at every training/evaluation loop.
_df_data = {}
_df_data_labels = {}


def get_input_fn(file_path, batch_size, num_epochs, shuffle):
  """Returns an input_fn closure for given parameters."""
  if file_path not in _df_data:
    _df_data[file_path] = pd.read_csv(
        tf.gfile.Open(file_path),
        names=CSV_COLUMNS,
        skipinitialspace=True,
        engine="python",
        skiprows=1)
    _df_data[file_path] = _df_data[file_path].dropna(how="any", axis=0)
    _df_data_labels[file_path] = _df_data[file_path]["income_bracket"].apply(
        lambda x: ">50K" in x).astype(int)
  return tf.estimator.inputs.pandas_input_fn(
      x=_df_data[file_path],
      y=_df_data_labels[file_path],
      batch_size=batch_size,
      shuffle=shuffle,
      num_epochs=num_epochs,
      num_threads=1)


def create_feature_columns():
  """Creates feature columns for UCI Census, some are sparse."""
  # Categorical features.
  gender = tf.feature_column.categorical_column_with_vocabulary_list(
      "gender", ["Female", "Male"])
  education = tf.feature_column.categorical_column_with_vocabulary_list(
      "education", [
          "Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college",
          "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school",
          "5th-6th", "10th", "1st-4th", "Preschool", "12th"
      ])
  marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
      "marital_status", [
          "Married-civ-spouse", "Divorced", "Married-spouse-absent",
          "Never-married", "Separated", "Married-AF-spouse", "Widowed"
      ])
  relationship = tf.feature_column.categorical_column_with_vocabulary_list(
      "relationship", [
          "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
          "Other-relative"
      ])
  workclass = tf.feature_column.categorical_column_with_vocabulary_list(
      "workclass", [
          "Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
          "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"
      ])
  occupation = tf.feature_column.categorical_column_with_vocabulary_list(
      "occupation", [
          "Prof-specialty", "Craft-repair", "Exec-managerial", "Adm-clerical",
          "Sales", "Other-service", "Machine-op-inspct", "?",
          "Transport-moving", "Handlers-cleaners", "Farming-fishing",
          "Tech-support", "Protective-serv", "Priv-house-serv", "Armed-Forces"
      ])
  race = tf.feature_column.categorical_column_with_vocabulary_list(
      "race", [
          "White",
          "Black",
          "Asian-Pac-Islander",
          "Amer-Indian-Eskimo",
          "Other",
      ])
  native_country = tf.feature_column.categorical_column_with_vocabulary_list(
      "native_country", [
          "United-States",
          "Mexico",
          "?",
          "Philippines",
          "Germany",
          "Canada",
          "Puerto-Rico",
          "El-Salvador",
          "India",
          "Cuba",
          "England",
          "Jamaica",
          "South",
          "China",
          "Italy",
          "Dominican-Republic",
          "Vietnam",
          "Guatemala",
          "Japan",
          "Poland",
          "Columbia",
          "Taiwan",
          "Haiti",
          "Iran",
          "Portugal",
          "Nicaragua",
          "Peru",
          "Greece",
          "France",
          "Ecuador",
          "Ireland",
          "Hong",
          "Trinadad&Tobago",
          "Cambodia",
          "Thailand",
          "Laos",
          "Yugoslavia",
          "Outlying-US(Guam-USVI-etc)",
          "Hungary",
          "Honduras",
          "Scotland",
          "Holand-Netherlands",
      ])

  # Numerical (continuous) base columns.
  age = tf.feature_column.numeric_column("age")
  education_num = tf.feature_column.numeric_column("education_num")
  capital_gain = tf.feature_column.numeric_column("capital_gain")
  capital_loss = tf.feature_column.numeric_column("capital_loss")
  hours_per_week = tf.feature_column.numeric_column("hours_per_week")

  # fnlwgt: this should be the weight, how representative this example is of
  #    the population, we don't use it here.
  # fnlwgt = tf.feature_column.numeric_column("fnlwgt")

  # income-bracket is the label, so, not returned here.
  return [
      age,
      workclass,
      education,
      education_num,
      marital_status,
      occupation,
      relationship,
      race,
      gender,
      capital_gain,
      capital_loss,
      hours_per_week,
      native_country,
  ]


def create_quantiles(quantiles_dir):
  """Creates quantiles directory if it doesn't yet exist."""
  batch_size = 10000
  input_fn = get_test_input_fn(
      batch_size=batch_size, num_epochs=1, shuffle=False)
  # Reads until input is exhausted, 10000 at a time.
  tfl.save_quantiles_for_keypoints(
      input_fn=input_fn,
      save_dir=quantiles_dir,
      feature_columns=create_feature_columns(),
      num_steps=None)


def _pprint_hparams(hparams):
  """Pretty-print hparams."""
  print("* hparams=[")
  for (key, value) in sorted(six.iteritems(hparams.values())):
    print("\t{}={}".format(key, value))
  print("]")


def create_calibrated_linear(feature_columns, config, quantiles_dir):
  feature_names = [fc.name for fc in feature_columns]
  hparams = tfl.CalibratedLinearHParams(
      feature_names=feature_names, num_keypoints=200, learning_rate=1e-4)
  hparams.parse(FLAGS.hparams)
  hparams.set_feature_param("capital_gain", "calibration_l2_laplacian_reg",
                            4.0e-3)
  _pprint_hparams(hparams)
  return tfl.calibrated_linear_classifier(
      feature_columns=feature_columns,
      model_dir=config.model_dir,
      config=config,
      hparams=hparams,
      quantiles_dir=quantiles_dir)


def create_calibrated_lattice(feature_columns, config, quantiles_dir):
  """Creates a calibrated lattice estimator."""
  feature_names = [fc.name for fc in feature_columns]
  hparams = tfl.CalibratedLatticeHParams(
      feature_names=feature_names,
      num_keypoints=200,
      lattice_l2_laplacian_reg=5.0e-3,
      lattice_l2_torsion_reg=1.0e-4,
      learning_rate=0.1,
      lattice_size=2)
  hparams.parse(FLAGS.hparams)
  _pprint_hparams(hparams)
  return tfl.calibrated_lattice_classifier(
      feature_columns=feature_columns,
      model_dir=config.model_dir,
      config=config,
      hparams=hparams,
      quantiles_dir=quantiles_dir)


def create_calibrated_rtl(feature_columns, config, quantiles_dir):
  """Creates a calibrated RTL estimator."""
  feature_names = [fc.name for fc in feature_columns]
  hparams = tfl.CalibratedRtlHParams(
      feature_names=feature_names,
      num_keypoints=200,
      learning_rate=0.02,
      lattice_l2_laplacian_reg=5.0e-4,
      lattice_l2_torsion_reg=1.0e-4,
      lattice_size=3,
      lattice_rank=4,
      num_lattices=100)
  # Specific feature parameters.
  hparams.set_feature_param("capital_gain", "lattice_size", 8)
  hparams.set_feature_param("native_country", "lattice_size", 8)
  hparams.set_feature_param("marital_status", "lattice_size", 4)
  hparams.set_feature_param("age", "lattice_size", 8)
  hparams.parse(FLAGS.hparams)
  _pprint_hparams(hparams)
  return tfl.calibrated_rtl_classifier(
      feature_columns=feature_columns,
      model_dir=config.model_dir,
      config=config,
      hparams=hparams,
      quantiles_dir=quantiles_dir)


def create_calibrated_etl(feature_columns, config, quantiles_dir):
  """Creates a calibrated ETL estimator."""
  # No enforced monotonicity in this example.
  feature_names = [fc.name for fc in feature_columns]
  hparams = tfl.CalibratedEtlHParams(
      feature_names=feature_names,
      num_keypoints=200,
      learning_rate=0.02,
      non_monotonic_num_lattices=200,
      non_monotonic_lattice_rank=2,
      non_monotonic_lattice_size=2,
      calibration_l2_laplacian_reg=4.0e-3,
      lattice_l2_laplacian_reg=1.0e-5,
      lattice_l2_torsion_reg=4.0e-4)
  hparams.parse(FLAGS.hparams)
  _pprint_hparams(hparams)
  return tfl.calibrated_etl_classifier(
      feature_columns=feature_columns,
      model_dir=config.model_dir,
      config=config,
      hparams=hparams,
      quantiles_dir=quantiles_dir)


def create_calibrated_dnn(feature_columns, config, quantiles_dir):
  """Creates a calibrated DNN model."""
  # This is an example of a hybrid model that uses input calibration layer
  # offered by TensorFlow Lattice library and connects it to a DNN.
  feature_names = [fc.name for fc in feature_columns]
  hparams = tfl.CalibratedHParams(
      feature_names=feature_names,
      num_keypoints=200,
      learning_rate=1.0e-3,
      calibration_output_min=-1.0,
      calibration_output_max=1.0,
      nodes_per_layer=10,  # All layers have the same number of nodes.
      layers=2,  # Includes output layer, therefore >= 1.
  )
  hparams.parse(FLAGS.hparams)
  _pprint_hparams(hparams)

  def _model_fn(features, labels, mode, params):
    """Model construction closure used when creating estimator."""
    del params  # Hyper-params are read directly from the bound variable hparams

    # Calibrate: since there is no monotonicity, there are no projection ops.
    # We also discard the ordered names of the features.
    (output, _, _, regularization) = tfl.input_calibration_layer_from_hparams(
        features, feature_columns, hparams, quantiles_dir)

    # Hidden-layers.
    for _ in range(hparams.layers - 1):
      output = tf.layers.dense(
          inputs=output, units=hparams.nodes_per_layer, activation=tf.sigmoid)

    # Classifier logits and prediction.
    logits = tf.layers.dense(inputs=output, units=1)
    predictions = tf.reshape(tf.sigmoid(logits), [-1])

    # Notice loss doesn't include regularization, which is added separately
    # by means of tf.contrib.layers.apply_regularization().
    loss_no_regularization = tf.losses.log_loss(labels, predictions)
    loss = loss_no_regularization
    if regularization is not None:
      loss += regularization
    optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)
    train_op = optimizer.minimize(
        loss,
        global_step=tf.train.get_global_step(),
        name="calibrated_dnn_minimize")

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels, predictions),

        # We want to report the loss without the regularization, so metric is
        # comparable with different regularizations. FutureWork, list both.
        "average_loss": tf.metrics.mean(loss_no_regularization),
    }

    return tf.estimator.EstimatorSpec(mode, predictions, loss, train_op,
                                      eval_metric_ops)

  # Hyper-parameters are passed directly to the model_fn closure by the context.
  return tf.estimator.Estimator(
      model_fn=_model_fn,
      model_dir=config.model_dir,
      config=config,
      params=None)


def create_estimator(config, quantiles_dir):
  """Creates estimator for given configuration based on --model_type."""
  feature_columns = create_feature_columns()
  if FLAGS.model_type == "calibrated_linear":
    return create_calibrated_linear(feature_columns, config, quantiles_dir)
  elif FLAGS.model_type == "calibrated_lattice":
    return create_calibrated_lattice(feature_columns, config, quantiles_dir)
  elif FLAGS.model_type == "calibrated_rtl":
    return create_calibrated_rtl(feature_columns, config, quantiles_dir)
  elif FLAGS.model_type == "calibrated_etl":
    return create_calibrated_etl(feature_columns, config, quantiles_dir)
  elif FLAGS.model_type == "calibrated_dnn":
    return create_calibrated_dnn(feature_columns, config, quantiles_dir)

  raise ValueError("Unknown model_type={}".format(FLAGS.model_type))


def evaluate_on_data(estimator, data):
  """Evaluates and prints results, set data to FLAGS.test or FLAGS.train."""
  name = os.path.basename(data)
  evaluation = estimator.evaluate(
      input_fn=get_input_fn(
          file_path=data,
          batch_size=FLAGS.batch_size,
          num_epochs=1,
          shuffle=False),
      name=name)
  print("  Evaluation on '{}':\taccuracy={:.4f}\taverage_loss={:.4f}".format(
      name, evaluation["accuracy"], evaluation["average_loss"]))


def train(estimator):
  """Trains estimator and optionally intermediary evaluations."""
  if not FLAGS.train_evaluate_on_train and not FLAGS.train_evaluate_on_test:
    estimator.train(input_fn=get_train_input_fn(
        batch_size=FLAGS.batch_size,
        num_epochs=FLAGS.train_epochs,
        shuffle=True))
  else:
    # Train 1/10th of the epochs requested per loop, but at least 1 per loop.
    epochs_trained = 0
    loops = 0
    while epochs_trained < FLAGS.train_epochs:
      loops += 1
      next_epochs_trained = int(loops * FLAGS.train_epochs / 10.0)
      epochs = max(1, next_epochs_trained - epochs_trained)
      epochs_trained += epochs
      estimator.train(input_fn=get_train_input_fn(
          batch_size=FLAGS.batch_size, num_epochs=epochs, shuffle=True))
      print("Trained for {} epochs, total so far {}:".format(
          epochs, epochs_trained))
      evaluate_on_data(estimator, FLAGS.train)
      evaluate_on_data(estimator, FLAGS.test)


def evaluate(estimator):
  """Runs straight evaluation on a currently trained model."""
  evaluate_on_data(estimator, FLAGS.train)
  evaluate_on_data(estimator, FLAGS.test)


def main(args):
  del args  # Not used.

  # Prepare directories.
  output_dir = FLAGS.output_dir
  if output_dir is None:
    output_dir = tempfile.mkdtemp()
    tf.logging.warning("Using temporary folder as model directory: %s",
                       output_dir)
  quantiles_dir = FLAGS.quantiles_dir or output_dir

  # Create quantiles if required.
  if FLAGS.create_quantiles:
    if FLAGS.run != "train":
      raise ValueError(
          "Can not create_quantiles for mode --run='{}'".format(FLAGS.run))
    create_quantiles(quantiles_dir)

  # Create config and then model.
  config = tf.estimator.RunConfig().replace(model_dir=output_dir)
  estimator = create_estimator(config, quantiles_dir)

  if FLAGS.run == "train":
    train(estimator)

  elif FLAGS.run == "evaluate":
    evaluate(estimator)

  else:
    raise ValueError("Unknonw --run={}".format(FLAGS.run))


if __name__ == "__main__":
  tf.app.run()
