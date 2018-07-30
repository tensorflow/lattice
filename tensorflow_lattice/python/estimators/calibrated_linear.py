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
"""CalibratedLinear canned estimators."""
# Dependency imports

from tensorflow_lattice.python.estimators import calibrated as calibrated_lib
from tensorflow_lattice.python.estimators import hparams as tfl_hparams

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope

# Scope for variable names.
_SCOPE_BIAS_WEIGHT = 'bias_weight'


class _CalibratedLinear(calibrated_lib.Calibrated):
  """Base class for CalibratedLinearClassifier and CalibratedLinearRegressor."""

  def __init__(self,
               n_classes,
               feature_columns=None,
               model_dir=None,
               quantiles_dir=None,
               keypoints_initializers_fn=None,
               optimizer=None,
               config=None,
               hparams=None,
               head=None,
               weight_column=None):
    """Construct CalibrateLinearClassifier/Regressor."""
    if not hparams:
      hparams = tfl_hparams.CalibratedLinearHParams([])
    self.check_hparams(hparams)

    super(_CalibratedLinear,
          self).__init__(n_classes, feature_columns, model_dir, quantiles_dir,
                         keypoints_initializers_fn, optimizer, config, hparams,
                         head, weight_column, 'linear')

  def _check_param_configuration(self, num_keypoints, missing_input_value,
                                 missing_output_value):
    error_messages = []
    if ((num_keypoints is None or num_keypoints < 2) and
        missing_input_value is not None):
      error_messages.append(
          'num_keypoints not set (or too low) so value is not calibrated, '
          'and cannot handle missing values')
    if missing_output_value is not None:
      error_messages.append('CalibratedLinear models do not support fixed '
                            'output for missing values')
    return error_messages

  def check_hparams(self, hparams):
    """Check pre-conditions of hparams.

    Args:
      hparams: (tfl_hparams.CalibratedLatticeHParams) Hyperparameter to
      be examined.
    Raises:
      ValueError: If the hyperparameter configuration is invalid, for example
      calibration_monotonic is None, but lattice_monotonic is True, then raise
      the error with a root cause.
    """
    error_messages = []

    # Check global params.
    feature_names = hparams.get_feature_names()
    global_values, per_feature_values = hparams.get_global_and_feature_params(
        ['num_keypoints', 'missing_input_value', 'missing_output_value'],
        feature_names)
    global_param_error_messages = self._check_param_configuration(
        *global_values)
    if global_param_error_messages:
      error_messages.append('Error message for global param:')
      error_messages += global_param_error_messages

    # Check per feature params. hparams.get_feature_names()  will only return
    # feature names that sets per feature parameters.
    for feature_idx in range(len(per_feature_values)):
      per_feature_param_error_messages = self._check_param_configuration(
          *per_feature_values[feature_idx])
      if per_feature_param_error_messages:
        error_messages.append(
            'Error message for %s feature param:' % feature_names[feature_idx])
        error_messages += per_feature_param_error_messages

    if error_messages:
      raise ValueError(
          'Hyperparameter configuration cannot be used in the calibrated '
          'lattice estimator. Error messages report the issue per feature, but'
          ' the parameter may be inherited from global parameter.\nDetailed '
          'error messsages\n%s' % '\n'.join(error_messages))

  def calibration_structure_builder(self, columns_to_tensors, hparams):
    """Returns the calibration structure of the model. See base class."""
    return None

  def prediction_builder_from_calibrated(
      self, mode, per_dimension_feature_names, hparams, calibrated):
    # No need for linear weights: since they are redundant, the calibration
    # can accommodate the weights. Same could be said for the bias, but
    # it turns out that a bias makes it easier to train in the presence of
    # many features.

    self.check_hparams(hparams)
    prediction = math_ops.reduce_sum(calibrated, 1, keepdims=True)
    bias = variable_scope.get_variable(
        _SCOPE_BIAS_WEIGHT,
        initializer=array_ops.zeros(shape=[], dtype=self._dtype))
    prediction += bias
    # Returns prediction Tensor, projection ops, and regularization ops.
    return prediction, None, None


def calibrated_linear_classifier(feature_columns=None,
                                 model_dir=None,
                                 quantiles_dir=None,
                                 keypoints_initializers_fn=None,
                                 optimizer=None,
                                 config=None,
                                 hparams=None,
                                 head=None,
                                 weight_column=None):
  """Calibrated linear classifier binary model.



  This model uses a piecewise linear calibration function on each of the
  real (as opposed to binary) inputs (parametrized) and then combines (sum up)
  the results. Optionally calibration can be made monotonic.

  It usually requires a preprocessing step on the data, to calculate the
  quantiles of each used feature. This can be done locally or in one worker
  only before training, in a separate invocation of your program (or directly).
  Typically this can be save (`save_dir` parameter) to the same
  directory where the data is.

  Hyper-parameters are given in the form of the object
  tfl_hparams.CalibrationHParams. It takes in per-feature calibration
  parameters.

  Internally values will be converted to tf.float32.

  Example:

  ```python
  def input_fn_train: ...
  def input_fn_eval: ...

  my_feature_columns=[...]

  # Have a separate program flag to generate the quantiles. Need to be run
  # only once.
  if FLAGS.create_quantiles:
    pwl_calibrators_layers.calculate_quantiles_for_keypoints(
      input_fn=input_fn_train,
      feature_columns=my_feature_columns,
      save_dir=FLAGS.data_dir,
      num_quantiles=1000,
      override=True)
    return  # Exit program.

  estimator = calibrated_linear.CalibratedLinearClassifier(
    feature_columns=feature_columns)
  estimator.train(input_fn=input_fn_train)
  estimator.evaluate(input_fn=input_fn_eval)
  estimator.predict(input_fn=input_fn_predict)
  ```

  Args:
    feature_columns: Optional, an iteratable containing all the feature
      columns used by the model. All items in the set should be instances of
      classes derived from `FeatureColumn`. If not given, the model will
      use as features the tensors returned by input_fn.
      Supported types of columns: RealValuedColumn.
    model_dir: Directory to save model parameters, graph and etc. This can
      also be used to load checkpoints from the directory into a estimator to
      continue training a previously saved model.
    quantiles_dir: location where quantiles for the data was saved. Typically
      the same directory as the training data. These quantiles can be
      generated only once with
      `pwl_calibration_layers.calculate_quantiles_for_keypoints` in a separate
      invocation of your program. If you don't want to use quantiles, you can
      set `keypoints_initializer` instead.
    keypoints_initializers_fn: if you know the distribution of your
      input features you can provide that directly instead of `quantiles_dir`.
      See `pwl_calibrators_layers.uniform_keypoints_for_signal`. It must be
      a closure that returns a pair of tensors with keypoints inputs and
      outputs to use for initialization (must match `num_keypoints` configured
      in `hparams`). Alternatively the closure can return a dict mapping
      feature name to pairs for initialization per feature. If `quantiles_dir`
      and `keypoints_initializers_fn` are set, the later takes precendence,
      and the features for which `keypoints_initializers` are not defined
      fallback to using the quantiles found in `quantiles_dir`. It uses a
      closure instead of the tensors themselves because the graph has to be
      created at the time the model is being build, which happens at a later
      time.
    optimizer: string, `Optimizer` object, or callable that defines the
      optimizer to use for training -- if a callable, it will be called with
      learning_rate=hparams.learning_rate.
    config: RunConfig object to configure the runtime settings. Typically set
      to learn_runner.EstimatorConfig().
    hparams: an instance of tfl_hparams.CalibrationHParams. If set to
      None default parameters are used.
    head: a `TensorFlow Estimator Head` which specifies how the loss function,
      final predictions, and so on are generated from model outputs. Defaults
      to using a sigmoid cross entropy head for binary classification and mean
      squared error head for regression.
    weight_column: A string or a `tf.feature_column.numeric_column` defining
      feature column representing weights. It is used to down weight or boost
      examples during training. It will be multiplied by the loss of the
      example.

  Returns:
    A `CalibratedLinearClassifier` estimator.

  Raises:
    ValueError: invalid parameters.
    KeyError: type of feature not supported.
  """
  return _CalibratedLinear(
      n_classes=2,
      feature_columns=feature_columns,
      model_dir=model_dir,
      quantiles_dir=quantiles_dir,
      keypoints_initializers_fn=keypoints_initializers_fn,
      optimizer=optimizer,
      config=config,
      hparams=hparams,
      head=head,
      weight_column=weight_column)


def calibrated_linear_regressor(feature_columns=None,
                                model_dir=None,
                                quantiles_dir=None,
                                keypoints_initializers_fn=None,
                                optimizer=None,
                                config=None,
                                hparams=None,
                                head=None,
                                weight_column=None):
  """Calibrated linear estimator (model) for regression.

  This model uses a piecewise linear calibration function on each of the
  inputs (parametrized) and then combine (sum up) the results. Optionally
  calibration can be made monotonic.

  It usually requires a preprocessing step on the data, to calculate the
  quantiles of each used feature. This can be done locally or in one worker
  only before training, in a separate invocation of your program (or directly)
  in . Typically this can be save (`save_dir` parameter) to the same
  directory where the data is.

  Hyper-parameters are given in the form of the object
  tfl_hparams.CalibrationHParams. It takes in per-feature calibration
  parameters.

  Internally values will be converted to tf.float32.



  Example:

  ```python
  def input_fn_train: ...
  def input_fn_eval: ...

  my_feature_columns=[...]

  # Have a separate program flag to generate the quantiles. Need to be run
  # only once.
  if FLAGS.create_quantiles:
    pwl_calibrators_layers.calculate_quantiles_for_keypoints(
      input_fn=input_fn_train,
      feature_columns=my_feature_columns,
      save_dir=FLAGS.data_dir,
      num_quantiles=1000,
      override=True)
    return  # Exit program.

  estimator = calibrated_linear.calibrated_linear_regressor(
    feature_columns=feature_columns)
  estimator.train(input_fn=input_fn_train)
  estimator.evaluate(input_fn=input_fn_eval)
  estimator.predict(input_fn=input_fn_predict)
  ```

  Args:
    feature_columns: Optional, if not set the model will use all features
      returned by input_fn. An iteratable containing all the feature
      columns used by the model. All items in the set should be instances of
      classes derived from `FeatureColumn`. If not given, the model will
      use as features the tensors returned by input_fn.
      Supported types: RealValuedColumn.
    model_dir: Directory to save model parameters, graph and etc. This can
      also be used to load checkpoints from the directory into a estimator to
      continue training a previously saved model.
    quantiles_dir: location where quantiles for the data was saved. Typically
      the same directory as the training data. These quantiles can be
      generated only once with
      `pwl_calibration_layers.calculate_quantiles_for_keypoints` in a separate
      invocation of your program. If you don't want to use quantiles, you can
      set `keypoints_initializer` instead.
    keypoints_initializers_fn: if you know the distribution of your
      input features you can provide that directly instead of `quantiles_dir`.
      See `pwl_calibrators_layers.uniform_keypoints_for_signal`. It must be
      a closure that returns a pair of tensors with keypoints inputs and
      outputs to use for initialization (must match `num_keypoints` configured
      in `hparams`). Alternatively the closure can return a dict mapping
      feature name to pairs for initialization per feature. If `quantiles_dir`
      and `keypoints_initializers_fn` are set, the later takes precendence,
      and the features for which `keypoints_initializers` are not defined
      fallback to using the quantiles found in `quantiles_dir`. It uses a
      closure instead of the tensors themselves because the graph has to be
      created at the time the model is being build, which happens at a later
      time.
    optimizer: string, `Optimizer` object, or callable that defines the
      optimizer to use for training -- if a callable, it will be called with
      learning_rate=hparams.learning_rate.
    config: RunConfig object to configure the runtime settings. Typically set
      to learn_runner.EstimatorConfig().
    hparams: an instance of tfl_hparams.CalibrationHParams. If set to
      None default parameters are used.
    head: a `TensorFlow Estimator Head` which specifies how the loss function,
      final predictions, and so on are generated from model outputs. Defaults
      to using a sigmoid cross entropy head for binary classification and mean
      squared error head for regression.
    weight_column: A string or a `tf.feature_column.numeric_column` defining
      feature column representing weights. It is used to down weight or boost
      examples during training. It will be multiplied by the loss of the
      example.

  Returns:
    A `CalibratedLinearRegressor` estimator.

  Raises:
    ValueError: invalid parameters.
    KeyError: type of feature not supported.
  """
  return _CalibratedLinear(
      n_classes=0,
      feature_columns=feature_columns,
      model_dir=model_dir,
      quantiles_dir=quantiles_dir,
      keypoints_initializers_fn=keypoints_initializers_fn,
      optimizer=optimizer,
      config=config,
      hparams=hparams,
      head=head,
      weight_column=weight_column)
