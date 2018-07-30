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
"""CalibratedLattice canned estimators."""
import copy

# Dependency imports

from tensorflow_lattice.python.estimators import calibrated as calibrated_lib
from tensorflow_lattice.python.estimators import hparams as tfl_hparams
from tensorflow_lattice.python.lib import lattice_layers
from tensorflow_lattice.python.lib import regularizers

_EPSILON = 1e-7


class _CalibratedLattice(calibrated_lib.Calibrated):
  """Base class for CalibratedLattice{Classifier|Regressor}."""

  def __init__(self,
               n_classes,
               feature_columns=None,
               model_dir=None,
               quantiles_dir=None,
               keypoints_initializers_fn=None,
               lattice_initializers_fn=None,
               optimizer=None,
               config=None,
               hparams=None,
               head=None,
               weight_column=None):
    """Construct CalibrateLatticeClassifier/Regressor."""
    if not hparams:
      hparams = tfl_hparams.CalibratedLatticeHParams([])
    self.check_hparams(hparams)
    hparams = self._set_calibration_params(hparams)

    self.lattice_initializers_fn_ = lattice_initializers_fn

    super(_CalibratedLattice,
          self).__init__(n_classes, feature_columns, model_dir, quantiles_dir,
                         keypoints_initializers_fn, optimizer, config, hparams,
                         head, weight_column, 'lattice')

  def _check_param_configuration(self, adjusted, monotonicity, lattice_size,
                                 calibration_output_min, calibration_output_max,
                                 calibration_bound, missing_input_value,
                                 missing_vertex, *unused_args):
    error_messages = []
    if monotonicity not in {-1, 0, +1}:
      error_messages.append('monotonicity should be an integer {-1, 0, +1} '
                            'but is %s' % monotonicity)
    if lattice_size < 2:
      error_messages.append('lattice_size should be greater than equal to 2'
                            'but is %d' % (lattice_size))

    if not calibration_bound:
      error_messages.append(
          'A lattice expects an bounded input from a calibration layer, but '
          'calibration_bound is set to be False')

    if not adjusted:
      if calibration_output_min is not None:
        error_messages.append(
            'calibration_output_min=%d should not be set, it is adjusted '
            'automatically to match the lattice_size' % calibration_output_min)
      if calibration_output_max is not None:
        error_messages.append(
            'calibration_output_max=%d should not be set, it is adjusted '
            'automatically to match the lattice_size' % calibration_output_max)

    if missing_input_value is None and missing_vertex:
      error_messages.append(
          'missing_vertex is True, however missing_input_value not set')

    return error_messages

  def _check_not_allowed_feature_params(self, hparams):
    not_allowed_feature_params = map(
        'lattice_{}'.format,
        regularizers.LATTICE_MULTI_DIMENSIONAL_REGULARIZERS)
    error_messages = []
    for param in not_allowed_feature_params:
      for feature_name in hparams.get_feature_names():
        if hparams.is_feature_set_param(feature_name, param):
          error_messages.append('feature %s sets %s, which is not allowed.' %
                                (feature_name, param))
    return error_messages

  def check_hparams(self, hparams, adjusted=False):
    """Check pre-conditions of hparams.

    Args:
      hparams: (tfl_hparams.CalibratedLatticeHParams) Hyperparameter to
      be examined.
      adjusted: if these are the parameters already adjusted
    Raises:
      ValueError: If the hyperparameter configuration is invalid, for example
      calibration_monotonic is None, but lattice_monotonic is True, then raise
      the error with a root cause.
    """
    error_messages = self._check_not_allowed_feature_params(hparams)

    # Check global params.
    feature_names = hparams.get_feature_names()
    param_list = [
        'monotonicity',
        'lattice_size',
        'calibration_output_min',
        'calibration_output_max',
        'calibration_bound',
        'missing_input_value',
        'missing_vertex',
    ] + ['lattice_{}'.format(r) for r in regularizers.LATTICE_REGULARIZERS]

    global_values, per_feature_values = hparams.get_global_and_feature_params(
        param_list, feature_names)
    global_param_error_messages = self._check_param_configuration(
        adjusted, *global_values)
    if global_param_error_messages:
      error_messages.append('Error message for global param:')
      error_messages += global_param_error_messages

    # Check per feature params. hparams.get_feature_names()  will only return
    # feature names that sets per feature parameters.
    for feature_idx in range(len(per_feature_values)):
      per_feature_param_error_messages = self._check_param_configuration(
          adjusted, *per_feature_values[feature_idx])
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

  def _set_calibration_params(self, hparams):
    hparams = copy.deepcopy(hparams)
    feature_names = hparams.get_feature_names()
    global_values, per_feature_values = hparams.get_global_and_feature_params(
        ['lattice_size', 'missing_input_value', 'missing_vertex'],
        feature_names)

    final_lattice_size, missing_output_value = self._calibration_params(
        *global_values)
    lattice_size = global_values[0]
    hparams.set_param('calibration_output_min', 0)
    hparams.set_param('calibration_output_max', lattice_size - 1)
    hparams.set_param('final_lattice_size', final_lattice_size)
    hparams.set_param('missing_output_value', missing_output_value)

    for feature_idx in range(len(per_feature_values)):
      feature_name = feature_names[feature_idx]
      final_lattice_size, missing_output_value = self._calibration_params(
          *per_feature_values[feature_idx])
      lattice_size = per_feature_values[feature_idx][0]
      hparams.set_feature_param(feature_name, 'calibration_output_min', 0)
      hparams.set_feature_param(feature_name, 'calibration_output_max',
                                lattice_size - 1)
      hparams.set_feature_param(feature_name, 'final_lattice_size',
                                final_lattice_size)
      hparams.set_feature_param(feature_name, 'missing_output_value',
                                missing_output_value)
    return hparams

  def _calibration_params(self, lattice_size, missing_input_value,
                          missing_vertex):
    """Returns final_lattice_size and missing_output_value."""
    if missing_input_value is None or not missing_vertex:
      return lattice_size, None

    # Last vertex of the lattice is reserved for missing values.
    return lattice_size + 1, lattice_size

  def calibration_structure_builder(self, columns_to_tensors, hparams):
    """Returns the calibration structure of the model. See base class."""
    return None

  def prediction_builder_from_calibrated(
      self, mode, per_dimension_feature_names, hparams, calibrated):
    """Construct the prediciton."""
    self.check_hparams(hparams, adjusted=True)
    lattice_sizes = [
        hparams.get_feature_param(f, 'final_lattice_size')
        for f in per_dimension_feature_names
    ]
    lattice_monotonic = [(hparams.get_feature_param(f, 'monotonicity') != 0)
                         for f in per_dimension_feature_names]
    interpolation_type = hparams.get_param('interpolation_type')

    # Setup the regularization.
    regularizer_amounts = {}
    for reg_name in regularizers.LATTICE_MULTI_DIMENSIONAL_REGULARIZERS:
      regularizer_amounts[reg_name] = hparams.get_param(
          'lattice_{}'.format(reg_name))
    for reg_name in regularizers.LATTICE_ONE_DIMENSIONAL_REGULARIZERS:
      regularizer_amounts[reg_name] = [
          hparams.get_feature_param(feature_name, 'lattice_{}'.format(reg_name))
          for feature_name in per_dimension_feature_names
      ]

    packed_results = lattice_layers.lattice_layer(
        calibrated,
        lattice_sizes,
        is_monotone=lattice_monotonic,
        interpolation_type=interpolation_type,
        lattice_initializer=self.lattice_initializers_fn_,
        **regularizer_amounts)
    (prediction, _, projection_ops, regularization) = packed_results
    # Returns prediction Tensor, projection ops, and regularization.
    return prediction, projection_ops, regularization


def calibrated_lattice_classifier(feature_columns=None,
                                  model_dir=None,
                                  quantiles_dir=None,
                                  keypoints_initializers_fn=None,
                                  optimizer=None,
                                  config=None,
                                  hparams=None,
                                  head=None,
                                  weight_column=None):
  """Calibrated lattice classifier binary model.



  This model uses a piecewise lattice calibration function on each of the
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

  estimator = calibrated_lattice.CalibratedLatticeClassifier(
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
      `pwl_calibrators_layers.calculate_quantiles_for_keypoints` in a separate
      invocation of your program. If you don't want to use quantiles, you can
      set `keypoints_initializer` instead.
    keypoints_initializers_fn: if you know the distribution of your
      input features you can provide that directly instead of `quantiles_dir`.
      See `pwl_calibrators_layers.uniform_keypoints_for_signal`. It must be
      as a closure that when called will return a pair of tensors with
      keypoints input and output initializes. Alternatively can be given as
      a dict mapping feature name to keypoints_initializers_fn, so one
      can have one initialization per feature. It uses a closure instead of
      the tensors themselves because the graph has to be created at the time
      the model is being build, which happens at a later time.
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
    A `CalibratedLatticeClassifier` estimator.

  Raises:
    ValueError: invalid parameters.
    KeyError: type of feature not supported.
  """
  return _CalibratedLattice(
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


def calibrated_lattice_regressor(feature_columns=None,
                                 model_dir=None,
                                 quantiles_dir=None,
                                 keypoints_initializers_fn=None,
                                 optimizer=None,
                                 config=None,
                                 hparams=None,
                                 head=None,
                                 weight_column=None):
  """Calibrated lattice estimator (model) for regression.

  This model uses a piecewise lattice calibration function on each of the
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

  estimator = calibrated_lattice.calibrated_lattice_regressor(
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
      `pwl_calibrators_layers.calculate_quantiles_for_keypoints` in a separate
      invocation of your program. If you don't want to use quantiles, you can
      set `keypoints_initializer` instead.
    keypoints_initializers_fn: if you know the distribution of your
      input features you can provide that directly instead of `quantiles_dir`.
      See `pwl_calibrators_layers.uniform_keypoints_for_signal`. It must be
      as a closure that when called will return a pair of tensors with
      keypoints input and output initializes. Alternatively can be given as
      a dict mapping feature name to keypoints_initializers_fn, so one
      can have one initialization per feature. It uses a closure instead of
      the tensors themselves because the graph has to be created at the time
      the model is being build, which happens at a later time.
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
    A `CalibratedLatticeRegressor` estimator.

  Raises:
    ValueError: invalid parameters.
    KeyError: type of feature not supported.
  """
  return _CalibratedLattice(
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
