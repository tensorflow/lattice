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
"""CalibratedEtl canned estimators."""
import copy

# Dependency imports

from tensorflow_lattice.python.estimators import calibrated as calibrated_lib
from tensorflow_lattice.python.estimators import hparams as tfl_hparams
from tensorflow_lattice.python.lib import keypoints_initialization
from tensorflow_lattice.python.lib import lattice_layers
from tensorflow_lattice.python.lib import monotone_linear_layers
from tensorflow_lattice.python.lib import pwl_calibration_layers
from tensorflow_lattice.python.lib import regularizers
from tensorflow_lattice.python.lib import tools

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope

_EPSILON = 1e-7


def _calibration_layer(input_tensor, input_dim, input_min, input_max,
                       num_keypoints, output_min, output_max):
  """Create an intermediate calibration layer."""
  init_keypoints = keypoints_initialization.uniform_keypoints_for_signal(
      num_keypoints=num_keypoints,
      input_min=input_min,
      input_max=input_max,
      output_min=output_min,
      output_max=output_max,
      dtype=input_tensor.dtype)
  packed_results = pwl_calibration_layers.calibration_layer(
      input_tensor,
      num_keypoints=num_keypoints,
      keypoints_initializers=[init_keypoints] * input_dim,
      bound=True,
      monotonic=+1)
  (calibrated_input_tensor, projection_ops, _) = packed_results
  return (calibrated_input_tensor, projection_ops)


def _ensemble_lattices_layer(
    input_tensor, input_dim, output_dim, interpolation_type, calibration_min,
    calibration_max, calibration_num_keypoints, num_lattices, lattice_rank,
    lattice_size, regularizer_amounts, is_monotone):
  """Creates an ensemble of lattices layer."""
  projections = []
  structures = [
      range(lattice_cnt * lattice_rank, (lattice_cnt + 1) * lattice_rank)
      for lattice_cnt in range(num_lattices)
  ]
  calibrated_input, proj = _calibration_layer(
      input_tensor,
      input_dim,
      calibration_min,
      calibration_max,
      calibration_num_keypoints,
      output_min=0,
      output_max=lattice_size - 1)
  if proj:
    projections += proj
  lattice_outputs, _, proj, reg = lattice_layers.ensemble_lattices_layer(
      calibrated_input, [lattice_size] * input_dim,
      structures,
      is_monotone=is_monotone,
      output_dim=output_dim,
      interpolation_type=interpolation_type,
      **regularizer_amounts)
  if proj:
    projections += proj
  return lattice_outputs, projections, reg


def _embedded_lattices(calibrated_input_tensor,
                       input_dim,
                       output_dim,
                       interpolation_type,
                       monotonic_num_lattices,
                       monotonic_lattice_rank,
                       monotonic_lattice_size,
                       non_monotonic_num_lattices,
                       non_monotonic_lattice_rank,
                       non_monotonic_lattice_size,
                       linear_embedding_calibration_min,
                       linear_embedding_calibration_max,
                       linear_embedding_calibration_num_keypoints,
                       regularizer_amounts,
                       is_monotone=None):
  """Creates an ensemble of lattices with a linear embedding.

  This function constructs the following deep lattice network:
  calibrated_input -> linear_embedding -> calibration -> ensemble of lattices.
  Then ensemble of lattices' output are averaged and bias term is added to make
  a final prediction.

  ensemble of lattices is consists of two parts: monotonic lattices and
  non-monotonic lattices. The input to the monotonic lattices is an output of
  linear_embedding that contains both monotonic and non-monotonic
  calibrated_input. All inputs to the monotonic lattices are set to be monotonic
  to preserve end-to-end monotonicity in the monotonic feature.
  The input to the non-monotonic lattices is an output of linear_embedding that
  only contains non-monotonic calibrated_input. All inputs to the non-monotonic
  lattices are set to be non-monotonic, since we do not need to guarantee
  monotonicity.

  Args:
    calibrated_input_tensor: [batch_size, input_dim] tensor.
    input_dim: (int) input dimnension.
    output_dim: (int) output dimension.
    interpolation_type: defines whether the lattice will interpolate using the
      full hypercube or only the simplex ("hyper-triangle") around the point
      being evaluated. Valid values: 'hypercube' or 'simplex'
    monotonic_num_lattices: (int) number of monotonic lattices in the ensemble
      lattices layer.
    monotonic_lattice_rank: (int) number of inputs to each monotonic lattice in
      the ensemble lattices layer.
    monotonic_lattice_size: (int) lattice cell size for each monotonic lattice
      in the ensemble lattices layer.
    non_monotonic_num_lattices: (int) number of non monotonic lattices in the
      ensemble lattices layer.
    non_monotonic_lattice_rank: (int) number of inputs to each non monotonic
      lattice in the ensemble lattices layer.
    non_monotonic_lattice_size: (int) lattice cell size for each non monotonic
      lattice in the ensemble lattices layer.
    linear_embedding_calibration_min: (float) a minimum input keypoints value
      for linear_embedding calibration.
    linear_embedding_calibration_max: (float) a maximum input keypoints value
      for linear_embedding calibration.
    linear_embedding_calibration_num_keypoints: (int) a number of eypoints for
      linear_embedding calibration.
    regularizer_amounts: Dict of regularization amounts passed as keyword args
      to regularizers.lattice_regularization().
    is_monotone: (bool, list of booleans) is_monotone[k] == true then
      calibrated_input_tensor[:, k] is considered to be a monotonic input.
  Returns:
    A tuple of (output_tensor, projection_ops, regularization).
  Raises:
    ValueError: If there is no non-monotonic inputs but
    non_monotonic_num_lattices is not zero.
  """
  projections = []
  regularization = None

  # Explictly assign number of lattices to zero for any empty cases.
  if not monotonic_num_lattices:
    monotonic_num_lattices = 0
  if not non_monotonic_num_lattices:
    non_monotonic_num_lattices = 0

  # Step 1. Create a linear embedding.
  if monotonic_num_lattices:
    monotonic_embedding_dim = monotonic_num_lattices * monotonic_lattice_rank
  else:
    monotonic_num_lattices = 0
    monotonic_embedding_dim = 0
  if non_monotonic_num_lattices:
    non_monotonic_embedding_dim = (
        non_monotonic_num_lattices * non_monotonic_lattice_rank)
  else:
    non_monotonic_num_lattices = 0
    non_monotonic_embedding_dim = 0

  if is_monotone is not None:
    is_monotone = tools.cast_to_list(is_monotone, input_dim, 'is_monotone')
  with variable_scope.variable_scope('linear_embedding'):
    packed_results = monotone_linear_layers.split_monotone_linear_layer(
        calibrated_input_tensor,
        input_dim,
        monotonic_embedding_dim,
        non_monotonic_embedding_dim,
        is_monotone=is_monotone)
    (monotonic_output, _, non_monotonic_output, _, proj, _) = packed_results
    if proj is not None:
      projections.append(proj)

  # Step 2. Create ensemble of monotonic lattices.
  if monotonic_num_lattices == 0:
    m_lattice_outputs = None
  else:
    with variable_scope.variable_scope('monotonic_lattices'):
      m_lattice_outputs, projs, reg = _ensemble_lattices_layer(
          monotonic_output,
          monotonic_embedding_dim,
          output_dim,
          interpolation_type,
          linear_embedding_calibration_min,
          linear_embedding_calibration_max,
          linear_embedding_calibration_num_keypoints,
          monotonic_num_lattices,
          monotonic_lattice_rank,
          monotonic_lattice_size,
          regularizer_amounts,
          is_monotone=True)
      if projs:
        projections += projs
      regularization = tools.add_if_not_none(regularization, reg)

  # Step 3. Construct non-monotonic ensembles.
  if non_monotonic_output is None and non_monotonic_num_lattices > 0:
    raise ValueError(
        'All input signals are monotonic but the number of non monotonic '
        'lattices is not zero.')
  if non_monotonic_num_lattices == 0:
    n_lattice_outputs = None
  else:
    with variable_scope.variable_scope('non_monotonic_lattices'):
      n_lattice_outputs, projs, reg = _ensemble_lattices_layer(
          non_monotonic_output,
          non_monotonic_embedding_dim,
          output_dim,
          interpolation_type,
          linear_embedding_calibration_min,
          linear_embedding_calibration_max,
          linear_embedding_calibration_num_keypoints,
          non_monotonic_num_lattices,
          non_monotonic_lattice_rank,
          non_monotonic_lattice_size,
          regularizer_amounts,
          is_monotone=False)
      if projs:
        projections += projs
      regularization = tools.add_if_not_none(regularization, reg)

  # Step 4. Take average to make a final prediction.
  with variable_scope.variable_scope('ensemble_average'):
    output = variable_scope.get_variable(
        name='ensemble_bias',
        initializer=[0.0] * output_dim,
        dtype=calibrated_input_tensor.dtype)
    if m_lattice_outputs:
      output += math_ops.divide(
          math_ops.add_n(m_lattice_outputs), monotonic_num_lattices)
    if n_lattice_outputs is not None:
      output += math_ops.divide(
          math_ops.add_n(n_lattice_outputs), non_monotonic_num_lattices)

  return (output, projections, regularization)


class _CalibratedEtl(calibrated_lib.Calibrated):
  """Base class for CalibratedEtl{Classifier|Regressor}."""

  def __init__(self,
               n_classes,
               feature_columns=None,
               model_dir=None,
               quantiles_dir=None,
               keypoints_initializers_fn=None,
               optimizer=None,
               config=None,
               hparams=None,
               feature_engineering_fn=None,
               head=None,
               weight_column=None):
    """Construct CalibrateEtlClassifier/Regressor."""
    if not hparams:
      hparams = tfl_hparams.CalibratedEtlHParams([])
    self.check_hparams(hparams)
    hparams = self._adjust_calibration_params(hparams)

    super(_CalibratedEtl,
          self).__init__(n_classes, feature_columns, model_dir, quantiles_dir,
                         keypoints_initializers_fn, optimizer, config, hparams,
                         head, weight_column, 'etl')
    # After initialization, we expect model_dir exists.
    if self._model_dir is None:
      raise ValueError('model_dir is not created')

  def _check_lattices_params(self, hparams):
    """Check lattice parameters."""
    monotonic_num_lattices = hparams.get_param('monotonic_num_lattices')
    monotonic_lattice_rank = hparams.get_param('monotonic_lattice_rank')
    monotonic_lattice_size = hparams.get_param('monotonic_lattice_size')
    non_monotonic_num_lattices = hparams.get_param('non_monotonic_num_lattices')
    non_monotonic_lattice_rank = hparams.get_param('non_monotonic_lattice_rank')
    non_monotonic_lattice_size = hparams.get_param('non_monotonic_lattice_size')

    error_messages = []
    if monotonic_num_lattices is None and non_monotonic_num_lattices is None:
      error_messages.append('At least one of monotonic_num_lattices or '
                            'non_monotonic_num_lattices should be provided')

    if monotonic_num_lattices:
      if monotonic_lattice_rank is None:
        error_messages.append('monotonic_lattice_rank should be specified.')
      if monotonic_lattice_size is None:
        error_messages.append('monotonic_lattice_size should be specified.')
      elif monotonic_lattice_size < 2:
        error_messages.append(
            'monotonic_lattice_size cannot be less than 2, but got %d' %
            monotonic_lattice_size)

    if non_monotonic_num_lattices:
      if non_monotonic_lattice_rank is None:
        error_messages.append('non_monotonic_lattice_rank should be specified.')
      if non_monotonic_lattice_size is None:
        error_messages.append('non_monotonic_lattice_size should be specified.')
      elif non_monotonic_lattice_size < 2:
        error_messages.append(
            'non_monotonic_lattice_size cannot be less than 2, but got %d' %
            non_monotonic_lattice_size)

    return error_messages

  def _adjust_calibration_params(self, hparams):
    """Makes sure we have the correct input calibration set up."""
    hparams = copy.deepcopy(hparams)
    hparams.set_param('calibration_output_min', -1.)
    hparams.set_param('calibration_output_max', 1.)
    return hparams

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

  def _check_per_feature_param_configuration(self, monotonicity,
                                             calibration_bound):
    """Check parameter configuration and returns the error messages."""
    error_messages = []

    if monotonicity not in {-1, 0, +1}:
      error_messages.append('monotonicity should be an integer {-1, 0, +1} '
                            'but is %s' % monotonicity)

    if not calibration_bound:
      error_messages.append(
          'A deep lattice network expects an bounded input from a calibration '
          'layer, but calibration_bound is set to be False')

    return error_messages

  def check_hparams(self, hparams):
    """Check pre-conditions of hparams.

    Args:
      hparams: (tfl_hparams.CalibratedEtlHParams) Hyperparameter to be
      examined.

    Raises:
      ValueError: If the hyperparameter configuration is invalid, for example
      calibration_monotonic is None, but lattice_monotonic is True, then raise
      the error with a root cause.
    """
    error_messages = self._check_lattices_params(hparams)
    # Check global params.
    feature_names = hparams.get_feature_names()
    packed_feature_values = hparams.get_global_and_feature_params(
        ['monotonicity', 'calibration_bound'], feature_names)
    default_feature_values, per_feature_values = packed_feature_values
    param_error_messages = self._check_per_feature_param_configuration(
        *default_feature_values)
    if param_error_messages:
      error_messages.append('Error message for default feature param:')
      error_messages += param_error_messages

    # Check per feature params. hparams.get_feature_names()  will only return
    # feature names that sets per feature parameters.
    for feature_idx in range(len(per_feature_values)):
      param_error_messages = self._check_per_feature_param_configuration(
          *per_feature_values[feature_idx])
      if param_error_messages:
        error_messages.append(
            'Error message for %s feature param:' % feature_names[feature_idx])
        error_messages += param_error_messages

    if error_messages:
      raise ValueError(
          'Hyperparameter configuration cannot be used in the calibrated '
          'etl estimator. Error messages report the issue per feature, but'
          ' the parameter may be inherited from global parameter.\nDetailed '
          'error messsages\n%s' % '\n'.join(error_messages))

  def calibration_structure_builder(self, columns_to_tensors, hparams):
    """Returns the calibration structure of the model. See base class."""
    return None

  def prediction_builder_from_calibrated(
      self, mode, per_dimension_feature_names, hparams, calibrated):
    """Construct the prediciton."""
    self.check_hparams(hparams)
    lattice_monotonic = [(hparams.get_feature_param(f, 'monotonicity') != 0)
                         for f in per_dimension_feature_names]
    monotonic_num_lattices = hparams.get_param('monotonic_num_lattices')
    monotonic_lattice_rank = hparams.get_param('monotonic_lattice_rank')
    monotonic_lattice_size = hparams.get_param('monotonic_lattice_size')
    non_monotonic_num_lattices = hparams.get_param('non_monotonic_num_lattices')
    non_monotonic_lattice_rank = hparams.get_param('non_monotonic_lattice_rank')
    non_monotonic_lattice_size = hparams.get_param('non_monotonic_lattice_size')
    linear_embedding_calibration_min = hparams.get_param(
        'linear_embedding_calibration_min')
    linear_embedding_calibration_max = hparams.get_param(
        'linear_embedding_calibration_max')
    linear_embedding_calibration_num_keypoints = hparams.get_param(
        'linear_embedding_calibration_num_keypoints')
    interpolation_type = hparams.get_param('interpolation_type')

    # Setup the regularization.
    regularizer_amounts = {}
    for regularizer_name in regularizers.LATTICE_REGULARIZERS:
      regularizer_amounts[regularizer_name] = hparams.get_param(
          'lattice_{}'.format(regularizer_name))

    input_dim = len(per_dimension_feature_names)
    output_dim = 1
    return _embedded_lattices(
        calibrated,
        input_dim,
        output_dim,
        interpolation_type,
        monotonic_num_lattices,
        monotonic_lattice_rank,
        monotonic_lattice_size,
        non_monotonic_num_lattices,
        non_monotonic_lattice_rank,
        non_monotonic_lattice_size,
        linear_embedding_calibration_min,
        linear_embedding_calibration_max,
        linear_embedding_calibration_num_keypoints,
        regularizer_amounts,
        is_monotone=lattice_monotonic)


def calibrated_etl_classifier(feature_columns=None,
                              model_dir=None,
                              quantiles_dir=None,
                              keypoints_initializers_fn=None,
                              optimizer=None,
                              config=None,
                              hparams=None,
                              head=None,
                              weight_column=None):
  """Calibrated etl binary classifier model.



  This model uses a piecewise lattice calibration function on each of the
  inputs (parametrized) and then feeds them to ensemble of random lattices.
  num_lattices and lattice_rank (number of inputs to each lattice) must be
  specified in the hyperparameter. Optionally calibration can be made monotonic.

  It usually requires a preprocessing step on the data, to calculate the
  quantiles of each used feature. This can be done locally or in one worker
  only before training, in a separate invocation of your program (or directly).
  Typically this can be saved (`save_dir` parameter) to the same
  directory where the data is.

  Hyper-parameters are given in the form of the object
  tfl_hparams.CalibrationEtlHParams. lattice_rank and num_lattices must
  be specified; there would be no default value for this. It also takes in
  per-feature parameters.

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

  hparams = hparams.CalibratedEtlHparams(num_lattices=10, lattice_rank=2)
  estimator = calibrated_etl.calibrated_etl_classifier(
    feature_columns=feature_columns, hparams=hparams)
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
    model_dir: Directory to save model parameters, graphs and etc. This can
      also be used to load checkpoints from the directory into an estimator to
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
    hparams: an instance of tfl_hparams.CalibrationEtlHParams. If set to
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
    A `calibrated_etl_classifier` estimator.

  Raises:
    ValueError: invalid parameters.
    KeyError: type of feature not supported.
  """
  return _CalibratedEtl(
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


def calibrated_etl_regressor(feature_columns=None,
                             model_dir=None,
                             quantiles_dir=None,
                             keypoints_initializers_fn=None,
                             optimizer=None,
                             config=None,
                             hparams=None,
                             head=None,
                             weight_column=None):
  """Calibrated etl regressor model.

  This model uses a piecewise lattice calibration function on each of the
  inputs (parametrized) and then feeds them to ensemble of random lattices.
  num_lattices and lattice_rank (number of inputs to each lattice) must be
  specified in the hyperparameter. Optionally calibration can be made monotonic.

  It usually requires a preprocessing step on the data, to calculate the
  quantiles of each used feature. This can be done locally or in one worker
  only before training, in a separate invocation of your program (or directly).
  Typically this can be saved (`save_dir` parameter) to the same
  directory where the data is.

  Hyper-parameters are given in the form of the object
  tfl_hparams.CalibrationEtlHParams. lattice_rank and num_lattices must
  be specified; there would be no default value for this. It also takes in
  per-feature parameters.

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

  hparams = hparams.CalibratedEtlHparams(num_lattices=10, lattice_rank=2)
  estimator = calibrated_etl.calibrated_etl_classifier(
    feature_columns=feature_columns, hparams=hparams)
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
    model_dir: Directory to save model parameters, graphs and etc. This can
      also be used to load checkpoints from the directory into an estimator to
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
    hparams: an instance of tfl_hparams.CalibrationEtlHParams. If set to
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
    A `calibrated_etl_regressor` estimator.

  Raises:
    ValueError: invalid parameters.
    KeyError: type of feature not supported.
  """
  return _CalibratedEtl(
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
