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
"""Base class for TensorFlow Lattice estimators with input calibration."""
import abc

# Dependency imports
import six

from tensorflow_lattice.python.estimators import hparams as tf_lattice_hparams
from tensorflow_lattice.python.lib import keypoints_initialization
from tensorflow_lattice.python.lib import pwl_calibration_layers
from tensorflow_lattice.python.lib import tools

from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training
from tensorflow.python.training import training_util

# Scope for variable names.
_SCOPE_CALIBRATED_TENSORFLOW_LATTICE = "calibrated_tf_lattice_model"
_SCOPE_INPUT_CALIBRATION = "input_calibration"
_SCOPE_TRAIN_OP = "calibrated_tf_lattice_train_op"


def _get_feature_dict(features):
  if isinstance(features, dict):
    return features
  return {"": features}


def _get_optimizer(optimizer, hparams):
  if callable(optimizer):
    return optimizer(learning_rate=hparams.learning_rate)
  else:
    return optimizer


def _get_per_feature_dict(hparams, param_name, default_value=None):
  """Creates dict with values returned by hparams for param for each feature."""
  if not issubclass(type(hparams), tf_lattice_hparams.PerFeatureHParams):
    raise ValueError(
        "hparams passed to Estimator is not a subclass of "
        "tensorflow_lattice.PerFeatureHParams, it can't figure out parameters "
        "for calibration")
  return {
      feature_name: hparams.get_feature_param(feature_name, param_name,
                                              default_value)
      for feature_name in hparams.get_feature_names()
  }


def _call_keypoints_inializers_fn(keypoints_initializers_fn):
  """Call the closure and check/return results."""
  if callable(keypoints_initializers_fn):
    kp_init = keypoints_initializers_fn()
    if (len(kp_init) != 2 or not issubclass(type(kp_init[0]), ops.Tensor) or
        not issubclass(type(kp_init[1]), ops.Tensor)):
      raise ValueError(
          "invalid value returned by keypoints_initializers_fn, expected a "
          "pair of tensors, got %s" % kp_init)
    return kp_init
  elif isinstance(keypoints_initializers_fn, dict):
    return {
        k: _call_keypoints_inializers_fn(v)
        for k, v in six.iteritems(keypoints_initializers_fn)
    }
  else:
    raise ValueError("Unknown type for keypoints_initializers_fn: %s" %
                     type(keypoints_initializers_fn))


def _update_keypoints(feature_name, asked_keypoints, kp_init_keypoints):
  """Updates num_keypoints according to availability."""
  if not asked_keypoints or kp_init_keypoints == asked_keypoints:
    # Meet asked_keypoints if no calibration was asked for this feature,
    # or if the correct number of kp_init_keypoints are available.
    return asked_keypoints
  if kp_init_keypoints < asked_keypoints:
    # If fewer keypoints were returned by init functions, emit debug
    # message and return those available.
    logging.debug("Using {} keypoints for calibration of {} instead of "
                  "the requested {}".format(kp_init_keypoints, feature_name,
                                            asked_keypoints))
    return kp_init_keypoints
  raise ValueError("Calibration initialization returned more keypoints ({}) "
                   "than requested ({}) for feature {}".format(
                       kp_init_keypoints, asked_keypoints, feature_name))


def input_calibration_layer_from_hparams(columns_to_tensors,
                                         feature_columns,
                                         hparams,
                                         quantiles_dir=None,
                                         keypoints_initializers=None,
                                         name=None,
                                         dtype=dtypes.float32):
  """Creates a calibration layer for the input using hyper-parameters.

  Similar to `input_calibration_layer` but reads its parameters from a
  `CalibratedHParams` object.

  Args:
    columns_to_tensors: A mapping from feature name to tensors. 'string' key
      means a base feature (not-transformed). If feature_columns is not set
      these are the features calibrated. Otherwise the transformed
      feature_columns are the ones calibrated.
    feature_columns: An iterable containing all the feature columns used by the
      model. Optional, if not set the model will use all features given in
      columns_to_tensors. All items in the set should be instances of
      classes derived from `FeatureColumn`.
    hparams: Hyper-parameters, need to inherit from `CalibratedHParams`.
      It is also changed to include all feature names found in
      `feature_columns`. See `CalibratedHParams` and `input_calibration_layer`
      for descriptions of how these hyper-parameters work.
    quantiles_dir: location where quantiles for the data was saved. Typically
      the same directory as the training data. These quantiles can be
      generated with `pwl_calibration_layers.calculate_quantiles_for_keypoints`,
      maybe in a separate invocation of your program. Different models that
      share the same quantiles information -- so this needs to be generated only
      once when hyper-parameter tuning. If you don't want to use quantiles, you
      can set `keypoints_initializers` instead.
    keypoints_initializers: if you know the distribution of your
      input features you can provide that directly instead of `quantiles_dir`.
      See `pwl_calibrators_layers.uniform_keypoints_for_signal`. It must be
      a pair of tensors with keypoints inputs and outputs to use for
      initialization (must match `num_keypoints` configured in `hparams`).
      Alternatively can be given as a dict mapping feature name to pairs,
      for initialization per feature. If `quantiles_dir` and
      `keypoints_initializer` are set, the later takes precendence, and the
      features for which `keypoints_initializers` are not defined fallback to
      using the quantiles found in `quantiles_dir`.
    name: Name scope for layer.
    dtype: If any of the scalars are not given as tensors, they are converted
      to tensors with this dtype.

  Returns:
    A tuple of:
    * calibrated tensor of shape [batch_size, sum(features dimensions)].
    * list of the feature names in the order they feature in the calibrated
      tensor. A name may appear more than once if the feature is
      multi-dimension (for instance a multi-dimension embedding)
    * list of projection ops, that must be applied at each step (or every so
      many steps) to project the model to a feasible space: used for bounding
      the outputs or for imposing monotonicity. Empty if none are requested.
    * None or tensor with regularization loss.

  Raises:
    ValueError: if dtypes are incompatible.


  """
  with ops.name_scope(name or "input_calibration_layer_from_hparams"):

    # Sort out list of feature names.
    unique_feature_names = tools.get_sorted_feature_names(
        columns_to_tensors=columns_to_tensors, feature_columns=feature_columns)
    for feature_name in unique_feature_names:
      hparams.add_feature(feature_name)

    # Get per-feature parameters.
    num_keypoints = _get_per_feature_dict(hparams, "num_keypoints")
    calibration_output_min = _get_per_feature_dict(hparams,
                                                   "calibration_output_min")
    calibration_output_max = _get_per_feature_dict(hparams,
                                                   "calibration_output_max")
    calibration_bound = _get_per_feature_dict(hparams, "calibration_bound")
    monotonicity = _get_per_feature_dict(hparams, "monotonicity")
    missing_input_values = _get_per_feature_dict(hparams, "missing_input_value")
    missing_output_values = _get_per_feature_dict(hparams,
                                                  "missing_output_value")
    calibration_l1_regs = None
    calibration_l2_regs = None
    calibration_l1_laplacian_regs = None
    calibration_l2_laplacian_regs = None

    # Define keypoints_initializers to use in this invocation of model_fn.
    kp_init = None
    if quantiles_dir is not None:
      # Skip features for which an explicit initializer was given.
      if isinstance(keypoints_initializers, dict):
        quantiles_feature_names = []
        for name in unique_feature_names:
          if name not in keypoints_initializers:
            quantiles_feature_names.append(name)
      else:
        quantiles_feature_names = unique_feature_names

      # Read initializers from quantiles_dir, for those not already
      # defined.
      #
      # Notice that output_min and output_max won't matter much if
      # they are not bounded, since they will be adjusted during training.
      kp_init = keypoints_initialization.load_keypoints_from_quantiles(
          feature_names=quantiles_feature_names,
          save_dir=quantiles_dir,
          num_keypoints=num_keypoints,
          output_min=calibration_output_min,
          output_max=calibration_output_max,
          dtype=dtype)

      # Merge with explicit initializers.
      if isinstance(keypoints_initializers, dict):
        kp_init.update(keypoints_initializers)

    else:
      # Take given initializers.
      kp_init = keypoints_initializers

    # Update num_keypoints according to keypoints actually used by the
    # initialization functions: some initialization functions may change
    # them, for instance if there are not enough unique values.
    if isinstance(kp_init, dict):
      # One initializer (kp_init) per feature.
      for (feature_name, initializers) in six.iteritems(kp_init):
        kp_init_keypoints = initializers[0].shape.as_list()[0]
        num_keypoints[feature_name] = _update_keypoints(
            feature_name, num_keypoints[feature_name], kp_init_keypoints)
    else:
      # Check generic initializer (kp_init).
      kp_init_keypoints = kp_init[0].shape.as_list()[0]
      for feature_name in six.iterkeys(num_keypoints):
        num_keypoints[feature_name] = _update_keypoints(
            feature_name, num_keypoints[feature_name], kp_init_keypoints)

    # Setup the regularization.
    calibration_l1_regs = _get_per_feature_dict(hparams, "calibration_l1_reg")
    calibration_l2_regs = _get_per_feature_dict(hparams, "calibration_l2_reg")
    calibration_l1_laplacian_regs = _get_per_feature_dict(
        hparams, "calibration_l1_laplacian_reg")
    calibration_l2_laplacian_regs = _get_per_feature_dict(
        hparams, "calibration_l2_laplacian_reg")

    return pwl_calibration_layers.input_calibration_layer(
        columns_to_tensors=columns_to_tensors,
        feature_columns=feature_columns,
        num_keypoints=num_keypoints,
        keypoints_initializers=kp_init,
        bound=calibration_bound,
        monotonic=monotonicity,
        missing_input_values=missing_input_values,
        missing_output_values=missing_output_values,
        l1_reg=calibration_l1_regs,
        l2_reg=calibration_l2_regs,
        l1_laplacian_reg=calibration_l1_laplacian_regs,
        l2_laplacian_reg=calibration_l2_laplacian_regs)


class _ProjectionHook(session_run_hook.SessionRunHook):
  """SessionRunHook to project to feasible space after each step."""

  def __init__(self):
    self._projection_ops = []

  def set_projection_ops(self, projection_ops):
    """Needs to be called in model_fn function, with ops to project."""
    self._projection_ops = projection_ops

  def after_run(self, run_context, run_values):
    if self._projection_ops:
      run_context.session.run(self._projection_ops)


class Calibrated(estimator.Estimator):
  """Base class for TensorFlow Lattice models.

  It provides preprocessing and calibration of the input features, and
  set up the hook that runs projections at each step -- typically used
  to project parameters to be monotone and within bounds.

  To extend one has to implement the method prediction_builder()
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self,
               n_classes,
               feature_columns=None,
               model_dir=None,
               quantiles_dir=None,
               keypoints_initializers_fn=None,
               optimizer=None,
               config=None,
               hparams=None,
               name=None):
    """Construct CalibrateLinearClassifier/Regressor.

    Args:
      n_classes: Number of classes, set to 0 if used for regression.
      feature_columns: Optional, if not set the model will use all features
        returned by input_fn. An iterable containing all the feature
        columns used by the model. All items in the set should be instances of
        classes derived from `FeatureColumn`.
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
      hparams: an instance of tf_lattice_hparams.CalibrationHParams. If set to
        None default parameters are used.
      name: Name to be used as top-level variable scope for model.

    Returns:
      A `Calibrated` base class: it doesn't fully implement an estimator yet,
      and needs to be extended.

    Raises:
      ValueError: invalid parameters.
      KeyError: type of feature not supported.
    """
    self._feature_columns = feature_columns
    self._quantiles_dir = quantiles_dir
    self._keypoints_initializers_fn = keypoints_initializers_fn
    self._optimizer = optimizer

    self._config = config
    if self._optimizer is None:
      self._optimizer = training.AdamOptimizer
    self._hparams = hparams
    if self._hparams is None:
      raise ValueError("hparams cannot be none")
    if not issubclass(
        type(self._hparams), tf_lattice_hparams.CalibratedHParams):
      raise ValueError("hparams is not instance of hparams.CalibratedHParams, "
                       "got type(params)=%s" % type(self._hparams))
    self._name = name
    self._n_classes = n_classes

    self._dtype = dtypes.float32


    if n_classes == 0:
      self._head = (
          head_lib.  # pylint: disable=protected-access
          _regression_head_with_mean_squared_error_loss(label_dimension=1))
    elif n_classes == 2:
      self._head = (
          head_lib.  # pylint: disable=protected-access
          _binary_logistic_head_with_sigmoid_cross_entropy_loss())
    else:
      raise ValueError("Invalid value for n_classes=%d" % n_classes)

    super(Calibrated, self).__init__(
        model_fn=self._calibrated_model_builder(),
        model_dir=model_dir,
        config=config)

    # Make sure model directory exists after initialization.
    # Notice self.model_dir is set by Estimator class.
    file_io.recursive_create_dir(self.model_dir)

    self._projection_hook = _ProjectionHook()


  @abc.abstractmethod
  def prediction_builder(self, mode, per_dimension_feature_names, hparams,
                         calibrated):
    """Method to be specialized that builds the prediction graph.

    Args:
      mode: Estimator's `ModeKeys`.
      per_dimension_feature_names: Name of features. The ordering should be
        matched with the ordering in calibrated feature tensor. Notice
        feature_names may be repeated, if some of the features were originally
        multi-dimensional.
      hparams: hyperparameters passed to object constructor.
      calibrated: calibrated feature tensor, shaped `[batch_size, num_features]`

    Returns:
      prediction_tensor: shaped `[batch_size/?,1]` for regression or binary
        classification, or `[batch_size, n_classes]` for multi-class
        classifiers. For classifier this will be the logit(s) value(s).
      projection_ops: list of projection ops to be applied after each batch,
        or None.
      regularization_loss: loss related to regularization or None.
    """
    raise NotImplementedError(
        "This method is expected to be implemented in a child class")

  def _calibrated_model_builder(self):
    """Returns a model_fn function, uses attributes in object (`self`)."""


    def model_fn(features, labels, mode, config):  # pylint: disable=unused-argument
      """Creates the prediction, loss, and train ops.

      Args:
        features: A dictionary of tensors keyed by the feature name.
        labels: A tensor representing the label.
        mode: The execution mode, as defined in model_fn_lib.ModeKeys.
        config: Optional configuration object. Will receive what is passed
          to Estimator in `config` parameter, or the default `config`.
          Allows updating things in your model_fn based on configuration
          such as `num_ps_replicas`.
      Returns:
        ModelFnOps, with the predictions, loss, and train_op.

      Raises:
        ValueError: if incompatible parameters are given, or if the keypoints
          initializers given during construction return invalid number of
          keypoints.
      """
      with variable_scope.variable_scope(
          "/".join([_SCOPE_CALIBRATED_TENSORFLOW_LATTICE, self._name])):

        if mode == model_fn_lib.ModeKeys.TRAIN:
          if (self._quantiles_dir is None and
              self._keypoints_initializers_fn is None):
            raise ValueError(
                "At least one of quantiles_dir or keypoints_initializers_fn "
                "must be given for training")

        # If keypoint_initializer closures were given, now it materializes
        # them, into the initializers tensors.
        kp_init_explicit = None
        if self._keypoints_initializers_fn is not None:
          kp_init_explicit = _call_keypoints_inializers_fn(
              self._keypoints_initializers_fn)

        # Calibrate.
        (calibrated, per_dimension_feature_names, projection_ops,
         regularization) = (input_calibration_layer_from_hparams(
             features,
             feature_columns=self._feature_columns,
             hparams=self._hparams,
             quantiles_dir=self._quantiles_dir,
             keypoints_initializers=kp_init_explicit,
             name=_SCOPE_INPUT_CALIBRATION,
             dtype=self._dtype))
        (prediction, prediction_projections,
         prediction_regularization) = self.prediction_builder(
             mode, per_dimension_feature_names, self._hparams, calibrated)
        projection_ops = tools.add_if_not_none(projection_ops,
                                               prediction_projections)
        regularization = tools.add_if_not_none(regularization,
                                               prediction_regularization)

        def _train_op_fn(loss):
          """Returns train_op tensor if TRAIN mode, or None."""
          train_op = None
          if mode == model_fn_lib.ModeKeys.TRAIN:
            if regularization is not None:
              loss += regularization
            optimizer = _get_optimizer(self._optimizer, self._hparams)
            train_op = optimizer.minimize(
                loss,
                global_step=training_util.get_global_step(),
                name=_SCOPE_TRAIN_OP)
            self._projection_hook.set_projection_ops(projection_ops)
          return train_op

        # Use head to generate model_fn outputs.
        estimator_spec = self._head.create_estimator_spec(
            features=features,
            labels=labels,
            mode=mode,
            train_op_fn=_train_op_fn,
            logits=prediction)
        # Update training hooks to include projection_hook in the training mode.
        if mode == model_fn_lib.ModeKeys.TRAIN:
          updated_training_hooks = (estimator_spec.training_hooks +
                                    (self._projection_hook,))
          estimator_spec = estimator_spec._replace(
              training_hooks=updated_training_hooks)

        return estimator_spec

    return model_fn
