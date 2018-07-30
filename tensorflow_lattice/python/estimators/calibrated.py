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

from tensorflow_lattice.python.estimators import base
from tensorflow_lattice.python.estimators import hparams as tf_lattice_hparams
from tensorflow_lattice.python.lib import keypoints_initialization
from tensorflow_lattice.python.lib import pwl_calibration_layers
from tensorflow_lattice.python.lib import regularizers
from tensorflow_lattice.python.lib import tools

from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training

# Scope for variable names.
_SCOPE_CALIBRATED_PREFIX = "calibrated_"
_SCOPE_INPUT_CALIBRATION = "input_calibration"


def _get_feature_dict(features):
  if isinstance(features, dict):
    return features
  return {"": features}


def _get_optimizer(optimizer, hparams):
  """Materializes the optimizer into a tf.train optimizer object."""
  if optimizer is None:
    optimizer = training.AdamOptimizer
  if callable(optimizer):
    learning_rate = hparams.get_param("learning_rate")
    if learning_rate is None:
      return optimizer()
    else:
      return optimizer(learning_rate=learning_rate)
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
                                         hparams,
                                         quantiles_dir=None,
                                         keypoints_initializers=None,
                                         name=None,
                                         dtype=dtypes.float32):
  """Creates a calibration layer for the input using hyper-parameters.

  Similar to `input_calibration_layer` but reads its parameters from a
  `CalibratedHParams` object.

  Args:
    columns_to_tensors: A mapping from feature name to tensors.
    hparams: Hyper-parameters, need to inherit from `CalibratedHParams`.
      See `CalibratedHParams` and `input_calibration_layer` for descriptions of
      how these hyper-parameters work.
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
      `keypoints_initializer` are set, the latter takes precendence, and the
      features for which `keypoints_initializers` are not defined fallback to
      using the quantiles found in `quantiles_dir`.
    name: Name scope for layer.
    dtype: If any of the scalars are not given as tensors, they are converted
      to tensors with this dtype.

  Returns:
    A tuple of:
    * calibrated tensor of shape [batch_size, sum(features dimensions)].
    * list of the feature names in the order they appear in the calibrated
      tensor. A name may appear more than once if the feature is
      multi-dimension (for instance a multi-dimension embedding)
    * list of projection ops, that must be applied at each step (or every so
      many steps) to project the model to a feasible space: used for bounding
      the outputs or for imposing monotonicity. Empty if none are requested.
    * tensor with regularization loss, or None for no regularization.

  Raises:
    ValueError: if dtypes are incompatible.


  """
  with ops.name_scope(name or "input_calibration_layer_from_hparams"):

    # Sort out list of feature names.
    unique_feature_names = tools.get_sorted_feature_names(
        columns_to_tensors=columns_to_tensors)

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

    # Convert keypoints_initializers to a dict if needed, or otherwise make a
    # copy of the original keypoints_initializers dict.
    if keypoints_initializers is None:
      keypoints_initializers = {}
    elif not isinstance(keypoints_initializers, dict):
      keypoints_initializers = {
          name: keypoints_initializers for name in unique_feature_names
      }
    else:
      keypoints_initializers = keypoints_initializers.copy()

    # If quantiles_dir is given, add any missing keypoint initializers with
    # keypoints based on quantiles.
    if quantiles_dir is not None:
      quantiles_feature_names = [
          name for name in unique_feature_names
          if name not in keypoints_initializers
      ]

      # Reverse initial output keypoints for decreasing monotonic features.
      reversed_dict = {
          feature_name: (monotonicity[feature_name] == -1)
          for feature_name in quantiles_feature_names
      }

      # Read initializers from quantiles_dir, for those not already
      # defined.
      #
      # Notice that output_min and output_max won't matter much if
      # they are not bounded, since they will be adjusted during training.
      quantiles_init = keypoints_initialization.load_keypoints_from_quantiles(
          feature_names=quantiles_feature_names,
          save_dir=quantiles_dir,
          num_keypoints=num_keypoints,
          output_min=calibration_output_min,
          output_max=calibration_output_max,
          reversed_dict=reversed_dict,
          missing_input_values_dict=missing_input_values,
          dtype=dtype)

      # Merge with explicit initializers.
      keypoints_initializers.update(quantiles_init)

    # Update num_keypoints according to keypoints actually used by the
    # initialization functions: some initialization functions may change
    # them, for instance if there are not enough unique values.
    for (feature_name, initializers) in six.iteritems(keypoints_initializers):
      kp_init_keypoints = initializers[0].shape.as_list()[0]
      num_keypoints[feature_name] = _update_keypoints(
          feature_name, num_keypoints[feature_name], kp_init_keypoints)

    # Setup the regularization.
    regularizer_amounts = {}
    for regularizer_name in regularizers.CALIBRATOR_REGULARIZERS:
      regularizer_amounts[regularizer_name] = _get_per_feature_dict(
          hparams, "calibration_{}".format(regularizer_name))

    return pwl_calibration_layers.input_calibration_layer(
        columns_to_tensors=columns_to_tensors,
        num_keypoints=num_keypoints,
        keypoints_initializers=keypoints_initializers,
        bound=calibration_bound,
        monotonic=monotonicity,
        missing_input_values=missing_input_values,
        missing_output_values=missing_output_values,
        **regularizer_amounts)


class _ProjectionHook(session_run_hook.SessionRunHook):
  """SessionRunHook to project to feasible space after each step."""

  def __init__(self):
    self._projection_ops = []

  def set_projection_ops(self, projection_ops):
    """Needs to be called in model_fn function, with ops to project."""
    self._projection_ops = projection_ops

  def after_run(self, run_context, run_values):
    if self._projection_ops is not None:
      run_context.session.run(self._projection_ops)


class Calibrated(base.Base):
  """Base class for TensorFlow calibrated models.

  It provides preprocessing and calibration of the input features, and
  sets up the hook that runs projections at each step -- typically used
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
               head=None,
               weight_column=None,
               name="model"):
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
      optimizer: `Optimizer` object, or callable that defines the
        optimizer to use for training -- if a callable, it will be called with
        learning_rate=hparams.learning_rate if provided.
      config: RunConfig object to configure the runtime settings. Typically set
        to learn_runner.EstimatorConfig().
      hparams: an instance of tf_lattice_hparams.CalibrationHParams. If set to
        None default parameters are used.
      head: a `TensorFlow Estimator Head` which specifies how the loss function,
        final predictions, and so on are generated from model outputs. Defaults
        to using a sigmoid cross entropy head for binary classification and mean
        squared error head for regression.
      weight_column: A string or a `tf.feature_column.numeric_column` defining
        feature column representing weights. It is used to down weight or boost
        examples during training. It will be multiplied by the loss of the
        example.
      name: Name to be used as suffix to top-level variable scope for model.

    Raises:
      ValueError: invalid parameters.
      KeyError: type of feature not supported.
    """
    super(Calibrated, self).__init__(
        n_classes=n_classes,
        feature_columns=feature_columns,
        model_dir=model_dir,
        optimizer=_get_optimizer(optimizer, hparams),
        config=config,
        hparams=hparams,
        head=head,
        weight_column=weight_column,
        dtype=dtypes.float32,
        name=_SCOPE_CALIBRATED_PREFIX + name)

    self._quantiles_dir = quantiles_dir
    self._keypoints_initializers_fn = keypoints_initializers_fn

    if self._hparams is None:
      raise ValueError("hparams cannot be none")
    if not issubclass(
        type(self._hparams), tf_lattice_hparams.CalibratedHParams):
      raise ValueError("hparams is not an instance of hparams.CalibratedHParams"
                       ", got type(params)=%s" % type(self._hparams))

  @abc.abstractmethod
  def calibration_structure_builder(self, columns_to_tensors, hparams):
    """Method to be specialized that builds the calibration structure.

    Derived classes should override this method to return the set of features
    used in each separately calibrated submodel, or return None to indicate
    all features should be calibrated only once.

    Args:
      columns_to_tensors: A mapping from feature name to tensors.
      hparams: hyperparameters passed to object constructor.

    Returns:
      calibration_structure: list of sub_columns_to_tensors corresponding to the
        features used in each sub-model, or None to indicate that this is a
        single model structure that uses all features. Each element is a dict
        from feature name to tensors in the same format as the input
        columns_to_tensors.
    """
    raise NotImplementedError(
        "This method must be implemented in a child class")


  @abc.abstractmethod
  def prediction_builder_from_calibrated(
      self, mode, per_dimension_feature_names, hparams, calibrated):
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
      A tuple of (prediction_tensor, oprojection_ops, regularization_loss) of
      type (tf.Tensor, list[], tf.Tensor):
      prediction_tensor: shaped `[batch_size/?,1]` for regression or binary
        classification, or `[batch_size, n_classes]` for multi-class
        classifiers. For classifier this will be the logit(s) value(s).
      projection_ops: list of projection ops to be applied after each batch,
        or None.
      regularization_loss: loss related to regularization or None.
    """
    raise NotImplementedError(
        "This method must be implemented in a child class")

  def prediction_builder(self, columns_to_tensors, mode, hparams, dtype):
    """Method that builds the prediction graph.

    Args:
      columns_to_tensors: A map from feature_name to raw features tensors,
      each with shape `[batch_size]` or `[batch_size, feature_dim]`.
      mode: Estimator's `ModeKeys`.
      hparams: hyperparameters object passed to prediction builder. This is not
        used by the Base estimator itself and is passed without checks or
        any processing and can be of any type.
      dtype: The dtype to be used for tensors.

    Returns:
      A tuple of (prediction_tensor, oprojection_ops, regularization_loss) of
      type (tf.Tensor, list[], tf.Tensor):
      prediction_tensor: shaped `[batch_size/?,1]` for regression or binary
        classification, or `[batch_size, n_classes]` for multi-class
        classifiers. For classifier this will be the logit(s) value(s).
      projection_ops: list of projection ops to be applied after each batch,
        or None.
      regularization_loss: loss related to regularization or None.
    Raises:
      ValueError: invalid parameters.
    """
    if (mode == model_fn_lib.ModeKeys.TRAIN and self._quantiles_dir is None and
        self._keypoints_initializers_fn is None):
      raise ValueError(
          "At least one of quantiles_dir or keypoints_initializers_fn "
          "must be given for training")

    # If keypoint_initializer closures were given, call them to create the
    # initializers tensors.
    kp_init_explicit = None
    if self._keypoints_initializers_fn is not None:
      kp_init_explicit = _call_keypoints_inializers_fn(
          self._keypoints_initializers_fn)

    # Add feature names to hparams so that builders can make use of them.
    for feature_name in columns_to_tensors:
      self._hparams.add_feature(feature_name)

    total_projection_ops = None
    total_regularization = None
    total_prediction = None

    # Get the ensemble structure.
    calibration_structure = self.calibration_structure_builder(
        columns_to_tensors, self._hparams)

    if calibration_structure is None:
      # Single model or shared calibration.
      (calibrated, per_dimension_feature_names, calibration_projections,
       calibration_regularization) = (
           input_calibration_layer_from_hparams(
               columns_to_tensors=columns_to_tensors,
               hparams=self._hparams,
               quantiles_dir=self._quantiles_dir,
               keypoints_initializers=kp_init_explicit,
               name=_SCOPE_INPUT_CALIBRATION,
               dtype=self._dtype))
      (total_prediction, prediction_projections,
       prediction_regularization) = self.prediction_builder_from_calibrated(
           mode, per_dimension_feature_names, self._hparams, calibrated)
      total_projection_ops = tools.add_if_not_none(calibration_projections,
                                                   prediction_projections)
      total_regularization = tools.add_if_not_none(calibration_regularization,
                                                   prediction_regularization)
    else:
      # Ensemble model with separate calibration.
      predictions = []
      for (index, sub_columns_to_tensors) in enumerate(calibration_structure):
        # Calibrate.
        with variable_scope.variable_scope("submodel_{}".format(index)):
          (calibrated, per_dimension_feature_names, calibration_projections,
           calibration_regularization) = (
               input_calibration_layer_from_hparams(
                   columns_to_tensors=sub_columns_to_tensors,
                   hparams=self._hparams,
                   quantiles_dir=self._quantiles_dir,
                   keypoints_initializers=kp_init_explicit,
                   name=_SCOPE_INPUT_CALIBRATION,
                   dtype=self._dtype))
          (prediction, prediction_projections,
           prediction_regularization) = self.prediction_builder_from_calibrated(
               mode, per_dimension_feature_names, self._hparams, calibrated)
          projection_ops = tools.add_if_not_none(calibration_projections,
                                                 prediction_projections)
          regularization = tools.add_if_not_none(calibration_regularization,
                                                 prediction_regularization)

        # Merge back the results.
        total_projection_ops = tools.add_if_not_none(total_projection_ops,
                                                     projection_ops)
        total_regularization = tools.add_if_not_none(total_regularization,
                                                     regularization)
        predictions.append(prediction)

      # Final prediction is a mean of predictions, plus a bias term.
      stacked_predictions = array_ops.stack(
          predictions, axis=0, name="stacked_predictions")
      ensemble_output = math_ops.reduce_mean(stacked_predictions, axis=0)
      ensemble_bias_init = self._hparams.get_param("ensemble_bias")
      bias = variables.Variable([ensemble_bias_init], name="ensemble_bias")
      total_prediction = ensemble_output + bias

    return total_prediction, total_projection_ops, total_regularization
