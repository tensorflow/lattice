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
"""Base class for generic estimators that handles boilerplate code."""
import abc
# Dependency imports

from tensorflow_lattice.python.lib import tools

from tensorflow.contrib.estimator.python.estimator import head as head_lib
from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses
from tensorflow.python.summary import summary
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training
from tensorflow.python.training import training_util

# Scope for variable names.
_SCOPE_TENSORFLOW_LATTICE_PREFIX = "tfl_"
_TRAIN_OP_NAME = "tfl_train_op"


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


class Base(estimator.Estimator):
  """Base class for generic models.

  It provides minimal preprocessing of the input features, sets up the hook that
  runs projections at each step (typically used to project parameters to be
  monotone and within bounds), and adds the appropriate head to the model.

  To extend one has to implement the method prediction_builder()
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self,
               n_classes,
               feature_columns=None,
               model_dir=None,
               optimizer=None,
               config=None,
               hparams=None,
               head=None,
               weight_column=None,
               dtype=dtypes.float32,
               name="model"):
    """Construct Classifier/Regressor.

    Args:
      n_classes: Number of classes, set to 0 if used for regression. If head
        is not provided, only n_classes = 0 or 2 are currently supported.
      feature_columns: Optional, if not set the model will use all features
        returned by input_fn. An iterable containing all the feature
        columns used by the model. All items in the set should be instances of
        classes derived from `FeatureColumn` and are used to transform the input
        columns into a numeric format that is fed into the rest of the graph.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      optimizer: `Optimizer` object, or callable (with no inputs) that
        returns an `Optimizer` object, defines the optimizer to use for
        training. This is typically one of the optimizers defined in tf.train.
      config: RunConfig object to configure the runtime settings. Typically set
        to learn_runner.EstimatorConfig().
      hparams: Hyper parameter object to be passed to prediction builder.
      head: a `TensorFlow Estimator Head` which specifies how the loss function,
        final predictions, and so on are generated from model outputs. Defaults
        to using a sigmoid cross entropy head for binary classification and mean
        squared error head for regression.
      weight_column: A string or a `tf.feature_column.numeric_column` defining
        feature column representing weights. It is used to down weight or boost
        examples during training. It will be multiplied by the loss of the
        example.
      dtype: The internal type to be used for tensors.
      name: Name to be used as suffix to top-level variable scope for model.

    Raises:
      ValueError: invalid parameters.
      KeyError: type of feature not supported.
    """
    # We sort the list of feature_columns here, since we will later create
    # the ops that implement their represented transformations (e.g. embedding)
    # in the order in which they are listed in self._feature_columns.
    # The constructed ops are then given names by the tensorflow framework
    # that depend on their creation order (for example, if two ops have the
    # same type they will be suffixed by an ordinal reflecting the creation
    # order). As this code must be deterministic (since it could be
    # executed in a multi-machine tensorflow cluster), we must have the order
    # of feature columns deterministic as well (which would not be the case if
    # it's, for example, the result of calling keys() on a dictionary); thus
    # we sort the feature columns here by their names.
    self._feature_columns = (
        None if feature_columns is None
        else tools.get_sorted_feature_columns(feature_columns)
    )
    self._weight_column = weight_column
    self._optimizer = optimizer
    self._config = config
    self._hparams = hparams
    self._name = _SCOPE_TENSORFLOW_LATTICE_PREFIX + name
    self._n_classes = n_classes
    self._dtype = dtype

    if head is not None:
      self._head = head
    else:
      if n_classes == 0:
        self._head = (
            head_lib.regression_head(
                label_dimension=1, weight_column=self._weight_column,
                loss_reduction=losses.Reduction.SUM))
      elif n_classes == 2:
        self._head = (
            head_lib.binary_classification_head(
                weight_column=self._weight_column,
                loss_reduction=losses.Reduction.SUM))
      else:
        raise ValueError("Invalid value for n_classes=%d" % n_classes)

    super(Base, self).__init__(
        model_fn=self._base_model_fn, model_dir=model_dir, config=config)

    # Make sure model directory exists after initialization.
    # Notice self.model_dir is set by Estimator class.
    file_io.recursive_create_dir(self.model_dir)

    self._projection_hook = _ProjectionHook()

  @abc.abstractmethod
  def prediction_builder(self, columns_to_tensors, mode, hparams, dtype):
    """Method to be specialized that builds the prediction graph.

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
    """
    raise NotImplementedError(
        "This method must be implemented in a child class")


  def _base_model_fn(self, features, labels, mode, config):  # pylint: disable=unused-argument
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
      ValueError: if incompatible parameters are given.
    """
    with variable_scope.variable_scope(self._name):
      if self._feature_columns is None:
        columns_to_tensors = features.copy()
      else:
        with variable_scope.variable_scope("feature_column_transformation"):
          columns_to_tensors = {
              feature_column.name: tools.input_from_feature_column(
                  features.copy(), feature_column, self._dtype)
              for feature_column in self._feature_columns
          }
      (prediction, projection_ops, regularization) = self.prediction_builder(
          columns_to_tensors, mode, self._hparams, self._dtype)

      def _train_op_fn(loss):
        """Returns train_op tensor if TRAIN mode, or None."""
        train_op = None
        if mode == model_fn_lib.ModeKeys.TRAIN:
          if regularization is not None:
            loss += regularization
            summary.scalar("loss_with_regularization", loss)
          optimizer = self._optimizer
          if optimizer is None:
            optimizer = training.AdamOptimizer
          if callable(optimizer):
            optimizer = optimizer()
          train_op = optimizer.minimize(
              loss,
              global_step=training_util.get_global_step(),
              name=_TRAIN_OP_NAME)
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
        updated_training_hooks = (
            estimator_spec.training_hooks + (self._projection_hook,))
        estimator_spec = estimator_spec._replace(
            training_hooks=updated_training_hooks)

      return estimator_spec
