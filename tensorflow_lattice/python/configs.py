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
"""TFL model configuration library for canned estimators.

To construct a TFL canned estimator, construct a model configuration and pass
it to the canned estimator constructor:

```python
feature_columns = ...
model_config = tfl.configs.CalibratedLatticeConfig(...)
feature_analysis_input_fn = create_input_fn(num_epochs=1, ...)
train_input_fn = create_input_fn(num_epochs=100, ...)
estimator = tfl.estimators.CannedClassifier(
    feature_columns=feature_columns,
    model_config=model_config,
    feature_analysis_input_fn=feature_analysis_input_fn)
estimator.train(input_fn=train_input_fn)
```

Supported models are:

*   **Calibrated linear model**: Constructed using
    `tfl.configs.CalibratedLinearConfig`.
    A calibrated linear model that applies piecewise-linear and categorical
    calibration on the input feature, followed by a linear combination and an
    optional output piecewise-linear calibration. When using output calibration
    or when output bounds are specified, the linear layer will apply weighted
    averaging on calibrated inputs.

*   **Calibrated lattice model**: Constructed using
    `tfl.configs.CalibratedLatticeConfig`.
    A calibrated lattice model applies piecewise-linear and categorical
    calibration on the input feature, followed by a lattice model and an
    optional output piecewise-linear calibration.

*   **Calibrated lattice ensemble model**: Constructed using
    `tfl.configs.CalibratedLatticeEnsembleConfig`.
    A calibrated lattice ensemble model applies piecewise-linear and categorical
    calibration on the input feature, followed by an ensemble of lattice models
    and an optional output piecewise-linear calibration.

Feature calibration and per-feature configurations are set using
`tfl.configs.FeatureConfig`. Feature configurations include monotonicity
constraints, per-feature regularization (see `tfl.configs.RegularizerConfig`),
and lattice sizes for lattice models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy

from absl import logging
import tensorflow as tf
# pylint: disable=g-import-not-at-top
# Use Keras 2.
version_fn = getattr(tf.keras, "version", None)
if version_fn and version_fn().startswith("3."):
  import tf_keras as keras
else:
  keras = tf.keras

_HPARAM_FEATURE_PREFIX = 'feature'
_HPARAM_REGULARIZER_PREFIX = 'regularizer'


class _Config(object):
  """Base class for configs."""

  def __init__(self, kwargs):
    if 'self' in kwargs:
      kwargs.pop('self')
    if '__class__' in kwargs:
      kwargs.pop('__class__')
    self.__dict__ = kwargs

  def __repr__(self):
    return self.__dict__.__repr__()

  def get_config(self):
    """Returns a configuration dictionary."""
    config = copy.deepcopy(self.__dict__)
    if 'self' in config:
      config.pop('self')
    if '__class__' in config:
      config.pop('__class__')
    if 'feature_configs' in config and config['feature_configs'] is not None:
      config['feature_configs'] = [
          keras.utils.legacy.serialize_keras_object(feature_config)
          for feature_config in config['feature_configs']
      ]
    if 'regularizer_configs' in config and config[
        'regularizer_configs'] is not None:
      config['regularizer_configs'] = [
          keras.utils.legacy.serialize_keras_object(regularizer_config)
          for regularizer_config in config['regularizer_configs']
      ]
    if ('reflects_trust_in' in config and
        config['reflects_trust_in'] is not None):
      config['reflects_trust_in'] = [
          keras.utils.legacy.serialize_keras_object(trust_config)
          for trust_config in config['reflects_trust_in']
      ]
    if 'dominates' in config and config['dominates'] is not None:
      config['dominates'] = [
          keras.utils.legacy.serialize_keras_object(dominance_config)
          for dominance_config in config['dominates']
      ]
    return config

  @classmethod
  def deserialize_nested_configs(cls, config, custom_objects=None):
    """Returns a deserialized configuration dictionary."""
    config = copy.deepcopy(config)
    if 'feature_configs' in config and config['feature_configs'] is not None:
      config['feature_configs'] = [
          keras.utils.legacy.deserialize_keras_object(
              feature_config, custom_objects=custom_objects
          )
          for feature_config in config['feature_configs']
      ]
    if 'regularizer_configs' in config and config[
        'regularizer_configs'] is not None:
      config['regularizer_configs'] = [
          keras.utils.legacy.deserialize_keras_object(
              regularizer_config, custom_objects=custom_objects
          )
          for regularizer_config in config['regularizer_configs']
      ]
    if ('reflects_trust_in' in config and
        config['reflects_trust_in'] is not None):
      config['reflects_trust_in'] = [
          keras.utils.legacy.deserialize_keras_object(
              trust_config, custom_objects=custom_objects
          )
          for trust_config in config['reflects_trust_in']
      ]
    if 'dominates' in config and config['dominates'] is not None:
      config['dominates'] = [
          keras.utils.legacy.deserialize_keras_object(
              dominance_config, custom_objects=custom_objects
          )
          for dominance_config in config['dominates']
      ]
    return config


class _HasFeatureConfigs(object):
  """Base class for configs with `feature_configs` attribute."""

  def feature_config_by_name(self, feature_name):
    """Returns existing or default FeatureConfig with the given name."""
    if self.feature_configs is None:
      self.feature_configs = []
    for feature_config in self.feature_configs:
      if feature_config.name == feature_name:
        return feature_config
    feature_config = FeatureConfig(feature_name)
    self.feature_configs.append(feature_config)
    return feature_config


class _HasRegularizerConfigs(object):
  """Base class for configs with `regularizer_configs` attribute."""

  def regularizer_config_by_name(self, regularizer_name):
    """Returns existing or default RegularizerConfig with the given name."""
    if self.regularizer_configs is None:
      self.regularizer_configs = []
    for regularizer_config in self.regularizer_configs:
      if regularizer_config.name == regularizer_name:
        return regularizer_config
    regularizer_config = RegularizerConfig(regularizer_name)
    self.regularizer_configs.append(regularizer_config)
    return regularizer_config


# pylint: disable=unused-argument


class CalibratedLatticeEnsembleConfig(_Config, _HasFeatureConfigs,
                                      _HasRegularizerConfigs):
  """Config for calibrated lattice model.

  A calibrated lattice ensemble model applies piecewise-linear and categorical
  calibration on the input feature, followed by an ensemble of lattice models
  and an optional output piecewise-linear calibration.

  The ensemble structure can be one of the following and set via the lattice
  flag:

    - Expliclit list of list of features specifying features used in each
      submodel.
    - A random arrangement (also called Random Tiny Lattices, or RTL).
    - Crystals growing algorithm: This algorithm first constructs a prefitting
      model to assess pairwise interactions between features, and then uses
      those estimates to construct a final model that puts interacting
      features in the same lattice. For details see "Fast and flexible monotonic
      functions with ensembles of lattices", Advances in Neural Information
      Processing Systems, 2016.

  Examples:

  Creating a random ensemble (RTL) model:

  ```python
  model_config = tfl.configs.CalibratedLatticeEnsembleConfig(
      num_lattices=6,  # number of lattices
      lattice_rank=5,  # number of features in each lattice
      feature_configs=[...],
  )
  feature_analysis_input_fn = create_input_fn(num_epochs=1, ...)
  train_input_fn = create_input_fn(num_epochs=100, ...)
  estimator = tfl.estimators.CannedClassifier(
      feature_columns=feature_columns,
      model_config=model_config,
      feature_analysis_input_fn=feature_analysis_input_fn)
  estimator.train(input_fn=train_input_fn)
  ```
  You can also construct a random ensemble (RTL) using a `tfl.layers.RTL`
  layer so long as all features have the same lattice size:
  ```python
  model_config = tfl.configs.CalibratedLatticeEnsembleConfig(
      lattices='rtl_layer',
      num_lattices=6,  # number of lattices
      lattice_rank=5,  # number of features in each lattice
      feature_configs=[...],
  )
  feature_analysis_input_fn = create_input_fn(num_epochs=1, ...)
  train_input_fn = create_input_fn(num_epochs=100, ...)
  estimator = tfl.estimators.CannedClassifier(
      feature_columns=feature_columns,
      model_config=model_config,
      feature_analysis_input_fn=feature_analysis_input_fn)
  estimator.train(input_fn=train_input_fn)
  ```
  To create a Crystals model, you will need to provide a *prefitting_input_fn*
  to the estimator constructor. This input_fn is used to train the prefitting
  model, as described above. The prefitting model does not need to be fully
  trained, so a few epochs should be enough.

  ```python
  model_config = tfl.configs.CalibratedLatticeEnsembleConfig(
      lattices='crystals',  # feature arrangement method
      num_lattices=6,  # number of lattices
      lattice_rank=5,  # number of features in each lattice
      feature_configs=[...],
  )
  feature_analysis_input_fn = create_input_fn(num_epochs=1, ...)
  prefitting_input_fn = create_input_fn(num_epochs=5, ...)
  train_input_fn = create_input_fn(num_epochs=100, ...)
  estimator = tfl.estimators.CannedClassifier(
      feature_columns=feature_columns,
      model_config=model_config,
      feature_analysis_input_fn=feature_analysis_input_fn
      prefitting_input_fn=prefitting_input_fn)
  estimator.train(input_fn=train_input_fn)
  ```
  """
  _HAS_DYNAMIC_ATTRIBUTES = True  # Required for pytype checks.

  def __init__(self,
               feature_configs=None,
               lattices='random',
               num_lattices=None,
               lattice_rank=None,
               interpolation='hypercube',
               parameterization='all_vertices',
               num_terms=2,
               separate_calibrators=True,
               use_linear_combination=False,
               use_bias=False,
               regularizer_configs=None,
               output_min=None,
               output_max=None,
               output_calibration=False,
               output_calibration_num_keypoints=10,
               output_initialization='quantiles',
               output_calibration_input_keypoints_type='fixed',
               fix_ensemble_for_2d_constraints=True,
               random_seed=0):
    # pyformat: disable
    """Initializes a `CalibratedLatticeEnsembleConfig` instance.

    Args:
      feature_configs: A list of `tfl.configs.FeatureConfig` instances that
        specify configurations for each feature. If a configuration is not
        provided for a feature, a default configuration will be used.
      lattices: Should be one of the following:
        - String `'random'` indicating that the features in each lattice should
          be selected randomly
        - String `'rtl_layer'` indicating that the features in each lattice
          should be selected randomly using a `tfl.layers.RTL` layer. Note that
          using a `tfl.layers.RTL` layer scales better than using separate
          `tfl.layers.Lattice` instances for the ensemble.
        - String `'crystals'` to use a heuristic to construct the lattice
          ensemble based on pairwise feature interactions
        - An explicit list of list of feature names to be used in each lattice
          in the ensemble.
      num_lattices: Number of lattices in the ensemble. Must be provided if
        lattices are not explicitly provided.
      lattice_rank: Number of features in each lattice. Must be provided if
        lattices are not explicitly provided.
      interpolation: One of 'hypercube' or 'simplex' interpolation. For a
        d-dimensional lattice, 'hypercube' interpolates 2^d parameters, whereas
        'simplex' uses d+1 parameters and thus scales better. For details see
        `tfl.lattice_lib.evaluate_with_simplex_interpolation` and
        `tfl.lattice_lib.evaluate_with_hypercube_interpolation`.
      parameterization: The parameterization of the lattice function class to
        use. A lattice function is uniquely determined by specifying its value
        on every lattice vertex. A parameterization scheme is a mapping from a
        vector of parameters to a multidimensional array of lattice vertex
        values. It can be one of:
          - String `'all_vertices'`: This is the "traditional" parameterization
            that keeps one scalar parameter per lattice vertex where the mapping
            is essentially the identity map. With this scheme, the number of
            parameters scales exponentially with the number of inputs to the
            lattice. The underlying lattices used will be `tfl.layers.Lattice`
            layers.
          - String `'kronecker_factored'`: With this parameterization, for each
            lattice input i we keep a collection of `num_terms` vectors each
            having `feature_configs[0].lattice_size` entries (note that all
            features must have the same lattice size). To obtain the tensor of
            lattice vertex values, for `t=1,2,...,num_terms` we compute the
            outer product of the `t'th` vector in each collection, multiply by a
            per-term scale, and sum the resulting tensors. Finally, we add a
            single shared bias parameter to each entry in the sum. With this
            scheme, the number of parameters grows linearly with `lattice_rank`
            (assuming lattice sizes and `num_terms` are held constant).
            Currently, only monotonicity shape constraint and bound constraint
            are supported for this scheme. Regularization is not currently
            supported. The underlying lattices used will be
            `tfl.layers.KroneckerFactoredLattice` layers.
      num_terms: The number of terms in a lattice using `'kronecker_factored'`
        parameterization. Ignored if parameterization is set to
        `'all_vertices'`.
      separate_calibrators: If features should be separately calibrated for each
        lattice in the ensemble.
      use_linear_combination: If set to true, a linear combination layer will be
        used to combine ensemble outputs. Otherwise an averaging layer will be
        used. If output is bounded or output calibration is used, then this
        layer will be a weighted average.
      use_bias: If a bias term should be used for the linear combination.
      regularizer_configs: A list of `tfl.configs.RegularizerConfig` instances
        that apply global regularization.
      output_min: Lower bound constraint on the output of the model.
      output_max: Upper bound constraint on the output of the model.
      output_calibration: If a piecewise-linear calibration should be used on
        the output of the lattice.
      output_calibration_num_keypoints: Number of keypoints to use for the
        output piecewise-linear calibration.
      output_initialization: The initial values to setup for the output of the
        model. When using output calibration, these values are used to
        initialize the output keypoints of the output piecewise-linear
        calibration. Otherwise the lattice parameters will be setup to form a
        linear function in the range of output_initialization. It can be one of:
          - String `'quantiles'`: Output is initliazed to label quantiles, if
            possible.
          - String `'uniform'`: Output is initliazed uniformly in label range.
          - A list of numbers: To be used for initialization of the output
            lattice or output calibrator.
      output_calibration_input_keypoints_type: One of "fixed" or
        "learned_interior". If "learned_interior", keypoints are initialized to
        the values in `pwl_calibration_input_keypoints` but then allowed to vary
        during training, with the exception of the first and last keypoint
        location which are fixed.
      fix_ensemble_for_2d_constraints: A boolean indicating whether to add
        missing features to some lattices to resolve potential 2d constraint
        violations which require lattices from ensemble to either contain both
        constrained features or none of them, e.g. trapezoid trust constraint
        requires a lattice that has the "conditional" feature to include the
        "main" feature. Note that this might increase the final lattice rank.
      random_seed: Random seed to use for randomized lattices.
    """
    # pyformat: enable
    super(CalibratedLatticeEnsembleConfig, self).__init__(locals())

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return CalibratedLatticeEnsembleConfig(**_Config.deserialize_nested_configs(
        config, custom_objects=custom_objects))


class CalibratedLatticeConfig(_Config, _HasFeatureConfigs,
                              _HasRegularizerConfigs):
  """Config for calibrated lattice model.

  A calibrated lattice model applies piecewise-linear and categorical
  calibration on the input feature, followed by a lattice model and an
  optional output piecewise-linear calibration.

  Example:

  ```python
  model_config = tfl.configs.CalibratedLatticeConfig(
      feature_configs=[...],
  )
  feature_analysis_input_fn = create_input_fn(num_epochs=1, ...)
  train_input_fn = create_input_fn(num_epochs=100, ...)
  estimator = tfl.estimators.CannedClassifier(
      feature_columns=feature_columns,
      model_config=model_config,
      feature_analysis_input_fn=feature_analysis_input_fn)
  estimator.train(input_fn=train_input_fn)
  ```
  """
  _HAS_DYNAMIC_ATTRIBUTES = True  # Required for pytype checks.

  def __init__(self,
               feature_configs=None,
               interpolation='hypercube',
               parameterization='all_vertices',
               num_terms=2,
               regularizer_configs=None,
               output_min=None,
               output_max=None,
               output_calibration=False,
               output_calibration_num_keypoints=10,
               output_initialization='quantiles',
               output_calibration_input_keypoints_type='fixed',
               random_seed=0):
    """Initializes a `CalibratedLatticeConfig` instance.

    Args:
      feature_configs: A list of `tfl.configs.FeatureConfig` instances that
        specify configurations for each feature. If a configuration is not
        provided for a feature, a default configuration will be used.
      interpolation: One of 'hypercube' or 'simplex' interpolation. For a
        d-dimensional lattice, 'hypercube' interpolates 2^d parameters, whereas
        'simplex' uses d+1 parameters and thus scales better. For details see
        `tfl.lattice_lib.evaluate_with_simplex_interpolation` and
        `tfl.lattice_lib.evaluate_with_hypercube_interpolation`.
      parameterization: The parameterization of the lattice function class to
        use. A lattice function is uniquely determined by specifying its value
        on every lattice vertex. A parameterization scheme is a mapping from a
        vector of parameters to a multidimensional array of lattice vertex
        values. It can be one of:
          - String `'all_vertices'`: This is the "traditional" parameterization
            that keeps one scalar parameter per lattice vertex where the mapping
            is essentially the identity map. With this scheme, the number of
            parameters scales exponentially with the number of inputs to the
            lattice. The underlying lattice used will be a `tfl.layers.Lattice`
            layer.
          - String `'kronecker_factored'`: With this parameterization, for each
            lattice input i we keep a collection of `num_terms` vectors each
            having `feature_configs[0].lattice_size` entries (note that all
            features must have the same lattice size). To obtain the tensor of
            lattice vertex values, for `t=1,2,...,num_terms` we compute the
            outer product of the `t'th` vector in each collection, multiply by a
            per-term scale, and sum the resulting tensors. Finally, we add a
            single shared bias parameter to each entry in the sum. With this
            scheme, the number of parameters grows linearly with
            `len(feature_configs)` (assuming lattice sizes and `num_terms` are
            held constant). Currently, only monotonicity shape constraint and
            bound constraint are supported for this scheme. Regularization is
            not currently supported. The underlying lattice used will be a
            `tfl.layers.KroneckerFactoredLattice` layer.
      num_terms: The number of terms in a lattice using `'kronecker_factored'`
        parameterization. Ignored if parameterization is set to
        `'all_vertices'`.
      regularizer_configs: A list of `tfl.configs.RegularizerConfig` instances
        that apply global regularization.
      output_min: Lower bound constraint on the output of the model.
      output_max: Upper bound constraint on the output of the model.
      output_calibration: If a piecewise-linear calibration should be used on
        the output of the lattice.
      output_calibration_num_keypoints: Number of keypoints to use for the
        output piecewise-linear calibration.
      output_initialization: The initial values to setup for the output of the
        model. When using output calibration, these values are used to
        initialize the output keypoints of the output piecewise-linear
        calibration. Otherwise the lattice parameters will be setup to form a
        linear function in the range of output_initialization. It can be one of:
          - String `'quantiles'`: Output is initliazed to label quantiles, if
            possible.
          - String `'uniform'`: Output is initliazed uniformly in label range.
          - A list of numbers: To be used for initialization of the output
            lattice or output calibrator.
      output_calibration_input_keypoints_type: One of "fixed" or
        "learned_interior". If "learned_interior", keypoints are initialized to
        the values in `pwl_calibration_input_keypoints` but then allowed to vary
        during training, with the exception of the first and last keypoint
        location which are fixed.
      random_seed: Random seed to use for initialization of a lattice with
        `'kronecker_factored'` parameterization. Ignored if parameterization is
        set to `'all_vertices'`.
    """
    super(CalibratedLatticeConfig, self).__init__(locals())

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return CalibratedLatticeConfig(**_Config.deserialize_nested_configs(
        config, custom_objects=custom_objects))


class CalibratedLinearConfig(_Config, _HasFeatureConfigs,
                             _HasRegularizerConfigs):
  """Config for calibrated lattice model.

  A calibrated linear model applies piecewise-linear and categorical
  calibration on the input feature, followed by a linear combination and an
  optional output piecewise-linear calibration. When using output calibration
  or when output bounds are specified, the linear layer will be apply weighted
  averaging on calibrated inputs.

  Example:

  ```python
  model_config = tfl.configs.CalibratedLinearConfig(
      feature_configs=[...],
  )
  feature_analysis_input_fn = create_input_fn(num_epochs=1, ...)
  train_input_fn = create_input_fn(num_epochs=100, ...)
  estimator = tfl.estimators.CannedClassifier(
      feature_columns=feature_columns,
      model_config=model_config,
      feature_analysis_input_fn=feature_analysis_input_fn)
  estimator.train(input_fn=train_input_fn)
  ```
  """
  _HAS_DYNAMIC_ATTRIBUTES = True  # Required for pytype checks.

  def __init__(self,
               feature_configs=None,
               regularizer_configs=None,
               use_bias=True,
               output_min=None,
               output_max=None,
               output_calibration=False,
               output_calibration_num_keypoints=10,
               output_initialization='quantiles',
               output_calibration_input_keypoints_type='fixed'):
    """Initializes a `CalibratedLinearConfig` instance.

    Args:
      feature_configs: A list of `tfl.configs.FeatureConfig` instances that
        specify configurations for each feature. If a configuration is not
        provided for a feature, a default configuration will be used.
      regularizer_configs: A list of `tfl.configs.RegularizerConfig` instances
        that apply global regularization.
      use_bias: If a bias term should be used for the linear combination.
      output_min: Lower bound constraint on the output of the model.
      output_max: Upper bound constraint on the output of the model.
      output_calibration: If a piecewise-linear calibration should be used on
        the output of the lattice.
      output_calibration_num_keypoints: Number of keypoints to use for the
        output piecewise-linear calibration.
      output_initialization: The initial values to setup for the output of the
        model. When using output calibration, these values are used to
        initialize the output keypoints of the output piecewise-linear
        calibration. Otherwise the lattice parameters will be setup to form a
        linear function in the range of output_initialization. It can be one of:
          - String `'quantiles'`: Output is initliazed to label quantiles, if
            possible.
          - String `'uniform'`: Output is initliazed uniformly in label range.
          - A list of numbers: To be used for initialization of the output
            lattice or output calibrator.
      output_calibration_input_keypoints_type: One of "fixed" or
        "learned_interior". If "learned_interior", keypoints are initialized to
        the values in `pwl_calibration_input_keypoints` but then allowed to vary
        during training, with the exception of the first and last keypoint
        location which are fixed.
    """
    super(CalibratedLinearConfig, self).__init__(locals())

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return CalibratedLinearConfig(**_Config.deserialize_nested_configs(
        config, custom_objects=custom_objects))


# TODO: add option for different pre-aggregation model (linear/ensemble)
class AggregateFunctionConfig(_Config, _HasFeatureConfigs,
                              _HasRegularizerConfigs):
  """Config for aggregate function learning model.

  An aggregate function learning model applies piecewise-linear and categorical
  calibration on the ragged input features, followed by an aggregation layer
  that aggregates the calibrated inputs. Lastly a lattice model and an optional
  output piecewise-linear calibration are applied.

  Example:

  ```python
  model_config = tfl.configs.AggregateFunctionConfig(
      feature_configs=[...],
  )
  model = tfl.premade.AggregateFunction(model_config)
  model.compile(...)
  model.fit(...)
  model.evaluate(...)
  ```
  """
  _HAS_DYNAMIC_ATTRIBUTES = True  # Required for pytype checks.

  def __init__(self,
               feature_configs,
               regularizer_configs=None,
               middle_dimension=1,
               middle_lattice_size=2,
               middle_calibration=False,
               middle_calibration_num_keypoints=10,
               middle_calibration_input_keypoints_type='fixed',
               middle_monotonicity=None,
               middle_lattice_interpolation='hypercube',
               aggregation_lattice_interpolation='hypercube',
               output_min=None,
               output_max=None,
               output_calibration=False,
               output_calibration_num_keypoints=10,
               output_initialization='uniform',
               output_calibration_input_keypoints_type='fixed'):
    """Initializes an `AggregateFunctionConfig` instance.

    Args:
      feature_configs: A list of `tfl.configs.FeatureConfig` instances that
        specify configurations for each feature.
      regularizer_configs: A list of `tfl.configs.RegularizerConfig` instances
        that apply global regularization.
      middle_dimension: The number of calibrated lattices that are applied to
        each block. The outputs of these lattices are then averaged over the
        blocks, and the middle_dimension resulting numbers are then passed into
        the "middle" calibrated lattice. This middle lattice therefore has input
        dimension equal to middle_dimension.
      middle_lattice_size: Size of each of the middle_lattice dimensions.
      middle_calibration: If a piecewise-linear calibration should be used on
        the inputs to the middle lattice.
      middle_calibration_num_keypoints: Number of keypoints to use for the
        middle piecewise-linear calibration.
      middle_calibration_input_keypoints_type: One of "fixed" or
        "learned_interior". If "learned_interior", keypoints are initialized to
        the values in `pwl_calibration_input_keypoints` but then allowed to vary
        during training, with the exception of the first and last keypoint
        location which are fixed.
      middle_monotonicity: Specifies if the middle calibrators should be
        monotonic, using 'increasing' or 1 to indicate increasing monotonicity,
        'decreasing' or -1 to indicate decreasing monotonicity, and 'none' or 0
        to indicate no monotonicity constraints.
      middle_lattice_interpolation: One of 'hypercube' or 'simplex'. For a
        d-dimensional lattice, 'hypercube' interpolates 2^d parameters, whereas
        'simplex' uses d+1 parameters and thus scales better. For details see
        `tfl.lattice_lib.evaluate_with_simplex_interpolation` and
        `tfl.lattice_lib.evaluate_with_hypercube_interpolation`.
      aggregation_lattice_interpolation: One of 'hypercube' or 'simplex'. For a
        d-dimensional lattice, 'hypercube' interpolates 2^d parameters, whereas
        'simplex' uses d+1 parameters and thus scales better. For details see
        `tfl.lattice_lib.evaluate_with_simplex_interpolation` and
        `tfl.lattice_lib.evaluate_with_hypercube_interpolation`.
      output_min: Lower bound constraint on the output of the model.
      output_max: Upper bound constraint on the output of the model.
      output_calibration: If a piecewise-linear calibration should be used on
        the output of the lattice.
      output_calibration_num_keypoints: Number of keypoints to use for the
        output piecewise-linear calibration.
      output_initialization: The initial values to setup for the output of the
        model. When using output calibration, these values are used to
        initialize the output keypoints of the output piecewise-linear
        calibration. Otherwise the lattice parameters will be setup to form a
        linear function in the range of output_initialization. It can be one of:
          - String `'uniform'`: Output is initliazed uniformly in label range.
          - A list of numbers: To be used for initialization of the output
            lattice or output calibrator.
      output_calibration_input_keypoints_type: One of "fixed" or
        "learned_interior". If "learned_interior", keypoints are initialized to
        the values in `pwl_calibration_input_keypoints` but then allowed to vary
        during training, with the exception of the first and last keypoint
        location which are fixed.
    """
    super(AggregateFunctionConfig, self).__init__(locals())

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return AggregateFunctionConfig(**_Config.deserialize_nested_configs(
        config, custom_objects=custom_objects))


class FeatureConfig(_Config, _HasRegularizerConfigs):
  """Per-feature configuration for TFL canned estimators.

  A feature can either be numerical or categorical. Numeric features will be
  calibrated using a piecewise-linear function with the given number of
  keypoints. Categorical features should have `num_buckets > 0` and the
  `vocabulary_list` represent their categories. Several of the config fields
  can be filled in automatically based on the `FeatureColumns` used by the
  model but can also be provided explicitly. See `__init__` args comments for
  details.

  Currently only one dimensional feature are supported.

  Examples:

  ```python
  feature_columns = [
      tf.feature_column.numeric_column.numeric_column(
          'age', default_value=-1),
      tf.feature_column.numeric_column.categorical_column_with_vocabulary_list(
          'thal', vocabulary_list=['normal', 'fixed', 'reversible']),
      ...
  ]

  model_config = tfl.configs.CalibratedLatticeConfig(
      feature_configs=[
          tfl.configs.FeatureConfig(
              name='age',
              lattice_size=3,
              # Monotonically increasing.
              monotonicity='increasing',
              # Per feature regularization.
              regularizer_configs=[
                  tfl.configs.RegularizerConfig(name='calib_hessian', l2=1e-4),
              ],
          ),
          tfl.configs.FeatureConfig(
              name='thal',
              # Partial monotonicity:
              # output(normal) <= output(fixed)
              # output(normal) <= output(reversible)
              monotonicity=[('normal', 'fixed'), ('normal', 'reversible')],
          ),
      ],
      # Global regularizers
      regularizer_configs=[...])
  feature_analysis_input_fn = create_input_fn(num_epochs=1, ...)
  train_input_fn = create_input_fn(num_epochs=100, ...)
  estimator = tfl.estimators.CannedClassifier(
      feature_columns=feature_columns,
      model_config=model_config,
      feature_analysis_input_fn=feature_analysis_input_fn)
  estimator.train(input_fn=train_input_fn)
  ```
  """
  _HAS_DYNAMIC_ATTRIBUTES = True  # Required for pytype checks.

  def __init__(self,
               name,
               is_missing_name=None,
               default_value=None,
               lattice_size=2,
               monotonicity='none',
               unimodality='none',
               reflects_trust_in=None,
               dominates=None,
               pwl_calibration_always_monotonic=False,
               pwl_calibration_convexity=0,
               pwl_calibration_num_keypoints=10,
               pwl_calibration_input_keypoints='quantiles',
               pwl_calibration_input_keypoints_type='fixed',
               pwl_calibration_clip_min=None,
               pwl_calibration_clip_max=None,
               pwl_calibration_clamp_min=False,
               pwl_calibration_clamp_max=False,
               num_buckets=0,
               vocabulary_list=None,
               regularizer_configs=None):
    """Initializes a `FeatureConfig` instance.

    Args:
      name: The name of the feature, which should match the name of a given
        FeatureColumn or a key in the input feature dict.
      is_missing_name: The name of a FeatureColumn or key in the input feature
        dict that indicates missing-ness of the main feature.
      default_value: [Automatically filled in from `FeatureColumns`] If set,
        this value in the input value represents missing. For numeric features,
        the output will be imputed. If default_value is provided for a
        categocial features, it would corresponds to the last bucket counted in
        num_buckets.
      lattice_size: The number of lattice verticies to be used along the axis
        for this feature.
      monotonicity: - For numeric features, specifies if the model output should
        be monotonic in this feature, using 'increasing' or 1 to indicate
        increasing monotonicity, 'decreasing' or -1 to indicate decreasing
        monotonicity, and 'none' or 0 to indicate no monotonicity constraints. -
        For categorical features, a list of (category_a, category_b) pairs from
        the vocabulary list indicating that with other features fixed, model
        output for category_b should be greater than or equal to category_a. If
        no vocabulary list is specified, we assume implcit vocabulary in the
        range `[0, num_buckets - 1]`.
      unimodality: For numeric features specifies if the model output should be
        unimodal in corresponding feature, using 'valley' or 1 to indicate that
        function first decreases then increases, using 'peak' or -1 to indicate
        that funciton first increases then decreases, using 'none' or 0 to
        indicate no unimodality constraints. Not used for categorical features.
      reflects_trust_in: None or a list of `tfl.configs.TrustConfig` instances.
      dominates: None or a list of `tfl.configs.DominanceConfig` instances.
      pwl_calibration_always_monotonic: Specifies if the piecewise-linear
        calibration should always be monotonic regardless of the specified
        end-to-end model output `monotonicity` with respect to this feature.
      pwl_calibration_convexity: Spefices the convexity constraints of the
        calibrators for numeric features. Convexity is indicated by 'convex' or
        1, concavity is indicated by 'concave' or -1, 'none' or 0 indicates no
        convexity/concavity constraints. Does not affect categorical features.
        Concavity together with increasing monotonicity as well as convexity
        together with decreasing monotonicity results in diminishing return
        constraints.
      pwl_calibration_num_keypoints: Number of keypoints to use for
        piecewise-linear calibration.
      pwl_calibration_input_keypoints: Indicates what should be used for the
        input keypoints of the piecewise-linear calibration. It can be one of:
          - String `'quantiles'`: Input keypoints are set to feature quantiles.
          - String `'uniform'`: Input keypoints are uniformly spaced in feature
            range.
          - A list of numbers: Explicitly specifies the keypoints.
      pwl_calibration_input_keypoints_type: One of "fixed" or
        "learned_interior". If "learned_interior", keypoints are initialized to
        the values in `pwl_calibration_input_keypoints` but then allowed to vary
        during training, with the exception of the first and last keypoint
        location which are fixed. Convexity can only be imposed with "fixed".
      pwl_calibration_clip_min: Input values are lower clipped by this value.
      pwl_calibration_clip_max: Input values are upper clipped by this value.
      pwl_calibration_clamp_min: for monotonic calibrators ensures that the
        minimum value in calibration output is reached.
      pwl_calibration_clamp_max: for monotonic calibrators ensures that the
        maximum value in calibration output is reached.
      num_buckets: [Automatically filled in from `FeatureColumns`] Number of
        categories for a categorical feature. Out-of-vocabulary and
        missing/default value should be counted into num_buckets (last buckets).
      vocabulary_list: [Automatically filled in from `FeatureColumns`] The input
        vocabulary of the feature.
      regularizer_configs: None or a list of per-feature
        `tfl.configs.RegularizerConfig` instances.
    """
    super(FeatureConfig, self).__init__(locals())

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return FeatureConfig(**_Config.deserialize_nested_configs(
        config, custom_objects=custom_objects))


class RegularizerConfig(_Config):
  """Regularizer configuration for TFL canned estimators.

  Regularizers can either be applied to specific features, or can be applied
  globally to all features or lattices.


  * **Calibrator regularizers:**

    These regularizers are applied to PWL calibration layers.

    - `'calib_laplacian'`: Creates an instance of
      `tfl.pwl_calibration_layer.LaplacianRegularizer`. A calibrator laplacian
      regularizer penalizes the changes in the output and results in a *flatter
      calibration function*.
    - `'calib_hessian'`: Creates an instance of
      `tfl.pwl_calibration_layer.HessianRegularizer`. A calibrator hessian
      regularizer penalizes changes in the slope, resulting in a *more linear
      calibration*.
    - `'calib_wrinkle'`: Creates an instance of
      `tfl.pwl_calibration_layer.WrinkleRegularizer`. A calibrator wrinkle
      regularizer penalizes the second derivative, resulting in a smoother
      function with *less changes in the curvature*.


  * **Lattice regularizers:**

    These regularizers are applied to lattice layers.

    - `'laplacian'`: Creates an instance of
      `tfl.lattice_layer.LaplacianRegularizer`. Laplacian regularizers penalize
      the difference between adjacent vertices in multi-cell lattice, resulting
      in a *flatter lattice function*.
    - `'torsion'`: Creates an instance of
      `tfl.lattice_layer.TorsionRegularizer`. Torsion regularizers penalizes
      how much the lattice function twists from side-to-side, a non-linear
      interactions in each 2 x 2 cell. Using this regularization results in a
      *more linear lattice function*.


  Examples:

  ```python
  model_config = tfl.configs.CalibratedLatticeConfig(
      feature_configs=[
          tfl.configs.FeatureConfig(
              name='age',
              lattice_size=3,
              # Per feature regularization.
              regularizer_configs=[
                  tfl.configs.RegularizerConfig(name='calib_hessian', l2=1e-4),
              ],
          ),
          tfl.configs.FeatureConfig(
              name='thal',
              # Partial monotonicity:
              # output(normal) <= output(fixed)
              # output(normal) <= output(reversible)
              monotonicity=[('normal', 'fixed'), ('normal', 'reversible')],
          ),
      ],
      # Global regularizers
      regularizer_configs=[
          # Torsion regularizer applied to the lattice to make it more linear.
          configs.RegularizerConfig(name='torsion', l2=1e-4),
          # Globally defined calibration regularizer is applied to all features.
          configs.RegularizerConfig(name='calib_hessian', l2=1e-4),
      ])
  feature_analysis_input_fn = create_input_fn(num_epochs=1, ...)
  train_input_fn = create_input_fn(num_epochs=100, ...)
  estimator = tfl.estimators.CannedClassifier(
      feature_columns=feature_columns,
      model_config=model_config,
      feature_analysis_input_fn=feature_analysis_input_fn)
  estimator.train(input_fn=train_input_fn)
  ```
  """
  _HAS_DYNAMIC_ATTRIBUTES = True  # Required for pytype checks.

  def __init__(self, name, l1=0.0, l2=0.0):
    """Initializes a `RegularizerConfig` instance.

    Args:
      name: The name of the regularizer.
      l1: l1 regularization amount.
      l2: l2 regularization amount.
    """
    super(RegularizerConfig, self).__init__(locals())

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return RegularizerConfig(**_Config.deserialize_nested_configs(
        config, custom_objects=custom_objects))


class TrustConfig(_Config):
  """Configuration for feature trusts in TFL canned estimators.

  You can specify how a feature reflects trust in another feature. Supported
  trust types (see `tfl.layers.Lattice` for details):

  - `'edgeworth'`: Edgeworth trust constrains the function to be more
      responsive to a main feature as a secondary conditional feature increases
      or decreases. For example, we may want the model to rely more on average
      rating (main feature) when the number of reviews (conditional feature) is
      high. In particular, the constraint guarantees that a given change in the
      main feature's value will change the model output by more when a secondary
      feature indicates higher trust in the main feature. Note that the
      constraint only works when the model is monotonic in the main feature.
  - `'trapezoid'`: Trapezoid trust is conceptually similar to edgeworth trust,
      but this constraint guarantees that the range of possible outputs along
      the main feature dimension, when a conditional feature indicates low
      trust, is a *subset* of the range of outputs when a conditional feature
      indicates high trust. When lattices have 2 vertices in each constrained
      dimension, this implies edgeworth trust (which only constrains the size of
      the relevant ranges). With more than 2 lattice vertices per dimension, the
      two constraints diverge and are not necessarily 'weaker' or 'stronger'
      than each other - edgeworth trust acts throughout the lattice interior on
      delta shifts in the main feature, while trapezoid trust only acts on the
      min and max extremes of the main feature, constraining the overall range
      of outputs across the domain of the main feature. The two types of trust
      constraints can be applied jointly.

  Trust constraints only affect lattices. When using trapezoid constraints in
  ensemble models, note that if a conditional feature is used in a lattice
  without the main feature also being used in the same lattice, then the
  trapezoid constraint might be violated for the ensemble function.

  Exampes:

  One feature reflecting trust in another:

  ```python
  model_config = tfl.configs.CalibratedLatticeConfig(
      feature_configs=[
          tfl.configs.FeatureConfig(
              name='num_reviews',
              reflects_trust_in=[
                  configs.TrustConfig(
                      feature_name='average_rating', trust_type='edgeworth'),
              ],
          ),
          tfl.configs.FeatureConfig(
              name='average_rating',
          ),
      ])
  ```

  Features can reflect positive or negative trust in other features. For example
  if the task is to estimate a property price in a neighborhood given two
  average prices for commercial and residential properties, you can use a trust
  feature `percentage_commercial_properties` to indicate that the model should
  more responsive to commercial estimate if more properties are commercial in
  the neighborhood. You can simultaneously have a negative trust constratins for
  residential properties, since higher commercial land usage indicates fewer
  houses, hence less market influence and less accurate estimate for residential
  property prices.

  ```python
  model_config = tfl.configs.CalibratedLatticeConfig(
      feature_configs=[
          tfl.configs.FeatureConfig(
              name='percentage_commercial_properties',
              reflects_trust_in=[
                  configs.TrustConfig(
                      feature_name='average_commercial_property_price',
                      direction='positive'),
                  configs.TrustConfig(
                      feature_name='average_residential_property_price',
                      direction='negative'),
              ],
          ),
          tfl.configs.FeatureConfig(
              name='average_commercial_property_price',
          ),
          tfl.configs.FeatureConfig(
              name='average_residential_property_price',
          ),
          tfl.configs.FeatureConfig(
              name='square_footage',
          ),
          ...
      ])
  ```
  """
  _HAS_DYNAMIC_ATTRIBUTES = True  # Required for pytype checks.

  def __init__(self,
               feature_name,
               trust_type='edgeworth',
               direction='positive'):
    """Initializes a `TrustConfig` instance.

    Args:
      feature_name: Name of the "main" feature for the trust constraint.
      trust_type: Type of trust constraint. Either `'edgeworth'` or
        `'trapezoid'`.
      direction: Direction of the trust. Should be: `'positive'`, `'negative'`,
        1 or -1.
    """
    super(TrustConfig, self).__init__(locals())

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return TrustConfig(**_Config.deserialize_nested_configs(
        config, custom_objects=custom_objects))


class DominanceConfig(_Config):
  """Configuration for dominance constraints in TFL canned estimators.

  You can specify how a feature dominantes another feature. Supported dominance
  types (see `tfl.layers.Lattice` and `tfl.layers.Linear` for details):

  - `'monotonic'`: Monotonic dominance constrains the function to require the
      effect (slope) in the direction of the *dominant* dimension to be greater
      than that of the *weak* dimension for any point in both lattice and linear
      models. Both dominant and weak dimensions must be monotonic. The
      constraint is guranteed to satisfy at the end of training for linear
      models, but might not be strictly satisified for lattice models. In such
      cases, increase the number of projection iterations.

  Example:

  ```python
  model_config = tfl.configs.CalibratedLatticeConfig(
      feature_configs=[
          tfl.configs.FeatureConfig(
              name='num_purchases',
              dominates=[
                  configs.DominanceConfig(
                      feature_name='num_clicks', dominance_type='monotonic'),
              ],
          ),
          tfl.configs.FeatureConfig(
              name='num_clicks',
          ),
      ])
  ```
  """
  _HAS_DYNAMIC_ATTRIBUTES = True  # Required for pytype checks.

  def __init__(self, feature_name, dominance_type='monotonic'):
    """Initializes a `DominanceConfig` instance.

    Args:
      feature_name: Name of the `"dominant"` feature for the dominance
        constraint.
      dominance_type: Type of dominance constraint. Currently, supports
        `'monotonic'`.
    """
    super(DominanceConfig, self).__init__(locals())

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return DominanceConfig(**_Config.deserialize_nested_configs(
        config, custom_objects=custom_objects))


class _TypeDict(collections.defaultdict):
  """Type dict that defaults to string type for hparams."""

  def __init__(self, hparams):
    super(_TypeDict,
          self).__init__(lambda: str,
                         {k: type(v) for k, v in hparams.values().items()})

  def __contains__(self, _):
    return True


def apply_updates(model_config, updates):
  """Updates a model config with the given set of (key, values) updates.

  Any value passed in the updates that matches a field of the config will be
  applied to the config. Nested configs can be updated as follows: to add/update
  a field `FIELD` in feature config for feature `FEATURE`, use
  `feature__FEATURE__FIELD` as the key. To add/update a field `FIELD` for
  regularizer with name `REGULARIZER` use `regularizer__REGULARIZER__FIELD` as
  the key. This naming scheme can be nested. When possible, string values will
  be converted to the corresponding value type in the model config.

  Example:

  ```python
  model_config = ...
  updates = [
      ('output_max', 1),
      ('regularizer__torsion__l1', 0.001),
      ('feature__some_feature_name__lattice_size', 4),
      ('feature__some_feature_name__regularizer__calib_hessian__l2', 0.001),
      ('unrelated_haparam_not_affecting_model_config', 42),
  ]
  configs.apply_updates(model_config, updates)
  ```

  Arguments:
    model_config: The model config object to apply the updates to.
    updates: A list of (key, value) pairs with potential config updates. Values
      that are not matched to a field in the model config will be ignored.

  Returns:
    Number of updates that are applied to the model config.
  """
  applied_updates = 0
  for k, v in updates:
    if _apply_update(model_config, k, v):
      applied_updates += 1
      logging.info('Updated model config with %s=%s', k, str(v))
  return applied_updates


def _apply_update(node, k, v):
  """Applies k, v updates to the given config node. See apply_updates."""
  while '__' in k:
    parts = k.split('__', 2)
    if len(parts) != 3:
      return False
    prefix, child_node_name, k = parts
    if (prefix == _HPARAM_FEATURE_PREFIX and
        isinstance(node, _HasFeatureConfigs)):
      node = node.feature_config_by_name(child_node_name)
    elif (prefix == _HPARAM_REGULARIZER_PREFIX and
          isinstance(node, _HasRegularizerConfigs)):
      node = node.regularizer_config_by_name(child_node_name)
    else:
      return False

  if hasattr(node, k):
    if isinstance(v, str):
      current_value = getattr(node, k)
      if current_value is None:
        raise ValueError(
            'Field `{}` has None value and can not be overridden by the '
            'hparams string value `{}` since the type cannot be inferred. An '
            'initial value must be set for the field to use string hparams.'
            .format(k, v))
      v = type(current_value)(v)

    setattr(node, k, v)
    return True

  return False
