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
"""Hyper-parameters support classes for TensorFlow Lattice estimators."""
from distutils.util import strtobool

import six
from tensorflow_lattice.python.lib import regularizers


class PerFeatureHParams(object):
  """Parameters object with per feature parametrization.

  Each parameter can be overwritten for specific features by setting
  `feature__<feature_name>__<parameter_name>`, otherwise it falls back to the
  global parameter name value `<parameter_name>`.

  Parameter types are set from their first value set -- but they can also be
  reset by `set_param_type`.

  Example: let's say we have a parameter `lattice_size` that should be 2 if not
  specified (global value), but can be overridden per feature; let's assume
  there are 3 features: `a`, `b`, and `c` (added after construction). Then:

  ```python
      hparams = PerFeatureHParams(["a", "b"], lattice_size=2,
                                  feature__b__lattice_size=3)
      hparams.add_feature(["c"])
      hparams.get_param("lattice_size") == 2
      hparams.get_feature_param("a", "lattice_size") == 2
      hparams.get_feature_param("b", "lattice_size") == 3
      hparams.get_feature_param("c", "lattice_size") == 2
      hparams.get_feature_param("d", "lattice_size") raises a ValueError
  ```

  Use the `get_feature_param` method to automatically get the specialized value,
  or fall-back to the global one.




  """

  # Used to separate feature prefix, name and parameter name.
  FEATURE_SEPARATOR = '__'

  # Feature prefix for feature specific parameter values.
  FEATURE_PREFIX = 'feature'

  def __init__(self, feature_names=None, **kwargs):
    """Construct with arbitrary list of parameters.

    Args:
      feature_names: list of feature names. Only features names listed here
        (or added later with add_feature) can have feature specific parameter
        values.
      **kwargs: parameters names.

    Returns:
      PerFeatureHParams object.

    Raises:
      ValueError: if a feature-specific parameter value is set for an
        unknown feature.
    """
    super(PerFeatureHParams, self).__init__()
    self._data = {}
    self._params_type = {}
    self._feature_names = set(
        feature_names) if feature_names is not None else set()
    for feature_name in self._feature_names:
      PerFeatureHParams._check_feature_name(feature_name)

    # First set the global parameters, so they become known and then feature
    # specific parameters.
    for param_name, value in six.iteritems(kwargs):
      if not PerFeatureHParams._is_feature_specific(param_name):
        self.set_param(param_name, value)
    for param_name, value in six.iteritems(kwargs):
      if PerFeatureHParams._is_feature_specific(param_name):
        self.set_param(param_name, value)

  @staticmethod
  def _check_feature_name(feature_name):
    """Raises ValueError if feature_name is not valid."""
    if (PerFeatureHParams.FEATURE_SEPARATOR in feature_name or
        '=' in feature_name):
      raise ValueError(
          'Invalid feature name "{}": "{}" and "=" are not supported in '
          'feature names'.format(feature_name,
                                 PerFeatureHParams.FEATURE_SEPARATOR))

  @staticmethod
  def _is_feature_specific(param_name):
    return param_name.startswith(PerFeatureHParams.FEATURE_PREFIX +
                                 PerFeatureHParams.FEATURE_SEPARATOR)

  def get_feature_names(self):
    """Returns copy of list of known feature names."""
    feature_names_list = list(self._feature_names)
    feature_names_list.sort()
    return feature_names_list

  def add_feature(self, feature_name):
    """Add feature_name (one name or list of names) to list of known names."""
    if isinstance(feature_name, list):
      # Add all elements in the list, if a list.
      for f in feature_name:
        if not isinstance(f, six.string_types):
          raise ValueError(
              'feature_name should either be a list of strings, or a string, '
              'got "%s"' % feature_name)
        PerFeatureHParams._check_feature_name(f)
        self._feature_names.add(f)
    elif isinstance(feature_name, six.string_types):
      PerFeatureHParams._check_feature_name(feature_name)
      self._feature_names.add(feature_name)
    else:
      raise ValueError(
          'feature_name should either be a list of strings, or a string, '
          'got "%s"' % feature_name)
    return self

  def param_name_for_feature(self, feature_name, param_name):
    """Returns parameter name for specific feature parameter."""
    if feature_name not in self._feature_names:
      raise ValueError('Unknown feature name "%s" for parameter "%s"' %
                       (feature_name, param_name))
    return PerFeatureHParams.FEATURE_SEPARATOR.join(
        [PerFeatureHParams.FEATURE_PREFIX, feature_name, param_name])

  def is_feature_set_param(self, feature_name, param_name):
    """Returns whether param_name parameter is set for feature_name."""
    key = self.param_name_for_feature(feature_name, param_name)
    return hasattr(self, key)

  def get_feature_param(self, feature_name, param_name, default=None):
    """Returns parameter for feature or falls back to global parameter."""
    key = self.param_name_for_feature(feature_name, param_name)
    if hasattr(self, key):
      return getattr(self, key, None)
    return getattr(self, param_name, default)

  def set_feature_param(self, feature_name, param_name, value):
    """Sets parameter value specific for feature. Returns self."""
    if feature_name not in self.get_feature_names():
      raise ValueError(
          'Unknown feature name "%s" when trying to set parameter "%s", known '
          'values are %s' % (feature_name, param_name,
                             self.get_feature_names()))
    if param_name not in self._params_type:
      raise ValueError(
          'Unknown parameter name "%s" when trying to set parameter for '
          'feature "%s"' % (param_name, feature_name))

    key = self.param_name_for_feature(feature_name, param_name)
    self._data[key] = value
    return self

  def get_param(self, param_name, default=None):
    """Returns the global parameter or falls back to default."""
    return self._data[param_name] if param_name in self._data else default

  def __getattr__(self, param_name):
    if param_name.startswith('_') or param_name not in self._data:
      raise AttributeError('No value set for "{}"'.format(param_name))
    return self._data[param_name]

  @staticmethod
  def _parse_value(value_str, value_type):
    """Parses string a the given value_type."""
    if value_type is str:
      return value_str
    elif value_type is int:
      return int(value_str)
    elif value_type is float:
      return float(value_str)
    elif value_type is bool:
      return strtobool(value_str)

    raise ValueError(
        'Do not know how to parse types {} -- value was {!r}'.format(
            value_type, value_str))

  def _set_param(self, param_name, value, parse):
    """Sets parameter, optionally parse it."""
    # Make sure that feature specific parameters are properly named.
    if PerFeatureHParams._is_feature_specific(param_name):
      parts = param_name.split(PerFeatureHParams.FEATURE_SEPARATOR, 3)
      if len(parts) != 3:
        raise ValueError(
            'Bad formatted feature specific parameter "{}", please use '
            '"{}{}<feature_name>{}<parameter_name>"'.format(
                param_name, PerFeatureHParams.FEATURE_PREFIX,
                PerFeatureHParams.FEATURE_SEPARATOR,
                PerFeatureHParams.FEATURE_SEPARATOR))
      if parts[1] not in self._feature_names:
        raise ValueError(
            'Unknown feature "{}" for feature specific parameter "{}"'.format(
                parts[1], param_name))
      if parts[2] not in self._params_type:
        raise ValueError(
            'Unknown parameter name "{}", can not set for feature "{}"'.format(
                parts[2], parts[1]))
      if parse:
        value = PerFeatureHParams._parse_value(value,
                                               self._params_type[parts[2]])
    else:
      # Non-feature specific parameter: set _param_type if not yet set.
      if param_name not in self._params_type:
        if parse:
          raise ValueError(
              'Parsing value for unknown parameter "{}"'.format(param_name))
        self._params_type[param_name] = type(value)
      elif parse:
        value = PerFeatureHParams._parse_value(value,
                                               self._params_type[param_name])
    self._data[param_name] = value

  def set_param(self, param_name, value):
    """Sets parameter value. Returns self."""
    self._set_param(param_name, value, parse=False)
    return self

  def set_param_type(self, param_name, param_type):
    """Sets the parameter type, it must already exist. Returns self."""
    if param_name not in self._params_type:
      raise ValueError(
          'Can not set parameter type if parameter has not been set for "{}"'.
          format(param_name))
    self._params_type[param_name] = param_type

  def parse_param(self, param_name, value_str):
    """Parses parameter values from string. Returns self."""
    self._set_param(param_name, value_str, parse=True)
    return self

  def get_global_and_feature_params(self, param_names, feature_names):
    """Returns values for multiple params, global and for each feature.

    Args:
      param_names: list of parameters to get values for.
      feature_names: list of features to get specific values for.

    Returns:
      * List of global values for parameters requested in `param_names`.
      * List of list of per feature values for parameters requested in
        `param_names` for features requested in `feature_names`.
    """
    global_values = [self.get_param(param_name) for param_name in param_names]
    feature_values = []
    for feature in feature_names:
      feature_values.append([
          self.get_feature_param(feature, param_name)
          for param_name in param_names
      ])
    return (global_values, feature_values)

  def values(self):
    """Returns shallow copy of the hyperparameter dict."""
    return {k: v for k, v in six.iteritems(self._data)}

  def __str__(self):
    return str(sorted(self.values().items()))

  def parse_hparams(self, hparams):
    """Incorporates hyper-parameters from another HParams object.

    Copies over values of hyper-parameters from the given object. New parameters
    may be set, but not new features. Also works with
    `tf.contrib.training.HParams` objects.

    Args:
      hparams: `PerFeatureHParams` object, but also works with the standard
        `tf.contrib.training.HParams` object.

    Returns:
      Changes affect self, but returns self for convenience.

    Raises:
      ValueError: if trying to set unknown features, or if setting a feature
        specific parameter for an unknown parameter.
    """
    # First set the global parameters, so they become known and then feature
    # specific parameters.
    if hparams is not None:
      for param_name, value in six.iteritems(hparams.values()):
        if not PerFeatureHParams._is_feature_specific(param_name):
          self.set_param(param_name, value)
      for param_name, value in six.iteritems(hparams.values()):
        if PerFeatureHParams._is_feature_specific(param_name):
          self.set_param(param_name, value)
    return self

  def parse(self, hparams_str):
    """Parses strings into hparams.

    Args:
      hparams_str: must be a comma separated list of "<key>=<value>",
      where "<key>" is a hyper-parameter name, and "<value>" its value.

    Returns:
      Changes affect self, but returns self for convenience.

    Raises:
      ValueError: if there is a problem with the input:
         * if trying to set an unknown parameter.
         * if trying to set unknown feature(s)
         * if can't convert value to parameter type.
    """
    if hparams_str:
      for pair in hparams_str.split(','):
        (key, value) = pair.split('=')
        self.parse_param(key, value)
    return self


class CalibratedHParams(PerFeatureHParams):
  """PerFeatureHParams specialization with input calibration parameters.

  The following hyper-parameters can be set as global, or per-feature (see
  base `PerFeatureHParams` for details):

    * `feature_names`: list of feature names. Only features names listed here
      (or added later with add_feature) can have feature specific parameter
      values.
    * `num_keypoints`: Number of keypoints to use for calibration, Set to 0 or
      `None` for no calibration.
    * `calibration_output_min`, `calibration_output_max`: initial and final
      values for calibrations. -1.0 to 1.0 works well for calibrated linear
      models. For lattices one will want to set these to (0, `lattice_size`-1).
      Only used during initialization of the calibration, if `quantiles_dir`
      is given to the calibrated model (as opposed to defining one's own value
      with `keypoints_initializers_fn`). It must be defined for calibration to
      work, no default is set.
    * `calibration_bound`: If output of calibration max/min are bound to the
      limits given in `calibration_output_min/max`.
    * `monotonicity`: Monotonicity for the feature. 0 for no monotonicity,
      1 and -1 for increasing and decreasing monotonicity respectively.
    * `missing_input_value`: If set, and if the input has this value it is
    assumed
      to be missing and the output will either be calibrated to some value
      between `[calibration_output_min, calibration_output_max]` or set to a
      fixed value set by missing_output_value.
    * `missing_output_value`: Requires missing_input_value also to be set. If
    set
      if will convert missing input to this value. Leave it undefined and the
      output will be learned.
    * `calibration_<regularizer_name>` for all regularizer_name's in
      regularizers.CALIBRATOR_REGULARIZERS. e.g. `calibration_l2_reg`.
  """

  def __init__(self, feature_names=None, **kwargs):
    # Set default args, and override with given ones.
    args = {
        'num_keypoints': 10,
        'calibration_output_min': None,
        'calibration_output_max': None,
        'calibration_bound': False,
        'monotonicity': 0,
        'missing_input_value': None,
        'missing_output_value': None,
    }
    regularizer_hparam_names = [
        'calibration_{}'.format(regularizer_name)
        for regularizer_name in regularizers.CALIBRATOR_REGULARIZERS
    ]
    args.update({
        regularizer_name: None for regularizer_name in regularizer_hparam_names
    })
    args.update(kwargs)
    super(CalibratedHParams, self).__init__(feature_names, **args)
    self.set_param_type('monotonicity', int)
    self.set_param_type('calibration_output_min', float)
    self.set_param_type('calibration_output_max', float)
    self.set_param_type('missing_input_value', float)
    self.set_param_type('missing_output_value', float)
    for regularizer_name in regularizer_hparam_names:
      self.set_param_type(regularizer_name, float)


class CalibratedLinearHParams(CalibratedHParams):
  """Hyper-parameters for CalibratedLinear models.

  Same as `CalibratedHParams` (hyper-parameters for input calibration) plus
  the global learning_rate.

  The parameters `calibration_output_min` and `calibration_output_max` shouldn't
  be changed (they are fixed at -1. and +1), since they are eventually re-scaled
  by the linear layer on top.

  It supports regularization, monotonicity and missing values (input and
  optionally output).
  """

  def __init__(self, feature_names=None, **kwargs):
    # Set default args, and override with given ones.
    args = {
        'learning_rate': 0.1,
        'calibration_output_min': -1.,
        'calibration_output_max': 1.,
    }
    args.update(kwargs)
    super(CalibratedLinearHParams, self).__init__(feature_names, **args)


class CalibratedLatticeHParams(CalibratedHParams):
  """Hyper-parameters for CalibratedLattice models.

  Supports regularization and monotonicity like described in `CalibratedHParam`.
  Values for `calibration_output_min`, `calibration_output_max` and
  `missing_output_value` get set automatically.

  Added parameters:

  * `learning_rate`: (float) a global parameter that assigns a step size of an
    optimizer.
  * `lattice_size`: (int) a global or per feature parameter that controls number
    of cells for a feature. Should be greater than equal to 2, and the
    recommended default value is 2. Also calibrator output min and max should be
    [0, lattice_size - 1], and the output should be bounded, since a lattice
    expects an input in the range [0, lattice_size - 1].
  * `interpolation_type`: a global parameter that defines if the lattice will
    interpolate using the full hypercube or only the simplex ("hyper-triangle",
    much faster for larger lattices) around the point being evaluated.
    Valid values: 'hypercube' or 'simplex'
  * `missing_input_value`: Value for which a feature is considered missing. Such
    values are either automatically learned to some calibrated value, or,
    if missing_vertex is set, they get their own value in the lattice.
  * `missing_vertex`: if missing_input_value is set, this boolean value indicate
    whether to create an extra vertex for missing values.
  * `lattice_<regularizer_name>` for all regularizer_name's in
    regularizers.LATTICE_REGULARIZERS. e.g. `lattice_l2_reg`.
  """

  def __init__(self, feature_names=None, **kwargs):
    # Set default args, and override with given ones.
    args = {
        'learning_rate': 0.1,
        'lattice_size': 2,
        'interpolation_type': 'hypercube',
        'calibration_bound': True,
        'missing_input_value': None,
        'missing_vertex': False,
    }
    regularizer_hparam_names = [
        'lattice_{}'.format(regularizer_name)
        for regularizer_name in regularizers.LATTICE_REGULARIZERS
    ]
    args.update({
        regularizer_name: None for regularizer_name in regularizer_hparam_names
    })
    args.update(kwargs)
    super(CalibratedLatticeHParams, self).__init__(feature_names, **args)
    self.set_param_type('missing_input_value', float)
    for regularizer_name in regularizer_hparam_names:
      self.set_param_type(regularizer_name, float)


class CalibratedRtlHParams(CalibratedHParams):
  """Hyper-parameters for CalibratedRtl (RandomTinyLattices) models.

  Supports regularization and monotonicity like described in `CalibratedHParam`.
  Values for `calibration_output_min`, `calibration_output_max` and
  `missing_output_value` get set automatically.

  Added parameters:

  * `learning_rate`: (float) a global parameter that assigns a step size of an
    optimizer.
  * `lattice_size`: (int) a global or per feature parameter that controls number
    of cells for a feature. Should be greater than equal to 2, and the
    recommended default value is 2. Also calibrator output min and max should be
    [0, lattice_size - 1], and the output should be bounded, since a lattice
    expects an input in the range [0, lattice_size - 1]. (Note if missing_vertex
    is True, then we add an extra vertex, so input range is [0, lattice_size])
  * `num_lattices`: (int) a number of lattices to be created.
  * `lattice_rank`: (int) a lattice rank in each lattice.
  * `interpolation_type`: a global parameter that defines if the lattice will
    interpolate using the full hypercube or only the simplex ("hyper-triangle",
    much faster for larger lattices) around the point being evaluated.
    Valid values: 'hypercube' or 'simplex'
  * `ensemble_bias`: (float) an initial value of bias term to be added to the
    output of ensemble.
  * `rtl_seed`: (int) a random seed for rtl construction.
  * `missing_input_value`: Value for which a feature is considered missing. Such
    values are either automatically learned to some calibrated value, or,
    if missing_vertex is set, they get their own value in the lattice.
  * `missing_vertex`: if missing_input_value is set, this boolean value indicate
    whether to create an extra vertex for missing values.
  * `lattice_<regularizer_name>` for all regularizer_name's in
    regularizers.LATTICE_REGULARIZERS. e.g. `lattice_l2_reg`.
  """

  def __init__(self, feature_names=None, **kwargs):
    # Set default args, and override with given ones.
    args = {
        'learning_rate': 0.1,
        'lattice_size': 2,
        'num_lattices': None,
        'lattice_rank': None,
        'interpolation_type': 'hypercube',
        'rtl_seed': 12345,
        'calibration_bound': True,
        'missing_input_value': None,
        'missing_vertex': False,
        'ensemble_bias': 0.0,
    }
    regularizer_hparam_names = [
        'lattice_{}'.format(regularizer_name)
        for regularizer_name in regularizers.LATTICE_REGULARIZERS
    ]
    args.update({
        regularizer_name: None for regularizer_name in regularizer_hparam_names
    })
    args.update(kwargs)
    super(CalibratedRtlHParams, self).__init__(feature_names, **args)
    self.set_param_type('num_lattices', int)
    self.set_param_type('lattice_rank', int)
    self.set_param_type('missing_input_value', float)
    for regularizer_name in regularizer_hparam_names:
      self.set_param_type(regularizer_name, float)


class CalibratedEtlHParams(CalibratedHParams):
  """Hyper-parameters for CalibratedEtl (Embedded tiny lattices) models.

  Supports regularization and monotonicity like described in `CalibratedHParam`.
  Values for `calibration_output_min`, `calibration_output_max` and
  `missing_output_value` get set automatically.

  Note that this architecture does not support any of per-feature based lattice
  hyper-parameters such as missing_vertex, per-feature missing_input_value,
  per-feature lattice_size, per-feature lattice regularization, because after
  the linear embedding, all of features are mixed together, so it is not clear
  how to merge per-feature parameters after the linear embedding layer.

  If there is no non-monotonic feature, but `non_monotonic_lattice_rank` or
  `non_monotonic_num_lattices` are not `None`, then this will raise the error.

  Added parameters:

  * `learning_rate`: (float) a global parameter that assigns a step size of an
    optimizer.
  * `lattice_size`: (int) a global parameter that controls number of
    cells for a feature. Should be greater than equal to 2, and the recommended
    default value is 2. Also calibrator output min and max should be
    [0, `lattice_size` - 1], and the output should be bounded.
  * `interpolation_type`: a global parameter that defines if the lattice will
    interpolate using the full hypercube or only the simplex ("hyper-triangle",
    much faster for larger lattices) around the point being evaluated.
    Valid values: 'hypercube' or 'simplex'
  * `monotonic_lattice_rank`: (int) a lattice rank in each monotonic lattice.
  * `monotonic_num_lattices`: (int) a number of monotonic lattices to be
    created.
  * `monotonic_lattice_size`: (int) lattice cell size for each monotonic lattice
    in the ensemble lattices layer.
  * `non_monotonic_lattice_rank`: (int) a lattice rank in each non monotonic
    lattice. If all features are monotonic, this parameter should be None.
  * `non_monotonic_num_lattices`: (int) a number of non-monotonic lattices to be
    created. If all features are monotonic, this parameter should be None.
  * `monotonic_lattice_size`: (int) lattice cell size for each non-monotonic
    lattice in the ensemble lattices layer.
  * `linear_embedding_calibration_min`: (float) a global parameter that controls
    a minimum value of intermediate calibration layers. Default is -100.
  * `linear_embedding_calibration_max`: (float) a global parameter that controls
    a maximum value of intermediate calibration layers. Default is 100.
  * `linear_embedding_calibration_num_keypoints`: (float) a global parameter
    that controls a `num_keypoints` in intermediate calibration layers. Default
    is 100.
  * `lattice_<regularizer_name>` for all regularizer_name's in
    regularizers.LATTICE_REGULARIZERS. e.g. `lattice_l2_reg`.
  """

  def __init__(self, feature_names=None, **kwargs):
    # Set default args, and override with given ones.
    args = {
        'learning_rate': 0.1,
        'monotonic_lattice_rank': None,
        'monotonic_num_lattices': None,
        'monotonic_lattice_size': None,
        'non_monotonic_lattice_rank': None,
        'non_monotonic_num_lattices': None,
        'non_monotonic_lattice_size': None,
        'interpolation_type': 'hypercube',
        'calibration_bound': True,
        'linear_embedding_calibration_min': -100.0,
        'linear_embedding_calibration_max': 100.0,
        'linear_embedding_calibration_num_keypoints': 100,
    }
    regularizer_hparam_names = [
        'lattice_{}'.format(regularizer_name)
        for regularizer_name in regularizers.LATTICE_REGULARIZERS
    ]
    args.update({
        regularizer_name: None for regularizer_name in regularizer_hparam_names
    })
    args.update(kwargs)
    super(CalibratedEtlHParams, self).__init__(feature_names, **args)
    self.set_param_type('monotonic_lattice_rank', int)
    self.set_param_type('monotonic_num_lattices', int)
    self.set_param_type('monotonic_lattice_size', int)
    self.set_param_type('non_monotonic_lattice_rank', int)
    self.set_param_type('non_monotonic_num_lattices', int)
    self.set_param_type('non_monotonic_lattice_size', int)
    self.set_param_type('linear_embedding_calibration_min', float)
    self.set_param_type('linear_embedding_calibration_max', float)
    self.set_param_type('linear_embedding_calibration_num_keypoints', int)
    for regularizer_name in regularizer_hparam_names:
      self.set_param_type(regularizer_name, float)
