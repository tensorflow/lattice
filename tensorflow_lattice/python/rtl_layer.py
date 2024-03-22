# Copyright 2020 Google LLC
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
"""Layer which represents an ensemble of Random Tiny Lattices (RTL).

See class level comment.

This layer can take multiple inputs and use them in an ensemble of lattices.
The output can be set to be monotonic with respect to a subset of features. This
layer can output either a single dense tensor, or can have separate monotonic
and unconstrained outputs to be fed into another RTL layer.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools

from absl import logging
import numpy as np
import six
import tensorflow as tf
# pylint: disable=g-import-not-at-top
# Use Keras 2.
version_fn = getattr(tf.keras, 'version', None)
if version_fn and version_fn().startswith('3.'):
  import tf_keras as keras
else:
  keras = tf.keras

from . import kronecker_factored_lattice_layer as kfll
from . import lattice_layer
from . import rtl_lib

_MAX_RTL_SWAPS = 10000
_RTLInput = collections.namedtuple('_RTLInput',
                                   ['monotonicity', 'group', 'input_index'])
RTL_KFL_NAME = 'rtl_kronecker_factored_lattice'
RTL_LATTICE_NAME = 'rtl_lattice'
INPUTS_FOR_UNITS_PREFIX = 'inputs_for_lattice'
RTL_CONCAT_NAME = 'rtl_concat'


class RTL(keras.layers.Layer):
  # pyformat: disable
  """Layer which includes a random ensemble of lattices.

  RTL (Random Tiny Lattices) is an ensemble of `tfl.layers.Lattice` layers that
  takes in a collection of monotonic and unconstrained features and randomly
  arranges them into lattices of a given rank. The input is taken as "groups",
  and inputs from the same group will not be used in the same lattice. E.g. the
  input can be the output of a calibration layer with multiple units applied to
  the same input feature. If there are more slots in the RTL than the number of
  inputs, inputs will be repeatedly used. Repeats will be approximately uniform
  across all inputs.

  Input shape:
  One of:
    - A dict with keys in `['unconstrained', 'increasing']`, and the values
      either a list of tensors of shape (batch_size, D_i), or a single tensor
      of shape (batch_size, D) that will be conceptually split into a list of D
      tensors of size (batch_size, 1). Each tensor in the list is considered a
      "group" of features that the RTL layer should try not to use in the same
      lattice.
    - A single tensor of shape (batch_size, D), which is considered to be
      unconstrained and will be conceptually split into a list of D tensors of
      size (batch_size, 1).

  Output shape:
  If `separate_outputs == True`, the output will be in the same format as the
  input and can be passed to follow on RTL layers:
  `{'unconstrained': unconstrained_out, 'increasing': mon_out}` where
  `unconstrained_out` and `mon_out` are of (batch_size, num_unconstrained_out)
  and (batch_size, num_mon_out) respectively, and
  `num_unconstrained_out + num_mon_out == num_lattices`. If
  `separate_outputs == False` the output will be a rank-2 tensor with shape:
  (batch_size, num_lattices) if average_outputs is False, or (batch_size, 1) if
  True.

  Attributes:
    - All `__init__ `arguments.

  Example:

  ```python
  a = keras.Input(shape=(1,))
  b = keras.Input(shape=(1,))
  c = keras.Input(shape=(1,))
  d = keras.Input(shape=(1,))
  cal_a = tfl.layers.CategoricalCalibration(
      units=10, output_min=0, output_max=1, ...)(a)
  cal_b = tfl.layers.PWLCalibration(
      units=20, output_min=0, output_max=1, ...)(b)
  cal_c = tfl.layers.PWLCalibration(
      units=10, output_min=0, output_max=1, monotonicity='increasing', ...)(c)
  cal_d = tfl.layers.PWLCalibration(
      units=20, output_min=0, output_max=1, monotonicity='decreasing', ...)(d)
  rtl_0 = RTL(
      num_lattices=20,
      lattice_rank=3,
      output_min=0,
      output_max=1,
      separate_outputs=True,
  )({
      'unconstrained': [cal_a, cal_b],
      'increasing': [cal_c, cal_d],
  })
  rtl_1 = RTL(num_lattices=5, lattice_rank=4)(rtl_0)
  outputs = tfl.layers.Linear(
      num_input_dims=5,
      monotonicities=['increasing'] * 5,
  )(rtl_1)
  model = keras.Model(inputs=[a, b, c, d], outputs=outputs)
  ```
  """
  # pyformat: enable

  def __init__(self,
               num_lattices,
               lattice_rank,
               lattice_size=2,
               output_min=None,
               output_max=None,
               init_min=None,
               init_max=None,
               separate_outputs=False,
               random_seed=42,
               num_projection_iterations=10,
               monotonic_at_every_step=True,
               clip_inputs=True,
               interpolation='hypercube',
               parameterization='all_vertices',
               num_terms=2,
               avoid_intragroup_interaction=True,
               kernel_initializer='random_monotonic_initializer',
               kernel_regularizer=None,
               average_outputs=False,
               **kwargs):
    # pyformat: disable
    """Initializes an instance of `RTL`.

    Args:
      num_lattices: Number of lattices in the ensemble.
      lattice_rank: Number of features used in each lattice.
      lattice_size: Number of lattice vertices per dimension (minimum is 2).
      output_min: None or lower bound of the output.
      output_max: None or upper bound of the output.
      init_min: None or lower bound of lattice kernel initialization.
      init_max: None or upper bound of lattice kernel initialization.
      separate_outputs: If set to true, the output will be a dict in the same
        format as the input to the layer, ready to be passed to another RTL
        layer. If false, the output will be a single tensor of shape
        (batch_size, num_lattices). See output shape for details.
      random_seed: Random seed for the randomized feature arrangement in the
        ensemble. Also used for initialization of lattices using
        `'kronecker_factored'` parameterization.
      num_projection_iterations: Number of iterations of Dykstra projections
        algorithm. Projection updates will be closer to a true projection (with
        respect to the L2 norm) with higher number of iterations. Increasing
        this number has diminishing return on projection precsion. Infinite
        number of iterations would yield perfect projection. Increasing this
        number might slightly improve convergence by cost of slightly increasing
        running time. Most likely you want this number to be proportional to
        number of lattice vertices in largest constrained dimension.
      monotonic_at_every_step: Whether to strictly enforce monotonicity and
        trust constraints after every gradient update by applying a final
        imprecise projection. Setting this parameter to True together with small
        num_projection_iterations parameter is likely to hurt convergence.
      clip_inputs: If inputs should be clipped to the input range of the
        lattice.
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
            having `feature_configs[0].lattice_size` entries (note that the
            lattice size of the first feature will be used as the lattice size
            for all other features as well). To obtain the tensor of lattice
            vertex values, for `t=1,2,...,num_terms` we compute the outer
            product of the `t'th` vector in each collection, multiply by a
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
      avoid_intragroup_interaction: If set to true, the RTL algorithm will try
        to avoid having inputs from the same group in the same lattice.
      kernel_initializer: One of:
        - `'linear_initializer'`: initialize parameters to form a linear
          function with positive and equal coefficients for monotonic dimensions
          and 0.0 coefficients for other dimensions. Linear function is such
          that minimum possible output is equal to output_min and maximum
          possible output is equal to output_max. See
          `tfl.lattice_layer.LinearInitializer` class docstring for more
          details. This initialization is not supported when using the
          `'kronecker_factored'` parameterization.
        - `'random_monotonic_initializer'`: initialize parameters uniformly at
          random such that all parameters are monotonically increasing for each
          input. Parameters will be sampled uniformly at random from the range
          `[init_min, init_max]` if specified, otherwise
          `[output_min, output_max]`. See
          `tfl.lattice_layer.RandomMonotonicInitializer` class docstring for
          more details. This initialization is not supported when using the
          `'kronecker_factored'` parameterization.
        - `'kfl_random_monotonic_initializer'`: initialize parameters uniformly
          at random such that all parameters are monotonically increasing for
          each monotonic input. Parameters will be sampled uniformly at random
          from the range `[init_min, init_max]` if specified. Otherwise, the
          initialization range will be algorithmically determined depending on
          output_{min/max}. See `tfl.layers.KroneckerFactoredLattice` and
          `tfl.kronecker_factored_lattice.KFLRandomMonotonicInitializer` class
          docstrings for more details. This initialization is not supported when
          using `'all_vertices'` parameterization.
      kernel_regularizer: None or a single element or a list of following:
        - Tuple `('torsion', l1, l2)` or List `['torsion', l1, l2]` where l1 and
          l2 represent corresponding regularization amount for graph Torsion
          regularizer. l1 and l2 must be single floats. Lists of floats to
          specify different regularization amount for every dimension is not
          currently supported.
        - Tuple `('laplacian', l1, l2)` or List `['laplacian', l1, l2]` where l1
          and l2 represent corresponding regularization amount for graph
          Laplacian regularizer. l1 and l2 must be single floats. Lists of
          floats to specify different regularization amount for every dimension
          is not currently supported.
      average_outputs: Whether to average the outputs of this layer. Ignored
        when separate_outputs is True.
      **kwargs: Other args passed to `keras.layers.Layer` initializer.

    Raises:
      ValueError: If layer hyperparameters are invalid.
      ValueError: If `parameterization` is not one of `'all_vertices'` or
        `'kronecker_factored'`.
    """
    # pyformat: enable
    rtl_lib.verify_hyperparameters(
        lattice_size=lattice_size,
        output_min=output_min,
        output_max=output_max,
        interpolation=interpolation,
        parameterization=parameterization,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer)
    super(RTL, self).__init__(**kwargs)
    self.num_lattices = num_lattices
    self.lattice_rank = lattice_rank
    self.lattice_size = lattice_size
    self.output_min = output_min
    self.output_max = output_max
    self.init_min = init_min
    self.init_max = init_max
    self.separate_outputs = separate_outputs
    self.random_seed = random_seed
    self.num_projection_iterations = num_projection_iterations
    self.monotonic_at_every_step = monotonic_at_every_step
    self.clip_inputs = clip_inputs
    self.interpolation = interpolation
    self.parameterization = parameterization
    self.num_terms = num_terms
    self.avoid_intragroup_interaction = avoid_intragroup_interaction
    self.kernel_initializer = kernel_initializer
    self.kernel_regularizer = kernel_regularizer
    self.average_outputs = average_outputs

  def build(self, input_shape):
    """Standard Keras build() method."""
    rtl_lib.verify_hyperparameters(
        lattice_size=self.lattice_size, input_shape=input_shape)
    # Convert kernel regularizers to proper form (tuples).
    kernel_regularizer = self.kernel_regularizer
    if isinstance(self.kernel_regularizer, list):
      if isinstance(self.kernel_regularizer[0], six.string_types):
        kernel_regularizer = tuple(self.kernel_regularizer)
      else:
        kernel_regularizer = [tuple(r) for r in self.kernel_regularizer]
    self._rtl_structure = self._get_rtl_structure(input_shape)
    # dict from monotonicities to the lattice layers with those monotonicities.
    self._lattice_layers = {}
    for monotonicities, inputs_for_units in self._rtl_structure:
      monotonicities_str = ''.join(
          [str(monotonicity) for monotonicity in monotonicities])
      # Passthrough names for reconstructing model graph.
      inputs_for_units_name = '{}_{}'.format(INPUTS_FOR_UNITS_PREFIX,
                                             monotonicities_str)
      # Use control dependencies to save inputs_for_units as graph constant for
      # visualisation toolbox to be able to recover it from saved graph.
      # Wrap this constant into pure op since in TF 2.0 there are issues passing
      # tensors into control_dependencies.
      with tf.control_dependencies([
          tf.constant(
              inputs_for_units, dtype=tf.int32, name=inputs_for_units_name)
      ]):
        units = len(inputs_for_units)
        if self.parameterization == 'all_vertices':
          layer_name = '{}_{}'.format(RTL_LATTICE_NAME, monotonicities_str)
          lattice_sizes = [self.lattice_size] * self.lattice_rank
          kernel_initializer = lattice_layer.create_kernel_initializer(
              kernel_initializer_id=self.kernel_initializer,
              lattice_sizes=lattice_sizes,
              monotonicities=monotonicities,
              output_min=self.output_min,
              output_max=self.output_max,
              unimodalities=None,
              joint_unimodalities=None,
              init_min=self.init_min,
              init_max=self.init_max)
          self._lattice_layers[str(monotonicities)] = lattice_layer.Lattice(
              lattice_sizes=lattice_sizes,
              units=units,
              monotonicities=monotonicities,
              output_min=self.output_min,
              output_max=self.output_max,
              num_projection_iterations=self.num_projection_iterations,
              monotonic_at_every_step=self.monotonic_at_every_step,
              clip_inputs=self.clip_inputs,
              interpolation=self.interpolation,
              kernel_initializer=kernel_initializer,
              kernel_regularizer=kernel_regularizer,
              name=layer_name,
          )
        elif self.parameterization == 'kronecker_factored':
          layer_name = '{}_{}'.format(RTL_KFL_NAME, monotonicities_str)
          kernel_initializer = kfll.create_kernel_initializer(
              kernel_initializer_id=self.kernel_initializer,
              monotonicities=monotonicities,
              output_min=self.output_min,
              output_max=self.output_max,
              init_min=self.init_min,
              init_max=self.init_max)
          self._lattice_layers[str(
              monotonicities)] = kfll.KroneckerFactoredLattice(
                  lattice_sizes=self.lattice_size,
                  units=units,
                  num_terms=self.num_terms,
                  monotonicities=monotonicities,
                  output_min=self.output_min,
                  output_max=self.output_max,
                  clip_inputs=self.clip_inputs,
                  kernel_initializer=kernel_initializer,
                  scale_initializer='scale_initializer',
                  name=layer_name)
        else:
          raise ValueError('Unknown type of parameterization: {}'.format(
              self.parameterization))
    super(RTL, self).build(input_shape)

  def call(self, x, **kwargs):
    """Standard Keras call() method."""
    if not isinstance(x, dict):
      x = {'unconstrained': x}

    # Flatten the input.
    # The order for flattening should match the order in _get_rtl_structure.
    input_tensors = []
    for input_key in sorted(x.keys()):
      items = x[input_key]
      if isinstance(items, list):
        input_tensors.extend(items)
      else:
        input_tensors.append(items)
    if len(input_tensors) == 1:
      flattened_input = input_tensors[0]
    else:
      flattened_input = tf.concat(input_tensors, axis=1, name=RTL_CONCAT_NAME)

    # outputs_for_monotonicity[0] are non-monotonic outputs
    # outputs_for_monotonicity[1] are monotonic outputs
    outputs_for_monotonicity = [[], []]
    for monotonicities, inputs_for_units in self._rtl_structure:
      if len(inputs_for_units) == 1:
        inputs_for_units = inputs_for_units[0]
      lattice_inputs = tf.gather(flattened_input, inputs_for_units, axis=1)
      output_monotonicity = max(monotonicities)
      # Call each lattice layer and store based on output monotonicy.
      outputs_for_monotonicity[output_monotonicity].append(
          self._lattice_layers[str(monotonicities)](lattice_inputs))

    if self.separate_outputs:
      separate_outputs = {}
      for monotoncity, output_key in [(0, 'unconstrained'), (1, 'increasing')]:
        lattice_outputs = outputs_for_monotonicity[monotoncity]
        if not lattice_outputs:
          # Do not need to add empty list to the output.
          pass
        elif len(lattice_outputs) == 1:
          separate_outputs[output_key] = lattice_outputs[0]
        else:
          separate_outputs[output_key] = tf.concat(lattice_outputs, axis=1)
      return separate_outputs
    else:
      joint_outputs = outputs_for_monotonicity[0] + outputs_for_monotonicity[1]
      if len(joint_outputs) > 1:
        joint_outputs = tf.concat(joint_outputs, axis=1)
      else:
        joint_outputs = joint_outputs[0]
      if self.average_outputs:
        joint_outputs = tf.reduce_mean(joint_outputs, axis=-1, keepdims=True)
      return joint_outputs

  def compute_output_shape(self, input_shape):
    """Standard Keras compute_output_shape() method."""
    if isinstance(input_shape, dict):
      batch_size = list(input_shape.values())[0][0]
    else:
      batch_size = input_shape[0]
    if not self.separate_outputs:
      if self.average_outputs:
        return (batch_size, 1)
      else:
        return (batch_size, self.num_lattices)
    num_outputs = [0, 0]
    for monotonicities, inputs_for_units in self._rtl_structure:
      output_monotonicity = max(monotonicities)
      num_outputs[output_monotonicity] += len(inputs_for_units)
    output_shape = {}
    if num_outputs[0]:
      output_shape['unconstrained'] = (batch_size, num_outputs[0])
    if num_outputs[1]:
      output_shape['increasing'] = (batch_size, num_outputs[1])
    return output_shape

  def get_config(self):
    """Standard Keras get_config() method."""
    config = super(RTL, self).get_config()
    config.update({
        'num_lattices': self.num_lattices,
        'lattice_rank': self.lattice_rank,
        'lattice_size': self.lattice_size,
        'output_min': self.output_min,
        'output_max': self.output_max,
        'init_min': self.init_min,
        'init_max': self.init_max,
        'separate_outputs': self.separate_outputs,
        'random_seed': self.random_seed,
        'num_projection_iterations': self.num_projection_iterations,
        'monotonic_at_every_step': self.monotonic_at_every_step,
        'clip_inputs': self.clip_inputs,
        'interpolation': self.interpolation,
        'parameterization': self.parameterization,
        'num_terms': self.num_terms,
        'avoid_intragroup_interaction': self.avoid_intragroup_interaction,
        'kernel_initializer': self.kernel_initializer,
        'kernel_regularizer': self.kernel_regularizer,
        'average_outputs': self.average_outputs,
    })
    return config

  def finalize_constraints(self):
    """Ensures layers weights strictly satisfy constraints.

    Applies approximate projection to strictly satisfy specified constraints.
    If `monotonic_at_every_step == True` there is no need to call this function.

    Returns:
      In eager mode directly updates weights and returns variable which stores
      them. In graph mode returns a list of `assign_add` op which has to be
      executed to updates weights.
    """
    return list(lattice_layer.finalize_constraints()
                for lattice_layer in self._lattice_layers.values())

  def assert_constraints(self, eps=1e-6):
    """Asserts that weights satisfy all constraints.

    In graph mode builds and returns a list of assertion ops.
    In eager mode directly executes assertions.

    Args:
      eps: allowed constraints violation.

    Returns:
      List of assertion ops in graph mode or immediately asserts in eager mode.
    """
    assertions = []
    for layer in self._lattice_layers.values():
      assertions.extend(layer.assert_constraints(eps))
    return assertions

  def _get_rtl_structure(self, input_shape):
    """Returns the RTL structure for the given input_shape.

    Args:
      input_shape: Input shape to the layer. Must be a dict matching the format
        described in the layer description.

    Raises:
      ValueError: If the structure is too small to include all the inputs.

    Returns:
      A list of `(monotonicities, lattices)` tuples, where `monotonicities` is
      the tuple of lattice monotonicites, and `lattices` is a list of list of
      indices into the flattened input to the layer.
    """
    if not isinstance(input_shape, dict):
      input_shape = {'unconstrained': input_shape}

    # Calculate the flattened input to the RTL layer. rtl_inputs will be a list
    # of _RTLInput items, each including information about the monotonicity,
    # input group and input index for each input to the layer.
    # The order for flattening should match the order in the call method.
    rtl_inputs = []
    group = 0  # group id for the input
    input_index = 0  # index into the flattened input
    for input_key in sorted(input_shape.keys()):
      shapes = input_shape[input_key]
      if input_key == 'unconstrained':
        monotonicity = 0
      elif input_key == 'increasing':
        monotonicity = 1
      else:
        raise ValueError(
            'Unrecognized key in the input to the RTL layer: {}'.format(
                input_key))

      if not isinstance(shapes, list):
        # Get the shape after a split. See single dense tensor input format in
        # the layer comments.
        shapes = [(shapes[0], 1)] * shapes[1]

      for shape in shapes:
        for _ in range(shape[1]):
          rtl_inputs.append(
              _RTLInput(
                  monotonicity=monotonicity,
                  group=group,
                  input_index=input_index))
          input_index += 1
        group += 1

    total_usage = self.num_lattices * self.lattice_rank
    if total_usage < len(rtl_inputs):
      raise ValueError(
          'RTL layer with {}x{}D lattices is too small to use all the {} input '
          'features'.format(self.num_lattices, self.lattice_rank,
                            len(rtl_inputs)))

    # Repeat the features to fill all the slots in the RTL layer.
    rs = np.random.RandomState(self.random_seed)
    rs.shuffle(rtl_inputs)
    rtl_inputs = rtl_inputs * (1 + total_usage // len(rtl_inputs))
    rtl_inputs = rtl_inputs[:total_usage]
    rs.shuffle(rtl_inputs)

    # Start with random lattices, possibly with repeated groups in lattices.
    lattices = []
    for lattice_index in range(self.num_lattices):
      lattices.append(
          rtl_inputs[lattice_index * self.lattice_rank:(lattice_index + 1) *
                     self.lattice_rank])

    # Swap features between lattices to make sure only a single input from each
    # group is used in each lattice.
    changed = True
    iteration = 0
    while changed and self.avoid_intragroup_interaction:
      if iteration > _MAX_RTL_SWAPS:
        logging.info('Some lattices in the RTL layer might use features from '
                     'the same input group')
        break
      changed = False
      iteration += 1
      for lattice_0, lattice_1 in itertools.combinations(lattices, 2):
        # For every pair of lattices: lattice_0, lattice_1
        for index_0, index_1 in itertools.product(
            range(len(lattice_0)), range(len(lattice_1))):
          # Consider swapping lattice_0[index_0] with lattice_1[index_1]
          rest_lattice_0 = list(lattice_0)
          rest_lattice_1 = list(lattice_1)
          feature_0 = rest_lattice_0.pop(index_0)
          feature_1 = rest_lattice_1.pop(index_1)
          if feature_0.group == feature_1.group:
            continue

          # Swap if a group is repeated and a swap fixes it.
          rest_lattice_groups_0 = list(
              lattice_input.group for lattice_input in rest_lattice_0)
          rest_lattice_groups_1 = list(
              lattice_input.group for lattice_input in rest_lattice_1)
          if ((feature_0.group in rest_lattice_groups_0) and
              (feature_0.group not in rest_lattice_groups_1) and
              (feature_1.group not in rest_lattice_groups_0)):
            lattice_0[index_0], lattice_1[index_1] = (lattice_1[index_1],
                                                      lattice_0[index_0])
            changed = True

    # Arrange into combined lattices layers. Lattices with similar monotonicites
    # can use the same tfl.layers.Lattice layer.
    # Create a dict: monotonicity -> list of list of input indices.
    lattices_for_monotonicities = collections.defaultdict(list)
    for lattice in lattices:
      lattice.sort(key=lambda lattice_input: lattice_input.monotonicity)
      monotonicities = tuple(
          lattice_input.monotonicity for lattice_input in lattice)
      lattice_input_indices = list(
          lattice_input.input_index for lattice_input in lattice)
      lattices_for_monotonicities[monotonicities].append(lattice_input_indices)

    return sorted(lattices_for_monotonicities.items())
