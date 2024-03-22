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
"""Lattice layer with monotonicity, unimodality, trust and bound constraints.

Keras implementation of tensorflow lattice layer. This layer takes one or more
d-dimensional input(s) and combines them using a lattice function, satisfying
monotonicity, unimodality, trust and bound constraints if specified.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf
# pylint: disable=g-import-not-at-top
# Use Keras 2.
version_fn = getattr(tf.keras, "version", None)
if version_fn and version_fn().startswith("3."):
  import tf_keras as keras
else:
  keras = tf.keras
from . import lattice_lib
from . import utils

LATTICE_KERNEL_NAME = "lattice_kernel"
LATTICE_SIZES_NAME = "lattice_sizes"


class Lattice(keras.layers.Layer):
  # pyformat: disable
  """Lattice layer.

  Layer performs interpolation using one of `units` d-dimensional lattices with
  arbitrary number of keypoints per dimension. There are trainable weights
  associated with lattice vertices. Input to this layer is considered to be a
  d-dimensional point within the lattice. If point coincides with one of the
  lattice vertex then interpolation result for this point is equal to weight
  associated with that vertex. Otherwise, all surrounding vertices contribute to
  the interpolation result inversely proportional to the distance from them.

  For example lattice sizes: [2, 3] produce following lattice:

  ```
  o---o---o
  |   |   |
  o---o---o
  ```

  First coordinate of input tensor must be within [0, 1], and the second within
  [0, 2]. If coordinates are outside of this range they will be clipped into it.

  There are several types of constraints on the shape of the learned function
  that are either 1 or 2 dimensional:

  ![Shape constraint visual example images](https://www.tensorflow.org/lattice/images/2D_shape_constraints_picture_color.png)

  * **Monotonicity:** constrains the function to be increasing in the
    corresponding dimension. To achieve decreasing monotonicity, either pass the
    inputs through a `tfl.layers.PWLCalibration` with `decreasing` monotonicity,
    or manually reverse the inputs as `lattice_size - 1 - inputs`.
  * **Unimodality:** constrains the function to be unimodal in that dimension
    with minimum being in the center lattice vertex of that dimension. Single
    dimension can not be constrained to be both monotonic and unimodal.
    Unimodal dimensions must have at least 3 lattice vertices.
  * **Edgeworth Trust:** constrains the function to be more responsive to a main
    feature as a secondary conditional feature increases or decreases. For
    example, we may want the model to rely more on average rating (main
    feature) when the number of reviews (conditional feature) is high. In
    particular, the constraint guarantees that a given change in the main
    feature's value will change the model output by more when a secondary
    feature indicates higher trust in the main feature. Note that the
    constraint only works when the model is monotonic in the main feature.
  * **Trapezoid Trust:** conceptually similar to edgeworth trust, but this
    constraint guarantees that the range of possible outputs along the main
    feature dimension, when a conditional feature indicates low trust, is a
    *subset* of the range of outputs when a conditional feature indicates high
    trust. When lattices have 2 vertices in each constrained dimension, this
    implies edgeworth trust (which only constrains the size of the relevant
    ranges). With more than 2 lattice vertices per dimension, the two
    constraints diverge and are not necessarily 'weaker' or 'stronger' than
    each other - edgeworth trust acts throughout the lattice interior on delta
    shifts in the main feature, while trapezoid trust only acts on the min and
    max extremes of the main feature, constraining the overall range of
    outputs across the domain of the main feature. The two types of trust
    constraints can be applied jointly.
  * **Monotonic Dominance:** constrains the function to require the effect
    (slope) in the direction of the *dominant* dimension to be greater than that
    of the *weak* dimension for any point in the lattice. Both dominant and weak
    dimensions must be monotonic. Note that this constraint might not be
    strictly satisified at the end of training. In such cases, increase the
    number of projection iterations.
  * **Range Dominance:** constraints the function to require the range of
    possible outputs to be greater than if one varies the *dominant* dimension
    than if one varies the *weak* dimension for any point. Both dominant and
    weak dimensions must be monotonic. Note that this constraint might not be
    strictly satisified at the end of training. In such cases, increase the
    number of projection iterations.
  * **Joint Monotonicity:** constrains the function to be monotonic along a
    diagonal direction of a two dimensional subspace when all other dimensions
    are fixed. For example, if our function is scoring the profit given *A*
    hotel guests and *B* hotel beds, it may be wrong to constrain the profit to
    be increasing in either hotel guests or hotel beds in-dependently, but along
    the diagonal (+ 1 guest and +1 bed), the profit should be monotonic. Note
    that this constraint might not be strictly satisified at the end of
    training. In such cases, increase the number of projection iterations.

  There are upper and lower bound constraints on the output.

  All units share the same layer configuration, but each has their separate set
  of trained parameters.

  Input shape:
    - if `units == 1`: tensor of shape: `(batch_size, ..., len(lattice_sizes))`
      or list of `len(lattice_sizes)` tensors of same shape:
      `(batch_size, ..., 1)`
    - if `units > 1`: tensor of shape:
      `(batch_size, ..., units, len(lattice_sizes))` or list of
      `len(lattice_sizes)` tensors of same shape: `(batch_size, ..., units, 1)`

    A typical shape is: `(batch_size, len(lattice_sizes))`

  Output shape:
    Tensor of shape: `(batch_size, ..., units)`

  Attributes:
    - All `__init__` arguments.
    kernel: weights of the lattice.

  Example:

  ```python
  lattice = tfl.layers.Lattice(
      # Number of vertices along each dimension.
      lattice_sizes=[2, 2, 3, 4, 2, 2, 3],
      # You can specify monotonicity constraints.
      monotonicities=['increasing', 'none', 'increasing', 'increasing',
                      'increasing', 'increasing', 'increasing'],
      # You can specify trust constraints between pairs of features. Here we
      # constrain the function to be more responsive to a main feature (index 4)
      # as a secondary conditional feature (index 3) increases (positive
      # direction).
      edgeworth_trusts=(4, 3, 'positive'),
      # Output can be bounded.
      output_min=0.0,
      output_max=1.0)
  ```
  """
  # pyformat: enable

  def __init__(self,
               lattice_sizes,
               units=1,
               monotonicities=None,
               unimodalities=None,
               edgeworth_trusts=None,
               trapezoid_trusts=None,
               monotonic_dominances=None,
               range_dominances=None,
               joint_monotonicities=None,
               joint_unimodalities=None,
               output_min=None,
               output_max=None,
               num_projection_iterations=10,
               monotonic_at_every_step=True,
               clip_inputs=True,
               interpolation="hypercube",
               kernel_initializer="random_uniform_or_linear_initializer",
               kernel_regularizer=None,
               **kwargs):
    # pyformat: disable
    """Initializes an instance of `Lattice`.

    Args:
      lattice_sizes: List or tuple of length d of integers which represents
        number of lattice vertices per dimension (minimum is 2). Second
        dimension of input shape must match the number of elements in lattice
        sizes.
      units: Output dimension of the layer. See class comments for details.
      monotonicities: None or list or tuple of same length as lattice_sizes of
        {'none', 'increasing', 0, 1} which specifies if the model output should
        be monotonic in corresponding feature, using 'increasing' or 1 to
        indicate increasing monotonicity and 'none' or 0 to indicate no
        monotonicity constraints.
      unimodalities: None or list or tuple of same length as lattice_sizes of
        {'none', 'valley', 'peak', 0, 1, -1} which specifies if the model output
        should be unimodal in corresponding feature, using 'valley' or 1 to
        indicate that function first decreases then increases, using 'peak' or
        -1 to indicate that funciton first increases then decreases, using
        'none' or 0 to indicate no unimodality constraints.
      edgeworth_trusts: None or three-element tuple or iterable of three-element
        tuples. First element is the index of the main (monotonic) feature.
        Second element is the index of the conditional feature. Third element is
        the direction of trust: 'positive' or 1 if higher values of the
        conditional feature should increase trust in the main feature and
        'negative' or -1 otherwise.
      trapezoid_trusts: None or three-element tuple or iterable of three-element
        tuples. First element is the index of the main (monotonic) feature.
        Second element is the index of the conditional feature. Third element is
        the direction of trust: 'positive' or 1 if higher values of the
        conditional feature should increase trust in the main feature and
        'negative' or -1 otherwise.
      monotonic_dominances: None or two-element tuple or iterable of two-element
        tuples. First element is the index of the dominant feature. Second
        element is the index of the weak feature.
      range_dominances: None or two-element tuple or iterable of two-element
        tuples. First element is the index of the dominant feature. Second
        element is the index of the weak feature.
      joint_monotonicities: None or two-element tuple or iterable of two-element
        tuples which represents indices of two features requiring joint
        monotonicity.
      joint_unimodalities: None or tuple or iterable of tuples. Each tuple
        contains 2 elements: iterable of indices of single group of jointly
        unimodal features followed by string 'valley' or 'peak', using 'valley'
        to indicate that function first decreases then increases, using 'peak'
        to indicate that funciton first increases then decreases. For example:
        ([0, 3, 4], 'valley').
      output_min: None or lower bound of the output.
      output_max: None or upper bound of the output.
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
      kernel_initializer: None or one of:
        - `'linear_initializer'`: initialize parameters to form a linear
          function with positive and equal coefficients for monotonic dimensions
          and 0.0 coefficients for other dimensions. Linear function is such
          that minimum possible output is equal to output_min and maximum
          possible output is equal to output_max. See
          `tfl.lattice_layer.LinearInitializer` class docstring for more
          details.
        - `'random_monotonic_initializer'`: initialize parameters uniformly at
          random such that all parameters are monotonically increasing for each
          input. Parameters will be sampled uniformly at random from the range
          `[output_min, output_max]`. See
          `tfl.lattice_layer.RandomMonotonicInitializer` class docstring for
          more details.
        - `random_uniform_or_linear_initializer`: if the lattice has a single
          joint unimodality constraint group encompassing all features then use
          the Keras 'random_uniform' initializer; otherwise, use TFL's
          'linear_initializer'.
        - Any Keras initializer object.
      kernel_regularizer: None or a single element or a list of following:
        - Tuple `('torsion', l1, l2)` where l1 and l2 represent corresponding
          regularization amount for graph Torsion regularizer. l1 and l2 can
          either be single floats or lists of floats to specify different
          regularization amount for every dimension.
        - Tuple `('laplacian', l1, l2)` where l1 and l2 represent corresponding
          regularization amount for graph Laplacian regularizer. l1 and l2 can
          either be single floats or lists of floats to specify different
          regularization amount for every dimension.
        - Any Keras regularizer object.
      **kwargs: Other args passed to `keras.layers.Layer` initializer.

    Raises:
      ValueError: If layer hyperparameters are invalid.
    """
    # pyformat: enable
    lattice_lib.verify_hyperparameters(
        lattice_sizes=lattice_sizes,
        monotonicities=monotonicities,
        unimodalities=unimodalities,
        interpolation=interpolation)
    super(Lattice, self).__init__(**kwargs)

    self.lattice_sizes = lattice_sizes
    self.units = units
    self.monotonicities = monotonicities
    self.unimodalities = unimodalities
    # Check if inputs are a single tuple of ints (vs an iterable of tuples)
    if (isinstance(edgeworth_trusts, tuple) and
        isinstance(edgeworth_trusts[0], int)):
      self.edgeworth_trusts = [edgeworth_trusts]
    else:
      self.edgeworth_trusts = edgeworth_trusts
    if (isinstance(trapezoid_trusts, tuple) and
        isinstance(trapezoid_trusts[0], int)):
      self.trapezoid_trusts = [trapezoid_trusts]
    else:
      self.trapezoid_trusts = trapezoid_trusts
    if (isinstance(monotonic_dominances, tuple) and
        isinstance(monotonic_dominances[0], int)):
      self.monotonic_dominances = [monotonic_dominances]
    else:
      self.monotonic_dominances = monotonic_dominances
    if (isinstance(range_dominances, tuple) and
        isinstance(range_dominances[0], int)):
      self.range_dominances = [range_dominances]
    else:
      self.range_dominances = range_dominances
    if (isinstance(joint_monotonicities, tuple) and
        isinstance(joint_monotonicities[0], int)):
      self.joint_monotonicities = [joint_monotonicities]
    else:
      self.joint_monotonicities = joint_monotonicities
    if (isinstance(joint_unimodalities, tuple) and
        len(joint_unimodalities) == 2 and
        isinstance(joint_unimodalities[1], six.string_types)):
      self.joint_unimodalities = [joint_unimodalities]
    else:
      self.joint_unimodalities = joint_unimodalities
    self.output_min = output_min
    self.output_max = output_max
    self.num_projection_iterations = num_projection_iterations
    self.monotonic_at_every_step = monotonic_at_every_step
    self.clip_inputs = clip_inputs
    self.interpolation = interpolation

    self.kernel_initializer = create_kernel_initializer(
        kernel_initializer, self.lattice_sizes, self.monotonicities,
        self.output_min, self.output_max, self.unimodalities,
        self.joint_unimodalities)

    self.kernel_regularizer = []
    if kernel_regularizer:
      if (callable(kernel_regularizer) or
          (isinstance(kernel_regularizer, tuple) and
           isinstance(kernel_regularizer[0], six.string_types))):
        kernel_regularizer = [kernel_regularizer]

      for regularizer in kernel_regularizer:
        if isinstance(regularizer, tuple):
          (name, l1, l2) = regularizer
          if name.lower() == "torsion":
            self.kernel_regularizer.append(
                TorsionRegularizer(
                    lattice_sizes=self.lattice_sizes, l1=l1, l2=l2))
          elif name.lower() == "laplacian":
            self.kernel_regularizer.append(
                LaplacianRegularizer(
                    lattice_sizes=self.lattice_sizes, l1=l1, l2=l2))
          else:
            raise ValueError("Unknown custom lattice regularizer: %s" %
                             regularizer)
        else:
          # This is needed for Keras deserialization logic to be aware of our
          # custom objects.
          with keras.utils.custom_object_scope({
              "TorsionRegularizer": TorsionRegularizer,
              "LaplacianRegularizer": LaplacianRegularizer,
          }):
            self.kernel_regularizer.append(keras.regularizers.get(regularizer))

  def build(self, input_shape):
    """Standard Keras build() method."""
    lattice_lib.verify_hyperparameters(
        lattice_sizes=self.lattice_sizes,
        units=self.units,
        input_shape=input_shape)
    constraints = LatticeConstraints(
        lattice_sizes=self.lattice_sizes,
        monotonicities=self.monotonicities,
        unimodalities=self.unimodalities,
        edgeworth_trusts=self.edgeworth_trusts,
        trapezoid_trusts=self.trapezoid_trusts,
        monotonic_dominances=self.monotonic_dominances,
        range_dominances=self.range_dominances,
        joint_monotonicities=self.joint_monotonicities,
        joint_unimodalities=self.joint_unimodalities,
        output_min=self.output_min,
        output_max=self.output_max,
        num_projection_iterations=self.num_projection_iterations,
        enforce_strict_monotonicity=self.monotonic_at_every_step)

    if not self.kernel_regularizer:
      kernel_reg = None
    elif len(self.kernel_regularizer) == 1:
      kernel_reg = self.kernel_regularizer[0]
    else:
      # Keras interface assumes only one regularizer, so summ all regularization
      # losses which we have.
      kernel_reg = lambda x: tf.add_n([r(x) for r in self.kernel_regularizer])

    num_weights = 1
    for dim_size in self.lattice_sizes:
      num_weights *= dim_size
    self.kernel = self.add_weight(
        LATTICE_KERNEL_NAME,
        shape=[num_weights, self.units],
        initializer=self.kernel_initializer,
        regularizer=kernel_reg,
        constraint=constraints,
        dtype=self.dtype)

    if self.kernel_regularizer and not tf.executing_eagerly():
      # Keras has its own mechanism to handle regularization losses which does
      # not use GraphKeys, but we want to also add losses to graph keys so they
      # are easily accessable when layer is being used outside of Keras. Adding
      # losses to GraphKeys will not interfer with Keras.
      for reg in self.kernel_regularizer:
        tf.compat.v1.add_to_collection(
            tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES, reg(self.kernel))

    # Constraints with enforce_strict_monotonicity always set to True. Intended
    # to be run at the end of training or any time when you need everything to
    # be strictly projected.
    self._final_constraints = LatticeConstraints(
        lattice_sizes=self.lattice_sizes,
        monotonicities=self.monotonicities,
        unimodalities=self.unimodalities,
        edgeworth_trusts=self.edgeworth_trusts,
        trapezoid_trusts=self.trapezoid_trusts,
        monotonic_dominances=self.monotonic_dominances,
        range_dominances=self.range_dominances,
        joint_monotonicities=self.joint_monotonicities,
        joint_unimodalities=self.joint_unimodalities,
        output_min=self.output_min,
        output_max=self.output_max,
        num_projection_iterations=20,
        enforce_strict_monotonicity=True)

    self.lattice_sizes_tensor = tf.constant(
        self.lattice_sizes, dtype=tf.int32, name=LATTICE_SIZES_NAME)
    super(Lattice, self).build(input_shape)

  def call(self, inputs):
    """Standard Keras call() method."""
    # Use control dependencies to save lattice sizes as graph constant for
    # visualisation toolbox to be able to recover it from saved graph.
    # Wrap this constant into pure op since in TF 2.0 there are issues passing
    # tensors into control_dependencies.
    with tf.control_dependencies([tf.identity(self.lattice_sizes_tensor)]):
      if self.interpolation == "simplex":
        return lattice_lib.evaluate_with_simplex_interpolation(
            inputs=inputs,
            kernel=self.kernel,
            units=self.units,
            lattice_sizes=self.lattice_sizes,
            clip_inputs=self.clip_inputs)
      elif self.interpolation == "hypercube":
        return lattice_lib.evaluate_with_hypercube_interpolation(
            inputs=inputs,
            kernel=self.kernel,
            units=self.units,
            lattice_sizes=self.lattice_sizes,
            clip_inputs=self.clip_inputs)
      else:
        raise ValueError("Unknown interpolation type: %s" % self.interpolation)

  def compute_output_shape(self, input_shape):
    """Standard Keras compute_output_shape() method."""
    if isinstance(input_shape, list):
      input_shape = input_shape[0]
    if self.units == 1:
      return tuple(input_shape[:-1]) + (1,)
    else:
      # Second to last dimension must be equal to 'units'. Nothing to append.
      return input_shape[:-1]

  def get_config(self):
    """Standard Keras config for serialization."""
    config = {
        "lattice_sizes": self.lattice_sizes,
        "units": self.units,
        "monotonicities": self.monotonicities,
        "unimodalities": self.unimodalities,
        "edgeworth_trusts": self.edgeworth_trusts,
        "trapezoid_trusts": self.trapezoid_trusts,
        "monotonic_dominances": self.monotonic_dominances,
        "range_dominances": self.range_dominances,
        "joint_monotonicities": self.joint_monotonicities,
        "joint_unimodalities": self.joint_unimodalities,
        "output_min": self.output_min,
        "output_max": self.output_max,
        "num_projection_iterations": self.num_projection_iterations,
        "monotonic_at_every_step": self.monotonic_at_every_step,
        "clip_inputs": self.clip_inputs,
        "interpolation": self.interpolation,
        "kernel_initializer":
            keras.initializers.serialize(
                self.kernel_initializer, use_legacy_format=True),
        "kernel_regularizer":
            [keras.regularizers.serialize(r, use_legacy_format=True)
             for r in self.kernel_regularizer],
    }  # pyformat: disable
    config.update(super(Lattice, self).get_config())
    return config

  def finalize_constraints(self):
    """Ensures layers weights strictly satisfy constraints.

    Applies approximate projection to strictly satisfy specified constraints.
    If `monotonic_at_every_step == True` there is no need to call this function.

    Returns:
      In eager mode directly updates weights and returns variable which stores
      them. In graph mode returns `assign_add` op which has to be executed to
      updates weights.
    """
    return self.kernel.assign_add(
        self._final_constraints(self.kernel) - self.kernel)

  def assert_constraints(self, eps=1e-6):
    """Asserts that weights satisfy all constraints.

    In graph mode builds and returns list of assertion ops.
    In eager mode directly executes assertions.

    Args:
      eps: allowed constraints violation.

    Returns:
      List of assertion ops in graph mode or immediately asserts in eager mode.
    """
    return lattice_lib.assert_constraints(
        weights=self.kernel,
        lattice_sizes=self.lattice_sizes,
        monotonicities=utils.canonicalize_monotonicities(
            self.monotonicities, allow_decreasing=False),
        edgeworth_trusts=utils.canonicalize_trust(self.edgeworth_trusts),
        trapezoid_trusts=utils.canonicalize_trust(self.trapezoid_trusts),
        monotonic_dominances=self.monotonic_dominances,
        range_dominances=self.range_dominances,
        joint_monotonicities=self.joint_monotonicities,
        joint_unimodalities=self.joint_unimodalities,
        output_min=self.output_min,
        output_max=self.output_max,
        eps=eps)


def create_kernel_initializer(kernel_initializer_id,
                              lattice_sizes,
                              monotonicities,
                              output_min,
                              output_max,
                              unimodalities,
                              joint_unimodalities,
                              init_min=None,
                              init_max=None):
  """Returns a kernel Keras initializer object from its id.

  This function is used to convert the 'kernel_initializer' parameter in the
  constructor of tfl.Lattice into the corresponding initializer object.

  Args:
    kernel_initializer_id: See the documentation of the 'kernel_initializer'
      parameter in the constructor of tfl.Lattice.
    lattice_sizes: See the documentation of the same parameter in the
      constructor of tfl.Lattice.
    monotonicities: See the documentation of the same parameter in the
      constructor of tfl.Lattice.
    output_min: See the documentation of the same parameter in the constructor
      of tfl.Lattice.
    output_max: See the documentation of the same parameter in the constructor
      of tfl.Lattice.
    unimodalities: See the documentation of the same parameter in the
      constructor of tfl.Lattice.
    joint_unimodalities: See the documentation of the same parameter in the
      constructor of tfl.Lattice.
    init_min: None or lower bound of kernel initialization. If set, init_max
      must also be set.
    init_max: None or upper bound of kernel initialization. If set, init_min
      must also be set.

  Returns:
    The Keras initializer object for the tfl.Lattice kernel variable.

  Raises:
    ValueError: If only one of init_{min/max} is set.
  """
  if ((init_min is not None and init_max is None) or
      (init_min is None and init_max is not None)):
    raise ValueError("Both or neither of init_{min/max} must be set")

  def do_joint_unimodalities_contain_all_features(joint_unimodalities):
    if (joint_unimodalities is None) or (len(joint_unimodalities) != 1):
      return False
    [joint_unimodalities] = joint_unimodalities
    return set(joint_unimodalities[0]) == set(range(len(lattice_sizes)))

  # Initialize joint unimodalities identical to regular ones.
  all_unimodalities = [0] * len(lattice_sizes)
  if unimodalities:
    for i, value in enumerate(unimodalities):
      if value:
        all_unimodalities[i] = value
  if joint_unimodalities:
    for dimensions, direction in joint_unimodalities:
      for dim in dimensions:
        all_unimodalities[dim] = direction

  if kernel_initializer_id in ["linear_initializer", "LinearInitializer"]:
    if init_min is None and init_max is None:
      init_min, init_max = lattice_lib.default_init_params(
          output_min, output_max)

    return LinearInitializer(
        lattice_sizes=lattice_sizes,
        monotonicities=monotonicities,
        output_min=init_min,
        output_max=init_max,
        unimodalities=all_unimodalities)
  elif kernel_initializer_id in [
      "random_monotonic_initializer", "RandomMonotonicInitializer"
  ]:
    if init_min is None and init_max is None:
      init_min, init_max = lattice_lib.default_init_params(
          output_min, output_max)

    return RandomMonotonicInitializer(
        lattice_sizes=lattice_sizes,
        output_min=init_min,
        output_max=init_max,
        unimodalities=all_unimodalities)
  elif kernel_initializer_id in [
      "random_uniform_or_linear_initializer", "RandomUniformOrLinearInitializer"
  ]:
    if do_joint_unimodalities_contain_all_features(joint_unimodalities):
      return create_kernel_initializer("random_uniform", lattice_sizes,
                                       monotonicities, output_min, output_max,
                                       unimodalities, joint_unimodalities,
                                       init_min, init_max)
    return create_kernel_initializer("linear_initializer", lattice_sizes,
                                     monotonicities, output_min, output_max,
                                     unimodalities, joint_unimodalities,
                                     init_min, init_max)
  else:
    # This is needed for Keras deserialization logic to be aware of our custom
    # objects.
    with keras.utils.custom_object_scope({
        "LinearInitializer": LinearInitializer,
        "RandomMonotonicInitializer": RandomMonotonicInitializer,
    }):
      return keras.initializers.get(kernel_initializer_id)


class LinearInitializer(keras.initializers.Initializer):
  # pyformat: disable
  """Initializes a `tfl.layers.Lattice` as linear function.

  - The linear function will have positive coefficients for monotonic dimensions
    and 0 otherwise. If all dimensions are unconstrained, all coefficients will
    be positive.
  - Linear coefficients are set such that the minimum/maximum output of the
    lattice matches the given output_min/output_max.
  - Each monotonic dimension contributes with same weight regardless of number
    of vertices per dimension.
  - No dimension can be both monotonic and unimodal.
  - Unimodal dimensions contribute with same weight as monotonic dimensions.
  - Unimodal dimensions linearly decrease for first `(dim_size + 1) // 2`
    vertices and then linearly increase for following vertices.

  Attributes:
    - All `__init__` arguments.
  """
  # pyformat: enable

  def __init__(self,
               lattice_sizes,
               monotonicities,
               output_min,
               output_max,
               unimodalities=None):
    """Initializes an instance of `LinearInitializer`.

    Args:
      lattice_sizes: Lattice sizes of `tfl.layers.Lattice` to initialize.
      monotonicities: Monotonic dimensions for initialization. Does not need to
        match `monotonicities` of `tfl.layers.Lattice`.
      output_min: Minimum layer output after initialization.
      output_max: Maximum layer output after initialization.
      unimodalities: None or unimodal dimensions after initialization. Does not
        need to match `unimodalities` of `tfl.layers.Lattice`.

    Raises:
      ValueError: If there is a mismatch between `monotonicities` and
      `lattice_sizes`.
    """
    lattice_lib.verify_hyperparameters(
        lattice_sizes=lattice_sizes,
        monotonicities=monotonicities,
        unimodalities=unimodalities,
        output_min=output_min,
        output_max=output_max)

    self.lattice_sizes = lattice_sizes
    self.monotonicities = monotonicities
    self.output_min = output_min
    self.output_max = output_max
    self.unimodalities = unimodalities

  def __call__(self, shape, dtype=None, partition_info=None):
    """Returns weights of `tfl.layers.Lattice` layer.

    Args:
      shape: Must be: `(prod(lattice_sizes), units)`.
      dtype: Standard Keras initializer param.
      partition_info: Standard Keras initializer param. Not used.
    """
    # TODO: figure out whether it should be used.
    del partition_info
    return lattice_lib.linear_initializer(
        lattice_sizes=self.lattice_sizes,
        monotonicities=utils.canonicalize_monotonicities(
            self.monotonicities, allow_decreasing=False),
        unimodalities=utils.canonicalize_unimodalities(self.unimodalities),
        output_min=self.output_min,
        output_max=self.output_max,
        units=shape[1],
        dtype=dtype)

  def get_config(self):
    """Standard Keras config for serialization."""
    config = {
        "lattice_sizes": self.lattice_sizes,
        "monotonicities": self.monotonicities,
        "output_min": self.output_min,
        "output_max": self.output_max,
        "unimodalities": self.unimodalities,
    }  # pyformat: disable
    return config


class RandomMonotonicInitializer(keras.initializers.Initializer):
  # pyformat: disable
  """Initializes a `tfl.layers.Lattice` as uniform random monotonic function.

  - The uniform random monotonic function will initilaize the lattice parameters
    uniformly at random and make it such that the parameters are monotonically
    increasing for each input.
  - The random parameters will be sampled from `[output_min, output_max]`

  Attributes:
    - All `__init__` arguments.
  """
  # pyformat: enable

  def __init__(self, lattice_sizes, output_min, output_max, unimodalities=None):
    """Initializes an instance of `RandomMonotonicInitializer`.

    Args:
      lattice_sizes: Lattice sizes of `tfl.layers.Lattice` to initialize.
      output_min: Minimum layer output after initialization.
      output_max: Maximum layer output after initialization.
      unimodalities: None or unimodal dimensions after initialization. Does not
        need to match `unimodalities` of `tfl.layers.Lattice`.

    Raises:
      ValueError: If there are invalid hyperparameters.
    """
    lattice_lib.verify_hyperparameters(
        lattice_sizes=lattice_sizes,
        unimodalities=unimodalities,
        output_min=output_min,
        output_max=output_max)

    self.lattice_sizes = lattice_sizes
    self.output_min = output_min
    self.output_max = output_max
    self.unimodalities = unimodalities

  def __call__(self, shape, dtype=None, partition_info=None):
    """Returns weights of `tfl.layers.Lattice` layer.

    Args:
      shape: Must be: `(prod(lattice_sizes), units)`.
      dtype: Standard Keras initializer param.
      partition_info: Standard Keras initializer param. Not used.
    """
    del partition_info
    return lattice_lib.random_monotonic_initializer(
        lattice_sizes=self.lattice_sizes,
        output_min=self.output_min,
        output_max=self.output_max,
        units=shape[1],
        dtype=dtype)

  def get_config(self):
    """Standard Keras config for serialization."""
    config = {
        "lattice_sizes": self.lattice_sizes,
        "output_min": self.output_min,
        "output_max": self.output_max,
        "unimodalities": self.unimodalities,
    }  # pyformat: disable
    return config


class LatticeConstraints(keras.constraints.Constraint):
  # pyformat: disable
  """Constraints for `tfl.layers.Lattice` layer.

  Applies all constraints to the lattice weights. See `tfl.layers.Lattice`
  for more details.

  Attributes:
    - All `__init__` arguments.
  """
  # pyformat: enable

  def __init__(self,
               lattice_sizes,
               monotonicities=None,
               unimodalities=None,
               edgeworth_trusts=None,
               trapezoid_trusts=None,
               monotonic_dominances=None,
               range_dominances=None,
               joint_monotonicities=None,
               joint_unimodalities=None,
               output_min=None,
               output_max=None,
               num_projection_iterations=1,
               enforce_strict_monotonicity=True):
    """Initializes an instance of `LatticeConstraints`.

    Args:
      lattice_sizes: Lattice sizes of `Lattice` layer to constraint.
      monotonicities: Same meaning as corresponding parameter of `Lattice`.
      unimodalities: Same meaning as corresponding parameter of `Lattice`.
      edgeworth_trusts: Same meaning as corresponding parameter of `Lattice`.
      trapezoid_trusts: Same meaning as corresponding parameter of `Lattice`.
      monotonic_dominances: Same meaning as corresponding parameter of
        `Lattice`.
      range_dominances: Same meaning as corresponding parameter of `Lattice`.
      joint_monotonicities: Same meaning as corresponding parameter of
        `Lattice`.
      joint_unimodalities: Same meaning as corresponding parameter of `Lattice`.
      output_min: Minimum possible output.
      output_max: Maximum possible output.
      num_projection_iterations: Same meaning as corresponding parameter of
        `Lattice`.
      enforce_strict_monotonicity: Whether to use approximate projection to
        ensure that constratins are strictly satisfied.

    Raises:
      ValueError: If weights to project don't correspond to `lattice_sizes`.
    """
    lattice_lib.verify_hyperparameters(
        lattice_sizes=lattice_sizes,
        monotonicities=monotonicities,
        unimodalities=unimodalities,
        edgeworth_trusts=edgeworth_trusts,
        trapezoid_trusts=trapezoid_trusts,
        monotonic_dominances=monotonic_dominances,
        range_dominances=range_dominances,
        joint_monotonicities=joint_monotonicities,
        joint_unimodalities=joint_unimodalities)

    self.lattice_sizes = lattice_sizes
    self.monotonicities = utils.canonicalize_monotonicities(
        monotonicities, allow_decreasing=False)
    self.unimodalities = utils.canonicalize_unimodalities(unimodalities)
    self.edgeworth_trusts = utils.canonicalize_trust(edgeworth_trusts)
    self.trapezoid_trusts = utils.canonicalize_trust(trapezoid_trusts)
    self.monotonic_dominances = monotonic_dominances
    self.range_dominances = range_dominances
    self.joint_monotonicities = joint_monotonicities
    self.joint_unimodalities = joint_unimodalities
    self.output_min = output_min
    self.output_max = output_max
    self.num_projection_iterations = num_projection_iterations
    self.enforce_strict_monotonicity = enforce_strict_monotonicity
    self.num_constraint_dims = utils.count_non_zeros(self.monotonicities,
                                                     self.unimodalities)

  def __call__(self, w):
    """Applies constraints to `w`."""
    # No need to separately check for trust constraints and monotonic dominance,
    # since monotonicity is required to impose them. The only exception is joint
    # monotonicity.
    if (self.num_constraint_dims > 0 or self.joint_monotonicities or
        self.joint_unimodalities):
      w = lattice_lib.project_by_dykstra(
          w,
          lattice_sizes=self.lattice_sizes,
          monotonicities=self.monotonicities,
          unimodalities=self.unimodalities,
          edgeworth_trusts=self.edgeworth_trusts,
          trapezoid_trusts=self.trapezoid_trusts,
          monotonic_dominances=self.monotonic_dominances,
          range_dominances=self.range_dominances,
          joint_monotonicities=self.joint_monotonicities,
          joint_unimodalities=self.joint_unimodalities,
          num_iterations=self.num_projection_iterations)
      if self.enforce_strict_monotonicity:
        w = lattice_lib.finalize_constraints(
            w,
            lattice_sizes=self.lattice_sizes,
            monotonicities=self.monotonicities,
            edgeworth_trusts=self.edgeworth_trusts,
            trapezoid_trusts=self.trapezoid_trusts,
            output_min=self.output_min,
            output_max=self.output_max)
    # TODO: come up with a better solution than separately applying
    # bounds again after other projections.
    if self.output_min is not None:
      w = tf.maximum(w, self.output_min)
    if self.output_max is not None:
      w = tf.minimum(w, self.output_max)
    return w

  def get_config(self):
    """Standard Keras config for serialization."""
    return {
        "lattice_sizes": self.lattice_sizes,
        "monotonicities": self.monotonicities,
        "unimodalities": self.unimodalities,
        "edgeworth_trusts": self.edgeworth_trusts,
        "trapezoid_trusts": self.trapezoid_trusts,
        "monotonic_dominances": self.monotonic_dominances,
        "range_dominances": self.range_dominances,
        "joint_monotonicities": self.joint_monotonicities,
        "joint_unimodalities": self.joint_unimodalities,
        "output_min": self.output_min,
        "output_max": self.output_max,
        "num_projection_iterations": self.num_projection_iterations,
        "enforce_strict_monotonicity": self.enforce_strict_monotonicity
    }  # pyformat: disable


class TorsionRegularizer(keras.regularizers.Regularizer):
  # pyformat: disable
  """Torsion regularizer for `tfl.layers.Lattice` layer.

  Lattice torsion regularizer penalizes how much the lattice function twists
  from side-to-side (see
  [publication](http://jmlr.org/papers/v17/15-243.html)).

  Consider a 3 x 2 lattice with weights `w`:

  ```
  w[3]-----w[4]-----w[5]
    |        |        |
    |        |        |
  w[0]-----w[1]-----w[2]
  ```

  In this case, the torsion regularizer is defined as:

  ```
  l1 * (|w[4] + w[0] - w[3] - w[1]| + |w[5] + w[1] - w[4] - w[2]|) +
  l2 * ((w[4] + w[0] - w[3] - w[1])^2 + (w[5] + w[1] - w[4] - w[2])^2)
  ```

  Attributes:
    - All `__init__` arguments.
  """
  # pyformat: enable

  def __init__(self, lattice_sizes, l1=0.0, l2=0.0):
    """Initializes an instance of `TorsionRegularizer`.

    Args:
      lattice_sizes: Lattice sizes of `tfl.layers.Lattice` to regularize.
      l1: l1 regularization amount. Either single float or list or tuple of
        floats to specify different regularization amount per dimension. The
        amount of regularization for the interaction term between two dimensions
        is the product of the corresponding per dimension amounts.
      l2: l2 regularization amount. Either single float or list or tuple of
        floats to specify different regularization amount per dimension. The
        amount of regularization for the interaction term between two dimensions
        is the product of the corresponding per dimension amounts.
    """
    self.lattice_sizes = lattice_sizes
    self.l1 = l1
    self.l2 = l2

  def __call__(self, x):
    """Returns regularization loss for `x`."""
    lattice_lib.verify_hyperparameters(
        lattice_sizes=self.lattice_sizes, weights_shape=x.shape)
    return lattice_lib.torsion_regularizer(x, self.lattice_sizes, self.l1,
                                           self.l2)

  def get_config(self):
    """Standard Keras config for serialization."""
    return {
        "lattice_sizes": self.lattice_sizes,
        "l1": self.l1,
        "l2": self.l2,
    }  # pyformat: disable


class LaplacianRegularizer(keras.regularizers.Regularizer):
  # pyformat: disable
  """Laplacian regularizer for `tfl.layers.Lattice` layer.

  Laplacian regularizer penalizes the difference between adjacent vertices in
  multi-cell lattice (see
  [publication](http://jmlr.org/papers/v17/15-243.html)).

  Consider a 3 x 2 lattice with weights `w`:

  ```
  w[3]-----w[4]-----w[5]
    |        |        |
    |        |        |
  w[0]-----w[1]-----w[2]
  ```

  where the number at each node represents the weight index.
  In this case, the laplacian regularizer is defined as:

  ```
  l1[0] * (|w[1] - w[0]| + |w[2] - w[1]| +
           |w[4] - w[3]| + |w[5] - w[4]|) +
  l1[1] * (|w[3] - w[0]| + |w[4] - w[1]| + |w[5] - w[2]|) +

  l2[0] * ((w[1] - w[0])^2 + (w[2] - w[1])^2 +
           (w[4] - w[3])^2 + (w[5] - w[4])^2) +
  l2[1] * ((w[3] - w[0])^2 + (w[4] - w[1])^2 + (w[5] - w[2])^2)
  ```

  Attributes:
    - All `__init__` arguments.
  """
  # pyformat: enable

  def __init__(self, lattice_sizes, l1=0.0, l2=0.0):
    """Initializes an instance of `LaplacianRegularizer`.

    Args:
      lattice_sizes: Lattice sizes of `tfl.layers.Lattice` to regularize.
      l1: l1 regularization amount. Either single float or list or tuple of
        floats to specify different regularization amount per dimension.
      l2: l2 regularization amount. Either single float or list or tuple of
        floats to specify different regularization amount per dimension.

    Raises:
      ValueError: If provided input does not correspond to `lattice_sizes`.
    """
    lattice_lib.verify_hyperparameters(
        lattice_sizes=lattice_sizes,
        regularization_amount=l1,
        regularization_info="l1")
    lattice_lib.verify_hyperparameters(
        lattice_sizes=lattice_sizes,
        regularization_amount=l2,
        regularization_info="l2")
    self.lattice_sizes = lattice_sizes
    self.l1 = l1
    self.l2 = l2

  def __call__(self, x):
    """Returns regularization loss for `x`."""
    lattice_lib.verify_hyperparameters(
        lattice_sizes=self.lattice_sizes, weights_shape=x.shape)
    return lattice_lib.laplacian_regularizer(x, self.lattice_sizes, self.l1,
                                             self.l2)

  def get_config(self):
    """Standard Keras config for serialization."""
    return {
        "lattice_sizes": self.lattice_sizes,
        "l1": self.l1,
        "l2": self.l2
    }  # pyformat: disable
