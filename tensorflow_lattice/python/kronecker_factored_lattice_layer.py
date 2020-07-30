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
"""Kronecker-Factored Lattice layer with monotonicity constraints.

Keras implementation of tensorflow Kronecker-Factored Lattice layer. This layer
takes one or more d-dimensional input(s) and combines them using a
Kronecker-Factored Lattice function, satisfying monotonicity constraints if
specified.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import kronecker_factored_lattice_lib as kfl_lib
from . import utils
from tensorflow import keras

KFL_SCALE_NAME = "kronecker_factored_lattice_scale"
KFL_BIAS_NAME = "kronecker_factored_lattice_bias"
KFL_KERNEL_NAME = "kronecker_factored_lattice_kernel"


# TODO: add support for different lattice_sizes for each input
# dimension.
class KroneckerFactoredLattice(keras.layers.Layer):
  # pyformat: disable
  """Kronecker-Factored Lattice layer.

  A Kronecker-Factored Lattice is a reparameterization of a Lattice using
  kronecker-facotrization, which gives us linear time and space complexity.
  While the underlying representation is different, the input-output behavior
  remains the same.

  A Kronecker-Factored Lattice consists of 'units' lattices. Each unit computes
  the function described below on a distinct 'dims'-dimensional vector x taken
  from the input tensor. Each unit has its own set of parameters. The function
  each unit computes is given by:

  f(x) = b + (1/num_terms) * sum_{t=1}^{num_terms} scale_t * prod_{d=1}^{dims} PLF(x[d];w[d])

  where bias and each scale_t are scalar parameters, w[d] is a
  'lattice_size'-dimensional vector of parameters, and  PLF(;w) denotes the
  one-dimensional piecewise-linear function with domain [0, lattice_sizes-1]
  whose graph consists of lattice_sizes-1 linear segments interpolating the
  points (i, w[i]), for i=0,1,...,lattice_size-1.

  There is currently one type of constraint on the shape of the learned
  function.

  * **Monotonicity:** constrains the function to be increasing in the
    corresponding dimension.

  Input shape:
    - if `units == 1`: tensor of shape: `(batch_size, ..., dims)`
      or list of `dims` tensors of same shape: `(batch_size, ..., 1)`
    - if `units > 1`: tensor of shape: `(batch_size, ..., units, dims)` or list
      of `dims` tensors of same shape: `(batch_size, ..., units, 1)`

    A typical shape is: `(batch_size, len(monotonicities))`

  Output shape:
    Tensor of shape: `(batch_size, ..., units)`

  Attributes:
    - All `__init__` arguments.
    scale: A tensor of shape `(units, num_terms)`. Contains the `scale_t`
      parameter for each unit for each term.
    bias: A tensor of shape `(units)`. Contains the `b` parameter for each unit.
    kernel: The `w` weights parameter of the Kronecker-Factored Lattice of
      shape: `(1, lattice_sizes, units * dims, num_terms)`. Note that the kernel
      is unit-major in its second to last dimension.

  Example:

  ```python
  kfl = tfl.layers.KroneckerFactoredLattice(
      # Number of vertices along each dimension.
      lattice_sizes=2,
      # Number of output units.
      units=2,
      # Number of independently trained submodels per unit, the outputs
      # of which are averaged to get the final output.
      num_terms=4,
      # You can specify monotonicity constraints.
      monotonicities=['increasing', 'none', 'increasing', 'increasing',
                      'increasing', 'increasing', 'increasing'])
  ```
  """
  # pyformat: enable

  def __init__(self,
               lattice_sizes,
               units=1,
               num_terms=2,
               monotonicities=None,
               clip_inputs=True,
               satisfy_constraints_at_every_step=True,
               kernel_initializer="random_monotonic_initializer",
               **kwargs):
    # pyformat: disable
    """Initializes an instance of `KroneckerFactoredLattice`.

    Args:
      lattice_sizes: Number of vertices per dimension (minimum is 2).
      units: Output dimension of the layer. See class comments for details.
      num_terms: Number of independently trained submodels per unit, the outputs
        of which are averaged to get the final output.
      monotonicities: None or list or tuple of same length as input dimension of
        {'none', 'increasing', 0, 1} which specifies if the model output should
        be monotonic in the corresponding feature, using 'increasing' or 1 to
        indicate increasing monotonicity and 'none' or 0 to indicate no
        monotonicity constraints.
      clip_inputs: If inputs should be clipped to the input range of the
        Kronecker-Factored Lattice.
      satisfy_constraints_at_every_step: Whether to strictly enforce constraints
        after every gradient update by applying an imprecise projection.
      kernel_initializer: None or one of:
        - `'random_monotonic_initializer'`: initializes parameters as uniform
          random functions that are monotonic in monotonic dimensions.
        - Any Keras initializer object.
      **kwargs: Other args passed to `tf.keras.layers.Layer` initializer.

    Raises:
      ValueError: If layer hyperparameters are invalid.
    """
    # pyformat: enable
    kfl_lib.verify_hyperparameters(
        lattice_sizes=lattice_sizes, units=units, num_terms=num_terms)
    super(KroneckerFactoredLattice, self).__init__(**kwargs)

    self.lattice_sizes = lattice_sizes
    self.units = units
    self.num_terms = num_terms
    self.monotonicities = monotonicities
    self.clip_inputs = clip_inputs
    self.satisfy_constraints_at_every_step = satisfy_constraints_at_every_step

    self.kernel_initializer = create_kernel_initializer(
        kernel_initializer_id=kernel_initializer,
        monotonicities=self.monotonicities)

  def build(self, input_shape):
    """Standard Keras build() method."""
    kfl_lib.verify_hyperparameters(
        units=self.units,
        input_shape=input_shape,
        monotonicities=self.monotonicities)
    # input_shape: (batch, ..., units, dims)
    if isinstance(input_shape, list):
      dims = len(input_shape)
    else:
      dims = input_shape.as_list()[-1]

    self.scale = self.add_weight(
        KFL_SCALE_NAME,
        shape=[self.units, self.num_terms],
        initializer=ScaleInitializer(),
        dtype=self.dtype)
    self.bias = self.add_weight(
        KFL_BIAS_NAME,
        shape=[self.units],
        initializer="zeros",
        dtype=self.dtype)

    if self.monotonicities:
      constraints = KroneckerFactoredLatticeConstraints(
          units=self.units,
          scale=self.scale,
          monotonicities=self.monotonicities,
          satisfy_constraints_at_every_step=self
          .satisfy_constraints_at_every_step)
    else:
      constraints = None

    # Note that the first dimension of shape is 1 to work with
    # tf.nn.depthwise_conv2d.
    self.kernel = self.add_weight(
        KFL_KERNEL_NAME,
        shape=[1, self.lattice_sizes, self.units * dims, self.num_terms],
        initializer=self.kernel_initializer,
        constraint=constraints,
        dtype=self.dtype)

    self._final_constraints = KroneckerFactoredLatticeConstraints(
        units=self.units,
        scale=self.scale,
        monotonicities=self.monotonicities,
        satisfy_constraints_at_every_step=True)

    super(KroneckerFactoredLattice, self).build(input_shape)

  def call(self, inputs):
    """Standard Keras call() method."""
    return kfl_lib.evaluate_with_hypercube_interpolation(
        inputs=inputs,
        scale=self.scale,
        bias=self.bias,
        kernel=self.kernel,
        units=self.units,
        num_terms=self.num_terms,
        lattice_sizes=self.lattice_sizes,
        clip_inputs=self.clip_inputs)

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
        "num_terms": self.num_terms,
        "monotonicities": self.monotonicities,
        "clip_inputs": self.clip_inputs,
        "satisfy_constraints_at_every_step":
            self.satisfy_constraints_at_every_step,
        "kernel_initializer":
            keras.initializers.serialize(self.kernel_initializer),
    }  # pyformat: disable
    config.update(super(KroneckerFactoredLattice, self).get_config())
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
    return kfl_lib.assert_constraints(
        weights=self.kernel,
        units=self.units,
        scale=self.scale,
        monotonicities=utils.canonicalize_monotonicities(
            self.monotonicities, allow_decreasing=False),
        eps=eps)


def create_kernel_initializer(kernel_initializer_id, monotonicities):
  """Returns a kernel Keras initializer object from its id.

  This function is used to convert the 'kernel_initializer' parameter in the
  constructor of tfl.layers.KroneckerFactoredLattice into the corresponding
  initializer object.

  Args:
    kernel_initializer_id: See the documentation of the 'kernel_initializer'
      parameter in the constructor of `tfl.layers.KroneckerFactoredLattice`.
    monotonicities: See the documentation of the same parameter in the
      constructor of `tfl.layers.KroneckerFactoredLattice`.

  Returns:
    The Keras initializer object for the `tfl.layers.KroneckerFactoredLattice`
    kernel variable.
  """
  # Construct initializer.
  if kernel_initializer_id in [
      "random_monotonic_initializer", "RandomMonotonicInitializer"
  ]:
    return RandomMonotonicInitializer(monotonicities)
  else:
    # This is needed for Keras deserialization logic to be aware of our custom
    # objects.
    with keras.utils.custom_object_scope({
        "RandomMonotonicInitializer": RandomMonotonicInitializer,
    }):
      return keras.initializers.get(kernel_initializer_id)


class RandomMonotonicInitializer(keras.initializers.Initializer):
  # pyformat: disable
  """Initializes a `tfl.layers.KroneckerFactoredLattice` as random monotonic."""
  # pyformat: enable

  def __init__(self, monotonicities, seed=None):
    """Initializes an instance of `RandomMonotonicInitializer`.

    Args:
      monotonicities: Monotonic dimensions for initialization. Does not need to
        match `monotonicities` of `tfl.layers.KroneckerFactoredLattice`.
      seed: A Python integer. Used to create a random seed for the distribution.
    """
    self.monotonicities = monotonicities
    self.seed = seed

  def __call__(self, shape, dtype=None, partition_info=None):
    """Returns weights of `tfl.layers.KroneckerFactoredLattice` layer.

    Args:
      shape: Must be: `(1, lattice_sizes, units * dims, num_terms)`.
      dtype: Standard Keras initializer param.
      partition_info: Standard Keras initializer param. Not used.
    """
    del partition_info
    return kfl_lib.random_monotonic_initializer(
        shape=shape,
        monotonicities=utils.canonicalize_monotonicities(
            self.monotonicities, allow_decreasing=False),
        dtype=dtype,
        seed=self.seed)

  def get_config(self):
    """Standard Keras config for serializaion."""
    config = {
        "monotonicities": self.monotonicities,
        "seed": self.seed,
    }  # pyformat: disable
    return config


class ScaleInitializer(keras.initializers.Initializer):
  # pyformat: disable
  """Initializes scale to alternate between 1 and -1 for each term."""
  # pyformat: enable

  def __call__(self, shape, dtype=None, partition_info=None):
    """Returns weights of `tfl.layers.KroneckerFactoredLattice` scale.

    Args:
      shape: Must be: `(units, num_terms)`.
      dtype: Standard Keras initializer param.
      partition_info: Standard Keras initializer param. Not used.
    """
    del partition_info
    units, num_terms = shape
    return kfl_lib.scale_initializer(units=units, num_terms=num_terms)


class KroneckerFactoredLatticeConstraints(keras.constraints.Constraint):
  # pyformat: disable
  """Constraints for `tfl.layers.KroneckerFactoredLattice` layer.

  Applies all constraints to the Kronecker-Factored Lattice weights. See
  `tfl.layers.KroneckerFactoredLattice` for more details.

  Attributes:
    - All `__init__` arguments.
  """
  # pyformat: enable

  def __init__(self,
               units,
               scale,
               monotonicities=None,
               satisfy_constraints_at_every_step=True):
    """Initializes an instance of `KroneckerFactoredLatticeConstraints`.

    Args:
      units: Same meaning as corresponding parameter of
        `KroneckerFactoredLattice`.
      scale: Scale variable of shape: `(units, num_terms)`.
      monotonicities: Same meaning as corresponding parameter of
        `KroneckerFactoredLattice`.
      satisfy_constraints_at_every_step: Whether to use approximate projection
        to ensure that constratins are strictly satisfied.
    """
    self.units = units
    self.scale = scale
    self.monotonicities = utils.canonicalize_monotonicities(
        monotonicities, allow_decreasing=False)
    self.num_constraint_dims = utils.count_non_zeros(self.monotonicities)
    self.satisfy_constraints_at_every_step = satisfy_constraints_at_every_step

  def __call__(self, w):
    """Applies constraints to `w`.

    Args:
      w: Kronecker-Factored Lattice weights tensor of shape: `(1, lattice_sizes,
        units * dims, num_terms)`.

    Returns:
      Constrained and projected w.
    """
    if self.num_constraint_dims and self.satisfy_constraints_at_every_step:
      w = kfl_lib.finalize_constraints(
          w,
          units=self.units,
          scale=self.scale,
          monotonicities=self.monotonicities)
    return w

  def get_config(self):
    """Standard Keras config for serialization."""
    return {
        "units": self.units,
        "scale": self.scale,
        "monotonicities": self.monotonicities,
        "satisfy_constraints_at_every_step":
            self.satisfy_constraints_at_every_step,
    }  # pyformat: disable
