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

import functools
import inspect

import tensorflow as tf
# pylint: disable=g-import-not-at-top
# Use Keras 2.
version_fn = getattr(tf.keras, "version", None)
if version_fn and version_fn().startswith("3."):
  import tf_keras as keras
else:
  keras = tf.keras
from . import kronecker_factored_lattice_lib as kfl_lib
from . import utils

DIMS_NAME = "dims"
KFL_SCALE_NAME = "kronecker_factored_lattice_scale"
KFL_BIAS_NAME = "kronecker_factored_lattice_bias"
KFL_KERNEL_NAME = "kronecker_factored_lattice_kernel"
LATTICE_SIZES_NAME = "lattice_sizes"
NUM_TERMS_NAME = "num_terms"
UNITS_NAME = "units"


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
    corresponding dimension. To achieve decreasing monotonicity, either pass the
    inputs through a `tfl.layers.PWLCalibration` with `decreasing` monotonicity,
    or manually reverse the inputs as `lattice_size - 1 - inputs`.

  There are upper and lower bound constraints on the output.

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
               output_min=None,
               output_max=None,
               clip_inputs=True,
               kernel_initializer="kfl_random_monotonic_initializer",
               scale_initializer="scale_initializer",
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
      output_min: None or lower bound of the output.
      output_max: None or upper bound of the output.
      clip_inputs: If inputs should be clipped to the input range of the
        Kronecker-Factored Lattice.
      kernel_initializer: None or one of:
        - `'kfl_random_monotonic_initializer'`: initializes parameters as uniform
          random functions that are monotonic in monotonic dimensions.
        - Any Keras initializer object.
      scale_initializer: None or one of:
        - `'scale_initializer'`: Initializes scale depending on output_min and
          output_max. If both output_min and output_max are set, scale is
          initialized to half their difference, alternating signs for each term.
          If only output_min is set, scale is initialized to 1 for each term. If
          only output_max is set, scale is initialized to -1 for each term.
          Otherwise scale is initialized to alternate between 1 and -1 for each
          term.
      **kwargs: Other args passed to `keras.layers.Layer` initializer.

    Raises:
      ValueError: If layer hyperparameters are invalid.
    """
    # pyformat: enable
    kfl_lib.verify_hyperparameters(
        lattice_sizes=lattice_sizes,
        units=units,
        num_terms=num_terms,
        output_min=output_min,
        output_max=output_max)
    super(KroneckerFactoredLattice, self).__init__(**kwargs)

    self.lattice_sizes = lattice_sizes
    self.units = units
    self.num_terms = num_terms
    self.monotonicities = monotonicities
    self.output_min = output_min
    self.output_max = output_max
    self.clip_inputs = clip_inputs

    self.kernel_initializer = create_kernel_initializer(
        kernel_initializer_id=kernel_initializer,
        monotonicities=self.monotonicities,
        output_min=self.output_min,
        output_max=self.output_max)

    self.scale_initializer = create_scale_initializer(
        scale_initializer_id=scale_initializer,
        output_min=self.output_min,
        output_max=self.output_max)

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

    if self.output_min is not None or self.output_max is not None:
      scale_constraints = ScaleConstraints(
          output_min=self.output_min, output_max=self.output_max)
    else:
      scale_constraints = None
    self.scale = self.add_weight(
        KFL_SCALE_NAME,
        shape=[self.units, self.num_terms],
        initializer=self.scale_initializer,
        constraint=scale_constraints,
        dtype=self.dtype)
    self.bias = self.add_weight(
        KFL_BIAS_NAME,
        shape=[self.units],
        initializer=BiasInitializer(self.output_min, self.output_max),
        trainable=(self.output_min is None and self.output_max is None),
        dtype=self.dtype)

    if (self.monotonicities or self.output_min is not None or
        self.output_max is not None):
      constraints = KroneckerFactoredLatticeConstraints(
          units=self.units,
          scale=self.scale,
          monotonicities=self.monotonicities,
          output_min=self.output_min,
          output_max=self.output_max)
    else:
      constraints = None

    # Note that the first dimension of shape is 1 to work with
    # tf.nn.depthwise_conv2d. We also provide scale to the __call__ method
    # of the initializer using partial functions if it accepts scale.
    parameters = inspect.signature(self.kernel_initializer).parameters.keys()
    if "scale" in parameters:
      # initial_value needs the lambda because it is a class property and the
      # second and third arguments to tf.cond should be functions,
      # but read_value is already a function, so the lambda is not needed.
      kernel_initializer = functools.partial(
          self.kernel_initializer,
          scale=tf.cond(
              tf.compat.v1.is_variable_initialized(self.scale),
              self.scale.read_value,
              lambda: self.scale.initial_value))
    else:
      kernel_initializer = self.kernel_initializer
    self.kernel = self.add_weight(
        KFL_KERNEL_NAME,
        shape=[1, self.lattice_sizes, self.units * dims, self.num_terms],
        initializer=kernel_initializer,
        constraint=constraints,
        dtype=self.dtype)

    self._final_kernel_constraints = KroneckerFactoredLatticeConstraints(
        units=self.units,
        scale=self.scale,
        monotonicities=self.monotonicities,
        output_min=self.output_min,
        output_max=self.output_max)

    self._final_scale_constraints = ScaleConstraints(
        output_min=self.output_min, output_max=self.output_max)

    # These tensors are meant for book keeping. Note that this slightly
    # increases the size of the graph.
    self.lattice_sizes_tensor = tf.constant(
        self.lattice_sizes, dtype=tf.int32, name=LATTICE_SIZES_NAME)
    self.units_tensor = tf.constant(
        self.units, dtype=tf.int32, name=UNITS_NAME)
    self.dims_tensor = tf.constant(dims, dtype=tf.int32, name=DIMS_NAME)
    self.num_terms_tensor = tf.constant(
        self.num_terms, dtype=tf.int32, name=NUM_TERMS_NAME)

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
        "output_min": self.output_min,
        "output_max": self.output_max,
        "clip_inputs": self.clip_inputs,
        "kernel_initializer":
            keras.initializers.serialize(
                self.kernel_initializer, use_legacy_format=True),
        "scale_initializer":
            keras.initializers.serialize(
                self.scale_initializer, use_legacy_format=True),
    }  # pyformat: disable
    config.update(super(KroneckerFactoredLattice, self).get_config())
    return config

  # TODO: can we remove this now that we always project at every step?
  def finalize_constraints(self):
    """Ensures layers weights strictly satisfy constraints.

    Applies approximate projection to strictly satisfy specified constraints.

    Returns:
      In eager mode directly updates kernel and scale and returns the variables
      which store them. In graph mode returns a `group` op containing the
      `assign_add` ops which have to be executed to update the kernel and scale.
    """
    finalize_kernel = self.kernel.assign_add(
        self._final_kernel_constraints(self.kernel) - self.kernel)
    finalize_scale = self.scale.assign_add(
        self._final_scale_constraints(self.scale) - self.scale)
    return tf.group([finalize_kernel, finalize_scale])

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
        output_min=self.output_min,
        output_max=self.output_max,
        eps=eps)


def create_kernel_initializer(kernel_initializer_id,
                              monotonicities,
                              output_min,
                              output_max,
                              init_min=None,
                              init_max=None):
  """Returns a kernel Keras initializer object from its id.

  This function is used to convert the 'kernel_initializer' parameter in the
  constructor of tfl.layers.KroneckerFactoredLattice into the corresponding
  initializer object.

  Args:
    kernel_initializer_id: See the documentation of the 'kernel_initializer'
      parameter in the constructor of `tfl.layers.KroneckerFactoredLattice`.
    monotonicities: See the documentation of the same parameter in the
      constructor of `tfl.layers.KroneckerFactoredLattice`.
    output_min: See the documentation of the same parameter in the constructor
      of `tfl.layers.KroneckerFactoredLattice`.
    output_max: See the documentation of the same parameter in the constructor
      of `tfl.layers.KroneckerFactoredLattice`.
    init_min: None or lower bound of kernel initialization. If set, init_max
      must also be set. Ignored if kernel_initializer_id is a Keras object.
    init_max: None or upper bound of kernel initialization. If set, init_min
      must also be set. Ignored if kernel_initializer_id is a Keras object.

  Returns:
    The Keras initializer object for the `tfl.layers.KroneckerFactoredLattice`
    kernel variable.

  Raises:
    ValueError: If only one of init_{min/max} is set.
  """
  if init_min is None and init_max is None:
    init_min, init_max = kfl_lib.default_init_params(output_min, output_max)
  elif init_min is not None and init_max is not None:
    # We have nothing to set here.
    pass
  else:
    raise ValueError("Both or neither of init_{min/max} must be set")

  # Construct initializer.
  if kernel_initializer_id in [
      "kfl_random_monotonic_initializer", "KFLRandomMonotonicInitializer"
  ]:
    return KFLRandomMonotonicInitializer(
        monotonicities=monotonicities, init_min=init_min, init_max=init_max)
  else:
    # This is needed for Keras deserialization logic to be aware of our custom
    # objects.
    with keras.utils.custom_object_scope({
        "KFLRandomMonotonicInitializer": KFLRandomMonotonicInitializer,
    }):
      return keras.initializers.get(kernel_initializer_id)


def create_scale_initializer(scale_initializer_id, output_min, output_max):
  """Returns a scale Keras initializer object from its id.

  This function is used to convert the 'scale_initializer' parameter in the
  constructor of tfl.layers.KroneckerFactoredLattice into the corresponding
  initializer object.

  Args:
    scale_initializer_id: See the documentation of the 'scale_initializer'
      parameter in the constructor of `tfl.layers.KroneckerFactoredLattice`.
    output_min: See the documentation of the same parameter in the constructor
      of `tfl.layers.KroneckerFactoredLattice`.
    output_max: See the documentation of the same parameter in the constructor
      of `tfl.layers.KroneckerFactoredLattice`.

  Returns:
    The Keras initializer object for the `tfl.layers.KroneckerFactoredLattice`
    scale variable.
  """
  # Construct initializer.
  if scale_initializer_id in ["scale_initializer", "ScaleInitializer"]:
    return ScaleInitializer(output_min=output_min, output_max=output_max)
  else:
    # This is needed for Keras deserialization logic to be aware of our custom
    # objects.
    with keras.utils.custom_object_scope({
        "ScaleInitializer": ScaleInitializer,
    }):
      return keras.initializers.get(scale_initializer_id)


class KFLRandomMonotonicInitializer(keras.initializers.Initializer):
  # pyformat: disable
  """Initializes a `tfl.layers.KroneckerFactoredLattice` as random monotonic."""
  # pyformat: enable

  def __init__(self, monotonicities, init_min=0.5, init_max=1.5, seed=None):
    """Initializes an instance of `KFLRandomMonotonicInitializer`.

    Args:
      monotonicities: Monotonic dimensions for initialization. Does not need to
        match `monotonicities` of `tfl.layers.KroneckerFactoredLattice`.
      init_min: The lower bound on the range of initialized weights.
      init_max: The upper bound on the range of initialized weights.
      seed: A Python integer. Used to create a random seed for the distribution.
    """
    self.monotonicities = monotonicities
    self.init_min = init_min
    self.init_max = init_max
    self.seed = seed

  def __call__(self, shape, scale, dtype=None, **kwargs):
    """Returns weights of `tfl.layers.KroneckerFactoredLattice` layer.

    Args:
      shape: Must be: `(1, lattice_sizes, units * dims, num_terms)`.
      scale: Scale variable of shape: `(units, num_terms)`.
      dtype: Standard Keras initializer param.
      **kwargs: Other args passed to `keras.initializers.Initializer` __call__
        method.
    """
    return kfl_lib.kfl_random_monotonic_initializer(
        shape=shape,
        scale=scale,
        monotonicities=utils.canonicalize_monotonicities(
            self.monotonicities, allow_decreasing=False),
        init_min=self.init_min,
        init_max=self.init_max,
        dtype=dtype,
        seed=self.seed)

  def get_config(self):
    """Standard Keras config for serializaion."""
    config = {
        "monotonicities": self.monotonicities,
        "init_min": self.init_min,
        "init_max": self.init_max,
        "seed": self.seed,
    }  # pyformat: disable
    return config


class ScaleInitializer(keras.initializers.Initializer):
  # pyformat: disable
  """Initializes scale depending on output_min and output_max.

  If both output_min and output_max are set, scale is initialized to half their
  difference, alternating signs for each term. If only output_min is set, scale
  is initialized to 1 for each term. If only output_max is set, scale is
  initialized to -1 for each term. Otherwise scale is initialized to alternate
  between 1 and -1 for each term.
  """
  # pyformat: enable

  def __init__(self, output_min, output_max):
    """Initializes an instance of `ScaleInitializer`.

    Args:
      output_min: None or minimum layer output.
      output_max: None or maximum layer output.
    """
    self.output_min = output_min
    self.output_max = output_max

  def __call__(self, shape, dtype=None, **kwargs):
    """Returns weights of `tfl.layers.KroneckerFactoredLattice` scale.

    Args:
      shape: Must be: `(units, num_terms)`.
      dtype: Standard Keras initializer param.
      **kwargs: Other args passed to `keras.initializers.Initializer` __call__
        method.
    """
    units, num_terms = shape
    return kfl_lib.scale_initializer(
        units=units,
        num_terms=num_terms,
        output_min=self.output_min,
        output_max=self.output_max)

  def get_config(self):
    """Standard Keras config for serializaion."""
    config = {
        "output_min": self.output_min,
        "output_max": self.output_max,
    }  # pyformat: disable
    return config


class BiasInitializer(keras.initializers.Initializer):
  # pyformat: disable
  """Initializes bias depending on output_min and output_max.

  If both output_min and output_max are set, bias is initialized to their
  average. If only output_min is set, bias is initialized to output_min. If only
  output_max is set, bias is initialized to output_max. Otherwise bias is
  initialized to zeros.
  """
  # pyformat: enable

  def __init__(self, output_min, output_max):
    """Initializes an instance of `BiasInitializer`.

    Args:
      output_min: None or minimum layer output.
      output_max: None or maximum layer output.
    """
    self.output_min = output_min
    self.output_max = output_max

  def __call__(self, shape, dtype=None, **kwargs):
    """Returns weights of `tfl.layers.KroneckerFactoredLattice` bias.

    Args:
      shape: Must be: `(units, num_terms)`.
      dtype: Standard Keras initializer param.
      **kwargs: Other args passed to `keras.initializers.Initializer` __call__
        method.
    """
    return kfl_lib.bias_initializer(
        units=shape[0],
        output_min=self.output_min,
        output_max=self.output_max,
        dtype=dtype)

  def get_config(self):
    """Standard Keras config for serializaion."""
    config = {
        "output_min": self.output_min,
        "output_max": self.output_max,
    }  # pyformat: disable
    return config


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
               output_min=None,
               output_max=None):
    """Initializes an instance of `KroneckerFactoredLatticeConstraints`.

    Args:
      units: Same meaning as corresponding parameter of
        `KroneckerFactoredLattice`.
      scale: Scale variable of shape: `(units, num_terms)`.
      monotonicities: Same meaning as corresponding parameter of
        `KroneckerFactoredLattice`.
      output_min: Same meaning as corresponding parameter of
        `KroneckerFactoredLattice`.
      output_max: Same meaning as corresponding parameter of
        `KroneckerFactoredLattice`.
    """
    self.units = units
    self.scale = scale
    self.monotonicities = utils.canonicalize_monotonicities(
        monotonicities, allow_decreasing=False)
    self.num_constraint_dims = utils.count_non_zeros(self.monotonicities)
    self.output_min = output_min
    self.output_max = output_max

  def __call__(self, w):
    """Applies constraints to `w`.

    Args:
      w: Kronecker-Factored Lattice weights tensor of shape: `(1, lattice_sizes,
        units * dims, num_terms)`.

    Returns:
      Constrained and projected w.
    """
    if self.num_constraint_dims:
      w = kfl_lib.finalize_weight_constraints(
          w,
          units=self.units,
          scale=self.scale,
          monotonicities=self.monotonicities,
          output_min=self.output_min,
          output_max=self.output_max)
    return w

  def get_config(self):
    """Standard Keras config for serialization."""
    return {
        "units": self.units,
        "scale": self.scale,
        "monotonicities": self.monotonicities,
        "output_min": self.output_min,
        "output_max": self.output_max,
    }  # pyformat: disable


class ScaleConstraints(keras.constraints.Constraint):
  # pyformat: disable
  """Constraints for `tfl.layers.KroneckerFactoredLattice` scale.

  Constraints the scale variable to be between
  `[output_min-output_max, output_max-output_min]` such that the final output
  of the layer is within the desired `[output_min, output_max]` range, assuming
  bias is properly fixed to be `output_min`.

  Attributes:
    - All `__init__` arguments.
  """
  # pyformat: enable

  def __init__(self, output_min=None, output_max=None):
    """Initializes an instance of `ScaleConstraints`.

    Args:
      output_min: Same meaning as corresponding parameter of
        `KroneckerFactoredLattice`.
      output_max: Same meaning as corresponding parameter of
        `KroneckerFactoredLattice`.
    """
    self.output_min = output_min
    self.output_max = output_max

  def __call__(self, scale):
    """Applies constraints to `scale`.

    Args:
      scale: Kronecker-Factored Lattice scale tensor of shape: `(units,
        num_terms)`.

    Returns:
      Constrained and clipped scale.
    """
    if self.output_min is not None or self.output_max is not None:
      scale = kfl_lib.finalize_scale_constraints(
          scale, output_min=self.output_min, output_max=self.output_max)
    return scale

  def get_config(self):
    """Standard Keras config for serialization."""
    return {
        "output_min": self.output_min,
        "output_max": self.output_max,
    }  # pyformat: disable
