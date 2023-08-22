# Copyright 2023 Google LLC
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
"""Implements CDF transformation with derived parameters (kernels).

`cdf_fn` is similar to `tfl.layers.CDF`, which is an additive / multiplicative
average of a few shifted and scaled `sigmoid` or `relu6` basis functions,
with the difference that the functions are parametrized by the provided
parameters instead of learnable weights belonging to a `tfl.layers.CDF` layer.

These parameters can be one of:

  - constants,
  - trainable variables,
  - outputs from other TF modules.

For inputs of shape `(batch_size, input_dim)`, two sets of free-form
parameters are used to configure the CDF function:

- `location_parameters` for where to place the sigmoid / relu6 transformation
basis,
- `scaling_parameters` (optional) for the horizontal scaling before applying
the transformation basis.
"""

from typing import Optional, Union, Tuple
import tensorflow as tf


def _verify_cdf_params(
    inputs: tf.Tensor,
    location_parameters: tf.Tensor,
    scaling_parameters: Optional[tf.Tensor],
    units: int,
    activation: str,
    reduction: str,
    sparsity_factor: int,
) -> None:
  """Verifies the arguments of cdf_fn call.

  Args:
    inputs: inputs to the CDF function.
    location_parameters: parameters for deciding the locations of the
      transformations.
    scaling_parameters: parameters for deciding the horizontal scaling of the
      transformations.
    units: output dimension.
    activation: either `sigmoid` or `relu6` for selecting the transformation.
    reduction: either `mean`, `geometric_mean`, or `none` to specify whether to
      perform averaging and which average to perform.
    sparsity_factor: deciding the level of sparsity during reduction.
      `input_dim` and `units` should both be divisible by `sparsity_factor`.
  """
  if activation not in ("sigmoid", "relu6"):
    raise ValueError(
        f"activation = {activation} is not supported. Use 'sigmoid' or 'relu6'."
    )
  if reduction not in ("mean", "geometric_mean", "none"):
    raise ValueError(
        f"reduction = {reduction} is not supported. Use 'mean',"
        " 'geometric_mean' or 'none'."
    )

  if len(inputs.shape) != 2:
    raise ValueError(
        f"inputs shape {inputs.shape} is not (batch_size, input_dim)."
    )

  input_dim = inputs.shape[1]
  if units % sparsity_factor != 0:
    raise ValueError(
        f"units = {units} is not divisible by sparsity_factor ="
        f" {sparsity_factor}."
    )
  if input_dim % sparsity_factor != 0:
    raise ValueError(
        f"input_dim = {input_dim} is not divisible by sparsity_factor ="
        f" {sparsity_factor}."
    )

  if (
      len(location_parameters.shape) != 4
      or location_parameters.shape[1] != input_dim
      or location_parameters.shape[3] != units // sparsity_factor
  ):
    raise ValueError(
        "location_parameters shape"
        f" {location_parameters.shape} is not (batch, input_dim, "
        f"num_functions, units / sparsity_factor = {units // sparsity_factor})."
    )

  if scaling_parameters is not None:
    try:
      _ = tf.broadcast_to(
          scaling_parameters,
          location_parameters.shape,
          name="cdf_fn_try_broadcasting",
      )
    except Exception as err:
      raise ValueError(
          "scaling_parameters and location_parameters likely"
          " are not broadcastable. Shapes of scaling_parameters:"
          f" {scaling_parameters.shape}, location_parameters:"
          f" {location_parameters.shape}."
      ) from err


@tf.function
def cdf_fn(
    inputs: tf.Tensor,
    location_parameters: tf.Tensor,
    scaling_parameters: Optional[tf.Tensor] = None,
    units: int = 1,
    activation: str = "relu6",
    reduction: str = "mean",
    sparsity_factor: int = 1,
    scaling_exp_transform_multiplier: Optional[float] = None,
    return_derived_parameters: bool = False,
) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
  r"""Maps `inputs` through a CDF function specified by keypoint parameters.

  `cdf_fn` is similar to `tfl.layers.CDF`, which is an additive / multiplicative
  average of a few shifted and scaled `sigmoid` or `relu6` basis functions,
  with the difference that the functions are parametrized by the provided
  parameters instead of learnable weights belonging to a `tfl.layers.CDF` layer.

  These parameters can be one of:

    - constants,
    - trainable variables,
    - outputs from other TF modules.

  For inputs of shape `(batch_size, input_dim)`, two sets of free-form
  parameters are used to configure the CDF function:

  - `location_parameters` for where to place the sigmoid / relu6 transformation
  basis,
  - `scaling_parameters` (optional) for the horizontal scaling before applying
  the transformation basis.

  The transformation per dimension is `x -> activation(scale * (x - location))`,
  where:

  - `scale` (specified via `scaling_parameter`) is the input scaling for each
  dimension and needs to be strictly positive for the CDF function to become
  monotonic. If needed, you can set `scaling_exp_transform_multiplier` to get
  `scale = exp(scaling_parameter * scaling_exp_transform_multiplier)` and
  guarantees strict positivity.
  - `location` (specified via `location_parameter`) is the input shift. Notice
  for `relu6` this is where the transformation starts to be nonzero, whereas for
  `sigmoid` this is where the transformation hits 0.5.
  - `activation` is either `sigmoid` or `relu6` (for `relu6 / 6`).

  An optional `reduction` operation will compute the additive / multiplicative
  average for the input dims after their individual CDF transformation. `mean`
  and `geometric_mean` are supported if sepcified.

  `sparsity_factor` decides the level of sparsity during reduction. For
  instance, default of `sparsity = 1` calculates the average of *all* input
  dims, whereas `sparsity = 2` calculates the average of *every other* input
  dim, and so on.

  Input shape:
    We denote `num_functions` as the number of `sigmoid` or `relu6 / 6` basis
    functions used for each CDF transformation.

    `inputs` should be:

    - `(batch_size, input_dim)`.

    `location_parameters` should be:

    - `(batch_size, input_dim, num_functions, units // sparsity_factor)`.

    `scaling_parameters` when provided should be broadcast friendly
    with `location_parameters`, e.g. one of

    - `(batch_size, input_dim, 1, 1)`,
    - `(batch_size, input_dim, num_functions, 1)`,
    - `(batch_size, input_dim, 1, units // sparsity_factor)`,
    - `(batch_size, input_dim, num_functions, units // sparsity_factor)`.

  Args:
    inputs: inputs to the CDF function.
    location_parameters: parameters for deciding the locations of the
      transformations.
    scaling_parameters: parameters for deciding the horizontal scaling of the
      transformations.
    units: output dimension.
    activation: either `sigmoid` or `relu6` for selecting the transformation.
    reduction: either `mean`, `geometric_mean`, or `none` to specify whether to
      perform averaging and which average to perform.
    sparsity_factor: deciding the level of sparsity during reduction.
      `input_dim` and `units` should both be divisible by `sparsity_factor`.
    scaling_exp_transform_multiplier: if provided, will be used inside an
      exponential transformation for `scaling_parameters`. This can be useful if
      `scaling_parameters` is free-form.
    return_derived_parameters: Whether `location_parameters` and
      `scaling_parameters` should be output along with the model output (e.g.
      for loss function computation purpoeses).

  Returns:
    If `return_derived_parameters = False`:

      - The CDF transformed outputs as a tensor with shape either
        `(batch_size, units)` if `reduction = 'mean' / 'geometric_mean'`, or
        `(batch_size, input_dim // sparsity_factor, units)` if
        `reduction = 'none'`.

    If `return_derived_parameters = True`:

      - A tuple of three elements:

        1. The CDF transformed outputs.
        2. `location_parameters`.
        3. `scaling_parameters`, with `exp` transformation applied if specified.
  """

  _verify_cdf_params(
      inputs,
      location_parameters,
      scaling_parameters,
      units,
      activation,
      reduction,
      sparsity_factor,
  )
  input_dim = inputs.shape[1]
  x = inputs[..., tf.newaxis, tf.newaxis] - location_parameters
  if scaling_parameters is not None:
    if scaling_exp_transform_multiplier is not None:
      scaling_parameters = tf.math.exp(
          scaling_parameters * scaling_exp_transform_multiplier
      )
    x *= scaling_parameters
  else:
    # For use when return_derived_parameters = True.
    scaling_parameters = tf.ones_like(location_parameters, dtype=tf.float32)

  # Shape: (batch, input_dim, 1, 1)
  #    --> (batch, input_dim, num_functions, units / factor)
  #    --> (batch, input_dim, units / factor).
  if activation == "relu6":
    result = tf.reduce_mean(tf.nn.relu6(x), axis=2) / 6
  else:  # activation == "sigmoid":
    result = tf.reduce_mean(tf.nn.sigmoid(x), axis=2)

  if sparsity_factor != 1:
    # Shape: (batch, input_dim, units / factor)
    #    --> (batch, input_dim / factor, units).
    result = tf.reshape(result, (-1, input_dim // sparsity_factor, units))

  # Shape: (batch, input_dim / factor, units) --> (batch, units).
  if reduction == "mean":
    result = tf.reduce_mean(result, axis=1)
  elif reduction == "geometric_mean":
    # We use the log form so that we can add the epsilon term
    # tf.pow(tf.reduce_prod(cdfs, axis=1), 1. / num_terms).
    result = tf.math.exp(tf.reduce_mean(tf.math.log(result + 1e-8), axis=1))
  # Otherwise reduction == "none".

  if return_derived_parameters:
    return (result, location_parameters, scaling_parameters)
  else:
    return result
