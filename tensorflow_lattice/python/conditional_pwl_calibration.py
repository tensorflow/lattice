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
"""Implements PWLCalibration with derived parameters (kernels).

`pwl_calibration_fn` is similar to `tfl.layers.PWLCalibration` with the key
difference that the keypoints are decided by the given parameters instead
of learnable weights belonging to a layer. These parameters can be one of:

  - constants,
  - trainable variables,
  - outputs from other TF modules.

For inputs of shape `(batch_size, units)`, two sets of parameters are required
to configure the piece-wise linear calibrator in terms of its x and y values:

 - `keypoint_input_parameters` for configuring the x values,
 - `keypoint_output_parameters` for configuring the y values.

This function is a general form of conditional calibration, that one input
variable is calibrated based on free form parameters coming from other variables
and their transformations.

Shapes:
The last dimension sizes of `keypoint_input_parameters` (input_param_size) and
`keypoint_output_parameters` (output_param_size) depend on the number of
keypoints used by the calibrator. We follow the relationships that

 - input_param_size = # keypoints - 2, as the leftmost and rightmost keypoints
   are given.
 - output_param_size = # keypoints initially, and we then modify it by

   1. if cyclic calibrator: output_param_size -= 1,
   2. if clamp_min: output_param_size -= 1,
   3. if clamp_max: output_param_size -= 1,
   4. if need to learn how to impute missing: output_param_size += 1.

The final shapes need to be broadcast friendly with `(batch_size, units, 1)`:

 - `keypoint_input_parameters`:
   `(1 or batch_size, 1 or units, input_param_size)`.
 - `keypoint_output_parameters`:
   `(1 or batch_size, 1 or units, output_param_size)`.
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf


def _front_pad(x: tf.Tensor, constant_values: float) -> tf.Tensor:
  return tf.pad(x, [[0, 0], [0, 0], [1, 0]], constant_values=constant_values)


def default_keypoint_output_parameters(
    num_keypoints: int,
    units: int = 1,
    monotonicity: str = "none",
    is_cyclic: bool = False,
    clamp_min: bool = False,
    clamp_max: bool = False,
    derived_missing_output: bool = False,
) -> Optional[tf.Tensor]:
  """Helper creating default `keypoint_output_parameters`.

  Primarily used for testing.

  Args:
    num_keypoints: number of keypoints for inputs.
    units: number of parallel calibrations on one input.
    monotonicity: `none` or `increasing`, monotonicity of the calibration.
    is_cyclic: whether the calibration is cyclic. Only works if `monotonicity ==
      none`.
    clamp_min: whether the leftmost keypoint should be clamped. Only works if
      `monotonicity == increasing`.
    clamp_max: whether the rightmost keypoint should be clamped. Only works if
      `monotonicity == increasing`.
    derived_missing_output: whether to reserve a placeholder for the missing
      output value.

  Returns:
    A tensor with a shape of `(1, units, output_param_size)`.

  Raises:
    `ValueError` if parsing failed.
  """
  if monotonicity == "none":
    output_param_size = num_keypoints - is_cyclic + derived_missing_output
    # default output = midpoint between
    # keypoint_output_min and keypoint_output_max, flat.
    return tf.zeros((1, units, output_param_size), dtype=tf.float32)
  elif monotonicity == "increasing":
    output_param_size = (
        num_keypoints - clamp_min - clamp_max + derived_missing_output
    )
    # default output = equal increments between
    # keypoint_output_min and keypoint_output_max.
    return tf.zeros((1, units, output_param_size), dtype=tf.float32)
  else:
    raise ValueError(f"Unknown monotonicity: {monotonicity}")


def default_keypoint_input_parameters(
    num_keypoints: Optional[int] = None,
    keypoints: Optional[Sequence[float]] = None,
    units: int = 1,
) -> Optional[tf.Tensor]:
  """Helper creating default `keypoint_input_parameters`.

  Primarily used for testing.

  Args:
    num_keypoints: number of keypoints. If provided, keypoints will be equally
      spaced.
    keypoints: sequence of increasing keypoints.
    units: number of parallel calibrations on one input.

  Returns:
    A tensor with a shape of `(1, units, input_param_size)` or
      `(1, units, input_param_size)`.

  Raises:
    `ValueError` if parsing failed.
  """
  if num_keypoints is not None and num_keypoints > 2:
    return tf.zeros((1, units, num_keypoints - 2), dtype=tf.float32)
  elif keypoints is not None:
    keypoints = np.array(keypoints)
    deltas = keypoints[1:] - keypoints[:-1]
    if np.all(deltas > 0):
      deltas = deltas / np.sum(deltas)
      deltas = np.log(deltas / deltas[0])[1:]
      deltas = tf.reshape(tf.constant(deltas, dtype=tf.float32), (1, 1, -1))
      return tf.tile(deltas, [1, units, 1])
  else:
    raise ValueError("Neither num_keypoints nor keypoints is specified.")


def _verify_pwl_calibration(
    inputs,
    keypoint_input_parameters,
    keypoint_output_parameters,
    units,
    keypoint_input_min,
    keypoint_input_max,
    keypoint_output_min,
    keypoint_output_max,
    clamp_min,
    clamp_max,
    monotonicity,
    is_cyclic,
    missing_input_value,
    missing_output_value,
):
  """Validates calibration arguments."""
  # Validate keypoint input_min and input_max.
  if keypoint_input_min > keypoint_input_max:
    raise ValueError(
        f"keypoint_input_min = {keypoint_input_min} > keypoint_input_max ="
        f" {keypoint_input_max}."
    )

  # Validate pwl shape arguments.
  if monotonicity not in ("none", "increasing"):
    raise ValueError(
        "Monotonicity should be 'none' or 'increasing'. "
        f"Given '{monotonicity}'."
    )

  if monotonicity == "none" and (clamp_min or clamp_max):
    raise ValueError("Cannot clamp to min or max when monotonicity is 'none'.")

  if keypoint_output_min > keypoint_output_max:
    raise ValueError(
        f"keypoint_output_min = {keypoint_output_min} > keypoint_output_max ="
        f" {keypoint_output_max}."
    )

  if monotonicity == "increasing" and is_cyclic:
    raise ValueError("Monotonicity should be 'none' when is_cyclic=True.")

  # Validate missingness indicators.
  if missing_output_value is not None and missing_input_value is None:
    raise ValueError(
        "missing_output_value is set, but missing_input_value is None"
    )

  # Validate parameter shapes. See module level doc string for details.
  num_keypoints = (
      keypoint_input_parameters.shape[-1] + 2
      if keypoint_input_parameters is not None
      else 0
  )
  output_param_size = (
      num_keypoints
      - clamp_max
      - clamp_min
      - is_cyclic
      + (missing_input_value is not None)
      - (missing_output_value is not None)
  )

  if output_param_size <= 0:
    raise ValueError(
        f"Required keypoint_output_parameters per example = {output_param_size}"
        " <= 0: Creating a trivial function, e.g. identity or constant."
    )

  if units > 1 and len(keypoint_output_parameters.shape) != 3:
    raise ValueError(
        "keypoint_output_parameters should be 3 dimensional when units > 1. "
        f"Given {keypoint_output_parameters.shape}."
    )
  if (
      len(keypoint_output_parameters.shape) == 3
      and keypoint_output_parameters.shape[1] != units
  ):
    raise ValueError(
        "2nd dimension of keypoint_output_parameters does not match units, "
        f"units = {units} vs keypoint_output_parameters = "
        f"{keypoint_output_parameters.shape[1]}."
    )
  if keypoint_output_parameters.shape[-1] != output_param_size:
    raise ValueError(
        "keypoint_output_parameters shape is "
        f"{keypoint_output_parameters.shape} whose last dimension needs to be "
        f"{output_param_size}."
    )

  # Validate input shape.
  if inputs.shape[1] > 1 and inputs.shape[1] != units:
    raise ValueError(
        "2nd dimension of input shape does not match units > 1, "
        f"Require (batch_size, 1) or (batch_size, units = {units})."
    )


def _compute_interpolation_weights(inputs, keypoints, lengths):
  """Computes weights for PWL calibration.

  Args:
    inputs: Tensor of shape: `(batch_size, units, 1)`. For multi-unit
      calibration, broadcasting will be used if needed.
    keypoints: Tensor of shape `(num_keypoints-1)` which represents left
      keypoint of pieces of piecewise linear function along X axis.
    lengths: Tensor of shape `(num_keypoints-1)` which represents lengths of
      pieces of piecewise linear function along X axis.

  Returns:
    Interpolation weights tensor of shape: `(batch_size, units, num_keypoints)`.
  """
  # weights always matches the shape of inputs.
  weights = (inputs - keypoints) / lengths
  weights = tf.clip_by_value(weights, 0.0, 1.0)
  return _front_pad(weights, 1.0)


@tf.function
def pwl_calibration_fn(
    inputs: tf.Tensor,
    keypoint_input_parameters: Optional[tf.Tensor],
    keypoint_output_parameters: tf.Tensor,
    keypoint_input_min: float = 0.0,
    keypoint_input_max: float = 1.0,
    keypoint_output_min: float = 0.0,
    keypoint_output_max: float = 1.0,
    units: int = 1,
    monotonicity: str = "none",
    clamp_min: bool = False,
    clamp_max: bool = False,
    is_cyclic: bool = False,
    missing_input_value: Optional[float] = None,
    missing_output_value: Optional[float] = None,
    return_derived_parameters: bool = False,
) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
  """Calibrates `inputs` using derived parameters (kernels).

  `pwl_calibration_fn` is similar to `tfl.layers.PWLCalibration` with the key
  difference that the keypoints are decided by the given parameters instead
  of learnable weights belonging to a layer. These parameters can be one of:

    - constants,
    - trainable variables,
    - outputs from other TF modules.

  Shapes:
  The last dimension of `keypoint_input_parameters` (`input_param_size`) and
  `keypoint_output_parameters` (`output_param_size`) depend on the number of
  keypoints used by the calibrator. We follow the relationships that

  - `input_param_size = # keypoints - 2`, as the leftmost and rightmost
    keypoints are given.
  - `output_param_size = # keypoints` initially, and we then modify it by

    1. if cyclic calibrator: `output_param_size -= 1`,
    2. if clamp_min: `output_param_size -= 1`,
    3. if clamp_max: `output_param_size -= 1`,
    4. if need to learn how to impute missing: `output_param_size += 1`.

  The final shapes need to be broadcast friendly with `(batch_size, units, 1)`:

  - `keypoint_input_parameters`:
    `(1 or batch_size, 1 or units, input_param_size)`.
  - `keypoint_output_parameters`:
    `(1 or batch_size, 1 or units, output_param_size)`.

  Input shape:
    `inputs` should be one of:

      - `(batch_size, 1)` if `units == 1`.
      - `(batch_size, 1)` or `(batch_size, units)` if `units > 1`.
        The former will be broadcast to match units.

    `keypoint_input_parameters` should be one of:

      - `None` if only the leftmost and the rightmost keypoints are required.
      - `(1, input_param_size)`.
      - `(batch_size, input_param_size)`.
      - `(1, 1, input_param_size)`.
      - `(batch_size, 1, input_param_size)`.
      - `(1, units, input_param_size)`.
      - `(batch_size, units, input_param_size)`.

    `keypoint_output_parameters` should be one of:

      - `(1, output_param_size)`.
      - `(batch_size, output_param_size)`.
      - `(1, 1, output_param_size)`.
      - `(batch_size, 1, output_param_size)`.
      - `(1, units, output_param_size)`.
      - `(batch_size, units, output_param_size)`.

  Args:
    inputs: inputs to the calibration fn.
    keypoint_input_parameters: parameters for keypoint x's of calibration fn.
    keypoint_output_parameters: parameters for keypoint y's of calibration fn.
    keypoint_input_min: the leftmost keypoint.
    keypoint_input_max: the rightmost keypoint.
    keypoint_output_min: lower bound of the fn output.
    keypoint_output_max: upper bound of the fn output.
    units: number of parallel calibrations on one input.
    monotonicity: `none` or `increasing`. Whether the calibration is monotonic.
    clamp_min: only applies when monotonicity == `increasing`. Whether to clamp
      the LHS keypoint to the calibration `keypoint_output_min`.
    clamp_max: only applies when monotonicity == `increasing`. Whether to clamp
      the RHS keypoint to the calibration `keypoint_output_max`.
    is_cyclic: only applies when monotonicity == `none`. Whether the LHS and the
      RHS keypoints have the same calibration output.
    missing_input_value: if set, use as the value indicating a missing input.
    missing_output_value: if set, use as the output for `missing_input_value`.
    return_derived_parameters: if True, return the derived kernel parameters
      used for interpolation.

  Returns:
    If `return_derived_parameters = False`:

      - The calibrated output as a tensor with shape `(batch_size, units)`.

    If `return_derived_parameters == True`:

      - A tuple of three elements:

        1. The calibrated output as a tensor with shape `(batch_size, units)`.
        2. The deltas between the keypoints x's with shape
          `(batch_size, units, # keypoints - 1)`.
        3. The initial value and the deltas between the keypoints y's, with
          shape shape `(batch_size, units, # keypoints)`. Apply `cumsum` will
          reconstruct the y values.
  """
  _verify_pwl_calibration(
      inputs=inputs,
      keypoint_input_parameters=keypoint_input_parameters,
      keypoint_output_parameters=keypoint_output_parameters,
      units=units,
      keypoint_input_min=keypoint_input_min,
      keypoint_input_max=keypoint_input_max,
      keypoint_output_min=keypoint_output_min,
      keypoint_output_max=keypoint_output_max,
      clamp_min=clamp_min,
      clamp_max=clamp_max,
      monotonicity=monotonicity,
      is_cyclic=is_cyclic,
      missing_input_value=missing_input_value,
      missing_output_value=missing_output_value,
  )

  if keypoint_input_parameters is None:
    keypoint_input_parameters = tf.zeros((1, units, 1), dtype=tf.float32)
  else:
    if len(keypoint_input_parameters.shape) == 2:
      keypoint_input_parameters = keypoint_input_parameters[:, tf.newaxis, :]
    if keypoint_input_parameters.shape[1] == 1 and units > 1:
      keypoint_input_parameters = tf.tile(
          keypoint_input_parameters, [1, units, 1]
      )
    # Front-pad 0 to normalize softmax.
    keypoint_input_parameters = _front_pad(keypoint_input_parameters, 0.0)

  keypoint_deltas = tf.nn.softmax(keypoint_input_parameters, axis=-1) * (
      keypoint_input_max - keypoint_input_min
  )
  # Front-pad `input_min` as the leftmost keypoint.
  # Trim the rightmost keypoint not required for interpolation.
  keypoints = (
      tf.cumsum(keypoint_deltas, exclusive=True, axis=-1) + keypoint_input_min
  )

  # Rename since its value will be modified as part of the output.
  kernel_outputs = keypoint_output_parameters
  if len(kernel_outputs.shape) == 2:
    kernel_outputs = kernel_outputs[:, tf.newaxis, :]
  if kernel_outputs.shape[1] == 1 and units > 1:
    kernel_outputs = tf.tile(kernel_outputs, [1, units, 1])

  missing_output = None
  if missing_input_value is not None:
    if missing_output_value is None:
      # The last parameter is used to derive the imputed output value after
      # sigmoid and rescale.
      missing_output = keypoint_output_min + tf.sigmoid(
          kernel_outputs[:, :, -1]
      ) * (keypoint_output_max - keypoint_output_min)
      kernel_outputs = kernel_outputs[:, :, :-1]
    else:
      missing_output = tf.fill(
          kernel_outputs[:, :, -1].shape, missing_output_value
      )

  if monotonicity == "none":
    kernel_outputs = (
        tf.sigmoid(kernel_outputs) * (keypoint_output_max - keypoint_output_min)
        + keypoint_output_min
    )
    if is_cyclic:
      kernel_outputs = tf.concat(
          [kernel_outputs, kernel_outputs[:, :, :1]], axis=-1
      )
    # Transform to [initial value, delta_0, delta_1,...].
    kernel_outputs = tf.concat(
        [
            kernel_outputs[:, :, :1],
            kernel_outputs[:, :, 1:] - kernel_outputs[:, :, :-1],
        ],
        axis=-1,
    )
  else:  # monotonicity == "increasing"
    # Front-pad zero to normalize softmax.
    kernel_outputs = _front_pad(kernel_outputs, 0.0)
    kernel_outputs = tf.nn.softmax(kernel_outputs, axis=-1) * (
        keypoint_output_max - keypoint_output_min
    )
    if clamp_min:
      # Front-pad keypoint_output_min to the kernel_outputs.
      kernel_outputs = _front_pad(kernel_outputs, keypoint_output_min)
    else:
      # Add keypoint_output_min to the LHS element in the kernel_outputs.
      # TODO: test tf.tensor_scatter_nd_add.
      kernel_outputs = tf.concat(
          [
              kernel_outputs[:, :, :1] + keypoint_output_min,
              kernel_outputs[:, :, 1:],
          ],
          axis=-1,
      )
    if not clamp_max:
      # Drop the RHS element in the kernel_outputs which made cumsum = 1.
      kernel_outputs = kernel_outputs[:, :, :-1]

  if units > 1 and inputs.shape[-1] == 1:
    inputs = tf.tile(inputs, [1, units])
  weights = _compute_interpolation_weights(
      tf.reshape(inputs, (-1, units, 1)), keypoints, keypoint_deltas
  )
  outputs = tf.reduce_sum(weights * kernel_outputs, axis=-1, keepdims=False)

  if missing_input_value is not None:
    outputs = tf.where(
        tf.equal(inputs, missing_input_value), missing_output, outputs
    )

  if return_derived_parameters:
    return outputs, keypoint_deltas, kernel_outputs
  else:
    return outputs
