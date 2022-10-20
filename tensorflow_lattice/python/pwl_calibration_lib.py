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
"""Implementation of algorithms required for PWL calibration layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import enum
from . import utils
import tensorflow as tf


class BoundConstraintsType(enum.Enum):
  """Type of bound constraints for PWL calibration.

  - NONE: no constraints.
  - BOUND: output range can be anywhere within bounds.
  - CLAMPED: output range must exactly match bounds.
  """
  NONE = 0
  BOUND = 1
  CLAMPED = 2


def convert_all_constraints(output_min, output_max, clamp_min, clamp_max):
  """Converts parameters of PWL calibration layer to internal format.

  Args:
    output_min: None for unconstrained bound or some numeric value.
    output_max: None for unconstrained bound or some numeric value.
    clamp_min: Whether to clamp pwl calibrator to value if `output_min` is not
      None.
    clamp_max: Whether to clamp pwl calibrator to value if `output_max` is not
      None.

  Returns:
    "value" as float and appropriate value of
    `tfl.pwl_calibration_lib.BoundConstraintsType` enum which corresponds to
    `output_min(max)` and `clamp_min(max)`.
  """
  if output_min is None:
    output_max, output_max_constraints = _convert_constraints(
        output_max, clamp_max)
    output_min = output_max
    output_min_constraints = BoundConstraintsType.NONE
  elif output_max is None:
    output_min, output_min_constraints = _convert_constraints(
        output_min, clamp_min)
    output_max = output_min
    output_max_constraints = BoundConstraintsType.NONE
  else:
    output_min, output_min_constraints = _convert_constraints(
        output_min, clamp_min)
    output_max, output_max_constraints = _convert_constraints(
        output_max, clamp_max)
  return output_min, output_max, output_min_constraints, output_max_constraints


def _convert_constraints(value, clamp_to_value):
  """Converts constraints for output_min/max to internal format.

  Args:
    value: None for unconstrained bound or some numeric value.
    clamp_to_value: Whether to clamp pwl calibrator to value if value isn't None

  Returns:
    "value" as float and appropriate value of
    `tfl.pwl_calibration_lib.BoundConstraintsType` enum which
    corresponds to `value` and `clamp_to_value`.
  """
  if value is None:
    return 0.0, BoundConstraintsType.NONE
  else:
    value = float(value)
    if clamp_to_value:
      return value, BoundConstraintsType.CLAMPED
    else:
      return value, BoundConstraintsType.BOUND


def compute_interpolation_weights(inputs, keypoints, lengths):
  """Computes weights for PWL calibration.

  Args:
    inputs: Tensor of shape: `(batch_size, 1)`, `(batch_size, units, 1)` or
    `(batch_size, 1, 1)`. For multi-unit calibration, broadcasting will be used
    if needed.
    keypoints: Tensor of shape `(num_keypoints-1)` or `(units, num_keypoints-1)`
      which represents left keypoint of pieces of piecewise linear function
      along X axis.
    lengths: Tensor of shape `(num_keypoints-1)` or `(units, num_keypoints-1)`
      which represents lengths of pieces of piecewise linear function along X
      axis.

  Returns:
    Interpolation weights tensor of shape: `(batch_size, num_keypoints)` or
    `(batch_size, units, num_keypoints)`.
  """
  weights = (inputs - keypoints) / lengths
  weights = tf.minimum(weights, 1.0)
  weights = tf.maximum(weights, 0.0)
  # Prepend 1.0 at the beginning to add bias unconditionally. Worth testing
  # different strategies, including those commented out, on different hardware.
  if len(keypoints.shape) == 1:
    return tf.concat([tf.ones_like(inputs), weights], axis=-1)
  else:
    shape = tf.concat([tf.shape(weights)[:-1], [1]], axis=0)
    return tf.concat([tf.ones(shape), weights], axis=-1)
  # return tf.concat([tf.ones_like(weights)[..., :1], weights], axis=-1)
  # return tf.concat([tf.ones_like(weights[..., :1]), weights], axis=-1)
  # paddings = [[0, 0]] * (len(weights.shape) - 1) + [[1, 0]]
  # return tf.pad(weights, paddings, constant_values=1.)


def linear_initializer(shape,
                       output_min,
                       output_max,
                       monotonicity,
                       keypoints=None,
                       dtype=None):
  """Initializes PWL calibration layer to represent linear function.

  PWL calibration layer weights have shape `(num_keypoints, units)`. First row
  represents bias. All remaining represent delta in y-value compare to previous
  point. Aka heights of segments.

  Args:
    shape: Requested shape. Must be `(num_keypoints, units)`.
    output_min: Minimum value of PWL calibration output after initialization.
    output_max: Maximum value of PWL calibration output after initialization.
    monotonicity: If one of {0, 1}, the returned function will go from
      `(input_min, output_min)` to `(input_max, output_max)`. If set to -1, the
      returned function will go from `(input_min, output_max)` to `(input_max,
      output_min)`.
    keypoints: If not provided (None or []), all pieces of returned function
      will have equal heights (i.e. `y[i+1] - y[i]` is constant). If provided,
      all pieces of returned function will have equal slopes (i.e. `(y[i+1] -
      y[i]) / (x[i+1] - x[i])` is constant).
    dtype: dtype.

  Returns:
    PWLCalibration layer weights initialized according to params.

  Raises:
    ValueError: If given parameters are inconsistent.
  """
  verify_hyperparameters(
      input_keypoints=keypoints,
      output_min=output_min,
      output_max=output_max,
      monotonicity=monotonicity,
      weights_shape=shape)

  num_keypoints, units = int(shape[0]), int(shape[1])
  if keypoints is None:
    # Subtract 1 for bias which will be handled separately.
    num_pieces = num_keypoints - 1
    segment_height = (output_max - output_min) / num_pieces
    heights_tensor = tf.constant(
        [segment_height] * num_pieces, shape=[num_pieces, 1], dtype=dtype)
  else:
    keypoints_tensor = tf.constant(
        keypoints, shape=[num_keypoints, 1], dtype=dtype)
    lengths_tensor = keypoints_tensor[1:] - keypoints_tensor[0:-1]
    output_range = output_max - output_min
    heights_tensor = (
        lengths_tensor * (output_range / tf.reduce_sum(lengths_tensor)))

  if units > 1:
    heights_tensor = tf.tile(heights_tensor, multiples=[1, units])

  if monotonicity == -1:
    bias = output_max
    heights_tensor = -heights_tensor
  else:
    bias = output_min
  bias_tensor = tf.constant(bias, shape=[1, units], dtype=dtype)

  return tf.concat([bias_tensor, heights_tensor], axis=0)


def _approximately_project_bounds_only(bias, heights, output_min, output_max,
                                       output_min_constraints,
                                       output_max_constraints):
  """Bounds constraints implementation for PWL calibration layer.

  Maps given weights of PWL calibration layer into some point which satisfies
  given bounds by capping the function based on the bounds. This is not an exact
  projection in L2 norm, but it is sufficiently accurate and efficient in
  practice for non monotonic functions.

  Args:
    bias: `(1, units)`-shape tensor which represents bias.
    heights: `(num_heights, units)`-shape tensor which represents heights.
    output_min: Minimum possible output of pwl function.
    output_max: Maximum possible output of pwl function.
    output_min_constraints: A `tfl.pwl_calibration_lib.BoundConstraintsType`
      describing the constraints on the layer's minimum value.
    output_max_constraints: A `tfl.pwl_calibration_lib.BoundConstraintsType`
      describing the constraints on the layer's maximum value.

  Raises:
    ValueError: If `output_min(max)_constraints` is set to "CLAMPED" which is
      not supported.

  Returns:
    Projected bias and heights.
  """
  if (output_min_constraints == BoundConstraintsType.CLAMPED or
      output_max_constraints == BoundConstraintsType.CLAMPED):
    raise ValueError("Clamping is not implemented for non monotonic functions.")
  if (output_min_constraints == BoundConstraintsType.NONE and
      output_max_constraints == BoundConstraintsType.NONE):
    return bias, heights

  # Compute cumulative sums - they correspond to our calibrator outputs at
  # keypoints. Simply clip them according to config and compute new heights
  # using clipped cumulative sums.
  sums = tf.cumsum(tf.concat([bias, heights], axis=0))
  if output_min_constraints == BoundConstraintsType.BOUND:
    sums = tf.maximum(sums, output_min)
  if output_max_constraints == BoundConstraintsType.BOUND:
    sums = tf.minimum(sums, output_max)

  bias = sums[0:1]
  heights = sums[1:] - sums[:-1]
  return bias, heights


def _project_bounds_considering_monotonicity(bias, heights, monotonicity,
                                             output_min, output_max,
                                             output_min_constraints,
                                             output_max_constraints):
  """Bounds projection given monotonicity constraints.

  Projects weights of PWLCalibration layer into nearest in terms of l2 distance
  point which satisfies bounds constraints taking into account that function
  is monotonic.

  Algorithm:
  To minimize L2 distance to projected point we want to distribute update
  through heights as evenly as possible. A simplified description of the
  algorithm for and increasing function is as follows:
  Consider only increasing function.

  ```
  delta = (output_max - (bias + sum(heights[:]))) / (num_heights + 1)
  bias = max(bias + delta, output_min)
  heights[:] += delta
  ```

  Some details which were omitted above:
  * If `output_min_constraints == "CAPPED"` then `bias` variable becomes
    constant (this means we can't add delta to it).
  * if `output_max_constraints != "CAPPED"` we are looking only for negative
    delta because we are not required to stretch function to meet upper bound.
  * If function is decreasing we multiply everything by -1 and switch min and
    max to make it increasing.

  Args:
    bias: `(1, units)`-shape tensor which represents bias.
    heights: `(num_heights, units)`-shape tensor which represents heights.
    monotonicity: 1 for increasing, -1 for decreasing.
    output_min: Lower bound constraint of PWL calibration layer.
    output_max: Upper bound constraint of PWL calibration layer.
    output_min_constraints: A `tfl.pwl_calibration_lib.BoundConstraintsType`
      describing the constraints on the layer's minimum value.
    output_max_constraints: A `tfl.pwl_calibration_lib.BoundConstraintsType`
      describing the constraints on the layer's maximum value.

  Returns:
    Projected bias and heights tensors.

  Raises:
    ValueError: If monotonicity is not in: {-1, 1}
  """
  if monotonicity not in [-1, 1]:
    raise ValueError("Monotonicity should be one of: [-1, 1]. It is: " +
                     str(monotonicity))
  if monotonicity == -1:
    # Reduce computation of projection of decreasing function to computation of
    # projection of increasing function by multiplying everything by -1 and
    # swapping maximums and minimums.
    (projected_bias,
     projected_heights) = _project_bounds_considering_monotonicity(
         bias=-bias,
         heights=-heights,
         monotonicity=1,
         output_min=None if output_max is None else -output_max,
         output_max=None if output_min is None else -output_min,
         output_min_constraints=output_max_constraints,
         output_max_constraints=output_min_constraints)
    return -projected_bias, -projected_heights

  bct = BoundConstraintsType
  if output_max_constraints != bct.NONE:
    num_heights = float(heights.shape.dims[0].value)
    sum_heights = tf.reduce_sum(heights, axis=0)

    # For each possible output_min_constraints value compute projected bias and
    # heights_delta.
    if output_min_constraints == bct.CLAMPED:
      # If output_min is clamped - bias must have fixed value and number of free
      # parameters is equal to number of heights.
      bias = tf.constant(output_min, shape=bias.shape, dtype=bias.dtype)
      heights_delta = (output_max - (bias + sum_heights)) / num_heights
    elif output_min_constraints == bct.BOUND:
      # If output_min is not clamped then number of free parameters is
      # num_heights + 1.
      bias_delta = (output_max - (bias + sum_heights)) / (num_heights + 1)
      if output_max_constraints != bct.CLAMPED:
        # If output_max is not clamped - there is no need to stretch our
        # function. We need only to squeeze it.
        bias_delta = tf.minimum(bias_delta, 0.0)
      bias = tf.maximum(bias + bias_delta, output_min)
      # For this branch compute heights delta _after_ we applied bias projection
      # because heights are not bound by output_min constraint unlike bias.
      heights_delta = (output_max - (bias + sum_heights)) / num_heights
    else:
      bias_delta = (output_max - (bias + sum_heights)) / (num_heights + 1)
      # For this branch heights delta and bias delta are same because none of
      # them are bounded from below.
      heights_delta = bias_delta
      if output_max_constraints != bct.CLAMPED:
        # If output_max is not clamped - there is no need to stretch our
        # function. We need only to squeeze it.
        bias_delta = tf.minimum(bias_delta, 0.0)
      bias += bias_delta

    if output_max_constraints != bct.CLAMPED:
      # If output_max is not clamped - there is no need to stretch our function.
      # We need only to squeeze it.
      heights_delta = tf.minimum(heights_delta, 0.0)
    heights += heights_delta
  else:
    # No need to do anything with heights if there are no output_max
    # constraints.
    if output_min_constraints == bct.CLAMPED:
      bias = tf.constant(output_min, shape=bias.shape, dtype=bias.dtype)
    elif output_min_constraints == bct.BOUND:
      bias = tf.maximum(bias, output_min)

  return bias, heights


def _project_convexity(heights, lengths, convexity, constraint_group):
  """Convexity projection for given 'constraint_group'.

  Since an exact single step projection is not possible for convexity
  constraints, we break the constraints into two independent groups and apply
  Dykstra's alternating projections algorithm. Each group consists of a list of
  pairs where each pair represents constraints on 2 consequtive heights.

  Groups:

  ```
  g0 = [(h0, h1), (h2, h3), (h4, h5), ...]
  g1 = [(h1, h2), (h3, h4), (h5, h6), ...]
  ```

  We know how to project single pair of adjacent heights:
  h0_prime = min/max(h0, (l0 / (l0 + l1)) * (h0 + h1))
  h1_prime = min/max(h1, (l1 / (l0 + l1)) * (h0 + h1))
  where l0 and l1 stand for lengths of segment which correspond to h0 and h1 and
  choise of min or max functions depends on convexity direction.

  We can see that all pairs within same group are independent so we know how to
  project such group of constraints in single pass.

  This function breaks heights and their lengths into given constraint group
  and does projection for this group.

  Args:
    heights: `(num_heights, units)`-shape tensor which represents heights.
    lengths: `(num_heights)`-shape tensor which represents lengths of segments
      which correspond to heights.
    convexity: -1 or 1 where 1 stands for convex function and -1 for concave.
    constraint_group: 0 or 1 which represent group from description above.

  Returns:
    Projected heights for given constraint group.
  """
  verify_hyperparameters(
      convexity=convexity,
      lengths=lengths,
      weights_shape=[heights.shape[0] + 1, heights.shape[1]])
  if constraint_group not in [0, 1]:
    raise ValueError("constraint_group must be one of: [0, 1]. "
                     "Given: %s" % constraint_group)

  if convexity == 0 or heights.shape[0] == 1:
    return heights

  num_heights = heights.shape.dims[0].value
  # To avoid broadcasting when performing math ops with 'heights'.
  lengths = tf.reshape(lengths, shape=(-1, 1))

  # Split heigths and lengths into pairs which correspond to given constraint
  # group. In order to do this we need to split heights into odd and even. We
  # can possibly omit last element of larger set to ensure that both sets have
  # same number of elements.
  num_0 = (num_heights - constraint_group + 1) // 2
  num_1 = (num_heights - constraint_group) // 2
  if num_1 == num_0:
    last_index = None
  else:
    last_index = -1
  heights_0 = heights[constraint_group:last_index:2]
  lengths_0 = lengths[constraint_group:last_index:2]
  heights_1 = heights[constraint_group + 1::2]
  lengths_1 = lengths[constraint_group + 1::2]

  # h0_prime = (l0 / (l0 + l1)) * (h0 + h1) = l0 * base
  # h1_prime = (l1 / (l0 + l1)) * (h0 + h1) = l1 * base
  base = (heights_0 + heights_1) / (lengths_0 + lengths_1)
  heights_0_prime = lengths_0 * base
  heights_1_prime = lengths_1 * base
  if convexity == 1:
    heights_0 = tf.minimum(heights_0, heights_0_prime)
    heights_1 = tf.maximum(heights_1, heights_1_prime)
  else:
    heights_0 = tf.maximum(heights_0, heights_0_prime)
    heights_1 = tf.minimum(heights_1, heights_1_prime)

  # Now we need to merge heights in such way that elements from 'heights_0' and
  # 'heights_1' alternate:
  # merged = [heights_0[0], heights_1[0], heights_0[1], heights_1[1], ...]
  # Achieve this by concatenating along axis=1 so after concatenation elements
  # from 'heights_0' and 'heights_1' will alternate in memory and reshape will
  # give us desired result.
  projected_heights = tf.reshape(
      tf.concat([heights_0, heights_1], axis=1), shape=[-1, heights.shape[1]])

  weights_pieces = [projected_heights]
  if constraint_group == 1:
    # First height was skipped during initial split.
    weights_pieces = [heights[0:1]] + weights_pieces
  if last_index == -1:
    # Last height was skipped during initial split.
    weights_pieces.append(heights[-1:])

  if len(weights_pieces) == 1:
    return weights_pieces[0]
  else:
    return tf.concat(weights_pieces, axis=0)


def _project_monotonicity(heights, monotonicity):
  """Projects into monotonic function."""
  if monotonicity == 0:
    return heights
  elif monotonicity == 1:
    return tf.maximum(heights, 0.0)
  else:
    return tf.minimum(heights, 0.0)


def project_all_constraints(weights,
                            monotonicity,
                            output_min,
                            output_max,
                            output_min_constraints,
                            output_max_constraints,
                            convexity,
                            lengths,
                            num_projection_iterations=8):
  """Jointly projects into all supported constraints.

  For all combinations of constraints except the case where bounds constraints
  are specified without monotonicity constraints we properly project into
  nearest point with respect to L2 norm. For latter case we use a heuristic to
  map input point into some feasible point with no guarantees on how close this
  point is to the true projection.

  If only bounds or only monotonicity constraints are specified there will be a
  single step projection. For all other combinations of constraints we use
  num_projection_iterations iterations of Dykstra's alternating projection
  algorithm to jointly project onto all the given constraints. Dykstra's
  algorithm gives us proper projection with respect to L2 norm but approaches it
  from "wrong" side. That's why in order to ensure that constraints are strictly
  met we'll do approximate projections in the end which project strictly into
  feasible space, but it's not an exact projection with respect to the L2 norm.
  With enough iterations of the Dykstra's algorithm, the impact of such
  approximate projection should be negligible.

  With bound and convexity constraints and no specified monotonicity, this
  method does not fully satisfy the constrains. Increasing the number of
  iterations can reduce the constraint violation in such cases.

  Args:
    weights: `(num_keypoints, units)`-shape tensor which represents weights of
      PWL calibration layer.
    monotonicity: 1 for increasing, -1 for decreasing, 0 for no monotonicity
      constraints.
    output_min: Lower bound constraint of PWL calibration layer.
    output_max: Upper bound constraint of PWL calibration layer.
    output_min_constraints: A `tfl.pwl_calibration_lib.BoundConstraintsType`
      describing the constraints on the layer's minimum value.
    output_max_constraints: A `tfl.pwl_calibration_lib.BoundConstraintsType`
      describing the constraints on the layer's maximum value.
    convexity: 1 for convex, -1 for concave, 0 for no convexity constraints.
    lengths: Lengths of pieces of piecewise linear function. Needed only if
      convexity projection is specified.
    num_projection_iterations: Number of iterations of Dykstra's alternating
      projection algorithm.

  Returns:
    Projected weights tensor.
  """
  bias = weights[0:1]
  heights = weights[1:]

  def body(projection_counter, bias, heights, last_bias_change,
           last_heights_change):
    """The body of tf.while_loop implementing a step of Dykstra's projection.

    Args:
      projection_counter: The counter tensor or number at the beginning of the
        iteration.
      bias: Bias tensor at the beginning of the iteration.
      heights: Heights tensor at the beginning of the iteration.
      last_bias_change: Dict that stores the last change in the bias after
        projecting onto each subset of constraints.
      last_heights_change: Dict that stores the last change in the heights after
        projecting onto each subset of constraints.

    Returns:
      The tuple `(num_projection_counter, bias, heights, last_bias_change,
      last_heights_change)` at the end of the iteration.
    """
    last_bias_change = copy.copy(last_bias_change)
    last_heights_change = copy.copy(last_heights_change)
    num_projections = 0
    # ******************** BOUNDS *********************
    bct = BoundConstraintsType
    if output_min_constraints != bct.NONE or output_max_constraints != bct.NONE:
      rolled_back_bias = bias - last_bias_change["BOUNDS"]
      rolled_back_heights = heights - last_heights_change["BOUNDS"]
      if monotonicity != 0:
        bias, heights = _project_bounds_considering_monotonicity(
            bias=rolled_back_bias,
            heights=rolled_back_heights,
            monotonicity=monotonicity,
            output_min=output_min,
            output_max=output_max,
            output_min_constraints=output_min_constraints,
            output_max_constraints=output_max_constraints)
      else:
        bias, heights = _approximately_project_bounds_only(
            bias=rolled_back_bias,
            heights=rolled_back_heights,
            output_min=output_min,
            output_max=output_max,
            output_min_constraints=output_min_constraints,
            output_max_constraints=output_max_constraints)
      last_bias_change["BOUNDS"] = bias - rolled_back_bias
      last_heights_change["BOUNDS"] = heights - rolled_back_heights
      num_projections += 1

    # ******************** MONOTONICITY *********************
    if monotonicity != 0:
      rolled_back_heights = heights - last_heights_change["MONOTONICITY"]
      heights = _project_monotonicity(
          heights=rolled_back_heights, monotonicity=monotonicity)
      last_heights_change["MONOTONICITY"] = heights - rolled_back_heights
      num_projections += 1

    # ******************** CONVEXITY *********************
    if convexity != 0:
      if heights.shape[0] >= 2:
        rolled_back_heights = heights - last_heights_change["CONVEXITY_0"]
        heights = _project_convexity(
            heights=rolled_back_heights,
            lengths=lengths,
            convexity=convexity,
            constraint_group=0)
        last_heights_change["CONVEXITY_0"] = heights - rolled_back_heights
        num_projections += 1
      if heights.shape[0] >= 3:
        rolled_back_heights = heights - last_heights_change["CONVEXITY_1"]
        heights = _project_convexity(
            heights=rolled_back_heights,
            lengths=lengths,
            convexity=convexity,
            constraint_group=1)
        last_heights_change["CONVEXITY_1"] = heights - rolled_back_heights
        num_projections += 1

    return (projection_counter + num_projections, bias, heights,
            last_bias_change, last_heights_change)

  # Call the body of the loop once to see if Dykstra's is needed.
  # If there is only one set of projections, apply it without a loop.
  # Running the body of the loop also finds the required last_bias_change
  # and last_heights_change keys. The set of keys in the input and output of the
  # body of tf.while_loop must be the same across iterations.
  zero_bias = tf.zeros_like(bias)
  zero_heights = tf.zeros_like(heights)
  last_bias_change = collections.defaultdict(lambda: zero_bias)
  last_heights_change = collections.defaultdict(lambda: zero_heights)
  (num_projections, projected_bias, projected_heights, last_bias_change,
   last_heights_change) = body(0, bias, heights, last_bias_change,
                               last_heights_change)
  if num_projections <= 1:
    return tf.concat([projected_bias, projected_heights], axis=0)

  def cond(projection_counter, bias, heights, last_bias_change,
           last_heights_change):
    del bias, heights, last_bias_change, last_heights_change
    return tf.less(projection_counter,
                   num_projection_iterations * num_projections)

  # Apply Dykstra's algorithm with tf.while_loop.
  projection_counter = tf.constant(0)
  last_bias_change = {k: zero_bias for k in last_bias_change}
  last_heights_change = {k: zero_heights for k in last_heights_change}
  (_, bias, heights, _,
   _) = tf.while_loop(cond, body, (projection_counter, bias, heights,
                                   last_bias_change, last_heights_change))

  # Since Dykstra's algorithm is iterative in order to strictly meet constraints
  # we use approximate projection algorithm to finalize them.
  return _finalize_constraints(
      bias=bias,
      heights=heights,
      monotonicity=monotonicity,
      output_min=output_min,
      output_max=output_max,
      output_min_constraints=output_min_constraints,
      output_max_constraints=output_max_constraints,
      convexity=convexity,
      lengths=lengths)


def _squeeze_by_scaling(bias, heights, monotonicity, output_min, output_max,
                        output_min_constraints, output_max_constraints):
  """Squeezes monotonic calibrators by scaling in order to meet bounds.

  Projection by scaling is not exact with respect to the L2 norm, but maintains
  convexity unlike projection by shift.

  Args:
    bias: `(1, units)`-shape tensor which represents bias.
    heights: `(num_heights, units)`-shape tensor which represents heights.
    monotonicity: 1 for increasing, -1 for decreasing.
    output_min: Lower bound constraint of PWL calibration layer.
    output_max: Upper bound constraint of PWL calibration layer.
    output_min_constraints: A `tfl.pwl_calibration_lib.BoundConstraintsType`
      describing the constraints on the layer's minimum value.
    output_max_constraints: A `tfl.pwl_calibration_lib.BoundConstraintsType`
      describing the constraints on the layer's maximum value.

  Returns:
    Projected bias and heights.
  """
  if monotonicity == -1:
    if output_min_constraints == BoundConstraintsType.NONE:
      return bias, heights
    # Reduce computation of projection of decreasing function to computation of
    # projection of increasing function by multiplying everything by -1 and
    # swapping maximums and minimums.
    bias, heights = _squeeze_by_scaling(
        bias=-bias,
        heights=-heights,
        monotonicity=1,
        output_min=None if output_max is None else -output_max,
        output_max=None if output_min is None else -output_min,
        output_min_constraints=output_max_constraints,
        output_max_constraints=output_min_constraints)
    return -bias, -heights
  if output_max_constraints == BoundConstraintsType.NONE:
    return bias, heights

  delta = output_max - bias
  # For better stability use tf.where rather than the more standard approach:
  # heights *= tf.reduce_sum(heights) / max(delta, eps)
  # in order to keep everything strictly unchanged for small deltas, rather than
  # increase heights by factor 1/eps and still don't meet constraints.
  scaling_factor = tf.where(delta > 0.001,
                            tf.reduce_sum(heights, axis=0) / delta,
                            tf.ones_like(delta))
  heights = heights / tf.maximum(scaling_factor, 1.0)
  return bias, heights


def _approximately_project_convexity(heights, lengths, convexity):
  """Strictly projects convexity, but is not exact with respect to the L2 norm.

  Projects by iterating over pieces of piecewise linear function left to right
  and aligning current slope with previous one if it violates convexity.

  Args:
    heights: `(num_heights, units)`-shape tensor which represents heights.
    lengths: `(num_heights)`-shape tensor which represents lengths of segments
      which correspond to heights.
    convexity: -1 or 1 where 1 stands for convex function and -1 for concave.

  Returns:
    Projected heights.
  """
  if convexity == 0:
    return heights
  heights = tf.unstack(heights, axis=0)
  lengths = tf.unstack(lengths, axis=0)
  for i in range(1, len(heights)):
    temp = heights[i - 1] * (lengths[i] / lengths[i - 1])
    if convexity == 1:
      heights[i] = tf.maximum(heights[i], temp)
    else:
      heights[i] = tf.minimum(heights[i], temp)

  return tf.stack(heights, axis=0)


def _finalize_constraints(bias, heights, monotonicity, output_min, output_max,
                          output_min_constraints, output_max_constraints,
                          convexity, lengths):
  """Strictly projects onto the given constraint, approximate w.r.t the L2 norm.

  Dykstra's algorithm gives us proper projection with respect to L2 norm but
  approaches it from "wrong" side. In order to ensure that constraints are
  strictly met we'll do approximate projections in the end which project
  strictly into feasible space, but it's not an exact projection with respect to
  the L2 norm. With enough iterations of the Dykstra's algorithm, the impact of
  such approximate projection should be negligible.

  With bound and convexity constraints and no specified monotonicity, this
  method does not fully satisfy the constrains. Increasing the number of
  iterations can reduce the constraint violation in such cases. Fortunately it
  does not seem to be common config.

  Args:
    bias: `(1, units)`-shape tensor which represents bias.
    heights: `(num_heights, units)`-shape tensor which represents heights.
    monotonicity: 1 for increasing, -1 for decreasing, 0 for no monotonicity
      constraints.
    output_min: Lower bound constraint of PWL calibration layer.
    output_max: Upper bound constraint of PWL calibration layer.
    output_min_constraints: A `tfl.pwl_calibration_lib.BoundConstraintsType`
      describing the constraints on the layer's minimum value.
    output_max_constraints: A `tfl.pwl_calibration_lib.BoundConstraintsType`
      describing the constraints on the layer's maximum value.
    convexity: 1 for convex, -1 for concave, 0 for no convexity constraints.
    lengths: Lengths of pieces of piecewise linear function. Needed only if
      convexity projection is specified.

  Returns:
    Projected weights tensor.
  """
  # Convexity and monotonicity projections don't violate each other, but both
  # might lead to bounds violation, so do them first and fix bounds after.
  if monotonicity != 0:
    heights = _project_monotonicity(heights=heights, monotonicity=monotonicity)
  if convexity != 0:
    heights = _approximately_project_convexity(
        heights=heights, lengths=lengths, convexity=convexity)

  bct = BoundConstraintsType
  if output_min_constraints != bct.NONE or output_max_constraints != bct.NONE:
    if monotonicity != 0 and convexity != 0:
      # Both monotonicity and convexity projection can only increase upper bound
      # so we only need to take care of decreasing it back.
      bias, heights = _squeeze_by_scaling(
          bias=bias,
          heights=heights,
          monotonicity=monotonicity,
          output_min=output_min,
          output_max=output_max,
          output_min_constraints=output_min_constraints,
          output_max_constraints=output_max_constraints)
    else:
      # This bounds projection might violate convexity. Unfortunately bounds
      # projections with convexity and without monotonicity are are difficult to
      # achieve strictly and might be violated. so ignore this for now. In order
      # to minimize projection error consider increasing
      # num_projection_iterations.
      if output_min_constraints == bct.CLAMPED:
        output_min_constraints = bct.BOUND
      if output_max_constraints == bct.CLAMPED:
        output_max_constraints = bct.BOUND
      bias, heights = _approximately_project_bounds_only(
          bias=bias,
          heights=heights,
          output_min=output_min,
          output_max=output_max,
          output_min_constraints=output_min_constraints,
          output_max_constraints=output_max_constraints)
  return tf.concat([bias, heights], axis=0)


def assert_constraints(outputs,
                       monotonicity,
                       output_min,
                       output_max,
                       clamp_min=False,
                       clamp_max=False,
                       debug_tensors=None,
                       eps=1e-6):
  """Asserts that 'outputs' satisfiy constraints.

  Args:
    outputs: Tensor of shape `(num_output_values, units)` which represents
      outputs of pwl calibration layer which will be tested against the given
      constraints. If monotonicity is specified these outputs must be for
      consequtive inputs.
    monotonicity: One of {-1, 0, 1}. -1 for decreasing, 1 for increasing 0 means
      no monotonicity checks.
    output_min: Lower bound or None.
    output_max: Upper bound or None.
    clamp_min: Whether one of outputs must match output_min.
    clamp_max: Whther one of outputs must match output_max.
    debug_tensors: None or list of anything convertible to tensor (for example
      tensors or strings) which will be printed in case of constraints
      violation.
    eps: Allowed constraints violation.

  Raises:
    ValueError: If monotonicity is not one of {-1, 0, 1}

  Returns:
    List of assertion ops in graph mode or immideately asserts in eager mode.
  """

  info = ["Outputs: ", outputs, "Epsilon: ", eps]
  if debug_tensors:
    info += debug_tensors
  asserts = []

  if output_min is not None:
    min_output = tf.reduce_min(outputs, axis=0)
    if clamp_min:
      asserts.append(
          tf.Assert(
              tf.reduce_all(tf.abs(min_output - output_min) <= eps),
              data=["Clamp_min violation.", "output_min:", output_min] + info,
              summarize=outputs.shape[0]))
    else:
      asserts.append(
          tf.Assert(
              tf.reduce_all(min_output >= output_min - eps),
              data=["Lower bound violation.", "output_min:", output_min] + info,
              summarize=outputs.shape[0]))

  if output_max is not None:
    max_output = tf.reduce_max(outputs, axis=0)
    if clamp_max:
      asserts.append(
          tf.Assert(
              tf.reduce_all(tf.abs(max_output - output_max) <= eps),
              data=["Clamp_max violation.", "output_max:", output_max] + info,
              summarize=outputs.shape[0]))
    else:
      asserts.append(
          tf.Assert(
              tf.reduce_all(max_output <= output_max + eps),
              data=["Upper bound violation.", "output_max:", output_max] + info,
              summarize=outputs.shape[0]))

  if monotonicity not in [-1, 0, 1]:
    raise ValueError("'monotonicity' must be one of: [-1, 0, 1]. It is: %s" %
                     monotonicity)
  if monotonicity != 0:
    diffs = (outputs[1:] - outputs[0:-1])
    asserts.append(
        tf.Assert(
            tf.reduce_min(diffs * monotonicity) >= -eps,
            data=["Monotonicity violation.", "monotonicity:", monotonicity] +
            info,
            summarize=outputs.shape[0]))

  return asserts


def verify_hyperparameters(input_keypoints=None,
                           output_min=None,
                           output_max=None,
                           monotonicity=None,
                           convexity=None,
                           is_cyclic=False,
                           lengths=None,
                           weights_shape=None,
                           input_keypoints_type=None):
  """Verifies that all given hyperparameters are consistent.

  See PWLCalibration class level comment for detailed description of arguments.

  Args:
    input_keypoints: `input_keypoints` of PWLCalibration layer.
    output_min: Smallest output of PWLCalibration layer.
    output_max: Largest output of PWLCalibration layer.
    monotonicity: `monotonicity` hyperparameter of PWLCalibration layer.
    convexity: `convexity` hyperparameter of PWLCalibration layer.
    is_cyclic: `is_cyclic` hyperparameter of PWLCalibration layer.
    lengths: Lengths of pieces of piecewise linear function.
    weights_shape: Shape of weights of PWLCalibration layer.
    input_keypoints_type: The type of input keypoints of a PWLCalibration layer.

  Raises:
    ValueError: If something is inconsistent.
  """
  if input_keypoints is not None:
    if tf.is_tensor(input_keypoints):
      if len(input_keypoints.shape) != 1 or input_keypoints.shape[0] < 2:
        raise ValueError("Input keypoints must be rank-1 tensor of size at "
                         "least 2. It is: " + str(input_keypoints))
    else:
      if len(input_keypoints) < 2:
        raise ValueError("At least 2 input keypoints must be provided. "
                         "Given: " + str(input_keypoints))
      if not all(input_keypoints[i] < input_keypoints[i + 1]
                 for i in range(len(input_keypoints) - 1)):
        raise ValueError("Keypoints must be strictly increasing. They are: " +
                         str(input_keypoints))

  if output_min is not None and output_max is not None:
    if output_max < output_min:
      raise ValueError("If specified output_max must be greater than "
                       "output_min. "
                       "They are: ({}, {})".format(output_min, output_max))

  # It also raises errors if monotonicities specified incorrectly.
  monotonicity = utils.canonicalize_monotonicity(monotonicity)
  convexity = utils.canonicalize_convexity(convexity)

  if is_cyclic and (monotonicity or convexity):
    raise ValueError("'is_cyclic' can not be specified together with "
                     "'monotonicity'({}) or 'convexity'({}).".format(
                         monotonicity, convexity))

  if weights_shape is not None:
    if len(weights_shape) != 2 or weights_shape[0] < 2:
      raise ValueError("PWLCalibrator weights must have shape: [k, units] where"
                       " k > 1. It is: " + str(weights_shape))

  if lengths is not None and weights_shape is not None:
    if tf.is_tensor(lengths):
      num_lengths = lengths.shape[0]
    else:
      num_lengths = len(lengths)
    if num_lengths + 1 != weights_shape[0]:
      raise ValueError("Number of lengths must be equal to number of weights "
                       "minus one. Lengths: %s, weights_shape: %s" %
                       (lengths, weights_shape))

  if (input_keypoints_type is not None and input_keypoints_type != "fixed" and
      input_keypoints_type != "learned_interior"):
    raise ValueError(
        "input_keypoints_type must be one of 'fixed' or 'learned_interior': %s"
        % input_keypoints_type)
