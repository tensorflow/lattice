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

"""Helpers to train simple model for tests and print debug output."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl import logging
import numpy as np


class TimeTracker(object):
  """Tracks time.

  Keeps track of time spent in its scope and appends it to 'list_to_append'
  on exit from scope divided by 'num_steps' if provided.

  Example:
    training_step_times = []
    with TimeTracker(training_step_times, num_steps=num_epochs):
      model.fit(... epochs=num_epochs ...)
    print np.median(training_step_times)
  """

  def __init__(self, list_to_append, num_steps=1):
    self._list_to_append = list_to_append
    self._num_steps = float(num_steps)

  def __enter__(self):
    self._start_time = time.time()
    return self

  def __exit__(self, unuesd_type, unuesd_value, unuesd_traceback):
    duration = time.time() - self._start_time
    self._list_to_append.append(
        duration / self._num_steps if self._num_steps else 0.0)


def run_training_loop(config,
                      training_data,
                      keras_model,
                      input_dtype=np.float32,
                      label_dtype=np.float32):
  """Trains models and prints debug info.

  Args:
    config: dictionary of test case parameters. See tests for TensorFlow Lattice
      layers.
    training_data: tuple: (training_inputs, labels) where
      training_inputs and labels are proper data to train models passed via
      other parameters.
    keras_model: Keras model to train on training_data.
    input_dtype: dtype for input conversion.
    label_dtype: dtype for label conversion.

  Returns:
    Loss measured on training data and tf.session() if one was initialized
    explicitly during training.
  """
  (training_inputs, training_labels) = training_data
  np_training_inputs = np.asarray(training_inputs).astype(input_dtype)
  np_training_labels = np.asarray(training_labels).astype(label_dtype)

  logging.info(" {0: <10}{1: <10}".format("it", "Loss"))

  num_steps = 10
  training_step_times = []
  for step in range(num_steps):
    begin = (config["num_training_epoch"] * step) // num_steps
    end = (config["num_training_epoch"] * (step + 1)) // num_steps
    num_epochs = end - begin
    if num_epochs == 0:
      continue

    loss = keras_model.evaluate(np_training_inputs, np_training_labels,
                                batch_size=len(np_training_inputs),
                                verbose=0)
    with TimeTracker(training_step_times, num_steps=num_epochs):
      keras_model.fit(np_training_inputs, np_training_labels,
                      batch_size=len(np_training_inputs),
                      epochs=num_epochs,
                      verbose=0)
    logging.info("{0: <10}{1: <10,.6f}".format(begin, loss))
  # End of: 'for step in range(num_steps):'

  loss = keras_model.evaluate(np_training_inputs, np_training_labels,
                              batch_size=len(np_training_inputs),
                              verbose=0)
  logging.info("Final loss: %f", loss)

  if training_step_times:
    logging.info("Median training step time: %f",
                 np.median(training_step_times))

  return loss


def two_dim_mesh_grid(num_points, x_min, y_min, x_max, y_max):
  """Generates uniform 2-d mesh grid for 3-d surfaces visualisation via pyplot.

  Uniformly distributes 'num_points' within rectangle:
  (x_min, y_min) - (x_max, y_max)
  'num_points' should be such that uniform distribution is possible. In other
  words there should exist such integers 'x_points' and 'y_points' that:
  - x_points * y_points == num_points
  - x_points / y_points == (x_max - x_min) / (y_max - y_min)

  Args:
    num_points: number of points in the grid.
    x_min: bounds of the grid.
    y_min: bounds of the grid.
    x_max: bounds of the grid.
    y_max: bounds of the grid.

  Returns:
    Tuple containing 2 numpy arrays which represent X and Y coordinates of mesh
    grid

  Raises:
    ValueError: if it's impossible to uniformly distribute 'num_points' across
    specified grid.

  """
  x_size = x_max - x_min
  y_size = y_max - y_min
  x_points = (num_points * x_size / y_size)**0.5
  y_points = num_points / x_points

  eps = 1e-7
  is_int = lambda x: abs(x - int(x + eps)) < eps
  if not is_int(x_points) or not is_int(y_points):
    raise ValueError("Cannot evenly distribute %d points across sides of "
                     "lengths: %f and %f" % (num_points, x_size, y_size))

  x_grid = np.linspace(start=x_min, stop=x_max, num=int(x_points + eps))
  y_grid = np.linspace(start=y_min, stop=y_max, num=int(y_points + eps))

  # Convert list returned by meshgrid() to tuple so we can easily distinguish
  # mesh grid vs list of points.
  return tuple(np.meshgrid(x_grid, y_grid))


def sample_uniformly(num_points, lower_bounds, upper_bounds):
  """Deterministically generates num_point random points within bounds.

  Points will be such that:
  lower_bounds[i] <= p[i] <= upper_bounds[i]

  Number of dimensions is defined by lengths of lower_bounds list.

  Args:
    num_points: number of points to generate.
    lower_bounds: list or tuple of lower bounds.
    upper_bounds: list or tuple of upper bounds.

  Returns:
    List of generated points.
  """
  if len(lower_bounds) != len(upper_bounds):
    raise ValueError("Lower and upper bounds must have same length. They are: "
                     "lower_bounds: %s, upper_bounds: %s" %
                     (lower_bounds, upper_bounds))
  np.random.seed(41)
  x = []
  for _ in range(num_points):
    point = [
        lower + np.random.random() * (upper - lower)
        for lower, upper in zip(lower_bounds, upper_bounds)
    ]
    x.append(np.asarray(point))
  return x


def get_hypercube_interpolation_fn(coefficients):
  """Returns function which does hypercube interpolation.

  This is only for 2^d lattice aka hypercube.

  Args:
    coefficients: coefficients of hypercube ordered according to index of
      corresponding vertex.

  Returns:
    Function which takes d-dimension point and performs hypercube interpolation
    with given coefficients.
  """

  def hypercube_interpolation_fn(x):
    """Does hypercube interpolation."""
    if 2**len(x) != len(coefficients):
      raise ValueError("Number of coefficients(%d) does not correspond to "
                       "dimension 'x'(%s)" % (len(coefficients), x))
    result = 0.0
    for coefficient_index in range(len(coefficients)):
      weight = 1.0
      for input_dimension in range(len(x)):
        if coefficient_index & (1 << input_dimension):
          # If statement checks whether 'input_dimension' bit of
          # 'coefficient_index' is set to 1.
          weight *= x[input_dimension]
        else:
          weight *= (1.0 - x[input_dimension])
      result += coefficients[coefficient_index] * weight
    return result

  return hypercube_interpolation_fn


def get_linear_lattice_interpolation_fn(lattice_sizes, monotonicities,
                                        output_min, output_max):
  """Returns function which does lattice interpolation.

  Returned function matches lattice_layer.LinearInitializer with corresponding
  parameters.

  Args:
    lattice_sizes: list or tuple of integers which represents lattice sizes.
    monotonicities: monotonicity constraints.
    output_min: minimum output of linear function.
    output_max: maximum output of linear function.

  Returns:
    Function which takes d-dimension point and performs lattice interpolation
    assuming lattice weights are such that lattice represents linear function
    with given output_min and output_max. All monotonic dimesions of this linear
    function cotribute with same weight despite of numer of vertices per
    dimension. All non monotonic dimensions have weight 0.0.
  """

  def linear_interpolation_fn(x):
    """Linear along monotonic dims and 0.0 along non monotonic."""
    result = output_min
    num_monotonic_dims = len(monotonicities) - monotonicities.count(0)
    if num_monotonic_dims == 0:
      local_monotonicities = [1] * len(lattice_sizes)
      num_monotonic_dims = len(lattice_sizes)
    else:
      local_monotonicities = monotonicities

    weight = (output_max - output_min) / num_monotonic_dims
    for i in range(len(x)):
      if local_monotonicities[i]:
        result += x[i] * weight / (lattice_sizes[i] - 1.0)
    return result

  return linear_interpolation_fn
