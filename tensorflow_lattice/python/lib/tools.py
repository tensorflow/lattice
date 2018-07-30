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
"""Library of internal functions used by TensorFlow Lattice modules."""

import datetime
import os
import socket
import time
import traceback

from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

# Name used as a default for all per-feature configurations, see
# cast_to_dict.
DEFAULT_NAME = 'tensorflow_lattice_internal_default'


def cast_to_scalar_tensor_of_dtype(t, dtype):
  """If not yet a tensor, casts it to a constant scalar tensor."""
  if issubclass(type(t), ops.Tensor):
    return t
  return array_ops.constant(t, shape=[], dtype=dtype)


def cast_to_list(v, n, param_name):
  if isinstance(v, list):
    if len(v) != n:
      raise ValueError('List given to %s has %d values, but we need %d' %
                       (param_name, len(v), n))
    return v
  return [v] * n


def cast_to_dict(v, feature_names, param_name):
  """If value not yet a dict, cast it to a dict of all feature names to values.

  Args:
    v: can be a dict or a value. If a dict, missing feature names are
      set to the value of v[DEFAULT_NAME] -- an exception is raised if
      some feature name is not found v[DEFAULT_NAME] is not set.
    feature_names: list of feature names that must be present in the returned
      dict.
    param_name: name shown in case of error, if value is not set for some
      feature.

  Returns:
    If value is not a dict, a new dict with the same value repeated for all
    feature names.

    If value is a dict, returns a new dict with the values copied, or if not
    present, copied from v[DEFAULT_NAME].

  Raises:
    ValueError: if a value is not set for a feature in feature_names, and no
    default value is set.
  """
  if isinstance(v, dict):
    v_copy = {}
    for feature_name in feature_names:
      if feature_name in v:
        v_copy[feature_name] = v[feature_name]
      else:
        if DEFAULT_NAME in v:
          v_copy[feature_name] = v[DEFAULT_NAME]
        else:
          raise ValueError(
              'Dict given for %s does not contain definition for feature '
              '"%s"' % (param_name, feature_name))
    return v_copy
  return {feature_name: v for feature_name in feature_names}


def cast_to_dict_of_tensor_scalars(v, feature_names, dtype, param_name):
  """Cast value to a dict mapping feature names to tensor scalars."""
  if isinstance(v, dict):
    # Convert each value to scalar.
    res = {}
    for feature_name in feature_names:
      if feature_name in v:
        res[feature_name] = cast_to_scalar_tensor_of_dtype(
            v[feature_name], dtype)
      else:
        if DEFAULT_NAME in v:
          res[feature_name] = cast_to_scalar_tensor_of_dtype(
              v[DEFAULT_NAME], dtype)
        else:
          raise ValueError(
              'Dict given for %s does not contain definition for feature '
              '"%s"' % (param_name, feature_name))
    return res

  v = cast_to_scalar_tensor_of_dtype(v, dtype)
  return {feature_name: v for feature_name in feature_names}


def input_from_feature_column(columns_to_tensors,
                              feature_column,
                              dtype=dtypes.float32):
  """Convert one feature_column to `Tensor`, making necessary transformations.

  DenseColumns are taken as is, see  `tf.feature_column.input_layer`.
  CategoricalColumns are assumed to be exclusive and it takes only the value
  of the category.

  Args:
    columns_to_tensors: Returned by input_fn. Consider processing first by
       `layers.transform_features(columns_to_tensors, feature_columns))`, since
       it may share tf ops for different FeatureColumns. This function
       transforms one at a time.
    feature_column: feature_column to transform to `Tensor`.
    dtype: `_CategoricalColumn`s are converted to this type.

  Returns:
    Tensor with transformed feature column for calibration consumption.

  Raises:
    ValueError: if type of FeatureColumn is unknown, and this function doesn't
      know how to handle it.
  """
  # pylint: disable=protected-access
  if isinstance(feature_column, feature_column_lib._DenseColumn):
    return feature_column_lib.input_layer(
        features=columns_to_tensors, feature_columns=set([feature_column]))
  elif isinstance(feature_column, feature_column_lib._CategoricalColumn):
    categorical_ids = math_ops.cast(
        feature_column._transform_feature(columns_to_tensors).values, dtype)
    return array_ops.stack([categorical_ids], axis=1)
  # pylint: enable=protected-access
  raise ValueError('Cannot handle FeatureColumn {}: only _DenseColumn and '
                   '_CategoricalColumn are implemented, consider converting '
                   'your column to float32 until this FeatureColumn is '
                   'supported'.format(feature_column))

def get_sorted_feature_columns(feature_columns):
  """Sorts an iterable of feature columns by their names in ascending order.

  Args:
    feature_columns: An iterable that yields instances of a tensorflow
      FeatureColumn.
  Returns:
    A copy of the input sorted by name in ascending order.
  """
  return sorted(feature_columns, key=lambda fc : fc.name)

def get_sorted_feature_names(columns_to_tensors, feature_columns=None):
  """List feature names: from feature_columns or columns_to_tensors.

  This function will return the list of feature names for layers or Estimators
  that use either feature_columns or directly the inputs returned by an
  input_fn.

  Args:
    columns_to_tensors: str-->tf.Tensor dict. A mapping from feature name to
      tensors.
    feature_columns: Optional set containing all the feature columns. If not
      set it is assumed all features come from columns_to_tensors. Otherwise
      this defines the list of features to use.
      All items in the set should be instances of classes derived by
      FeatureColumn.

  Returns:
    List of feature names.
  """
  if feature_columns is not None:
    return [f_col.name for f_col in get_sorted_feature_columns(feature_columns)]
  return [k for k in sorted(columns_to_tensors.keys())]

def assert_shape(tensor, expected_shape, tensor_name):
  """Assert shapes that must be known in graph construction time."""
  got_shape = tensor.shape.as_list()
  if got_shape != expected_shape:
    raise ValueError('Invalid shape for %s: got %s, expected %s' %
                     (tensor_name, got_shape, expected_shape))


def add_if_not_none(a, b):
  """Returns a/b is one of them is None, or their sum if both are not None."""
  if a is None:
    return b
  if b is None:
    return a
  return a + b



class LatticeStructure(object):
  """Lattice structure class.

  This class represents lattice vertices in a column-major order indexing that
  are used in C++ lattice operators.

  With the column-major indexing, the lattice with lattice_sizes
  [m_0, m_1, ..., m_{n-1}] will have:
  dimension: n
  number of vertices: m_0 * ... * m_{n-1}
  number of vertices in each cell: 2 ** (n-1)
  stride[0] = 1
  stride[1] = 1 * m_{0}
       ...
  stride[n-1] = 1 * m_{n-2} ... * m_0

  LatticeStructure keeps a reference copy of lattice_sizes, so if any of element
  in lattice_sizes changes, this structure's output is not useful anymore.

  """

  def __init__(self, lattice_sizes):
    """Initialize lattice structure.

    Args:
      lattice_sizes: (list of ints) constains lattice size of each dimension.

    Raises:
      ValueError: If any element of lattice_sizes is less than 2.
    """

    # This is a reference copy.
    self.lattice_sizes = lattice_sizes
    self.dimension = len(lattice_sizes)
    self.num_vertices_per_cell = 2**self.dimension
    self.num_vertices = 1
    self.strides = []
    for lattice_size in lattice_sizes:
      self.strides.append(self.num_vertices)
      if lattice_size < 2:
        raise ValueError(
            'Lattice size cannot be less than 2, but one (or more) of '
            'lattice_size is less than 2', lattice_sizes)
      self.num_vertices *= lattice_size


def lattice_indices_generator(lattice_structure):
  """lattice_indices_generator iterators all vertices in a multi-cell lattice.

  It returns a global index and per-dimension index. So a lattice of sizes
  [2,3] would yield the sequence:

     (0, [0, 0])
     (1, [1, 0])
     (2, [0, 1])
     (3, [1, 1])
     (4, [0, 2])
     (5, [1, 2])

  The access order is in the column-major order, that is consistent with C++
  lattice operators indexing convention.

  Example usage:
    for (index, per_dim_index) in lattice_indices_generator(lattice structure):
      flat_index = index
      first_dim_index = per_dim_index[0]

  Args:
    lattice_structure: (LatticeStructure) lattice structure to be used.
  Yields:
    (flat_index, per_dim_indices)
  """
  per_dim_indices = [0] * lattice_structure.dimension

  for flat_index in range(lattice_structure.num_vertices):
    yield (flat_index, per_dim_indices)
    for dim in range(lattice_structure.dimension):
      per_dim_indices[dim] += 1
      if per_dim_indices[dim] == lattice_structure.lattice_sizes[dim]:
        per_dim_indices[dim] = 0
      else:
        break


def lattice_1d_slice(lattice_param_tensor, lattice_sizes, lattice_axis, begin,
                     size):
  """Produce 1d slice of lattice param.

  Suppose we have d dimensional lattices. Recall that lattice_param_tensor
  is a 2d tensor, where the first dimension is output_dim, and the second
  dimension is a flattened version of lattice parameters.

  This function interprets lattice_param_tensor as (d + 1) dimensional tensor
  of the form:
    lattice_param[output_dim, vertex[0], vertex[1], ..., vertex[d - 1]]
  and returns the flattened (2d) representation of
    lattice_param[output_dim, :, :, ..., begin : begin + size, :, ..., :]
  where the slicing happens at lattice_axis.

  Args:
    lattice_param_tensor: [output_dim, param_dim] tensor contains lattice
      parameters in the column-major order.
    lattice_sizes: (list of ints) lattice size of each dimension.
    lattice_axis: (int) axis slice.
    begin: (int) slice beginning index at lattice_axis.
    size: (int) slice size along the axis slice.

  Returns:
    [output_dim, sliced_param_dim] tensor that contains sliced lattice params.

  Raises:
    ValueError: * If lattice_param's shape is not a 2d tensor.
      * If lattice_axis is not in [0, len(lattice_sizes) - 1].
      * If [begin, begin + size] is not a subset of
        [0, lattice_sizes[lattice_axis] - 1]
  """
  lattice_rank = len(lattice_sizes)
  param_shape = lattice_param_tensor.shape.as_list()
  if len(param_shape) != 2:
    raise ValueError('Expect 2d tensor, but got %s' % param_shape)
  if lattice_axis < 0 or lattice_axis >= lattice_rank:
    raise ValueError('lattice_axis (%d) is out of range' % lattice_axis)
  if begin < 0 or (begin + size) > lattice_sizes[lattice_axis]:
    raise ValueError(
        '[begin, begin + size] ([%d, %d]) is out of range [0, %d]' %
        (begin, begin + size, lattice_sizes[lattice_axis]))

  output_dim = param_shape[0]

  pre_axis_param_dim = 1
  for index in range(0, lattice_axis):
    pre_axis_param_dim *= lattice_sizes[index]
  post_axis_param_dim = 1
  for index in range(lattice_axis + 1, lattice_rank):
    post_axis_param_dim *= lattice_sizes[index]

  # Lattice param in each output dimension is in the column-major order, but
  # tf.reshape works in the row-major order. So we put post_axis_param_dim
  # first, and then pre_axis_param_dim.
  target_shape = [output_dim, post_axis_param_dim, lattice_sizes[lattice_axis],
                  pre_axis_param_dim]
  # reshape param to [output_dim, :, target_axis, :].
  reshaped_param = array_ops.reshape(lattice_param_tensor, shape=target_shape)
  sliced_param = array_ops.slice(
      reshaped_param, begin=[0, 0, begin, 0], size=[-1, -1, size, -1])
  final_slice = array_ops.reshape(sliced_param, shape=[output_dim, -1])

  return final_slice


class SaveOnceOrWaitTimeOutError(Exception):
  pass


def save_once_or_wait_for_chief(
    write_fn,
    metadata_dir,
    is_chief,
    timeout_secs=600):
  """Synchronizes saving data to disk across multiple tensorflow processes.

  This function can be used for synchronizing creation of data on disk that
  needs to be available to all processes in a Tensorflow cluster. Each process
  should call this function prior to using the data. The function makes the
  designated chief process write the data and every other process blocks until
  the data has been written.

  Args:
    write_fn: A function taking no arguments that executes the write to disk.
    metadata_dir: A path on the filesystem used for storing internal data
      used in this function (currently, a "done" sentinal file). If this
      directory doesn't exist it would be created; otherwise it should be
      writeable.
    is_chief: Whether the current process is the designated chief. Only one
      process should pass this as "True".
    timeout_secs: The (approximate) time in seconds a non-chief process should
      wait for the data to be created.
  Raises:
    SaveOnceOrWaitTimeOutError if this is a non-chief process and the data has
      not been created by the chief after timeout_secs seconds.
  """
  done_file = os.path.join(metadata_dir, '__tensorflow_lattice__done')
  if not is_chief:
    _poll_for_file(done_file, timeout_secs)
    return

  if file_io.file_exists(done_file):
    return

  write_fn()

  # Create an empty done file.
  file_io.recursive_create_dir(metadata_dir)
  file_io.write_string_to_file(done_file,
                               'Time created [UTC]: %s'
                               '\nHostname: %s'
                               '\nProcess id: %s'
                               '\nTraceback:\n%s' % (
                                   datetime.datetime.utcnow(),
                                   socket.gethostname(),
                                   os.getpid(),
                                   '\n'.join(traceback.format_stack())
                               ))


POLL_INTERVAL_SECS = 30


def _poll_for_file(filename, timeout_secs):
  start = time.time()
  while not file_io.file_exists(filename):
    time.sleep(POLL_INTERVAL_SECS)
    if time.time() - start > timeout_secs:
      raise SaveOnceOrWaitTimeOutError('Waiting for file %s timed-out' %
                                       filename)
