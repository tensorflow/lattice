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
"""Collection of test datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import tensorflow as tf

_NUM_EXAMPLES = 10000
_BATCH_SIZE = 100
_NUM_EPOCHS = 1


class TestData(object):
  """A test dataset class."""

  def __init__(self,
               num_examples=_NUM_EXAMPLES,
               batch_size=_BATCH_SIZE,
               num_epochs=_NUM_EPOCHS):
    self.num_examples = num_examples
    self.batch_size = batch_size
    self.num_epochs = num_epochs

  # Collection of transformations that generates the label, y.
  def _f(self, x):
    return np.power(x, 3) + 0.1 * np.sin(x * math.pi * 8)

  def _g(self, x0, x1):
    return self._f(x0) + 0.3 * (1.0 - np.square(x1))

  def _h(self, x0, x1):
    radius2 = (x0 * x0 + x1 * x1)
    max_radius2 = 0.25
    return radius2 < max_radius2

  def _i(self, x0, x1, x2):
    return self._g(x0, x1) + np.choose(x2.astype(int) + 1, [11., 7., 13.])

  def oned_input_fn(self):
    """Returns an input function for one dimensional learning task.

    column 'x' is a feature column, and column 'y' is a label column.
    The transformation is deterministic, where y = _f(x).

    Returns:
      Function, that has signature of ()->({'x': data}, `target`)

    FutureWork: Make this use keypoints_initialization from quantiles.
    """
    x = np.random.uniform(-1.0, 1.0, size=self.num_examples)
    y = self._f(x)
    return tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'x': x},
        y=y,
        batch_size=self.batch_size,
        num_epochs=self.num_epochs,
        shuffle=False)

  def oned_zero_weight_input_fn(self):
    """Returns an input function for one dimensional learning task.

    column 'x' is a feature column, column 'zero' is a numerical column that
    contains zero values and column 'y' is a label column.
    The transformation is deterministic, where y = _f(x).

    Returns:
      Function, that has signature of ()->({'x': data, 'zero': zeros}, `target`)
    """
    x = np.random.uniform(-1.0, 1.0, size=self.num_examples)
    zeros = np.zeros(shape=(self.num_examples))
    y = self._f(x)
    return tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={
            'x': x,
            'zero': zeros
        },
        y=y,
        batch_size=self.batch_size,
        num_epochs=self.num_epochs,
        shuffle=False)

  def twod_input_fn(self):
    """Returns an input function for two dimensional learning task.

    column 'x0' and 'x1' are feature columns, and column 'y' is a label column.
    The transformation is deterministic, where y = _g(x0, x1).

    Returns:
      Function, that has signature of ()->({'x0': data, 'x1': data}, `target`)
    """
    x0 = np.random.uniform(-1.0, 1.0, size=self.num_examples)
    x1 = np.random.uniform(-1.0, 1.0, size=self.num_examples)
    y = self._g(x0, x1)

    return tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'x0': x0,
           'x1': x1},
        y=y,
        batch_size=self.batch_size,
        num_epochs=self.num_epochs,
        shuffle=False)

  def twod_classificer_input_fn(self):
    """Returns an input function for two dimensional classification task.

    column 'x0' and 'x1' are feature columns, and column 'y' is a label column.
    The transformation is deterministic, where y = _h(x0, x1).

    Returns:
      Function, that has signature of ()->({'x0': data, 'x1': data}, `target`)
    """
    x0 = np.random.uniform(-1.0, 1.0, size=self.num_examples)
    x1 = np.random.uniform(-1.0, 1.0, size=self.num_examples)
    y = np.vectorize(self._h)(x0, x1)

    return tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'x0': x0,
           'x1': x1},
        y=y,
        batch_size=self.batch_size,
        num_epochs=self.num_epochs,
        shuffle=False)

  def threed_input_fn(self, full_data, num_epochs=None):
    """Returns an input function for three-dimensional learning task.

    'x0' and 'x1' are numeric, and 'x2' is categorical with values {-1, 0, 1}.
    The transformation is deterministic and decomposable on the inputs,
    that is y = _i(x0, x1, x2) = _i_0(x0)+_i_1(x1)+_i_2(x2).

    Args:
      full_data: if set to true the whole data is returned in one batch.
      num_epochs: number of epochs to go over the data. Takes default used
        in construction if not set.

    Returns:
      Function, that has signature of
      ()->({'x0': data, 'x1': data, 'x2': data}, `target`)
    """
    x0 = np.random.uniform(-1.0, 1.0, size=self.num_examples)
    x1 = np.random.uniform(-1.0, 1.0, size=self.num_examples)
    x2 = np.random.choice(
        [-1., 0., 1.], size=self.num_examples, replace=True, p=[0.1, 0.7, 0.2])
    y = self._i(x0, x1, x2)

    x2_str = np.choose(x2.astype(int) + 1, ['?', 'N', 'Y'])
    if num_epochs is None:
      num_epochs = self.num_epochs
    return tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'x0': x0,
           'x1': x1,
           'x2': x2_str},
        y=y,
        batch_size=self.batch_size if not full_data else self.num_examples,
        num_epochs=num_epochs,
        shuffle=False)

  def multid_feature_input_fn(self):
    """Returns an input function with one multi-dimensional feature.

    column 'x' is the feature column, and column 'y' is a label column.
    The transformation is deterministic, where y = _g(x[0], x[1]).

    Returns:
      Function, that has signature of ()->({'x': data}, `target`)
    """
    x = np.random.uniform(-1.0, 1.0, size=[self.num_examples, 2])
    x_split = np.split(x, 2, axis=1)
    y = self._g(x_split[0], x_split[1])

    return tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'x': x},
        y=y,
        batch_size=self.batch_size,
        num_epochs=self.num_epochs,
        shuffle=False)
