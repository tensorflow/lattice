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
"""ParallelCombination layer for combining several parallel calibration layers.

This layer wraps several calibration layers under single ParallelCombination one
that can be used by `Sequential` Keras model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_lattice.python import categorical_calibration_layer
from tensorflow_lattice.python import lattice_layer
from tensorflow_lattice.python import linear_layer
from tensorflow_lattice.python import pwl_calibration_layer
# pylint: disable=g-import-not-at-top
# Use Keras 2.
version_fn = getattr(tf.keras, "version", None)
if version_fn and version_fn().startswith("3."):
  import tf_keras as keras
else:
  keras = tf.keras


# TODO: Add support for calibrators with units > 1.
class ParallelCombination(keras.layers.Layer):
  # pyformat: disable
  """Wraps several parallel calibration layers under single one.

  `ParallelCombination` is designed for combning several calibration layers
  which output goes into single `Lattice` or `Linear` layer in order to be able
  to use calibration layers within `Sequential` model.

  Difference from `keras.layers.Concatenate` is that last one operates on
  already built objects and thus cannot be used to group layers for `Sequential`
  model.

  Input shape:
    `(batch_size, k)` or list of length `k` of shapes: `(batch_size, 1)` where
    `k` is a number of associated calibration layers.

  Output shape:
    `(batch_size, k)` or list of length `k` of shapes: `(batch_size, 1)` where
    `k` is a number of associated calibration layers. Shape of output depends on
    `single_output` parameter.

  Attributes:
    - All `__init__` arguments.

  Example:

  Example usage with a Sequential model:

  ```python
  model = keras.models.Sequential()
  combined_calibrators = ParallelCombination()
  for i in range(num_dims):
    calibration_layer = PWLCalibration(...)
    combined_calibrators.append(calibration_layer)
  model.add(combined_calibrators)
  model.add(Lattice(...))
  ```
  """
  # pyformat: enable

  def __init__(self, calibration_layers=None, single_output=True, **kwargs):
    """Initializes an instance of `ParallelCombination`.

    Args:
      calibration_layers: List of `PWLCalibration` or `CategoricalCalibration`
        objects or any other layers taking and returning tensor of shape
        `(batch_size, 1)`.
      single_output: if True returns output as single tensor of shape
        `(batch_size, k)`. Otherwise returns list of `k` tensors of shape
        `(batch_size, 1)`.
      **kwargs: other args passed to `keras.layers.Layer` initializer.
    """
    super(ParallelCombination, self).__init__(**kwargs)
    self.calibration_layers = []
    for calibration_layer in calibration_layers or []:
      if not isinstance(calibration_layer, dict):
        self.calibration_layers.append(calibration_layer)
      else:
        # Keras deserialization logic must have explicit acceess to all custom
        # classes. This is standard way to provide such access.
        with keras.utils.custom_object_scope({
            "Lattice":
                lattice_layer.Lattice,
            "Linear":
                linear_layer.Linear,
            "PWLCalibration":
                pwl_calibration_layer.PWLCalibration,
            "CategoricalCalibration":
                categorical_calibration_layer.CategoricalCalibration,
        }):
          self.calibration_layers.append(
              keras.layers.deserialize(
                  calibration_layer, use_legacy_format=True
              )
          )
    self.single_output = single_output

  def append(self, calibration_layer):
    """Appends new calibration layer to the end."""
    self.calibration_layers.append(calibration_layer)

  def build(self, input_shape):
    """Standard Keras build() method."""
    if isinstance(input_shape, list):
      if len(input_shape) != len(self.calibration_layers):
        raise ValueError("Number of ParallelCombination input tensors does not "
                         "match number of calibration layers. input_shape: %s, "
                         "layers: %s" % (input_shape, self.calibration_layers))
    else:
      if input_shape[1] != len(self.calibration_layers):
        raise ValueError("Second dimension of ParallelCombination input tensor "
                         "does not match number of calibration layers. "
                         "input_shape: %s, layers: %s" %
                         (input_shape, self.calibration_layers))
    super(ParallelCombination, self).build(input_shape)

  def call(self, inputs):
    """Standard Keras call() method."""
    if not isinstance(inputs, list):
      if len(inputs.shape) != 2:
        raise ValueError("'inputs' is expected to have rank-2. "
                         "Given: %s" % inputs)
      inputs = tf.split(inputs, axis=1, num_or_size_splits=inputs.shape[1])
    if len(inputs) != len(self.calibration_layers):
      raise ValueError("Number of ParallelCombination input tensors does not "
                       "match number of calibration layers. inputs: %s, "
                       "layers: %s" % (inputs, self.calibration_layers))
    outputs = [
        layer(one_d_input)
        for layer, one_d_input in zip(self.calibration_layers, inputs)
    ]
    if self.single_output:
      return tf.concat(outputs, axis=1)
    else:
      return outputs

  def compute_output_shape(self, input_shape):
    if self.single_output:
      return tf.TensorShape([None, len(self.calibration_layers)])
    else:
      return [tf.TensorShape([None, 1])] * len(self.calibration_layers)

  def get_config(self):
    """Standard Keras config for serialization."""
    config = {
        "calibration_layers": [
            keras.layers.serialize(layer, use_legacy_format=True)
            for layer in self.calibration_layers
        ],
        "single_output": self.single_output,
    }  # pyformat: disable
    config.update(super(ParallelCombination, self).get_config())
    return config
