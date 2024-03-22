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
"""Layer which represents aggregation function.

See class level comment.

This layer applies the provided model to the ragged input tensor and aggregates
the results.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# pylint: disable=g-import-not-at-top
# Use Keras 2.
version_fn = getattr(tf.keras, 'version', None)
if version_fn and version_fn().startswith('3.'):
  import tf_keras as keras
else:
  keras = tf.keras


class Aggregation(keras.layers.Layer):
  # pyformat: disable
  """Layer which represents an aggregation function.

  Calls the model on each of the ragged dimensions and takes the mean.

  Input shape:
  A list or dictionary with num_input_dims Rank-2 ragged tensors with
  shape: (batch_size, ?)

  Output shape:
  Rank-2 tensor with shape: (batch_size, 1)

  Attributes:
    - All `__init__ `arguments.

  Example:

  ```python
  model = keras.Model(inputs=inputs, outputs=outputs)
  layer = tfl.layers.Aggregation(model)
  ```
  """
  # pyformat: enable

  def __init__(self, model, **kwargs):
    """initializes an instance of `Aggregation`.

    Args:
      model: A keras.Model instance.
      **kwargs: Other args passed to `keras.layers.Layer` initializer.

    Raises:
      ValueError: if model is not at `keras.Model` instance.
    """
    if not isinstance(model, keras.Model):
      raise ValueError('Model must be a keras.Model instance.')
    super(Aggregation, self).__init__(**kwargs)
    # This flag enables inputs to be Ragged Tensors
    self._supports_ragged_inputs = True
    self.model = model

  def call(self, x):
    """Standard Keras call() method."""
    return tf.reduce_mean(tf.ragged.map_flat_values(self.model, x), axis=1)

  def get_config(self):
    """Standard Keras get_config() method."""
    config = super(Aggregation, self).get_config().copy()
    config.update(
        {'model': keras.utils.legacy.serialize_keras_object(self.model)}
    )
    return config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    model = keras.utils.legacy.deserialize_keras_object(
        config.pop('model'), custom_objects=custom_objects
    )
    return cls(model, **config)
