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
"""A quick test script for TensorFlow Lattice's calibrated RTL estimator."""
import numpy as np

import tensorflow as tf
import tensorflow_lattice as tfl

# Feature definition.
feature_columns = [
    tf.feature_column.numeric_column('x0'),
    tf.feature_column.numeric_column('x1'),
]

# Hyperparameters.
num_keypoints = 10
hparams = tfl.CalibratedRtlHParams(
    num_keypoints=num_keypoints,
    num_lattices=5,
    lattice_rank=2,
    learning_rate=0.1)
def init_fn():
  return tfl.uniform_keypoints_for_signal(num_keypoints,
                                          input_min=-1.0,
                                          input_max=1.0,
                                          output_min=0.0,
                                          output_max=1.0)

# Estimator.
rtl_estimator = tfl.calibrated_rtl_regressor(feature_columns=feature_columns,
                                             hparams=hparams,
                                             keypoints_initializers_fn=init_fn)

# Prepare the dataset.
num_examples = 1000
x0 = np.random.uniform(-1.0, 1.0, size=num_examples)
x1 = np.random.uniform(-1.0, 1.0, size=num_examples)
y = x0 ** 2 + x1 ** 2

# Example input function.
twod_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={
        'x0': x0,
        'x1': x1
    }, y=y, batch_size=10, num_epochs=1, shuffle=False)

# Train!
rtl_estimator.train(input_fn=twod_input_fn)
# Evaluate!
print(rtl_estimator.evaluate(input_fn=twod_input_fn))
