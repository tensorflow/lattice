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
"""Tests for lattice estimators."""
import numpy as np
import tensorflow as tf
import tensorflow_lattice as tfl

# Example training and testing data.
train_features = {
    'distance': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    'quality': np.array([2.0, 5.0, 1.0, 2.0, 5.0]),
}
train_labels = np.array([0.2, 1.0, 0.0, 0.0, 1.0])

# Same quality but different distance.
test_features = {
    'distance': np.array([5.0, 10.0]),
    'quality': np.array([3.0, 3.0]),
}

# Feature definition.
feature_columns = [
    tf.feature_column.numeric_column('distance'),
    tf.feature_column.numeric_column('quality'),
]

# Hyperparameters.
num_keypoints = 10
hparams = tfl.CalibratedLatticeHParams(
    feature_names=['distance', 'quality'],
    num_keypoints=num_keypoints,
    learning_rate=0.1,
)

# Set feature monotonicity.
hparams.set_feature_param('distance', 'monotonicity', -1)
hparams.set_feature_param('quality', 'monotonicity', +1)

# Define keypoint init.
keypoints_init_fns = {
    'distance': lambda: tfl.uniform_keypoints_for_signal(num_keypoints,
                                                         input_min=0.0,
                                                         input_max=10.0,
                                                         output_min=0.0,
                                                         output_max=1.0),
    'quality': lambda: tfl.uniform_keypoints_for_signal(num_keypoints,
                                                        input_min=0.0,
                                                        input_max=5.0,
                                                        output_min=0.0,
                                                        output_max=1.0),
}

lattice_estimator = tfl.calibrated_lattice_regressor(
    feature_columns=feature_columns,
    hparams=hparams,
    keypoints_initializers_fn=keypoints_init_fns)

# Train!
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=train_features,
    y=train_labels,
    batch_size=1,
    num_epochs=100,
    shuffle=False)

lattice_estimator.train(input_fn=train_input_fn)

# Test.
test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=test_features,
    y=None,
    batch_size=1,
    num_epochs=1,
    shuffle=False)

print(list(lattice_estimator.predict(input_fn=test_input_fn)))
