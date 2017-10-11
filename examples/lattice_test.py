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
"""A quick test script for TensorFlow Lattice's lattice layer."""
import tensorflow as tf
import tensorflow_lattice as tfl

x = tf.placeholder(tf.float32, shape=(None, 2))
(y, _, _, _) = tfl.lattice_layer(x, lattice_sizes=(2, 2))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print(sess.run(y, feed_dict={x: [[0.0, 0.0]]}))
