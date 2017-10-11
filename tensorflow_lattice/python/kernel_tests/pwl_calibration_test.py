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
"""Tests for piecewise-linear calibration gradient."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensorflow_lattice.python.ops import pwl_calibration_ops

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


_MAX_ABSOLUTE_NUMERIC_ERROR = 1e-4


class PWLCalibrationOpsTest(test.TestCase):

  def _testInBetweenGradients(self, kp_inputs):
    """Compares numerical with the calculated gradient and checks the error."""
    # Create batch with all values in between the keypoints inputs.
    x_values = []
    for ii in range(len(kp_inputs) - 1):
      x_values += [(kp_inputs[ii] + kp_inputs[ii + 1]) / 2]
    x_values = np.asarray(x_values, dtype=np.float32)

    tf_logging.info("kp_inputs = %s" % kp_inputs)
    tf_logging.info("x_values = %s" % x_values)
    with ops.Graph().as_default():
      with self.test_session(use_gpu=False):
        x_shape = [x_values.size]
        x = array_ops.placeholder(dtype=np.float32, shape=x_shape, name="x")
        y_shape = [x_values.size, len(kp_inputs)]

        # Dense version.
        y_dense = pwl_calibration_ops.pwl_indexing_calibrator(
            input=x,
            kp_inputs=array_ops.constant(
                kp_inputs, dtype=dtypes.float32))
        y_dense_values = y_dense.eval(feed_dict={x: x_values})
        tf_logging.info("y_dense=%s" % (y_dense_values,))
        dense_error = gradient_checker.compute_gradient_error(
            x, x_shape, y_dense, y_shape, x_init_value=x_values)
        tf_logging.info("dense_error = %f" % dense_error)
        self.assertTrue(dense_error < _MAX_ABSOLUTE_NUMERIC_ERROR)

        # Sparse version.
        sparse_indices, sparse_weights = (
            pwl_calibration_ops.pwl_indexing_calibrator_sparse(
                input=x,
                kp_inputs=array_ops.constant(
                    kp_inputs, dtype=dtypes.float32)))
        y_sparse = sparse_ops.sparse_to_dense(sparse_indices, y_shape,
                                              sparse_weights)
        y_sparse_values = y_sparse.eval(feed_dict={x: x_values})
        tf_logging.info("y_sparse=%s" % (y_sparse_values,))
        sparse_weights_values = sparse_weights.eval(feed_dict={x: x_values})
        sparse_error = gradient_checker.compute_gradient_error(
            x,
            x_shape,
            sparse_weights,
            sparse_weights_values.shape,
            x_init_value=x_values)
        tf_logging.info("sparse_error = %f" % sparse_error)
        self.assertTrue(sparse_error < _MAX_ABSOLUTE_NUMERIC_ERROR)

    self.assertTrue(  # Checks dense and sparse y's are the same.
        np.allclose(
            y_dense_values, y_sparse_values, atol=_MAX_ABSOLUTE_NUMERIC_ERROR))

  def testInBetweenGradients(self):
    # Notice we don't test the gradients on top of the keypoints (including
    # edges) because the gradient cannot be calculated numerically on those
    # points.
    # But our op define arbitrary values for them, and they are tested
    # in the C++ implementation. Here it suffices to test that the proper op
    # gradient c++ implementation is being called.
    self._testInBetweenGradients([0.0, 1.0])
    self._testInBetweenGradients([0.0, 1.0, 2.0, 3.0, 4.0])
    self._testInBetweenGradients([0.0, 1.0, 10.0, 100.0])


if __name__ == "__main__":
  test.main()
