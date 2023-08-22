# Copyright 2023 Google LLC
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
"""TF tests for pwl_calibration_fn.py."""

import tensorflow as tf

from tensorflow_lattice.python.conditional_pwl_calibration import default_keypoint_input_parameters
from tensorflow_lattice.python.conditional_pwl_calibration import pwl_calibration_fn

_EPSILON = 1e-4


class PwlCalibrationFnTest(tf.test.TestCase):

  def assertAllClose(self, x, y):
    super().assertAllClose(x, y, rtol=_EPSILON, atol=_EPSILON)

  def assertAllGreaterEqual(self, a, comparison_target):
    super().assertAllGreaterEqual(a, comparison_target - _EPSILON)

  def assertAllLessEqual(self, a, comparison_target):
    super().assertAllLessEqual(a, comparison_target + _EPSILON)

  def assertAllEqual(self, a, comparison_target):
    super().assertAllInRange(
        a, comparison_target - _EPSILON, comparison_target + _EPSILON
    )

  def setUp(self):
    super().setUp()
    self.kernel_4 = tf.constant(
        [
            [-0.38, -0.41, -0.34, -0.29],
            [0.17, -0.32, 0.33, -0.1],
        ],
        dtype=tf.float32,
    )
    self.kernel_5 = tf.constant(
        [
            [-0.38, -0.41, -0.34, -0.29, 0.42],
            [0.17, -0.32, 0.33, -0.1, -0.36],
        ],
        dtype=tf.float32,
    )
    self.multi_unit_kernel_4 = tf.constant(
        [
            [
                [-0.26, 0.43, 0.49, 0.26],
                [0.39, 0.42, -0.33, 0.41],
                [0.28, 0.04, 0.46, 0.09],
            ],
            [
                [-0.27, -0.23, 0.29, -0.12],
                [-0.4, -0.24, -0.31, 0.01],
                [0.03, 0.01, -0.42, -0.42],
            ],
        ],
        dtype=tf.float32,
    )
    self.multi_unit_kernel_5 = tf.constant(
        [
            [
                [-0.26, 0.43, 0.49, 0.26, -0.32],
                [0.39, 0.42, -0.33, 0.41, 0.11],
                [0.28, 0.04, 0.46, 0.09, -0.33],
            ],
            [
                [-0.27, -0.23, 0.29, -0.12, 0.46],
                [-0.4, -0.24, -0.31, 0.01, 0.21],
                [0.03, 0.01, -0.42, -0.42, 0.37],
            ],
        ],
        dtype=tf.float32,
    )

  def test_suite_none_monotonic(self):
    """Tests non-monotonic calibration."""
    # basic call
    y = pwl_calibration_fn(
        inputs=tf.constant([[0.5], [0.8]]),
        keypoint_output_parameters=self.kernel_4,
        keypoint_input_parameters=default_keypoint_input_parameters(
            keypoints=[0.0, 0.1, 0.4, 1.0]
        ),
        keypoint_input_min=0.0,
        keypoint_input_max=1.0,
    )
    print(default_keypoint_input_parameters(keypoints=[0.0, 0.1, 0.4, 1.0]))
    self.assertAllClose(y, tf.constant([[0.41784188], [0.51060027]]))

    # if is_cyclic, starting and ending keypoints give the same prediction
    y1 = pwl_calibration_fn(
        inputs=tf.constant([[0.5], [0.5]]),
        keypoint_output_parameters=self.kernel_4,
        is_cyclic=True,
        keypoint_input_parameters=default_keypoint_input_parameters(
            keypoints=[0.5, 0.6, 0.65, 0.7, 0.8]
        ),
        keypoint_input_min=0.5,
        keypoint_input_max=0.8,
    )
    y2 = pwl_calibration_fn(
        inputs=tf.constant([[0.8], [0.8]]),
        keypoint_output_parameters=self.kernel_4,
        is_cyclic=True,
        keypoint_input_parameters=default_keypoint_input_parameters(
            keypoints=[0.5, 0.6, 0.65, 0.7, 0.8]
        ),
        keypoint_input_min=0.5,
        keypoint_input_max=0.8,
    )
    self.assertAllClose(y1, y2)

    # basic multi-unit call, input needs broadcast
    y = pwl_calibration_fn(
        inputs=tf.constant([[0.5], [0.8]]),
        units=3,
        keypoint_output_parameters=self.multi_unit_kernel_4,
        keypoint_input_parameters=default_keypoint_input_parameters(
            keypoints=[0.0, 0.1, 0.4, 1.0]
        ),
        keypoint_input_min=0.0,
        keypoint_input_max=1.0,
    )
    self.assertAllClose(
        y,
        tf.constant([
            [0.6108614, 0.44871515, 0.5979259],
            [0.50402266, 0.47603822, 0.39651677],
        ]),
    )

    # basic multi-unit call
    y = pwl_calibration_fn(
        inputs=tf.constant([[0.5, 0.5, 0.5], [0.8, 0.8, 0.8]]),
        units=3,
        keypoint_output_parameters=self.multi_unit_kernel_4,
        keypoint_input_parameters=default_keypoint_input_parameters(
            keypoints=[0.0, 0.1, 0.4, 1.0]
        ),
        keypoint_input_min=0.0,
        keypoint_input_max=1.0,
    )
    self.assertAllClose(
        y,
        tf.constant([
            [0.6108614, 0.44871515, 0.5979259],
            [0.50402266, 0.47603822, 0.39651677],
        ]),
    )

    # keypoint_output_min and keypoint_output_max scales correctly
    y1 = pwl_calibration_fn(
        inputs=tf.constant([[0.5, 0.5, 0.5], [0.8, 0.8, 0.8]]),
        units=3,
        keypoint_output_parameters=self.multi_unit_kernel_4,
        keypoint_input_parameters=default_keypoint_input_parameters(
            keypoints=[0.0, 0.1, 0.4, 1.0]
        ),
        keypoint_input_min=0.0,
        keypoint_input_max=1.0,
    )
    y2 = pwl_calibration_fn(
        inputs=tf.constant([[0.5, 0.5, 0.5], [0.8, 0.8, 0.8]]),
        units=3,
        keypoint_output_parameters=self.multi_unit_kernel_4,
        keypoint_input_parameters=default_keypoint_input_parameters(
            keypoints=[0.0, 0.1, 0.4, 1.0]
        ),
        keypoint_output_min=-1.0,
        keypoint_output_max=10.0,
        keypoint_input_min=0.0,
        keypoint_input_max=1.0,
    )
    self.assertAllClose(y1 * 11.0 - 1.0, y2)

    # multi-unit is_cyclic gives cyclic predictions
    y1 = pwl_calibration_fn(
        inputs=tf.constant([[-0.1], [1.1]]),
        units=3,
        is_cyclic=True,
        keypoint_output_parameters=self.multi_unit_kernel_4,
        keypoint_input_parameters=default_keypoint_input_parameters(
            keypoints=[0.0, 0.1, 0.2, 0.5, 1.0]
        ),
        keypoint_input_min=0.0,
        keypoint_input_max=1.0,
    )
    y2 = pwl_calibration_fn(
        inputs=tf.constant([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]),
        units=3,
        is_cyclic=True,
        keypoint_output_parameters=self.multi_unit_kernel_4,
        keypoint_input_parameters=default_keypoint_input_parameters(
            keypoints=[0.0, 0.1, 0.2, 0.5, 1.0]
        ),
        keypoint_input_min=0.0,
        keypoint_input_max=1.0,
    )
    self.assertAllClose(y1, y2)

    # missing input with given missing output imputed correctly
    y = pwl_calibration_fn(
        inputs=tf.constant([[0.5], [-1.0]]),
        keypoint_output_parameters=self.kernel_4,
        keypoint_input_parameters=default_keypoint_input_parameters(
            keypoints=[0.0, 0.1, 0.4, 1.0]
        ),
        keypoint_input_min=0.0,
        keypoint_input_max=1.0,
        missing_input_value=-1.0,
        missing_output_value=3.0,
    )
    self.assertAllClose(y, tf.constant([[0.41784188], [3.0]]))

    # missing input imputed correctly with derived missing output
    y = pwl_calibration_fn(
        inputs=tf.constant([[0.5], [-1.0]]),
        keypoint_output_parameters=self.kernel_5,
        keypoint_input_parameters=default_keypoint_input_parameters(
            keypoints=[0.0, 0.1, 0.4, 1.0]
        ),
        keypoint_input_min=0.0,
        keypoint_input_max=1.0,
        missing_input_value=-1.0,
    )
    self.assertAllClose(y, tf.constant([[0.41784188], [0.41095957]]))

  def test_suite_increasing_monotonic(self):
    """Tests monotonic calibration."""
    # basic call
    y = pwl_calibration_fn(
        inputs=tf.constant([[0.5], [0.8]]),
        keypoint_output_parameters=self.kernel_4,
        monotonicity='increasing',
        keypoint_input_parameters=default_keypoint_input_parameters(
            keypoints=[0.0, 0.1, 0.4, 1.0]
        ),
        keypoint_input_min=0.0,
        keypoint_input_max=1.0,
    )
    self.assertAllClose(y, tf.constant([[0.64769804], [0.7371951]]))

    # outputs are monotonic
    y1 = pwl_calibration_fn(
        inputs=tf.constant([[-0.5], [0.3]]),
        keypoint_output_parameters=self.kernel_4,
        monotonicity='increasing',
        keypoint_input_parameters=default_keypoint_input_parameters(
            keypoints=[0.0, 0.1, 0.4, 1.0]
        ),
        keypoint_input_min=0.0,
        keypoint_input_max=1.0,
    )
    y2 = pwl_calibration_fn(
        inputs=tf.constant([[0.5], [0.8]]),
        keypoint_output_parameters=self.kernel_4,
        monotonicity='increasing',
        keypoint_input_parameters=default_keypoint_input_parameters(
            keypoints=[0.0, 0.1, 0.4, 1.0]
        ),
        keypoint_input_min=0.0,
        keypoint_input_max=1.0,
    )
    y3 = pwl_calibration_fn(
        inputs=tf.constant([[0.6], [1.2]]),
        keypoint_output_parameters=self.kernel_4,
        monotonicity='increasing',
        keypoint_input_parameters=default_keypoint_input_parameters(
            keypoints=[0.0, 0.1, 0.4, 1.0]
        ),
        keypoint_input_min=0.0,
        keypoint_input_max=1.0,
    )
    self.assertAllGreaterEqual(y2 - y1, 0.0)
    self.assertAllGreaterEqual(y3 - y2, 0.0)

    # clamp_min works as expected
    y = pwl_calibration_fn(
        inputs=tf.constant([[0.0], [-0.2]]),
        keypoint_output_parameters=self.multi_unit_kernel_4,
        keypoint_input_parameters=default_keypoint_input_parameters(
            num_keypoints=5
        ),
        keypoint_input_min=0.0,
        keypoint_input_max=2.0,
        monotonicity='increasing',
        keypoint_output_min=-10.0,
        clamp_min=True,
        units=3,
    )
    self.assertAllEqual(y, -10.0)

    # clamp_out works as expected
    y = pwl_calibration_fn(
        inputs=tf.constant([[2.0], [2.5]]),
        keypoint_output_parameters=self.multi_unit_kernel_4,
        keypoint_input_parameters=default_keypoint_input_parameters(
            num_keypoints=5
        ),
        keypoint_input_min=0.0,
        keypoint_input_max=2.0,
        monotonicity='increasing',
        keypoint_output_max=10.0,
        clamp_max=True,
        units=3,
    )
    self.assertAllEqual(y, 10.0)

    # clamp_min and clamp_out work as expected together, min
    y = pwl_calibration_fn(
        inputs=tf.constant([[0.0, 0.0, -10.0], [-0.2, 0.0, -100.0]]),
        keypoint_output_parameters=self.multi_unit_kernel_4,
        keypoint_input_parameters=default_keypoint_input_parameters(
            keypoints=[0.0, 0.1, 0.4, 1.0, 1.5, 2.0]
        ),
        keypoint_input_min=0.0,
        keypoint_input_max=2.0,
        monotonicity='increasing',
        keypoint_output_min=-10.0,
        clamp_min=True,
        keypoint_output_max=5.0,
        clamp_max=True,
        units=3,
    )
    self.assertAllEqual(y, -10.0)

    # clamp_min and clamp_out work as expected together, max
    y = pwl_calibration_fn(
        inputs=tf.constant([[2.0, 3.0, 4.0], [2.5, 2.5, 2.5]]),
        keypoint_output_parameters=self.multi_unit_kernel_4,
        units=3,
        keypoint_input_parameters=default_keypoint_input_parameters(
            keypoints=[0.0, 0.1, 0.4, 1.0, 1.5, 2.0]
        ),
        keypoint_input_min=0.0,
        keypoint_input_max=2.0,
        monotonicity='increasing',
        keypoint_output_min=-10.0,
        clamp_min=True,
        keypoint_output_max=5.0,
        clamp_max=True,
    )
    self.assertAllEqual(y, 5.0)

    # clamp_min, clamp_out, missing_input_value and derived missing_output_value
    # work as expected together
    y = pwl_calibration_fn(
        inputs=tf.constant([[0.0, 1.0, 2.0], [-0.5, 1.5, 2.5]]),
        keypoint_output_parameters=self.multi_unit_kernel_5,
        units=3,
        keypoint_input_parameters=default_keypoint_input_parameters(
            keypoints=[0.0, 0.1, 0.4, 1.0, 1.5, 2.0]
        ),
        keypoint_input_min=0.0,
        keypoint_input_max=2.0,
        monotonicity='increasing',
        keypoint_output_min=-10.0,
        clamp_min=True,
        keypoint_output_max=5.0,
        clamp_max=True,
        missing_input_value=-1.0,
    )
    self.assertAllClose(
        y, tf.constant([[-10.0, -0.3635044, 5.0], [-10.0, 1.3930602, 5.0]])
    )

    # clamp_min, clamp_out and missing_input_value work as expected together
    y = pwl_calibration_fn(
        inputs=tf.constant([[0.0, -1.0, 2.0], [-0.5, -1.0, 2.5]]),
        keypoint_output_parameters=self.multi_unit_kernel_4,
        units=3,
        keypoint_input_parameters=default_keypoint_input_parameters(
            keypoints=[0.0, 0.1, 0.4, 1.0, 1.5, 2.0]
        ),
        keypoint_input_min=0.0,
        keypoint_input_max=2.0,
        monotonicity='increasing',
        keypoint_output_min=-10.0,
        clamp_min=True,
        keypoint_output_max=5.0,
        clamp_max=True,
        missing_input_value=-1.0,
        missing_output_value=3.0,
    )
    self.assertAllClose(y, tf.constant([[-10.0, 3.0, 5.0], [-10.0, 3.0, 5.0]]))

  def test_gradient_step(self):
    """Tests gradient computation."""
    trainable = tf.Variable(
        tf.zeros_like(self.multi_unit_kernel_5, dtype=tf.float32),
        trainable=True,
        name='trainable',
    )

    with tf.GradientTape() as tape:
      y = pwl_calibration_fn(
          inputs=tf.constant([[-1.0, 0.0, 1.0], [0.8, 2.0, 3.0]]),
          keypoint_output_parameters=trainable,
          units=3,
          keypoint_input_parameters=default_keypoint_input_parameters(
              keypoints=[0.0, 0.1, 0.4, 1.0, 2.0]
          ),
          keypoint_input_min=0.0,
          keypoint_input_max=2.0,
          monotonicity='increasing',
          keypoint_output_max=10.0,
          clamp_max=True,
          missing_input_value=-1.0,
      )
      loss = tf.reduce_mean(y * y)
    grads = tape.gradient(loss, trainable)
    self.assertAllClose(
        grads,
        tf.constant([
            [
                [0.0, 0.0, 0.0, 0.0, 4.166667],
                [-0.26666668, -0.26666668, -0.26666668, -0.26666668, 0.0],
                [1.0666668, 1.0666668, 1.0666668, -4.266667, 0.0],
            ],
            [
                [1.3037037, 1.3037037, -0.3259262, -3.5851853, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ]),
    )

  def test_suite_raises(self):
    """Tests verifiable ValueErrors."""

    with self.assertRaises(ValueError):
      _ = pwl_calibration_fn(
          inputs=tf.constant([[0.5], [0.8]]),
          keypoint_output_parameters=self.kernel_4,
          keypoint_input_parameters=default_keypoint_input_parameters(
              keypoints=[0.0, 0.1, 0.1, 1.0]
          ),
      )
    with self.assertRaises(ValueError):
      _ = pwl_calibration_fn(
          inputs=tf.constant([[0.5], [0.8]]),
          keypoint_output_parameters=self.kernel_4,
          keypoint_input_parameters=default_keypoint_input_parameters(
              keypoints=[0.0, 0.3, 0.1, 1.0]
          ),
      )
    with self.assertRaises(ValueError):
      _ = pwl_calibration_fn(
          inputs=tf.constant([[0.5], [0.8]]),
          keypoint_output_parameters=self.kernel_4,
          keypoint_input_parameters=default_keypoint_input_parameters(
              keypoints=[0.0, 0.1, 0.2, 0.3, 1.0]
          ),
      )
    with self.assertRaises(ValueError):
      _ = pwl_calibration_fn(
          inputs=tf.constant([[0.5], [0.8]]),
          keypoint_output_parameters=self.kernel_4,
          is_cyclic=True,
          keypoint_input_parameters=default_keypoint_input_parameters(
              keypoints=[0.0, 0.1, 0.3, 1.0]
          ),
      )
    with self.assertRaises(ValueError):
      _ = pwl_calibration_fn(
          inputs=tf.constant([[0.5], [0.8]]),
          keypoint_output_parameters=self.kernel_4,
          units=3,
          keypoint_input_parameters=default_keypoint_input_parameters(
              keypoints=[0.0, 0.1, 0.5, 1.0]
          ),
      )
    with self.assertRaises(ValueError):
      _ = pwl_calibration_fn(
          inputs=tf.constant([[0.5], [0.8]]),
          keypoint_output_parameters=self.kernel_4,
          keypoint_input_parameters=default_keypoint_input_parameters(
              keypoints=[0.0, 0.1, 0.4, 1.0]
          ),
          keypoint_output_min=1.0,
          keypoint_output_max=0.0,
      )
    with self.assertRaises(ValueError):
      _ = pwl_calibration_fn(
          inputs=tf.constant([[0.5], [0.8]]),
          keypoint_output_parameters=self.kernel_4,
          keypoint_input_parameters=default_keypoint_input_parameters(
              keypoints=[0.0, 0.1, 0.4, 1.0]
          ),
          missing_output_value=1.0,
      )
    with self.assertRaises(ValueError):
      _ = pwl_calibration_fn(
          inputs=tf.constant([[0.5], [0.8]]),
          keypoint_output_parameters=self.kernel_4,
          keypoint_input_parameters=default_keypoint_input_parameters(
              keypoints=[0.0, 0.1, 0.4, 1.0]
          ),
          keypoint_input_min=1.0,
          keypoint_input_max=0.0,
      )
    with self.assertRaises(ValueError):
      _ = pwl_calibration_fn(
          inputs=tf.constant([[0.5], [0.8]]),
          keypoint_output_parameters=self.multi_unit_kernel_4,
          keypoint_input_parameters=default_keypoint_input_parameters(
              keypoints=[0.0, 0.1, 0.4, 1.0]
          ),
      )
    with self.assertRaises(ValueError):
      _ = pwl_calibration_fn(
          inputs=tf.constant([[0.5], [0.8]]),
          keypoint_output_parameters=self.kernel_5,
          monotonicity='increasing',
          keypoint_input_parameters=default_keypoint_input_parameters(
              keypoints=[0.0, 0.1, 0.4, 1.0]
          ),
      )
    with self.assertRaises(ValueError):
      _ = pwl_calibration_fn(
          inputs=tf.constant([[0.5, 0.6, 0.7, 0.8], [0.0, 0.1, 0.2, 0.8]]),
          units=3,
          keypoint_output_parameters=self.multi_unit_kernel_5,
          monotonicity='increasing',
          keypoint_input_parameters=default_keypoint_input_parameters(
              keypoints=[0.0, 0.1, 0.4, 0.7, 1.0]
          ),
      )


if __name__ == '__main__':
  tf.test.main()
