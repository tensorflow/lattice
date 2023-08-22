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
"""TF tests for conditional_cdf.py."""

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_lattice.python.conditional_cdf import cdf_fn

_EPSILON = 1e-4


class CdfFnTest(parameterized.TestCase, tf.test.TestCase):

  def assertAllClose(self, x, y):
    super().assertAllClose(x, y, atol=1e-4)

  @parameterized.named_parameters(
      dict(
          testcase_name="trivial",
          inputs=[[-1.0], [0.0], [1.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]]],
              [[[-1.0], [0.0], [1.0]]],
              [[[-1.0], [0.0], [1.0]]],
          ],
          scaling_parameters=[[[[1.0]]], [[[1.0]]], [[[1.0]]]],
          reduction="none",
          expected=[[[0.29604811]], [[0.5]], [[0.70395189]]],
      ),
      dict(
          testcase_name="trivial_mean",
          inputs=[[-1.0], [0.0], [1.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]]],
              [[[-1.0], [0.0], [1.0]]],
              [[[-1.0], [0.0], [1.0]]],
          ],
          scaling_parameters=[[[[1.0]]], [[[1.0]]], [[[1.0]]]],
          reduction="mean",
          expected=[[0.29604811], [0.5], [0.70395189]],
      ),
      dict(
          testcase_name="moderate",
          inputs=[[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]], [[-2.0], [0.0], [2.0]]],
              [[[-1.0], [0.0], [1.0]], [[-3.0], [0.0], [3.0]]],
              [[[-1.0], [0.0], [1.0]], [[-4.0], [0.0], [4.0]]],
          ],
          scaling_parameters=[
              [[[1.0]], [[1.0]]],
              [[[1.0]], [[1.0]]],
              [[[1.0]], [[1.0]]],
          ],
          reduction="none",
          expected=[
              [[0.29604811], [0.5]],
              [[0.5], [0.61075843]],
              [[0.70395189], [0.66584245]],
          ],
      ),
      dict(
          testcase_name="moderate_scaling",
          inputs=[[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]], [[-2.0], [0.0], [2.0]]],
              [[[-1.0], [0.0], [1.0]], [[-3.0], [0.0], [3.0]]],
              [[[-1.0], [0.0], [1.0]], [[-4.0], [0.0], [4.0]]],
          ],
          scaling_parameters=[
              [[[1.0]], [[1.0]]],
              [[[2.0]], [[2.0]]],
              [[[5.0]], [[7.0]]],
          ],
          reduction="none",
          expected=[
              [[0.29604811], [0.5]],
              [[0.5], [0.632815979]],
              [[0.8310872504], [0.6666666666]],
          ],
      ),
      dict(
          testcase_name="moderate_mean",
          inputs=[[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]], [[-2.0], [0.0], [2.0]]],
              [[[-1.0], [0.0], [1.0]], [[-3.0], [0.0], [3.0]]],
              [[[-1.0], [0.0], [1.0]], [[-4.0], [0.0], [4.0]]],
          ],
          scaling_parameters=None,
          reduction="mean",
          expected=[[0.398024055], [0.555379215], [0.684897170]],
      ),
      dict(
          testcase_name="moderate_geometric_mean",
          inputs=[[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]], [[-2.0], [0.0], [2.0]]],
              [[[-1.0], [0.0], [1.0]], [[-3.0], [0.0], [3.0]]],
              [[[-1.0], [0.0], [1.0]], [[-4.0], [0.0], [4.0]]],
          ],
          scaling_parameters=[
              [[[1.0]], [[1.0]]],
              [[[1.0]], [[1.0]]],
              [[[1.0]], [[1.0]]],
          ],
          reduction="geometric_mean",
          expected=[[0.38473894], [0.55261127], [0.68463206]],
      ),
  )
  def test_compute_sigmoid(
      self,
      inputs,
      location_parameters,
      scaling_parameters,
      reduction,
      expected,
  ):
    result = cdf_fn(
        inputs=tf.constant(inputs, dtype=tf.float32),
        location_parameters=tf.constant(location_parameters, dtype=tf.float32),
        scaling_parameters=(
            tf.constant(scaling_parameters, dtype=tf.float32)
            if scaling_parameters is not None
            else None
        ),
        units=1,
        activation="sigmoid",
        reduction=reduction,
    )
    self.assertAllClose(result, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name="trivial",
          inputs=[[-1.0], [0.0], [1.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]]],
              [[[-1.0], [0.0], [1.0]]],
              [[[-1.0], [0.0], [1.0]]],
          ],
          scaling_parameters=[[[[1.0]]], [[[1.0]]], [[[1.0]]]],
          reduction="none",
          expected=[[[0.0]], [[1.0 / 18]], [[3.0 / 18]]],
      ),
      dict(
          testcase_name="trivial_none_scaling",
          inputs=[[-1.0], [0.0], [1.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]]],
              [[[-1.0], [0.0], [1.0]]],
              [[[-1.0], [0.0], [1.0]]],
          ],
          scaling_parameters=None,
          reduction="none",
          expected=[[[0.0]], [[1.0 / 18]], [[3.0 / 18]]],
      ),
      dict(
          testcase_name="trivial_mean",
          inputs=[[-1.0], [0.0], [1.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]]],
              [[[-1.0], [0.0], [1.0]]],
              [[[-1.0], [0.0], [1.0]]],
          ],
          scaling_parameters=[[[[1.0]]], [[[1.0]]], [[[1.0]]]],
          reduction="mean",
          expected=[[0.0], [1.0 / 18], [3.0 / 18]],
      ),
      dict(
          testcase_name="moderate",
          inputs=[[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]], [[-2.0], [0.0], [2.0]]],
              [[[-1.0], [0.0], [1.0]], [[-3.0], [0.0], [3.0]]],
              [[[-1.0], [0.0], [1.0]], [[-4.0], [0.0], [4.0]]],
          ],
          scaling_parameters=[
              [[[1.0]], [[1.0]]],
              [[[1.0]], [[1.0]]],
              [[[1.0]], [[1.0]]],
          ],
          reduction="none",
          expected=[
              [[0.0], [2.0 / 18]],
              [[1.0 / 18], [5.0 / 18]],
              [[3.0 / 18], [8.0 / 18]],
          ],
      ),
      dict(
          testcase_name="moderate_none_scaling",
          inputs=[[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]], [[-2.0], [0.0], [2.0]]],
              [[[-1.0], [0.0], [1.0]], [[-3.0], [0.0], [3.0]]],
              [[[-1.0], [0.0], [1.0]], [[-4.0], [0.0], [4.0]]],
          ],
          scaling_parameters=None,
          reduction="none",
          expected=[
              [[0.0], [2.0 / 18]],
              [[1.0 / 18], [5.0 / 18]],
              [[3.0 / 18], [8.0 / 18]],
          ],
      ),
      dict(
          testcase_name="moderate_scaling",
          inputs=[[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]], [[-2.0], [0.0], [2.0]]],
              [[[-1.0], [0.0], [1.0]], [[-3.0], [0.0], [3.0]]],
              [[[-1.0], [0.0], [1.0]], [[-4.0], [0.0], [4.0]]],
          ],
          scaling_parameters=[
              [[[1.0]], [[1.0]]],
              [[[2.0]], [[2.0]]],
              [[[5.0]], [[0.5]]],
          ],
          reduction="none",
          expected=[
              [[0.0], [2.0 / 18]],
              [[2.0 / 18], [8.0 / 18]],
              [[11.0 / 18], [4.0 / 18]],
          ],
      ),
      dict(
          testcase_name="moderate_mean",
          inputs=[[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]], [[-2.0], [0.0], [2.0]]],
              [[[-1.0], [0.0], [1.0]], [[-3.0], [0.0], [3.0]]],
              [[[-1.0], [0.0], [1.0]], [[-4.0], [0.0], [4.0]]],
          ],
          scaling_parameters=[
              [[[1.0]], [[1.0]]],
              [[[1.0]], [[1.0]]],
              [[[1.0]], [[1.0]]],
          ],
          reduction="mean",
          expected=[[1.0 / 18], [3.0 / 18], [5.5 / 18]],
      ),
      dict(
          testcase_name="moderate_geometric_mean",
          inputs=[[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]], [[-2.0], [0.0], [2.0]]],
              [[[-1.0], [0.0], [1.0]], [[-3.0], [0.0], [3.0]]],
              [[[-1.0], [0.0], [1.0]], [[-4.0], [0.0], [4.0]]],
          ],
          scaling_parameters=[
              [[[1.0]], [[1.0]]],
              [[[1.0]], [[1.0]]],
              [[[1.0]], [[1.0]]],
          ],
          reduction="geometric_mean",
          expected=[[0.0], [2.23606797 / 18], [4.898979485 / 18]],
      ),
  )
  def test_compute_relu6(
      self,
      inputs,
      location_parameters,
      scaling_parameters,
      reduction,
      expected,
  ):
    result = cdf_fn(
        inputs=tf.constant(inputs, dtype=tf.float32),
        location_parameters=tf.constant(location_parameters, dtype=tf.float32),
        scaling_parameters=(
            tf.constant(scaling_parameters, dtype=tf.float32)
            if scaling_parameters is not None
            else None
        ),
        units=1,
        activation="relu6",
        reduction=reduction,
    )
    self.assertAllClose(result, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name="0.0",
          scaling_exp_transform_multiplier=0.0,
          expected=[[0.398024055], [0.555379215], [0.684897170]],
      ),
      dict(
          testcase_name="1.0",
          scaling_exp_transform_multiplier=1.0,
          expected=[[0.344373118], [0.58323046], [0.6278357037]],
      ),
      dict(
          testcase_name="-1.0",
          scaling_exp_transform_multiplier=-1.0,
          expected=[[0.4554976295], [0.51644151635], [0.66798191003]],
      ),
  )
  def test_scaling_exp_transformation(
      self, scaling_exp_transform_multiplier, expected
  ):
    result = cdf_fn(
        inputs=tf.constant([[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]]),
        location_parameters=tf.constant([
            [[[-1.0], [0.0], [1.0]], [[-2.0], [0.0], [2.0]]],
            [[[-1.0], [0.0], [1.0]], [[-3.0], [0.0], [3.0]]],
            [[[-1.0], [0.0], [1.0]], [[-4.0], [0.0], [4.0]]],
        ]),
        scaling_parameters=tf.constant([
            [[[1.0]], [[1.0]]],
            [[[0.0]], [[2.0]]],
            [[[-1.0]], [[3.0]]],
        ]),
        reduction="mean",
        activation="sigmoid",
        scaling_exp_transform_multiplier=scaling_exp_transform_multiplier,
    )
    self.assertAllClose(result, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name="sigmoid_repeat",
          inputs=[[0.0], [0.0], [0.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]]],
              [[[-1.0], [0.0], [1.0]]],
              [[[-1.0], [0.0], [1.0]]],
          ],
          scaling_parameters=[[[[1.0]]], [[[1.0]]], [[[1.0]]]],
          units=1,
          activation="sigmoid",
          sparsity_factor=1,
          scaling_exp_transform_multiplier=None,
          expected=[
              [
                  [[[-0.06553731], [-0.08333334], [-0.06553732]]],
                  [[[-0.06553731], [-0.08333334], [-0.06553732]]],
                  [[[-0.06553731], [-0.08333334], [-0.06553732]]],
              ],
              [
                  [[[-7.4505806e-09]]],
                  [[[-7.4505806e-09]]],
                  [[[-7.4505806e-09]]],
              ],
          ],
      ),
      dict(
          testcase_name="sigmoid_trivial",
          inputs=[[-1.0], [0.0], [1.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]]],
              [[[-1.0], [0.0], [1.0]]],
              [[[-1.0], [0.0], [1.0]]],
          ],
          scaling_parameters=[[[[1.0]]], [[[1.0]]], [[[1.0]]]],
          units=1,
          activation="sigmoid",
          sparsity_factor=1,
          scaling_exp_transform_multiplier=None,
          expected=[
              [
                  [[[-0.04934135], [-0.03880439], [-0.0207221]]],
                  [[[-0.06553731], [-0.08333334], [-0.06553732]]],
                  [[[-0.04927362], [-0.09227023], [-0.11732531]]],
              ],
              [[[[-8.0248594e-02]]], [[[-7.4505806e-09]]], [[[1.9081746e-01]]]],
          ],
      ),
      dict(
          testcase_name="relu6",
          inputs=[[-1.0], [0.0], [1.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]]],
              [[[-1.0], [0.0], [1.0]]],
              [[[-1.0], [0.0], [1.0]]],
          ],
          scaling_parameters=[[[[1.0]]], [[[1.0]]], [[[1.0]]]],
          units=1,
          activation="relu6",
          sparsity_factor=1,
          scaling_exp_transform_multiplier=None,
          expected=[
              [
                  [[[-0.0], [-0.0], [-0.0]]],
                  [[[-0.00617284], [-0.0], [-0.0]]],
                  [[[-0.01851852], [-0.01851852], [-0.0]]],
              ],
              [[[[0.0]]], [[[0.00617284]]], [[[0.05555556]]]],
          ],
      ),
      dict(
          testcase_name="units_multiplier_sigmoid",
          inputs=[[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]], [[-2.0], [0.0], [2.0]]],
              [[[-1.0], [0.0], [1.0]], [[-3.0], [0.0], [3.0]]],
              [[[-1.0], [0.0], [1.0]], [[-4.0], [0.0], [4.0]]],
          ],
          scaling_parameters=[
              [[[1.0]], [[1.0]]],
              [[[2.0]], [[2.0]]],
              [[[5.0]], [[7.0]]],
          ],
          units=2,
          activation="sigmoid",
          sparsity_factor=2,
          scaling_exp_transform_multiplier=0.0,
          expected=[
              [
                  [
                      [[-0.04934135], [-0.03880439], [-0.0207221]],
                      [[-0.03499786], [-0.08333334], [-0.03499787]],
                  ],
                  [
                      [[-0.06553731], [-0.08333334], [-0.06553732]],
                      [[-0.00719178], [-0.08005493], [-0.04275048]],
                  ],
                  [
                      [[-0.04927362], [-0.09227023], [-0.11732531]],
                      [[-0.00109488], [-0.04660612], [-0.04660612]],
                  ],
              ],
              [[[[-0.0]], [[-0.0]]], [[[-0.0]], [[0.0]]], [[[0.0]], [[0.0]]]],
          ],
      ),
      dict(
          testcase_name="units_multiplier_relu6",
          inputs=[[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]], [[-2.0], [0.0], [2.0]]],
              [[[-1.0], [0.0], [1.0]], [[-3.0], [0.0], [3.0]]],
              [[[-1.0], [0.0], [1.0]], [[-4.0], [0.0], [4.0]]],
          ],
          scaling_parameters=[
              [[[1.0]], [[1.0]]],
              [[[2.0]], [[2.0]]],
              [[[5.0]], [[7.0]]],
          ],
          units=2,
          activation="relu6",
          sparsity_factor=2,
          scaling_exp_transform_multiplier=0.01,
          expected=[
              [
                  [
                      [[-0.000], [-0.000], [-0.000]],
                      [[-0.01259508], [-0.0], [-0.0]],
                  ],
                  [
                      [[-0.00642476], [-0.0], [-0.0]],
                      [[-0.03212379], [-0.03212379], [-0.0]],
                  ],
                  [
                      [[-0.02046613], [-0.02046613], [-0.0]],
                      [[-0.0], [-0.05392344], [-0.0]],
                  ],
              ],
              [
                  [[[0.0000000e00]], [[2.5190154e-04]]],
                  [[[6.4247579e-05]], [[1.6061892e-03]]],
                  [[[6.1398384e-04]], [[1.0784689e-03]]],
              ],
          ],
      ),
  )
  def test_gradient(
      self,
      inputs,
      location_parameters,
      scaling_parameters,
      units,
      activation,
      sparsity_factor,
      scaling_exp_transform_multiplier,
      expected,
  ):
    location_parameters = tf.Variable(
        location_parameters,
        trainable=True,
        dtype=tf.float32,
        name="location_parameters",
    )
    scaling_parameters = tf.Variable(
        scaling_parameters,
        trainable=True,
        dtype=tf.float32,
        name="scaling_parameters",
    )

    with tf.GradientTape() as tape:
      y = cdf_fn(
          inputs=tf.constant(inputs, dtype=tf.float32),
          location_parameters=location_parameters,
          scaling_parameters=scaling_parameters,
          reduction="mean",
          units=units,
          activation=activation,
          sparsity_factor=sparsity_factor,
          scaling_exp_transform_multiplier=scaling_exp_transform_multiplier,
      )
      loss = tf.reduce_sum(y * y)
    grads = tape.gradient(loss, [location_parameters, scaling_parameters])
    self.assertAllClose(grads, expected)

  @parameterized.named_parameters(
      dict(
          testcase_name="activation",
          inputs=[[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]], [[-2.0], [0.0], [2.0]]],
              [[[-1.0], [0.0], [1.0]], [[-3.0], [0.0], [3.0]]],
              [[[-1.0], [0.0], [1.0]], [[-4.0], [0.0], [4.0]]],
          ],
          scaling_parameters=[
              [[[1.0]], [[1.0]]],
              [[[2.0]], [[2.0]]],
              [[[5.0]], [[7.0]]],
          ],
          units=2,
          activation="relu",
          reduction="none",
          sparsity_factor=2,
          expected="activation = .* is not supported.*",
      ),
      dict(
          testcase_name="reduction",
          inputs=[[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]], [[-2.0], [0.0], [2.0]]],
              [[[-1.0], [0.0], [1.0]], [[-3.0], [0.0], [3.0]]],
              [[[-1.0], [0.0], [1.0]], [[-4.0], [0.0], [4.0]]],
          ],
          scaling_parameters=None,
          units=2,
          activation="sigmoid",
          reduction="some_reduction",
          sparsity_factor=2,
          expected="reduction = .* is not supported.*",
      ),
      dict(
          testcase_name="input_shape",
          inputs=[-1.0, 0.0],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]], [[-2.0], [0.0], [2.0]]],
              [[[-1.0], [0.0], [1.0]], [[-3.0], [0.0], [3.0]]],
              [[[-1.0], [0.0], [1.0]], [[-4.0], [0.0], [4.0]]],
          ],
          scaling_parameters=[
              [[[1.0]], [[1.0]]],
              [[[2.0]], [[2.0]]],
              [[[5.0]], [[7.0]]],
          ],
          units=2,
          activation="sigmoid",
          reduction="none",
          sparsity_factor=2,
          expected="inputs shape.*is not.*",
      ),
      dict(
          testcase_name="units_and_sparsity_factor",
          inputs=[[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]], [[-2.0], [0.0], [2.0]]],
              [[[-1.0], [0.0], [1.0]], [[-3.0], [0.0], [3.0]]],
              [[[-1.0], [0.0], [1.0]], [[-4.0], [0.0], [4.0]]],
          ],
          scaling_parameters=None,
          units=2,
          activation="sigmoid",
          reduction="mean",
          sparsity_factor=3,
          expected="units.*is not divisible by sparsity_factor.*",
      ),
      dict(
          testcase_name="input_dim_and_sparsity_factor",
          inputs=[[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]], [[-2.0], [0.0], [2.0]]],
              [[[-1.0], [0.0], [1.0]], [[-3.0], [0.0], [3.0]]],
              [[[-1.0], [0.0], [1.0]], [[-4.0], [0.0], [4.0]]],
          ],
          scaling_parameters=[
              [[[1.0]], [[1.0]]],
              [[[2.0]], [[2.0]]],
              [[[5.0]], [[7.0]]],
          ],
          units=3,
          activation="sigmoid",
          reduction="mean",
          sparsity_factor=3,
          expected="input_dim.*is not divisible by sparsity_factor.*",
      ),
      dict(
          testcase_name="location_parameters_shape_1",
          inputs=[[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
          location_parameters=[
              [[-1.0], [0.0], [1.0]],
              [[-2.0], [0.0], [2.0]],
              [[-1.0], [0.0], [1.0]],
              [[-3.0], [0.0], [3.0]],
              [[-1.0], [0.0], [1.0]],
              [[-4.0], [0.0], [4.0]],
          ],
          scaling_parameters=[
              [[[1.0]], [[1.0]]],
              [[[2.0]], [[2.0]]],
              [[[5.0]], [[7.0]]],
          ],
          units=2,
          activation="sigmoid",
          reduction="mean",
          sparsity_factor=2,
          expected="location_parameters shape.*is not.*",
      ),
      dict(
          testcase_name="location_parameters_shape_2",
          inputs=[[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]]],
              [[[-1.0], [0.0], [1.0]]],
              [[[-1.0], [0.0], [1.0]]],
          ],
          scaling_parameters=None,
          units=2,
          activation="sigmoid",
          reduction="mean",
          sparsity_factor=2,
          expected="location_parameters shape.*is not.*",
      ),
      dict(
          testcase_name="location_parameters_shape_3",
          inputs=[[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]], [[-2.0], [0.0], [2.0]]],
              [[[-1.0], [0.0], [1.0]], [[-3.0], [0.0], [3.0]]],
              [[[-1.0], [0.0], [1.0]], [[-4.0], [0.0], [4.0]]],
          ],
          scaling_parameters=None,
          units=2,
          activation="sigmoid",
          reduction="mean",
          sparsity_factor=1,
          expected="location_parameters shape.*is not.*",
      ),
      dict(
          testcase_name="location_and_scaling_shape_1",
          inputs=[[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]], [[-2.0], [0.0], [2.0]]],
              [[[-1.0], [0.0], [1.0]], [[-3.0], [0.0], [3.0]]],
              [[[-1.0], [0.0], [1.0]], [[-4.0], [0.0], [4.0]]],
          ],
          scaling_parameters=[
              [[[1.0]], [[1.0]]],
              [[[2.0]], [[2.0]]],
              [[[5.0]], [[7.0]]],
              [[[5.0]], [[7.0]]],
          ],
          units=2,
          activation="sigmoid",
          reduction="mean",
          sparsity_factor=2,
          expected=(
              "scaling_parameters and location_parameters"
              " likely are not broadcastable.*"
          ),
      ),
      dict(
          testcase_name="location_and_scaling_shape_2",
          inputs=[[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
          location_parameters=[
              [[[-1.0], [0.0], [1.0]], [[-2.0], [0.0], [2.0]]],
              [[[-1.0], [0.0], [1.0]], [[-3.0], [0.0], [3.0]]],
              [[[-1.0], [0.0], [1.0]], [[-4.0], [0.0], [4.0]]],
          ],
          scaling_parameters=[
              [[[1.0, 1.0]], [[1.0, 1.0]]],
              [[[2.0, 1.0]], [[2.0, 1.0]]],
              [[[5.0, 1.0]], [[7.0, 1.0]]],
          ],
          units=2,
          activation="sigmoid",
          reduction="mean",
          sparsity_factor=2,
          expected=(
              "scaling_parameters and location_parameters"
              " likely are not broadcastable.*"
          ),
      ),
  )
  def test_raise(
      self,
      inputs,
      location_parameters,
      scaling_parameters,
      units,
      activation,
      reduction,
      sparsity_factor,
      expected,
  ):
    with self.assertRaisesRegex(ValueError, expected):
      _ = cdf_fn(
          inputs=tf.constant(inputs, dtype=tf.float32),
          location_parameters=tf.constant(
              location_parameters, dtype=tf.float32
          ),
          scaling_parameters=(
              tf.constant(scaling_parameters, dtype=tf.float32)
              if scaling_parameters is not None
              else None
          ),
          units=units,
          reduction=reduction,
          activation=activation,
          sparsity_factor=sparsity_factor,
      )


if __name__ == "__main__":
  tf.test.main()
