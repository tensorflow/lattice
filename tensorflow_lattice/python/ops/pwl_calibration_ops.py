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
"""Piecewise-linear calibration ops.

Piecewise-linear calibration works particularly well with lattice models, and
is therefore part of the "TensorFlow Lattice" package.

But it can be used in conjunction with other types of models as well, in
particular with linear models: it increases its power without breaking
independence of the variables (desirable in some situations).

This file exports the basic graph operations used for calibrators. See
pwl_calibration_layers.py for more details and higher level calibration
functions, for building models.
"""
# pylint: disable=unused-import
from tensorflow_lattice.python.ops.gen_monotonic_projection import monotonic_projection
from tensorflow_lattice.python.ops.gen_pwl_indexing_calibrator import pwl_indexing_calibrator
from tensorflow_lattice.python.ops.gen_pwl_indexing_calibrator import pwl_indexing_calibrator_gradient
from tensorflow_lattice.python.ops.gen_pwl_indexing_calibrator import pwl_indexing_calibrator_sparse
from tensorflow_lattice.python.ops.gen_pwl_indexing_calibrator import pwl_indexing_calibrator_sparse_gradient
# pylint: enable=unused-import

from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader

_pwl_calibration_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile(
        '../../cc/ops/_pwl_calibration_ops.so'))


@ops.RegisterGradient('PwlIndexingCalibrator')
def _pwl_indexing_calibrator_grad(op, grad_wrt_weights):
  """Register gradient for PwlIndexingCalibrator."""
  grad_wrt_input, grad_wrt_kp_inputs = pwl_indexing_calibrator_gradient(
      input=op.inputs[0],
      kp_inputs=op.inputs[1],
      grad_wrt_weights=grad_wrt_weights)
  return [grad_wrt_input, grad_wrt_kp_inputs]


@ops.RegisterGradient('PwlIndexingCalibratorSparse')
def _pwl_indexing_calibrator_sparse_grad(op, unused_grad_wrt_indices,
                                         grad_wrt_weights):
  """Register gradient for PwlIndexingCalibratorSparse."""
  # unused_grad_wrt_indices is None and not used. But the optimizers do pass
  # the extra parameter, so it needs to be there.
  grad_wrt_input, grad_wrt_params = pwl_indexing_calibrator_sparse_gradient(
      input=op.inputs[0],
      kp_inputs=op.inputs[1],
      indices=op.outputs[0],
      grad_wrt_weights=grad_wrt_weights)
  return [grad_wrt_input, grad_wrt_params]
