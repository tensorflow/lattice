/* Copyright 2017 The TensorFlow Lattice Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Implementations of the piecewise linear "Indexing" calibrator: operators
// related to the calculation of the interpolation weights and gradients.
//
// Sparse and dense implementations.
//
// FutureWork: Zero tensors using functor::SetZeroFunctor (device dependent),
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace lattice {

namespace {

bool IsSameShape(shape_inference::InferenceContext* c,
                 const shape_inference::ShapeHandle& shape1,
                 const shape_inference::ShapeHandle& shape2) {
  if (c->Rank(shape1) != c->Rank(shape2)) return false;
  for (int ii = 0; ii < c->Rank(shape1); ++ii) {
    if (c->Value(c->Dim(shape1, ii)) != c->Value(c->Dim(shape2, ii))) {
      return false;
    }
  }
  return true;
}

}  // namespace

REGISTER_OP("PwlIndexingCalibrator")
    .Input("input: Dtype")
    .Input("kp_inputs: Dtype")
    .Output("weights: Dtype")
    .Attr("Dtype: {float, double} = DT_FLOAT")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_shape));
      shape_inference::DimensionHandle batch_size = c->Dim(input_shape, 0);
      shape_inference::ShapeHandle kp_input_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &kp_input_shape));
      shape_inference::DimensionHandle num_keypoints =
          c->Dim(kp_input_shape, 0);
      auto output_shape = c->Matrix(batch_size, num_keypoints);
      c->set_output(0, output_shape);
      return Status::OK();
    })
    .Doc(R"doc(
Returns tensor representing interpolation weights in a piecewise linear
function. If using a large number of keypoints, try PwlIndexingCalibratorSparse.

Notice that in this version the keypoints inputs (given by kp_inputs) is kept
fixed by forcing its gradient to be always 0. FutureWork: allow kp_inputs to
also be optimized, by providing a gradient.

Inputs
  input: uncalibrated weights, `[batch_size]`
  kp_input: keypoints' input weights, can be initialized with the
            pwl_calibrator_initialize_input_keypoints op. `[num_keypoints]`

Outputs
  weights: Interpolation weights for a piecewise linear function. Its shape is
    `[batch_size, num_keypoints]`. The dot product of this and the keypoints
    output will give the calibrated value.
)doc");

REGISTER_OP("PwlIndexingCalibratorGradient")
    .Input("input: Dtype")
    .Input("kp_inputs: Dtype")
    .Input("grad_wrt_weights: Dtype")
    .Output("grad_wrt_input: Dtype")
    .Output("grad_wrt_kp_inputs: Dtype")
    .Attr("Dtype: {float, double} = DT_FLOAT")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &input_shape));
      shape_inference::DimensionHandle batch_size = c->Dim(input_shape, 0);

      shape_inference::ShapeHandle kp_input_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &kp_input_shape));
      const auto num_keypoints = c->Dim(kp_input_shape, 0);

      auto weights_shape = c->Matrix(batch_size, num_keypoints);
      shape_inference::ShapeHandle grad_wrt_weights_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &grad_wrt_weights_shape));
      if (!IsSameShape(c, weights_shape, grad_wrt_weights_shape)) {
        return errors::InvalidArgument("grad_wrt_weights has shape ",
                                       c->DebugString(grad_wrt_weights_shape),
                                       ", but weights has shape ",
                                       c->DebugString(weights_shape));
      }

      auto grad_wrt_input_shape = c->Vector(batch_size);
      c->set_output(0, grad_wrt_input_shape);
      auto grad_wrt_kp_inputs_shape = c->Vector(num_keypoints);
      c->set_output(1, grad_wrt_kp_inputs_shape);
      return Status::OK();
    })
    .Doc(R"doc(
Computes gradients of PwlIndexingCalibrator. Returns a dense gradient.

As FutureWork we want to allow kp_inputs to be adjusted dynamically.

Inputs
  input: uncalibrated value, `[batch_size]`.
  kp_inputs: keypoints' input weights, can be initialized with the
      pwl_calibrator_initialize_input_keypoints op, `[num_keypoints]`.
  weights_grad: Gradient with respect to the weights outputs of this operator,
      `[batch_size, num_keypoints]`.

Outputs
  grad_wrt_input: gradient with respect to input, `[batch_size]`.
  grad_wrt_kp_inputs: gradient with respect to the kp_inputs. This is fixed in 0
      because (for now) the keypoints inputs are fixed, `[num_keypoints]`.

)doc");

REGISTER_OP("PwlIndexingCalibratorSparse")
    .Input("input: Dtype")
    .Input("kp_inputs: Dtype")
    .Output("indices: int64")
    .Output("weights: Dtype")
    .Attr("Dtype: {float, double} = DT_FLOAT")
    .Doc(R"doc(
Returns sparse tensor representing interpolation weights in a piecewise linear
function.

Inputs
  input: uncalibrated weights, `[batch_size]`
  kp_input: keypoints' input weights, can be initialized with the
            pwl_calibrator_initialize_input_keypoints op. `[num_keypoints]`

Outputs
  indices, weights: Tensors with sparse representation of interpolation weights
    for a piecewise linear function in the form of a SparseTensor. At most two
    weights will be set per uncalibrated value given. This can be multiplied
    by the keypoints' output weights. The tensor will be shaped
    `[batch_size, num_keypoints]`.
)doc");

REGISTER_OP("PwlIndexingCalibratorSparseGradient")
    .Input("input: Dtype")
    .Input("kp_inputs: Dtype")
    .Input("indices: int64")
    .Input("grad_wrt_weights: Dtype")
    .Output("grad_wrt_input: Dtype")
    .Output("grad_wrt_kp_inputs: Dtype")
    .Attr("Dtype: {float, double} = DT_FLOAT")
    .Doc(R"doc(
Computes gradients of PwlIndexingCalibratorSparse. Returns (dense) gradients
with respect to the input and to the kp_inputs.

As FutureWork we want to allow kp_inputs to be adjusted dynamically.

Inputs
  input: uncalibrated value, `[batch_size]`.
  kp_inputs: keypoints' input weights, can be initialized with the
      pwl_calibrator_initialize_input_keypoints op, `[num_keypoints]`.
  indices, weights_grad: indices and weights gradient (gradient
      of the loss function with respect to output weights calculated by
      PwlIndexingCalibratorSparseOp). They are the sparse representation of a
      Tensor of shape `[batch_size, num_keypoints]`.

Outputs
  grad_wrt_input: gradient with respect to input, `[batch_size]`.
  grad_wrt_kp_inputs: gradient with respect to the kp_inputs. This is fixed in 0
      because (for now) the keypoints inputs are fixed, `[num_keypoints]`.
)doc");

}  // namespace lattice
}  // namespace tensorflow
