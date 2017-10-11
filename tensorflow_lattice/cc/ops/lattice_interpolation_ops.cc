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
#include <vector>

#include "tensorflow_lattice/cc/lib/lattice_structure.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace lattice {

namespace {
Status InterpolationShapeFn(shape_inference::InferenceContext* c) {
  std::vector<int> lattice_sizes;
  TF_RETURN_IF_ERROR(c->GetAttr("lattice_sizes", &lattice_sizes));
  if (!LatticeStructure::IsValidLatticeSizes(lattice_sizes)) {
    return errors::InvalidArgument(str_util::Join(lattice_sizes, ","),
                                   " is not a valid lattice sizes");
  }

  // input_shape = [?,lattice_sizes.size()].
  shape_inference::ShapeHandle input_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));
  shape_inference::DimensionHandle unused_lattice_input_size;
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(input_shape, 1), lattice_sizes.size(),
                                  &unused_lattice_input_size));

  shape_inference::DimensionHandle batch_size = c->Dim(input_shape, 0);
  LatticeStructure lattice_structure(lattice_sizes);
  c->set_output(0, c->Matrix(batch_size, lattice_structure.NumVertices()));

  return Status::OK();
}

Status GradWrtInputShapeFn(shape_inference::InferenceContext* c) {
  std::vector<int> lattice_sizes;
  TF_RETURN_IF_ERROR(c->GetAttr("lattice_sizes", &lattice_sizes));
  if (!LatticeStructure::IsValidLatticeSizes(lattice_sizes)) {
    return errors::InvalidArgument(str_util::Join(lattice_sizes, ","),
                                   " is not a valid lattice sizes");
  }
  LatticeStructure lattice_structure(lattice_sizes);

  // input_shape = [?,lattice_sizes.size()].
  shape_inference::ShapeHandle input_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));
  shape_inference::DimensionHandle batch_size = c->Dim(input_shape, 0);
  shape_inference::DimensionHandle input_size;
  TF_RETURN_IF_ERROR(
      c->WithValue(c->Dim(input_shape, 1), lattice_sizes.size(), &input_size));

  // weight_shape = [?,LatticeStructure.NumVertcies()].
  shape_inference::ShapeHandle weight_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &weight_shape));
  if (c->Value(c->Dim(weight_shape, 0)) != c->Value(c->Dim(input_shape, 0))) {
    return errors::InvalidArgument(strings::StrCat(
        "Input batch size (", c->DebugString(c->Dim(input_shape, 0)),
        ") != Weight batch size (", c->DebugString(c->Dim(weight_shape, 0)),
        ")"));
  }
  shape_inference::DimensionHandle unused_weight_size;
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(weight_shape, 1),
                                  lattice_structure.NumVertices(),
                                  &unused_weight_size));

  // grad_wrt_weight_shape = [?,LatticeStructure.NumVertcies()].
  shape_inference::ShapeHandle grad_wrt_weight_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &grad_wrt_weight_shape));
  if ((c->Value(c->Dim(weight_shape, 0)) !=
       c->Value(c->Dim(grad_wrt_weight_shape, 0))) ||
      (c->Value(c->Dim(weight_shape, 1)) !=
       c->Value(c->Dim(grad_wrt_weight_shape, 1)))) {
    return errors::InvalidArgument(
        strings::StrCat("Weight shape (", c->DebugString(weight_shape),
                        ") != GradWrtWeight shape (",
                        c->DebugString(grad_wrt_weight_shape), ")"));
  }

  c->set_output(0, c->Matrix(batch_size, input_size));

  return Status::OK();
}
}  // namespace

REGISTER_OP("HypercubeInterpolation")
    .Input("input: Dtype")
    .Output("weights: Dtype")
    .Attr("Dtype: {float, double} = DT_FLOAT")
    .Attr("lattice_sizes: list(int) = []")
    .SetShapeFn(InterpolationShapeFn)
    .Doc(R"doc(
Returns a tensor representing interpolation weights in a hypercube lattice
interpolation.

Inputs
  input: 2D tensor, `[?, d]`

Params
  lattice_sizes: 1D int tensor that contains a lattice size per each dimension,
  [m_0, ..., m_{d - 1}].

Outputs
  weights: 2D tensor that contains interpolation weights.
  [?, m_0 x m_1 ... x m_{d - 1}].
)doc");

REGISTER_OP("HypercubeGradient")
    .Input("input: Dtype")
    .Input("weight: Dtype")
    .Input("grad_wrt_weight: Dtype")
    .Output("grad_wrt_input: Dtype")
    .Attr("Dtype: {float, double} = DT_FLOAT")
    .Attr("lattice_sizes: list(int) = []")
    .SetShapeFn(GradWrtInputShapeFn)
    .Doc(R"doc(
Computes gradients of HypercubeInterpolation. Returns a dense gradient.

Inputs
  input: input tensor, `[?, d]`.
  grad_wrt_weight: Gradient with respect to the outputs of this operator,
  `[?, m_0 x m_1 x .. x m_{d - 1}]`

Outputs
  grad_wrt_input: A gradient tensor, `[?, d]`, with respect to input.
)doc");

REGISTER_OP("SimplexInterpolation")
    .Input("input: Dtype")
    .Output("weights: Dtype")
    .Attr("Dtype: {float, double} = DT_FLOAT")
    .Attr("lattice_sizes: list(int) = []")
    .SetShapeFn(InterpolationShapeFn)
    .Doc(R"doc(
Returns a tensor representing interpolation weights in a simplex lattice
interpolation.

Inputs
  input: 2D tensor, `[?, d]`

Params
  lattice_sizes: 1D int tensor that contains a lattice size per each dimension,
  [m_0, ..., m_{d - 1}].

Outputs
  weights: 2D tensor that contains interpolation weights.
  [?, m_0 x m_1 ... x m_{d - 1}].
)doc");

REGISTER_OP("SimplexGradient")
    .Input("input: Dtype")
    .Input("weight: Dtype")
    .Input("grad_wrt_weight: Dtype")
    .Output("grad_wrt_input: Dtype")
    .Attr("Dtype: {float, double} = DT_FLOAT")
    .Attr("lattice_sizes: list(int) = []")
    .SetShapeFn(GradWrtInputShapeFn)
    .Doc(R"doc(
Computes gradients of SimplexInterpolation. Returns a dense gradient.

Inputs
  input: input tensor, `[?, d]`.
  grad_wrt_weight: Gradient with respect to the outputs of this operator,
  `[?, m_0 x m_1 x .. x m_{d - 1}]`

Outputs
  grad_wrt_input: A gradient tensor, `[?, d]`, with respect to input.
)doc");

}  // namespace lattice
}  // namespace tensorflow
