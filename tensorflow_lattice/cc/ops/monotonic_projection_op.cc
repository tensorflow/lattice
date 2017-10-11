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
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {
namespace lattice {

REGISTER_OP("MonotonicProjection")
    .Input("values: Dtype")
    .Input("increasing: bool")
    .Output("monotonic: Dtype")
    .Attr("Dtype: {float, double} = DT_FLOAT")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // Input must be a vector, and output is the same shape as input.
      shape_inference::ShapeHandle values_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &values_shape));
      shape_inference::ShapeHandle increasing_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &increasing_shape));

      c->set_output(0, values_shape);
      return Status::OK();
    })
    .Doc(R"doc(
Returns a not-strict monotonic projection of the vector.

The returned vector is of the same size as the input and values (optionally)
changed to make them monotonically, minimizing the sum of the square distance
to the original values.

This is part of the set of ops that support monotonicity in piecewise-linear
calibration.

Note that the gradient is undefined for this function.

  values: `Tensor` with values to be made monotonic.
  increasing: Defines if projection it to monotonic increasing values
    or to monotonic decreasing ones.

  monotonic: output `Tensor` with values made monotonic.
)doc");

}  // namespace lattice
}  // namespace tensorflow
