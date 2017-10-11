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
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace lattice {

REGISTER_OP("MonotoneLattice")
    .Input("lattice_params: Dtype")
    .Output("projected_lattice_params: Dtype")
    .Attr("Dtype: {float, double} = DT_FLOAT")
    .Attr("is_monotone: list(bool) = []")
    .Attr("lattice_sizes: list(int) = []")
    .Attr("tolerance: float = 1e-7")
    .Attr("max_iter: int = 1000")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // Check pre-conditions.
      std::vector<int> lattice_sizes;
      TF_RETURN_IF_ERROR(c->GetAttr("lattice_sizes", &lattice_sizes));
      if (!LatticeStructure::IsValidLatticeSizes(lattice_sizes)) {
        return errors::InvalidArgument(str_util::Join(lattice_sizes, ","),
                                       " is not a valid lattice sizes");
      }
      LatticeStructure lattice_structure(lattice_sizes);

      shape_inference::ShapeHandle lattice_params_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &lattice_params_shape));
      if (c->Value(c->Dim(lattice_params_shape, 1)) !=
          lattice_structure.NumVertices()) {
        return errors::InvalidArgument(
            strings::StrCat("lattice_params' number of parameters (",
                            c->DebugString(c->Dim(lattice_params_shape, 1)),
                            ") != expected number of parameters (",
                            lattice_structure.NumVertices(), ")"));
      }
      // Returns the shape of the output.
      return shape_inference::UnchangedShapeWithRank(c, 2);
    })
    .Doc(R"doc(
Returns a projected lattice parameters onto the monotonicity constraints.

Monotonicity constraints are specified is_monotone. If is_monotone[k] == True,
then the kth input has a non-decreasing monotonicity, otherwise there will be no
constraints.

This operator uses an iterative algorithm, Alternating Direction Method of
Multipliers (ADMM) method, to find the projection, so tolerance and max_iter can
be used to control the accuracy vs. the time spent trade-offs in the ADMM
method.

Inputs
  lattice_params: 2D tensor, `[number of outputs, number of parameters]`

Params
  is_monotone: 1D bool tensor that contains whether the kth dimension should be
  monotonic.
  lattice_sizes: 1D int tensor that contains a lattice size per each dimension,
  [m_0, ..., m_{d - 1}].
  tolerance: The tolerance in ||true projection - projection|| in the ADMM
  method.
  max_iter: Maximum number of iterations in the ADMM method.

Outputs
  projected_lattice_params: 2D tensor,
  `[number of outputs, number of parameters]`, that contains the projected
  parameters.
)doc");

}  // namespace lattice
}  // namespace tensorflow
