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
#include "tensorflow_lattice/cc/kernels/lattice_interpolation_base.h"

#include <vector>

#include "tensorflow_lattice/cc/lib/lattice_structure.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {
namespace lattice {

using errors::InvalidArgument;
using str_util::Join;

LatticeOpBase::LatticeOpBase(OpKernelConstruction* context)
    : OpKernel(context), cost_per_example_(1.0) {
  std::vector<int> lattice_sizes;
  OP_REQUIRES_OK(context, context->GetAttr("lattice_sizes", &lattice_sizes));
  OP_REQUIRES(context, LatticeStructure::IsValidLatticeSizes(lattice_sizes),
              InvalidArgument(Join(lattice_sizes, ","),
                              " is not a valid lattice size"));
  lattice_structure_ =
      std::unique_ptr<LatticeStructure>(new LatticeStructure(lattice_sizes));
}

void LatticeOpBase::CheckShape(OpKernelContext* context, const Tensor& tensor,
                               const std::vector<int64>& expected_shape) const {
  OP_REQUIRES(context, tensor.dims() == expected_shape.size(),
              InvalidArgument("expect rank ", expected_shape.size(), "but got ",
                              tensor.DebugString()));

  for (int ii = 0; ii < expected_shape.size(); ++ii) {
    OP_REQUIRES(context, tensor.dim_size(ii) == expected_shape[ii],
                InvalidArgument("expect ", ii, "-dim: ", expected_shape[ii],
                                "but got ", tensor.DebugString()));
  }
}

}  // namespace lattice
}  // namespace tensorflow
