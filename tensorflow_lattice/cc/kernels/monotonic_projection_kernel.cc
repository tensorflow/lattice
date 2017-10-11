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
#include <functional>
#include <vector>

#include "tensorflow_lattice/cc/kernels/monotonic_projections.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace lattice {

namespace {

template <typename Dtype>
bool CmpLesserOrEqual(const Dtype a, const Dtype b) {
  return a <= b;
}

template <typename Dtype>
bool CmpGreaterOrEqual(const Dtype a, const Dtype b) {
  return a >= b;
}

}  // namespace

template <typename Dtype>
class MonotonicProjectionOpKernel : public OpKernel {
 public:
  explicit MonotonicProjectionOpKernel(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& values_tensor = context->input(0);
    const Tensor& increasing_tensor = context->input(1);

    OP_REQUIRES(
        context, values_tensor.dims() == 1,
        errors::InvalidArgument("values must have dims=1, got values.dims=",
                                values_tensor.dims()));
    OP_REQUIRES(context, increasing_tensor.dims() == 0,
                errors::InvalidArgument(
                    "increasing must be a boolean scalar, got increasing.dims=",
                    increasing_tensor.dims()));
    OP_REQUIRES(
        context, increasing_tensor.dtype() == DT_BOOL,
        errors::InvalidArgument(
            "increasing must be a boolean scalar, got increasing.dtype=",
            DataType_Name(increasing_tensor.dtype())));

    Tensor* monotonic_tensor = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(0, values_tensor.shape(), &monotonic_tensor));

    // Copy the current non-monotonic values and project them to monotonicity.
    *monotonic_tensor = values_tensor;
    if (increasing_tensor.scalar<bool>()()) {
      TensorVectorMonotonicProjection<Dtype>(monotonic_tensor->vec<Dtype>(),
                                             CmpLesserOrEqual<Dtype>);
    } else {
      TensorVectorMonotonicProjection<Dtype>(monotonic_tensor->vec<Dtype>(),
                                             CmpGreaterOrEqual<Dtype>);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MonotonicProjection")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("Dtype"),
                        MonotonicProjectionOpKernel<float>);
REGISTER_KERNEL_BUILDER(Name("MonotonicProjection")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("Dtype"),
                        MonotonicProjectionOpKernel<double>);

}  // namespace lattice
}  // namespace tensorflow
