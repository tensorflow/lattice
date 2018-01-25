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
// Lattice interpolation base class.
#ifndef TENSORFLOW_LATTICE_CC_KERNELS_LATTICE_INTERPOLATION_BASE_H_
#define TENSORFLOW_LATTICE_CC_KERNELS_LATTICE_INTERPOLATION_BASE_H_

#include <functional>
#include <memory>
#include <vector>

#include "tensorflow_lattice/cc/lib/lattice_structure.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace lattice {

template <typename Dtype>
struct InterpolationWeights {
  std::vector<int64> indices;
  std::vector<Dtype> weights;
};

// LatticeOpBase class contains common part of all lattice operators as lattice
// structure initialization.
class LatticeOpBase : public OpKernel {
 public:
  explicit LatticeOpBase(OpKernelConstruction* context);

  // Returns the lattice_structure.
  const LatticeStructure& GetLatticeStructure() const {
    return *lattice_structure_;
  }

  // Check whether the shape of tensor is same with expected_shape.
  void CheckShape(OpKernelContext* context, const Tensor& tensor,
                  const std::vector<int64>& expected_shape) const;

  // Cost per example.
  const int64 CostPerExample() const { return cost_per_example_; }
  void SetCostPerExample(const int64 cost_per_example) {
    cost_per_example_ = cost_per_example;
  }

 private:
  std::unique_ptr<LatticeStructure> lattice_structure_;
  int64 cost_per_example_;
};

// LatticeInterpolationOpBase is a base class for
// HypercubeInterpolationOpKernel and SimplexInterpolationOpKernel.
// The InterpolationWeights computation should be implemented in
// ComputeInterpolationWeights method.
template <typename Dtype>
class LatticeInterpolationOpBase : public LatticeOpBase {
 public:
  explicit LatticeInterpolationOpBase(OpKernelConstruction* context)
      : LatticeOpBase(context) {}

  void Compute(OpKernelContext* context) override;

 protected:
  virtual InterpolationWeights<Dtype> ComputeInterpolationWeights(
      const LatticeStructure& lattice_structure,
      typename TTypes<Dtype>::UnalignedConstFlat input_vector) const = 0;

 private:
  // Apply InterpolationWeights to each slice of tensors.
  void BatchInterpolationWorker(const Tensor& input_tensor, const int start,
                                const int limit,
                                Tensor* interpolation_weights_tensor) const;
};

template <typename Dtype>
void LatticeInterpolationOpBase<Dtype>::BatchInterpolationWorker(
    const Tensor& input_tensor, const int start, const int limit,
    Tensor* interpolation_weights_tensor) const {
  for (int ii = start; ii < limit; ++ii) {
    // Get iith input vector.
    const auto input_row_ii = input_tensor.Slice(ii, ii + 1);

    // Compute weight-index pairs.
    const InterpolationWeights<Dtype> interpolation_weights =
        ComputeInterpolationWeights(GetLatticeStructure(),
                                    input_row_ii.unaligned_flat<Dtype>());

    // Get iith interpolation weight vector (output).
    auto interpolation_weights_row_ii =
        interpolation_weights_tensor->Slice(ii, ii + 1).unaligned_flat<Dtype>();

    // Assign values to interpolation weight vector.
    interpolation_weights_row_ii.setZero();
    DCHECK_EQ(interpolation_weights.indices.size(),
              interpolation_weights.weights.size());
    for (int jj = 0; jj < interpolation_weights.indices.size(); ++jj) {
      interpolation_weights_row_ii(interpolation_weights.indices[jj]) =
          interpolation_weights.weights[jj];
    }
  }
}

template <typename Dtype>
void LatticeInterpolationOpBase<Dtype>::Compute(OpKernelContext* context) {
  const LatticeStructure& lattice_structure = GetLatticeStructure();
  // Grab the input tensor.
  const Tensor& input_tensor = context->input(0);
  // Check the shapes.
  const int64 batch_dim = input_tensor.dim_size(0);
  const int64 input_dim = lattice_structure.Dimension();
  CheckShape(context, input_tensor, {batch_dim, input_dim});

  // Allocate interpolation_weights_tensor.
  Tensor* interpolation_weights_tensor = nullptr;
  OP_REQUIRES_OK(
      context,
      context->allocate_output(
          0, TensorShape({batch_dim, lattice_structure.NumVertices()}),
          &interpolation_weights_tensor));

  auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());

  // Launch threads.
  Shard(worker_threads.num_threads, worker_threads.workers, batch_dim,
        CostPerExample(), [&](int start, int limit) {
          BatchInterpolationWorker(input_tensor, start, limit,
                                   interpolation_weights_tensor);
        });
}

// LatticeGradientOpBase is a base class for HypercubeGradientOpKernel and
// SimplexGradientOpKernel.
// Computing Gradient with respect to input should be should be implemented in
// ComputeGradWrtInput method.
template <typename Dtype>
class LatticeGradientOpBase : public LatticeOpBase {
 public:
  explicit LatticeGradientOpBase(OpKernelConstruction* context)
      : LatticeOpBase(context) {}

  void Compute(OpKernelContext* context) override;

 protected:
  virtual std::vector<Dtype> ComputeGradWrtInput(
      const LatticeStructure& lattice_structure,
      typename TTypes<Dtype>::UnalignedConstFlat input_vector,
      typename TTypes<Dtype>::UnalignedConstFlat weight_vector,
      typename TTypes<Dtype>::UnalignedConstFlat grad_wrt_weight_vector)
      const = 0;

 private:
  // Apply grad_wrt_input_fn_ to each slice of tensors.
  void BatchGradientWorker(const Tensor& input_tensor,
                           const Tensor& weight_tensor,
                           const Tensor& grad_wrt_weight_tensor,
                           const int start, const int limit,
                           Tensor* grad_wrt_input_tensor) const;
};

// BatchGradientWorker computes the gradient with respect to the input of each
// row.
template <typename Dtype>
void LatticeGradientOpBase<Dtype>::BatchGradientWorker(
    const Tensor& input_tensor, const Tensor& weight_tensor,
    const Tensor& grad_wrt_weight_tensor, const int start, const int limit,
    Tensor* grad_wrt_input_tensor) const {
  auto grad_wrt_input_matrix = grad_wrt_input_tensor->matrix<Dtype>();
  for (int ii = start; ii < limit; ++ii) {
    const auto input_row_ii = input_tensor.Slice(ii, ii + 1);
    const auto weight_row_ii = weight_tensor.Slice(ii, ii + 1);
    const auto grad_wrt_weight_row_ii =
        grad_wrt_weight_tensor.Slice(ii, ii + 1);

    const std::vector<Dtype> grad_wrt_input = ComputeGradWrtInput(
        GetLatticeStructure(), input_row_ii.unaligned_flat<Dtype>(),
        weight_row_ii.unaligned_flat<Dtype>(),
        grad_wrt_weight_row_ii.unaligned_flat<Dtype>());

    for (int jj = 0; jj < grad_wrt_input.size(); ++jj) {
      grad_wrt_input_matrix(ii, jj) = grad_wrt_input[jj];
    }
  }
}

template <typename Dtype>
void LatticeGradientOpBase<Dtype>::Compute(OpKernelContext* context) {
  const LatticeStructure& lattice_structure = this->GetLatticeStructure();
  const Tensor& input_tensor = context->input(0);
  const Tensor& weight_tensor = context->input(1);
  const Tensor& grad_wrt_weight_tensor = context->input(2);
  // Check the shapes.
  const int64 batch_dim = input_tensor.dim_size(0);
  const int64 input_dim = lattice_structure.Dimension();
  CheckShape(context, input_tensor, {batch_dim, input_dim});
  CheckShape(context, weight_tensor,
             {batch_dim, lattice_structure.NumVertices()});
  CheckShape(context, grad_wrt_weight_tensor,
             {batch_dim, lattice_structure.NumVertices()});

  // Dense implementation.
  Tensor* grad_wrt_input_tensor = nullptr;
  OP_REQUIRES_OK(
      context,
      context->allocate_output(0, TensorShape({batch_dim, input_dim}),
                               &grad_wrt_input_tensor));

  auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());

  // Launch threads.
  Shard(worker_threads.num_threads, worker_threads.workers, batch_dim,
        CostPerExample(), [&](int start, int limit) {
          BatchGradientWorker(input_tensor, weight_tensor,
                              grad_wrt_weight_tensor, start, limit,
                              grad_wrt_input_tensor);
        });
}

}  // namespace lattice
}  // namespace tensorflow

#endif  // TENSORFLOW_LATTICE_CC_KERNELS_LATTICE_INTERPOLATION_BASE_H_
