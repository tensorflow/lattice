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
#include <type_traits>
#include <vector>

#include "tensorflow_lattice/cc/kernels/lattice_interpolation_base.h"
#include "tensorflow_lattice/cc/kernels/monotonic_lattice_projections.h"
#include "tensorflow_lattice/cc/lib/lattice_structure.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace lattice {

// MonotoneLatticeOp returns the projected lattice param vectors onto the
// subspace that satisfies monotonicity constraints specified by is_monotone.
// If is_monotone[k] == true, then kth input will have a non-decreasing
// monotonicity constraint, and is_monotone[k] == false, then then kth input has
// no monotonicity constraints.
//
// Lattice param tensor is expected to be a 2d tensor, [num_outputs,
// num_parameters], where each row represents a parameter from multi-cell
// lattice.
template <typename Dtype>
class MonotoneLatticeOp : public LatticeOpBase {
 public:
  static_assert(std::is_floating_point<Dtype>::value,
                "Dtype needs to be a floating point");

  explicit MonotoneLatticeOp(OpKernelConstruction* context);
  void Compute(OpKernelContext* context) final;

  TF_DISALLOW_COPY_AND_ASSIGN(MonotoneLatticeOp);

 private:
  void ProjectionWorker(const Tensor& lattice_params_tensor, int start,
                        int limit, int num_parameters,
                        Tensor* projection_tensor,
                        OpKernelContext* context) const;

  std::unique_ptr<MonotoneLatticeProjector<Dtype>> projector_;
};

template <typename Dtype>
MonotoneLatticeOp<Dtype>::MonotoneLatticeOp(OpKernelConstruction* context)
    : LatticeOpBase(context) {
  std::vector<bool> is_monotone;
  float tolerance;
  int64 max_iter;

  OP_REQUIRES_OK(context, context->GetAttr("is_monotone", &is_monotone));
  OP_REQUIRES_OK(context, context->GetAttr("tolerance", &tolerance));
  OP_REQUIRES_OK(context, context->GetAttr("max_iter", &max_iter));

  const int64 lattice_dim = GetLatticeStructure().Dimension();
  OP_REQUIRES(context, (is_monotone.size() == lattice_dim),
              errors::InvalidArgument(
                  "lattice dimension :", lattice_dim,
                  " != ", "is_monotone dimension: ", is_monotone.size()));

  std::vector<int> monotone_dims;
  for (int ii = 0; ii < lattice_dim; ++ii) {
    if (is_monotone[ii]) {
      monotone_dims.push_back(ii);
    }
  }

  projector_ = std::unique_ptr<MonotoneLatticeProjector<Dtype>>(
      new MonotoneLatticeProjector<Dtype>(GetLatticeStructure(), monotone_dims,
                                          tolerance, max_iter));


  constexpr int64 kInitCost = 20;
  constexpr int64 kBaseCost = 20;
  constexpr int64 kConstraintCost = 20;
  // For initilaization: constant0 * GetLatticeStructure().NumVertices().
  // Each iteration in ADMM:
  //   1. Projection for each constraint: constant1 * NumVertices().
  //   2. Center variable update: constant2 * NumVertices()
  //   3. Dual variable update for each constraint: constant3 *
  //   NumVertices().
  // Therefore, the total cost of each iteration is
  //  ((constant1 + constant3) * number of monotone dimensions + constant2) *
  //  NumVertices().
  // The number of iteration is bounded by min(max_iter, O(||true_projection -
  // initial_point||_2/epsilon)). But since the latter is hard to obtain, we use
  // max_iter as an upper bound.
  // So the total cost is given by
  //
  // ((kConstraintCost * monotone_dims.size() + kBaseCost) * max_iter +
  // kInitCost) * GetLatticeStructure().NumVertices()
  const int64 cost_per_example =
      ((kConstraintCost * monotone_dims.size() + kBaseCost) * max_iter +
       kInitCost) *
      GetLatticeStructure().NumVertices();
  SetCostPerExample(cost_per_example);
}

template <typename Dtype>
void MonotoneLatticeOp<Dtype>::Compute(OpKernelContext* context) {
  // Grab the param tensor. Expect [num_ouputs, num_parameters] tensor.
  const Tensor& lattice_params_tensor = context->input(0);

  OP_REQUIRES(context, lattice_params_tensor.dims() == 2,
              errors::InvalidArgument("expected a 2d tensor, got ",
                                      lattice_params_tensor.dims()));
  OP_REQUIRES(
      context,
      lattice_params_tensor.dim_size(1) == GetLatticeStructure().NumVertices(),
      errors::InvalidArgument(
          "expected parameter dimension: ", GetLatticeStructure().NumVertices(),
          "got: ", lattice_params_tensor.dim_size(1)));
  const int64 num_outputs = lattice_params_tensor.dim_size(0);
  const int64 num_parameters = lattice_params_tensor.dim_size(1);

  Tensor* projection_tensor = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(
                              0, TensorShape({num_outputs, num_parameters}),
                              &projection_tensor));

  auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());

  // A worker that projects lattice_params_tensor[start : start + limit - 1, :]
  // and saves the result to from the
  // projection_tensor[start : start + limit - 1, :].
  // This lambda captures everything including "this" to use ProjectionWorker
  // method and all of captured states' lifetime is longer than Shard operation.
  auto worker = [&](int start, int limit) {
    ProjectionWorker(lattice_params_tensor, start, limit, num_parameters,
                     projection_tensor, context);
  };
  // Launch threads.
  Shard(worker_threads.num_threads, worker_threads.workers, num_outputs,
        CostPerExample(), worker);
}

template <typename Dtype>
void MonotoneLatticeOp<Dtype>::ProjectionWorker(
    const Tensor& lattice_params_tensor, const int start, const int limit,
    const int num_parameters, Tensor* projection_tensor,
    OpKernelContext* context) const {
  auto lattice_params_matrix = lattice_params_tensor.matrix<Dtype>();
  auto projection_matrix = projection_tensor->matrix<Dtype>();
  for (int row = start; row < limit; ++row) {
    // Computing the projection per each row.
    std::vector<Dtype> lattice_params_vec(num_parameters);
    std::vector<Dtype> projected_lattice_params_vec(num_parameters, 0.0);

    // Fetching the lattice parameter.
    for (int ii = 0; ii < num_parameters; ++ii) {
      lattice_params_vec[ii] = lattice_params_matrix(row, ii);
    }
    OP_REQUIRES_OK(context, projector_->Project(lattice_params_vec,
                                                &projected_lattice_params_vec));
    // Fill-in projected params.
    for (int ii = 0; ii < num_parameters; ++ii) {
      projection_matrix(row, ii) = projected_lattice_params_vec[ii];
    }
  }
}

// Register kernels for float and double.
REGISTER_KERNEL_BUILDER(
    Name("MonotoneLattice").Device(DEVICE_CPU).TypeConstraint<float>("Dtype"),
    MonotoneLatticeOp<float>);
REGISTER_KERNEL_BUILDER(
    Name("MonotoneLattice").Device(DEVICE_CPU).TypeConstraint<double>("Dtype"),
    MonotoneLatticeOp<double>);

}  // namespace lattice
}  // namespace tensorflow
