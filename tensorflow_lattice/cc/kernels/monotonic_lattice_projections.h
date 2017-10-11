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
// Project lattice parameter vector onto monotonicity constraints.
#ifndef TENSORFLOW_LATTICE_CC_KERNELS_MONOTONIC_LATTICE_PROJECTIONS_H_
#define TENSORFLOW_LATTICE_CC_KERNELS_MONOTONIC_LATTICE_PROJECTIONS_H_

#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "tensorflow_lattice/cc/kernels/lattice_raw_iterator.h"
#include "tensorflow_lattice/cc/kernels/monotonic_projections.h"
#include "tensorflow_lattice/cc/lib/lattice_structure.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace lattice {

// Monotone Lattice projector projects lattice parameter to monotonicity
// constraints specified by monotone_dimensions.
// monotone_dimensions contains a index of (increasing) monotonic dimension.
// For example, if we want to impose the monotonicity constraint in the 0th and
// 2th dimensions, then monotone_dimensions = {0, 2}.
//
// The implementation uses Alternating Direction Method of Multipliers (ADMM)
// parallel projection. See Distributed Optimization and Statistical Learning
// via the Alternating Direction Method of Multipliers
// (http://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf) Section
// 5.1.2. Parallel projeciton and Chapter 7. for the theoritical background.
//
// Suppose we have K number of convex sets, C_1, ..., C_K, and we want to
// project a variable x_0 in R^n to the intersection of C_1, ..., C_K.
// Let x_1, ..., x_K in R^n, and d_1, ..., d_K in R^n.
// The ADMM parallel projection works as follows.
//
//  Step 0: Initialize x_center = x_0, and d_k = 0 for all k = 1, ..., K.
//  Step 1: x_k <- Projection of (d_k + x_center) onto C_i for all i = 1, ...,
//  K.
//  Step 2: x_center <- 0.5 * x_0 + 0.5 * 1/K * sum_k (x_k - d_k).
//  Step 3: d_k <- d_k + x_center - x_k.
//  Step 4: Go back to the Step 1 if sum_k ||x_center - x_k||_1 > eps.
//
// Step 1 generates x_k in C_k. However, x_k may not be in C_i where i \neq k.
// However, the algorithm is guaranteed to converge, which implies d_k should
// stop being updated after many iterations.
// Therefore, x_center == x_k for all k eventually. Since x_center == x_1 == ...
// == x_K, we can conclude that x_center is in the intersection of C_1, ...,
// C_K.
// Step 2 generates x_center that minimizes ||x_center - x_0||_2^2 + some
// regularization terms. Upon convergence, regularization temrs are zero.
// Therefore, x_center == the projeciton of x_0 onto the intersection of C_1,
// ..., C_K, when the algorithm converges.
//
// In the following implementation, we set each C_k to be the set of
// lattice_param_vec that satisfies one dimensional monotonicity constraint.
// Assuming we have K number of monotone dimensions, the ADMM algorithm
// perform the projection for a given lattice_param_vec as follows:
//   Step 0: Initialize center = lattice_param_vec and duals[k] =
//     std::vector<Dtype>(param_size, 0.0) for k = 0, ..., K - 1.
//   Step 1: params[k] <- Projection of (duals[k] + center) onto the kth 1D
//   monotonicity constraint. (Here + means an elementwise summation.) for k =
//   0, ..., K - 1.
//   Step 2: center <- 0.5 * lattice_param_vec + 0.5 * 1/K * sum_k (params[k] -
//     duals[k])
//   Step 3: duals[k] += (center - params[k]) for k = 0, ..., K - 1.
//   Step 4: Repeat Step 1 until sum_k ||center - params[k]||_1 < epsilon, or
//     Step 1 was repeated more than max_iter times.
template <typename Dtype>
class MonotoneLatticeProjector {
 public:
  static_assert(std::is_floating_point<Dtype>::value,
                "Dtype needs to be a floating point");

  explicit MonotoneLatticeProjector(const LatticeStructure& lattice_structure,
                                    const std::vector<int>& monotone_dimensions,
                                    const Dtype epsilon = 1e-7,
                                    const int64 max_iter = 100000);

  // Apply ADMM projections, and save the result to the projected_param.
  Status Project(const std::vector<Dtype>& lattice_param_vec,
                 std::vector<Dtype>* projected_lattice_param_vec) const;

 private:
  // This projector computes the projection of lattice parameter vector onto the
  // per dimension monotonicity constraints.
  //
  // For example, consider 3 x 3 lattice:
  //
  // 2---------5--------8
  // |         |        |
  // |         |        |
  // 1---------4--------7
  // |         |        |
  // |         |        |
  // 0---------3--------6
  //
  // For the 0th dimension, we have
  //   weight[0] <= weight[3] <= weight[6]
  //   weight[1] <= weight[4] <= weight[7]
  //   weight[2] <= weight[5] <= weight[8].
  //
  // So PerDimensionProjector(lattice_structure, 0) will project the
  // lattice_param_vec onto the constraints of the given dimension.
  //
  // For the 1th dimension, we have
  //   weight[0] <= weight[1] <= weight[2]
  //   weight[3] <= weight[4] <= weight[5]
  //   weight[6] <= weight[7] <= weight[8].
  //
  // So PerDimensionProjector(lattice_structure, 1) will project
  // lattice_param_vec onto the constraints of the given dimension.
  class PerDimensionProjector {
   public:
    explicit PerDimensionProjector(const LatticeStructure& lattice_structure,
                                   const int64 dimension);

    // Apply projection, and save the result to the lattice_param_vec.
    void Project(std::vector<Dtype>* lattice_param_vec) const;

   private:
    // Helper function that returns the base indices of a given LatticeStructure
    // and dimension.
    static std::vector<int64> BaseIndices(
        const LatticeStructure& lattice_structure, const int64 dimension);

    const int64 lattice_size_;
    const int64 stride_;
    const std::vector<int64> base_indices_;
  };

  const Dtype epsilon_;
  const int64 max_iter_;
  int64 param_size_;
  std::vector<PerDimensionProjector> projectors_;
};

// Implementation of PerDimensionProjector's methods.
template <typename Dtype>
MonotoneLatticeProjector<Dtype>::PerDimensionProjector::PerDimensionProjector(
    const LatticeStructure& lattice_structure, const int64 dimension)
    : lattice_size_(lattice_structure.LatticeSize(dimension)),
      stride_(lattice_structure.Stride(dimension)),
      base_indices_(BaseIndices(lattice_structure, dimension)) {}

template <typename Dtype>
std::vector<int64>
MonotoneLatticeProjector<Dtype>::PerDimensionProjector::BaseIndices(
    const LatticeStructure& lattice_structure, const int64 dimension) {
  std::vector<int64> base_indices;

  for (LatticeRawIterator iter(lattice_structure); !iter.IsDone();
       iter.Next()) {
    if (iter.VertexDim(dimension) == 0) {
      base_indices.push_back(iter.Index());
    }
  }
  return base_indices;
}


template <typename Dtype>
void MonotoneLatticeProjector<Dtype>::PerDimensionProjector::Project(
    std::vector<Dtype>* lattice_param_vec_ptr) const {
  DCHECK(lattice_param_vec_ptr);

  std::vector<Dtype>& lattice_param_vec = *lattice_param_vec_ptr;
  for (const int64 base_index : base_indices_) {
    std::vector<Dtype> lattice_slice(lattice_size_);
    // Find the slice of lattice parameter vector.
    int64 current_index = base_index;
    for (Dtype& value : lattice_slice) {
      value = lattice_param_vec[current_index];
      current_index += stride_;
    }

    // Make a projection.
    std::vector<Dtype> projected_slice =
        VectorMonotonicProjection(lattice_slice, std::less_equal<Dtype>());

    // Fill in the result.
    current_index = base_index;
    for (const Dtype value : projected_slice) {
      lattice_param_vec[current_index] = value;
      current_index += stride_;
    }
  }
}

// Implementation of MonotoneLatticeProjector's methods.
template <typename Dtype>
MonotoneLatticeProjector<Dtype>::MonotoneLatticeProjector(
    const LatticeStructure& lattice_structure,
    const std::vector<int>& monotone_dimensions, const Dtype epsilon,
    const int64 max_iter)
    : epsilon_(epsilon),
      max_iter_(max_iter),
      param_size_(lattice_structure.NumVertices()) {
  for (const int dim : monotone_dimensions) {
    projectors_.push_back(PerDimensionProjector(lattice_structure, dim));
  }
}

// Apply ADMM projections.
template <typename Dtype>
Status MonotoneLatticeProjector<Dtype>::Project(
    const std::vector<Dtype>& lattice_param_vec,
    std::vector<Dtype>* projected_lattice_param_vec) const {
  if (lattice_param_vec.size() != param_size_) {
    return errors::InvalidArgument("lattice_param_vec's size (",
                                   lattice_param_vec.size(),
                                   ") != param_size (", param_size_, ")");
  }

  if (!projected_lattice_param_vec) {
    return errors::InvalidArgument("projected_lattice_param_vec is nullptr");
  }
  if (projected_lattice_param_vec->size() != param_size_) {
    return errors::InvalidArgument("projected_lattice_param_vec's size (",
                                   projected_lattice_param_vec->size(),
                                   ") != param_size (", param_size_, ")");
  }

  // No projection at all. Make a deep copy, then return.
  if (projectors_.empty()) {
    *projected_lattice_param_vec = lattice_param_vec;
    return Status::OK();
  }

  // Only one projection. No need for running a complicated projection.
  if (projectors_.size() == 1) {
    // Make a deep copy, then project.
    *projected_lattice_param_vec = lattice_param_vec;
    projectors_[0].Project(projected_lattice_param_vec);
    return Status::OK();
  }

  // Initialize all variables.
  // 1. Center: This contains a reference to the projected lattice parameter
  // vector.
  // 2. Param_per_cluster.
  // 3. Deviation_per_cluster.
  std::vector<Dtype>& center = *projected_lattice_param_vec;
  const int param_size = lattice_param_vec.size();
  const int num_clusters = projectors_.size();

  // Initial point is a deep copy of lattice_param_vec.
  center = lattice_param_vec;
  std::vector<std::vector<Dtype>> param_per_cluster(
      num_clusters, std::vector<Dtype>(param_size, 0.0));
  std::vector<std::vector<Dtype>> duals(num_clusters,
                                        std::vector<Dtype>(param_size, 0.0));

  Dtype residual = std::numeric_limits<Dtype>::max();
  int64 iter = 0;
  const Dtype average_scale = 0.5 / static_cast<Dtype>(num_clusters);


  while (residual > epsilon_) {
    // Step 1. Update parameter in each cluster by applying projections.
    for (int ii = 0; ii < num_clusters; ++ii) {
      // Step 1-1. Update param_per_cluster[ii] == center + duals[ii].
      const std::vector<Dtype>& duals_ii = duals[ii];
      std::vector<Dtype>& param_ii = param_per_cluster[ii];
      for (int jj = 0; jj < param_size; ++jj) {
        param_ii[jj] = duals_ii[jj] + center[jj];
      }
      // Step 1-2. Project onto the monotonicity constraint.
      projectors_[ii].Project(&param_ii);
    }

    // Step 2. Update the center.
    // center = 1/2 * lattice_param_vec + 1/2 * (Average(param_per_cluster) -
    // Average(dual))
    center.assign(param_size, 0);
    for (int ii = 0; ii < num_clusters; ++ii) {
      const std::vector<Dtype>& dual = duals[ii];
      const std::vector<Dtype>& param = param_per_cluster[ii];
      for (int jj = 0; jj < param_size; ++jj) {
        center[jj] += (param[jj] - dual[jj]);
      }
    }
    for (int ii = 0; ii < param_size; ++ii) {
      center[ii] *= average_scale;
      center[ii] += 0.5 * lattice_param_vec[ii];
    }

    // Step 3. Update the dual and residual
    residual = 0;
    for (int ii = 0; ii < num_clusters; ++ii) {
      std::vector<Dtype>& dual = duals[ii];
      const std::vector<Dtype>& param = param_per_cluster[ii];
      for (int jj = 0; jj < param_size; ++jj) {
        const Dtype diff = center[jj] - param[jj];
        dual[jj] += diff;
        residual += std::abs(diff);
      }
    }

    ++iter;
    if (iter > max_iter_) {
      break;
    }
  }
  return Status::OK();
}

}  // namespace lattice
}  // namespace tensorflow

#endif  // TENSORFLOW_LATTICE_CC_KERNELS_MONOTONIC_LATTICE_PROJECTIONS_H_
