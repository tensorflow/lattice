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
#include <numeric>
#include <vector>

#include "tensorflow_lattice/cc/kernels/lattice_interpolation_base.h"
#include "tensorflow_lattice/cc/lib/lattice_structure.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace lattice {

namespace {

// Returns the permutation such that
// values[permutation[0]] >= ... >= values[permutation[d - 1]] where
// d == values.size().
template <typename Dtype>
std::vector<int64> DescendingPermutation(const std::vector<Dtype>& values) {
  std::vector<int64> permutation(values.size());
  std::iota(permutation.begin(), permutation.end(), 0);

  auto cmp = [&values](const int64 left, const int64 right) -> bool {
    return values[left] > values[right];
  };

  std::sort(permutation.begin(), permutation.end(), cmp);

  return permutation;
}

}  // namespace

// SimplexInterpolationOpKernel returns interpolation weights.
template <typename Dtype>
class SimplexInterpolationOpKernel : public LatticeInterpolationOpBase<Dtype> {
 public:
  explicit SimplexInterpolationOpKernel(OpKernelConstruction* context)
      : LatticeInterpolationOpBase<Dtype>(context) {

    constexpr int64 kBaseCost = 20;
    constexpr int64 kCostPerCellVertex = 20;
    constexpr int64 kWeightInitializationCost = 1;
    const int64 input_dim = this->GetLatticeStructure().Dimension();
    this->SetCostPerExample(kCostPerCellVertex * input_dim * log(input_dim) +
                            kWeightInitializationCost *
                                this->GetLatticeStructure().NumVertices() +
                            kBaseCost);
  }

 private:
  InterpolationWeights<Dtype> ComputeInterpolationWeights(
      const LatticeStructure& lattice_structure,
      typename TTypes<Dtype>::UnalignedConstFlat input_vector) const final;

  TF_DISALLOW_COPY_AND_ASSIGN(SimplexInterpolationOpKernel);
};

// SimplexGradientOpKernel returns gradient with respect to the
// input. See details in CalculateGradientWrtInput above.
template <typename Dtype>
class SimplexGradientOpKernel : public LatticeGradientOpBase<Dtype> {
 public:
  explicit SimplexGradientOpKernel(OpKernelConstruction* context)
      : LatticeGradientOpBase<Dtype>(context) {

    constexpr int64 kBaseCost = 20;
    constexpr int64 kCostPerCellVertex = 20;
    const int64 input_dim = this->GetLatticeStructure().Dimension();
    this->SetCostPerExample(kCostPerCellVertex * input_dim * log(input_dim) +
                            kBaseCost);
  }

 private:
  std::vector<Dtype> ComputeGradWrtInput(
      const LatticeStructure& lattice_structure,
      typename TTypes<Dtype>::UnalignedConstFlat input_vector,
      typename TTypes<Dtype>::UnalignedConstFlat weight_vector,
      typename TTypes<Dtype>::UnalignedConstFlat grad_wrt_weight_vector)
      const final;

  TF_DISALLOW_COPY_AND_ASSIGN(SimplexGradientOpKernel);
};

// Produces simplex interpolation weights for an input that is in the unit
// hypercube (the residual), as well as the corresponding indices in the lattice
// (based on the bottom_corner). See http://jmlr.org/papers/v17/15-243.html for
// more details.
//
// Calculating the linear interpolation weights
// --------------------------------------------
// We compute the linear interpolation weights using Lovasz's extension.
// The formula for Lovasz's extension:
// 1. Find the permuation such that
//
//   input[permutation[0]] >= ... >= input[permutation[d-1]]
//
// 2. Assign the weight such that
//
//   weight on e0 = 1 - input[permutation[0]]
//   weight on e0 + e[permutation[0]] = input[permutation[0]] -
//   input[permutation[1]]
//   ...
//   weight on e0 + \sum_{i=0}^k e[permutation[i]] = input[permutation[k]] -
//   input[permutation[k + 1]]
//   ....
//   weight on e0 + \sum_{i=0}^{d - 1} e[permutation[i]] =
//   input[permutation[d - 1]]
//
// where e0 = [0,...0], e[i] = [0,...,1,...,0] whose ith component is 1.
// (Note that the weight is in the 2 ** D dimensional probability simplex, hence
// the valid interpolation weight.)
//
// This is equivalent to partition the hypercube into d! simplices, where each
// simplex has d+1 vertices, and each simplex's vertices includes the all-zeros
// vertex, one vertex with one ones, one vertex with two ones, ... and the
// all-ones vertex.
//
// For example, for a three-dimensional unit hypercube the 3! = 6 simplices
// are:
// 1: [0,0,0], [0,0,1], [0,1,1], [1,1,1]
// 2: [0,0,0], [0,0,1], [1,0,1], [1,1,1]
// 3: [0,0,0], [0,1,0], [0,1,1], [1,1,1]
// 4: [0,0,0], [0,1,0], [1,1,0], [1,1,1]
// 5: [0,0,0], [1,0,0], [1,1,0], [1,1,1]
// 6: [0,0,0], [1,0,0], [1,0,1], [1,1,1]
//
// Thus we can specify one of the d! simplices by a d-dim vector stating the
// order in which the vertices add 1. In the example above, the first simplex
// can be specified as [2,1,0], and the second simplex as [2,0,1].
//
// For the first simplex, the weights are given by
//
//   weight on [0,0,0] = 1 - input[2]
//   weight on [0,0,1] = input[2] - input[1]
//   weight on [0,1,1] = input[1] - input[0]
//   weight on [1,1,1] = input[0]
//   weight on others  = 0.
//
// For the second simplex, the weights are given by
//   weight on [0,0,0] = 1 - input[2]
//   weight on [0,0,1] = input[2] - input[0]
//   weight on [1,0,1] = input[0] - input[1]
//   weight on [1,1,1] = input[0]
//   weight on others  = 0.
//
// An extension to the multi-cell case is done by
//  1. Finding the bottom corner index and the residual vector.
//  2. Compute the interpolation weight using the residual vector.
//  3. Modify e[i] = strides[i] + bottom_corner_index.
//

template <typename Dtype>
InterpolationWeights<Dtype>
SimplexInterpolationOpKernel<Dtype>::ComputeInterpolationWeights(
    const LatticeStructure& lattice_structure,
    typename TTypes<Dtype>::UnalignedConstFlat input) const {
  const BottomCornerIndexAndResidual<Dtype> bottom_corner_index_and_residual =
      lattice_structure.GetBottomCornerIndexAndResidual<Dtype>(input);
  const std::vector<Dtype>& residual =
      bottom_corner_index_and_residual.residual;
  const std::vector<int64> descending_permutation =
      DescendingPermutation<Dtype>(residual);

  const int64 input_dim = lattice_structure.Dimension();
  // interpolation weight contains upto d + 1 non-zero elements.
  // Number of non-zero weights.
  const int64 nnz_weight = input_dim + 1;
  InterpolationWeights<Dtype> interpolation_weights;
  std::vector<int64>& index = interpolation_weights.indices;
  std::vector<Dtype>& weight = interpolation_weights.weights;
  index.resize(nnz_weight);
  weight.resize(nnz_weight);

  Dtype current_residual = 1.0;
  int64 current_index = bottom_corner_index_and_residual.bottom_corner_index;
  const std::vector<int64>& strides = lattice_structure.Strides();
  for (int ii = 0; ii < input_dim; ++ii) {
    const int64 current_dim = descending_permutation[ii];
    const Dtype next_residual = residual[current_dim];
    // Assigning index and weight.
    index[ii] = current_index;
    weight[ii] = current_residual - next_residual;
    // Proceed to the next item.
    current_index += strides[current_dim];
    current_residual = next_residual;
  }
  // The boundary case.
  index[input_dim] = current_index;
  weight[input_dim] = current_residual;

  return interpolation_weights;
}

// The goal of the gradient op is, given grad_wrt_weight:
//   (dy / dweight[0], dy / dweight[1], dy / dweight[2], dy / dweight[3]),
// to compute the grad_wrt_input:
//   (dy / dx[0], ..., dy / dx[D-1]).
//
// We know that:
//   dy/dx[jj] = sum_{ii \in weights} dy/dweight[ii] * dweight[ii]/dx[jj]
//
// So we need to calculate dweight[ii]/dx[jj]. Let us consider 2 x 2 lattice
// case first. Recall that
//
//   weight = \sum_k input[k] * (e[permutation[k + 1]] - e[permutation[k]])
//
// which is a linear function in input[k]. Therefore the gradient can be
// computed easily once we have the permutation.
//
// The boudnary case (out_of_bound):
//  When input[k] > 1 or input[k] < 0, we assign the zero gradient.
//

template <typename Dtype>
std::vector<Dtype> SimplexGradientOpKernel<Dtype>::ComputeGradWrtInput(
    const LatticeStructure& lattice_structure,
    typename TTypes<Dtype>::UnalignedConstFlat input,
    typename TTypes<Dtype>::UnalignedConstFlat unused_weight,
    typename TTypes<Dtype>::UnalignedConstFlat grad_wrt_weight) const {
  const BottomCornerIndexAndResidual<Dtype> bottom_corner_index_and_residual =
      lattice_structure.GetBottomCornerIndexAndResidual<Dtype>(input);
  const std::vector<Dtype>& residual =
      bottom_corner_index_and_residual.residual;
  const std::vector<int64> descending_permutation =
      DescendingPermutation<Dtype>(residual);

  const int64 input_dim = lattice_structure.Dimension();
  int64 current_index = bottom_corner_index_and_residual.bottom_corner_index;
  int64 current_coefficient = grad_wrt_weight(current_index);
  const std::vector<int64>& strides = lattice_structure.Strides();
  const std::vector<bool>& out_of_bound =
      bottom_corner_index_and_residual.out_of_bound;

  // Initialization.
  std::vector<Dtype> grad_wrt_input(input_dim, 0.0);
  for (const int64 current_dim : descending_permutation) {
    current_index += strides[current_dim];
    const Dtype next_coefficient = grad_wrt_weight(current_index);
    // Only update the gradient if it is not out of bound.
    if (!out_of_bound[current_dim]) {
      grad_wrt_input[current_dim] = (next_coefficient - current_coefficient);
    }
    current_coefficient = next_coefficient;
  }
  return grad_wrt_input;
}

// Register kernels for float and double.
REGISTER_KERNEL_BUILDER(Name("SimplexInterpolation")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("Dtype"),
                        SimplexInterpolationOpKernel<float>);

REGISTER_KERNEL_BUILDER(Name("SimplexInterpolation")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("Dtype"),
                        SimplexInterpolationOpKernel<double>);

REGISTER_KERNEL_BUILDER(
    Name("SimplexGradient").Device(DEVICE_CPU).TypeConstraint<float>("Dtype"),
    SimplexGradientOpKernel<float>);

REGISTER_KERNEL_BUILDER(
    Name("SimplexGradient").Device(DEVICE_CPU).TypeConstraint<double>("Dtype"),
    SimplexGradientOpKernel<double>);

}  // namespace lattice
}  // namespace tensorflow
