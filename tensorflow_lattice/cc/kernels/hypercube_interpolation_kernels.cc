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
#include <memory>
#include <vector>

#include "tensorflow_lattice/cc/kernels/lattice_interpolation_base.h"
#include "tensorflow_lattice/cc/lib/lattice_structure.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace lattice {

// HypercubeInterpolationOpKernel returns interpolation weights.
template <typename Dtype>
class HypercubeInterpolationOpKernel
    : public LatticeInterpolationOpBase<Dtype> {
 public:
  explicit HypercubeInterpolationOpKernel(OpKernelConstruction* context)
      : LatticeInterpolationOpBase<Dtype>(context) {

    constexpr int64 kBaseCost = 20;
    constexpr int64 kCostPerCellVertex = 20;
    constexpr int64 kWeightInitializationCost = 1;
    this->SetCostPerExample(
        kCostPerCellVertex * this->GetLatticeStructure().NumVerticesPerCell() +
        kWeightInitializationCost * this->GetLatticeStructure().NumVertices() +
        kBaseCost);
  }

 private:
  InterpolationWeights<Dtype> ComputeInterpolationWeights(
      const LatticeStructure& lattice_structure,
      typename TTypes<Dtype>::UnalignedConstFlat input_vector) const final;

  TF_DISALLOW_COPY_AND_ASSIGN(HypercubeInterpolationOpKernel);
};

// HypercubeGradientOpKernel returns gradient with respect to the
// input.
template <typename Dtype>
class HypercubeGradientOpKernel : public LatticeGradientOpBase<Dtype> {
 public:
  explicit HypercubeGradientOpKernel(OpKernelConstruction* context)
      : LatticeGradientOpBase<Dtype>(context) {

    constexpr int64 kBaseCost = 20;
    constexpr int64 kCostPerCellVertex = 20;
    this->SetCostPerExample(
        kCostPerCellVertex * this->GetLatticeStructure().Dimension() *
            this->GetLatticeStructure().NumVerticesPerCell() +
        kBaseCost);
  }

 private:
  std::vector<Dtype> ComputeGradWrtInput(
      const LatticeStructure& lattice_structure,
      typename TTypes<Dtype>::UnalignedConstFlat input_vector,
      typename TTypes<Dtype>::UnalignedConstFlat weight_vector,
      typename TTypes<Dtype>::UnalignedConstFlat grad_wrt_weight_vector)
      const final;

  TF_DISALLOW_COPY_AND_ASSIGN(HypercubeGradientOpKernel);
};

// Produces linear interpolation weights for an input that is in the unit
// hypercube (the residual), as well as the corresponding indices in the lattice
// (based on the bottom_corner). Both the weights and the indices are computed
// during the same loop for efficiency, but we'll explain their computations
// separately. Also returns the residual vector from the bottom corner of the
// hypercube cell. See http://jmlr.org/papers/v17/15-243.html for more details.
//
// Calculating the linear interpolation weights
// --------------------------------------------
// The linear interpolation weights on each vertex are the volumes of the
// hyperrectangles formed by partitioning the unit hypercube at the input.
// Example: 2D case  - Draw a unit square. Draw an input x in the square.
// Draw horizontal and vertical lines through x.  That forms 4 boxes - the
// volume of these boxes are the weights. Note the boxes are a partition of
// the unit square, so the sum of the areas (volumes) of the boxes sums to 1.
// The linear interpolation weight on each vertex is the volume of the box in
// the opposite corner (so that if you move x close to one corner, the weight
// on that corner grows). Mathematically in the 2D case (and generalizes
// directly to higher D) the weights are:
// weight([0, 0]) = (1 - input[0]) * (1 - input[1])
// weight([0, 1]) = (1 - input[0]) * input[1]
// weight([1, 0]) = input[0] * (1 - input[1])
// weight([1, 1]) =  input[0] * input[1]
//
// Computing each of the 2^D weights directly using above formula would take
// O(2^D * D) operations. Instead we take advantage of the many repeated
// calculations to reduce this to a O(2^D) computation as follows:
// Let's start by initializing weight to 1 for every vertex. Lets consider
// current vertex. Suppose its bit representation is "0110". For every "0" we
// should multiply its weight on (1 - input[i]), where i is a sequence number
// of correspondent bit. And for each "1" we should multiply its weight on
// input[i].
// Let us iterate through all vertices in dfs (lexicographical) order. Let
// current_highest_dimension be a sequence number of highest bit in binary
// representation of current vertex. At this moment we multiplied
// correspondent weights for all dimensions below current_highest_dimension.
// Now, let us update current_highest_dimension.
// Example:
// If "ii" is iterating on "??x010" (the location of memory where finally the
// weight for "00_1_010" will be stored), then we set the value for
//
//   // Resetting bit x of ??x010.
//   earlier_ii = ii ^ (1 << current_highest_dimension)
//   // Now ii represents ?x1010.
//   weight[ii] = weight[earlier_ii] * input[current_highest_dimension]
//   // earlier_ii represents ?x0010.
//   weight[earlier_ii] *= (1 - input[current_highest_dimension])
//
// Example for 2x2 case:
//   weight[0] is weight on [0,0]
//   weight[1] is weight on [1,0]
//   weight[2] is weight on [0,1]
//   weight[3] is weight on [1,1]
// Initialization:  weight[0] = 1, no other weight set.
// Loop: ii = 1. current_highest_dimension = 0
//   weight[1] = weight[0] * input[0];
//   weight[0] = weight[0] * (1 - input[0])
// ii = 2. current_highest_dimension = 1. (highest bit of ii got index 1 at
// this step, so update current_highest_dimension to reflect this)
//   weight[2] = weight[0] * input[1];
//   weight[0] = weight[0] * (1 - input[1])
// ii = 3. current_highest_dimension = 1.
//   weight[3] = weight[1] * input[1];
//   weight[1] = weight[1] * (1 - input[1])
//
// Calculating the corresponding indices. Notice if the lattice sizes are larger
// than 2, the indices of the wieghts will be adjusted according to the
// LatticeStructure.strides.
// -------------------------------------
// The lattice index for the iith vertex in the cell is the same as the index
// we computed for an earlier neighbor vertex, but offset by
// lattice_strides[(dimensions - 1) - current_highest_dimension].
// Example:
// Suppose we have a 2x2 lattice. We should output vertices in the order:
//   [0,0], [1,0], [0,1], [1,1].
// Bottom corner is [0,0], so vertices[0] = 0 already set.
// let ii = 1. It corresponds to vertex [0,1]. current_highest_dimension = 0.
//   lattice index of vertices[1] is different from lattice index of
//   vertices[0] in dimension current_highest_dimension = 0 (counting from the
// end).
//   vertices[1] = vertices[0] + lattice_strides[0] = [1, 0];
// let ii = 2, it corresponds to [1,0]. current_highest_dimension becomes 1.
//   vertices[2] = vertices[0] + lattice_strides[1] = [0, 1];
//   vertices[3] = vertices[1] + lattice_strides[1] = [1, 1];
//

template <typename Dtype>
InterpolationWeights<Dtype>
HypercubeInterpolationOpKernel<Dtype>::ComputeInterpolationWeights(
    const LatticeStructure& lattice_structure,
    typename TTypes<Dtype>::UnalignedConstFlat input) const {
  const BottomCornerIndexAndResidual<Dtype> bottom_corner_index_and_residual =
      lattice_structure.GetBottomCornerIndexAndResidual<Dtype>(input);
  const std::vector<Dtype>& residual =
      bottom_corner_index_and_residual.residual;
  const int64 num_vertices_per_cell = lattice_structure.NumVerticesPerCell();
  // interpolation weight contains upto num_vertices_per_cell non-zero elements.
  InterpolationWeights<Dtype> interpolation_weights;
  std::vector<int64>& index = interpolation_weights.indices;
  std::vector<Dtype>& weight = interpolation_weights.weights;

  index.resize(num_vertices_per_cell);
  weight.resize(num_vertices_per_cell);
  index[0] = bottom_corner_index_and_residual.bottom_corner_index;
  weight[0] = 1.0;

  const int64 input_dim = lattice_structure.Dimension();
  const std::vector<int64>& strides = lattice_structure.Strides();

  int64 current_highest_dimension = 0;
  Dtype current_residual_value = residual[current_highest_dimension];
  for (int64 ii = 1; ii < num_vertices_per_cell; ++ii) {
    // Make sure that we're within the bounds of the unit hypercube.
    DCHECK_GE(current_residual_value, 0);
    DCHECK_LE(current_residual_value, 1);
    // Sanity check: current_highest_dimension has better respect the bounds.
    DCHECK_GE(current_highest_dimension, 0);
    DCHECK_LT(current_highest_dimension, input_dim);

    const int64 earlier_ii = ii ^ (1 << current_highest_dimension);
    index[ii] = index[earlier_ii] + strides[current_highest_dimension];
    weight[ii] = weight[earlier_ii] * current_residual_value;
    weight[earlier_ii] *= (1.0 - current_residual_value);

    if ((ii & (ii + 1)) == 0) {
      // If ii + 1 is power of 2, then current_highest_dimension has changed,
      // that means, that we are processing next dimension.
      ++current_highest_dimension;
      if (input_dim >= current_highest_dimension + 1) {
        current_residual_value = residual[current_highest_dimension];
      }
    }
  }
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
// For dweight[ii]/dx[jj], we use the following observation.
// For any 2 x ... x 2 lattices:
//  weight[ii] + weight[jj] == constant.
// for all (ii, jj) pair such that ii ^ jj == 2 ** k and ii < jj. (This means
// ii's kth vertex is 0, and jj's kth vertex is 1, and other vertices are same.)
// Moreover, for such (ii, jj) pair, we have
//   dweight[ii] / dx[k] == -(weight[ii] + weight[jj])
//   dweight[jj] / dx[k] == (weight[ii] + weight[jj])
//
// To see this, let us consider 2 x 2 lattice case.
//
// Recall that
//   weight[0] = (1 - x[0]) * (1 - x[1])
//   weight[1] = x[0] * (1 - x[1])
//   weight[2] = (1 - x[0]) * x[1]
//   weight[3] = x[0] * x[1]
//
// Therefore,
//   dweight[0] / dx[0] = -(1 - x[1]) == -(weight[0] + weight[1])
//   dweight[1] / dx[0] = (1 - x[1]) == (weight[0] + weight[1])
//   dweight[2] / dx[0] = -x[1] == -(weight[2] + weight[3])
//   dweight[3] / dx[0] = x[1] == (weight[2] + weight[3]),
// and
//   dweight[0] / dx[1] = -(1 - x[0]) == -(weight[0] + weight[2])
//   dweight[1] / dx[1] = -x[0] == -(weight[1] + weight[3])
//   dweight[2] / dx[1] = (1 - x[0]) == (weight[0] + weight[2])
//   dweight[3] / dx[1] = x[0] == (weight[1] + weight[3]).
//
// So the summation part marginalize the dependency of x[k], and the sign is
// minus if the kth vertex is 0, and plus if the kth vertex is 1.
// The following code computes the gradient using the (ii, jj) pair by
// enumerating all indices whose kth vertex is 0.
// In order to support the multi-cell lattice, the code constructs a list
// (nnz_weight below) that maps the indices in the 2 x .... x 2 cell holding x
// into indices in the multi-cell.
//
// Including this construction, the overall complexity is
//  O((input_dim + 2) * 2 ** (input_dim - 1)).
//
// Also when x[jj] < 0 or x[jj] > lattice_size[jj], the input is out of bound.
// So the change in the input should not change the output, therefore the
// gradient should be zero.
//

template <typename Dtype>
std::vector<Dtype> HypercubeGradientOpKernel<Dtype>::ComputeGradWrtInput(
    const LatticeStructure& lattice_structure,
    typename TTypes<Dtype>::UnalignedConstFlat input,
    typename TTypes<Dtype>::UnalignedConstFlat weight,
    typename TTypes<Dtype>::UnalignedConstFlat grad_wrt_weight) const {
  const BottomCornerIndexAndResidual<Dtype> bottom_corner_index_and_residual =
      lattice_structure.GetBottomCornerIndexAndResidual<Dtype>(input);
  const int64 input_dim = lattice_structure.Dimension();
  std::vector<Dtype> grad_wrt_input(input_dim, 0.0);

  // There are at most 2 ** n number of non-zero elements in weight.
  // nnz_weight_index keeps the index of non-zero element in the weight.
  // The following loop enumerats all vertices in cell in the following order.
  // [0, 0, ..., 0], [1, 0, ...,0], [0, 1, ..., 0], ..., [1, 1, ..., 1].
  std::vector<int64> nnz_weight_index(lattice_structure.NumVerticesPerCell());

  int64 current_dim = 0;
  int64 current_bit = 1;  // Always 1 << current_dim;
  nnz_weight_index[0] = bottom_corner_index_and_residual.bottom_corner_index;
  const std::vector<int64>& strides = lattice_structure.Strides();
  for (int64 ii = 1; ii < nnz_weight_index.size(); ++ii) {
    if ((ii & current_bit) == 0) {
      ++current_dim;
      current_bit <<= 1;
    }
    // ii - current_bit is the base.
    // ii is the current one, which is always an upper layer in the current
    // dimension.
    nnz_weight_index[ii] =
        nnz_weight_index[ii - current_bit] + strides[current_dim];
  }

  // Compute the gradient for each input.
  for (int64 ii = 0; ii < input_dim; ++ii) {
    // If out_of_bound, gradient is 0.
    if (bottom_corner_index_and_residual.out_of_bound[ii]) {
      continue;
    }
    // Only process the bottom faces.
    int64 bit = 1 << ii;
    int64 stride = strides[ii];
    Dtype grad_ii = 0.0;
    for (int64 index = 0; index < lattice_structure.NumVerticesPerCell();
         ++index) {
      // Upper face. Skip this index.
      if (index & bit) {
        continue;
      }
      // Bottom face.
      int64 lower_index = nnz_weight_index[index];
      int64 upper_index = lower_index + stride;
      grad_ii += (weight(lower_index) + weight(upper_index)) *
                 (grad_wrt_weight(upper_index) - grad_wrt_weight(lower_index));
    }
    grad_wrt_input[ii] = grad_ii;
  }

  return grad_wrt_input;
}
// Register kernels for float and double.
REGISTER_KERNEL_BUILDER(Name("HypercubeInterpolation")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("Dtype"),
                        HypercubeInterpolationOpKernel<float>);

REGISTER_KERNEL_BUILDER(Name("HypercubeInterpolation")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("Dtype"),
                        HypercubeInterpolationOpKernel<double>);

REGISTER_KERNEL_BUILDER(
    Name("HypercubeGradient").Device(DEVICE_CPU).TypeConstraint<float>("Dtype"),
    HypercubeGradientOpKernel<float>);

REGISTER_KERNEL_BUILDER(Name("HypercubeGradient")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("Dtype"),
                        HypercubeGradientOpKernel<double>);

}  // namespace lattice
}  // namespace tensorflow
