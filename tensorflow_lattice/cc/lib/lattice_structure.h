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
// Lattice structure class that represents a lattice with column-major indexing.
#ifndef TENSORFLOW_LATTICE_CC_LIB_LATTICE_STRUCTURE_H_
#define TENSORFLOW_LATTICE_CC_LIB_LATTICE_STRUCTURE_H_

#include <math.h>
#include <vector>

#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace lattice {

template <typename Dtype>
Dtype ClipToBounds(const Dtype value, const Dtype lower_bound,
                   const Dtype upper_bound) {
  return value > upper_bound ? upper_bound
                             : (value < lower_bound ? lower_bound : value);
}

// BottomCornerIndexAndResidual contains a bottom corner index in the multi-cell
// lattice and residual vector for a given input. If out_of_bound[k] is true,
// then kth input is outside of multi-cell lattice's boundary.
template <typename Dtype>
struct BottomCornerIndexAndResidual {
  int64 bottom_corner_index;
  std::vector<Dtype> residual;
  std::vector<bool> out_of_bound;
};

// This class represents a structure of a multi-cell lattice including the
// dimension of a lattice, number of vertices, number of vertices in each cell,
// and strides for a global index.
// For example, in 2d case, a multi-cell lattice is a grid. The following
// example shows a 3 x 2 multi-cell lattice. Each cell has four vertices, and
// in total, this multi-cell lattice contains 12 vertices.
//
//   --------------------------
//   |       |        |       |
//   |       |        |       |
//   --------------------------
//   |       |        |       |
//   |       |        |       |
//   --------------------------
//
// With the column-major indexing, the lattice with lattice_sizes
// [m_0, m_1, ..., m_{n - 1}] will have:
//   dimension: n
//   number of vertices: m_0 * ... * m_{n-1}
//   number of vertices in each cell: 2 ** (n-1)
//   stride[0] = 1
//   stride[1] = 1 * m_{0}
//        ...
//   stride[n-1] = 1 * m_{n - 2} ... * m_0
//
// Moreover, BottomCornerIndexAndResidual method returns the bottom corner index
// and residual vector of a input vector in a multi-cell lattice.
class LatticeStructure {
 public:
  // lattice_sizes[ii] is expected to contain a lattice size of the iith
  // coordinate.
  explicit LatticeStructure(const std::vector<int>& lattice_sizes);

  // Returns true if all elements >= 2.
  static bool IsValidLatticeSizes(const std::vector<int>& lattice_sizes);

  const int64 Dimension() const { return dimension_; }
  const int64 NumVertices() const { return num_vertices_; }
  const int64 NumVerticesPerCell() const { return num_vertices_per_cell_; }

  int64 Stride(int64 dimension) const { return strides_[dimension]; }
  const std::vector<int64>& Strides() const { return strides_; }
  int LatticeSize(int64 dimension) const { return lattice_sizes_[dimension]; }
  const std::vector<int>& LatticeSizes() const { return lattice_sizes_; }

  // Returns the bottom corner index of a cell that the input_vec belongs to and
  // the residual of vector, which is input_vec minus the vector corresponding
  // to the bottom corner index.
  // For example, consider the following 5 x 3 lattice in 2d plane.
  //
  //   x2
  //   |
  //   |
  //   10 ---- 11 ---- 12 ---- 13 --- 14
  //   |       |       |   x   |       |
  //   |       |       |       |       |
  //   5 ----- 6 ----- 7 ----- 8 ---- -9
  //   |       |       |       |       |
  //   |       |       |       |       |
  //   0 ----- 1 ----- 2 ----- 3 ----- 4----x1
  //
  // where the number at each vertex is the global index of each vertex. Each
  // cell is a square with the width 1. So the coordinate representation of
  // 0-indexed vertex is (0, 0), 1-indexed vertex is (1, 0), and 14-indexed
  // vertex is (4, 2).
  // Let x be the input vector, located at (2.5, 1.8). In this case, the
  // cell's bottom corner index is 7, and the residual is (0.5, 0.8).
  template <typename Dtype>
  BottomCornerIndexAndResidual<Dtype> GetBottomCornerIndexAndResidual(
      typename TTypes<Dtype>::UnalignedConstFlat input_vec) const;

 private:
  int64 dimension_;
  int64 num_vertices_;
  int64 num_vertices_per_cell_;
  std::vector<int> lattice_sizes_;
  std::vector<int64> strides_;
};

template <typename Dtype>
BottomCornerIndexAndResidual<Dtype>
LatticeStructure::GetBottomCornerIndexAndResidual(
    typename TTypes<Dtype>::UnalignedConstFlat vec) const {
  BottomCornerIndexAndResidual<Dtype> bottom_corner_index_and_residual;
  int64& bottom_corner_index =
      bottom_corner_index_and_residual.bottom_corner_index;
  std::vector<Dtype>& residual = bottom_corner_index_and_residual.residual;
  std::vector<bool>& out_of_bound =
      bottom_corner_index_and_residual.out_of_bound;

  residual.resize(dimension_);
  out_of_bound.resize(dimension_);

  bottom_corner_index = 0;
  for (int64 ii = 0; ii < dimension_; ++ii) {
    const int64 max_vertex_in_ii = lattice_sizes_[ii] - 1;
    const Dtype input_ii = vec(ii);
    // Find the bottom corner lattice coordinates for the "ii"th feature of
    // this point. We clip to the bounds of the lattice, [0, max_vertex_in_ii].

    const int64 bottom_corner_ii = ClipToBounds<int64>(
        static_cast<int64>(std::floor(input_ii)), 0, max_vertex_in_ii - 1);
    const Dtype residual_ii =
        ClipToBounds<Dtype>(input_ii - bottom_corner_ii, 0.0, 1.0);

    bottom_corner_index += strides_[ii] * bottom_corner_ii;
    residual[ii] = residual_ii;
    out_of_bound[ii] = (input_ii < 0.0 || input_ii > max_vertex_in_ii);
  }
  return bottom_corner_index_and_residual;
}

}  // namespace lattice
}  // namespace tensorflow

#endif  // TENSORFLOW_LATTICE_CC_LIB_LATTICE_STRUCTURE_H_
