/* Copyright 2018 The TensorFlow Lattice Authors.

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
#ifndef TENSORFLOW_LATTICE_CC_TFLITE_OPS_HELPERS_H_
#define TENSORFLOW_LATTICE_CC_TFLITE_OPS_HELPERS_H_

#include <math.h>
#include <vector>

namespace tflite_lattice {

template <typename T>
T ClipToBounds(const T value, const T lower_bound, const T upper_bound) {
  return value > upper_bound ? upper_bound
                             : (value < lower_bound ? lower_bound : value);
}

// BottomCornerIndexAndResidual contains a bottom corner index in the multi-cell
// lattice and residual vector for a given input. If out_of_bound[k] is true,
// then kth input is outside of multi-cell lattice's boundary.
template <typename Dtype>
struct BottomCornerIndexAndResidual {
  int bottom_corner_index;
  std::vector<Dtype> residual;
  std::vector<bool> out_of_bound;
};

template <typename Dtype>
BottomCornerIndexAndResidual<Dtype> GetBottomCornerIndexAndResidual(
    std::vector<int> lattice_sizes, const float* input_row,
    std::vector<int> strides) {
  int dimension = lattice_sizes.size();
  BottomCornerIndexAndResidual<Dtype> bottom_corner_index_and_residual;
  int& bottom_corner_index =
      bottom_corner_index_and_residual.bottom_corner_index;
  std::vector<Dtype>& residual = bottom_corner_index_and_residual.residual;
  std::vector<bool>& out_of_bound =
      bottom_corner_index_and_residual.out_of_bound;

  residual.resize(dimension);
  out_of_bound.resize(dimension);

  bottom_corner_index = 0;
  for (int i = 0; i < dimension; ++i) {
    const int max_vertex_in_i = lattice_sizes[i] - 1;
    const float input_i = input_row[i];
    // Find the bottom corner lattice coordinates for the "i"th feature of
    // this point.
    // We clip to the bounds of the lattice, [0, max_vertex_in_i].

    const int bottom_corner_i = ClipToBounds<int>(
        static_cast<int>(floor(input_i)), 0, max_vertex_in_i - 1);
    const Dtype residual_i =
        ClipToBounds<Dtype>(input_i - bottom_corner_i, 0.0, 1.0);

    bottom_corner_index += strides[i] * bottom_corner_i;
    residual[i] = residual_i;
    out_of_bound[i] = (input_i < 0.0 || input_i > max_vertex_in_i);
  }
  return bottom_corner_index_and_residual;
}

typedef struct {
  // lattice_sizes is provided by user and records the number of nodes in each
  // lattice dimension.  All other members derived from this one.
  // Naming matches tensoflow op.
  std::vector<int> lattice_sizes;
  // Number of dimensions present.  Total nodes = k ^ dimension if all
  // lattice_sizes are k.
  int dimension;
  // One node index in dimension k equals strides[k] array indices
  std::vector<int> strides;
  // Total number of nodes, is the product of numbers in lattice_sizes
  int num_vertices;
  // Nodes in a cell.  Note that this is the number of non-zero weights that
  // will result from interpolating one point.  Also note that each vertex can
  // belong to many cells.
  int num_vertices_per_cell;
} InterpolationParams;

}  // namespace tflite_lattice

#endif  // TENSORFLOW_LATTICE_CC_TFLITE_OPS_HELPERS_H_
