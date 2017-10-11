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
#include "tensorflow_lattice/cc/lib/lattice_structure.h"

#include <memory>
#include <vector>

#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace lattice {

LatticeStructure::LatticeStructure(const std::vector<int>& lattice_sizes)
    : lattice_sizes_(lattice_sizes) {
  dimension_ = lattice_sizes_.size();
  strides_.resize(dimension_);
  num_vertices_ = 1;
  for (int ii = 0; ii < dimension_; ++ii) {
    strides_[ii] = num_vertices_;
    num_vertices_ *= lattice_sizes_[ii];
  }
  num_vertices_per_cell_ = 1 << dimension_;
}

bool LatticeStructure::IsValidLatticeSizes(
    const std::vector<int>& lattice_sizes) {
  if (lattice_sizes.empty()) {
    return false;
  }
  for (int size : lattice_sizes) {
    if (size < 2) {
      return false;
    }
  }
  return true;
}

}  // namespace lattice
}  // namespace tensorflow
