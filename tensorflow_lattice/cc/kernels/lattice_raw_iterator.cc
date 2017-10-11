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
#include "tensorflow_lattice/cc/kernels/lattice_raw_iterator.h"

#include <vector>

#include "tensorflow_lattice/cc/lib/lattice_structure.h"

namespace tensorflow {
namespace lattice {

void LatticeRawIterator::Next() {
  ++index_;
  for (int64 dim = 0; dim < lattice_sizes_.size(); ++dim) {
    ++vertex_[dim];
    if (vertex_[dim] == lattice_sizes_[dim]) {
      vertex_[dim] = 0;
    } else {
      break;
    }
  }
}

}  // namespace lattice
}  // namespace tensorflow
