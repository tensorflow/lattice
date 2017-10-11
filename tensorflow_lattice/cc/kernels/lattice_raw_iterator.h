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
// LatticeRawIterator iterates all vertices in a multi-cell lattice in the
// column-major order. Note that this indexing (column-major order) should be
// consistent with LatticeStructure.
//
// Iteration example:
// for (LatticeRawIterator iter(lattice_structure) ; !iter.IsDone();
// iter.Next()) {
//   const int64 global_index = iter.Index();
//   const int64 vertex_first_dim = iter.VertexDim(0);
//   const int64 vertex_second_dim = iter.VertexDim(1);
// }

#ifndef TENSORFLOW_LATTICE_CC_KERNELS_LATTICE_RAW_ITERATOR_H_
#define TENSORFLOW_LATTICE_CC_KERNELS_LATTICE_RAW_ITERATOR_H_

#include <memory>
#include <vector>

#include "tensorflow_lattice/cc/lib/lattice_structure.h"

namespace tensorflow {
namespace lattice {

class LatticeRawIterator {
 public:
  explicit LatticeRawIterator(const LatticeStructure& lattice_structure)
      : lattice_sizes_(lattice_structure.LatticeSizes()),
        vertex_(lattice_structure.Dimension(), 0),
        index_(0),
        last_index_(lattice_structure.NumVertices()) {}

  // Forwards the iterator.
  void Next();

  bool IsDone() const { return index_ >= last_index_; }
  int64 Index() const { return index_; }
  const std::vector<int64>& Vertex() const { return vertex_; }
  int64 VertexDim(const int64 dim) const { return vertex_[dim]; }

 private:
  const std::vector<int> lattice_sizes_;
  std::vector<int64> vertex_;
  int64 index_;
  const int64 last_index_;
};

}  // namespace lattice
}  // namespace tensorflow
#endif  // TENSORFLOW_LATTICE_CC_KERNELS_LATTICE_RAW_ITERATOR_H_
