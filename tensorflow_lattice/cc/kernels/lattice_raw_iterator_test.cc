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
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace lattice {

namespace {
struct IndexVertexPair {
  int64 index;
  std::vector<int64> vertex;
};
}  // namespace

// The fixture for testing LatticeRawIteration.
class LatticeRawIteratorTest : public ::testing::Test {
 protected:
  // Given the lattice sizes, iterate using RawIterator and check whether the
  // iterator visits all expected index_vertex_pairs.
  void CheckFullIteration(
      const std::vector<int>& lattice_sizes,
      const std::vector<IndexVertexPair>& expected_index_vertex_pairs) {
    LatticeStructure lattice_structure(lattice_sizes);

    // Iterate and collect indices and vertices.
    std::vector<IndexVertexPair> visited_index_vertex_pairs;
    for (LatticeRawIterator iter(lattice_structure); !iter.IsDone();
         iter.Next()) {
      visited_index_vertex_pairs.push_back(
          IndexVertexPair{iter.Index(), iter.Vertex()});
      LOG(INFO) << "visited_index : " << iter.Index() << " visited_vertex: ["
                << str_util::Join(iter.Vertex(), ",") << "]";
    }

    // Check the result with the expected results.
    CompareIndexVertexPairs(expected_index_vertex_pairs,
                            visited_index_vertex_pairs);
  }

 private:
  void CompareIndexVertexPairs(
      const std::vector<IndexVertexPair>& index_vertex_pairs1,
      const std::vector<IndexVertexPair>& index_vertex_pairs2) {
    ASSERT_EQ(index_vertex_pairs1.size(), index_vertex_pairs2.size());
    const int num_pairs = index_vertex_pairs1.size();
    std::vector<bool> visited(num_pairs, false);
    // n ** 2 comparsion.
    for (const auto& index_vertex_pair2 : index_vertex_pairs2) {
      for (int ii = 0; ii < num_pairs; ++ii) {
        if (index_vertex_pair2.index == index_vertex_pairs1[ii].index &&
            index_vertex_pair2.vertex == index_vertex_pairs1[ii].vertex) {
          visited[ii] = true;
          break;
        }
      }
    }
    // Now check that we visited all index_vertex_pair in index_vertex_pairs1.
    for (const bool is_visited : visited) {
      EXPECT_TRUE(is_visited);
    }
  }
};

TEST_F(LatticeRawIteratorTest, FullIterationWithTwoByThree) {
  CheckFullIteration(
      /*lattice_sizes=*/{2, 3}, /*expected_index_vertex_pairs=*/{{0, {0, 0}},
                                                                 {1, {1, 0}},
                                                                 {2, {0, 1}},
                                                                 {3, {1, 1}},
                                                                 {4, {0, 2}},
                                                                 {5, {1, 2}}});
}

TEST_F(LatticeRawIteratorTest, FullIterationWithThreeByTwoByTwo) {
  CheckFullIteration(
      /*lattice_sizes=*/{3, 2, 2},
      /*expected_index_vertex_pairs=*/{{0, {0, 0, 0}},
                                       {1, {1, 0, 0}},
                                       {2, {2, 0, 0}},
                                       {3, {0, 1, 0}},
                                       {4, {1, 1, 0}},
                                       {5, {2, 1, 0}},
                                       {6, {0, 0, 1}},
                                       {7, {1, 0, 1}},
                                       {8, {2, 0, 1}},
                                       {9, {0, 1, 1}},
                                       {10, {1, 1, 1}},
                                       {11, {2, 1, 1}}});
}

}  // namespace lattice
}  // namespace tensorflow
