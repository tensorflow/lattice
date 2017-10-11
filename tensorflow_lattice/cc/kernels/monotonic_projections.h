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
// Functions that calculate monotonic projections.
#ifndef TENSORFLOW_LATTICE_CC_KERNELS_MONOTONIC_PROJECTIONS_H_
#define TENSORFLOW_LATTICE_CC_KERNELS_MONOTONIC_PROJECTIONS_H_

#include <vector>

#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
namespace lattice {

// Converts a vector to a non-strictly monotonic vector that minimizes squared
// distance to original vector values.
//
// monotonic_cmp is the comparison function that defines the direction of
//   the monotonicity. monotonic_cmp(a,b) should return true if a followed
//   by b is considered monotonic (equal values should always be considered
//   monotonic). monotonic_cmp should be transitive and
//   monotonic_cmp(a,b) && monotonic_cmp(b,a) iff a == b.
template <typename Dtype, typename CmpFn>
std::vector<Dtype> VectorMonotonicProjection(const std::vector<Dtype>& input,
                                             const CmpFn monotonic_cmp);

// Converts a Tensor vector to a non-strictly monotonic vector that minimizes
// squared distance to original vector values.
//
// monotonic_cmp is the comparison function that defines the direction of
//   the monotonicity. monotonic_cmp(a,b) should return true if a followed
//   by b is considered monotonic (equal values should always be considered
//   monotonic). monotonic_cmp should be transitive and
//   monotonic_cmp(a,b) && monotonic_cmp(b,a) iff a == b.
template <typename Dtype, typename CmpFn>
void TensorVectorMonotonicProjection(typename TTypes<Dtype>::Vec values,
                                     const CmpFn monotonic_cmp);

// Converts a vector to a non-strictly monotonic vector that minimizes squared
// distance to original vector values.
//
// Given a vector, input, it finds a non-strictly monotonic vector, output, such
// that:
//
//     1. cmp_fn(output[i], output[i + 1]) == true for all 0 <= i < n -1
//        (e.g., output[0] <= output[1] <= ... <= output[n -1])
//     2. minimizes || input - output ||_2
//
// This is a implementation special case of pool adjacent violators (PAV)
// algorithm.
//
// To use it one provides a comparison function (that defines the desired
// monotonicity direction) and Insert() one value at a time, in order.
//
// In the end one can project the monotonic vector into a std::vector or
// directly into a Tensor vector.
template <typename Dtype, typename CmpFn>
class MonotonicProjector {
 public:
  // size is the size of the vector to be projected to monotonicity.
  // monotonic_cmp is the comparison function that defines the direction of
  //   the monotonicity. monotonic_cmp(a,b) should return true if a followed
  //   by b is considered monotonic (equal values should always be considered
  //   monotonic). monotonic_cmp should be transitive and
  //   monotonic_cmp(a,b) && monotonic_cmp(b,a) iff a == b.
  explicit MonotonicProjector(const int size, const CmpFn monotonic_cmp)
      : size_(size), monotonic_cmp_(monotonic_cmp) {
    pool_list_.reserve(size);
  }

  // Insert value to end of pool list keeping list monotonic according to
  // monotonic_cmp_.
  void Insert(Dtype value) {
    Pool new_pool{1, value, value};
    // While new_pool wouldn't be properly monotonic, merge the pool with the
    // previous one.
    while (!pool_list_.empty() &&
           !monotonic_cmp_(pool_list_.back().mean, new_pool.mean)) {
      // If last pool would break monotonicity,
      new_pool.size += pool_list_.back().size;
      new_pool.sum += pool_list_.back().sum;
      new_pool.mean = new_pool.sum / new_pool.size;
      pool_list_.pop_back();
    }
    pool_list_.push_back(new_pool);
  }

  // Copies monotonic projection to Tensor vector.
  void ProjectToTensorVector(typename TTypes<Dtype>::Vec output) {
    int output_index = 0;
    for (const auto& pool : pool_list_) {
      for (const int limit = output_index + pool.size; output_index < limit;
           ++output_index) {
        output(output_index) = pool.mean;
      }
    }
  }

  // Returns monotonic projection as vector.
  std::vector<Dtype> ProjectToVector() {
    std::vector<Dtype> output(size_);
    int output_index = 0;
    for (const auto& pool : pool_list_) {
      for (const int limit = output_index + pool.size; output_index < limit;
           ++output_index) {
        output[output_index] = pool.mean;
      }
    }
    return output;
  }

 private:
  struct Pool {
    int size;         // Number of elements in pool.
    Dtype sum, mean;  // Sum and mean of all values in pool.
  };

  const int size_;
  std::vector<Pool> pool_list_;
  const CmpFn monotonic_cmp_;
};

// Implementation details

// START_SKIP_DOXYGEN
template <typename Dtype, typename CmpFn>
std::vector<Dtype> VectorMonotonicProjection(const std::vector<Dtype>& input,
                                             const CmpFn monotonic_cmp) {
  MonotonicProjector<Dtype, CmpFn> projector(input.size(), monotonic_cmp);
  for (const Dtype value : input) {
    projector.Insert(value);
  }
  return projector.ProjectToVector();
}

template <typename Dtype, typename CmpFn>
void TensorVectorMonotonicProjection(typename TTypes<Dtype>::Vec values,
                                     const CmpFn monotonic_cmp) {
  MonotonicProjector<Dtype, CmpFn> projector(values.size(), monotonic_cmp);
  for (int i = 0; i < values.size(); ++i) {
    projector.Insert(values(i));
  }
  projector.ProjectToTensorVector(values);
}
// END_SKIP_DOXYGEN

}  // namespace lattice
}  // namespace tensorflow

#endif  // TENSORFLOW_LATTICE_CC_KERNELS_MONOTONIC_PROJECTIONS_H_
