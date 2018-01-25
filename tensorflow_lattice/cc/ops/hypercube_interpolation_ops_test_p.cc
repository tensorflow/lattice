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
#define EIGEN_USE_THREADS

#include <string>
#include <vector>

#include "tensorflow_lattice/cc/ops/hypercube_interpolation_ops_test.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace lattice {
namespace {

TEST_P(HypercubeInterpolationOpsTest, ThreeDoubleLattice) {
  const std::vector<int> lattice_sizes = {3};
  TF_ASSERT_OK(
      NodeDefBuilder("hypercube_interpolation", "HypercubeInterpolation")
          .Input(FakeInput(DT_DOUBLE))
          .Attr("lattice_sizes", lattice_sizes)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  // Input tensor = [[-1], [0], [0.2], [0.8], [1.0], [1.3], [2.0], [2.5]].
  AddInputFromArray<double>(TensorShape({8, 1}),
                            {-1.0, 0.0, 0.2, 0.8, 1.0, 1.3, 2.0, 2.5});
  TF_ASSERT_OK(RunOpKernel());
  // expected weight = [[1, 0, 0], [1, 0, 0], [0.8, 0.2, 0], [0.2, 0.8, 0],
  // [0, 1, 0], [0, 0.7, 0.3], [0, 0, 1.0], [0, 0, 1.0]].
  Tensor expected_weights(DT_DOUBLE, TensorShape({8, 3}));
  test::FillValues<double>(
      &expected_weights,
      {1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.2, 0.8, 0.0,
       0.0, 1.0, 0.0, 0.0, 0.7, 0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0});

  LOG(INFO) << "Input: " << GetInput(0).SummarizeValue(8);
  LOG(INFO) << "Expected weight: " << expected_weights.SummarizeValue(24);
  LOG(INFO) << "Result: " << GetOutput(0)->SummarizeValue(24);
  test::ExpectTensorEqual<double>(expected_weights, *GetOutput(0));
}

TEST_P(HypercubeInterpolationOpsTest, TwoByTwoFloatLattice) {
  const std::vector<int> lattice_sizes = {2, 2};
  TF_ASSERT_OK(
      NodeDefBuilder("hypercube_interpolation", "HypercubeInterpolation")
          .Input(FakeInput(DT_FLOAT))
          .Attr("lattice_sizes", lattice_sizes)
          .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());
  // Input tensor = [[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5], [0.2, 0.8],
  // [0.2, 0.3]]
  AddInputFromArray<float>(
      TensorShape({7, 2}),
      {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.5, 0.5, 0.2, 0.8, 0.2, 0.3});
  TF_ASSERT_OK(RunOpKernel());
  // expected weight = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1],
  // [0.25, 0.25, 0.25, 0.25], [0.16, 0.04, 0.64, 0.16], [0.56, 0.14, 0.24,
  // 0.06]]
  Tensor expected_weights(DT_FLOAT, TensorShape({7, 4}));
  test::FillValues<float>(
      &expected_weights,
      {1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0,
       0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.25, 0.25, 0.25, 0.25,
       0.16, 0.04, 0.64, 0.16, 0.56, 0.14, 0.24, 0.06});

  LOG(INFO) << "Input: " << GetInput(0).SummarizeValue(14);
  LOG(INFO) << "Expected weight: " << expected_weights.SummarizeValue(28);
  LOG(INFO) << "Result: " << GetOutput(0)->SummarizeValue(28);
  test::ExpectTensorEqual<float>(expected_weights, *GetOutput(0));
}

}  // namespace
}  // namespace lattice
}  // namespace tensorflow
