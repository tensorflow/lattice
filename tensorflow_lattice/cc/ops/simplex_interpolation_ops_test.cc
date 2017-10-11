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
#include <string>
#include <vector>

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace lattice {

class SimplexInterpolationOpsTest : public OpsTestBase {
 protected:
  SimplexInterpolationOpsTest() {}
};

TEST_F(SimplexInterpolationOpsTest, ThreeDoubleLattice) {
  const std::vector<int> lattice_sizes = {3};
  TF_ASSERT_OK(NodeDefBuilder("simplex_interpolation", "SimplexInterpolation")
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

TEST_F(SimplexInterpolationOpsTest, TwoByTwoFloatLattice) {
  const std::vector<int> lattice_sizes = {2, 2};
  TF_ASSERT_OK(NodeDefBuilder("simplex_interpolation", "SimplexInterpolation")
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
  // [0.5, 0, 0, 0.5], [0.2, 0, 0.6, 0.2], [0.7, 0, 0.1, 0.2]
  Tensor expected_weights(DT_FLOAT, TensorShape({7, 4}));
  test::FillValues<float>(
      &expected_weights,
      {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
       0.0, 1.0, 0.5, 0.0, 0.0, 0.5, 0.2, 0.0, 0.6, 0.2, 0.7, 0.0, 0.1, 0.2});

  LOG(INFO) << "Input: " << GetInput(0).SummarizeValue(14);
  LOG(INFO) << "Expected weight: " << expected_weights.SummarizeValue(28);
  LOG(INFO) << "Result: " << GetOutput(0)->SummarizeValue(28);
  test::ExpectTensorEqual<float>(expected_weights, *GetOutput(0));
}

TEST(SimplexInterpolationOpsShapeTest, SimplexInterpolation_ShapeFn) {
  ShapeInferenceTestOp op("SimplexInterpolation");

  // Total number of weights = 3 x 2 x 3 = 18.
  // Output dimension is always 18.
  std::vector<int> lattice_sizes = {3, 2, 3};
  TF_ASSERT_OK(NodeDefBuilder("test", "SimplexInterpolation")
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("lattice_sizes", lattice_sizes)
                   .Finalize(&op.node_def));

  INFER_OK(op, "[10,3]", "[d0_0,18]");
  INFER_OK(op, "[?,3]", "[d0_0,18]");

  INFER_ERROR("", op, "[?,?]");
  INFER_ERROR("Shape must be rank 2 but is rank 3", op, "[?,?,1]");
  INFER_ERROR("Shape must be rank 2 but is rank 1", op, "[10]");
  INFER_ERROR("Dimension must be 3 but is 2", op, "[?,2]");
  INFER_ERROR("Dimension must be 3 but is 2", op, "[5,2]");
}

TEST(SimplexGradientOpsShapeTest, SimplexGradient_ShapeFn) {
  ShapeInferenceTestOp op("SimplexGradient");

  // Total number of weights = 3 x 2 x 3 = 18.
  // Output dimension is always 18.
  std::vector<int> lattice_sizes = {3, 2, 3};
  TF_ASSERT_OK(NodeDefBuilder("test", "SimplexGradient")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("lattice_sizes", lattice_sizes)
                   .Finalize(&op.node_def));

  INFER_OK(op, "[10,3];[10,18];[10,18]", "[d0_0,d0_1]");
  INFER_OK(op, "[?,3];[?,18];[?,18]", "[d0_0,d0_1]");

  INFER_ERROR("", op, "[?,?]");
  INFER_ERROR("Shape must be rank 2 but is rank 3", op, "[?,?,1];[?,1];[?,1]");
  INFER_ERROR("Shape must be rank 2 but is rank 1", op, "[10];[?,?,1];[?,?,1]");
  INFER_ERROR("Dimension must be 3 but is 2", op, "[?,2];[2,3];[2,3]");
  INFER_ERROR("Dimension must be 3 but is 2", op, "[5,2];[5,5];[5,5]");
  INFER_ERROR("Shape must be rank 2 but is rank 3", op, "[2,3];[?,1,3];[?,1]");
  INFER_ERROR("Shape must be rank 2 but is rank 1", op, "[2,3];[10];[10]");
  INFER_ERROR("Input batch size (2) != Weight batch size (5)", op,
              "[2,3];[5,18];[5,18]");
  INFER_ERROR("Weight shape ([2,18]) != GradWrtWeight shape ([5,18])", op,
              "[2,3];[2,18];[5,18]");
  INFER_ERROR("Weight shape ([2,18]) != GradWrtWeight shape ([2,15])", op,
              "[2,3];[2,18];[2,15]");
  INFER_ERROR("Dimension must be 18 but is 17", op, "[?,3];[?,17];[?,17]");
  INFER_ERROR("Dimension must be 18 but is 5", op, "[5,3];[5,5];[5,5]");
}

}  // namespace lattice
}  // namespace tensorflow
