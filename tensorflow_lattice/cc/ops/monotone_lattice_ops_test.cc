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

namespace {
class MonotoneLatticeOpsTest : public OpsTestBase {
 protected:
  MonotoneLatticeOpsTest() {}
  // Computes the projected_lattice_param_vec and compares the output
  // with expected_projected_lattice_param_vec.
  // In order to test batch parameter projection, this test method accepts a
  // list of lattice_param_vec and a list of
  // expected_projected_lattice_param_vec.
  void CheckProjection(
      const std::vector<int>& lattice_sizes,
      const std::vector<bool>& is_monotone,
      const std::vector<std::vector<double>>& lattice_param_vecs,
      const std::vector<std::vector<double>>&
          expected_projected_lattice_param_vecs) {
    constexpr double kEpsilon = 1e-5;
    const int num_inputs = lattice_param_vecs.size();
    ASSERT_GT(num_inputs, 0);
    const int num_parameters = lattice_param_vecs[0].size();

    // Pre-condition.
    ASSERT_EQ(expected_projected_lattice_param_vecs.size(), num_inputs);

    // Flattening vectors to fill-in tensors.
    std::vector<double> flattened_lattice_param_vecs;
    std::vector<double> flattened_expected_projection;
    flattened_lattice_param_vecs.reserve(num_inputs * num_parameters);
    flattened_expected_projection.reserve(num_inputs * num_parameters);
    for (int ii = 0; ii < num_inputs; ++ii) {
      ASSERT_EQ(lattice_param_vecs[ii].size(), num_parameters);
      ASSERT_EQ(expected_projected_lattice_param_vecs[ii].size(),
                num_parameters);
      for (int jj = 0; jj < num_parameters; ++jj) {
        flattened_lattice_param_vecs.push_back(lattice_param_vecs[ii][jj]);
        flattened_expected_projection.push_back(
            expected_projected_lattice_param_vecs[ii][jj]);
      }
    }

    // Define tensorflow ops to be tested.
    TF_ASSERT_OK(NodeDefBuilder("monotone_lattice", "MonotoneLattice")
                     .Input(FakeInput(DT_DOUBLE))
                     .Attr("lattice_sizes", lattice_sizes)
                     .Attr("is_monotone", is_monotone)
                     .Finalize(node_def()));

    TF_ASSERT_OK(InitOp());
    AddInputFromArray<double>(TensorShape({num_inputs, num_parameters}),
                              flattened_lattice_param_vecs);
    TF_ASSERT_OK(RunOpKernel());
    Tensor expected_projection_tensor(
        DT_DOUBLE, TensorShape({num_inputs, num_parameters}));
    test::FillValues<double>(&expected_projection_tensor,
                             flattened_expected_projection);

    VLOG(1) << "Lattice parameter tensor: "
            << GetInput(0).SummarizeValue(num_parameters);
    VLOG(1) << "Expected projection tensor: "
            << expected_projection_tensor.SummarizeValue(num_parameters);
    VLOG(1) << "Result tensor: "
            << GetOutput(0)->SummarizeValue(num_parameters);
    test::ExpectTensorNear<double>(expected_projection_tensor, *GetOutput(0),
                                   kEpsilon);
  }
};

TEST_F(MonotoneLatticeOpsTest, ProjectToNothing) {
  CheckProjection(
      /*lattice_sizes=*/{2, 2}, /*is_monotone=*/{false, false},
      /*lattice_param_vecs=*/{{3.0, 0.0, 2.0, 5.0}},
      /*expected_projected_lattice_param_vecs=*/{{3.0, 0.0, 2.0, 5.0}});
}

TEST_F(MonotoneLatticeOpsTest, ProjectTo0thDimension) {
  CheckProjection(
      /*lattice_sizes=*/{2, 2}, /*is_monotone=*/{true, false},
      /*lattice_param_vecs=*/{{3.0, 0.0, 2.0, 5.0}},
      /*expected_projected_lattice_param_vecs=*/{{1.5, 1.5, 2.0, 5.0}});
}

TEST_F(MonotoneLatticeOpsTest, ProjectTo1stDimension) {
  CheckProjection(
      /*lattice_sizes=*/{2, 2}, /*is_monotone=*/{false, true},
      /*lattice_param_vecs=*/{{3.0, 0.0, 2.0, 5.0}},
      /*expected_projected_lattice_param_vecs=*/{{2.5, 0.0, 2.5, 5.0}});
}

TEST_F(MonotoneLatticeOpsTest, ProjectToAllDimensions) {
  CheckProjection(
      /*lattice_sizes=*/{2, 2}, /*is_monotone=*/{true, true},
      /*lattice_param_vecs=*/{{3.0, 0.0, 2.0, 5.0}},
      /*expected_projected_lattice_param_vecs=*/{{1.5, 1.5, 2.0, 5.0}});
}

TEST_F(MonotoneLatticeOpsTest, ProjectThreeByTwoLatticeToAllDimensions) {
  CheckProjection(
      /*lattice_sizes=*/{3, 2}, /*is_monotone=*/{true, true},
      /*lattice_param_vecs=*/{{3.0, 1.0, 0.0, 0.0, 2.0, 5.0}},
      /*expected_projected_lattice_param_vecs=*/{
          {1.0, 1.0, 1.0, 1.0, 2.0, 5.0}});
}

TEST_F(MonotoneLatticeOpsTest, ProjectMultipleTwoByTwoLatticesToAllDimensions) {
  CheckProjection(
      /*lattice_sizes=*/{2, 2}, /*is_monotone=*/{true, true},
      /*lattice_param_vecs=*/{{3.0, 0.0, 2.0, 5.0},
                              {3.0, 0.0, 2.0, 5.0},
                              {0.0, 1.0, 2.0, 3.0},
                              {3.0, 3.0, 1.0, 1.0},
                              {-1.0, -5.0, 2.0, 3.0}},
      /*expected_projected_lattice_param_vecs=*/{{1.5, 1.5, 2.0, 5.0},
                                                 {1.5, 1.5, 2.0, 5.0},
                                                 {0.0, 1.0, 2.0, 3.0},
                                                 {2.0, 2.0, 2.0, 2.0},
                                                 {-3.0, -3.0, 2.0, 3.0}});
}

TEST(MonotoneLatticeOpsShapeTest, CorrectInference) {
  ShapeInferenceTestOp op("MonotoneLattice");

  // 2 x 2 x 2 lattice = 8 parameters.
  std::vector<int> lattice_sizes = {2, 2, 2};
  std::vector<bool> is_monotone = {true, true, true};
  TF_ASSERT_OK(NodeDefBuilder("test", "MonotoneLattice")
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("lattice_sizes", lattice_sizes)
                   .Attr("is_monotone", is_monotone)
                   .Finalize(&op.node_def));

  INFER_OK(op, "[3,8]", "in0");
  INFER_OK(op, "[10,8]", "in0");
}

TEST(MonotoneLatticeOpsShapeTest, WrongShapeShouldFail) {
  ShapeInferenceTestOp op("MonotoneLattice");

  // 2 x 2 x 2 lattice = 8 parameters.
  std::vector<int> lattice_sizes = {2, 2, 2};
  std::vector<bool> is_monotone = {true, true, true};
  TF_ASSERT_OK(NodeDefBuilder("test", "MonotoneLattice")
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("lattice_sizes", lattice_sizes)
                   .Attr("is_monotone", is_monotone)
                   .Finalize(&op.node_def));

  INFER_ERROR("Shape must be rank 2 but is rank 1", op, "[1]");
  INFER_ERROR("Shape must be rank 2 but is rank 3", op, "[1,2,3]");
  INFER_ERROR(
      "lattice_params' number of parameters (3) != expected number of "
      "parameters (8)",
      op, "[10,3]");
}

}  // namespace

}  // namespace lattice
}  // namespace tensorflow
