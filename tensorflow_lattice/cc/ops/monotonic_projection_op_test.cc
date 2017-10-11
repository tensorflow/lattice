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
#include <vector>

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace lattice {

class MonotonicProjectionOpTest : public OpsTestBase {};

TEST_F(MonotonicProjectionOpTest, MonotonicProjection) {
  struct Test {
    bool increasing;
    gtl::ArraySlice<double> before;
    gtl::ArraySlice<double> expected;
  };
  std::vector<Test> tests{
      // No-op.
      {true, {}, {}},
      {true, {0, 1}, {0, 1}},
      {false, {1, 0}, {1, 0}},
      {true, {22.9}, {22.9}},
      {false, {22.9}, {22.9}},
      {true, {6.0, 7.0, 8.0}, {6.0, 7.0, 8.0}},

      // Short dependency.
      {true, {1, 0}, {0.5, 0.5}},
      {false, {0, 1}, {0.5, 0.5}},

      // Long dependencies.
      {true, {6.0, 1, 2, 3.5}, {3, 3, 3, 3.5}},
      {true, {10.0, 9.0, 8.0, 7.0, 6.0}, {8.0, 8.0, 8.0, 8.0, 8.0}},

      // Examples that require back-tracking of pools.
      {false, {2, 1, 6}, {3, 3, 3}},
      {true, {4, 5, 0}, {3, 3, 3}},
      {true, {4, 5, 0, 4, -3}, {2, 2, 2, 2, 2}},
      {true, {5.0, 6.0, 5.0, 6.0, 7.0, 6.0}, {5.0, 5.5, 5.5, 6.0, 6.5, 6.5}},
  };

  for (const auto &test : tests) {
    inputs_.clear();
    const int64 test_size = test.before.size();
    LOG(INFO) << "Testing for increasing=" << test.increasing << ", values=["
              << ::tensorflow::str_util::Join(test.before, ", ") << "]";

    TF_ASSERT_OK(NodeDefBuilder("monotonic_projection:0", "MonotonicProjection")
                     .Input("values", 0, DT_DOUBLE)
                     .Input("increasing", 0, DT_BOOL)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
    AddInputFromArray<double>(TensorShape({test_size}), test.before);
    AddInputFromList<bool>(TensorShape(), {test.increasing});
    TF_ASSERT_OK(RunOpKernel());

    Tensor expected(allocator(), DT_DOUBLE, TensorShape({test_size}));
    test::FillValues<double>(&expected, test.expected);
    test::ExpectTensorEqual<double>(expected, *GetOutput(0));
  }
}

TEST_F(MonotonicProjectionOpTest, MonotonicProjection_ShapeFn) {
  ShapeInferenceTestOp op("MonotonicProjection");
  TF_ASSERT_OK(NodeDefBuilder("monotonic_projection:1", "MonotonicProjection")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_BOOL))
                   .Finalize(&op.node_def));

  INFER_OK(op, "[11];[]", "in0");
  INFER_OK(op, "[17];[]", "in0");

  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[20,1];[]");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "[20];[1]");
}

}  // namespace lattice
}  // namespace tensorflow
