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
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace lattice {

using ::tensorflow::gtl::ArraySlice;

extern void PwlSetTestMode(bool split_batch);

class PwlIndexingCalibratorOpTest : public OpsTestBase {
 protected:
  void PwlIndexingCalibratorHelper(const bool use_sparse) {
    ArraySlice<double> keypoints_inputs{0.0, 20.0, 40.0, 60.0, 80.0, 100.0};
    const int num_keypoints = keypoints_inputs.size();

    struct Test {
      ArraySlice<double> uncalibrated;
      ArraySlice<double> expected_weights;
      ArraySlice<int64> expected_indices;
    };
    std::vector<Test> tests{
        // Bounded min.
        {{-10.0}, {1.0, 0.0}, {0, 1}},

        // Bounded max.
        {{200.0}, {0.0, 1.0}, {4, 5}},

        // Exact match.
        {{80.0}, {0.0, 1.0, 0.0}, {3, 4, 5}},

        // Interpolated examples.
        {{10.0}, {0.5, 0.5}, {0, 1}},
        {{35.0}, {0.25, 0.75}, {1, 2}},
    };

    LOG(INFO) << "Keypoints inputs: "
              << "[" << str_util::Join(keypoints_inputs, ",") << "]";
    for (const auto &test : tests) {
      inputs_.clear();
      TF_ASSERT_OK(NodeDefBuilder("pwl_indexing_calibrator:0",
                                  use_sparse ? "PwlIndexingCalibratorSparse"
                                             : "PwlIndexingCalibrator")
                       .Input(FakeInput(DT_DOUBLE))  //
                       .Input(FakeInput(DT_DOUBLE))
                       .Finalize(node_def()));
      TF_ASSERT_OK(InitOp());
      int batch_size = test.uncalibrated.size();
      AddInputFromArray<double>(TensorShape({batch_size}), test.uncalibrated);
      AddInputFromArray<double>(TensorShape({6}), keypoints_inputs);
      LOG(INFO) << "Testing for uncalibrated="
                << "[" << str_util::Join(test.uncalibrated, ",") << "]";
      TF_ASSERT_OK(RunOpKernel());

      if (use_sparse) {
        // Sparse implementation.
        Tensor expected_weights(
            allocator(), DT_DOUBLE,
            TensorShape({static_cast<int64>(test.expected_weights.size())}));
        test::FillValues<double>(&expected_weights, test.expected_weights);
        test::ExpectTensorEqual<double>(expected_weights, *GetOutput(1));

        Tensor expected_indices(
            allocator(), DT_INT64,
            TensorShape({static_cast<int64>(test.expected_indices.size()), 2}));
        std::vector<int64> flattened_indices_with_batch;
        for (int64 index : test.expected_indices) {
          flattened_indices_with_batch.push_back(0);  // batch index, always 0
          flattened_indices_with_batch.push_back(index);
        }
        test::FillValues<int64>(&expected_indices,
                                flattened_indices_with_batch);
        LOG(INFO) << "Expected: "
                  << "[" << str_util::Join(test.expected_indices, ",") << "]";

        test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(0));

      } else {
        // Dense implementation.
        Tensor expected_weights(
            allocator(), DT_DOUBLE,
            TensorShape({1, static_cast<int64>(num_keypoints)}));
        std::vector<double> weights(num_keypoints, 0);
        for (int i = 0; i < test.expected_weights.size(); i++) {
          weights[test.expected_indices[i]] = test.expected_weights[i];
        }
        test::FillValues<double>(&expected_weights, weights);
        test::ExpectTensorEqual<double>(expected_weights, *GetOutput(0));
      }
    }

    // Test batch version
    inputs_.clear();
    TF_ASSERT_OK(NodeDefBuilder("pwl_indexing_calibrator:1",
                                use_sparse ? "PwlIndexingCalibratorSparse"
                                           : "PwlIndexingCalibrator")
                     .Input(FakeInput(DT_DOUBLE))  //
                     .Input(FakeInput(DT_DOUBLE))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());

    std::vector<double> all_uncalibrated;
    for (const auto &test : tests) {
      all_uncalibrated.push_back(test.uncalibrated[0]);
    }
    AddInputFromArray<double>(
        TensorShape({static_cast<int64>(all_uncalibrated.size())}),
        all_uncalibrated);

    AddInputFromArray<double>(TensorShape({6}), keypoints_inputs);

    LOG(INFO) << "Testing for batch of all uncalibrated values: uncalibrated="
              << "[" << str_util::Join(all_uncalibrated, ",") << "]";
    TF_ASSERT_OK(RunOpKernel());

    if (use_sparse) {
      // Sparse implementation.
      std::vector<int64> vec_indices;
      std::vector<double> vec_weights;
      for (int j = 0; j < tests.size(); j++) {
        const Test &test = tests[j];
        for (int64 idx : test.expected_indices) {
          // Each example takes two coordinates.
          vec_indices.push_back(j);
          vec_indices.push_back(idx);
        }
        for (double w : test.expected_weights) {
          vec_weights.push_back(w);
        }
      }

      Tensor expected_weights(
          allocator(), DT_DOUBLE,
          TensorShape({static_cast<int64>(vec_weights.size())}));
      test::FillValues<double>(&expected_weights, vec_weights);

      Tensor expected_indices(
          allocator(), DT_INT64,
          TensorShape({static_cast<int64>(vec_weights.size()), 2}));
      test::FillValues<int64>(&expected_indices, vec_indices);

      const Tensor &output_indices = *GetOutput(0);
      const Tensor &output_weights = *GetOutput(1);
      test::ExpectTensorEqual<int64>(expected_indices, output_indices);
      test::ExpectTensorEqual<double>(expected_weights, output_weights);

    } else {
      // Batch dense version.
      Tensor expected_weights(allocator(), DT_DOUBLE,
                              TensorShape({static_cast<int64>(tests.size()),
                                           static_cast<int64>(num_keypoints)}));
      std::vector<double> weights(tests.size() * num_keypoints, 0);
      for (int j = 0; j < tests.size(); j++) {
        const Test &test = tests[j];
        for (int i = 0; i < test.expected_weights.size(); i++) {
          weights[j * num_keypoints + test.expected_indices[i]] =
              test.expected_weights[i];
        }
      }
      test::FillValues<double>(&expected_weights, weights);
      test::ExpectTensorEqual<double>(expected_weights, *GetOutput(0));
    }
  }

  void PwlIndexingCalibratorFloatHelper(const bool use_sparse) {
    ArraySlice<float> keypoints_inputs{0.0, 20.0, 40.0, 60.0, 80.0, 100.0};
    const int num_keypoints = keypoints_inputs.size();

    TF_ASSERT_OK(NodeDefBuilder("pwl_indexing_calibrator:0",
                                use_sparse ? "PwlIndexingCalibratorSparse"
                                           : "PwlIndexingCalibrator")
                     .Input(FakeInput(DT_FLOAT))  //
                     .Input(FakeInput(DT_FLOAT))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
    constexpr float uncalibrated = 200.0;
    AddInputFromArray<float>(TensorShape({1}), {uncalibrated});
    AddInputFromArray<float>(TensorShape({6}), keypoints_inputs);
    TF_ASSERT_OK(RunOpKernel()) << "Failed for uncalibrated="
                                << "[" << uncalibrated << "]";
    LOG(INFO) << "Testing for uncalibrated="
              << "[" << uncalibrated << "]";

    if (use_sparse) {
      // Sparse implementation.
      Tensor expected_weights(allocator(), DT_FLOAT, TensorShape({2}));
      test::FillValues<float>(&expected_weights, {0.0, 1.0});
      test::ExpectTensorEqual<float>(expected_weights, *GetOutput(1));

      Tensor expected_indices(allocator(), DT_INT64, TensorShape({2, 2}));
      test::FillValues<int64>(&expected_indices, {0, 4, 0, 5});
      test::ExpectTensorEqual<int64>(expected_indices, *GetOutput(0));

    } else {
      // Dense implementation.
      Tensor expected_weights(
          allocator(), DT_FLOAT,
          TensorShape({1, static_cast<int64>(num_keypoints)}));
      std::vector<float> values(num_keypoints, 0);
      values[5] = 1;
      test::FillValues<float>(&expected_weights, values);
      test::ExpectTensorEqual<float>(expected_weights, *GetOutput(0));
    }
  }

  void PwlIndexingCalibratorGradientHelper(const bool use_sparse) {
    ArraySlice<double> keypoints_inputs{0.0, 20.0, 40.0, 60.0, 80.0, 100.0};
    ArraySlice<double> weights_grad{0.0, 1.0, 2.0, 4.0, 8.0, 10.0};
    ArraySlice<double> grad_wrt_kp_inputs_values{0, 0, 0, 0, 0, 0};

    struct Test {
      ArraySlice<double> uncalibrated;
      ArraySlice<double> interpolation_weights;

      // interpolation_indices: 2 numbers per value: batch_index, weight_index.
      ArraySlice<int64> interpolation_indices;

      ArraySlice<double> grad_wrt_input;

      // Indices that would been used by the sparse interpolation, see
      // FindExpandedInterpolationIndices.
      std::vector<int> keypoint_indices;
    };
    std::vector<Test> tests{
        // At min, gradient should be based on slope of the first piece.
        {{-10.0}, {1.0, 0.0}, {0, 1}, {1.0 / 20.0}},

        // At max, gradient should be based on slope of the last piece.
        {{200.0}, {0.0, 1.0}, {4, 5}, {2.0 / 20.0}},

        // At a keypoint, slope should be mean of two slopes:
        {{40.0}, {0.0, 1.0, 0.0}, {1, 2, 3}, {(1.0 / 20 + 2.0 / 20) / 2}},

        // Interpolated examples.
        {{10.0}, {0.5, 0.5}, {0, 1}, {1.0 / 20}},
        {{75.0}, {0.25, 0.75}, {3, 4}, {4.0 / 20}},
    };

    LOG(INFO) << "Keypoints inputs: "
              << "[" << str_util::Join(keypoints_inputs, ",") << "]";

    for (const auto &test : tests) {
      inputs_.clear();
      if (use_sparse) {
        TF_ASSERT_OK(NodeDefBuilder("pwl_indexing_calibrator_gradient:0",
                                    "PwlIndexingCalibratorSparseGradient")
                         .Input(FakeInput(DT_DOUBLE))
                         .Input(FakeInput(DT_DOUBLE))
                         .Input(FakeInput(DT_INT64))
                         .Input(FakeInput(DT_DOUBLE))
                         .Finalize(node_def()));
      } else {
        TF_ASSERT_OK(NodeDefBuilder("pwl_indexing_calibrator_gradient:0",
                                    "PwlIndexingCalibratorGradient")
                         .Input(FakeInput(DT_DOUBLE))
                         .Input(FakeInput(DT_DOUBLE))
                         .Input(FakeInput(DT_DOUBLE))
                         .Finalize(node_def()));
      }
      TF_ASSERT_OK(InitOp());

      // Input being calibrated.
      AddInputFromArray<double>(TensorShape({1}), test.uncalibrated);

      // Parameters of calibration: the keypoints input values.
      AddInputFromArray<double>(TensorShape({6}), keypoints_inputs);

      // The gradient with respect to the output: presumably the keypoints
      // outputs if they are the last layer.
      if (use_sparse) {
        // Add interpolation indices, that will be provided for sparse
        // gradients.
        std::vector<int64> flattened_interpolation_indices_with_batch;
        for (const int64 weight_index : test.interpolation_indices) {
          flattened_interpolation_indices_with_batch.push_back(0);  // batch_idx
          flattened_interpolation_indices_with_batch.push_back(weight_index);
        }
        AddInputFromArray<int64>(
            TensorShape(
                {static_cast<int64>(test.interpolation_indices.size()), 2}),
            flattened_interpolation_indices_with_batch);

        std::vector<double> sparse_weights_grad;
        for (const int64 weight_index : test.interpolation_indices) {
          sparse_weights_grad.push_back(weights_grad[weight_index]);
        }
        AddInputFromArray<double>(
            TensorShape({static_cast<int64>(sparse_weights_grad.size())}),
            sparse_weights_grad);
      } else {
        AddInputFromArray<double>(TensorShape({1, 6}), weights_grad);
      }
      LOG(INFO) << "Testing for uncalibrated="
                << "[" << str_util::Join(test.uncalibrated, ",") << "]";
      TF_ASSERT_OK(RunOpKernel());

      Tensor grad_wrt_input(allocator(), DT_DOUBLE, TensorShape({1}));
      test::FillValues<double>(&grad_wrt_input, test.grad_wrt_input);
      test::ExpectTensorEqual<double>(grad_wrt_input, *GetOutput(0));

      Tensor grad_wrt_kp_inputs(
          allocator(), DT_DOUBLE,
          TensorShape({static_cast<int64>(grad_wrt_kp_inputs_values.size())}));
      test::FillValues<double>(&grad_wrt_kp_inputs, grad_wrt_kp_inputs_values);
      test::ExpectTensorEqual<double>(grad_wrt_kp_inputs, *GetOutput(1));
    }

    // Evaluate all tests in one batch.
    inputs_.clear();
    if (use_sparse) {
      TF_ASSERT_OK(NodeDefBuilder("pwl_indexing_calibrator_gradient:0",
                                  "PwlIndexingCalibratorSparseGradient")
                       .Input(FakeInput(DT_DOUBLE))
                       .Input(FakeInput(DT_DOUBLE))
                       .Input(FakeInput(DT_INT64))
                       .Input(FakeInput(DT_DOUBLE))
                       .Finalize(node_def()));
    } else {
      TF_ASSERT_OK(NodeDefBuilder("pwl_indexing_calibrator_gradient:0",
                                  "PwlIndexingCalibratorGradient")
                       .Input(FakeInput(DT_DOUBLE))
                       .Input(FakeInput(DT_DOUBLE))
                       .Input(FakeInput(DT_DOUBLE))
                       .Finalize(node_def()));
    }
    TF_ASSERT_OK(InitOp());

    // Input being calibrated.
    std::vector<double> all_uncalibrated;
    for (const auto &test : tests) {
      all_uncalibrated.push_back(test.uncalibrated[0]);
    }
    AddInputFromArray<double>(TensorShape({static_cast<int64>(tests.size())}),
                              all_uncalibrated);

    // Parameters of calibration: the keypoints' input values.
    AddInputFromArray<double>(TensorShape({6}), keypoints_inputs);

    // The gradient with respect to the output: presumably the keypoints'
    // outputs if they are the last layer.
    if (use_sparse) {
      std::vector<double> grad_wrt_weights_sparse;
      std::vector<int64> interpolation_indices_with_batch;
      for (int batch_index = 0; batch_index < tests.size(); batch_index++) {
        const auto &test = tests[batch_index];
        for (const int weight_index : test.interpolation_indices) {
          grad_wrt_weights_sparse.push_back(weights_grad[weight_index]);
          interpolation_indices_with_batch.push_back(batch_index);
          interpolation_indices_with_batch.push_back(weight_index);
        }
      }
      AddInputFromArray<int64>(
          TensorShape({static_cast<int64>(grad_wrt_weights_sparse.size()), 2}),
          interpolation_indices_with_batch);
      AddInputFromArray<double>(
          TensorShape({static_cast<int64>(grad_wrt_weights_sparse.size())}),
          grad_wrt_weights_sparse);
    } else {
      // Repeat weights_grad for each test.
      std::vector<double> repeated_weights_grad;
      for (int i = 0; i < tests.size(); i++) {
        for (const double w : weights_grad) {
          repeated_weights_grad.push_back(w);
        }
      }
      AddInputFromArray<double>(
          TensorShape({static_cast<int64>(tests.size()), 6}),
          repeated_weights_grad);
    }
    LOG(INFO) << "Testing for all tests in one batch";
    TF_ASSERT_OK(RunOpKernel());

    Tensor grad_wrt_input(allocator(), DT_DOUBLE,
                          TensorShape({static_cast<int64>(tests.size())}));
    std::vector<double> all_grad_wrt_input;
    for (const auto &test : tests) {
      all_grad_wrt_input.push_back(test.grad_wrt_input[0]);
    }
    test::FillValues<double>(&grad_wrt_input, all_grad_wrt_input);
    test::ExpectTensorEqual<double>(grad_wrt_input, *GetOutput(0));

    Tensor grad_wrt_kp_inputs(
        allocator(), DT_DOUBLE,
        TensorShape({static_cast<int64>(grad_wrt_kp_inputs_values.size())}));
    test::FillValues<double>(&grad_wrt_kp_inputs, grad_wrt_kp_inputs_values);
    test::ExpectTensorEqual<double>(grad_wrt_kp_inputs, *GetOutput(1));
  }
};

TEST_F(PwlIndexingCalibratorOpTest, PwlIndexingCalibratorDense) {
  LOG(INFO) << "Process whole batch at once: (split_batch=false)";
  PwlSetTestMode(/*split_batch=*/false);
  PwlIndexingCalibratorHelper(false);

  LOG(INFO) << "Process whole batch in splits: (split_batch=true)";
  PwlSetTestMode(/*split_batch=*/true);
  PwlIndexingCalibratorHelper(false);
  PwlSetTestMode(/*split_batch=*/false);
}

TEST_F(PwlIndexingCalibratorOpTest, PwlIndexingCalibratorSparse) {
  PwlIndexingCalibratorHelper(true);
}

TEST_F(PwlIndexingCalibratorOpTest, PwlIndexingCalibratorFloatDense) {
  PwlIndexingCalibratorFloatHelper(false);
}

TEST_F(PwlIndexingCalibratorOpTest, PwlIndexingCalibratorFloatSparse) {
  PwlIndexingCalibratorFloatHelper(true);
}

TEST_F(PwlIndexingCalibratorOpTest, PwlIndexingCalibratorGradientDense) {
  PwlIndexingCalibratorGradientHelper(false);
}

TEST_F(PwlIndexingCalibratorOpTest, PwlIndexingCalibratorGradientSparse) {
  PwlIndexingCalibratorGradientHelper(true);
}

TEST_F(PwlIndexingCalibratorOpTest, PwlIndexingCalibrator_ShapeFn) {
  ShapeInferenceTestOp op("PwlIndexingCalibrator");
  TF_ASSERT_OK(NodeDefBuilder("test", "PwlIndexingCalibrator")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Finalize(&op.node_def));

  INFER_OK(op, "[20];[10]", "[d0_0,d1_0]");
  INFER_OK(op, "[?];[10]", "[d0_0,d1_0]");

  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[20,1];[10]");
}

TEST_F(PwlIndexingCalibratorOpTest, PwlIndexingCalibratorGradient_ShapeFn) {
  ShapeInferenceTestOp op("PwlIndexingCalibratorGradient");
  TF_ASSERT_OK(NodeDefBuilder("pwl_indexing_calibrator_gradient:1",
                              "PwlIndexingCalibratorGradient")
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Input(FakeInput(DT_FLOAT))
                   .Finalize(&op.node_def));

  INFER_OK(op, "[11];[13];[11,13]", "[d0_0];[d1_0]");
  INFER_OK(op, "[?];[7];[?,7]", "[d0_0];[d1_0]");

  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[20,1];[11];[20,11]");
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[20];[11,1];[20,11]");
  INFER_ERROR("Shape must be rank 2 but is rank 1", op, "[20];[11];[20]");
  INFER_ERROR(
      "grad_wrt_weights has shape [17,11], but weights has shape [20,11]", op,
      "[20];[11];[17,11]");
}

}  // namespace lattice
}  // namespace tensorflow
