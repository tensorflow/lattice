/* Copyright 2018 The TensorFlow Lattice Authors.

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
#include <math.h>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flexbuffers.h"
#include "tensorflow_lattice/cc/tflite_ops/tflite_ops.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace ops {
namespace custom {

namespace {

class PWLIndexingCalibratorSparseOp : public SingleOpModel {
 public:
  PWLIndexingCalibratorSparseOp(const TensorData& input,
                                const TensorData& kp_inputs,
                                const TensorData& indices_output,
                                const TensorData& weights_output) {
    input_ = AddInput(input);
    kp_inputs_ = AddInput(kp_inputs);
    indices_output_ = AddOutput(indices_output);
    weights_output_ = AddOutput(weights_output);
    SetCustomOp("PWLIndexingCalibratorSparseOp", {},
                Register_PWL_INDEXING_CALIBRATOR_SPARSE);

    BuildInterpreter({GetShape(input_), GetShape(kp_inputs_)});
  }

  int input() { return input_; }
  int kp_inputs() { return kp_inputs_; }
  std::vector<int> GetIndicesOutput() {
    return ExtractVector<int>(indices_output_);
  }
  std::vector<int> GetIndicesOutputShape() {
    return GetTensorShape(indices_output_);
  }
  std::vector<float> GetWeightsOutput() {
    return ExtractVector<float>(weights_output_);
  }
  std::vector<int> GetWeightsOutputShape() {
    return GetTensorShape(weights_output_);
  }

 private:
  int input_;
  int kp_inputs_;
  int indices_output_;
  int weights_output_;
};

TEST(TestBasic, PWLIndexingCalibratorSparseTest) {
  PWLIndexingCalibratorSparseOp m(
      {TensorType_FLOAT32, {6}}, {TensorType_FLOAT32, {4}},
      {TensorType_INT32, {}}, {TensorType_FLOAT32, {}});
  m.PopulateTensor<float>(m.input(), {-3.0, 0.0, 0.1, 0.5, 0.75, 2.0});
  m.PopulateTensor<float>(m.kp_inputs(), {0.0, 0.25, 0.5, 0.75});
  m.Invoke();
  std::vector<int> indices_out = {
      0, 0,
      1, 0,
      2, 0,
      2, 1,
      3, 2,
      4, 3,
      5, 3,
  };
  std::vector<float> weights_out = {
      1.0,
      1.0,
      0.6,
      0.4,
      1.0,
      1.0,
      1.0,
  };
  EXPECT_THAT(m.GetIndicesOutput(), testing::ElementsAreArray(indices_out));
  EXPECT_THAT(m.GetWeightsOutput(),
              ElementsAreArray(ArrayFloatNear(weights_out, 1e-3)));
}

TEST(TestTF, PWLIndexingCalibratorSparseTest) {
  std::vector<float> keypoints = {0.0, 20.0, 40.0, 60.0, 80.0, 100.0};
  struct Test {
    std::vector<float> uncalibrated;
    std::vector<float> expected_weights;
    std::vector<int> expected_indices;
  };
  std::vector<Test> tests{
      // Bounded min.
      {{-10.0}, {1.0}, {0}},

      // Bounded max.
      {{200.0}, {1.0}, {5}},

      // Exact match.
      {{80.0}, {1.0}, {4}},

      // Interpolated examples.
      {{10.0}, {0.5, 0.5}, {0, 1}},
      {{35.0}, {0.25, 0.75}, {1, 2}},
  };

  for (auto &test : tests) {
    PWLIndexingCalibratorSparseOp m(
        {TensorType_FLOAT32, {(int)test.uncalibrated.size()}},
        {TensorType_FLOAT32, {(int)keypoints.size()}}, {TensorType_FLOAT32, {}},
        {TensorType_FLOAT32, {}});
    m.PopulateTensor<float>(
        m.input(), 0, test.uncalibrated.data(),
        test.uncalibrated.data() + test.uncalibrated.size());
    m.PopulateTensor<float>(m.kp_inputs(), 0, keypoints.data(),
                            keypoints.data() + keypoints.size());
    m.Invoke();
    std::vector<int> indices_out(2 * test.expected_weights.size());
    std::vector<float> weights_out(test.expected_weights.size());
    for (int kk = 0; kk < test.expected_weights.size(); ++kk) {
      int indices_out_row_offset = 2 * kk;
      indices_out[indices_out_row_offset] = 0;
      indices_out[indices_out_row_offset + 1] = test.expected_indices[kk];
      weights_out[kk] = test.expected_weights[kk];
    }
    EXPECT_THAT(m.GetIndicesOutput(), testing::ElementsAreArray(indices_out));
    EXPECT_THAT(m.GetWeightsOutput(),
                ElementsAreArray(ArrayFloatNear(weights_out, 1e-3)));
  }
}

}  // namespace
}  // namespace custom
}  // namespace ops
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
