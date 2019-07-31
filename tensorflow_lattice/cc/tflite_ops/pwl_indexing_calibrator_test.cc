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

class PWLIndexingCalibratorOp : public SingleOpModel {
 public:
  PWLIndexingCalibratorOp(const TensorData& input, const TensorData& kp_inputs,
                          const TensorData& output) {
    input_ = AddInput(input);
    kp_inputs_ = AddInput(kp_inputs);
    output_ = AddOutput(output);
    SetCustomOp("PWLIndexingCalibratorOp", {},
                Register_PWL_INDEXING_CALIBRATOR);

    BuildInterpreter({GetShape(input_), GetShape(kp_inputs_)});
  }

  int input() { return input_; }
  int kp_inputs() { return kp_inputs_; }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int kp_inputs_;
  int output_;
};

TEST(TestBasic, PWLIndexingCalibratorTest) {
  PWLIndexingCalibratorOp m({TensorType_FLOAT32, {6}},
                            {TensorType_FLOAT32, {4}},
                            {TensorType_FLOAT32, {}});
  m.PopulateTensor<float>(m.input(), {-3.0, 0.0, 0.1, 0.5, 0.75, 2.0});
  m.PopulateTensor<float>(m.kp_inputs(), {0.0, 0.25, 0.5, 0.75});
  m.Invoke();
  std::vector<float> out = {
      1.0, 0.0, 0.0, 0.0,
      1.0, 0.0, 0.0, 0.0,
      0.6, 0.4, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0,
      0.0, 0.0, 0.0, 1.0,
  };
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(out, 1e-3)));
}

TEST(TestSingle, PWLIndexingCalibratorTest) {
  PWLIndexingCalibratorOp m({TensorType_FLOAT32, {3}},
                            {TensorType_FLOAT32, {1}},
                            {TensorType_FLOAT32, {}});
  m.PopulateTensor<float>(m.input(), {-1.0, 0.0, 1.1});
  m.PopulateTensor<float>(m.kp_inputs(), {0.0});
  m.Invoke();
  std::vector<float> out = {
      1.0,
      1.0,
      1.0,
  };
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(out, 1e-3)));
}

TEST(TestTF, PWLIndexingCalibratorTest) {
  std::vector<float> keypoints = {0.0, 20.0, 40.0, 60.0, 80.0, 100.0};
  struct Test {
    std::vector<float> uncalibrated;
    std::vector<float> expected_weights;
    std::vector<int> expected_indices;
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

  for (auto& test : tests) {
    PWLIndexingCalibratorOp m(
        {TensorType_FLOAT32, {(int)test.uncalibrated.size()}},
        {TensorType_FLOAT32, {(int)keypoints.size()}},
        {TensorType_FLOAT32, {}});
    m.PopulateTensor<float>(
        m.input(), 0, test.uncalibrated.data(),
        test.uncalibrated.data() + test.uncalibrated.size());
    m.PopulateTensor<float>(m.kp_inputs(), 0, keypoints.data(),
                            keypoints.data() + keypoints.size());
    m.Invoke();
    std::vector<float> out(keypoints.size() * test.uncalibrated.size());
    for (int ii = 0; ii < out.size(); ++ii) {
      out[ii] = 0.0;
    }
    for (int kk = 0; kk < test.expected_weights.size(); ++kk) {
      out[test.expected_indices[kk]] = test.expected_weights[kk];
    }
    EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(out, 1e-3)));
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
