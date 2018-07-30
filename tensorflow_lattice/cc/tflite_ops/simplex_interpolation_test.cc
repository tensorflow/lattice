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
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/kernels/test_util.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow_lattice/cc/tflite_ops/tflite_ops.h"

namespace tflite {
namespace ops {
namespace custom {


namespace {

class SimplexInterpolationOp : public SingleOpModel {
 public:
  SimplexInterpolationOp(const TensorData& input, const TensorData& output,
                           std::vector<int> lattice_sizes) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    flexbuffers::Builder fbb;
    size_t map_start = fbb.StartMap();
    auto vec_start = fbb.StartVector("lattice_sizes");
    for (int i = 0; i < lattice_sizes.size(); ++i) {
      fbb.Add(lattice_sizes[i]);
    }
    fbb.EndVector(vec_start, /* typed */ true, /* fixed */ false);
    fbb.EndMap(map_start);
    fbb.Finish();
    SetCustomOp("SimplexInterpolation", fbb.GetBuffer(),
                Register_SIMPLEX_INTERPOLATION);

    BuildInterpreter({GetShape(input_)});
  }

  int input() { return input_; }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int output_;
};

TEST(Test1D, SimplexInterpolationTest) {
  SimplexInterpolationOp m({TensorType_FLOAT32, {8, 1}},
                             {TensorType_FLOAT32, {}}, {3});
  m.PopulateTensor<float>(m.input(), {
    -1.0, 0.0, 0.2, 0.8, 1.0, 1.3, 2.0, 2.5
  });
  m.Invoke();
  std::vector<float> out = {
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    0.8, 0.2, 0.0,
    0.2, 0.8, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.7, 0.3,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0
  };
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(out, 1e-3)));
}

TEST(Test2D, SimplexInterpolationTest) {
  SimplexInterpolationOp m({TensorType_FLOAT32, {7, 2}},
                             {TensorType_FLOAT32, {}}, {2, 2});
  m.PopulateTensor<float>(m.input(), {
    0.0, 0.0,
    0.0, 1.0,
    1.0, 0.0,
    1.0, 1.0,
    0.5, 0.5,
    0.2, 0.8,
    0.2, 0.3
  });
  m.Invoke();
  std::vector<float> out = {
    1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
    0.5, 0.0, 0.0, 0.5,
    0.2, 0.0, 0.6, 0.2,
    0.7, 0.0, 0.1, 0.2
  };
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(out, 1e-3)));
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
