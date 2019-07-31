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

class HypercubeInterpolationOp : public SingleOpModel {
 public:
  HypercubeInterpolationOp(const TensorData& input, const TensorData& output,
                           std::vector<int> lattice_sizes) {
    input_ = AddInput(input);
    output_ = AddOutput(output);
    flexbuffers::Builder fbb;
    size_t map_start = fbb.StartMap();
    auto vec_start = fbb.StartVector("lattice_sizes");
    for (int ii = 0; ii < lattice_sizes.size(); ++ii) {
      fbb.Add(lattice_sizes[ii]);
    }
    fbb.EndVector(vec_start, /* typed */ true, /* fixed */ false);
    fbb.EndMap(map_start);
    fbb.Finish();
    SetCustomOp("HypercubeInterpolation", fbb.GetBuffer(),
                Register_HYPERCUBE_INTERPOLATION);

    BuildInterpreter({GetShape(input_)});
  }

  int input() { return input_; }
  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }
  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

 private:
  int input_;
  int output_;
};

TEST(Test2D, HypercubeInterpolationTest) {
  const float equal_vertex_weight_2d = 0.25;
  const int out_row_length = 9;  // 3 nodes ^ 2 dimensions
  const int n_cells = 4;         // (3 nodes - 1) ^ 2 dimensions
  HypercubeInterpolationOp m({TensorType_FLOAT32, {n_cells, 2}},
                             {TensorType_FLOAT32, {}}, {3, 3});
  m.PopulateTensor<float>(m.input(), {
    0.5, 0.5,
    0.5, 1.5,
    1.5, 0.5,
    1.5, 1.5,
  });
  m.Invoke();
  std::vector<float> out(out_row_length * n_cells, 0.0);
  int non_zero_indices[n_cells][4] = {
      {0, 1, 3, 4},
      {3, 4, 6, 7},
      {1, 2, 4, 5},
      {4, 5, 7, 8},
  };
  int row_offset;
  for (int ii = 0; ii < n_cells; ii++) {
    for (int ij : non_zero_indices[ii]) {
      row_offset = ii * out_row_length;
      out[row_offset + ij] = equal_vertex_weight_2d;
    }
  }
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(out, 1e-3)));
}

TEST(Test3D, HypercubeInterpolationTest) {
  const float equal_vertex_weight_3d = 0.125;
  const int out_row_length = 27;  // 3 nodes ^ 3 dimensions
  const int n_cells = 8;  // (3 nodes - 1) ^ 3 dimensions
  const int tier_stride = 9;  // 3 nodes ^ 2 dimensions
  HypercubeInterpolationOp m(
      {TensorType_FLOAT32, {n_cells, 3}}, {TensorType_FLOAT32, {}}, {3, 3, 3});
  m.PopulateTensor<float>(m.input(), {
      0.5, 0.5, 0.5,
      0.5, 1.5, 0.5,
      1.5, 0.5, 0.5,
      1.5, 1.5, 0.5,
      0.5, 0.5, 1.5,
      0.5, 1.5, 1.5,
      1.5, 0.5, 1.5,
      1.5, 1.5, 1.5,
  });
  m.Invoke();
  std::vector<float> out(out_row_length * n_cells, 0.0);
  // the 3D lattice of 9 cells is a stack of 2 'tiers' of 3x3 (4 cell) latice
  // the non-zero entries follow the same pattern as they did for the 2D case
  // except that the value must be propagated vertically to 8 vertices of a cube
  // instead of the 4 vertices of a square.  Also, the 2D must be iterated over
  // twice, once for each tier, but both will have the same 2D projection.
  int non_zero_indices[n_cells / 2][4] = {
      {0, 1, 3, 4},
      {3, 4, 6, 7},
      {1, 2, 4, 5},
      {4, 5, 7, 8},
  };
  int row_offset;
  int tier_offset;
  int on_tier;
  for (int ii = 0; ii < n_cells; ++ii) {
    row_offset = ii * out_row_length;
    on_tier = ii < 4 ? 0 : 1;
    for (int tier = 0; tier < 2; ++tier) {
      tier_offset = (on_tier + tier) * tier_stride;
      for (int ij : non_zero_indices[ii % 4]) {
        out[row_offset + tier_offset + ij] = equal_vertex_weight_3d;
      }
    }
  }
  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(out, 1e-3)));
}

TEST(ThreeDoubleLattice, HypercubeInterpolationTest) {
  HypercubeInterpolationOp m({TensorType_FLOAT32, {8, 1}},
                             {TensorType_FLOAT32, {}}, {3});
  m.PopulateTensor<float>(m.input(), {
                                         -1.0,
                                         0.0,
                                         0.2,
                                         0.8,
                                         1.0,
                                         1.3,
                                         2.0,
                                         2.5,
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
      0.0, 0.0, 1.0,
  };

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(out, 1e-3)));
}

TEST(TwoByTwoFloatLattice, HypercubeInterpolationTest) {
  HypercubeInterpolationOp m({TensorType_FLOAT32, {7, 2}},
                             {TensorType_FLOAT32, {}}, {2, 2});
  m.PopulateTensor<float>(m.input(), {
      0.0, 0.0,
      0.0, 1.0,
      1.0, 0.0,
      1.0, 1.0,
      0.5, 0.5,
      0.2, 0.8,
      0.2, 0.3,
  });
  m.Invoke();
  std::vector<float> out = {
      1.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 1.0,
      0.25, 0.25, 0.25, 0.25,
      0.16, 0.04, 0.64, 0.16,
      0.56, 0.14, 0.24, 0.06,
  };

  EXPECT_THAT(m.GetOutput(), ElementsAreArray(ArrayFloatNear(out, 1e-3)));
}

TEST(TestOOB, HypercubeInterpolationTest) {
  HypercubeInterpolationOp m({TensorType_FLOAT32, {3, 2}},
                             {TensorType_FLOAT32, {}}, {2, 2});
  m.PopulateTensor<float>(m.input(), {
      0.0,  3.0,
      1.4,  .1,
      1.0,  1.0,
  });
  m.Invoke();
  std::vector<float> out = {
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.9, 0.0, 0.1,
      0.0, 0.0, 0.0, 1.0,
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
