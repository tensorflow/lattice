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
// tf-lite op corresponding to PwlIndexingCalibrator op defined by tf-lattice

#include "flatbuffers/flexbuffers.h"
#include "tensorflow_lattice/cc/tflite_ops/tflite_ops.h"
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace pwl_indexing_calibrator {

// Typically the two nearest keypoints are returned for interpolation.
// If the input coincides with a keypoint, then only that one is returned.
// If the input is outside the keypoint range, then only the nearest keypoint
// is returned.
constexpr int kMaxNumInterpolationPoints = 2;

template <typename Dtype>
struct InterpolationPoints {
  int num_points;
  int64_t lower_index;
  Dtype weights[kMaxNumInterpolationPoints];
};



// Gets the index of an element in a flat representation given its row and col
inline int Get2DIndex(int n_cols, int row, int col) {
  return n_cols * row + col;
}

// Find the interpolation points, but _not the weights_, for the given
// uncalibrated value and keypoints inputs (kp_inputs).
// The interpolation will be between kp_inputs[lower_index] and
// kp_inputs[lower_index + 1]. Except outside the edges or if x (uncalibrated)
// is exactly on top of a keypoint, in which case the function returns 1 point.
// It uses a simple binary-search, so it is O(log(|kp_inputs|)).
template <typename Dtype>
InterpolationPoints<Dtype> FindInterpolationPoints(const Dtype uncalibrated,
                                                   const float* kp_inputs,
                                                   int num_kp) {
  if (uncalibrated <= kp_inputs[0]) {
    return InterpolationPoints<Dtype>{1, 0};
  }
  if (uncalibrated >= kp_inputs[num_kp - 1]) {
    return InterpolationPoints<Dtype>{1, num_kp - 1};
  }

  // Binary search the keypoints inputs.
  int64_t min_idx = 0, max_idx = num_kp;
  while (max_idx > min_idx + 1) {
    const int64_t idx = (max_idx + min_idx) / 2;
    const float value = kp_inputs[idx];
    if (uncalibrated == value) {
      return InterpolationPoints<Dtype>{1, idx};
    }
    if (uncalibrated > value) {
      min_idx = idx;
    } else {
      max_idx = idx;
    }
  }

  // Two points, where lower_index is min_idx.
  return InterpolationPoints<Dtype>{2, min_idx};
}

// Find interpolations points and associated weights for the given
// uncalibrated value and keypoints inputs (kp_inputs).
// Returns 1 interpolation point if uncalibrated is exactly on top of an
// input keypoint (or if beyond the edges), or 2 if in between two
// keypoints.
// See FindInterpolationPoints.
template <typename Dtype>
InterpolationPoints<Dtype> FindInterpolationPointsWithWeights(
    const Dtype uncalibrated, const float* kp_inputs, int num_kp) {
  // Get points an calculates weights.
  InterpolationPoints<Dtype> interpolation_points =
      FindInterpolationPoints<Dtype>(uncalibrated, kp_inputs, num_kp);
  if (interpolation_points.num_points == 1) {
    // All weight goes to the exact one keypoint where the uncalibrated value
    // lies.
    interpolation_points.weights[0] = 1.0;
    return interpolation_points;
  }
  const Dtype delta = kp_inputs[interpolation_points.lower_index + 1] -
                      kp_inputs[interpolation_points.lower_index];
  interpolation_points.weights[1] =
      (uncalibrated - kp_inputs[interpolation_points.lower_index]) / delta;
  interpolation_points.weights[0] = 1.0 - interpolation_points.weights[1];
  return interpolation_points;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* output = GetOutput(context, node, 0);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 1);
  const TfLiteTensor* kp_inputs = GetInput(context, node, 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(kp_inputs), 1);
  // output tensor shape is number of input rows x number of vertices
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(2);
  output_size->data[0] = SizeOfDimension(input, 0);
  output_size->data[1] = SizeOfDimension(kp_inputs, 0);
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_size));

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  const float* input_flat = GetTensorData<float>(input);

  const TfLiteTensor* kp_inputs = GetInput(context, node, 1);
  const float* kp_inputs_flat = GetTensorData<float>(kp_inputs);

  TfLiteTensor* output = GetOutput(context, node, 0);
  float* output_flat = GetTensorData<float>(output);

  for (int row = 0; row < SizeOfDimension(input, 0); ++row) {
    const float uncalibrated = input_flat[row];
    InterpolationPoints<float> pts = FindInterpolationPointsWithWeights(
        uncalibrated, kp_inputs_flat, SizeOfDimension(kp_inputs, 0));
    float* output_row = output_flat + row * SizeOfDimension(kp_inputs, 0);
    for (int i = 0; i < SizeOfDimension(kp_inputs, 0); ++i) {
      output_row[i] = 0.0;
    }
    for (int k = 0; k < pts.num_points; ++k) {
      output_row[pts.lower_index + k] = pts.weights[k];
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Prepare_Sparse(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* indices_output = GetOutput(context, node, 0);
  TfLiteTensor* weights_output = GetOutput(context, node, 1);
  SetTensorToDynamic(indices_output);
  SetTensorToDynamic(weights_output);
  weights_output->type = kTfLiteFloat32;
  indices_output->type = kTfLiteInt32;
  return kTfLiteOk;
}

TfLiteStatus Eval_Sparse(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  const float* input_flat = GetTensorData<float>(input);

  const TfLiteTensor* kp_inputs = GetInput(context, node, 1);
  const float* kp_inputs_flat = GetTensorData<float>(kp_inputs);

  TfLiteTensor* indices_output = GetOutput(context, node, 0);
  TfLiteIntArray* indices_output_size = TfLiteIntArrayCreate(2);
  indices_output_size->data[0] =
      kMaxNumInterpolationPoints * SizeOfDimension(input, 0);
  indices_output_size->data[1] = 2;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, indices_output,
                                                   indices_output_size));
  int* indices_output_flat = GetTensorData<int>(indices_output);

  TfLiteTensor* weights_output = GetOutput(context, node, 1);
  TfLiteIntArray* weights_output_size = TfLiteIntArrayCreate(1);
  weights_output_size->data[0] =
      kMaxNumInterpolationPoints * SizeOfDimension(input, 0);
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, weights_output,
                                                   weights_output_size));
  float* weights_output_flat = GetTensorData<float>(weights_output);

  int current_output_row = 0;
  for (int row = 0; row < SizeOfDimension(input, 0); ++row) {
    const float uncalibrated = input_flat[row];
    InterpolationPoints<float> pts = FindInterpolationPointsWithWeights(
        uncalibrated, kp_inputs_flat, SizeOfDimension(kp_inputs, 0));
    for (int i = 0; i < pts.num_points; ++i) {
      weights_output_flat[current_output_row] = pts.weights[i];
      indices_output_flat[Get2DIndex(2, current_output_row, 0)] = row;
      indices_output_flat[Get2DIndex(2, current_output_row, 1)] =
          pts.lower_index + i;
      ++current_output_row;
    }
  }

  TfLiteIntArray* indices_output_size_ = TfLiteIntArrayCreate(2);
  indices_output_size_->data[0] = current_output_row;
  indices_output_size_->data[1] = 2;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, indices_output,
                                                   indices_output_size_));

  TfLiteIntArray* weights_output_size_ = TfLiteIntArrayCreate(1);
  weights_output_size_->data[0] = current_output_row;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, weights_output,
                                                   weights_output_size_));

  return kTfLiteOk;
}

}  // namespace pwl_indexing_calibrator

TfLiteRegistration* Register_PWL_INDEXING_CALIBRATOR() {
  static TfLiteRegistration r = {nullptr, nullptr,
                                 pwl_indexing_calibrator::Prepare,
                                 pwl_indexing_calibrator::Eval};
  return &r;
}

TfLiteRegistration* Register_PWL_INDEXING_CALIBRATOR_SPARSE() {
  static TfLiteRegistration r = {nullptr, nullptr,
                                 pwl_indexing_calibrator::Prepare_Sparse,
                                 pwl_indexing_calibrator::Eval_Sparse};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
