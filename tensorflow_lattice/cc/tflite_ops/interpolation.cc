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
// tf-lite op corresponding to hypercube_interpolation op defined by tf-lattice

#include <numeric>
#include "flatbuffers/flexbuffers.h"
#include "tensorflow_lattice/cc/tflite_ops/helpers.h"
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace interpolation {

using tflite_lattice::BottomCornerIndexAndResidual;
using tflite_lattice::GetBottomCornerIndexAndResidual;
using tflite_lattice::InterpolationParams;


// See tensorflow_lattice/cc/kernels/hypercube_interpolation_kernels.cc
// for an in depth explanation of this routine and further references.
// Edge behavior is inherited from the tensorflow op, which specifies that
// points out of bounds are clipped to nearest cell boundary.
void ComputeInterpolationWeightsHyper(const std::vector<int>& lattice_sizes,
                                        int num_vertices_per_cell,
                                        const std::vector<int>& strides,
                                        const float* input_row,
                                        float* output_row) {
  int dimension = lattice_sizes.size();
  std::vector<int> indices(num_vertices_per_cell);
  std::vector<float> weights(num_vertices_per_cell);

  const BottomCornerIndexAndResidual<float> index_and_residual =
      GetBottomCornerIndexAndResidual<float>(lattice_sizes, input_row, strides);
  const std::vector<float>& residual =
      index_and_residual.residual;

  indices[0] = index_and_residual.bottom_corner_index;
  weights[0] = 1.0;
  int current_highest_dimension = 0;
  float current_residual_value = residual[current_highest_dimension];
  for (int i = 1; i < num_vertices_per_cell; ++i) {
    // Make sure that we're within the bounds of the unit hypercube.
    TFLITE_DCHECK_GE(current_residual_value, 0);
    TFLITE_DCHECK_LE(current_residual_value, 1);
    // Sanity check: current_highest_dimension has better respect the bounds.
    TFLITE_DCHECK_GE(current_highest_dimension, 0);
    TFLITE_DCHECK_LT(current_highest_dimension, dimension);
    const int earlier_i = i ^ (1 << current_highest_dimension);
    indices[i] = indices[earlier_i] + strides[current_highest_dimension];
    weights[i] = weights[earlier_i] * current_residual_value;
    weights[earlier_i] *= (1.0 - current_residual_value);

    if ((i & (i + 1)) == 0) {
      // If i + 1 is power of 2, then current_highest_dimension has changed,
      // that means, that we are processing next dimension.
      ++current_highest_dimension;
      if (dimension >= current_highest_dimension + 1) {
        current_residual_value = residual[current_highest_dimension];
      }
    }
  }
  // initialize output tensor to zeros
  // This is the number of vertices, which is the length of the output rows,
  // See Init for more context
  const int row_size = strides[dimension - 1] * lattice_sizes[dimension - 1];
  for (int i = 0; i < row_size; ++i) {
    output_row[i] = 0.0;
  }
  for (int jj = 0; jj < indices.size(); ++jj) {
    output_row[indices[jj]] = weights[jj];
  }
}

// Returns the permutation such that
// values[permutation[0]] >= ... >= values[permutation[d - 1]] where
// d == values.size().
std::vector<int> DescendingPermutation(const std::vector<float>& values) {
  std::vector<int> permutation(values.size());
  std::iota(permutation.begin(), permutation.end(), 0);

  auto cmp = [&values](const int left, const int right) -> bool {
    return values[left] > values[right];
  };
  std::sort(permutation.begin(), permutation.end(), cmp);
  return permutation;
}

// This function is adapted from ComputeInterpolationWeights in
// tensorflow_lattice/cc/kernels/simplex_interpolation_kernels.cc,
// see there for a detailed exposition.
// Produces simplex interpolation weights for an input that is in the unit
// hypercube (the residual), as well as the corresponding indices in the lattice
// (based on the bottom_corner). See http://jmlr.org/papers/v17/15-243.html for
// more details.
void ComputeInterpolationWeightsSimplex(const std::vector<int>& lattice_sizes,
                                        int num_vertices_per_cell,
                                        const std::vector<int>& strides,
                                        const float* input_row,
                                        float* output_row) {
  int dimension = lattice_sizes.size();

  const BottomCornerIndexAndResidual<float> bottom_corner_index_and_residual =
      GetBottomCornerIndexAndResidual<float>(lattice_sizes, input_row, strides);
  const std::vector<float>& residual =
      bottom_corner_index_and_residual.residual;

  const std::vector<int> descending_permutation =
      DescendingPermutation(residual);

  const int input_dim = dimension;
  // interpolation weight contains upto d + 1 non-zero elements.
  // Number of non-zero weights.
  const int max_nonzero = input_dim + 1;
  std::vector<int> indices(max_nonzero);
  std::vector<float> weights(max_nonzero);

  float current_residual = 1.0;
  int current_index = bottom_corner_index_and_residual.bottom_corner_index;
  for (int i = 0; i < input_dim; ++i) {
    const int current_dim = descending_permutation[i];
    const float next_residual = residual[current_dim];
    // Assigning index and weight.
    indices[i] = current_index;
    weights[i] = current_residual - next_residual;
    // Proceed to the next item.
    current_index += strides[current_dim];
    current_residual = next_residual;
  }
  // The boundary case.
  indices[input_dim] = current_index;
  weights[input_dim] = current_residual;

  // initialize output tensor to zeros
  // This is the number of vertices, which is the length of the output rows,
  // See Init for more context
  const int row_size = strides[dimension - 1] * lattice_sizes[dimension - 1];
  for (int i = 0; i < row_size; ++i) {
    output_row[i] = 0.0;
  }
  for (int j = 0; j < indices.size(); ++j) {
    output_row[indices[j]] = weights[j];
  }
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new InterpolationParams;
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  auto sizes = m["lattice_sizes"].AsTypedVector();
  data->dimension = sizes.size();
  for (int i = 0; i < data->dimension; ++i) {
    data->lattice_sizes.push_back(sizes[i].AsInt64());
  }
  data->strides.resize(data->dimension);
  data->num_vertices = 1;
  for (int i = 0; i < data->dimension; ++i) {
    data->strides[i] = data->num_vertices;
    data->num_vertices *= data->lattice_sizes[i];
  }
  data->num_vertices_per_cell = 1 << data->dimension;
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<InterpolationParams*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* output = GetOutput(context, node, 0);

  const auto* params =
      reinterpret_cast<InterpolationParams*>(node->user_data);
  const TfLiteTensor* input = GetInput(context, node, 0);
  // output tensor shape is number of input rows x number of vertices
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(2);
  output_size->data[0] = input->dims->data[0];
  output_size->data[1] = params->num_vertices;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_size));

  return kTfLiteOk;
}

using WeightCalculator = const std::function<void(
    const std::vector<int>&,
    int,
    const std::vector<int>&,
    const float*,
    float*
)>;

TfLiteStatus Eval(
    TfLiteContext* context, TfLiteNode* node,
    WeightCalculator compute_weights_f) {
  const auto* params =
      reinterpret_cast<InterpolationParams*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, 0);
  const float* input_flat = GetTensorData<float>(input);
  TfLiteTensor* output = GetOutput(context, node, 0);
  float* output_flat = GetTensorData<float>(output);

  for (int row_i = 0; row_i < input->dims->data[0]; ++row_i) {
    const float* input_row = input_flat + row_i * input->dims->data[1];
    float* output_row = output_flat + row_i * params->num_vertices;
    compute_weights_f(params->lattice_sizes,
                      params->num_vertices_per_cell, params->strides, input_row,
                      output_row);
  }
  return kTfLiteOk;
}

TfLiteStatus EvalHyper(TfLiteContext* context, TfLiteNode* node) {
  return Eval(context, node, ComputeInterpolationWeightsHyper);
}

TfLiteStatus EvalSimplex(TfLiteContext* context, TfLiteNode* node) {
  return Eval(context, node, ComputeInterpolationWeightsSimplex);
}

}  // namespace interpolation

TfLiteRegistration* Register_HYPERCUBE_INTERPOLATION() {
  static TfLiteRegistration r = {interpolation::Init, interpolation::Free,
                                 interpolation::Prepare,
                                 interpolation::EvalHyper};
  return &r;
}

TfLiteRegistration* Register_SIMPLEX_INTERPOLATION() {
  static TfLiteRegistration r = {interpolation::Init, interpolation::Free,
                                 interpolation::Prepare,
                                 interpolation::EvalSimplex};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
