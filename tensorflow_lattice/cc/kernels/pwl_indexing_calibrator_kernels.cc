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
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace lattice {

namespace {

// Maximum number of points used by interpolation. It may use up to 3 when it's
// exactly on top of a keypoint input -- it returns also the left and right
// keypoints inputs (indices). See explanation below on
// FindExpandedInterpolation.
constexpr int kMaxNumInterpolationPoints = 3;

// Changed with PwlSetDebugMode function. This variable forces each row of a
// batch to be processed by a separate worker, only used for testing.
bool test_force_split = false;

}  // namespace

extern void PwlSetTestMode(bool split_batches);
void PwlSetTestMode(const bool split_batches) {
  test_force_split = split_batches;
}

// Helper struct that holds all information needed to resolve one interpolation:
// the number of consecutive points used (num_points), the index of the first
// one (lower_index) the associated weights -- not used in every case.
template <typename Dtype>
struct InterpolationPoints {
  int num_points;
  int64_t lower_index;
  Dtype weights[kMaxNumInterpolationPoints];
};

namespace {

// Find the interpolation points, but _not the weights_, for the given
// uncalibrated value and keypoints inputs (kp_inputs).
// The interpolation will be between kp_inputs[lower_index] and
// kp_inputs[lower_index + 1]. Except outside the edges or if x (uncalibrated)
// is exactly on top of a keypoint, in which case the function returns 1 point.
// It uses a simple binary-search, so it is O(log(|kp_inputs|)).
template <typename Dtype>
InterpolationPoints<Dtype> FindInterpolationPoints(
    const Dtype uncalibrated,
    const typename TTypes<const Dtype>::Vec& kp_inputs) {
  if (uncalibrated <= kp_inputs(0)) {
    return InterpolationPoints<Dtype>{1, 0};
  }
  const int64_t kp_inputs_last_idx = static_cast<int64_t>(kp_inputs.size() - 1);
  if (uncalibrated >= kp_inputs(kp_inputs_last_idx)) {
    return InterpolationPoints<Dtype>{1, kp_inputs_last_idx};
  }

  // Binary search the keypoints inputs.
  int64_t min_idx = 0, max_idx = kp_inputs.size();
  while (max_idx > min_idx + 1) {
    const int64_t idx = (max_idx + min_idx) / 2;
    const Dtype value = kp_inputs(idx);
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
    const Dtype uncalibrated,
    const typename TTypes<const Dtype>::Vec& kp_inputs) {
  // Get points an calculates weights.
  InterpolationPoints<Dtype> interpolation_points =
      FindInterpolationPoints<Dtype>(uncalibrated, kp_inputs);
  if (interpolation_points.num_points == 1) {
    // All weight goes to the exact one keypoint where the uncalibrated value
    // lies.
    interpolation_points.weights[0] = 1.0;
    return interpolation_points;
  }

  // assert(interpolation_points.num_points == 2)
  // The piecewise linear interpolation weights (w) when x (uncalibrated)
  // is in between two keypoints, is given by:
  //
  //   w[lower_index] = 1.0 - theta(x, lower_index)
  //   w[lower_index + 1] = theta(x, lower_index)
  // Where:
  //   theta(x, lower_index) = (x - keypoint_inputs[lower_index]) /
  //                           delta(lower_index)
  //   delta(lower_index) = kp_inputs[lower_index+1] - kp_inputs[lower_index]
  //
  // Note: the calibration later will apply the weights to the keypoints
  // outputs, in the following format:
  //
  //   calibrated(x) = sum ( w(x) .* kp_outputs )
  //
  // So in this particular case, down the pipe, we'll have:
  //
  //   calibrated(x) = w[lower_index] * kp_outputs[lower_index] +
  //       w[lower_index + 1] * kp_outputs[lower_index + 1]
  //
  // And since w(x) is a linear in x, calibrated(x) will be linear in x as well.
  const Dtype delta = kp_inputs(interpolation_points.lower_index + 1) -
                      kp_inputs(interpolation_points.lower_index);
  interpolation_points.weights[1] =
      (uncalibrated - kp_inputs(interpolation_points.lower_index)) / delta;
  interpolation_points.weights[0] = 1.0 - interpolation_points.weights[1];
  return interpolation_points;
}

template <typename Dtype>
void IndexingCalibratorWorker(
    const typename TTypes<const Dtype>::Vec& kp_inputs,
    const typename TTypes<const Dtype>::Vec& uncalibrated_flat, const int start,
    const int limit, typename TTypes<Dtype, 2>::Tensor interpolation) {
  // Loop over input weights.
  for (int i = start; i < limit; i++) {
    // Find interpolation lower_index and weights (weights).
    const InterpolationPoints<Dtype> interpolation_points =
        FindInterpolationPointsWithWeights<Dtype>(uncalibrated_flat(i),
                                                  kp_inputs);

    // Copy interpolation weights.
    for (int j = 0; j < interpolation_points.num_points; j++) {
      interpolation(i, interpolation_points.lower_index + j) =
          interpolation_points.weights[j];
    }
  }
}

}  // namespace

template <typename Dtype>
class PwlIndexingCalibratorOpKernel : public OpKernel {
 public:
  explicit PwlIndexingCalibratorOpKernel(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab keypoints inputs: it provides the num_keypoints.
    const Tensor& kp_inputs_tensor = context->input(1);
    OP_REQUIRES(context, kp_inputs_tensor.dims() == 1,
                errors::InvalidArgument(
                    "keypoints must have dims=1, got kp_inputs.dims=",
                    kp_inputs_tensor.dims()));
    auto kp_inputs = kp_inputs_tensor.vec<Dtype>();
    const int num_keypoints = kp_inputs.size();

    // Uncalibrated value(s): it provides the batch_size.
    const Tensor& uncalibrated_tensor = context->input(0);
    OP_REQUIRES(
        context, uncalibrated_tensor.dims() == 1,
        errors::InvalidArgument("input must have dims=1, got input.dims=",
                                uncalibrated_tensor.dims()));
    const auto& uncalibrated_flat = uncalibrated_tensor.vec<Dtype>();
    const int64 batch_size = uncalibrated_flat.size();

    // Output tensor.
    Tensor* interpolation_tensor = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({uncalibrated_flat.size(), num_keypoints}),
            &interpolation_tensor));
    auto interpolation_tensor_flat =
        interpolation_tensor->flat_inner_dims<Dtype, 2>();
    interpolation_tensor_flat.setZero();

    if (test_force_split) {
      // Debug mode: do one example at a time.
      for (int ii = 0; ii < batch_size; ii++) {
        IndexingCalibratorWorker<Dtype>(kp_inputs, uncalibrated_flat, ii,
                                        ii + 1, interpolation_tensor_flat);
      }
    } else {
      // Sharded (multi-threaded) calculation:
      auto worker_threads =
          *(context->device()->tensorflow_cpu_worker_threads());
      // Cost is O(N) because of having to zero out the weights.

      constexpr int64 kBaseCost = 20;
      constexpr int64 kCostPerKeypoint = 20;
      const int64 cost_per_unit = kBaseCost + num_keypoints * kCostPerKeypoint;
      Shard(worker_threads.num_threads, worker_threads.workers, batch_size,
            cost_per_unit, [&kp_inputs, &uncalibrated_flat,
                            &interpolation_tensor_flat](int start, int limit) {
              IndexingCalibratorWorker<Dtype>(kp_inputs, uncalibrated_flat,
                                              start, limit,
                                              interpolation_tensor_flat);
            });
    }
  }

  TF_DISALLOW_COPY_AND_ASSIGN(PwlIndexingCalibratorOpKernel);
};

namespace {

// This worker computes the partial derivative w.r.t. input (uncalibrated).
//
// To simplify, let's call x the input uncalibrated value of one example
// (as opposed to the full batch). Let's call w(x) the vector of interpolation
// weights, returned by PwlIndexingCalibratorOp.
//
// The optimizer needs to find the gradient of loss(x), which can be written
// as loss(w(x)).
//
//     d(loss)/dx = d(loss)/d(w) * d(w)/dx
//
// grad_wrt_weights is the d(loss)/d(w) and is given. So this function needs to
// calculate d(w)/dx in order to return d(loss)/dx.
//
// For the common case, x is in between two keypoints (kp_inputs). Recall from
// comment in FindInterpolation above that:
//
//     w[lower_index] = 1.0 - theta
//     w[lower_index + 1] = theta
// Where:
//     theta = (x - keypoint_inputs[lower_index]) / delta[lower_index]
//     delta[lower_index] = kp_inputs[lower_index + 1] - kp_inputs[lower_index]
//
// For d(w)/dx we have:
//     d(w[i])/dx = 0, for i != lower_index and i != lower_index +1
//
// And for i = lower_index and lower_index+1 (notice that kp_inputs are
// constants):
//
//     d(w[lower_index])/dx = - 1/delta[lower_index]
//     d(w[lower_index+1)])/dx = 1/delta[lower_index]
//
// Since d(loss)/dx = d(loss)/d(w) * d(w)/dx, d(loss)/d(w) = grad_wrt_weights,
// we have:
//
//     d(loss)/dx = (grad_wrt_weights[lower_index+1] -
//                   grad_wrt_weights[lower_index]) / delta[lower_index]

template <typename Dtype>
void IndexingCalibratorInputGradientWorker(
    const typename TTypes<const Dtype>::Vec& kp_inputs,
    const typename TTypes<const Dtype>::Vec& uncalibrated_vec,
    const typename TTypes<const Dtype>::Matrix& grad_wrt_weights_mat,
    const int start, const int limit,
    typename TTypes<Dtype>::Vec* grad_wrt_input_vec) {
  const int num_keypoints = kp_inputs.size();

  // Loop over examples (batch_index) of the batch.
  for (int batch_index = start; batch_index < limit; batch_index++) {
    // Simpler non-batch (single value) version:
    const Dtype uncalibrated = uncalibrated_vec(batch_index);

    // Find interpolation lower_index and weights (weights).
    const InterpolationPoints<Dtype> interpolation_points =
        FindInterpolationPoints<Dtype>(uncalibrated, kp_inputs);

    // Input grad has to be multiplied by the output grad.
    if (interpolation_points.num_points == 2) {
      // Input is in between 2 keypoints.
      const Dtype delta = kp_inputs(interpolation_points.lower_index + 1) -
                          kp_inputs(interpolation_points.lower_index);
      (*grad_wrt_input_vec)(batch_index) =
          (grad_wrt_weights_mat(batch_index,
                                interpolation_points.lower_index + 1) -
           grad_wrt_weights_mat(batch_index,
                                interpolation_points.lower_index)) /
          delta;

    } else {  // assert(interpolation_points.num_points == 1)
      // Input is exactly on top of a keypoint. d(w)/dx is not defined in this
      // case, and what we do is to average the d(w)/dx that comes to the right
      // of it and the d(w)/dx to the left of it.
      //
      // To the right of lower_index we have, from above:
      //
      // d(loss)/dx = (grad_wrt_weights[lower_index+1] -
      //     grad_wrt_weights[lower_index]) / delta[lower_index]
      //
      // And from the left:
      //
      // d(loss)/dx = (grad_wrt_weights[lower_index] -
      //     grad_wrt_weights[lower_index-1]) / delta[lower_index - 1]
      //
      // And we take a sub-gradient (or super-gradient), by averaging of those
      // two gradients, except if the keypoint is in one of the edges (start
      // or end of the kp_inputs), in which case we just get the d(w)/dx from
      // the side we have.
      Dtype grad = 0.0;  // == d(loss)/dx
      int count = 0;
      if (interpolation_points.lower_index > 0) {
        const Dtype delta = kp_inputs(interpolation_points.lower_index) -
                            kp_inputs(interpolation_points.lower_index - 1);
        grad = (grad_wrt_weights_mat(batch_index,
                                     interpolation_points.lower_index) -
                grad_wrt_weights_mat(batch_index,
                                     interpolation_points.lower_index - 1)) /
               delta;
        ++count;
      }
      if (interpolation_points.lower_index < num_keypoints - 1) {
        const Dtype delta = kp_inputs(interpolation_points.lower_index + 1) -
                            kp_inputs(interpolation_points.lower_index);
        grad += (grad_wrt_weights_mat(batch_index,
                                      interpolation_points.lower_index + 1) -
                 grad_wrt_weights_mat(batch_index,
                                      interpolation_points.lower_index)) /
                delta;
        ++count;
      }
      if (count > 0) grad /= count;  // Take mean.
      (*grad_wrt_input_vec)(batch_index) = grad;
    }
  }
}
}  // namespace

template <typename Dtype>
class PwlIndexingCalibratorGradientOpKernel : public OpKernel {
 public:
  explicit PwlIndexingCalibratorGradientOpKernel(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab keypoints inputs: it provides the num_keypoints.
    const Tensor& kp_inputs_tensor = context->input(1);
    OP_REQUIRES(context, kp_inputs_tensor.dims() == 1,
                errors::InvalidArgument(
                    "keypoints must have dims=1, got kp_inputs.dims=",
                    kp_inputs_tensor.dims()));
    auto kp_inputs = kp_inputs_tensor.vec<Dtype>();
    const int num_keypoints = kp_inputs.size();

    // Uncalibrated value(s): it provides the batch_size.
    const Tensor& uncalibrated_tensor = context->input(0);
    OP_REQUIRES(
        context, uncalibrated_tensor.dims() == 1,
        errors::InvalidArgument("input must have dims=1, got input.dims=",
                                uncalibrated_tensor.dims()));
    const auto& uncalibrated_vec = uncalibrated_tensor.vec<Dtype>();
    const int64 batch_size = uncalibrated_vec.size();

    // Gradient with respect to outputs, needed for chain rule.
    const Tensor& grad_wrt_weights_tensor = context->input(2);
    OP_REQUIRES(
        context, grad_wrt_weights_tensor.dims() == 2,
        errors::InvalidArgument("grad_wrt_weights_tensor must have dims=2, "
                                "got grad_wrt_weights_tensor.dims=",
                                grad_wrt_weights_tensor.dims()));
    OP_REQUIRES(
        context, grad_wrt_weights_tensor.dim_size(0) == batch_size,
        errors::InvalidArgument(
            "grad_wrt_weights_tensor (output gradient) has shape [batch_size=",
            grad_wrt_weights_tensor.dim_size(0),
            ", num_keypoints], expected batch_size=", batch_size, " instead"));
    OP_REQUIRES(
        context, grad_wrt_weights_tensor.dim_size(1) == num_keypoints,
        errors::InvalidArgument(
            "grad_wrt_weights_tensor (output gradient) has shape [batch_size, "
            "num_keypoints=",
            grad_wrt_weights_tensor.dim_size(1), "], expected num_keypoints=",
            num_keypoints, " instead"));
    const auto grad_wrt_weights_mat = grad_wrt_weights_tensor.matrix<Dtype>();

    // Keypoints' inputs are fixed, so their gradient are always zero.
    // The kp_inputs is of fixed size ([num_keypoints]) independent of the
    // batch size. So the gradient wrt kp_inputs is summed over all batch,
    // as opposed to the gradient wrt to the input.
    Tensor* grad_wrt_kp_inputs = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape({num_keypoints}),
                                            &grad_wrt_kp_inputs));
    grad_wrt_kp_inputs->vec<Dtype>().setZero();

    // Gradient with respect to input:
    Tensor* grad_wrt_input_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({batch_size}),
                                            &grad_wrt_input_tensor));
    auto grad_wrt_input_vec = grad_wrt_input_tensor->vec<Dtype>();

    // Sharded (multi-threaded) calculation:
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    // Cost is O(N) because of having to zero out the weights.

    constexpr int64 kBaseCost = 20;
    constexpr int64 kCostPerKeypoint = 20;
    const int64 cost_per_unit = kBaseCost + num_keypoints * kCostPerKeypoint;
    Shard(worker_threads.num_threads, worker_threads.workers, batch_size,
          cost_per_unit, [&kp_inputs, &uncalibrated_vec, &grad_wrt_weights_mat,
                          &grad_wrt_input_vec](int start, int limit) {
            IndexingCalibratorInputGradientWorker<Dtype>(
                kp_inputs, uncalibrated_vec, grad_wrt_weights_mat, start, limit,
                &grad_wrt_input_vec);
          });
  }

  TF_DISALLOW_COPY_AND_ASSIGN(PwlIndexingCalibratorGradientOpKernel);
};

//////////////////////////////////////////////////////////////////////////////
// Sparse implementation
//////////////////////////////////////////////////////////////////////////////

namespace {

// Calculates expanded interpolation points: Similar to
// FindInterpolationPointsWithWeights above, but expand interpolation
// on exact keypoints to the ones around it. So it returns either 2 or
// 3 keypoints.
// The expansion is helpful in the sparse implementation because it makes the
// optimizing feature provide the d(loss)/d(w) for those points around, even
// when their weights are zero: see description of
// IndexingCalibratorInputGradientWorker above.
// Returns a InterpolationPoints struct with 2 or 3 points with the
// weights properly set.
template <typename Dtype>
InterpolationPoints<Dtype> FindExpandedInterpolationPointsWithWeights(
    const Dtype uncalibrated,
    const typename TTypes<const Dtype>::Vec& kp_inputs) {
  // Find interpolation points without expansion.
  InterpolationPoints<Dtype> interpolation_points =
      FindInterpolationPointsWithWeights<Dtype>(uncalibrated, kp_inputs);

  // Nothing changes for interpolation between keypoints.
  if (interpolation_points.num_points == 2 || kp_inputs.size() == 1) {
    return interpolation_points;
  }

  // assert(interpolation_points.num_points == 1)
  // Add second keypoint if on the first keypoint.
  if (interpolation_points.lower_index == 0) {
    interpolation_points.num_points = 2;
    interpolation_points.weights[0] = 1;
    interpolation_points.weights[1] = 0;
    return interpolation_points;
  }

  // Add second keypoint if on the last keypoint.
  const auto kp_last = kp_inputs.size() - 1;
  if (interpolation_points.lower_index == kp_last) {
    interpolation_points.num_points = 2;
    interpolation_points.lower_index--;
    interpolation_points.weights[0] = 0;
    interpolation_points.weights[1] = 1;
    return interpolation_points;
  }

  // Add keypoints on the sides when exactly on a middle keypoint.
  interpolation_points.num_points = 3;
  interpolation_points.lower_index--;
  interpolation_points.weights[0] = 0;
  interpolation_points.weights[1] = 1;
  interpolation_points.weights[2] = 0;
  return interpolation_points;
}

// Calculates the gradient w.r.t the input, for the given interpolation points.
// This is a simple adaptation of IndexingCalibratorInputGradientWorker for
// sparse tensors. Please see the documentation in that function for the math
// details.
template <typename Dtype>
Dtype GradWRTInputSparse(
    const int num_interpolation_points,
    const typename TTypes<const Dtype>::Vec& kp_inputs, const int64 lower_index,
    const typename TTypes<const Dtype>::Vec& grad_wrt_weights,
    const int64 weights_base_idx) {
  Dtype grad;
  if (num_interpolation_points == 2) {
    // Input is in between 2 keypoints.
    const Dtype delta = kp_inputs(lower_index + 1) - kp_inputs(lower_index);
    grad = (grad_wrt_weights(weights_base_idx + 1) -
            grad_wrt_weights(weights_base_idx + 0)) /
           delta;

  } else {
    // assert(num_interpolation_points == 3)
    // Input is exactly on top of a keypoint: average the slope of the
    // previous and next keypoints: it's not correct since it is a point
    // of discontinuity, but allows the weights to move.
    const Dtype delta1 = kp_inputs(lower_index + 1) - kp_inputs(lower_index);
    grad = (grad_wrt_weights(weights_base_idx + 1) -
            grad_wrt_weights(weights_base_idx)) /
           delta1;

    const Dtype delta2 =
        kp_inputs(lower_index + 2) - kp_inputs(lower_index + 1);
    grad += (grad_wrt_weights(weights_base_idx + 2) -
             grad_wrt_weights(weights_base_idx + 1)) /
            delta2;

    // Divided by 2 to get the mean of the gradients.
    grad /= 2.0;
  }
  return grad;
}

}  // namespace

template <typename Dtype>
class PwlIndexingCalibratorSparseOpKernel : public OpKernel {
 public:
  explicit PwlIndexingCalibratorSparseOpKernel(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab keypoints inputs.
    const Tensor& kp_inputs_tensor = context->input(1);
    OP_REQUIRES(context, kp_inputs_tensor.dims() == 1,
                errors::InvalidArgument(
                    "keypoints must have dims=1, got kp_inputs.dims=",
                    kp_inputs_tensor.dims()));
    auto kp_inputs = kp_inputs_tensor.vec<Dtype>();

    // Uncalibrated value(s): it provides the batch_size.
    const Tensor& uncalibrated_tensor = context->input(0);
    OP_REQUIRES(
        context, uncalibrated_tensor.dims() == 1,
        errors::InvalidArgument("input must have dims=1, got input.dims=",
                                uncalibrated_tensor.dims()));
    const auto& uncalibrated_vec = uncalibrated_tensor.vec<Dtype>();
    const int64 batch_size = uncalibrated_vec.size();

    // Find interpolation points and weights for each uncalibrated
    // value.
    std::vector<int64> batch_lower_weight_indices(batch_size);
    std::vector<Dtype> batch_weights(batch_size * kMaxNumInterpolationPoints);
    std::vector<int> batch_num_interpolation_points(batch_size);
    int64 total_interpolation_points = 0;
    for (int i = 0; i < batch_size; i++) {
      const InterpolationPoints<Dtype> interpolation_points =
          FindExpandedInterpolationPointsWithWeights<Dtype>(uncalibrated_vec(i),
                                                            kp_inputs);
      for (int j = 0; j < interpolation_points.num_points; j++) {
        batch_weights[total_interpolation_points + j] =
            interpolation_points.weights[j];
      }
      batch_num_interpolation_points[i] = interpolation_points.num_points;
      batch_lower_weight_indices[i] = interpolation_points.lower_index;
      total_interpolation_points += interpolation_points.num_points;
    }

    // Copy interpolation weights into sparse tensor components: indices,
    // weights.
    // Build indices tensor: each index is a vector of 2 numbers: batch_index
    // and the weight index.
    Tensor* tensor_indices = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({total_interpolation_points, 2}), &tensor_indices));
    auto tensor_indices_mat = tensor_indices->matrix<int64>();
    int64 sparse_index = 0;
    for (int batch_index = 0; batch_index < batch_size; batch_index++) {
      for (int col = 0; col < batch_num_interpolation_points[batch_index];
           col++) {
        tensor_indices_mat(sparse_index, 0) = batch_index;
        tensor_indices_mat(sparse_index, 1) =
            batch_lower_weight_indices[batch_index] + col;
        sparse_index++;
      }
    }

    // The weights in order of the sparse index is already calculated in
    // batch_weights so we just need to copy them.
    Tensor* tensor_weights = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(1, TensorShape({total_interpolation_points}),
                                 &tensor_weights));
    // Notice batch_weights has some overhead space, we can only copy
    // total_interpolation_points weights.
    std::copy(batch_weights.begin(),
              batch_weights.begin() + total_interpolation_points,
              tensor_weights->flat<Dtype>().data());
  }

  TF_DISALLOW_COPY_AND_ASSIGN(PwlIndexingCalibratorSparseOpKernel);
};

template <typename Dtype>
class PwlIndexingCalibratorSparseGradientOpKernel : public OpKernel {
 public:
  explicit PwlIndexingCalibratorSparseGradientOpKernel(
      OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab keypoints inputs: it provides the num_keypoints.
    const Tensor& kp_inputs_tensor = context->input(1);
    OP_REQUIRES(context, kp_inputs_tensor.dims() == 1,
                errors::InvalidArgument(
                    "keypoints must have dims=1, got kp_inputs.dims=",
                    kp_inputs_tensor.dims()));
    auto kp_inputs = kp_inputs_tensor.vec<Dtype>();
    const int num_keypoints = kp_inputs.size();

    // Uncalibrated value(s): it provides the batch_size.
    const Tensor& uncalibrated_tensor = context->input(0);
    OP_REQUIRES(
        context, uncalibrated_tensor.dims() == 1,
        errors::InvalidArgument("input must have dims=1, got input.dims=",
                                uncalibrated_tensor.dims()));
    const auto& uncalibrated_vec = uncalibrated_tensor.vec<Dtype>();
    const int64 batch_size = uncalibrated_vec.size();

    // Interpolation indices returned by PwlIndexingCalibratorSparse op.
    // It will be a matrix where each row represent an interpolation point
    // given by (batch_index, weight_index), 0 <= batch_index < batch_size,
    // 0 <= weight_index < kp_inputs.size().
    const Tensor& interpolation_indices_tensor = context->input(2);
    OP_REQUIRES(context, interpolation_indices_tensor.dims() == 2,
                errors::InvalidArgument(
                    "interpolation_indicesmust have dims=2, got input.dims=",
                    uncalibrated_tensor.dims()));
    const auto interpolation_indices =
        interpolation_indices_tensor.matrix<int64>();
    const int64 total_interpolation_points =
        interpolation_indices_tensor.dim_size(0);

    // Gradient with respect to outputs, needed for chain rule. One value
    // per sparse index in interpolation_indices.
    const Tensor& grad_wrt_weights_tensor = context->input(3);
    const auto grad_wrt_weights = grad_wrt_weights_tensor.vec<Dtype>();
    OP_REQUIRES(
        context, grad_wrt_weights.size() == total_interpolation_points,
        errors::InvalidArgument("grad_wrt_weights (", grad_wrt_weights.size(),
                                " elements) must have as many elements as the "
                                "total number of interpolation indices (",
                                total_interpolation_points, " elements)"));

    // Keypoints' inputs are fixed, so their gradient are always zero. Fixed
    // size, invariant to the size of the batch.
    Tensor* grad_wrt_kp_inputs = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape({num_keypoints}),
                                            &grad_wrt_kp_inputs));
    grad_wrt_kp_inputs->vec<Dtype>().setZero();

    // Gradient with respect to inputs is dense and of the same dimension as
    // the input, that is batch_size.
    Tensor* grad_wrt_input_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({batch_size}),
                                            &grad_wrt_input_tensor));
    auto grad_wrt_input = grad_wrt_input_tensor->vec<Dtype>();

    // Each interpolation point is associated to one weigh in grad_wrt_weights
    // and a pair of indices (batch index, lower_index) in
    // interpolation_indices.
    int64 sparse_index = 0;  // Loops over all interpolation points.
    while (sparse_index < total_interpolation_points) {
      const int batch_index = interpolation_indices(sparse_index, 0);
      OP_REQUIRES(context, batch_index >= 0 && batch_index < batch_size,
                  errors::InvalidArgument(
                      "invalid batch_index index for sparse "
                      "interpolation, expected 0 <= batch_index <= ",
                      batch_size, " got ", batch_index));
      const int64 weights_base_idx = sparse_index;
      int64 lower_index = interpolation_indices(sparse_index, 1);
      int num_interpolation_points = 0;
      do {
        num_interpolation_points++;
        sparse_index++;
      } while (sparse_index < total_interpolation_points &&
               interpolation_indices(sparse_index, 0) == batch_index &&
               num_interpolation_points < kMaxNumInterpolationPoints + 1);
      OP_REQUIRES(
          context,
          num_interpolation_points == 2 || num_interpolation_points == 3,
          errors::InvalidArgument(
              "only interpolations with 2 or 3 points are supported, got ",
              num_interpolation_points));
      grad_wrt_input(batch_index) = GradWRTInputSparse<Dtype>(
          num_interpolation_points, kp_inputs, lower_index, grad_wrt_weights,
          weights_base_idx);
    }
  }

  TF_DISALLOW_COPY_AND_ASSIGN(PwlIndexingCalibratorSparseGradientOpKernel);
};

//////////////////////////////////////////////////////////////////////////////
// Kernels registration for all operation defined here.
//////////////////////////////////////////////////////////////////////////////
REGISTER_KERNEL_BUILDER(Name("PwlIndexingCalibrator")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("Dtype"),
                        PwlIndexingCalibratorOpKernel<float>);
REGISTER_KERNEL_BUILDER(Name("PwlIndexingCalibrator")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("Dtype"),
                        PwlIndexingCalibratorOpKernel<double>);

REGISTER_KERNEL_BUILDER(Name("PwlIndexingCalibratorGradient")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("Dtype"),
                        PwlIndexingCalibratorGradientOpKernel<float>);
REGISTER_KERNEL_BUILDER(Name("PwlIndexingCalibratorGradient")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("Dtype"),
                        PwlIndexingCalibratorGradientOpKernel<double>);

REGISTER_KERNEL_BUILDER(Name("PwlIndexingCalibratorSparse")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("Dtype"),
                        PwlIndexingCalibratorSparseOpKernel<float>);
REGISTER_KERNEL_BUILDER(Name("PwlIndexingCalibratorSparse")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("Dtype"),
                        PwlIndexingCalibratorSparseOpKernel<double>);

REGISTER_KERNEL_BUILDER(Name("PwlIndexingCalibratorSparseGradient")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("Dtype"),
                        PwlIndexingCalibratorSparseGradientOpKernel<float>);
REGISTER_KERNEL_BUILDER(Name("PwlIndexingCalibratorSparseGradient")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("Dtype"),
                        PwlIndexingCalibratorSparseGradientOpKernel<double>);

}  // namespace lattice
}  // namespace tensorflow
