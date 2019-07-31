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
#include "tensorflow_lattice/cc/tflite_ops/tflite_ops.h"
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/op_resolver.h"

namespace tflite {

void RegisterTfLatticeOps(MutableOpResolver* resolver) {
  resolver->AddCustom("HypercubeInterpolation",
            tflite::ops::custom::Register_HYPERCUBE_INTERPOLATION());
  resolver->AddCustom("SimplexInterpolation",
            tflite::ops::custom::Register_SIMPLEX_INTERPOLATION());
  resolver->AddCustom("PWLIndexingCalibration",
                      tflite::ops::custom::Register_PWL_INDEXING_CALIBRATOR());
  resolver->AddCustom(
      "PWLIndexingCalibrationSparse",
      tflite::ops::custom::Register_PWL_INDEXING_CALIBRATOR_SPARSE());
}

}  // namespace tflite
