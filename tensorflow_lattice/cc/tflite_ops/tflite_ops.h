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
#ifndef TENSORFLOW_LATTICE_CC_TFLITE_OPS_TFLITE_OPS_H_
#define TENSORFLOW_LATTICE_CC_TFLITE_OPS_TFLITE_OPS_H_
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/op_resolver.h"

// This file provides declarations and utilities useful for consumers of TF-Lite
// ops in TF-Lattice project.  In particular, there are headers for registration
// functions for each op, as well as a function that performs the registration.
namespace tflite {
namespace ops {
namespace custom {

TfLiteRegistration* Register_HYPERCUBE_INTERPOLATION();
TfLiteRegistration* Register_SIMPLEX_INTERPOLATION();
TfLiteRegistration* Register_PWL_INDEXING_CALIBRATOR();
TfLiteRegistration* Register_PWL_INDEXING_CALIBRATOR_SPARSE();

}  // namespace custom
}  // namespace ops

// Registers the custom ops so that tflite interpreter can find them.  Must be
// called by clients that intend to use these ops.
void RegisterTfLatticeOps(MutableOpResolver* resolver);

}  // namespace tflite

#endif  // TENSORFLOW_LATTICE_CC_TFLITE_OPS_TFLITE_OPS_H_
