# Copyright 2018 The TensorFlow Lattice Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# This package makes the custom ops developed for tensorflow_lattice available
# in the form of tf-lite ops.  For help with integration, reach out to
# tensorflow_lattice team
load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_cc_test",
)

package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0 License

exports_files(["LICENSE"])

cc_library(
    name = "tflite_ops",
    srcs = [
        "helpers.h",
        "interpolation.cc",
        "pwl_indexing_calibrator.cc",
    ],
    hdrs = [
        "tflite_ops.h",
    ],
    deps = [
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
        "@org_tensorflow//tensorflow/lite/kernels/internal:reference",
        "@flatbuffers",
    ],
)

cc_library(
    name = "tflite_ops_cc",
    srcs = ["tflite_ops.cc"],
    hdrs = ["tflite_ops.h"],
    deps = ["@org_tensorflow//tensorflow/lite:framework"],
)

tf_cc_test(
    name = "hypercube_interpolation_test",
    size = "small",
    srcs = [
        "hypercube_interpolation_test.cc",
    ],
    deps = [
        ":tflite_ops",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
        "@org_tensorflow//tensorflow/lite/kernels/internal:reference",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "simplex_interpolation_test",
    size = "small",
    srcs = [
        "simplex_interpolation_test.cc",
    ],
    deps = [
        ":tflite_ops",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
        "@org_tensorflow//tensorflow/lite/kernels/internal:reference",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "pwl_indexing_calibrator_test",
    size = "small",
    srcs = [
        "pwl_indexing_calibrator_test.cc",
    ],
    deps = [
        ":tflite_ops",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
        "@org_tensorflow//tensorflow/lite/kernels/internal:reference",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

tf_cc_test(
    name = "pwl_indexing_calibrator_sparse_test",
    size = "small",
    srcs = [
        "pwl_indexing_calibrator_sparse_test.cc",
    ],
    deps = [
        ":tflite_ops",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite:string_util",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
        "@org_tensorflow//tensorflow/lite/kernels:kernel_util",
        "@org_tensorflow//tensorflow/lite/kernels:test_util",
        "@org_tensorflow//tensorflow/lite/kernels/internal:reference",
        "@com_google_googletest//:gtest",
        "@flatbuffers",
    ],
)

py_binary(
    name = "toco_wrapper",
    srcs = ["toco_wrapper.py"],
    python_version = "PY2",
    deps = [
        ":tflite_ops",
        "//tensorflow_lattice",
        "@org_tensorflow//tensorflow/lite/python:tflite_convert_main_lib",
    ],
)

py_binary(
    name = "freeze_graph_wrapper",
    srcs = ["freeze_graph_wrapper.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_lattice",
        "@org_tensorflow//tensorflow/python/tools:freeze_graph_main_lib",
    ],
)
