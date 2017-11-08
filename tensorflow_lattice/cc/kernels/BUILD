# Copyright 2017 The TensorFlow Lattice Authors.
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
licenses(["notice"])  # Apache 2.0

package(
    default_visibility = [
        "//tensorflow_lattice:__subpackages__",
    ],
)

load("//tensorflow_lattice:tensorflow.bzl", "tf_kernel_library")
load("//tensorflow_lattice:tensorflow.bzl", "tf_cc_test")

# Piecewise-linear calibration kernels
cc_library(
    name = "pwl_calibration_kernels",
    deps = [
        ":monotonic_projection_kernel",
        ":pwl_indexing_calibrator_kernels",
    ],
)

tf_kernel_library(
    name = "pwl_indexing_calibrator_kernels",
    srcs = ["pwl_indexing_calibrator_kernels.cc"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core:framework_lite",
        "@protobuf_archive//:protobuf",
    ],
)

tf_kernel_library(
    name = "monotonic_projection_kernel",
    srcs = ["monotonic_projection_kernel.cc"],
    deps = [
        ":monotonic_projections",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@protobuf_archive//:protobuf",
    ],
)

# Lattice interpolation kernels
cc_library(
    name = "lattice_kernels",
    deps = [
        ":hypercube_interpolation_kernels",
        ":monotone_lattice_kernels",
        ":simplex_interpolation_kernels",
    ],
)

cc_library(
    name = "lattice_interpolation_base",
    srcs = ["lattice_interpolation_base.cc"],
    hdrs = ["lattice_interpolation_base.h"],
    deps = [
        "//tensorflow_lattice/cc/lib:lattice_structure",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
)

tf_kernel_library(
    name = "hypercube_interpolation_kernels",
    srcs = ["hypercube_interpolation_kernels.cc"],
    deps = [
        ":lattice_interpolation_base",
        "//tensorflow_lattice/cc/lib:lattice_structure",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
    alwayslink = 1,
)

tf_kernel_library(
    name = "simplex_interpolation_kernels",
    srcs = ["simplex_interpolation_kernels.cc"],
    deps = [
        ":lattice_interpolation_base",
        "//tensorflow_lattice/cc/lib:lattice_structure",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
    alwayslink = 1,
)

# Monotonic projections.
cc_library(
    name = "monotonic_projections",
    hdrs = ["monotonic_projections.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core:framework_lite",
    ],
)

cc_library(
    name = "lattice_raw_iterator",
    srcs = ["lattice_raw_iterator.cc"],
    hdrs = ["lattice_raw_iterator.h"],
    deps = ["//tensorflow_lattice/cc/lib:lattice_structure"],
)

tf_cc_test(
    name = "lattice_raw_iterator_test",
    srcs = ["lattice_raw_iterator_test.cc"],
    deps = [
        ":lattice_raw_iterator",
        "//tensorflow_lattice/cc:test_main",
        "//tensorflow_lattice/cc/lib:lattice_structure",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

cc_library(
    name = "monotonic_lattice_projections",
    hdrs = ["monotonic_lattice_projections.h"],
    deps = [
        ":lattice_raw_iterator",
        ":monotonic_projections",
        "//tensorflow_lattice/cc/lib:lattice_structure",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
)

tf_cc_test(
    name = "monotonic_lattice_projections_test",
    srcs = ["monotonic_lattice_projections_test.cc"],
    deps = [
        ":monotonic_lattice_projections",
        "//tensorflow_lattice/cc:test_main",
        "//tensorflow_lattice/cc/lib:lattice_structure",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

# Monotone lattice kernels.
tf_kernel_library(
    name = "monotone_lattice_kernels",
    srcs = ["monotone_lattice_kernels.cc"],
    deps = [
        ":lattice_interpolation_base",
        ":monotonic_lattice_projections",
        "//tensorflow_lattice/cc/lib:lattice_structure",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
    alwayslink = 1,
)
