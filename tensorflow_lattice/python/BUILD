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
licenses(["notice"])  # Apache 2.0 License

package(
    default_visibility = [
        "//tensorflow_lattice:__subpackages__",
    ],
)

exports_files(["LICENSE"])

# All python tests can run under python 2 and 3.
load(
    "//tensorflow_lattice:tensorflow_lattice.bzl",
    "rpath_linkopts",
)
load(
    "@org_tensorflow//tensorflow:tensorflow.bzl",
    "tf_gen_op_wrapper_py",
)
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

tf_gen_op_wrapper_py(
    name = "pwl_indexing_calibrator_py_wrapper",
    out = "ops/gen_pwl_indexing_calibrator.py",
    cc_linkopts = rpath_linkopts("pwl_indexing_calibrator_py_wrapper"),
    hidden = [],
    deps = [
        "//tensorflow_lattice/cc:pwl_indexing_calibrator_ops_op_lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "monotonic_projection_py_wrapper",
    out = "ops/gen_monotonic_projection.py",
    cc_linkopts = rpath_linkopts("monotonic_projection_py_wrapper"),
    hidden = [],
    deps = [
        "//tensorflow_lattice/cc:monotonic_projection_op_op_lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "lattice_interpolation_py_wrapper",
    out = "ops/gen_lattice_interpolation.py",
    cc_linkopts = rpath_linkopts("lattice_interpolation_py_wrapper"),
    hidden = [],
    require_shape_functions = True,
    deps = [
        "//tensorflow_lattice/cc:lattice_interpolation_ops_op_lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "monotone_lattice_py_wrapper",
    out = "ops/gen_monotone_lattice.py",
    cc_linkopts = rpath_linkopts("monotone_lattice_py_wrapper"),
    hidden = [],
    require_shape_functions = True,
    deps = [
        "//tensorflow_lattice/cc:monotone_lattice_ops_op_lib",
    ],
)

tf_custom_op_py_library(
    name = "lattice_ops_py",
    srcs = [
        "ops/lattice_ops.py",
    ],
    dso = [
        "//tensorflow_lattice/cc:ops/_lattice_ops.so",
    ],
    kernels = [
        "//tensorflow_lattice/cc:lattice_ops",
        "//tensorflow_lattice/cc/kernels:lattice_kernels",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":lattice_interpolation_py_wrapper",
        ":monotone_lattice_py_wrapper",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python:framework_for_generated_wrappers",
    ],
)

tf_custom_op_py_library(
    name = "pwl_calibration_ops_py",
    srcs = [
        "ops/pwl_calibration_ops.py",
    ],
    dso = [
        "//tensorflow_lattice/cc:ops/_pwl_calibration_ops.so",
    ],
    kernels = [
        "//tensorflow_lattice/cc:pwl_calibration_ops",
        "//tensorflow_lattice/cc/kernels:pwl_calibration_kernels",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":monotonic_projection_py_wrapper",
        ":pwl_indexing_calibrator_py_wrapper",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python:framework_for_generated_wrappers",
    ],
)

py_library(
    name = "keypoints_initialization",
    srcs = ["lib/keypoints_initialization.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tools",
        "@org_tensorflow//third_party/py/numpy",
        "@six_archive//:six",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "keypoints_initialization_test",
    size = "medium",
    srcs = ["lib/keypoints_initialization_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keypoints_initialization",
        "@absl_py//absl/testing:parameterized",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "pwl_calibration_layers",
    srcs = ["lib/pwl_calibration_layers.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":keypoints_initialization",
        ":pwl_calibration_ops_py",
        ":regularizers",
        ":tools",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "pwl_calibration_layers_test",
    size = "medium",
    srcs = ["lib/pwl_calibration_layers_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pwl_calibration_layers",
        ":tools",
        "@org_tensorflow//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "lattice_layers",
    srcs = ["lib/lattice_layers.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":lattice_ops_py",
        ":regularizers",
        ":tools",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "lattice_layers_test",
    size = "medium",
    srcs = ["lib/lattice_layers_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":lattice_layers",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "monotone_linear_layers",
    srcs = ["lib/monotone_linear_layers.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":regularizers",
        ":tools",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "monotone_linear_layers_test",
    size = "medium",
    srcs = ["lib/monotone_linear_layers_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":monotone_linear_layers",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "regularizers",
    srcs = ["lib/regularizers.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":tools",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "regularizers_test",
    size = "large",
    srcs = ["lib/regularizers_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":regularizers",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

# TensorFlow Lattice internal libraries.
py_library(
    name = "tools",
    srcs = ["lib/tools.py"],
    srcs_version = "PY2AND3",
    deps = [
        "@org_tensorflow//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "tools_test",
    size = "medium",
    srcs = ["lib/tools_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":test_data",
        ":tools",
        "@org_tensorflow//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "test_data",
    srcs = ["lib/test_data.py"],
    srcs_version = "PY2AND3",
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
