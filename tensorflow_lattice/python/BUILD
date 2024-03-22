# Copyright 2019 The TensorFlow Lattice Authors.
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

load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")
load("//third_party/bazel_rules/rules_python/python:py_test.bzl", "py_test")

package(
    default_visibility = [
        "//tensorflow_lattice:__subpackages__",
    ],
)

licenses(["notice"])

# Build rules are alphabetized. Please add new rules alphabetically
# to maintain the ordering.
py_library(
    name = "aggregation_layer",
    srcs = ["aggregation_layer.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "aggregation_test",
    srcs = ["aggregation_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":aggregation_layer",
        # tensorflow dep,
    ],
)

py_library(
    name = "categorical_calibration_layer",
    srcs = ["categorical_calibration_layer.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":categorical_calibration_lib",
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_library(
    name = "categorical_calibration_lib",
    srcs = ["categorical_calibration_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":internal_utils",
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_test(
    name = "categorical_calibration_test",
    size = "large",
    srcs = ["categorical_calibration_test.py"],
    python_version = "PY3",
    # shard_count = 4,
    srcs_version = "PY2AND3",
    deps = [
        ":categorical_calibration_layer",
        ":parallel_combination_layer",
        ":test_utils",
        # absl/logging dep,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "cdf_layer",
    srcs = ["cdf_layer.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
        # tensorflow dep,
    ],
)

py_test(
    name = "cdf_test",
    size = "large",
    srcs = ["cdf_test.py"],
    python_version = "PY3",
    # shard_count = 12,
    srcs_version = "PY3",
    deps = [
        ":cdf_layer",
        ":test_utils",
        ":utils",
        # absl/logging dep,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "configs",
    srcs = ["configs.py"],
    srcs_version = "PY2AND3",
    deps = [
        # absl/logging dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "configs_test",
    size = "small",
    srcs = ["configs_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":categorical_calibration_layer",
        ":configs",
        ":lattice_layer",
        ":linear_layer",
        ":premade",
        ":pwl_calibration_layer",
        # absl/logging dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "internal_utils",
    srcs = ["internal_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "internal_utils_test",
    srcs = ["internal_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":internal_utils",
        # tensorflow dep,
    ],
)

py_library(
    name = "kronecker_factored_lattice_layer",
    srcs = ["kronecker_factored_lattice_layer.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":kronecker_factored_lattice_lib",
        ":utils",
        # tensorflow dep,
    ],
)

py_library(
    name = "kronecker_factored_lattice_lib",
    srcs = ["kronecker_factored_lattice_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        # numpy dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "kronecker_factored_lattice_test",
    size = "large",
    srcs = ["kronecker_factored_lattice_test.py"],
    python_version = "PY3",
    # shard_count = 12,
    srcs_version = "PY2AND3",
    deps = [
        ":kronecker_factored_lattice_layer",
        ":kronecker_factored_lattice_lib",
        ":test_utils",
        # absl/logging dep,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "lattice_layer",
    srcs = ["lattice_layer.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":lattice_lib",
        ":pwl_calibration_layer",
        ":utils",
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_library(
    name = "lattice_lib",
    srcs = ["lattice_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        # absl/logging dep,
        # numpy dep,
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_test(
    name = "lattice_test",
    size = "large",
    srcs = ["lattice_test.py"],
    python_version = "PY3",
    # shard_count = 12,
    srcs_version = "PY2AND3",
    deps = [
        ":lattice_layer",
        ":test_utils",
        # absl/logging dep,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "linear_layer",
    srcs = ["linear_layer.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":linear_lib",
        ":utils",
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_library(
    name = "linear_lib",
    srcs = ["linear_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":internal_utils",
        ":utils",
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_test(
    name = "linear_test",
    size = "large",
    srcs = ["linear_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":linear_layer",
        ":test_utils",
        ":utils",
        # absl/logging dep,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "model_info",
    srcs = ["model_info.py"],
    srcs_version = "PY2AND3",
)

py_library(
    name = "parallel_combination_layer",
    srcs = ["parallel_combination_layer.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":categorical_calibration_layer",
        ":lattice_layer",
        ":linear_layer",
        ":pwl_calibration_layer",
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_test(
    name = "parallel_combination_test",
    size = "large",
    srcs = ["parallel_combination_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":lattice_layer",
        ":parallel_combination_layer",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "premade",
    srcs = ["premade.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":aggregation_layer",
        ":categorical_calibration_layer",
        ":configs",
        ":kronecker_factored_lattice_layer",
        ":lattice_layer",
        ":parallel_combination_layer",
        ":premade_lib",
        ":pwl_calibration_layer",
        # absl/logging dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "premade_lib",
    srcs = ["premade_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":aggregation_layer",
        ":categorical_calibration_layer",
        ":configs",
        ":kronecker_factored_lattice_layer",
        ":kronecker_factored_lattice_lib",
        ":lattice_layer",
        ":lattice_lib",
        ":linear_layer",
        ":pwl_calibration_layer",
        ":rtl_layer",
        ":utils",
        # absl/logging dep,
        # numpy dep,
        # six dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "premade_test",
    size = "large",
    srcs = ["premade_test.py"],
    python_version = "PY3",
    # shard_count = 10,
    srcs_version = "PY2AND3",
    deps = [
        ":configs",
        ":premade",
        ":premade_lib",
        # absl/logging dep,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "pwl_calibration_layer",
    srcs = ["pwl_calibration_layer.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pwl_calibration_lib",
        ":utils",
        # absl/logging dep,
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_library(
    name = "pwl_calibration_lib",
    srcs = ["pwl_calibration_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_test(
    name = "pwl_calibration_test",
    size = "large",
    srcs = ["pwl_calibration_test.py"],
    python_version = "PY3",
    # shard_count = 12,
    srcs_version = "PY2AND3",
    deps = [
        ":parallel_combination_layer",
        ":pwl_calibration_layer",
        ":test_utils",
        ":utils",
        # absl/logging dep,
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_library(
    name = "rtl_layer",
    srcs = ["rtl_layer.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":kronecker_factored_lattice_layer",
        ":lattice_layer",
        ":rtl_lib",
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_library(
    name = "rtl_lib",
    srcs = ["rtl_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        # six dep,
    ],
)

py_test(
    name = "rtl_test",
    size = "large",
    srcs = ["rtl_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":linear_layer",
        ":pwl_calibration_layer",
        ":rtl_layer",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        # absl/logging dep,
        # numpy dep,
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        # six dep,
    ],
)

py_library(
    name = "conditional_pwl_calibration",
    srcs = ["conditional_pwl_calibration.py"],
    deps = [
        # numpy dep,
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_library(
    name = "conditional_cdf",
    srcs = ["conditional_cdf.py"],
    deps = [
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_test(
    name = "conditional_cdf_test",
    srcs = ["conditional_cdf_test.py"],
    deps = [
        ":conditional_cdf",
        # absl/testing:parameterized dep,
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_test(
    name = "conditional_pwl_calibration_test",
    srcs = ["conditional_pwl_calibration_test.py"],
    deps = [
        ":conditional_pwl_calibration",
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        # absl/testing:parameterized dep,
        # tensorflow dep,
    ],
)
