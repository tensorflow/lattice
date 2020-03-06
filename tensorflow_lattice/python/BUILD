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

package(
    default_visibility = [
        "//tensorflow_lattice:__subpackages__",
    ],
)

licenses(["notice"])

py_library(
    name = "pwl_calibration_layer",
    srcs = ["pwl_calibration_layer.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":pwl_calibration_lib",
        # absl/logging dep,
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_library(
    name = "pwl_calibration_lib",
    srcs = ["pwl_calibration_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        # enum dep,
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
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_library(
    name = "linear_lib",
    srcs = ["linear_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        # tensorflow:tensorflow_no_contrib dep,
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
        ":utils",
        # enum dep,
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

py_test(
    name = "linear_test",
    size = "large",
    srcs = ["linear_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":linear_layer",
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
        ":categorical_calibration_layer",
        ":lattice_lib",
        ":pwl_calibration_layer",
        # tensorflow:tensorflow_no_contrib dep,
    ],
)

py_library(
    name = "lattice_lib",
    srcs = ["lattice_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
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
        ":configs",
        # absl/logging dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        # tensorflow dep,
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        # tensorflow dep,
    ],
)

py_library(
    name = "test_utils",
    srcs = ["test_utils.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":visualization",
        # absl/logging dep,
        # numpy dep,
    ],
)

py_library(
    name = "visualization",
    srcs = ["visualization.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":model_info",
        # graphviz dep,
        # matplotlib dep,
        # mpl_toolkits/mplot3d dep,
        # numpy dep,
    ],
)

py_library(
    name = "estimators",
    srcs = ["estimators.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":categorical_calibration_layer",
        ":configs",
        ":lattice_layer",
        ":lattice_lib",
        ":linear_layer",
        ":model_info",
        ":premade",
        ":premade_lib",
        ":pwl_calibration_layer",
        # absl/logging dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "estimators_test",
    size = "enormous",
    srcs = ["estimators_test.py"],
    python_version = "PY3",
    # shard_count = 10,
    srcs_version = "PY2AND3",
    deps = [
        ":configs",
        ":estimators",
        ":model_info",
        # absl/logging dep,
        # sklearn dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "premade",
    srcs = ["premade.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":configs",
        ":premade_lib",
        # tensorflow dep,
    ],
)

py_library(
    name = "premade_lib",
    srcs = ["premade_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":categorical_calibration_layer",
        ":configs",
        ":lattice_layer",
        ":linear_layer",
        ":pwl_calibration_layer",
        # absl/logging dep,
        # enum dep,
        # numpy dep,
        # six dep,
        # tensorflow dep,
    ],
)

py_test(
    name = "premade_test",
    srcs = ["premade_test.py"],
    python_version = "PY3",
    srcs_version = "PY2AND3",
    deps = [
        ":configs",
        ":premade_lib",
        # absl/testing:parameterized dep,
        # numpy dep,
        # tensorflow dep,
    ],
)

py_library(
    name = "model_info",
    srcs = ["model_info.py"],
    srcs_version = "PY2AND3",
    deps = [],
)
