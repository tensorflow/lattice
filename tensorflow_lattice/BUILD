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

package(
    default_visibility = [
        "//visibility:public",
    ],
)

licenses(["notice"])

exports_files(["LICENSE"])

py_library(
    name = "tensorflow_lattice",
    srcs = [
        "__init__.py",
        "layers/__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_lattice/python:aggregation_layer",
        "//tensorflow_lattice/python:categorical_calibration_layer",
        "//tensorflow_lattice/python:categorical_calibration_lib",
        "//tensorflow_lattice/python:cdf_layer",
        "//tensorflow_lattice/python:conditional_cdf",
        "//tensorflow_lattice/python:conditional_pwl_calibration",
        "//tensorflow_lattice/python:configs",
        "//tensorflow_lattice/python:kronecker_factored_lattice_layer",
        "//tensorflow_lattice/python:kronecker_factored_lattice_lib",
        "//tensorflow_lattice/python:lattice_layer",
        "//tensorflow_lattice/python:lattice_lib",
        "//tensorflow_lattice/python:linear_layer",
        "//tensorflow_lattice/python:linear_lib",
        "//tensorflow_lattice/python:model_info",
        "//tensorflow_lattice/python:parallel_combination_layer",
        "//tensorflow_lattice/python:premade",
        "//tensorflow_lattice/python:premade_lib",
        "//tensorflow_lattice/python:pwl_calibration_layer",
        "//tensorflow_lattice/python:pwl_calibration_lib",
        "//tensorflow_lattice/python:rtl_layer",
        "//tensorflow_lattice/python:test_utils",
        "//tensorflow_lattice/python:utils",
    ],
)
