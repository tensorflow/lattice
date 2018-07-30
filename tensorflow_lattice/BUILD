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

exports_files(["LICENSE"])

package(
    default_visibility = [
        "//visibility:public",
    ],
)

py_library(
    name = "tensorflow_lattice",
    srcs = ["__init__.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow_lattice/python:keypoints_initialization",
        "//tensorflow_lattice/python:lattice_layers",
        "//tensorflow_lattice/python:lattice_ops_py",
        "//tensorflow_lattice/python:pwl_calibration_layers",
        "//tensorflow_lattice/python:pwl_calibration_ops_py",
        "//tensorflow_lattice/python:regularizers",
        "//tensorflow_lattice/python:tools",
        "//tensorflow_lattice/python/estimators:base",
        "//tensorflow_lattice/python/estimators:calibrated",
        "//tensorflow_lattice/python/estimators:calibrated_etl",
        "//tensorflow_lattice/python/estimators:calibrated_lattice",
        "//tensorflow_lattice/python/estimators:calibrated_linear",
        "//tensorflow_lattice/python/estimators:calibrated_rtl",
        "//tensorflow_lattice/python/estimators:hparams",
        "//tensorflow_lattice/python/estimators:separately_calibrated_rtl",
    ],
)

# Depend on this if you have a C++ library or binary that uses TensorFlow
# lattice ops.
cc_library(
    name = "tensorflow_lattice_cc",
    deps = [
        "//tensorflow_lattice/cc:lattice_ops",
        "//tensorflow_lattice/cc:pwl_calibration_ops",
    ],
    alwayslink = 1,
)
