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

# estimators.
py_library(
    name = "hparams",
    srcs = ["hparams.py"],
    srcs_version = "PY2AND3",
    deps = [
        "@six_archive//:six",
    ],
)

py_test(
    name = "hparams_test",
    size = "small",
    srcs = ["hparams_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":hparams",
        "@org_tensorflow//third_party/py/numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "calibrated",
    srcs = ["calibrated.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":hparams",
        "@six_archive//:six",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tensorflow_lattice/python:keypoints_initialization",
        "//tensorflow_lattice/python:pwl_calibration_layers",
        "//tensorflow_lattice/python:tools",
    ],
)

py_test(
    name = "calibrated_test",
    size = "small",
    srcs = ["calibrated_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":calibrated",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "calibrated_linear",
    srcs = ["calibrated_linear.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":calibrated",
        ":hparams",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "calibrated_linear_test",
    size = "large",
    srcs = ["calibrated_linear_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":calibrated_linear",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tensorflow_lattice/python:test_data",
    ],
)

py_library(
    name = "calibrated_lattice",
    srcs = ["calibrated_lattice.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":calibrated",
        ":hparams",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tensorflow_lattice/python:lattice_layers",
    ],
)

py_test(
    name = "calibrated_lattice_test",
    size = "large",
    srcs = ["calibrated_lattice_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":calibrated_lattice",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tensorflow_lattice/python:test_data",
    ],
)

py_library(
    name = "calibrated_rtl",
    srcs = ["calibrated_rtl.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":calibrated",
        ":hparams",
        "@six_archive//:six",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tensorflow_lattice/python:lattice_layers",
    ],
)

py_test(
    name = "calibrated_rtl_test",
    size = "large",
    srcs = ["calibrated_rtl_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":calibrated_rtl",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tensorflow_lattice/python:test_data",
    ],
)

py_library(
    name = "calibrated_etl",
    srcs = ["calibrated_etl.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":calibrated",
        ":hparams",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tensorflow_lattice/python:lattice_layers",
        "//tensorflow_lattice/python:monotone_linear_layers",
        "//tensorflow_lattice/python:pwl_calibration_layers",
    ],
)

py_test(
    name = "calibrated_etl_test",
    size = "large",
    srcs = ["calibrated_etl_test.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":calibrated_etl",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tensorflow_lattice/python:test_data",
    ],
)
