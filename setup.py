# Copyright 2018 The TensorFlow Lattice Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# ==============================================================================
"""Package setup script for TensorFlow Lattice library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import sys

from setuptools import find_packages
from setuptools import setup

# This version number should always be that of the *next* (unreleased) version.
# Immediately after uploading a package to PyPI, you should increment the
# version number and push to gitHub.
__version__ = "2.1.1"

if "--release" in sys.argv:
  sys.argv.remove("--release")
  _name = "tensorflow_lattice"
else:
  # Build a nightly package by default.
  _name = "tensorflow_lattice_nightly"
  __version__ += datetime.datetime.now().strftime(".dev%Y%m%d")

_install_requires = [
    "absl-py",
    "numpy",
    "pandas",
    "six",
    "scikit-learn",
    "matplotlib",
    "graphviz",
    "tf-keras",
]

# Part of the visualization code uses colabtools and IPython libraries. These
# are not added as hard requirements as they are mainly used in jupyter/colabs.

_extras_require = {
    "tensorflow": "tensorflow>=1.15",
}

_classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Software Development",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

_description = (
    "A library that implements optionally monotonic lattice based models.")
_long_description = """\
TensorFlow Lattice is a library that implements fast-to-evaluate and
interpretable (optionally monotonic) lattice based models, which are also known
as *interpolated look-up tables*. The library includes a collection of Keras
layers for lattices and feature calibration that can be composed into custom
models or used inside generic premade models.
"""

setup(
    name=_name,
    version=__version__,
    author="Google Inc.",
    author_email="no-reply@google.com",
    license="Apache 2.0",
    classifiers=_classifiers,
    install_requires=_install_requires,
    extras_require=_extras_require,
    packages=find_packages(),
    include_package_data=True,
    description=_description,
    long_description=_long_description,
    long_description_content_type="text/markdown",
    keywords="tensorflow lattice calibration machine learning",
    url=(
        "https://github.com/tensorflow/lattice"
    ),
)
