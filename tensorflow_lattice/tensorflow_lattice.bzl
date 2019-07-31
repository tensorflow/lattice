# -*- Python -*-

# Copyright 2017 The TensorFlow Lattice Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Bazel macros for TensorFlow Lattice."""

def _add_tf_search_path(prefix, levels_to_root):
    root = "%s/%s" % (prefix, "/".join([".."] * (levels_to_root + 1)))
    tf_root = "%s/external/org_tensorflow/tensorflow" % root
    return "-rpath,%s" % tf_root

def rpath_linkopts(name):
    """Add proper rpath_linkopts to the build rule.

    This function adds tensorflow root to rpath for Darwin builds.

    Args:
      name: Name of the target.

    Returns:
      rpath linker options.
    """
    levels_to_root = native.package_name().count("/") + name.count("/")
    return select({
        "@org_tensorflow//tensorflow:macos": [
            "-Wl,%s" % (_add_tf_search_path("@loader_path", levels_to_root),),
        ],
        "//conditions:default": [
        ],
    })
