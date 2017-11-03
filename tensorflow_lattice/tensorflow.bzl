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
"""Corrections to tensorflow macros.

TensorFlow Lattice uses many tensorflow macros to create Bazel rules, but the
dependency paths in them must be corrected to point to the proper workspace.
"""

load("@org_tensorflow//tensorflow:tensorflow.bzl", "check_deps")
load(
    "@local_config_cuda//cuda:build_defs.bzl",
    "if_cuda",
)

# Sanitize a dependency so that it works correctly from code that includes
# TensorFlow as a submodule.
def clean_dep(dep):
  return str(Label(dep))

def tf_copts():
  return (["-fno-exceptions", "-DEIGEN_AVOID_STL_ARRAY",] +
          if_cuda(["-DGOOGLE_CUDA=1"]) +
          select({"@org_tensorflow//tensorflow:darwin": [],
                  "//conditions:default": ["-pthread"]}))

# Bazel-generated shared objects which must be linked into TensorFlow binaries
# to define symbols from //tensorflow/core:framework and //tensorflow/core:lib.
def tf_binary_additional_srcs():
  """Returns binaries needed for static linking."""
  return [clean_dep("@org_tensorflow//tensorflow:libtensorflow_framework.so")]

# Given a list of "op_lib_names" (a list of files in the ops directory
# without their .cc extensions), generate a library for that file.
def tf_gen_op_libs(op_lib_names, deps=None):
  # Make library out of each op so it can also be used to generate wrappers
  # for various languages.
  if deps:
    deps += ["@org_tensorflow//tensorflow/core:framework"]
  else:
    deps = ["@org_tensorflow//tensorflow/core:framework"]

  for n in op_lib_names:
    native.cc_library(name=n + "_op_lib",
                      copts=tf_copts(),
                      srcs=["ops/" + n + ".cc"],
                      deps=deps,
                      visibility=["//visibility:public"],
                      alwayslink=1,
                      linkstatic=1,)

def tf_gen_op_wrapper_py(name, out=None, hidden=[], visibility=None, deps=[],
                         require_shape_functions=False):
  """Construct a cc_binary containing the specified ops."""
  tool_name = "gen_" + name + "_py_wrappers_cc"
  if not deps:
    deps = ["@org_tensorflow//tensorflow/core:" + name + "_op_lib"]
  native.cc_binary(
      name = tool_name,
      srcs = tf_binary_additional_srcs(),
      linkopts = ["-lm"],
      copts = tf_copts(),
      linkstatic = 1,   # Faster to link this one-time-use binary dynamically
      deps = (["@org_tensorflow//tensorflow/core:framework",
               "@org_tensorflow//tensorflow/python:python_op_gen_main"] + deps),
  )

  # Invoke the previous cc_binary to generate a python file.
  if not out:
    out = "ops/gen_" + name + ".py"

  native.genrule(
      name=name + "_pygenrule",
      outs=[out],
      tools=[tool_name],
      cmd=("$(location " + tool_name + ") " + ",".join(hidden)
           + " " + ("1" if require_shape_functions else "0") + " > $@"))

  # Make a py_library out of the generated python file.
  native.py_library(
      name=name,
      srcs=[out],
      srcs_version="PY2AND3",
      visibility=visibility,
      deps=[
          "@org_tensorflow//tensorflow/python:framework_for_generated_wrappers_v2",
      ],
  )

def tf_custom_op_library_additional_deps():
  return [
      "@org_tensorflow//third_party/eigen3",
      "@org_tensorflow//tensorflow/core:framework_headers_lib",
      "@protobuf_archive//:protobuf_headers",
  ]

def tf_custom_op_py_library(name, srcs=[], dso=[], kernels=[],
                            srcs_version="PY2AND3", visibility=None, deps=[]):
  """Ties wrapper to its dynamic library."""
  kernels = kernels  # unused argument
  native.py_library(
      name=name,
      data=dso,
      srcs=srcs,
      srcs_version=srcs_version,
      visibility=visibility,
      deps=deps,
  )

def tf_cuda_library(deps=None, cuda_deps=None, copts=None, **kwargs):
  """Generate a cc_library with a conditional set of CUDA dependencies.

  When the library is built with --config=cuda:
  - both deps and cuda_deps are used as dependencies
  - the cuda runtime is added as a dependency (if necessary)
  - The library additionally passes -DGOOGLE_CUDA=1 to the list of copts
  Args:
    deps: dependencies which will always be linked.
    cuda_deps: BUILD dependencies which will be linked if and only if:
      '--config=cuda' is passed to the bazel command line.
    copts: copts always passed to the cc_library.
    **kwargs: Any other argument to cc_library.
  """
  if not deps:
    deps = []
  if not cuda_deps:
    cuda_deps = []
  if not copts:
    copts = []

  native.cc_library(
      deps = deps + if_cuda(cuda_deps + [
          "@org_tensorflow//tensorflow/core:cuda",
          "@local_config_cuda//cuda:cuda_headers"
      ]),
      copts = copts + if_cuda(["-DGOOGLE_CUDA=1"]),
      **kwargs)

def tf_kernel_library(name, prefix=None, srcs=None, gpu_srcs=None, hdrs=None,
                      deps=None, alwayslink=1, copts=tf_copts(), **kwargs):
  """A rule to build a TensorFlow OpKernel.
  May either specify srcs/hdrs or prefix.  Similar to tf_cuda_library,
  but with alwayslink=1 by default.  If prefix is specified:
    * prefix*.cc (except *.cu.cc) is added to srcs
    * prefix*.h (except *.cu.h) is added to hdrs
    * prefix*.cu.cc and prefix*.h (including *.cu.h) are added to gpu_srcs.
  With the exception that test files are excluded.
  For example, with prefix = "cast_op",
    * srcs = ["cast_op.cc"]
    * hdrs = ["cast_op.h"]
    * gpu_srcs = ["cast_op_gpu.cu.cc", "cast_op.h"]
    * "cast_op_test.cc" is excluded
  With prefix = "cwise_op"
    * srcs = ["cwise_op_abs.cc", ..., "cwise_op_tanh.cc"],
    * hdrs = ["cwise_ops.h", "cwise_ops_common.h"],
    * gpu_srcs = ["cwise_op_gpu_abs.cu.cc", ..., "cwise_op_gpu_tanh.cu.cc",
                  "cwise_ops.h", "cwise_ops_common.h",
                  "cwise_ops_gpu_common.cu.h"]
    * "cwise_ops_test.cc" is excluded
  Args:
    name: name of the rule
    prefix: prefix to be matched (see above)
    srcs: explicit source files to be used as dependencies
    gpu_srcs: explicit GPU source files
    hdrs: explicit headers
    deps: other dependencies
    alwayslink: whether to link all stactic symbols (default 1)
    copts: copts always passed to the cc_library.
    **kwargs: Any other argument to cc_library.
  """
  if not srcs:
    srcs = []
  if not hdrs:
    hdrs = []
  if not deps:
    deps = []

  if prefix:
    if native.glob([prefix + "*.cu.cc"], exclude = ["*test*"]):
      if not gpu_srcs:
        gpu_srcs = []
      gpu_srcs = gpu_srcs + native.glob([prefix + "*.cu.cc", prefix + "*.h"],
                                        exclude = [prefix + "*test*"])
    srcs = srcs + native.glob([prefix + "*.cc"],
                              exclude = [prefix + "*test*", prefix + "*.cu.cc"])
    hdrs = hdrs + native.glob([prefix + "*.h"], exclude = [prefix + "*test*",
                                                           prefix + "*.cu.h"])

  cuda_deps = ["@org_tensorflow//tensorflow/core:gpu_lib"]
  if gpu_srcs:
    for gpu_src in gpu_srcs:
      if gpu_src.endswith(".cc") and not gpu_src.endswith(".cu.cc"):
        fail("{} not allowed in gpu_srcs. .cc sources must end with .cu.cc".format(gpu_src))
    tf_gpu_kernel_library(
        name = name + "_gpu",
        srcs = gpu_srcs,
        deps = deps,
        **kwargs)
    cuda_deps.extend([":" + name + "_gpu"])

  tf_cuda_library(
      name = name,
      srcs = srcs,
      hdrs = hdrs,
      copts = copts,
      cuda_deps = cuda_deps,
      linkstatic = 1,
      alwayslink = alwayslink,
      deps = deps,
      **kwargs)

# Build defs for TensorFlow kernels

# When this target is built using --config=cuda, a cc_library is built
# that passes -DGOOGLE_CUDA=1 and '-x cuda', linking in additional
# libraries needed by GPU kernels.
def tf_gpu_kernel_library(srcs, copts=[], cuda_copts=[], deps=[], hdrs=[],
                          **kwargs):
  copts = copts + if_cuda(cuda_copts) + tf_copts()

  native.cc_library(
      srcs = srcs,
      hdrs = hdrs,
      copts = copts,
      deps = deps + if_cuda([
          "@org_tensorflow//tensorflow/core:cuda",
          "@org_tensorflow//tensorflow/core:gpu_lib",
      ]),
      alwayslink=1,
      **kwargs)

def tf_py_test(name,
               srcs,
               size="medium",
               data=[],
               main=None,
               args=[],
               tags=[],
               shard_count=1,
               additional_deps=[],
               flaky=0):
  native.py_test(
      name=name,
      size=size,
      srcs=srcs,
      main=main,
      args=args,
      tags=tags,
      visibility=[clean_dep("@org_tensorflow//tensorflow:internal")],
      shard_count=shard_count,
      data=data,
      deps=select({
          "//conditions:default": [
              clean_dep("@org_tensorflow//tensorflow/python:extra_py_tests_deps"),
              clean_dep("@org_tensorflow//tensorflow/python:gradient_checker"),
          ] + additional_deps,
          clean_dep("@org_tensorflow//tensorflow:no_tensorflow_py_deps"): []
      }),
      flaky=flaky,
      srcs_version="PY2AND3")

def tf_cc_test(name,
               srcs,
               deps,
               linkstatic=0,
               tags=[],
               data=[],
               size="medium",
               suffix="",
               args=None,
               linkopts=[]):
  native.cc_test(
      name="%s%s" % (name, suffix),
      srcs=srcs + tf_binary_additional_srcs(),
      size=size,
      args=args,
      copts=tf_copts(),
      data=data,
      deps=deps,
      linkopts=["-lpthread", "-lm"] + linkopts,
      linkstatic=linkstatic,
      tags=tags)

def tf_custom_op_library(name, srcs=[], gpu_srcs=[], deps=[]):
  """Helper to build a dynamic opeartor library (.so)."""
  cuda_deps = [
      clean_dep("@org_tensorflow//tensorflow/core:stream_executor_headers_lib"),
      "@local_config_cuda//cuda:cudart_static",
  ]
  deps = deps + tf_custom_op_library_additional_deps()

  check_deps(
      name=name + "_check_deps",
      deps=deps + if_cuda(cuda_deps),
      disallowed_deps=[
          clean_dep("@org_tensorflow//tensorflow/core:framework"),
          clean_dep("@org_tensorflow//tensorflow/core:lib")
      ])

  native.cc_binary(
      name=name,
      srcs=srcs + tf_binary_additional_srcs(),
      deps=deps + if_cuda(cuda_deps),
      data=[name + "_check_deps"],
      copts=tf_copts(),
      linkshared=1,
      linkopts=select({
          "//conditions:default": [
              "-lm",
          ],
          clean_dep("@org_tensorflow//tensorflow:darwin"): [],
      }),)
