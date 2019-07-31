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
workspace(name = "tensorflow_lattice")

local_repository(
    name = "org_tensorflow",
    path = "tensorflow",
)

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")

# This rule is from TensorFlow's WORKSPACE.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "e0a111000aeed2051f29fcc7a3f83be3ad8c6c93c186e64beb1ad313f0c7f9f9",
    strip_prefix = "rules_closure-cf1e44edb908e9616030cc83d085989b8e6cd6df",
    urls = [
        "http://mirror.tensorflow.org/github.com/bazelbuild/rules_closure/archive/cf1e44edb908e9616030cc83d085989b8e6cd6df.tar.gz",
        "https://github.com/bazelbuild/rules_closure/archive/cf1e44edb908e9616030cc83d085989b8e6cd6df.tar.gz",  # 2019-04-04
    ],
)

# Apple and Swift rules.
http_archive(
    name = "build_bazel_rules_apple",
    sha256 = "23792cd999f97fc97284d1c44cb1324bfdd0bc54aa68ad513fa3705aca3b1f9e",
    urls = ["https://github.com/bazelbuild/rules_apple/releases/download/0.15.0/rules_apple.0.15.0.tar.gz"],
)  # https://github.com/bazelbuild/rules_apple/releases
http_archive(
    name = "build_bazel_apple_support",
    sha256 = "7356dbd44dea71570a929d1d4731e870622151a5f27164d966dda97305f33471",
    urls = ["https://github.com/bazelbuild/apple_support/releases/download/0.6.0/apple_support.0.6.0.tar.gz"],
)  # https://github.com/bazelbuild/apple_support/releases
http_archive(
    name = "bazel_skylib",
    sha256 = "2ef429f5d7ce7111263289644d233707dba35e39696377ebab8b0bc701f7818e",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/0.8.0/bazel-skylib.0.8.0.tar.gz"],
)  # https://github.com/bazelbuild/bazel-skylib/releases
http_archive(
    name = "build_bazel_rules_swift",
    sha256 = "9efe9699e9765e6b4a5e063e4a08f6b163cccaf0443f775d935baf5c3cd6ed0e",
    urls = ["https://github.com/bazelbuild/rules_swift/releases/download/0.9.0/rules_swift.0.9.0.tar.gz"],
)  # https://github.com/bazelbuild/rules_swift/releases
http_archive(
    name = "com_github_apple_swift_swift_protobuf",
    type = "zip",
    strip_prefix = "swift-protobuf-1.5.0/",
    urls = ["https://github.com/apple/swift-protobuf/archive/1.5.0.zip"],
)  # https://github.com/apple/swift-protobuf/releases
http_file(
    name = "xctestrunner",
    executable = 1,
    urls = ["https://github.com/google/xctestrunner/releases/download/0.2.7/ios_test_runner.par"],
)  # https://github.com/google/xctestrunner/releases
# Use `swift_rules_dependencies` to fetch the toolchains. With the
# `git_repository` rules above, the following call will skip redefining them.
load("@build_bazel_rules_swift//swift:repositories.bzl", "swift_rules_dependencies")
swift_rules_dependencies()

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

tf_workspace(
    path_prefix = "",
    tf_repo_name = "org_tensorflow",
)
