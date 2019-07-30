#!/usr/bin/env bash
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

# Run tensorflow lattice bazel tests.
function tensorflow_lattice_test {
  # Cleaning up bazel workspace
  bazel clean

  if [[ "${IS_MAC}" == true ]]; then
    N_JOBS=$(sysctl -n hw.ncpu)
  else
    N_JOBS=$(grep -c ^processor /proc/cpuinfo)
  fi

  echo ""
  echo "Bazel will use ${N_JOBS} concurrent job(s)."
  echo ""

  bazel test --config=opt --test_tag_filters=-gpu -k \
      --jobs=${N_JOBS} --test_timeout 300,450,1200,3600 --build_tests_only \
      --test_output=errors -- \
      //tensorflow_lattice/...
}
