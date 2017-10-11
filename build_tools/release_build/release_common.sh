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
function build_pip_pkg {
  # Clean up bazel workspace
  bazel clean

  if [  "${TFL_NATIVE}" = true  ]; then
    # Build pip install package.
    bazel build \
      --compilation_mode=opt \
      --incompatible_dict_literal_has_no_duplicates=false \
      --incompatible_disallow_set_constructor=false \
      --distinct_host_configuration=false \
      :pip_pkg
  else
    bazel build \
      --compilation_mode=opt \
      --cpu=k8 \
      --incompatible_dict_literal_has_no_duplicates=false \
      --incompatible_disallow_set_constructor=false \
      --distinct_host_configuration=false \
      :pip_pkg
  fi

  if [  -z "${TFL_ARTIFACTS_DIR}"  ]; then
    echo "TFL_ARTIFACTS_DIR is empty, so set tp /tmp/tfl_artifacts"
    export TFL_ARTIFACTS_DIR="/tmp/tfl_artifacts"
  fi

  # Create wheel to artifacts dir.
  if  [  "${TFL_USE_GPU}" = true  ]; then
    echo 'Building pip package for gpu'
    ./bazel-bin/pip_pkg ${TFL_ARTIFACTS_DIR} --gpu
  else
    echo 'Building pip package for cpu'
    ./bazel-bin/pip_pkg ${TFL_ARTIFACTS_DIR}
  fi
}

function install_pip_and_test {
  # Check python version.
  python -V

  # Install TensorFlow Lattice
  pip install --upgrade ${TFL_ARTIFACTS_DIR}/*.whl

  # Run the example script to check whether it works or not.
  cd examples

  # Check TensorFlow version
  python -c 'import tensorflow as tf; print(tf.__version__)'

  echo 'running lattice example'
  python lattice_test.py
  echo 'running coffee example'
  python coffee_test.py
  echo 'running estimator example'
  python estimator_test.py
}
