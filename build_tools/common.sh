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

# Git initialization
function git_init {
  # Run configure.
  export TF_NEED_GCP=0
  export TF_NEED_HDFS=0
  export PYTHON_BIN_PATH=$(which python)

  # Initialize git.
  git init

  if [  -d "tensorflow"  ]; then
    echo "TensorFlow submodule exist. Checkout r1.4"
    cd tensorflow
    git checkout r1.4
    cd -
  else
    echo "Add TensorFlow r1.4 submodule."
    git submodule add -b r1.4 https://github.com/tensorflow/tensorflow.git
  fi

  # Fetch all submodules.
  git submodule update --init --recursive

  # Configure tensorflow.
  cd tensorflow
  git show --oneline -s
  yes "" | ./configure

  cd -
}

# Create virtualenv.
function create_virtualenv {
  if [  "${TFL_PY}" = "py3" ]; then
    echo "Setting up python 3 virtualenv"
    export TFL_ENV_PATH=${TFL_ROOT}/tensorflow-lattice-env-py3
    virtualenv --system-site-packages -p python3 ${TFL_ENV_PATH}
  else
    echo "Setting up python 2 virtualenv"
    export TFL_ENV_PATH=${TFL_ROOT}/tensorflow-lattice-env-py2
    virtualenv --system-site-packages -p python2.7 ${TFL_ENV_PATH}
  fi
  source ${TFL_ENV_PATH}/bin/activate
  python -V
  pip install --upgrade pip
  pip install six numpy wheel enum34
  deactivate
}

# Pointfix aws workspace build rule.
# Without this fix, bazel replaces @%ws% -> empty which makes all imports fail
# in aws.BUILD.
function aws_ws_fix {
  sed -i='' 's,@%ws%,@org_tensorflow,' tensorflow/third_party/aws.BUILD
}

# Prepare all necessary environment for bazel build & testing.
function prepare_build {
  # If TFL_ROOT does not exist, create one in here.
  if [  -z "${TFL_ROOT}"  ]; then
    echo "TFL_ROOT is empty, so set to /tmp/tfl_root."
    export TFL_ROOT="/tmp/tfl_root"
  fi

  # Create virtualenv.
  create_virtualenv

  # Activate virtualenv.
  source ${TFL_ENV_PATH}/bin/activate

  if [  "${TFL_USE_GPU}" = true  ]; then
    echo "GPU build -- Enable CUDA"
    export TF_NEED_CUDA=1
  else
    echo "CPU build -- No CUDA"
    export TF_NEED_CUDA=0
  fi

  echo "Initialize git repo."
  git_init
  echo "Initialization is done."

  echo "Pointfix aws.BUILD"
  aws_ws_fix

  deactivate
}
