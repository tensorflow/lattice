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
set -e
set -x

# Source common scripts.
source "build_tools/common.sh"

export TFL_PY="py2"
export TFL_USE_GPU=false

# Prepare build.
prepare_build

# Source common release scripts.
source "build_tools/release_build/release_common.sh"

# Activate virtualenv.
source ${TFL_ENV_PATH}/bin/activate

echo "Build pip package."
build_pip_pkg
echo "Done."

echo "Install pip package and test."
install_pip_and_test
echo "Done."

deactivate
