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

# This script will run the bash function tensorflow_lattice_test under a python2
# environment.

set -e
set -x

# Source common scripts.
source "build_tools/common.sh"

export IS_MAC=true
export TFL_PY="py2"
export TFL_USE_GPU=false

# Prepare build.
prepare_build

# Source common ci scripts.
source "build_tools/ci_build/ci_common.sh"

# Activate virtualenv.
source ${TFL_ENV_PATH}/bin/activate

echo "Running all tests."
tensorflow_lattice_test
echo "Done with testing."

deactivate
