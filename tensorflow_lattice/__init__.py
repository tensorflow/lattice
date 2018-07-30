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
"""Lattice modeling.

This package provides functions and classes for lattice modeling.

See full description in `README.md` file.


  use them.
"""

# pylint: disable=unused-import,wildcard-import, line-too-long

from __future__ import absolute_import

# Dependency imports

# Import all modules here, but only import functions and classes that are
# more likely to be used directly by users.
from tensorflow_lattice.python.estimators.calibrated import input_calibration_layer_from_hparams
from tensorflow_lattice.python.estimators.calibrated_etl import calibrated_etl_classifier
from tensorflow_lattice.python.estimators.calibrated_etl import calibrated_etl_regressor
from tensorflow_lattice.python.estimators.calibrated_lattice import calibrated_lattice_classifier
from tensorflow_lattice.python.estimators.calibrated_lattice import calibrated_lattice_regressor
from tensorflow_lattice.python.estimators.calibrated_linear import calibrated_linear_classifier
from tensorflow_lattice.python.estimators.calibrated_linear import calibrated_linear_regressor
from tensorflow_lattice.python.estimators.calibrated_rtl import calibrated_rtl_classifier
from tensorflow_lattice.python.estimators.calibrated_rtl import calibrated_rtl_regressor
from tensorflow_lattice.python.estimators.hparams import CalibratedEtlHParams
from tensorflow_lattice.python.estimators.hparams import CalibratedHParams
from tensorflow_lattice.python.estimators.hparams import CalibratedLatticeHParams
from tensorflow_lattice.python.estimators.hparams import CalibratedLinearHParams
from tensorflow_lattice.python.estimators.hparams import CalibratedRtlHParams
from tensorflow_lattice.python.estimators.hparams import PerFeatureHParams
from tensorflow_lattice.python.estimators.separately_calibrated_rtl import separately_calibrated_rtl_classifier
from tensorflow_lattice.python.estimators.separately_calibrated_rtl import separately_calibrated_rtl_regressor
from tensorflow_lattice.python.lib.keypoints_initialization import load_keypoints_from_quantiles
from tensorflow_lattice.python.lib.keypoints_initialization import save_quantiles_for_keypoints
from tensorflow_lattice.python.lib.keypoints_initialization import save_quantiles_for_keypoints_once
from tensorflow_lattice.python.lib.keypoints_initialization import uniform_keypoints_for_signal
from tensorflow_lattice.python.lib.lattice_layers import ensemble_lattices_layer
from tensorflow_lattice.python.lib.lattice_layers import lattice_layer
from tensorflow_lattice.python.lib.lattice_layers import monotone_lattice
from tensorflow_lattice.python.lib.pwl_calibration_layers import calibration_layer
from tensorflow_lattice.python.lib.pwl_calibration_layers import input_calibration_layer
from tensorflow_lattice.python.lib.regularizers import calibrator_regularization
from tensorflow_lattice.python.lib.regularizers import lattice_regularization
from tensorflow_lattice.python.lib.tools import DEFAULT_NAME
from tensorflow_lattice.python.ops.gen_monotonic_projection import monotonic_projection
from tensorflow_lattice.python.ops.gen_pwl_indexing_calibrator import pwl_indexing_calibrator
from tensorflow_lattice.python.ops.lattice_ops import lattice
# pylint: enable=unused-import,wildcard-import,line-too-long
