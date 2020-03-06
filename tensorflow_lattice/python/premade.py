# Copyright 2019 Google LLC
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
"""TF Lattice premade models implement typical monotonic model architectures.

You can use TFL premade models to easily construct commonly used monotonic model
architectures. To construct a TFL premade model, construct a model configuration
from `tfl.configs` and pass it to the premade model constructor. No fields in
the model config will be automatically filled in, so the config must be fully
specified. Note that the inputs to the model should match the order in which
they are defined in the feature configs.

```python
model_config = tfl.configs.CalibratedLatticeConfig(...)
calibrated_lattice_model = tfl.premade.CalibratedLattice(
    model_config=model_config)
calibrated_lattice_model.compile(...)
calibrated_lattice_model.fit(...)
```

Supported models are defined in `tfl.configs`. Each model architecture can be
used the same as any other `tf.keras.Model`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import configs
from . import premade_lib

import tensorflow as tf


# TODO: add support for serialization and object scoping or annoations.
class CalibratedLatticeEnsemble(tf.keras.Model):
  """Premade model for Tensorflow calibrated lattice ensemble models.

  Creates a `tf.keras.Model` for the model architecture specified by the
  `model_config`, which should be a
  `tfl.configs.CalibratedLatticeEnsembleConfig`. No fields in the model config
  will be automatically filled in, so the config must be fully specified. Note
  that the inputs to the model should match the order in which they are defined
  in the feature configs.

  Example:

  ```python
  model_config = tfl.configs.CalibratedLatticeEnsembleConfig(...)
  calibrated_lattice_ensemble_model = tfl.premade.CalibratedLatticeEnsemble(
      model_config=model_config)
  calibrated_lattice_ensemble_model.compile(...)
  calibrated_lattice_ensemble_model.fit(...)
  ```
  """

  def __init__(self, model_config, dtype=tf.float32):
    """Initializes a `CalibratedLatticeEnsemble` instance.

    Args:
      model_config: Model configuration object describing model architecutre.
        Should be one of the model configs in `tfl.configs`.
      dtype: dtype of layers used in the model.
    """
    # Check that proper config has been given.
    if not isinstance(model_config, configs.CalibratedLatticeEnsembleConfig):
      raise ValueError('Invalid config type: {}'.format(type(model_config)))
    # Verify that the config is fully specified.
    premade_lib.verify_config(model_config)
    # Get feature configs and construct model.
    input_layer = premade_lib.build_input_layer(
        feature_configs=model_config.feature_configs, dtype=dtype)

    submodels_inputs = premade_lib.build_calibration_layers(
        calibration_input_layer=input_layer,
        feature_configs=model_config.feature_configs,
        model_config=model_config,
        layer_output_range=premade_lib.LayerOutputRange.INPUT_TO_LATTICE,
        submodels=model_config.lattices,
        separate_calibrators=model_config.separate_calibrators,
        dtype=dtype)

    lattice_outputs = []
    for submodel_index, (lattice_feature_names, lattice_input) in enumerate(
        zip(model_config.lattices, submodels_inputs)):
      lattice_feature_configs = [
          model_config.feature_config_by_name(feature_name)
          for feature_name in lattice_feature_names
      ]

      lattice_layer_output_range = (
          premade_lib.LayerOutputRange.INPUT_TO_FINAL_CALIBRATION
          if model_config.output_calibration else
          premade_lib.LayerOutputRange.MODEL_OUTPUT)
      lattice_outputs.append(
          premade_lib.build_lattice_layer(
              lattice_input=lattice_input,
              feature_configs=lattice_feature_configs,
              model_config=model_config,
              layer_output_range=lattice_layer_output_range,
              submodel_index=submodel_index,
              is_inside_ensemble=True,
              dtype=dtype))

    if len(lattice_outputs) > 1:
      averaged_lattice_output = tf.keras.layers.Average()(lattice_outputs)
    else:
      averaged_lattice_output = lattice_outputs[0]
    if model_config.output_calibration:
      model_output = premade_lib.build_output_calibration_layer(
          output_calibration_input=averaged_lattice_output,
          model_config=model_config,
          dtype=dtype)
    else:
      model_output = averaged_lattice_output

    # Define inputs and initialize model.
    inputs = [
        input_layer[feature_config.name]
        for feature_config in model_config.feature_configs
    ]
    super(CalibratedLatticeEnsemble, self).__init__(
        inputs=inputs, outputs=model_output)


class CalibratedLattice(tf.keras.Model):
  """Premade model for Tensorflow calibrated lattice models.

  Creates a `tf.keras.Model` for the model architecture specified by the
  `model_config`, which should be a `tfl.configs.CalibratedLatticeConfig`. No
  fields in the model config will be automatically filled in, so the config
  must be fully specified. Note that the inputs to the model should match the
  order in which they are defined in the feature configs.

  Example:

  ```python
  model_config = tfl.configs.CalibratedLatticeConfig(...)
  calibrated_lattice_model = tfl.premade.CalibratedLattice(
      model_config=model_config)
  calibrated_lattice_model.compile(...)
  calibrated_lattice_model.fit(...)
  ```
  """

  def __init__(self, model_config, dtype=tf.float32):
    """Initializes a `CalibratedLattice` instance.

    Args:
      model_config: Model configuration object describing model architecutre.
        Should be one of the model configs in `tfl.configs`.
      dtype: dtype of layers used in the model.
    """
    # Check that proper config has been given.
    if not isinstance(model_config, configs.CalibratedLatticeConfig):
      raise ValueError('Invalid config type: {}'.format(type(model_config)))
    # Verify that the config is fully specified.
    premade_lib.verify_config(model_config)
    # Get feature configs and construct model.
    input_layer = premade_lib.build_input_layer(
        feature_configs=model_config.feature_configs, dtype=dtype)
    submodels_inputs = premade_lib.build_calibration_layers(
        calibration_input_layer=input_layer,
        feature_configs=model_config.feature_configs,
        model_config=model_config,
        layer_output_range=premade_lib.LayerOutputRange.INPUT_TO_LATTICE,
        submodels=[[
            feature_config.name
            for feature_config in model_config.feature_configs
        ]],
        separate_calibrators=False,
        dtype=dtype)

    lattice_layer_output_range = (
        premade_lib.LayerOutputRange.INPUT_TO_FINAL_CALIBRATION
        if model_config.output_calibration else
        premade_lib.LayerOutputRange.MODEL_OUTPUT)
    lattice_output = premade_lib.build_lattice_layer(
        lattice_input=submodels_inputs[0],
        feature_configs=model_config.feature_configs,
        model_config=model_config,
        layer_output_range=lattice_layer_output_range,
        submodel_index=0,
        is_inside_ensemble=False,
        dtype=dtype)

    if model_config.output_calibration:
      model_output = premade_lib.build_output_calibration_layer(
          output_calibration_input=lattice_output,
          model_config=model_config,
          dtype=dtype)
    else:
      model_output = lattice_output

    # Define inputs and initialize model.
    inputs = [
        input_layer[feature_config.name]
        for feature_config in model_config.feature_configs
    ]
    super(CalibratedLattice, self).__init__(inputs=inputs, outputs=model_output)


class CalibratedLinear(tf.keras.Model):
  """Premade model for Tensorflow calibrated linear models.

  Creates a `tf.keras.Model` for the model architecture specified by the
  `model_config`, which should be a `tfl.configs.CalibratedLinearConfig`. No
  fields in the model config will be automatically filled in, so the config
  must be fully specified. Note that the inputs to the model should match the
  order in which they are defined in the feature configs.

  Example:

  ```python
  model_config = tfl.configs.CalibratedLinearConfig(...)
  calibrated_linear_model = tfl.premade.CalibratedLinear(
      model_config=model_config)
  calibrated_linear_model.compile(...)
  calibrated_linear_model.fit(...)
  ```
  """

  def __init__(self, model_config, dtype=tf.float32):
    """Initializes a `CalibratedLinear` instance.

    Args:
      model_config: Model configuration object describing model architecutre.
        Should be one of the model configs in `tfl.configs`.
      dtype: dtype of layers used in the model.
    """
    # Check that proper config has been given.
    if not isinstance(model_config, configs.CalibratedLinearConfig):
      raise ValueError('Invalid config type: {}'.format(type(model_config)))
    # Verify that the config is fully specified.
    premade_lib.verify_config(model_config)
    # Get feature configs and construct model.
    input_layer = premade_lib.build_input_layer(
        feature_configs=model_config.feature_configs, dtype=dtype)

    calibration_layer_output_range = (
        premade_lib.LayerOutputRange.INPUT_TO_FINAL_CALIBRATION
        if model_config.output_calibration else
        premade_lib.LayerOutputRange.MODEL_OUTPUT)
    submodels_inputs = premade_lib.build_calibration_layers(
        calibration_input_layer=input_layer,
        feature_configs=model_config.feature_configs,
        model_config=model_config,
        layer_output_range=calibration_layer_output_range,
        submodels=[[
            feature_config.name
            for feature_config in model_config.feature_configs
        ]],
        separate_calibrators=False,
        dtype=dtype)

    weighted_average = (
        model_config.output_min is not None or
        model_config.output_max is not None or model_config.output_calibration)
    linear_output = premade_lib.build_linear_layer(
        linear_input=submodels_inputs[0],
        feature_configs=model_config.feature_configs,
        model_config=model_config,
        weighted_average=weighted_average,
        submodel_index=0,
        dtype=dtype)

    if model_config.output_calibration:
      model_output = premade_lib.build_output_calibration_layer(
          output_calibration_input=linear_output,
          model_config=model_config,
          dtype=dtype)
    else:
      model_output = linear_output

    # Define inputs and initialize model.
    inputs = [
        input_layer[feature_config.name]
        for feature_config in model_config.feature_configs
    ]
    super(CalibratedLinear, self).__init__(inputs=inputs, outputs=model_output)
