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

"""Classes defining trained TFL model structure and parameter information.

This package provides representations and tools for analysis of a trained
TF Lattice model, e.g. a canned estimator in saved model format.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


class ModelGraph(
    collections.namedtuple('ModelGraph', ['nodes', 'output_node'])):
  """Model info and parameter as a graph.

  Note that this is not a TF graph, but rather a graph of python object that
  describe model structure and parameters.

  Attributes:
    nodes: List of all the nodes in the model.
    output_node: The output node of the model.
  """


class InputFeatureNode(
    collections.namedtuple('InputFeatureNode',
                           ['name', 'is_categorical', 'vocabulary_list'])):
  """Input features to the model.

  Attributes:
    name: Name of the input feature.
    is_categorical: If the feature is categorical.
    vocabulary_list: Category values for categorical features or None.
  """


class PWLCalibrationNode(
    collections.namedtuple('PWLCalibrationNode', [
        'input_node', 'input_keypoints', 'output_keypoints', 'default_input',
        'default_output'
    ])):
  """Represetns a PWL calibration layer.

  Attributes:
    input_node: Input node for the calibration.
    input_keypoints: Input keypoints for PWL calibration.
    output_keypoints: Output keypoints for PWL calibration.
    default_input: Default/missing input value or None.
    default_output: Default/missing output value or None.
  """


class CategoricalCalibrationNode(
    collections.namedtuple('CategoricalCalibrationNode',
                           ['input_node', 'output_values', 'default_input'])):
  """Represetns a categorical calibration layer.

  Attributes:
    input_node: Input node for the calibration.
    output_values: Output calibration values. If the calibrated feature has
      default/missing values, the last value will be for default/missing.
    default_input: Default/missing input value or None.
  """


class LinearNode(
    collections.namedtuple('LinearNode',
                           ['input_nodes', 'coefficients', 'bias'])):
  """Represents a linear layer.

  Attributes:
    input_nodes: List of input nodes to the linear layer.
    coefficients: Linear weights.
    bias: Bias term for the linear layer.
  """


class LatticeNode(
    collections.namedtuple('LatticeNode', ['input_nodes', 'weights'])):
  """Represetns a lattice layer.

  Attributes:
    input_nodes: List of input nodes to the lattice layer.
    weights: Lattice parameters.
  """


class KroneckerFactoredLatticeNode(
    collections.namedtuple('KroneckerFactoredLatticeNode',
                           ['input_nodes', 'weights', 'scale', 'bias'])):
  """Represents a kronecker-factored lattice layer.

  Attributes:
    input_nodes: List of input nodes to the kronecker-factored lattice layer.
    weights: Kronecker-factored lattice kernel parameters of shape
      `(1, lattice_sizes, units * dims, num_terms)`.
    scale: Kronecker-factored lattice scale parameters of shape
      `(units, num_terms)`.
    bias: Kronecker-factored lattice bias parameters of shape `(units)`.
  """


class MeanNode(collections.namedtuple('MeanNode', ['input_nodes'])):
  """Represents an averaging layer.

  Attributes:
    input_nodes: List of input nodes to the average layer.
  """
