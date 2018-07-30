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
"""Tests for TensorFlow Lattice's lattice_layers module."""

# Dependency imports

from tensorflow_lattice.python.lib import lattice_layers

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class LatticeParamTestCase(test.TestCase):

  def testTwoByTwoOneOutput(self):
    lattice_param = lattice_layers.lattice_param_as_linear(
        lattice_sizes=[2, 2], output_dim=1)
    self.assertAllClose([[-0.5, 0.0, 0.0, 0.5]], lattice_param)

  def testTwoByTwoTwoOutputs(self):
    lattice_param = lattice_layers.lattice_param_as_linear(
        lattice_sizes=[2, 2],
        output_dim=2,
        linear_weights=[[1.0, 1.0], [-0.1, 0.3]])
    self.assertAllClose([[-0.5, 0.0, 0.0, 0.5], [-0.05, -0.1, 0.1, 0.05]],
                        lattice_param)

  def testTwoByThreeByTwoOneOutput(self):
    lattice_param = lattice_layers.lattice_param_as_linear(
        lattice_sizes=[2, 3, 2], output_dim=1, linear_weights=[-1.0, 1.0, 1.0])
    self.assertAllClose([[
        -0.1666667, -0.5, 0.0, -0.3333333, 0.1666667, -0.1666667, 0.1666667,
        -0.1666667, 0.3333333, 0.0, 0.5, 0.1666667
    ]], lattice_param)

  def testWrongLatticeSizesExpectError(self):
    with self.assertRaises(ValueError):
      _ = lattice_layers.lattice_param_as_linear(
          lattice_sizes=[1, -1], output_dim=1)

  def testEmptyLatticeSizesExpectError(self):
    with self.assertRaises(ValueError):
      _ = lattice_layers.lattice_param_as_linear(lattice_sizes=[], output_dim=1)

  def testMoreLinearWeightsThanLatticeRankExpectError(self):
    with self.assertRaises(ValueError):
      _ = lattice_layers.lattice_param_as_linear(
          lattice_sizes=[2, 2], output_dim=1, linear_weights=[1, 2, 3])

  def testLessLinearWeightsThanOutputDimExpectError(self):
    with self.assertRaises(ValueError):
      _ = lattice_layers.lattice_param_as_linear(
          lattice_sizes=[2, 2], output_dim=2, linear_weights=[[1, 2]])

  def testWrongLinearWeightsExpectError(self):
    with self.assertRaises(ValueError):
      _ = lattice_layers.lattice_param_as_linear(
          lattice_sizes=[2, 2], output_dim=2, linear_weights=[[1], [1, 2]])


class LatticeLayersTestCase(test.TestCase):

  def setUp(self):
    super(LatticeLayersTestCase, self).setUp()

  def _testLatticeLayerEvaluation(self, interpolation_type, lattice_sizes,
                                  output_dim, inputs, parameters,
                                  expected_outputs):
    """Test evaluation of lattice layers."""
    with ops.Graph().as_default():
      input_tensor = array_ops.constant(inputs, dtype=dtypes.float32)
      init_param = array_ops.constant(parameters, dtype=dtypes.float32)
      (output_tensor, _, _, _) = lattice_layers.lattice_layer(
          input_tensor,
          lattice_sizes=lattice_sizes,
          output_dim=output_dim,
          interpolation_type=interpolation_type,
          lattice_initializer=init_param)

      with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        output_tensor_values = sess.run(output_tensor)
      self.assertAllClose(output_tensor_values, expected_outputs)

  def testWrongInterpolationTypeExpectError(self):
    with self.assertRaises(ValueError):
      self._testLatticeLayerEvaluation(
          interpolation_type='wrong',
          output_dim=2,
          lattice_sizes=[2, 2],
          inputs=[[0.5, 0.5]],
          parameters=[[1.0, 2.0], [3.0, 4.0]],
          expected_outputs=[[2.5]])

  def testHypercubeEvaluation(self):
    inputs = [[-1.0, 0.0], [0.0, 0.0], [0.1, 0.9], [0.3, 1.1], [1.5, 2.0],
              [1.6, 3.0]]
    parameters = [[0.0, 0.1, 1.1, 2.3, 3.1, 4.2],
                  [5.1, 2.11, 1.11, 3.21, -1.02, -2.2]]
    expected_outputs = [[0.0, 5.1], [0.0, 5.1], [1.099, 1.6681],
                        [1.657, 1.4286], [4.2, -2.2], [4.2, -2.2]]
    self._testLatticeLayerEvaluation(
        interpolation_type='hypercube',
        output_dim=2,
        lattice_sizes=[2, 3],
        inputs=inputs,
        parameters=parameters,
        expected_outputs=expected_outputs)

  def testSimplexEvaluation(self):
    inputs = [[-1.0, 0.0], [0.0, 0.0], [0.1, 0.9], [0.3, 1.1], [1.5, 2.0],
              [1.6, 3.0]]
    parameters = [[0.0, 0.1, 1.1, 2.3, 3.1, 4.2],
                  [5.1, 2.11, 1.11, 3.21, -1.02, -2.2]]
    expected_outputs = [[0.0, 5.1], [0.0, 5.1], [1.11, 1.719], [1.65, 1.199],
                        [4.2, -2.2], [4.2, -2.2]]
    self._testLatticeLayerEvaluation(
        interpolation_type='simplex',
        output_dim=2,
        lattice_sizes=[2, 3],
        inputs=inputs,
        parameters=parameters,
        expected_outputs=expected_outputs)

  def testHypercubeEvaluationWithLinearParam(self):
    lattice_sizes = [2, 3]
    output_dim = 2
    inputs = [[0.0, 0.0], [0.1, 0.9], [0.3, 1.1], [1.5, 2.0]]
    # This parameter works as a linear function
    #   f(x1, x2) == 1/2 * (x1 + x2) - 0.75
    parameters = lattice_layers.lattice_param_as_linear(
        lattice_sizes=lattice_sizes, linear_weights=[1.0, 2.0], output_dim=2)
    expected_outputs = [[-0.75, -0.75], [-0.25, -0.25], [-0.05, -0.05],
                        [0.75, 0.75]]
    self._testLatticeLayerEvaluation(
        interpolation_type='hypercube',
        output_dim=output_dim,
        lattice_sizes=lattice_sizes,
        inputs=inputs,
        parameters=parameters,
        expected_outputs=expected_outputs)

  def testSimplexEvaluationWithLinearParam(self):
    lattice_sizes = [2, 3]
    output_dim = 2
    inputs = [[0.0, 0.0], [0.1, 0.9], [0.3, 1.1], [1.5, 2.0]]
    # This parameter works as linear functions
    #   f(x1, x2) = [0.5 * (x1 + x2) - 0.75, x1 + x2 - 1.5]
    parameters = lattice_layers.lattice_param_as_linear(
        lattice_sizes=lattice_sizes,
        output_dim=2,
        linear_weights=[[1.0, 2.0], [2.0, 4.0]])
    expected_outputs = [[-0.75, -1.5], [-0.25, -0.5], [-0.05, -0.1],
                        [0.75, 1.5]]
    self._testLatticeLayerEvaluation(
        interpolation_type='simplex',
        output_dim=output_dim,
        lattice_sizes=lattice_sizes,
        inputs=inputs,
        parameters=parameters,
        expected_outputs=expected_outputs)

  def testHypercubeNoRegularizationExpectsNone(self):
    lattice_sizes = [2, 3]
    with ops.Graph().as_default():
      input_tensor = array_ops.placeholder(
          shape=[None, 2], dtype=dtypes.float32)
      (_, _, _, regularization) = lattice_layers.lattice_layer(
          input_tensor,
          lattice_sizes=lattice_sizes,
          output_dim=1,
          interpolation_type='hypercube')
      self.assertEqual(regularization, None)

  def testSimplexNoRegularizationExpectsNone(self):
    lattice_sizes = [2, 3]
    with ops.Graph().as_default():
      input_tensor = array_ops.placeholder(
          shape=[None, 2], dtype=dtypes.float32)
      (_, _, _, regularization) = lattice_layers.lattice_layer(
          input_tensor,
          lattice_sizes=lattice_sizes,
          output_dim=1,
          interpolation_type='simplex')
      self.assertEqual(regularization, None)

  def testHypercubeRegularization(self):
    lattice_sizes = [2, 3]
    parameters = [[0.0, 0.1, 1.1, 2.3, 3.1, 4.2],
                  [5.1, 2.11, 1.11, 3.21, -1.02, -2.2]]
    output_dim = 2
    with ops.Graph().as_default():
      input_tensor = array_ops.placeholder(
          shape=[None, 2], dtype=dtypes.float32)
      init_param = array_ops.constant(parameters, dtype=dtypes.float32)
      (_, _, _, regularization) = lattice_layers.lattice_layer(
          input_tensor,
          lattice_sizes=lattice_sizes,
          output_dim=output_dim,
          interpolation_type='hypercube',
          l1_reg=0.1,
          l2_reg=0.1,
          l1_torsion_reg=0.1,
          l2_torsion_reg=0.1,
          l1_laplacian_reg=[0.1, 0.1],
          l2_laplacian_reg=[0.1, 0.1],
          lattice_initializer=init_param)

      with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        self.assertAlmostEqual(26.514278, sess.run(regularization), delta=1e-5)

  def testSimplexRegularization(self):
    lattice_sizes = [2, 3]
    parameters = [[0.0, 0.1, 1.1, 2.3, 3.1, 4.2],
                  [5.1, 2.11, 1.11, 3.21, -1.02, -2.2]]
    output_dim = 2
    with ops.Graph().as_default():
      input_tensor = array_ops.placeholder(
          shape=[None, 2], dtype=dtypes.float32)
      init_param = array_ops.constant(parameters, dtype=dtypes.float32)
      (_, _, _, regularization) = lattice_layers.lattice_layer(
          input_tensor,
          lattice_sizes=lattice_sizes,
          output_dim=output_dim,
          interpolation_type='simplex',
          l1_reg=0.1,
          l2_reg=0.1,
          l1_torsion_reg=0.1,
          l2_torsion_reg=0.1,
          l1_laplacian_reg=[0.1, 0.1],
          l2_laplacian_reg=[0.1, 0.1],
          lattice_initializer=init_param)

      with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        self.assertAlmostEqual(26.514278, sess.run(regularization), delta=1e-5)

  def _testLatticeLayerProjection(self, interpolation_type, lattice_sizes,
                                  output_dim, output_min, output_max,
                                  is_monotone, parameters,
                                  expected_projected_parameters):
    """Test monotonicity projection of lattice layers."""
    with ops.Graph().as_default():
      input_tensor = array_ops.zeros(
          [1, len(lattice_sizes)], dtype=dtypes.float32)
      (_, param_tensor, projection_op, _) = lattice_layers.lattice_layer(
          input_tensor,
          lattice_sizes=lattice_sizes,
          is_monotone=is_monotone,
          output_dim=output_dim,
          output_min=output_min,
          output_max=output_max,
          interpolation_type=interpolation_type)
      with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        sess.run(
            state_ops.assign(param_tensor,
                             array_ops.constant(
                                 parameters, dtype=dtypes.float32)))
        sess.run(projection_op)
        param_tensor_values = param_tensor.eval()

      self.assertAllClose(
          param_tensor_values, expected_projected_parameters, atol=1e-4)

  def testProjectionWithNonMonotonicHypercube(self):
    parameters = [[0.0, 0.1, 1.1, 2.3, 3.1, 4.2],
                  [5.1, 2.11, 1.11, 3.21, -1.02, -2.2]]
    expected_projected_parameters = parameters
    self._testLatticeLayerProjection(
        interpolation_type='hypercube',
        is_monotone=[False, False],
        output_dim=2,
        output_min=None,
        output_max=None,
        lattice_sizes=[2, 3],
        parameters=parameters,
        expected_projected_parameters=expected_projected_parameters)

  def testProjectionWithNonMonotonicSimplex(self):
    parameters = [[0.0, 0.1, 1.1, 2.3, 3.1, 4.2],
                  [5.1, 2.11, 1.11, 3.21, -1.02, -2.2]]
    expected_projected_parameters = parameters
    self._testLatticeLayerProjection(
        interpolation_type='simplex',
        is_monotone=[False, False],
        output_dim=2,
        output_min=None,
        output_max=None,
        lattice_sizes=[2, 3],
        parameters=parameters,
        expected_projected_parameters=expected_projected_parameters)

  def testProjectionWithFullMonotonicHypercube(self):
    parameters = [[0.0, 0.1, 1.1, 2.3, 3.1, 4.2],
                  [5.1, 2.11, 1.11, 3.21, -1.02, -2.2]]
    expected_projected_parameters = [[0, 0.1, 1.1, 2.3, 3.1, 4.2],
                                     [1.385, 1.385, 1.385, 1.385, 1.385, 1.385]]
    self._testLatticeLayerProjection(
        interpolation_type='hypercube',
        is_monotone=[True, True],
        output_dim=2,
        output_min=None,
        output_max=None,
        lattice_sizes=[2, 3],
        parameters=parameters,
        expected_projected_parameters=expected_projected_parameters)

  def testProjectionWithFullMonotonicSimplex(self):
    parameters = [[0.0, 0.1, 1.1, 2.3, 3.1, 4.2],
                  [5.1, 2.11, 1.11, 3.21, -1.02, -2.2]]
    expected_projected_parameters = [[0, 0.1, 1.1, 2.3, 3.1, 4.2],
                                     [1.385, 1.385, 1.385, 1.385, 1.385, 1.385]]
    self._testLatticeLayerProjection(
        interpolation_type='simplex',
        is_monotone=[True, True],
        output_dim=2,
        output_min=None,
        output_max=None,
        lattice_sizes=[2, 3],
        parameters=parameters,
        expected_projected_parameters=expected_projected_parameters)

  def testProjectionWithBoundedFullMonotonicHypercube(self):
    parameters = [[0.0, 0.1, 1.1, 2.3, 3.1, 4.2],
                  [5.1, 2.11, 1.11, 3.21, -1.02, -2.2]]
    expected_projected_parameters = [[0.3, 0.3, 1.1, 2.3, 3.0, 3.0],
                                     [1.385, 1.385, 1.385, 1.385, 1.385, 1.385]]
    self._testLatticeLayerProjection(
        interpolation_type='hypercube',
        is_monotone=[True, True],
        output_dim=2,
        output_min=0.3,
        output_max=3.0,
        lattice_sizes=[2, 3],
        parameters=parameters,
        expected_projected_parameters=expected_projected_parameters)


class EnsembleLatticesLayersTestCase(test.TestCase):

  def _testEnsembleLatticesLayerEvaluation(
      self, interpolation_type, lattice_sizes, structure, output_dim, inputs,
      parameters, expected_outputs_list):
    """Test evaluation of ensemble lattices layers."""
    with ops.Graph().as_default():
      input_tensor = array_ops.constant(inputs, dtype=dtypes.float32)
      init_params = [
          array_ops.constant(param, dtype=dtypes.float32)
          for param in parameters
      ]
      (output_tensor_lists, _, _, _) = lattice_layers.ensemble_lattices_layer(
          input_tensor,
          lattice_sizes=lattice_sizes,
          structure_indices=structure,
          output_dim=output_dim,
          interpolation_type=interpolation_type,
          lattice_initializers=init_params)
      self.assertEqual(len(output_tensor_lists), len(structure))
      with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        output_values_list = sess.run(output_tensor_lists)
      self.assertAllClose(output_values_list, expected_outputs_list)

  def testHypercubeEvaluation(self):
    inputs = [[-1.0, 0.0], [0.0, 0.0], [0.1, 0.9], [0.3, 1.1], [1.5, 2.0],
              [1.6, 3.0]]
    structure = [[0], [1], [0, 1]]

    # Construct params.
    parameters = []
    # First one is 1d lattice with two outputs:
    #  output[0] = x[0], output[1] = 1-x[0].
    parameters.append([[0.0, 1.0], [1.0, 0.0]])
    # Second one is 1d lattice with two outputs:
    #   output[0] = x[1] for 1 <= x[1] <= 2, 0 otherwise
    #   output[1] = 1 - x[1] for 0 <= x[1] <= 1, 0 otherwise.
    parameters.append([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    # Third one is 2d lattice.
    parameters.append([[0.0, 0.1, 1.1, 2.3, 3.1, 4.2],
                       [5.1, 2.11, 1.11, 3.21, -1.02, -2.2]])

    # Construct expected outputs.
    expected_outputs = []
    # Expected outputs from the first lattice.
    expected_outputs.append([[0.0, 1.0], [0.0, 1.0], [0.1, 0.9], [0.3, 0.7],
                             [1.0, 0.0], [1.0, 0.0]])
    # Expected outputs from the second lattice.
    expected_outputs.append([[0.0, 1.0], [0.0, 1.0], [0.0, 0.1], [0.1, 0.0],
                             [1.0, 0.0], [1.0, 0.0]])
    # Expected outputs from the third lattice.
    expected_outputs.append([[0.0, 5.1], [0.0, 5.1], [1.099, 1.6681],
                             [1.657, 1.4286], [4.2, -2.2], [4.2, -2.2]])

    self._testEnsembleLatticesLayerEvaluation(
        interpolation_type='hypercube',
        structure=structure,
        output_dim=2,
        lattice_sizes=[2, 3],
        inputs=inputs,
        parameters=parameters,
        expected_outputs_list=expected_outputs)

  def testSimplexEvaluation(self):
    inputs = [[-1.0, 0.0], [0.0, 0.0], [0.1, 0.9], [0.3, 1.1], [1.5, 2.0],
              [1.6, 3.0]]
    structure = [[0], [1], [0, 1]]

    # Construct params.
    parameters = []
    # First one is 1d lattice with two outputs:
    #   output[0] = x[0], output[1] = 1 - x[0].
    parameters.append([[0.0, 1.0], [1.0, 0.0]])
    # Second one is 1d lattice with two outputs:
    #   output[0] = x[1] for 1 <= x[1] <= 2, 0 otherwise
    #   output[1] = 1 - x[1] for 0 <= x[1] <= 1, 0 otherwise.
    parameters.append([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    # Third one is 2d lattice with two outputs.
    parameters.append([[0.0, 0.1, 1.1, 2.3, 3.1, 4.2],
                       [5.1, 2.11, 1.11, 3.21, -1.02, -2.2]])

    # Construct expected outputs.
    expected_outputs = []
    # Expected outputs from the first lattice.
    expected_outputs.append([[0.0, 1.0], [0.0, 1.0], [0.1, 0.9], [0.3, 0.7],
                             [1.0, 0.0], [1.0, 0.0]])
    # Expected outputs from the second lattice.
    expected_outputs.append([[0.0, 1.0], [0.0, 1.0], [0.0, 0.1], [0.1, 0.0],
                             [1.0, 0.0], [1.0, 0.0]])
    # Expected outputs from the third lattice.
    expected_outputs.append([[0.0, 5.1], [0.0, 5.1], [1.11, 1.719],
                             [1.65, 1.199], [4.2, -2.2], [4.2, -2.2]])

    self._testEnsembleLatticesLayerEvaluation(
        interpolation_type='simplex',
        structure=structure,
        output_dim=2,
        lattice_sizes=[2, 3],
        inputs=inputs,
        parameters=parameters,
        expected_outputs_list=expected_outputs)

  def testHypercubeRegularization(self):
    lattice_sizes = [2, 3]
    structure = [[0], [1], [0, 1]]
    # Construct params.
    parameters = []
    parameters.append([[0.0, 1.0], [1.0, 0.0]])
    parameters.append([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    parameters.append([[0.0, 0.1, 1.1, 2.3, 3.1, 4.2],
                       [5.1, 2.11, 1.11, 3.21, -1.02, -2.2]])
    output_dim = 2
    with ops.Graph().as_default():
      input_tensor = array_ops.placeholder(
          shape=[None, 2], dtype=dtypes.float32)
      init_params = [
          array_ops.constant(param, dtype=dtypes.float32)
          for param in parameters
      ]
      (_, _, _, regularization) = lattice_layers.ensemble_lattices_layer(
          input_tensor,
          lattice_sizes=lattice_sizes,
          structure_indices=structure,
          output_dim=output_dim,
          interpolation_type='hypercube',
          l1_reg=0.1,
          l2_reg=0.1,
          l1_torsion_reg=0.1,
          l2_torsion_reg=0.1,
          l1_laplacian_reg=[0.1, 0.1],
          l2_laplacian_reg=[0.1, 0.1],
          lattice_initializers=init_params)

      with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        self.assertAlmostEqual(28.114279, sess.run(regularization), delta=1e-5)

  def testSimplexRegularization(self):
    lattice_sizes = [2, 3]
    structure = [[0], [1], [0, 1]]
    # Construct params.
    parameters = []
    parameters.append([[0.0, 1.0], [1.0, 0.0]])
    parameters.append([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    parameters.append([[0.0, 0.1, 1.1, 2.3, 3.1, 4.2],
                       [5.1, 2.11, 1.11, 3.21, -1.02, -2.2]])
    output_dim = 2
    with ops.Graph().as_default():
      input_tensor = array_ops.placeholder(
          shape=[None, 2], dtype=dtypes.float32)
      init_params = [
          array_ops.constant(param, dtype=dtypes.float32)
          for param in parameters
      ]
      (_, _, _, regularization) = lattice_layers.ensemble_lattices_layer(
          input_tensor,
          lattice_sizes=lattice_sizes,
          structure_indices=structure,
          output_dim=output_dim,
          interpolation_type='simplex',
          l1_reg=0.1,
          l2_reg=0.1,
          l1_torsion_reg=0.1,
          l2_torsion_reg=0.1,
          l1_laplacian_reg=[0.1, 0.1],
          l2_laplacian_reg=[0.1, 0.1],
          lattice_initializers=init_params)

      with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        self.assertAlmostEqual(28.114279, sess.run(regularization), delta=1e-5)

  def _testEnsembleLatticesLayerProjection(
      self, interpolation_type, lattice_sizes, structure, output_dim,
      is_monotone, parameters, expected_projected_parameters):
    """Test monotonicity projection of lattice layers."""
    with ops.Graph().as_default():
      input_tensor = array_ops.zeros(
          [1, len(lattice_sizes)], dtype=dtypes.float32)
      (_, param_tensors, proj, _) = lattice_layers.ensemble_lattices_layer(
          input_tensor,
          structure_indices=structure,
          lattice_sizes=lattice_sizes,
          is_monotone=is_monotone,
          output_dim=output_dim,
          lattice_initializers=parameters,
          interpolation_type=interpolation_type)
      with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        # Check initialization.
        param_tensor_values = sess.run(param_tensors)
        self.assertEqual(len(param_tensor_values), len(parameters))
        for (param_value, expected_value) in zip(param_tensor_values,
                                                 parameters):
          self.assertAllClose(param_value, expected_value, atol=1e-4)
        # Check projection.
        sess.run(proj)
        param_tensor_values = sess.run(param_tensors)
        self.assertEqual(
            len(param_tensor_values), len(expected_projected_parameters))
        for (param_value, expected_value) in zip(param_tensor_values,
                                                 expected_projected_parameters):
          self.assertAllClose(param_value, expected_value, atol=1e-4)

  def testProjectionWithNonMonotonicHypercube(self):
    structure = [[0], [0, 1]]
    params = []
    params.append([[0.0, 1.0], [1.0, -1.0]])
    params.append([[0.0, 0.1, 1.1, 2.3, 3.1, 4.2],
                   [5.1, 2.11, 1.11, 3.21, -1.02, -2.2]])
    expected_proj_params = params
    self._testEnsembleLatticesLayerProjection(
        interpolation_type='hypercube',
        structure=structure,
        is_monotone=[False, False],
        output_dim=2,
        lattice_sizes=[2, 3],
        parameters=params,
        expected_projected_parameters=expected_proj_params)

  def testProjectionWithNonMonotonicSimplex(self):
    structure = [[0], [0, 1]]
    params = []
    params.append([[0.0, 1.0], [1.0, -1.0]])
    params.append([[0.0, 0.1, 1.1, 2.3, 3.1, 4.2],
                   [5.1, 2.11, 1.11, 3.21, -1.02, -2.2]])
    expected_proj_params = params
    self._testEnsembleLatticesLayerProjection(
        interpolation_type='hypercube',
        structure=structure,
        is_monotone=[False, False],
        output_dim=2,
        lattice_sizes=[2, 3],
        parameters=params,
        expected_projected_parameters=expected_proj_params)

  def testProjectionWithFullMonotonicHypercube(self):
    structure = [[0], [0, 1]]
    params = []
    params.append([[0.0, -10.0], [0.0, 5.0]])
    params.append([[0.0, 0.1, 1.1, 2.3, 3.1, 4.2],
                   [5.1, 2.11, 1.11, 3.21, -1.02, -2.2]])
    expected_proj_params = []
    expected_proj_params.append([[-5.0, -5.0], [0.0, 5.0]])
    expected_proj_params.append([[0, 0.1, 1.1, 2.3, 3.1, 4.2],
                                 [1.385, 1.385, 1.385, 1.385, 1.385, 1.385]])
    self._testEnsembleLatticesLayerProjection(
        interpolation_type='hypercube',
        structure=structure,
        is_monotone=[True, True],
        output_dim=2,
        lattice_sizes=[2, 3],
        parameters=params,
        expected_projected_parameters=expected_proj_params)

  def testProjectionWithFullMonotonicSimplex(self):
    structure = [[0], [0, 1]]
    params = []
    params.append([[0.0, -10.0], [0.0, 5.0]])
    params.append([[0.0, 0.1, 1.1, 2.3, 3.1, 4.2],
                   [5.1, 2.11, 1.11, 3.21, -1.02, -2.2]])
    expected_proj_params = []
    expected_proj_params.append([[-5.0, -5.0], [0.0, 5.0]])
    expected_proj_params.append([[0, 0.1, 1.1, 2.3, 3.1, 4.2],
                                 [1.385, 1.385, 1.385, 1.385, 1.385, 1.385]])
    self._testEnsembleLatticesLayerProjection(
        interpolation_type='simplex',
        structure=structure,
        is_monotone=True,
        output_dim=2,
        lattice_sizes=[2, 3],
        parameters=params,
        expected_projected_parameters=expected_proj_params)


if __name__ == '__main__':
  test.main()
