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
"""Base estimator is tested with a simple linear model implementation."""
# Dependency imports
import six

from tensorflow_lattice.python.estimators import base
from tensorflow_lattice.python.lib import test_data

from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test


class _BaseLinear(base.Base):
  """Base class for BaseLinearClassifier and BaseLinearRegressor."""

  def __init__(self,
               n_classes,
               feature_columns=None,
               model_dir=None,
               hparams=None):
    """Construct LinearClassifier/Regressor."""
    super(_BaseLinear, self).__init__(
        n_classes=n_classes,
        feature_columns=feature_columns,
        model_dir=model_dir,
        hparams=hparams,
        name='linear')

  def prediction_builder(self, columns_to_tensors, mode, hparams, dtype):
    unstacked_inputs = []
    for tensor in six.itervalues(columns_to_tensors):
      if tensor.shape.ndims == 1:
        unstacked_inputs.append(tensor)
      elif tensor.shape.ndims == 2:
        unstacked_inputs.extend(array_ops.unstack(tensor, axis=1))
    input_tensor = array_ops.stack(unstacked_inputs, axis=1, name='stack')
    weights = variable_scope.get_variable(
        'weights',
        initializer=array_ops.zeros(
            shape=[len(unstacked_inputs), 1], dtype=dtype))
    prediction = array_ops.reshape(
        math_ops.tensordot(input_tensor, weights, axes=1, name='tensordot'),
        [-1, 1])
    # Add ridge regularizer.
    regularization = math_ops.reduce_sum(math_ops.square(weights))
    # Add a projection that forces the weight vector to be 0.
    projeciton_ops = [weights.assign_sub(weights)]
    return prediction, projeciton_ops, regularization


class BaseTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(BaseTest, self).setUp()
    self._test_data = test_data.TestData()

  def _TestRegressor(self, feature_columns, input_fn):
    estimator = _BaseLinear(n_classes=0, feature_columns=feature_columns)
    estimator.train(input_fn=input_fn)
    preds = [p['predictions'][0] for p in estimator.predict(input_fn=input_fn)]
    self.assertAllClose(preds, [0.0] * len(preds), 1e-7)

  def _TestCalssifier(self, feature_columns, input_fn):
    estimator = _BaseLinear(n_classes=2, feature_columns=feature_columns)
    estimator.train(input_fn=input_fn)
    preds = [p['logits'][0] for p in estimator.predict(input_fn=input_fn)]
    self.assertAllClose(preds, [0.0] * len(preds), 1e-7)

  def testBaseLinearRegressorTraining1D(self):
    feature_columns = [
        feature_column_lib.numeric_column('x'),
    ]
    self._TestRegressor(feature_columns, self._test_data.oned_input_fn())

  def testBaseLinearRegressorTraining3D(self):
    # Tests also a categorical feature with vocabulary list.
    feature_columns = [
        feature_column_lib.numeric_column('x0'),
        feature_column_lib.numeric_column('x1'),
        feature_column_lib.categorical_column_with_vocabulary_list(
            'x2', ['Y', 'N'])
    ]
    self._TestRegressor(feature_columns,
                        self._test_data.threed_input_fn(False, 1))

  def testBaseLinearRegressorTrainingMultiDimensionalFeature(self):
    feature_columns = [
        feature_column_lib.numeric_column('x', shape=(2,)),
    ]
    self._TestRegressor(feature_columns,
                        self._test_data.multid_feature_input_fn())

  def testBaseLinearClassifierTraining(self):
    feature_columns = [
        feature_column_lib.numeric_column('x0'),
        feature_column_lib.numeric_column('x1'),
    ]
    self._TestCalssifier(feature_columns,
                         self._test_data.twod_classificer_input_fn())


if __name__ == '__main__':
  test.main()
