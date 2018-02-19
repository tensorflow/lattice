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
"""Tests for TensorFlow Lattice's keypoints_initialization module."""

import math
import os

# Dependency imports

from tensorflow_lattice.python.lib import keypoints_initialization

import numpy as np

from tensorflow.python.estimator.inputs import numpy_io
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class KeypointsInitializationTestCase(test.TestCase):

  def setUp(self):
    super(KeypointsInitializationTestCase, self).setUp()

  def testMaterializeLocally(self):
    num_examples = 100
    x = np.random.uniform(0.0, 1.0, size=num_examples)

    # Read to the end of a number of epochs.
    input_fn = numpy_io.numpy_input_fn(
        x={'x': x}, batch_size=13, num_epochs=1, shuffle=False)
    results = keypoints_initialization._materialize_locally(
        tensors=input_fn(), num_steps=None)
    self.assertEqual(len(results['x']), num_examples)
    input_fn = numpy_io.numpy_input_fn(
        x={'x': x}, batch_size=13, num_epochs=2, shuffle=False)
    results = keypoints_initialization._materialize_locally(
        tensors=input_fn(), num_steps=None)
    self.assertEqual(len(results['x']), 2 * num_examples)

    # Read a certain number of steps: just enough to read all data (last
    # batch will only be partially fulfilled).
    input_fn = numpy_io.numpy_input_fn(
        x={'x': x}, batch_size=13, num_epochs=1, shuffle=False)
    results = keypoints_initialization._materialize_locally(
        tensors=input_fn(), num_steps=1)
    self.assertEqual(len(results['x']), 13)

    input_fn = numpy_io.numpy_input_fn(
        x={'x': x}, batch_size=13, num_epochs=1, shuffle=False)
    results = keypoints_initialization._materialize_locally(
        tensors=input_fn(), num_steps=8)
    self.assertEqual(len(results['x']), num_examples)

    # Try to read beyond end of input, with num_steps set.
    input_fn = numpy_io.numpy_input_fn(
        x={'x': x}, batch_size=13, num_epochs=1, shuffle=False)
    with self.assertRaises(errors.OutOfRangeError):
      results = keypoints_initialization._materialize_locally(
          tensors=input_fn(), num_steps=100)

    # Try to read beyond safety limit.
    input_fn = numpy_io.numpy_input_fn(
        x={'x': x}, batch_size=13, num_epochs=None, shuffle=False)
    with self.assertRaises(ValueError):
      results = keypoints_initialization._materialize_locally(
          tensors=input_fn(), num_steps=None, safety_size=1000)

  def _BuildInputs(self, x0, x1, x2):
    """Returns input_fn, feature_names and feature_columns."""

    def _input_fn():
      return ({
          'x0': array_ops.constant(x0, dtype=dtypes.float32),
          'x1': array_ops.constant(x1, dtype=dtypes.float32),
          'x2': array_ops.constant(x2, dtype=dtypes.float32),
      }, None)

    feature_names = ['x0', 'x1', 'x2']
    feature_columns = set(
        [feature_column_lib.numeric_column(key=fn) for fn in feature_names])
    return _input_fn, feature_names, feature_columns

  def _CheckSaveQuantilesForKeypoints(self, name, num_examples, num_steps, x0,
                                      x1, x2, use_feature_columns, override):
    input_fn, feature_names, feature_columns = self._BuildInputs(x0, x1, x2)
    save_dir = os.path.join(self.get_temp_dir(), name)
    keypoints_initialization.save_quantiles_for_keypoints(
        input_fn,
        save_dir,
        feature_columns=(feature_columns if use_feature_columns else None),
        num_quantiles=5,
        override=override)

    # Check by reading files directly.
    subdir = os.path.join(save_dir,
                          keypoints_initialization._QUANTILES_SUBDIRECTORY)
    quantiles_x0 = keypoints_initialization._load_quantiles(subdir, 'x0')
    quantiles_x1 = keypoints_initialization._load_quantiles(subdir, 'x1')
    quantiles_x2 = keypoints_initialization._load_quantiles(subdir, 'x2')
    self.assertAllClose(
        quantiles_x0, [0, 2.5**2, 5.**2, 7.5**2, 100.], atol=0.2)
    self.assertAllClose(
        quantiles_x1, [1.,
                       math.pow(10., 0.5), 10.0,
                       math.pow(10., 1.5), 100.],
        atol=0.2)
    # x2 should start with [0,0,...] and end in [..., 1, 1], the middle value
    # can be either 0 or 1.
    self.assertAllClose(quantiles_x2[0:2], [0., 0.], atol=1e-3)
    self.assertAllClose(quantiles_x2[-2:], [1., 1.], atol=1e-3)

    # New graph is needed because default graph is changed by save
    # keypoints, and self.test_session() will by default try to reuse a cached
    # session, with a different graph.
    with ops.Graph().as_default() as g:
      # Check by using load_keypoints_from_quantiles.
      keypoints_init = keypoints_initialization.load_keypoints_from_quantiles(
          feature_names,
          save_dir,
          3,
          output_min={'x0': 0.,
                      'x1': 1.,
                      'x2': 7.},
          output_max={'x0': 1.,
                      'x1': 10.,
                      'x2': 13.}
      )
      with self.test_session(graph=g) as sess:
        keypoints_init = sess.run(keypoints_init)
    self.assertAllClose(keypoints_init['x0'][0], [0, 5.**2, 100.], atol=0.2)
    self.assertAllClose(keypoints_init['x0'][1], [0., 0.5, 1.])
    self.assertAllClose(keypoints_init['x1'][0], [1., 10.0, 100.], atol=0.2)
    self.assertAllClose(keypoints_init['x1'][1], [1., 5.5, 10.])

    # Notice x2 only has 2 unique values, so it should have lowered the
    # num_keypoints to 2.
    self.assertAllClose([0., 1.0], keypoints_init['x2'][0], atol=1e-3)
    self.assertAllClose([7., 13.0], keypoints_init['x2'][1], atol=1e-3)

    # Check that load_keypoints_from_quantiles don't generate anything
    # if num_keypoints is 0 or unset.
    with ops.Graph().as_default() as g:
      # Check by using load_keypoints_from_quantiles.
      keypoints_init = keypoints_initialization.load_keypoints_from_quantiles(
          feature_names,
          save_dir, {'x0': 3,
                     'x2': 3,
                     'x1': 0},
          output_min={'x0': 0.,
                      'x1': 1.,
                      'x2': 7.},
          output_max={'x0': 1.,
                      'x1': 10.,
                      'x2': 13.})
      with self.test_session(graph=g) as sess:
        keypoints_init = sess.run(keypoints_init)
    self.assertTrue('x0' in keypoints_init)
    self.assertTrue('x2' in keypoints_init)
    self.assertTrue('x1' not in keypoints_init)

  def testSaveQuantilesForKeypoints(self):
    """Tests quantiles are being calculated correctly."""
    num_examples = 100000
    num_steps = num_examples / num_examples

    # Verify for randomized input: try with/without feature_columns.
    x0 = np.random.uniform(0.0, 10.0, size=num_examples)
    x0 = np.square(x0)
    x1 = np.random.uniform(0.0, 2.0, size=num_examples)
    x1 = np.power(10., x1)
    x2 = np.random.randint(0, 2, size=num_examples).astype(float)
    self._CheckSaveQuantilesForKeypoints(
        'save_quantiles_for_keypoints',
        num_examples,
        num_steps,
        x0,
        x1,
        x2,
        use_feature_columns=True,
        override=True)
    self._CheckSaveQuantilesForKeypoints(
        'save_quantiles_for_keypoints',
        num_examples,
        num_steps,
        x0,
        x1,
        x2,
        use_feature_columns=False,
        override=False)

    # Second change: since we are not overriding, it shouldn't regenerate the
    # results. So we provide "wrong data": if the quantiles are regenerated
    # the test will fail.
    x0 = np.linspace(0.0, 1.0, num_examples)
    x1 = np.linspace(0.0, 1.0, num_examples)
    x2 = np.array([2.] * num_examples)
    self._CheckSaveQuantilesForKeypoints(
        'save_quantiles_for_keypoints',
        num_examples,
        num_steps,
        x0,
        x1,
        x2,
        use_feature_columns=False,
        override=False)

    # Verify that things work on a non-randomized set: this will break
    # if not all input is being considered.
    x0 = np.linspace(0.0, 10.0, num_examples)
    x0 = np.square(x0)
    x1 = np.linspace(0.0, 2.0, num_examples)
    x1 = np.power(10., x1)
    x2 = np.array([0.] * int(num_examples / 2) + [1.] * int(num_examples / 2))
    self._CheckSaveQuantilesForKeypoints(
        'save_quantiles_for_keypoints',
        num_examples,
        num_steps,
        x0,
        x1,
        x2,
        use_feature_columns=False,
        override=True)

  def testQuantileInitWithReversedDict(self):
    num_examples = 100
    x0 = np.linspace(0.0, 10.0, num_examples)
    x1 = np.linspace(0.0, 10.0, num_examples)
    x2 = np.linspace(0.0, 1.0, num_examples)

    input_fn, feature_names, feature_columns = self._BuildInputs(x0, x1, x2)
    save_dir = os.path.join(self.get_temp_dir(), 'reversed_dict')
    keypoints_initialization.save_quantiles_for_keypoints(
        input_fn,
        save_dir,
        feature_columns=feature_columns,
        num_quantiles=100,
        override=True)
    reversed_dict = {'x0': False,
                     'x1': True,
                     'x2': False}

    with ops.Graph().as_default() as g:
      # Check by using load_keypoints_from_quantiles.
      keypoints_init = keypoints_initialization.load_keypoints_from_quantiles(
          feature_names,
          save_dir,
          num_keypoints=3,
          output_min={'x0': 0.,
                      'x1': 0.,
                      'x2': 0.},
          output_max={'x0': 1.,
                      'x1': 1.,
                      'x2': 1.},
          reversed_dict=reversed_dict
      )
      with self.test_session(graph=g) as sess:
        keypoints_init = sess.run(keypoints_init)

    self.assertAllClose(keypoints_init['x0'][0], [0.0, 5.0, 10.0], atol=0.1)
    self.assertAllClose(keypoints_init['x0'][1], [0.0, 0.5, 1.0], atol=0.01)
    self.assertAllClose(keypoints_init['x1'][0], [0.0, 5.0, 10.0], atol=0.1)
    self.assertAllClose(keypoints_init['x1'][1], [1.0, 0.5, 0.0], atol=0.01)
    self.assertAllClose(keypoints_init['x2'][0], [0.0, 0.5, 1.0], atol=0.01)
    self.assertAllClose(keypoints_init['x2'][1], [0.0, 0.5, 1.0], atol=0.01)

  def testQuantileInitWithExcludeInputValuesDict(self):
    num_examples = 10
    x0 = np.linspace(-1.0, 1.0, num_examples)
    x1 = np.linspace(0.0, 1.0, num_examples)
    x2 = np.linspace(0.0, 1.0, num_examples)

    input_fn, feature_names, feature_columns = self._BuildInputs(x0, x1, x2)
    save_dir = os.path.join(self.get_temp_dir(), 'exclude_input_values_dict')
    keypoints_initialization.save_quantiles_for_keypoints(
        input_fn,
        save_dir,
        feature_columns=feature_columns,
        num_quantiles=num_examples,
        override=True)

    with ops.Graph().as_default() as g:
      # Check by using load_keypoints_from_quantiles.
      keypoints_init = keypoints_initialization.load_keypoints_from_quantiles(
          feature_names,
          save_dir,
          num_keypoints=3,
          output_min={'x0': 0.,
                      'x1': 0.,
                      'x2': 0.},
          output_max={'x0': 1.,
                      'x1': 1.,
                      'x2': 1.},
          exclude_input_values_dict={'x0': [-1.0],
                                     'x1': [0.0, 0.5],
                                     'x2': []},
      )
      with self.test_session(graph=g) as sess:
        keypoints_init = sess.run(keypoints_init)

    self.assertAllClose(keypoints_init['x0'][0], [-0.778, 0.111, 1.0], atol=0.1)
    self.assertAllClose(keypoints_init['x0'][1], [0.0, 0.5, 1.0], atol=0.01)
    self.assertAllClose(keypoints_init['x1'][0], [0.111, 0.556, 1.0], atol=0.1)
    self.assertAllClose(keypoints_init['x1'][1], [0.0, 0.5, 1.0], atol=0.01)
    self.assertAllClose(keypoints_init['x2'][0], [0.0, 0.444, 1.0], atol=0.01)
    self.assertAllClose(keypoints_init['x2'][1], [0.0, 0.5, 1.0], atol=0.01)

  def testUniformKeypointsForSignal(self):
    # New graph is needed because default graph is changed by save
    # keypoints, and self.test_session() will by default try to reuse a cached
    # session, with a different graph.
    with ops.Graph().as_default() as g:
      keypoints_init = keypoints_initialization.uniform_keypoints_for_signal(
          num_keypoints=5,
          input_min=array_ops.constant(0.0, dtype=dtypes.float64),
          input_max=array_ops.constant(1.0, dtype=dtypes.float64),
          output_min=10,
          output_max=100,
          dtype=dtypes.float64)
      self.assertEqual(keypoints_init[0].dtype, dtypes.float64)
      self.assertEqual(keypoints_init[1].dtype, dtypes.float64)
      with self.test_session(graph=g) as sess:
        keypoints_init = sess.run(keypoints_init)
        self.assertAllClose(keypoints_init[0], [0., 0.25, 0.5, 0.75, 1.])
        self.assertAllClose(keypoints_init[1], [10., 32.5, 55., 77.5, 100.])


if __name__ == '__main__':
  test.main()
