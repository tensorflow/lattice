# Copyright 2018 The TensorFlow Lattice Authors.
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
"""Trains a small (2 inputs, single lattice) on toy data and visualizes it."""
from __future__ import print_function

import tempfile
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
import tensorflow_lattice as tfl

np.random.seed(1)

_FEATURE_KEYPOINTS = 'tfl_calibrated_etl/pwl_calibration/X_{}_keypoints_'
_EMBED_KEYPOINTS = 'tfl_calibrated_etl/non_monotonic_lattices/'
_EMBED_KEYPOINTS += 'pwl_calibration/signal_{}_keypoints_'
_LATTICE_PARAMS = 'tfl_calibrated_etl/non_monotonic_lattices/lattice_{}/'
_LATTICE_PARAMS += 'hypercube_lattice_parameters'


def annulus_data(n_points, r_0, r_1):
  """Creates toy dataset in quadrant I with a quarter annulus.

  Args:
      n_points: (int) number of points
      r_0: (float) inner bounding radius
      r_1: (float) outer bounding radius

  Returns:
      x: (np.Array) covariates
      y: (np.Array) labels
  """
  x = np.random.random(size=(n_points, 2))
  r = (x**2).sum(1)**.5
  y = (r_0 < r) & (r < r_1)
  return x, y.astype(int)


def fit_model(x,
              y,
              lattice_size=5,
              non_monotonic_num_lattices=1,
              non_monotonic_lattice_rank=1):
  """Fits a single 1D lattice to the provided data.

  Args:
      x: covariates
      y: labels
      lattice_size: (int, optional) Number of knots in each lattice dimension,
        total knots is lattice_size^lattice_rank, for each lattice
      non_monotonic_num_lattices: (int, optional)
      non_monotonic_lattice_rank: (int, optional) number of inputs to each

  Returns:
      etl_estimator: fitted TF Estimator
  """
  # Hyperparameters.
  num_keypoints = 100
  hparams = tfl.CalibratedEtlHParams(
      non_monotonic_lattice_rank=non_monotonic_lattice_rank,
      non_monotonic_num_lattices=non_monotonic_num_lattices,
      non_monotonic_lattice_size=lattice_size,
      num_keypoints=num_keypoints,
      learning_rate=0.007,
      linear_embedding_calibration_num_keypoints=100)

  # Estimator.
  feature_columns = [
      tf.feature_column.numeric_column('X_0'),
      tf.feature_column.numeric_column('X_1'),
  ]

  # Training is sensitive to initialization
  config = tf.estimator.RunConfig(tf_random_seed=1)
  def keypoints_config():
    return tfl.uniform_keypoints_for_signal(
        num_keypoints,
        input_min=0.0,
        input_max=x.max(),
        output_min=0.0,
        output_max=lattice_size - 1
    )
  etl_estimator = tfl.calibrated_etl_classifier(
      feature_columns=feature_columns,
      hparams=hparams,
      keypoints_initializers_fn=keypoints_config,
      config=config
  )

  # Input function.
  input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x={
          'X_0': x[:, 0],
          'X_1': x[:, 1]
      },
      y=y.flatten(),
      batch_size=10000,
      num_epochs=100,
      shuffle=False)

  # Train!
  etl_estimator.train(input_fn=input_fn)

  # Evaluate
  eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x={
          'X_0': x[:, 0],
          'X_1': x[:, 1]
      },
      y=y.flatten(),
      batch_size=10000,
      num_epochs=1,
      shuffle=False)
  print(etl_estimator.evaluate(input_fn=eval_input_fn))

  return etl_estimator


def _get_calibration_params(estimator, dim, weight_key, prefix):
  """Helps extract calibration parameters from TFL graph."""
  input_key = '{}_keypoints_inputs'.format(prefix)
  output_key = '{}_keypoints_outputs'.format(prefix)
  calibrator_key = '{}_calibrators'.format(prefix)

  params = {}
  params[input_key], params[output_key], params[calibrator_key] = [], [], []
  for i in xrange(dim):
    params[input_key].append(
        estimator.get_variable_value(weight_key.format(i) + 'inputs'))
    params[output_key].append(
        estimator.get_variable_value(weight_key.format(i) + 'outputs'))
    params[calibrator_key].append(
        scipy.interpolate.interp1d(
            params[input_key][-1],
            params[output_key][-1],
            fill_value='extrapolate'))
  return params


def _get_parameters(etl_estimator):
  """Extracts all parameters necessary to evaluate an ETL from estimator."""
  params = {}
  params['embed_weighting'] = etl_estimator.get_variable_value(
      'tfl_calibrated_etl/linear_embedding/split_non_monotone/monotone_linear'
      '/weight')
  params['embed_bias'] = etl_estimator.get_variable_value(
      'tfl_calibrated_etl/linear_embedding/split_non_monotone/monotone_linear'
      '/bias')
  params['final_bias'] = etl_estimator.get_variable_value(
      'tfl_calibrated_etl/ensemble_average/ensemble_bias')
  params['n_embed'] = params['embed_weighting'].shape[0]
  params['n_feature'] = params['embed_weighting'].shape[1]

  params.update(
      _get_calibration_params(etl_estimator, params['n_feature'],
                              _FEATURE_KEYPOINTS, 'feature'))

  params.update(
      _get_calibration_params(
          etl_estimator,
          params['n_embed'],
          _EMBED_KEYPOINTS,
          'embed',
      ))

  n, ws = 0, []
  while _LATTICE_PARAMS.format(n) in etl_estimator.get_variable_names():
    ws.append(etl_estimator.get_variable_value(_LATTICE_PARAMS.format(n)))
    n += 1
  params['lattice_knots'] = np.vstack(ws)

  return params


def _apply_callibration(x, calibrators):
  x_ = x.copy()
  for n in xrange(x.shape[1]):
    x_[:, n] = calibrators[n](x[:, n])
  return x_


def _compress_0_1(x):
  return (x - x.min()) / (x.max() - x.min())


def plot_all(etl_estimator, x, y, save_dir):
  """Makes visualizations of ETL Estimator.

  Args:
      etl_estimator: (TF ETL Estimator)
      x: (np.Array) inputs
      y: (np.Array) labels, in [0, 1]
      save_dir: (string) directory for saving visualizations
  """
  params = _get_parameters(etl_estimator)

  x_cal = _apply_callibration(x, params['feature_calibrators'])
  x_cal_emb = x_cal.dot(params['embed_weighting'].T) + params['embed_bias']
  x_cal_emb_cal = _apply_callibration(x_cal_emb, params['embed_calibrators'])
  x_cal_emb_cal_lat = np.zeros_like(x_cal_emb_cal)
  for i in xrange(params['lattice_knots'].shape[0]):
    interpolator = scipy.interpolate.interp1d(
        range(params['lattice_knots'].shape[1]),
        params['lattice_knots'][i],
        fill_value='extrapolate')
    x_cal_emb_cal_lat[:, i] = interpolator(x_cal_emb_cal[:, i])

  predictions = (x_cal_emb_cal_lat.mean(1) + params['final_bias'] >
                 .5).astype(int)

  plt.figure()
  plt.title('Input Points Colored By Correct Classification')
  plt.scatter(x[:10000, 0], x[:10000, 1], c=y[:10000], alpha=.3)
  plt.savefig(save_dir + '/labeled.png')

  for i, (inputs, outputs) in enumerate(
      zip(params['feature_keypoints_inputs'],
          params['feature_keypoints_outputs'])):
    plt.figure()
    plt.title('Calibration Keypoints For Input Column Number {}'.format(i))
    plt.scatter(inputs, outputs)
    plt.savefig(save_dir + '/feature_cal_{}.png'.format(i))

  for i, (inputs, outputs) in enumerate(
      zip(params['embed_keypoints_inputs'], params['embed_keypoints_outputs'])):
    plt.figure()
    plt.title('Calibration Keypoints For Emedding Number {}'.format(i))
    plt.scatter(inputs, outputs)
    plt.savefig(save_dir + '/embed_cal_{}.png'.format(i))

  for i in xrange(params['lattice_knots'].shape[0]):
    plt.figure()
    plt.title('Lattice knots for lattice number {}'.format(i))
    plt.plot(
        range(params['lattice_knots'].shape[1]), params['lattice_knots'][i])
    plt.savefig(save_dir + '/lattice_{}.png'.format(i))

  plt.figure()
  plt.title('Input Points After Calibration, Colored By Correct Classification')
  plt.scatter(x_cal[:10000, 0], x_cal[:10000, 1], c=y[:10000], alpha=.3)
  plt.savefig(save_dir + '/calibrated.png')

  plt.figure()
  plt.title('Input Points Colored By Value'
            ' After Calibration and linear transformation')
  plt.scatter(
      x[:10000, 0],
      x[:10000, 1],
      c=_compress_0_1(x_cal_emb[:10000, 0]),
      alpha=.3)
  plt.savefig(save_dir + '/embed_colored.png')

  plt.figure()
  plt.title('Input Points Colored By Value After Calibration,'
            '\n Linear Transformation, Second Calibration')
  plt.scatter(
      x[:10000, 0],
      x[:10000, 1],
      c=_compress_0_1(x_cal_emb_cal[:10000, 0]),
      alpha=.3)
  plt.savefig(save_dir + '/embed_calibrated_colored.png')

  plt.figure()
  plt.title('Input Points Colored by Value After Calibration,'
            '\nlinear transformation, second calibration, and 1D lattice')
  plt.scatter(
      x[:10000, 0],
      x[:10000, 1],
      c=_compress_0_1(x_cal_emb_cal_lat[:10000, 0]),
      alpha=.3)
  plt.savefig(save_dir + '/lattice_colored.png')

  plt.figure()
  plt.title('Predictions')
  plt.scatter(
      x[:10000, 0],
      x[:10000, 1],
      c=_compress_0_1(predictions)[:10000],
      alpha=.3)
  plt.savefig(save_dir + '/predictions.png')


def main():
  # Make data
  x, y = annulus_data(300000, .5, .8)

  # Train model
  etl_estimator = fit_model(x, y)

  # Visualize
  temp_dir = tempfile.mkdtemp()
  print('Saving figures to {}'.format(temp_dir))
  plot_all(etl_estimator, x, y, temp_dir)


if __name__ == '__main__':
  main()
