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
"""A quick example of TensorFlow Lattice's calibrated RTL estimator."""
from __future__ import print_function
import sys
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_lattice as tfl


def _pixels(im):
  out = np.zeros((im.shape[0] * im.shape[1], 3))
  out[:, 0] = np.repeat(np.arange(im.shape[0]), im.shape[1])
  out[:, 1] = np.tile(np.arange(im.shape[1]), im.shape[0])
  out[:, 2] = im.ravel()
  return out


def _pixels_to_image(pixels):
  out = np.zeros((int(pixels[:, 0].max() + 1), int(pixels[:, 1].max() + 1)))
  out[pixels[:, 0].astype(int), pixels[:, 1].astype(int)] = pixels[:, 2]
  return out


def run_image(image_path, lattice_size=35):
  """Reads image and fits a 2D lattice to compress it."""
  im = plt.imread(image_path)[:, :, 2]
  im_pixels = _pixels(im)

  print('compression ratio is ', lattice_size**2 / float(im.size))

  # Hyperparameters.
  num_keypoints = 2
  hparams = tfl.CalibratedRtlHParams(
      num_keypoints=num_keypoints,
      num_lattices=1,
      lattice_rank=2,
      learning_rate=0.003,
      lattice_size=lattice_size)

  # Estimator.
  # input: coordinate of the pixel
  # output: value of the pixel
  feature_columns = [
      tf.feature_column.numeric_column('pixel_x'),
      tf.feature_column.numeric_column('pixel_y'),
  ]

  def keypoints_initializers():
    return tfl.uniform_keypoints_for_signal(
        num_keypoints,
        input_min=0.0,
        input_max=im_pixels.max(),
        output_min=0.0,
        output_max=lattice_size - 1
    )
  rtl_estimator = tfl.calibrated_rtl_regressor(
      feature_columns=feature_columns,
      hparams=hparams,
      keypoints_initializers_fn=keypoints_initializers
  )

  # Example input function.
  input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x={
          'pixel_x': im_pixels[:, 0],
          'pixel_y': im_pixels[:, 1]
      },
      y=im_pixels[:, 2],
      batch_size=5000,
      num_epochs=15,
      shuffle=True)

  # Train!
  rtl_estimator.train(input_fn=input_fn)

  # Evaluate!
  eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x={
          'pixel_x': im_pixels[:, 0],
          'pixel_y': im_pixels[:, 1]
      },
      y=im_pixels[:, 2],
      batch_size=5000,
      num_epochs=1,
      shuffle=True)
  print(rtl_estimator.evaluate(input_fn=eval_input_fn))

  return rtl_estimator


def visualize(estimator, input_img_path, output_dir):
  """Visualizes trained estimator."""
  # This example pulls one channel, also would make sense to convert to gray
  im = plt.imread(input_img_path)[:, :, 2]
  im_pixels = _pixels(im)

  input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x={
          'pixel_x': im_pixels[:, 0],
          'pixel_y': im_pixels[:, 1]
      },
      batch_size=10000,
      num_epochs=1,
      shuffle=False)

  y_test = np.array(
      [q['predictions'] for q in estimator.predict(input_fn=input_fn)])
  img = _pixels_to_image(np.c_[im_pixels[:, :2], y_test])

  plt.figure()
  plt.imshow(img, cmap='gray')
  plt.savefig(output_dir + '/image.png')
  return img


def main(image_path):
  """Fits image and provides visualization."""
  temp_dir = tempfile.mkdtemp()
  print('Saving output to {}'.format(temp_dir))
  estimator = run_image(image_path)
  visualize(estimator, image_path, temp_dir)

if __name__ == '__main__':
  input_image_path = sys.argv[1]
  main(input_image_path)
