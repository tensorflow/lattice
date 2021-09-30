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
"""Generate docs API for TF Lattice.

Example run:

```
python build_docs.py --output_dir=/path/to/output
```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl import app
from absl import flags

from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator import public_api

import tensorflow_lattice as tfl

flags.DEFINE_string('output_dir', '/tmp/tfl_api/',
                    'The path to output the files to')

flags.DEFINE_string(
    'code_url_prefix',
    'https://github.com/tensorflow/lattice/blob/master/tensorflow_lattice',
    'The url prefix for links to code.')

flags.DEFINE_bool('search_hints', True,
                  'Include metadata search hints in the generated files')

flags.DEFINE_string('site_path', 'lattice/api_docs/python',
                    'Path prefix in the _toc.yaml')

FLAGS = flags.FLAGS


def local_definitions_filter(path, parent, children):
  """Filters local imports, except for the tfl.layers module."""
  if path == ('tfl', 'layers'):
    return children
  return public_api.local_definitions_filter(path, parent, children)


def main(_):
  private_map = {
      'tfl': ['python'],
      'tfl.aggregation_layer': ['Aggregation'],
      'tfl.categorical_calibration_layer': ['CategoricalCalibration'],
      'tfl.cdf_layer': ['CDF'],
      'tfl.kronecker_factored_lattice_layer': ['KroneckerFactoredLattice'],
      'tfl.lattice_layer': ['Lattice'],
      'tfl.linear_layer': ['Linear'],
      'tfl.pwl_calibration_layer': ['PWLCalibration'],
      'tfl.parallel_combination_layer': ['ParallelCombination'],
      'tfl.rtl_layer': ['RTL'],
  }
  doc_generator = generate_lib.DocGenerator(
      root_title='TensorFlow Lattice 2.0',
      py_modules=[('tfl', tfl)],
      base_dir=os.path.dirname(tfl.__file__),
      code_url_prefix=FLAGS.code_url_prefix,
      search_hints=FLAGS.search_hints,
      site_path=FLAGS.site_path,
      private_map=private_map,
      callbacks=[local_definitions_filter])

  sys.exit(doc_generator.build(output_dir=FLAGS.output_dir))


if __name__ == '__main__':
  app.run(main)
