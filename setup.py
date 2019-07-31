# pylint: disable=g-bad-file-header
# Copyright 2017 The TensorFlow Lattice Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Setup for pip package."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import warnings

from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase
from setuptools.dist import Distribution


__version__ = '0.9.9'


REQUIRED_PACKAGES = [
    'six >= 1.11.0',
    'protobuf >= 3.6.1',
    'numpy >= 1.14.5',
]


if '--gpu' in sys.argv:
  use_gpu = True
  sys.argv.remove('--gpu')
else:
  use_gpu = False


if use_gpu:
  project_name = 'tensorflow-lattice-gpu'
  REQUIRED_PACKAGES.append('tensorflow-gpu==1.14.0')
else:
  project_name = 'tensorflow-lattice'
  REQUIRED_PACKAGES.append('tensorflow==1.14.0')

CONSOLE_SCRIPTS = [
    'freeze_graph_wrapper = '
    'tensorflow_lattice.cc.tflite.freeze_graph_wrapper:main',
    'toco_wrapper = tensorflow_lattice.cc.tflite.toco_wrapper:main',
]


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""

  def has_ext_modules(self):
    return True


warnings.warn('tensorflow-lattice is likley to fail when building from a '
              'source distribution (sdist). Please follow instructions in '
              '(https://github.com/tensorflow/lattice/INSTALL.md) '
              'to build this from the source.')


setup(
    name=project_name,
    version=__version__,
    description=('TensorFlow Lattice provides lattice models in TensorFlow'),
    long_description='',
    url='https://github.com/tensorflow/lattice',
    author='Google Inc.',
    author_email='tensorflow-lattice-releasing@google.com',
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    # Add in any packaged data.
    include_package_data=True,
    package_data={'': ['*.so']},
    exclude_package_data={'': ['BUILD', '*.h', '*.cc']},
    zip_safe=False,
    distclass=BinaryDistribution,
    cmdclass={
        'pip_pkg': InstallCommandBase,
    },
    entry_points={
        'console_scripts': CONSOLE_SCRIPTS
    },
    # PyPI package information.
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
        ],
    license='Apache 2.0',
    keywords='lattice tensorflow tensor machine learning',
)
