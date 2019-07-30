<!-- Copyright 2017 The TensorFlow Lattice Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=============================================================================-->
# TensorFlow Lattice installation

TensorFlow Lattice runs on Ubuntu and Mac OS X, and requires TensorFlow.

We highly recommend to read [TensorFlow installation
instructions](https://www.tensorflow.org/install), especially [Installing
TensorFlow on Ubuntu](https://www.tensorflow.org/install/install_linux) to
understand virtualenv and pip, and [Installing TensorFlow from
Sources](https://www.tensorflow.org/install/install_sources).

# Install the prebuilt pip package

## Activate virtualenv
If using virtualenv, activate your virtualenv for the rest of the installation,
otherwise skip this step:

``` shell
~$ virtualenv --system-site-packages tensorflow-lattice # for Python 2.7
~$ virtualenv --system-site-packages -p python3 tensorflow-lattice # for Python 3.n
```

Here you can change `tensorflow-lattice` to another target directory you want to
use.

```shell
~$ source tensorflow-lattice/bin/activate # bash, sh, ksh, or zsh
~$ source tensorflow-lattice/bin/activate.csh  # csh or tcsh
```

## Install pip packages.
You can use pip install to install tensorflow-lattice pip package.

```shell
(tensorflow-lattice)$ pip install --upgrade tensorflow-lattice # for Python 2.7
(tensorflow-lattice)$ pip3 install --upgrade tensorflow-lattice # for Python 3.n
(tensorflow-lattice)$ pip install --upgrade tensorflow-lattice-gpu # for Python 2.7 and GPU
(tensorflow-lattice)$ pip3 install --upgrade tensorflow-lattice-gpu # for Python 3.n and GPU
```
Our custom operators do not have GPU kernels. The main difference
between `tensorflow-lattice-gpu` and `tensorflow-lattice` pip package is that
the former requires `tensorflow-gpu` pip package whereas the latter requires
`tensorflow` pip package.

## Test TensorFlow and TensorFlow Lattice

Run the following python script to test TensorFlow Lattice.

```python
import tensorflow as tf
import tensorflow_lattice as tfl

x = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))
(y, _, _, _) = tfl.lattice_layer(x, lattice_sizes=(2, 2))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print(sess.run(y, feed_dict={x: [[0.0, 0.0]]}))
```

Now you are ready to use *TensorFlow Lattice*. Check out examples in the
[examples](https://github.com/tensorflow/lattice/tree/master/examples) directory
and run them if you need more examples to run.
[Tutorial](g3doc/tutorial/index.md) contains detailed explanation on how to use
TensorFlow Lattice.

You can stop here unless you want to build TensorFlow Lattice from the source.

# Build TensorFlow Lattice and TensorFlow pip package from the source.
You can also build TensorFlow Lattice packages from the source.
For this, you will need to compile all libraries using
[Bazel](https://bazel.build) against TensorFlow headers.


We will show how to build TensorFlow and TensorFlow Lattice pip package using
Bazel, and install it to your virtualenv.

## Activate virtualenv

If using virtualenv, activate your virtualenv for the rest of the installation,
otherwise skip this step:

```shell
~$ source $VIRTUALENV_PATH/bin/activate # bash, sh, ksh, or zsh
~$ source $VIRTUALENV_PATH/bin/activate.csh  # csh or tcsh
```

or if you are using virtualenv for the first time,

```shell
~$ sudo apt-get install python-virtualenv
~$ virtualenv --system-site-packages tensorflow-lattice
~$ source ~/tensorflow-lattice/bin/activate  # bash, sh, ksh, or zsh
~$ source ~/tensorflow-lattice/bin/activate.csh  # csh or tcsh
```
## Prepare TensorFlow envirnoment for Linux.

Please follow instructions in [Prepare environment for
Linux](https://www.tensorflow.org/install/install_sources#prepare_environment_for_linux)
to setup the environment for TensorFlow.

## Clone the TensorFlow Lattice repository.

Let us clone the TensorFlow Lattice repository, which contains TensorFlow as a
submodule:

```shell
(tensorflow-lattice)~$ git clone --recursive https://github.com/tensorflow/lattice.git
```

## Configure TensorFlow and build TensorFlow pip package.

### Configure TensorFlow

We now need to configure TensorFlow options. See [Configure the
installation](https://www.tensorflow.org/install/install_sources#configure_the_installation)
for the details.

```shell
(tensorflow-lattice)~$ cd lattice
(tensorflow-lattice)~/lattice$ cd tensorflow
(tensorflow-lattice)~/lattice/tensorflow$ ./configure
```

### Build TensorFlow pip packaging script

We are ready to build the TensorFlow pip package. See [Build the pip
package](https://www.tensorflow.org/install/install_sources#build_the_pip_package)
for the details.

To build a pip package for TensorFlow with CPU-only support:

```shell
(tensorflow-lattice)~/lattice/tensorflow$ bazel build \
  --config=opt \
  tensorflow/tools/pip_package:build_pip_package
```

To build a pip package for TensorFlow with GPU support:

```shell
(tensorflow-lattice)~/lattice/tensorflow$ bazel build \
  --config=cuda \
  tensorflow/tools/pip_package:build_pip_package
```

### Install TensorFlow pip package

```shell
(tensorflow-lattice)~/lattice/tensorflow$ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
(tensorflow-lattice)~/lattice/tensorflow$ pip install /tmp/tensorflow_pkg/*.whl
```

### Build TensorFlow Lattice pip packaging script

To build a pip package for TensorFlow with CPU-only support:

```shell
(tensorflow-lattice)~/$ cd ~/lattice
(tensorflow-lattice)~/lattice$ bazel build \
  --config=opt :pip_pkg
```

### Install TensorFlow Lattice pip package

```shell
(tensorflow-lattice)~/lattice$ bazel-bin/pip_pkg /tmp/tensorflow_lattice_pkg
(tensorflow-lattice)~/lattice$ pip install /tmp/tensorflow_lattice_pkg/*.whl
```

### Test TensorFlow and TensorFlow Lattice
```shell
(tensorflow-lattice)~/lattice$ cd examples
(tensorflow-lattice)~/lattice/examples$ python test.py
```

test.py is a simple python script.

```python
import tensorflow as tf
import tensorflow_lattice as tfl

x = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))
(y, _, _, _) = tfl.lattice_layer(x, lattice_sizes=(2, 2))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print(sess.run(y, feed_dict={x: [[0.0, 0.0]]}))
```
