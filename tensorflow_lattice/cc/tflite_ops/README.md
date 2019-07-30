<!-- Copyright 2018 The TensorFlow Lattice Authors.

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

# tensorflow_lattice on TF-Lite

## Concepts

__TF-Lite__ Framework in tensorflow/contrib for evaluating TF graphs on
low-power platforms <go/tf-lite>

__TOCO__ Tool for converting saved tensorflow graphs to tf-lite format
[TOCO docs](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/toco/g3doc/cmdline_examples.md)

## Introduction

This document describes how to use a lattice model on a low-power platform by
converting it to a Tensorflow-lite model which can then be run on the device.
This allows for inferences to be done without wifi or server costs.

## Use Notes

These tf-lite ops are necessary when running a tf-lite model that includes any
custom tf-lattice ops. Typically, a TF model saved in frozen_graph format will
be converted with TOCO. The output of TOCO can then be run (on device) with
TF-Lite. There are two integration tasks corresponding to these two steps:

### Use TOCO to convert a saved Tensorflow graph

TOCO, as explained above, operates on the output of the `tflite_convert`
utility. In order for this utility to work properly, Tensorflow itself must have
already loaded any custom ops that are needed. This is done 'lazily', so that
Tensorflow, and consequently TOCO, will fail to find a custom op that has not
yet been loaded. This is the purpose of the `toco_wrapper` target in this
directory. It triggers the loading of tensorflow_lattice ops by importing the
`tensorflow_lattice` python package. The wrapper script simply makes this import
and then calls `tflite_convert`. Use `toco_wrapper` with the same arguments that
you wish passed on to `tflite_convert`.

### Make the tf-lite op visible to the tf-lite interpreter

The low level code that instantiates and calls the tf-lite interpreter must be
modified to register the custom op. __The registration is done _in situ_ by the
team who wish to use the ops.__ Remember to add the `'tflite_ops` dependency to
build target.

## Example Commands

Example code for registering op:

```c++
#include "third_party/py/tensorflow_lattice/cc/tflite_ops/tflite_ops.h"

namespace tflite {

// ...

tflite::ops::builtin::BuiltinOpResolver resolver;
// this is the key addition
RegisterTfLatticeOps(&resolver);
```

Example commands, useful for testing that an op is reachable:

```
$ toco_wrapper \
  --output_file=/tmp/xo.tflite \
  --graph_def_file=/usr/local/google/home/epenn/Downloads/frozen_graph.pb \
  --input_arrays=deploy/Placeholder \
  --output_arrays=deploy/regression/MatMul \
  --allow_custom_ops

# This command will fail unless an edit like that described above is made to
# .../lite/tools/benchmark/benchmark_tflite_model.cc
$ bazel run tensorflow/lite/tools/benchmark:benchmark_model \
  -- --graph=/tmp/xo.tflite

```

If successful, the last command will print a summary of run timings.

## Full Example

### Build model

Consider the following simple tf_lattice model. Note where the model directory
is being set, this information will be important later.

```python
import numpy as np

import tensorflow as tf
import tensorflow_lattice as tfl

# Feature definition.
feature_columns = [
    tf.feature_column.numeric_column('x0'),
    tf.feature_column.numeric_column('x1'),
]

# Hyperparameters.
num_keypoints = 10
hparams = tfl.CalibratedRtlHParams(
    num_keypoints=num_keypoints,
    num_lattices=5,
    lattice_rank=2,
    learning_rate=0.1)
def init_fn():
  return tfl.uniform_keypoints_for_signal(num_keypoints,
                                          input_min=-1.0,
                                          input_max=1.0,
                                          output_min=0.0,
                                          output_max=1.0)

# Estimator.
rtl_estimator = tfl.calibrated_rtl_regressor(
    model_dir='/tmp/tfl_estimator_0',  # Set model directory
    feature_columns=feature_columns,
    hparams=hparams,
    keypoints_initializers_fn=init_fn
)

# Prepare the dataset.
num_examples = 1000
x0 = np.random.uniform(-1.0, 1.0, size=num_examples)
x1 = np.random.uniform(-1.0, 1.0, size=num_examples)
y = x0 ** 2 + x1 ** 2

# Example input function.
twod_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={'x0': x0,
       'x1': x1},
    y=y,
    batch_size=10,
    num_epochs=1,
    shuffle=False)

# Train!
rtl_estimator.train(input_fn=twod_input_fn)
# Evaluate!
print(rtl_estimator.evaluate(input_fn=twod_input_fn))
```

### Determine input and output nodes

In order to use the conversion utilities below, it is necessary to know which
nodes in the tensorflow model graph are to be used as input and output. This can
be tricky, especially when using the estimator API.

To visually inspect the graph, run the following:

```bash
$ MODEL_DIR=/tmp/tfl_estimator_0  # from above
$ tensorboard --logdir $MODEL_DIR  # use the model directory specified above
```

For this example, the following nodes will be used for input and output:

```bash
$ INPUT_NODE=tfl_calibrated_rtl/feature_column_transformation/input_layer/concat
$ OUTPUT_NODE=tfl_calibrated_rtl/add
```

### Convert trained model to frozen graph format using frozen_graph_wrapper

This conversion uses the tensorflow `frozen_graph` utility. As with
`tflite_convert` (TOCO), this utility requires that tensorflow has loaded the
tensorflow_lattice custom ops. In order to facilitate this, a simple wrapper is
provided.

```bash
$ freeze_graph_wrapper \
  --input_graph=$MODEL_DIR/graph.pbtxt \
  --input_checkpoint=$MODEL_DIR/model.ckpt-100 \
  --output_graph=$MODEL_DIR/output_graph.pb \
  --output_node_names=tfl_calibrated_rtl/add
```

### Convert frozen graph to tf-lite format using toco_wrapper

This step will produce a tf-lite artifact suitable for use. Note that use will
require edits to the low level C++ code as described above

```bash
$ toco_wrapper \
  --output_file=$MODEL_DIR/tflite.out \
  --graph_def_file=$MODEL_DIR/output_graph.pb \
  --input_arrays=$INPUT_NODE \
  --output_arrays=$OUTPUT_NODE \
  --allow_custom_ops
```
