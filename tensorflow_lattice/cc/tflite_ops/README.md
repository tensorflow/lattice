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

 __TOCO__  Tool for converting saved tensorflow graphs to tf-lite format
 [TOCO docs](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/toco/g3doc/cmdline_examples.md)

## Use Notes

These tf-lite ops are necessary when running a tf-lite model that includes any
custom tf-lattice ops.  Typically, a TF model saved in frozen_graph format will
be converted with TOCO.  The output of TOCO can then be run (on device)
with TF-Lite.  There are two integration tasks corresponding to these two steps:

### Use TOCO to convert a saved Tensorflow graph

TOCO, as explained above, operates on the output of the `tflite_convert`
utility.  In order for this utility to work properly, Tensorflow itself must
have already loaded any custom ops that are needed.  This is done 'lazily',
so that Tensorflow, and consequently TOCO, will fail to find a custom op that
has not yet been loaded.  This is the purpose of the `toco_wrapper` target in
this directory.  It triggers the loading of tensorflow_lattice ops by importing
the `tensorflow_lattice` python package.  The wrapper script simply makes this
import and then calls `tflite_convert`.  Use `toco_wrapper` with the same
arguments that you wish passed on to `tflite_convert`.


### Make the tf-lite op visible to the tf-lite interpreter
The low level code that instantiates and calls the tf-lite interpreter must be
modified to register the custom op.  __The registration is done _in situ_ by the
team who wish to use the ops.__  Remember to add the `'tflite_ops` dependency to
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
$ blaze-bin/third_party/py/tensorflow_lattice/cc/tflite_ops/toco_wrapper \
  --output_file=/tmp/xo.tflite \
  --graph_def_file=/usr/local/google/home/epenn/Downloads/frozen_graph.pb \
  --input_arrays=deploy/Placeholder \
  --output_arrays=deploy/regression/MatMul \
  --allow_custom_ops

# This command will fail unless an edit like that described above is made to
# .../lite/tools/benchmark/benchmark_tflite_model.cc
$ blaze run third_party/tensorflow/contrib/lite/tools/benchmark:benchmark_model \
  -- --graph=/tmp/xo.tflite

```
If successful, the last command will print a summary of run timings.
