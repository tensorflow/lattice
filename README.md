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
<div align="center">
<img src="g3doc/images/tensorflow_lattice.png" style="width: 100px"/>
</div>

# TensorFlow Lattice

This is an implementation of [Monotonic Calibrated Interpolated Look-Up Tables](http://jmlr.org/papers/v17/15-243.html) in [TensorFlow](https://www.tensorflow.org).

These are fast-to-evaluate and interpretable lattice models, also known as
interpolated look-up tables. This library also provides a rich and intuitive set
of regularizations and monotonicity constraints configurable per feature.

It includes
[__TensorFlow estimators__](https://www.tensorflow.org/extend/estimators) for
regression and classification with the most common set ups for lattice models:

* Calibrated Linear
* Calibrated Lattice
* Random Tiny Lattices (_RTL_)
* Embedded Tiny Lattices (_ETL_) (see [Deep Lattice Networks and Partial Monotonic Functions](https://research.google.com/pubs/pub46327.html))

Additionally this library provides two types of __model components__
(or __layers__) that can be combined with other types of models (including
neural networks):

* Calibration: piecewise linear calibration of signals.
* Lattice: interpolated look-up table implementation.


You can install our prebuilt pip package using

```bash
pip install tensorflow-lattice
```

but please see the [install](INSTALL.md) section for more detailed instructions.

This [tutorial](g3doc/tutorial/index.md) contains more detailed explanation
about lattice models and usage in TensorFlow, and check out
[API docs](g3doc/api_docs/python/index.md) for python APIs.

__TensorFlow Lattice is not an official Google product.__
