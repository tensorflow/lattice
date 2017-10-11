<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tensorflow_lattice.pwl_indexing_calibrator" />
</div>

# tensorflow_lattice.pwl_indexing_calibrator

``` python
pwl_indexing_calibrator(
    input,
    kp_inputs,
    name=None
)
```

Returns tensor representing interpolation weights in a piecewise linear

function. If using a large number of keypoints, try PwlIndexingCalibratorSparse.

Notice that in this version the keypoints inputs (given by kp_inputs) is kept
fixed by forcing its gradient to be always 0. FutureWork: allow kp_inputs to
also be optimized, by providing a gradient.

Inputs
  input: uncalibrated weights, `[batch_size]`
  kp_input: keypoints' input weights, can be initialized with the
            pwl_calibrator_initialize_input_keypoints op. `[num_keypoints]`

Outputs
  weights: Interpolation weights for a piecewise linear function. Its shape is
    `[batch_size, num_keypoints]`. The dot product of this and the keypoints
    output will give the calibrated value.

#### Args:

* <b>`input`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`.
* <b>`kp_inputs`</b>: A `Tensor`. Must have the same type as `input`.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

A `Tensor`. Has the same type as `input`.