<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tensorflow_lattice.uniform_keypoints_for_signal" />
</div>

# tensorflow_lattice.uniform_keypoints_for_signal

``` python
uniform_keypoints_for_signal(
    num_keypoints,
    input_min,
    input_max,
    output_min,
    output_max,
    dtype=dtypes.float32
)
```

Returns a pair of initialization tensors for calibration keypoints.

This is used when the input range to be calibrated is known.

#### Args:

* <b>`num_keypoints`</b>: number of keypoints to use for calibrating this signal.
* <b>`input_min`</b>: Scalar with the minimum value that the uncalibrated input
    can take.
* <b>`input_max`</b>: Scalar with the maximum value that the uncalibrated input
    can take.
* <b>`output_min`</b>: Scalar with calibrated value associated with input_min.
    Typically the minimum expected calibrated value, but not necessarily.
    Specially if the calibration is decreasing.
* <b>`output_max`</b>: Scalar with calibrated scalar value associated with
    input_max.
* <b>`dtype`</b>: If any of the scalars are not given as tensors, they are converted
    to tensors with this dtype.


#### Returns:

Two tensors to be used as the keypoints_inputs and keypoints_outputs
initialization, uniformly distributed over given ranges. Dtype is given
by input_min, input_max, output_min, output_max.


#### Raises:

* <b>`ValueError`</b>: if underlying types (dtype) don't match.