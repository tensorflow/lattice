<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tensorflow_lattice.load_keypoints_from_quantiles" />
</div>

# tensorflow_lattice.load_keypoints_from_quantiles

``` python
load_keypoints_from_quantiles(
    feature_names,
    save_dir,
    num_keypoints,
    output_min,
    output_max,
    dtype=dtypes.float32
)
```

Retrieves keypoints initialization values for selected features.

It expects that the quantiles have already been calculated and saved in the
save_dir by the save_quantiles_for_keypoints function. It will raise
an I/O error if not.

#### Args:

* <b>`feature_names`</b>: List of features names for which to get keypoints
    initialization values.
* <b>`save_dir`</b>: Directory where the quantiles have been saved to. Same value used
    when save_quantiles_for_keypoints was called.
* <b>`num_keypoints`</b>: Desired number of keypoints to use for calibration. This
    can either be a scalar to be used for all features, or a dict mapping
    feature name to num_keypoints. Fewer keypoints than requested can end
    up being used when for the given feature there are not enough different
    values. If num_keypoints for a feature is missing, None or 0, no
    initialization is generated.
* <b>`output_min`</b>: Initial calibrated value associated with the first calibration
    keypoint. The keypoints outputs in between will be linearly interpolated.
    It can be given as a scalar, in which case value is used for all features,
    or a dict mapping feature name to output_min.
* <b>`output_max`</b>: Like output_min, but the calibrated value associated to the
    last keypoint. Scalar or dict.
* <b>`dtype`</b>: Type to be used for calibration.


#### Returns:

Dict of feature name to pair of constant tensors that can be used to
initialize calibrators keypoints inputs and outputs.


#### Raises:

* <b>`tf.errors.NotFoundError`</b>: if quantiles file not found.


  values in the signal. This would probably be better handled as categorical,
  but still this should handle the case correctly.