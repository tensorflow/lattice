<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tensorflow_lattice.input_calibration_layer" />
</div>

# tensorflow_lattice.input_calibration_layer

``` python
input_calibration_layer(
    columns_to_tensors,
    num_keypoints,
    feature_columns=None,
    keypoints_initializers=None,
    keypoints_initializer_fns=None,
    bound=False,
    monotonic=None,
    missing_input_values=None,
    missing_output_values=None,
    l1_reg=None,
    l2_reg=None,
    l1_laplacian_reg=None,
    l2_laplacian_reg=None,
    dtype=dtypes.float32
)
```

Creates a calibration layer for the given input and feature_columns.

Returns a tensor with the calibrated values of the given features, a list
of the names of the features in the order they feature in the returned, and
a list of projection ops, that must be applied at each step (or every so many
steps) to project the model to a feasible space: used for bounding the outputs
or for imposing monotonic -- the list will be empty if bound and
monotonic are not set.

#### Args:

* <b>`columns_to_tensors`</b>: A mapping from feature name to tensors. 'string' key
    means a base feature (not-transformed). If feature_columns is not set
    these are the features calibrated. Otherwise the transformed
    feature_columns are the ones calibrated.
* <b>`num_keypoints`</b>: Number of keypoints to use. Either a single int, or a dict
    mapping feature names to num_keypoints. If a value of the dict is 0 or
    None the correspondent feature won't be calibrated.
* <b>`feature_columns`</b>: Optional. If set to a set of FeatureColumns, these will
    be the features used and calibrated.
* <b>`keypoints_initializers`</b>: For evaluation or inference (or when resuming
    training from a checkpoint) the values will be loaded from disk, so they
    don't need to be given (leave it as None).
    Either a tuple of two tensors of shape [num_keypoints], or a dict mapping
    feature names to pair of tensors of shape [num_keypoints[feature_name]].
    See load_keypoints_from_quantiles or uniform_keypoints_for_signal on how
    to generate these (module keypoints_initialization).
* <b>`keypoints_initializer_fns`</b>: Like keypoints_initializers but using lambda
    initializers. They should be compatible with tf.get_variable. If this is
    set, then keypoints_initializers must be None.
* <b>`bound`</b>: boolean whether output of calibration must be bound. Alternatively
    a dict mapping feature name to boundness.
* <b>`monotonic`</b>: whether calibration has to be kept monotonic: None or 0 means
    no monotonic. Positive or negative values mean increasing or decreasing
    monotonic respectively. Alternatively a dict mapping feature name
    to monotonic.
* <b>`missing_input_values`</b>: If set, and if the input has this value it is assumed
    to be missing and the output will either be calibrated to some value
    between `[calibration_output_min, calibration_output_max]` or set to a
    fixed value set by missing_output_value. Limitation: it only works for
    scalars. Either one value for all inputs, or a dict mapping feature name
    to missing_input_value for the respective feature.
* <b>`missing_output_values`</b>: Requires missing_input_value also to be set. If set
    if will convert missing input to this value. Either one value for all
    inputs, or a dict mapping feature name to missing_input_value for the
    respective feature.
* <b>`l1_reg`</b>: ({feature_name: float} dict or float) l1 regularization amount.
    If float, then same value is applied to all features.
* <b>`l2_reg`</b>: ({feature_name: float} dict or float) l2 regularization amount.
    If float, then same value is applied to all features.
* <b>`l1_laplacian_reg`</b>: ({feature_name: float} dict or float) l1 laplacian
    regularization amount. If float, then same value is applied to all
    features.
* <b>`l2_laplacian_reg`</b>:  ({feature_name: float} dict or float) l2 laplacian
    regularization amount. If float, then same value is applied to all
    features.
* <b>`dtype`</b>: If any of the scalars are not given as tensors, they are converted
    to tensors with this dtype.


#### Returns:

A tuple of:
* calibrated tensor of shape [batch_size, sum(features dimensions)].
* list of the feature names in the order they feature in the calibrated
  tensor. A name may appear more than once if the feature is
  multi-dimension (for instance a multi-dimension embedding)
* list of projection ops, that must be applied at each step (or every so
  many steps) to project the model to a feasible space: used for bounding
  the outputs or for imposing monotonicity. Empty if none are requested.
* None or tensor with regularization loss.


#### Raises:

* <b>`ValueError`</b>: if dtypes are incompatible.

