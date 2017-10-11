<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tensorflow_lattice.calibration_layer" />
</div>

# tensorflow_lattice.calibration_layer

``` python
calibration_layer(
    uncalibrated_tensor,
    num_keypoints,
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
    name=None
)
```

Creates a calibration layer for uncalibrated values.

Returns a calibrated tensor of the same shape as the uncalibrated continuous
signals passed in, and a list of projection ops, that must be applied at
each step (or every so many steps) to project the model to a feasible space:
used for bounding the outputs or for imposing monotonicity -- the list will be
empty if bound and monotonic are not set.

#### Args:

* <b>`uncalibrated_tensor`</b>: Tensor of shape [batch_size, ...] with uncalibrated
    values.
* <b>`num_keypoints`</b>: Number of keypoints to use. Either a scalar value that
    will be used for every uncalibrated signal, or a list of n values,
    per uncalibrated signal -- uncalibrated is first flattened (
    see tf.contrib.layers.flatten) to [batch_size, n], and there should
    be one value in the list per n. If a value of the list is 0 or None
    the correspondent signal won't be calibrated.
* <b>`keypoints_initializers`</b>: For evaluation or inference (or when resuming
    training from a checkpoint) the values will be loaded from disk, so they
    don't need to be given (leave it as None).
    Otherwise provide either a tuple of two tensors of shape [num_keypoints],
    or a list of n pairs of tensors, each of shape [num_keypoints]. In this
    list there should be one pair per uncalibrated signal, just like
    num_keypoints above. Notice that num_keypoints can be different per
    signal.
* <b>`keypoints_initializer_fns`</b>: Like keypoints_initializers but using lambda
    initializers. They should be compatible with tf.get_variable. If this is
    set, then keypoints_initializers must be None.
* <b>`bound`</b>: boolean whether output of calibration must be bound. Alternatively
    a list of n booleans, one per uncalibrated value, like num_keypoints
    above.
* <b>`monotonic`</b>: whether calibration is monotonic: None or 0 means no
    monotonicity. Positive or negative values mean increasing or decreasing
    monotonicity respectively. Alternatively a list of n monotonic values,
    one per uncalibrated value, like num_keypoints above.
* <b>`missing_input_values`</b>: If set, and if the input has this value it is assumed
    to be missing and the output will either be calibrated to some value
    between `[calibration_output_min, calibration_output_max]` or set to a
    fixed value set by missing_output_value. Limitation: it only works for
    scalars. Either one value for all inputs, or a list with one value per
    uncalibrated value.
* <b>`missing_output_values`</b>: Requires missing_input_value also to be set. If set
    if will convert missing input to this value. Either one value for all
    outputs, or a list with one value per uncalibrated value.
* <b>`l1_reg`</b>: (list of floats or float) l1 regularization amount.
    If float, then same value is applied to all dimensions.
* <b>`l2_reg`</b>: (list of floats or float) l2 regularization amount.
    If float, then same value is applied to all dimensions.
* <b>`l1_laplacian_reg`</b>: (list of floats or float) l1 laplacian
    regularization amount. If float, then same value is applied to all
    dimensions.
* <b>`l2_laplacian_reg`</b>:  (list of floats or float) l2 laplacian
    regularization amount. If float, then same value is applied to all
    dimensions.
* <b>`name`</b>: Name scope for operations.


#### Returns:

A tuple of:
* calibrated tensor of shape [batch_size, ...], the same shape as
  uncalibrated.
* list of projection ops, that must be applied at each step (or every so
  many steps) to project the model to a feasible space: used for bounding
  the outputs or for imposing monotonicity. Empty if none are requested.
* None or tensor with regularization loss.


#### Raises:

* <b>`ValueError`</b>: If dimensions don't match.