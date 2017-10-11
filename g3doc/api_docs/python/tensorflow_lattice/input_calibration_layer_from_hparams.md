<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tensorflow_lattice.input_calibration_layer_from_hparams" />
</div>

# tensorflow_lattice.input_calibration_layer_from_hparams

``` python
input_calibration_layer_from_hparams(
    columns_to_tensors,
    feature_columns,
    hparams,
    quantiles_dir=None,
    keypoints_initializers=None,
    name=None,
    dtype=dtypes.float32
)
```

Creates a calibration layer for the input using hyper-parameters.

Similar to `input_calibration_layer` but reads its parameters from a
`CalibratedHParams` object.

#### Args:

* <b>`columns_to_tensors`</b>: A mapping from feature name to tensors. 'string' key
    means a base feature (not-transformed). If feature_columns is not set
    these are the features calibrated. Otherwise the transformed
    feature_columns are the ones calibrated.
* <b>`feature_columns`</b>: An iterable containing all the feature columns used by the
    model. Optional, if not set the model will use all features given in
    columns_to_tensors. All items in the set should be instances of
    classes derived from `FeatureColumn`.
* <b>`hparams`</b>: Hyper-parameters, need to inherit from `CalibratedHParams`.
    It is also changed to include all feature names found in
    `feature_columns`. See `CalibratedHParams` and `input_calibration_layer`
    for descriptions of how these hyper-parameters work.
* <b>`quantiles_dir`</b>: location where quantiles for the data was saved. Typically
    the same directory as the training data. These quantiles can be
    generated with `pwl_calibration_layers.calculate_quantiles_for_keypoints`,
    maybe in a separate invocation of your program. Different models that
    share the same quantiles information -- so this needs to be generated only
    once when hyper-parameter tuning. If you don't want to use quantiles, you
    can set `keypoints_initializers` instead.
* <b>`keypoints_initializers`</b>: if you know the distribution of your
    input features you can provide that directly instead of `quantiles_dir`.
    See `pwl_calibrators_layers.uniform_keypoints_for_signal`. It must be
    a pair of tensors with keypoints inputs and outputs to use for
    initialization (must match `num_keypoints` configured in `hparams`).
    Alternatively can be given as a dict mapping feature name to pairs,
    for initialization per feature. If `quantiles_dir` and
    `keypoints_initializer` are set, the later takes precendence, and the
    features for which `keypoints_initializers` are not defined fallback to
    using the quantiles found in `quantiles_dir`.
* <b>`name`</b>: Name scope for layer.
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

