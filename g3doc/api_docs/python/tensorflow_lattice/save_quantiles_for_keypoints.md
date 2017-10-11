<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tensorflow_lattice.save_quantiles_for_keypoints" />
</div>

# tensorflow_lattice.save_quantiles_for_keypoints

``` python
save_quantiles_for_keypoints(
    input_fn,
    save_dir,
    feature_columns=None,
    num_steps=1,
    override=True,
    num_quantiles=1000,
    dtype=dtypes.float32
)
```

Calculates and saves quantiles for given features.

These values can later be retrieved and used by keypoints_from_quantiles()
below.

Repeated values are discarded before the quantiles are calculated. That means
that the quantiles of a very skewed distribution (for instance where 99%
of the values are 0), will be different. But for the purpose of calibration
this approach is more useful.

Nothing is returned, the values are simply saved in the given location.

This function can be called as a preprocessing step before actual training
starts. Typically one will run this in a separate process locally, before
starting training for instance.

#### Args:

* <b>`input_fn`</b>: Similar to input_fn provided to Estimators. Typically one
    doesn't need to go over the full data to get good quantiles. Typically
    some 100 random examples per quantile is good enough for the purpose of
    calibration. If you don't have too much data, just use everything.
    If input_fn returns a target (used in training) it is ignored.
* <b>`save_dir`</b>: Where to save these quantiles. Since when optimizing
    hyper-parameters we train various models, we can share the quantiles
    information generated here. So this should be a directory that can be
    accessed by all training sessions. A subdirectory called "quantiles" will
    be created, and inside one file per feature is created: named after the
    feature name, and with the quantiles stored in JSON format.
* <b>`feature_columns`</b>: If set, quantiles are generated for these feature columns.
    The file name used to save the quantiles uses a hash of the names of the
    feature_columns, so it can support different quantiles sets for different
    parts of the model if needed. If not set quantiles will be generated for
    all features returned by input_fn.
* <b>`num_steps`</b>: number of steps to take over input_fn to gather enough data to
    create quantiles. Set to 0 or None to run until queue is exhausted,
    like if you used num_epochs in your input_fn.
* <b>`override`</b>: if False it won't regenerate quantiles for files that are already
    there. This works as long as the features definition/distribution hasn't
    change from one run to another.
* <b>`num_quantiles`</b>: This value should be larger than the maximum number of
    keypoints that will be considered for calibrating these features. If
    there are not enough quantiles for the keypoints, the system is robust and
    will simply interpolate the missing quantiles. Similarly if there are not
    enough examples to represent the quantiles, it will interpolate the
    quantiles from the examples given.
* <b>`dtype`</b>: Deafult dtype to use, in particular for categorical values.

Returns: Nothing, results are saved to disk.


#### Raises:

* <b>`errors.OpError`</b>: For I/O errors.

FutureWork:
  * Use Munro-Paterson algorithm to calculate quantiles in a streaming
    fashion. See Squawd library.
  * Add support to weighted examples.
  * Handle cases where there are not enough different values in quantiles.