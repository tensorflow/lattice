<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tensorflow_lattice.calibrated_linear_regressor" />
</div>

# tensorflow_lattice.calibrated_linear_regressor

``` python
calibrated_linear_regressor(
    feature_columns=None,
    model_dir=None,
    quantiles_dir=None,
    keypoints_initializers_fn=None,
    optimizer=None,
    config=None,
    hparams=None
)
```

Calibrated linear estimator (model) for regression.

This model uses a piecewise linear calibration function on each of the
inputs (parametrized) and then combine (sum up) the results. Optionally
calibration can be made monotonic.

It usually requires a preprocessing step on the data, to calculate the
quantiles of each used feature. This can be done locally or in one worker
only before training, in a separate invocation of your program (or directly)
in . Typically this can be save (`save_dir` parameter) to the same
directory where the data is.

Hyper-parameters are given in the form of the object
tfl_hparams.CalibrationHParams. It takes in per-feature calibration
parameters.

Internally values will be converted to tf.float32.



Example:

```python
def input_fn_train: ...
def input_fn_eval: ...

my_feature_columns=[...]

# Have a separate program flag to generate the quantiles. Need to be run
# only once.
if FLAGS.create_quantiles:
  pwl_calibrators_layers.calculate_quantiles_for_keypoints(
    input_fn=input_fn_train,
    feature_columns=my_feature_columns,
    save_dir=FLAGS.data_dir,
    num_quantiles=1000,
    override=True)
  return  # Exit program.

estimator = calibrated_linear.calibrated_linear_regressor(
  feature_columns=feature_columns)
estimator.train(input_fn=input_fn_train)
estimator.evaluate(input_fn=input_fn_eval)
estimator.predict(input_fn=input_fn_predict)
```

#### Args:

* <b>`feature_columns`</b>: Optional, if not set the model will use all features
    returned by input_fn. An iteratable containing all the feature
    columns used by the model. All items in the set should be instances of
    classes derived from `FeatureColumn`. If not given, the model will
    use as features the tensors returned by input_fn.
    Supported types: RealValuedColumn.
* <b>`model_dir`</b>: Directory to save model parameters, graph and etc. This can
    also be used to load checkpoints from the directory into a estimator to
    continue training a previously saved model.
* <b>`quantiles_dir`</b>: location where quantiles for the data was saved. Typically
    the same directory as the training data. These quantiles can be
    generated only once with
    `pwl_calibration_layers.calculate_quantiles_for_keypoints` in a separate
    invocation of your program. If you don't want to use quantiles, you can
    set `keypoints_initializer` instead.
* <b>`keypoints_initializers_fn`</b>: if you know the distribution of your
    input features you can provide that directly instead of `quantiles_dir`.
    See `pwl_calibrators_layers.uniform_keypoints_for_signal`. It must be
    a closure that returns a pair of tensors with keypoints inputs and
    outputs to use for initialization (must match `num_keypoints` configured
    in `hparams`). Alternatively the closure can return a dict mapping
    feature name to pairs for initialization per feature. If `quantiles_dir`
    and `keypoints_initializers_fn` are set, the later takes precendence,
    and the features for which `keypoints_initializers` are not defined
    fallback to using the quantiles found in `quantiles_dir`. It uses a
    closure instead of the tensors themselves because the graph has to be
    created at the time the model is being build, which happens at a later
    time.
* <b>`optimizer`</b>: string, `Optimizer` object, or callable that defines the
    optimizer to use for training -- if a callable, it will be called with
    learning_rate=hparams.learning_rate.
* <b>`config`</b>: RunConfig object to configure the runtime settings. Typically set
    to learn_runner.EstimatorConfig().
* <b>`hparams`</b>: an instance of tfl_hparams.CalibrationHParams. If set to
    None default parameters are used.


#### Returns:

A `CalibratedLinearRegressor` estimator.


#### Raises:

* <b>`ValueError`</b>: invalid parameters.
* <b>`KeyError`</b>: type of feature not supported.