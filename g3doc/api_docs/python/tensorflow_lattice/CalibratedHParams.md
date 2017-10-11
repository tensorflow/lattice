<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tensorflow_lattice.CalibratedHParams" />
<meta itemprop="property" content="__getattr__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add_feature"/>
<meta itemprop="property" content="get_feature_names"/>
<meta itemprop="property" content="get_feature_param"/>
<meta itemprop="property" content="get_global_and_feature_params"/>
<meta itemprop="property" content="get_param"/>
<meta itemprop="property" content="is_feature_set_param"/>
<meta itemprop="property" content="param_name_for_feature"/>
<meta itemprop="property" content="parse"/>
<meta itemprop="property" content="parse_hparams"/>
<meta itemprop="property" content="parse_param"/>
<meta itemprop="property" content="set_feature_param"/>
<meta itemprop="property" content="set_param"/>
<meta itemprop="property" content="set_param_type"/>
<meta itemprop="property" content="values"/>
<meta itemprop="property" content="FEATURE_PREFIX"/>
<meta itemprop="property" content="FEATURE_SEPARATOR"/>
</div>

# tensorflow_lattice.CalibratedHParams

## Class `CalibratedHParams`

Inherits From: [`PerFeatureHParams`](../tensorflow_lattice/PerFeatureHParams.md)

PerFeatureHParams specialization with input calibration parameters.

The following hyper-parameters can be set as global, or per-feature (see
base `PerFeatureHParams` for details):

  * `feature_names`: list of feature names. Only features names listed here
    (or added later with add_feature) can have feature specific parameter
    values.
  * `num_keypoints`: Number of keypoints to use for calibration, Set to 0 or
    `None` for no calibration.
  * `calibration_output_min`, `calibration_output_max`: initial and final
    values for calibrations. -1.0 to 1.0 works well for calibrated linear
    models. For lattices one will want to set these to (0, `lattice_size`-1).
    Only used during initialization of the calibration, if `quantiles_dir`
    is given to the calibrated model (as opposed to defining one's own value
    with `keypoints_initializers_fn`). It must be defined for calibration to
    work, no default is set.
  * `calibration_bound`: If output of calibration max/min are bound to the
    limits given in `calibration_output_min/max`.
  * `monotonicity`: Monotonicity for the feature. 0 for no monotonicity,
    1 and -1 for increasing and decreasing monotonicity respectively.
  * `missing_input_value`: If set, and if the input has this value it is
  assumed
    to be missing and the output will either be calibrated to some value
    between `[calibration_output_min, calibration_output_max]` or set to a
    fixed value set by missing_output_value.
  * `missing_output_value`: Requires missing_input_value also to be set. If
  set
    if will convert missing input to this value. Leave it undefined and the
    output will be learned.
  * `calibration_l1_reg`, `calibration_l2_reg`,
    `calibration_l1_laplacian_reg`, `calibration_l2_laplacian_reg`: Calibrator
    regularizers regularization amount. Default is `None`.

## Methods

<h3 id="__init__"><code>__init__</code></h3>

``` python
__init__(
    feature_names=None,
    **kwargs
)
```



<h3 id="__getattr__"><code>__getattr__</code></h3>

``` python
__getattr__(param_name)
```



<h3 id="add_feature"><code>add_feature</code></h3>

``` python
add_feature(feature_name)
```

Add feature_name (one name or list of names) to list of known names.

<h3 id="get_feature_names"><code>get_feature_names</code></h3>

``` python
get_feature_names()
```

Returns copy of list of known feature names.

<h3 id="get_feature_param"><code>get_feature_param</code></h3>

``` python
get_feature_param(
    feature_name,
    param_name,
    default=None
)
```

Returns parameter for feature or falls back to global parameter.

<h3 id="get_global_and_feature_params"><code>get_global_and_feature_params</code></h3>

``` python
get_global_and_feature_params(
    param_names,
    feature_names
)
```

Returns values for multiple params, global and for each feature.

#### Args:

* <b>`param_names`</b>: list of parameters to get values for.
* <b>`feature_names`</b>: list of features to get specific values for.


#### Returns:

* List of global values for parameters requested in `param_names`.
* List of list of per feature values for parameters requested in
  `param_names` for features requested in `feature_names`.

<h3 id="get_param"><code>get_param</code></h3>

``` python
get_param(
    param_name,
    default=None
)
```

Returns the global parameter or falls back to default.

<h3 id="is_feature_set_param"><code>is_feature_set_param</code></h3>

``` python
is_feature_set_param(
    feature_name,
    param_name
)
```

Returns whether param_name parameter is set for feature_name.

<h3 id="param_name_for_feature"><code>param_name_for_feature</code></h3>

``` python
param_name_for_feature(
    feature_name,
    param_name
)
```

Returns parameter name for specific feature parameter.

<h3 id="parse"><code>parse</code></h3>

``` python
parse(hparams_str)
```

Parses strings into hparams.

#### Args:

* <b>`hparams_str`</b>: must be a comma separated list of "<key>=<value>",
  where "<key>" is a hyper-parameter name, and "<value>" its value.


#### Returns:

Changes affect self, but returns self for convenience.


#### Raises:

* <b>`ValueError`</b>: if there is a problem with the input:
     * if trying to set an unknown parameter.
     * if trying to set unknown feature(s)
     * if can't convert value to parameter type.

<h3 id="parse_hparams"><code>parse_hparams</code></h3>

``` python
parse_hparams(hparams)
```

Incorporates hyper-parameters from another HParams object.

Copies over values of hyper-parameters from the given object. New parameters
may be set, but not new features. Also works with
`tf.contrib.training.HParams` objects.

#### Args:

* <b>`hparams`</b>: `PerFeatureHParams` object, but also works with the standard
    `tf.contrib.training.HParams` object.


#### Returns:

Changes affect self, but returns self for convenience.


#### Raises:

* <b>`ValueError`</b>: if trying to set unknown features, or if setting a feature
    specific parameter for an unknown parameter.

<h3 id="parse_param"><code>parse_param</code></h3>

``` python
parse_param(
    param_name,
    value_str
)
```

Parses parameter values from string. Returns self.

<h3 id="set_feature_param"><code>set_feature_param</code></h3>

``` python
set_feature_param(
    feature_name,
    param_name,
    value
)
```

Sets parameter value specific for feature. Returns self.

<h3 id="set_param"><code>set_param</code></h3>

``` python
set_param(
    param_name,
    value
)
```

Sets parameter value. Returns self.

<h3 id="set_param_type"><code>set_param_type</code></h3>

``` python
set_param_type(
    param_name,
    param_type
)
```

Sets the parameter type, it must already exist. Returns self.

<h3 id="values"><code>values</code></h3>

``` python
values()
```

Returns shallow copy of the hyperparameter dict.



## Class Members

<h3 id="FEATURE_PREFIX"><code>FEATURE_PREFIX</code></h3>

<h3 id="FEATURE_SEPARATOR"><code>FEATURE_SEPARATOR</code></h3>

