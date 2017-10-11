<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tensorflow_lattice.CalibratedLinearHParams" />
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

# tensorflow_lattice.CalibratedLinearHParams

## Class `CalibratedLinearHParams`

Inherits From: [`CalibratedHParams`](../tensorflow_lattice/CalibratedHParams.md)

Hyper-parameters for CalibratedLinear models.

Same as `CalibratedHParams` (hyper-parameters for input calibration) plus
the global learning_rate.

The parameters `calibration_output_min` and `calibration_output_max` shouldn't
be changed (they are fixed at -1. and +1), since they are eventually re-scaled
by the linear layer on top.

It supports regularization, monotonicity and missing values (input and
optionally output).

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

