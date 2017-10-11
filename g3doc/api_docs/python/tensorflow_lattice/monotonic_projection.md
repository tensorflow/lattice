<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tensorflow_lattice.monotonic_projection" />
</div>

# tensorflow_lattice.monotonic_projection

``` python
monotonic_projection(
    values,
    increasing,
    name=None
)
```

Returns a not-strict monotonic projection of the vector.

The returned vector is of the same size as the input and values (optionally)
changed to make them monotonically, minimizing the sum of the square distance
to the original values.

This is part of the set of ops that support monotonicity in piecewise-linear
calibration.

Note that the gradient is undefined for this function.

  values: `Tensor` with values to be made monotonic.
  increasing: Defines if projection it to monotonic increasing values
    or to monotonic decreasing ones.

  monotonic: output `Tensor` with values made monotonic.

#### Args:

* <b>`values`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`.
* <b>`increasing`</b>: A `Tensor` of type `bool`.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

A `Tensor`. Has the same type as `values`.