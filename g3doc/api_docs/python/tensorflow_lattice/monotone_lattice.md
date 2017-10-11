<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tensorflow_lattice.monotone_lattice" />
</div>

# tensorflow_lattice.monotone_lattice

``` python
monotone_lattice(
    lattice_params,
    is_monotone=[],
    lattice_sizes=[],
    tolerance=1e-07,
    max_iter=1000,
    name=None
)
```

Returns a projected lattice parameters onto the monotonicity constraints.

Monotonicity constraints are specified is_monotone. If is_monotone[k] == True,
then the kth input has a non-decreasing monotonicity, otherwise there will be no
constraints.

This operator uses an iterative algorithm, Alternating Direction Method of
Multipliers (ADMM) method, to find the projection, so tolerance and max_iter can
be used to control the accuracy vs. the time spent trade-offs in the ADMM
method.

Inputs
  lattice_params: 2D tensor, `[number of outputs, number of parameters]`

Params
  is_monotone: 1D bool tensor that contains whether the kth dimension should be
  monotonic.
  lattice_sizes: 1D int tensor that contains a lattice size per each dimension,
  [m_0, ..., m_{d - 1}].
  tolerance: The tolerance in ||true projection - projection|| in the ADMM
  method.
  max_iter: Maximum number of iterations in the ADMM method.

Outputs
  projected_lattice_params: 2D tensor,
  `[number of outputs, number of parameters]`, that contains the projected
  parameters.

#### Args:

* <b>`lattice_params`</b>: A `Tensor`. Must be one of the following types: `float32`, `float64`.
* <b>`is_monotone`</b>: An optional list of `bools`. Defaults to `[]`.
* <b>`lattice_sizes`</b>: An optional list of `ints`. Defaults to `[]`.
* <b>`tolerance`</b>: An optional `float`. Defaults to `1e-07`.
* <b>`max_iter`</b>: An optional `int`. Defaults to `1000`.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

A `Tensor`. Has the same type as `lattice_params`.