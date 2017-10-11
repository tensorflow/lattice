<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tensorflow_lattice.lattice" />
</div>

# tensorflow_lattice.lattice

``` python
lattice(
    input_tensor,
    parameter_tensor,
    lattice_sizes,
    interpolation_type='hypercube'
)
```

Returns an interpolated look-up table (lattice) op.

#### Args:

* <b>`input_tensor`</b>: [batch_size, input_dim] tensor.
* <b>`parameter_tensor`</b>: [output_dim, param_dim] tensor, where param_dim ==
    lattice_sizes[0] * ... * lattice_sizes[input_dim - 1].
* <b>`lattice_sizes`</b>: A list of lattice sizes of each dimension.
* <b>`interpolation_type`</b>: 'hypercube' or 'simplex'.


#### Returns:

* <b>`output_tensor`</b>: [batch_size, num_outputs] tensor that contains the output of
  hypercube lattice.


#### Raises:

* <b>`ValueError`</b>: If interpolation_type is not 'hypercube' nor 'simplex'.

