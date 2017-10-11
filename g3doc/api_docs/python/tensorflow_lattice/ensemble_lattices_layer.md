<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tensorflow_lattice.ensemble_lattices_layer" />
</div>

# tensorflow_lattice.ensemble_lattices_layer

``` python
ensemble_lattices_layer(
    input_tensor,
    lattice_sizes,
    structure_indices,
    is_monotone=None,
    output_dim=1,
    interpolation_type='hypercube',
    lattice_initializers=None,
    l1_reg=None,
    l2_reg=None,
    l1_torsion_reg=None,
    l2_torsion_reg=None,
    l1_laplacian_reg=None,
    l2_laplacian_reg=None
)
```

Creates a ensemble of lattices layer.

Returns a list of output of lattices, lattice parameters, and projection ops.

#### Args:

* <b>`input_tensor`</b>: [batch_size, input_dim] tensor.
* <b>`lattice_sizes`</b>: A list of lattice sizes of each dimension.
* <b>`structure_indices`</b>: A list of list of ints. structure_indices[k] is a list
  of indices that belongs to kth lattices.
* <b>`is_monotone`</b>: A list of input_dim booleans, boolean or None. If None or
    False, lattice will not have monotonicity constraints. If
    is_monotone[k] == True, then the lattice output has the non-decreasing
    monotonicity with respect to input_tensor[?, k] (the kth coordinate). If
    True, all the input coordinate will have the non-decreasing monotonicity.
* <b>`output_dim`</b>: Number of outputs.
* <b>`interpolation_type`</b>: 'hypercube' or 'simplex'.
* <b>`lattice_initializers`</b>: (Optional) A list of initializer for each lattice
    parameter vectors. lattice_initializer[k] is a 2D tensor
    [output_dim, parameter_dim[k]], where parameter_dim[k] is the number of
    parameter in the kth lattice. If None, lattice_param_as_linear initializer
    will be used with
    linear_weights=[1 if monotone else 0 for monotone in is_monotone].
* <b>`l1_reg`</b>: (float) l1 regularization amount.
* <b>`l2_reg`</b>: (float) l2 regularization amount.
* <b>`l1_torsion_reg`</b>: (float) l1 torsion regularization amount.
* <b>`l2_torsion_reg`</b>: (float) l2 torsion regularization amount.
* <b>`l1_laplacian_reg`</b>: (list of floats or float) list of L1 Laplacian
     regularization amount per each dimension. If a single float value is
     provided, then all diemnsion will get the same value.
* <b>`l2_laplacian_reg`</b>: (list of floats or float) list of L2 Laplacian
     regularization amount per each dimension. If a single float value is
     provided, then all diemnsion will get the same value.


#### Returns:

A tuple of:
* a list of output tensors, [batch_size, output_dim], with length
  len(structure_indices), i.e., one for each lattice.
* a list of parameter tensors shape [output_dim, parameter_dim]
* None or projection ops, that must be applied at each
  step (or every so many steps) to project the model to a feasible space:
  used for bounding the outputs or for imposing monotonicity.
* None or a regularization loss, if regularization is configured.