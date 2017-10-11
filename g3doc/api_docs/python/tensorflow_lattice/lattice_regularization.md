<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tensorflow_lattice.lattice_regularization" />
</div>

# tensorflow_lattice.lattice_regularization

``` python
lattice_regularization(
    lattice_params,
    lattice_sizes,
    l1_reg=None,
    l2_reg=None,
    l1_torsion_reg=None,
    l2_torsion_reg=None,
    l1_laplacian_reg=None,
    l2_laplacian_reg=None,
    name='lattice_regularization'
)
```

Returns a lattice regularization op.

#### Args:

lattice_params: (Rank-2 tensor with shape [output_dim, param_dim]) Lattice
  parameter tensor.
lattice_sizes: (list of integers) lattice size of each dimension.
l1_reg: (float) l1 regularization amount.
l2_reg: (float) l2 regularization amount.
l1_torsion_reg: (float) l1 torsion regularization amount.
l2_torsion_reg: (float) l2 torsion regularization amount.
l1_laplacian_reg: (list of floats or float) list of L1 Laplacian
  regularization amount per each dimension. If a single float value is
  provided, then all diemnsion will get the same value.
l2_laplacian_reg: (list of floats or float) list of L2 Laplacian
  regularization amount per each dimension. If a single float value is
  provided, then all diemnsion will get the same value.
name: name scope of lattice regularization.


#### Returns:

Rank-0 tensor (scalar) that contains lattice regularization.


#### Raises:

* <b>`ValueError`</b>: * lattice_param is not rank-2 tensor.
              * output_dim or param_dim is unknown.