<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tensorflow_lattice.calibrator_regularization" />
</div>

# tensorflow_lattice.calibrator_regularization

``` python
calibrator_regularization(
    output_keypoints,
    l1_reg=None,
    l2_reg=None,
    l1_laplacian_reg=None,
    l2_laplacian_reg=None,
    name='calibrator_regularization'
)
```

Returns a calibrator regularization op.

#### Args:

output_keypoints: (Rank-1 tensor with shape [num_keypoints]) 1d calibrator's
   output keypoints tensor.
l1_reg: (float) l1 regularization amount.
l2_reg: (float) l2 regularization amount.
l1_laplacian_reg: (float) l1 Laplacian regularization amount.
l2_laplacian_reg: (float) l2 Laplacian regularization amount.
name: name scope of calibrator regularization.


#### Returns:

Rank-0 tensor (scalar) that contains calibrator regularization.


#### Raises:

* <b>`ValueError`</b>: * If output_keypoints is not rank-1 tensor.
              * If the shape of output_keypoints is unknown.