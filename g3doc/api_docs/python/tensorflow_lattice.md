<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tensorflow_lattice" />
<meta itemprop="property" content="DEFAULT_NAME"/>
<meta itemprop="property" content="absolute_import"/>
</div>

# Module: tensorflow_lattice

Lattice modeling.

This package provides functions and classes for lattice modeling.

See full description in `README.md` file.


  use them.

## Classes

[`class CalibratedEtlHParams`](./tensorflow_lattice/CalibratedEtlHParams.md): Hyper-parameters for CalibratedEtl (Embedded tiny lattices) models.

[`class CalibratedHParams`](./tensorflow_lattice/CalibratedHParams.md): PerFeatureHParams specialization with input calibration parameters.

[`class CalibratedLatticeHParams`](./tensorflow_lattice/CalibratedLatticeHParams.md): Hyper-parameters for CalibratedLattice models.

[`class CalibratedLinearHParams`](./tensorflow_lattice/CalibratedLinearHParams.md): Hyper-parameters for CalibratedLinear models.

[`class CalibratedRtlHParams`](./tensorflow_lattice/CalibratedRtlHParams.md): Hyper-parameters for CalibratedRtl (RandomTinyLattices) models.

[`class PerFeatureHParams`](./tensorflow_lattice/PerFeatureHParams.md): Parameters object with per feature parametrization.

## Functions

[`calibrated_etl_classifier(...)`](./tensorflow_lattice/calibrated_etl_classifier.md): Calibrated etl binary classifier model.

[`calibrated_etl_regressor(...)`](./tensorflow_lattice/calibrated_etl_regressor.md): Calibrated etl regressor model.

[`calibrated_lattice_classifier(...)`](./tensorflow_lattice/calibrated_lattice_classifier.md): Calibrated lattice classifier binary model.

[`calibrated_lattice_regressor(...)`](./tensorflow_lattice/calibrated_lattice_regressor.md): Calibrated lattice estimator (model) for regression.

[`calibrated_linear_classifier(...)`](./tensorflow_lattice/calibrated_linear_classifier.md): Calibrated linear classifier binary model.

[`calibrated_linear_regressor(...)`](./tensorflow_lattice/calibrated_linear_regressor.md): Calibrated linear estimator (model) for regression.

[`calibrated_rtl_classifier(...)`](./tensorflow_lattice/calibrated_rtl_classifier.md): Calibrated rtl binary classifier model.

[`calibrated_rtl_regressor(...)`](./tensorflow_lattice/calibrated_rtl_regressor.md): Calibrated rtl regressor model.

[`calibration_layer(...)`](./tensorflow_lattice/calibration_layer.md): Creates a calibration layer for uncalibrated values.

[`calibrator_regularization(...)`](./tensorflow_lattice/calibrator_regularization.md): Returns a calibrator regularization op.

[`ensemble_lattices_layer(...)`](./tensorflow_lattice/ensemble_lattices_layer.md): Creates a ensemble of lattices layer.

[`input_calibration_layer(...)`](./tensorflow_lattice/input_calibration_layer.md): Creates a calibration layer for the given input and feature_columns.

[`input_calibration_layer_from_hparams(...)`](./tensorflow_lattice/input_calibration_layer_from_hparams.md): Creates a calibration layer for the input using hyper-parameters.

[`lattice(...)`](./tensorflow_lattice/lattice.md): Returns an interpolated look-up table (lattice) op.

[`lattice_layer(...)`](./tensorflow_lattice/lattice_layer.md): Creates a lattice layer.

[`lattice_regularization(...)`](./tensorflow_lattice/lattice_regularization.md): Returns a lattice regularization op.

[`load_keypoints_from_quantiles(...)`](./tensorflow_lattice/load_keypoints_from_quantiles.md): Retrieves keypoints initialization values for selected features.

[`monotone_lattice(...)`](./tensorflow_lattice/monotone_lattice.md): Returns a projected lattice parameters onto the monotonicity constraints.

[`monotonic_projection(...)`](./tensorflow_lattice/monotonic_projection.md): Returns a not-strict monotonic projection of the vector.

[`pwl_indexing_calibrator(...)`](./tensorflow_lattice/pwl_indexing_calibrator.md): Returns tensor representing interpolation weights in a piecewise linear

[`save_quantiles_for_keypoints(...)`](./tensorflow_lattice/save_quantiles_for_keypoints.md): Calculates and saves quantiles for given features.

[`uniform_keypoints_for_signal(...)`](./tensorflow_lattice/uniform_keypoints_for_signal.md): Returns a pair of initialization tensors for calibration keypoints.

## Other Members

`DEFAULT_NAME`

`absolute_import`

