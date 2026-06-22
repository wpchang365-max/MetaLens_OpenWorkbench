# Physics Audit v5.1.1

## v5.1.1 image-plane clarification

- Letter-E geometry is now parameterized by overall width, overall height, stroke width and middle-arm length ratio. Numerical point spacing is explicitly a discretization control rather than a physical glyph dimension.
- The projection interface accepts object distance only. For a real image, `u > f` is enforced and the image distance is derived from `1/f = 1/u + 1/v`. The calibrated transverse magnification is `-angular_magnification * v/u`.
- In normal mode the design preview is the focal-plane PSF. In projection-lithography mode it is the Letter-E aerial image at the derived image plane. Exports preserve both arrays under distinct names.

## v5.1 lithography and phase audit

- Projection lithography contrast is evaluated from the image-side Letter-E pattern using robust foreground/background separation, structural correlation, stroke uniformity, shape overlap and explicit pixel resolvability. A high scalar Michelson contrast alone is not treated as readable imaging.
- The reported sidelobe ratio is the peak PSF intensity outside an adaptively measured main-lobe radius divided by the PSF peak. In lithography it is a secondary flare/proximity-exposure indicator; CD, NILS, exposure window and pattern fidelity remain distinct quantities.
- Propagation phase follows optical-path phase and inverse-wavelength scaling. Geometric phase follows the ideal Pancharatnam-Berry relation phi = 2 sigma theta and is not assigned propagation-phase chromatic scaling; conversion efficiency is applied to pupil amplitude.
- Letter-E geometry maps to the image plane with magnification m = angular_magnification * image_distance / object_distance. Distances and image sampling therefore affect resolvability.
- CUDA uses CuPy FFTs, a bounded memory pool and source-sample batching. CPU fallback remains available when CUDA/OpenCL runtimes are absent.

## Model boundaries

- The interactive propagation engine is scalar Fourier/Fresnel/angular-spectrum optics with an explicitly labelled fast vector-weighted Debye approximation.
- RCWA/FDTD is not impersonated by the fast engine. Complex transmission coefficients must be imported from an external full-wave solver or measurement database.
- LED spectral and angular samples are combined incoherently; coherent laser fields are propagated before intensity evaluation.
- Jones matrices represent deterministic polarization transformations. Mueller conversion uses the Pauli-matrix trace definition.

## Formula checks

- Fresnel kernel: exp(i k r^2 / 2z), output factor exp(i k z) exp(i k r_1^2 / 2z)/(i lambda z).
- Angular spectrum: H(fx,fy)=exp(i 2 pi z sqrt(1/lambda^2-fx^2-fy^2)); evanescent terms are filtered in the rapid far-field path.
- FWHM uses linearly interpolated half-maximum crossings.
- Conventional Letter-E Michelson contrast remains available as a diagnostic using median target/background intensity.
- The optimization metric is now a weighted geometric mean of P10-foreground/P95-background robust contrast, target correlation, stroke uniformity and binary Dice overlap. This prevents a high median contrast from hiding broken strokes or local background hot spots.
- A separate resolvability factor checks image-plane glyph width/height, physical stroke width and bar spacing in pixels. Undersampled glyphs are strongly downweighted even if their tiny raster mask matches a bright spot.
- NILS is CD times the normalized image-log-slope at the resist threshold.
- Critical dimension is the median printed run width at the selected threshold.
- Depth of focus is the contiguous/qualified z span satisfying the configured CD tolerance.
- Phase error is wrapped to [-pi, pi]. Meta-atom efficiency is |t|^2.

## Limitations that remain explicit

- The Fourier adjoint optimizes a scalar Fourier-plane complex-field objective, not a Maxwell volume solve.
- Fabrication perturbation response requires either a user evaluator/full-wave database or the UI screening surrogate; publication-grade tolerance claims should use the former.
- GDS export currently emits rectangular boundaries. Curved/freeform geometry should be polygonized by a dedicated process-design-kit exporter.
