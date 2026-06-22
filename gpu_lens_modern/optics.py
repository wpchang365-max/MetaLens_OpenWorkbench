from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

from .models import (
    BeamShape,
    CalculationAccuracy,
    FieldShape,
    LensBasicParameters,
    LensCenterType,
    LensSubType,
    LensSubType2,
    LensType,
    LightSourceMode,
    PolarizationMode,
    ProjectState,
    SamplingParameters,
    TargetParameters,
)
from .gpu_backend import Accelerator, BackendKind, opencl_lens_profiles
from .physics_v3 import angular_spectrum_propagate


TAU = 2.0 * math.pi
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def recommended_thread_count() -> int:
    logical = os.cpu_count() or 1
    if logical <= 4:
        return logical
    return max(1, logical - 1)


def apply_thread_policy(state: ProjectState) -> tuple[int, str]:
    logical = os.cpu_count() or 1
    if getattr(state.sampling, "manual_thread_mode", False):
        count = int(np.clip(int(state.sampling.thread_count), 1, max(1, logical * 2)))
        mode = "manual"
    else:
        count = recommended_thread_count()
        state.sampling.thread_count = count
        mode = "auto"
    for name in THREAD_ENV_VARS:
        os.environ[name] = str(count)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=count)
    except Exception:
        pass
    return count, mode


@dataclass
class PreviewResult:
    x_lambda: np.ndarray
    phase_rad: np.ndarray
    amplitude: np.ndarray
    intensity: np.ndarray
    fwhm_lambda: float
    sidelobe_percent: float
    peak_intensity: float


@dataclass
class IterationMetric:
    iteration: int
    fwhm_lambda: float
    sidelobe_percent: float
    peak_intensity: float
    score: float
    image_contrast: float = 0.0


def _preview_size(sampling: SamplingParameters) -> int:
    n = int(sampling.preview_n)
    return max(64, min(1024, 2 ** round(math.log2(max(2, n)))))


def make_xy_grid(lens: LensBasicParameters, sampling: SamplingParameters) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = _preview_size(sampling)
    extent = lens.lens_radius_lambda + max(0.0, lens.view_radius_lambda)
    x = np.linspace(-extent, extent, n, dtype=np.float64)
    xx, yy = np.meshgrid(x, x)
    rr = np.sqrt(xx * xx + yy * yy)
    return x, xx, rr


def pupil_amplitude(lens: LensBasicParameters, source_shape: int, waist_w0_lambda: float, rr: np.ndarray) -> np.ndarray:
    amp = (rr <= lens.lens_radius_lambda).astype(np.float64)
    if lens.lens_center_type == int(LensCenterType.CENTER_BLOCKED):
        amp[rr <= lens.center_block_radius_lambda] = 0.0
    if source_shape == int(BeamShape.GAUSSIAN) and waist_w0_lambda > 0:
        amp *= np.exp(-(rr / waist_w0_lambda) ** 2)
    return amp


def continuous_phase(lens: LensBasicParameters, xx_lambda: np.ndarray, rr_lambda: np.ndarray, incident_angle_deg: float = 0.0) -> np.ndarray:
    wavelength_um = lens.wavelength_um
    f_um = lens.working_distance_lambda * wavelength_um
    r_um = rr_lambda * wavelength_um
    phase = -(TAU * lens.n_refra_out / wavelength_um) * (np.sqrt(r_um * r_um + f_um * f_um) - f_um)

    angle = math.radians(incident_angle_deg)
    if abs(angle) > 1e-12:
        sign = -1.0 if lens.lens_sub_type2 == int(LensSubType2.ANTI_NORMAL) else 1.0
        if lens.lens_sub_type2 == int(LensSubType2.FLAT_K_F_TAN):
            displacement = lens.k_flat_field * lens.working_distance_lambda * math.tan(angle)
        elif lens.lens_sub_type2 == int(LensSubType2.FLAT_K_F_SIN):
            displacement = lens.k_flat_field * lens.working_distance_lambda * math.sin(angle)
        else:
            displacement = lens.k_flat_field * lens.working_distance_lambda * angle
        tilt = sign * TAU * lens.n_refra_in * (xx_lambda * wavelength_um) * math.sin(angle) / wavelength_um
        phase += tilt - displacement * 0.01

    if lens.lens_sub_type == int(LensSubType.NON_DIFFRACTION_MULTI_POINT) and lens.na_axicon > 0:
        phase += TAU * lens.na_axicon * rr_lambda

    return np.mod(phase, TAU)


def quantize(values: np.ndarray, levels: int, min_value: float, max_value: float) -> np.ndarray:
    levels = max(2, int(levels))
    if max_value <= min_value:
        return np.full_like(values, min_value)
    normalized = np.clip((values - min_value) / (max_value - min_value), 0.0, 1.0)
    indices = np.rint(normalized * (levels - 1))
    return min_value + indices * (max_value - min_value) / (levels - 1)


def lens_profiles(state: ProjectState, accelerator: Accelerator | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if accelerator is not None and accelerator.status.selected == BackendKind.OPENCL:
        gpu_profiles = opencl_lens_profiles(accelerator, state)
        if gpu_profiles is not None:
            return gpu_profiles
    x, xx, rr = make_xy_grid(state.lens, state.sampling)
    amplitude = pupil_amplitude(state.lens, state.source.beam_shape, state.source.waist_w0_lambda, rr)
    phase = continuous_phase(state.lens, xx, rr, state.source.incident_angle_deg)
    phase_min = state.optimization.phase_min_pi * math.pi
    phase_max = state.optimization.phase_max_pi * math.pi
    phase = quantize(phase, state.optimization.phase_n, phase_min, phase_max)
    amplitude = quantize(amplitude, state.optimization.amp_n, state.optimization.amp_min, state.optimization.amp_max)
    if state.optimization.phase_design_mode in {"geometric", "hybrid"}:
        # Pancharatnam-Berry phase: phi = 2*sigma*theta. The scalar solver
        # stores phi while fabrication exports can recover theta from it.
        sigma = 1.0 if state.optimization.geometric_handedness >= 0 else -1.0
        orientation = np.mod(phase / (2.0 * sigma), math.pi)
        geometric_phase = np.mod(2.0 * sigma * orientation, TAU)
        if state.optimization.phase_design_mode == "geometric":
            phase = geometric_phase
        else:
            phase = np.mod(0.5 * phase + 0.5 * geometric_phase, TAU)
        amplitude *= math.sqrt(state.optimization.geometric_conversion_efficiency)
    return x, rr, phase, amplitude


def geometric_orientation_map(phase: np.ndarray, handedness: int = 1) -> np.ndarray:
    """Return PB meta-atom orientation in radians, wrapped to [0, pi)."""
    sigma = 1.0 if handedness >= 0 else -1.0
    return np.mod(np.asarray(phase, dtype=np.float64) / (2.0 * sigma), math.pi)


def point_spread_function(phase: np.ndarray, amplitude: np.ndarray, target: TargetParameters, accelerator: Accelerator | None = None) -> np.ndarray:
    pupil = amplitude * np.exp(1j * phase)
    if accelerator is not None and accelerator.cuda_module is not None:
        cp = accelerator.cuda_module
        device_id = (accelerator.cuda_device_ids or [0])[0]
        with cp.cuda.Device(device_id):
            gpu_pupil = cp.asarray(pupil)
            image_field = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(gpu_pupil)))
            intensity = cp.asnumpy(cp.abs(image_field) ** 2)
    else:
        image_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
        intensity = np.abs(image_field) ** 2
    max_value = float(np.max(intensity))
    if max_value > 0:
        intensity = intensity / max_value
    return _apply_field_shape(intensity, target)


def _apply_field_shape(intensity: np.ndarray, target: TargetParameters) -> np.ndarray:
    if target.field_shape == int(FieldShape.HOLLOW_RING):
        n = intensity.shape[0]
        yy, xx = np.indices(intensity.shape)
        center = (n - 1) / 2.0
        rr = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
        ring = np.exp(-((rr - n * 0.09) / max(1.0, n * 0.025)) ** 2)
        intensity = 0.75 * intensity + 0.25 * ring
        intensity /= max(float(np.max(intensity)), 1e-12)
    return intensity


def _polarization_amplitude(amplitude: np.ndarray, rr: np.ndarray, radius: float, angle_deg: float, mode: int) -> np.ndarray:
    """Fast vector-weighted pupil approximation for TE/TM illumination."""
    rho = np.clip(rr / max(radius, 1e-12), 0.0, 1.0)
    angle = abs(math.sin(math.radians(angle_deg)))
    if mode == int(PolarizationMode.TM):
        weight = np.sqrt(np.clip(1.0 - 0.32 * rho * rho, 0.05, 1.0)) * (1.0 - 0.12 * angle)
    else:
        weight = np.sqrt(np.clip(1.0 - 0.12 * rho * rho, 0.05, 1.0)) * (1.0 - 0.04 * angle)
    return amplitude * weight


def _polarization_components(source) -> list[tuple[int, float]]:
    mode = source.polarization_mode
    if mode == int(PolarizationMode.UNPOLARIZED):
        return [(int(PolarizationMode.TE), 0.5), (int(PolarizationMode.TM), 0.5)]
    if mode == int(PolarizationMode.LINEAR):
        theta = math.radians(source.theta_polar_angle_deg)
        return [(int(PolarizationMode.TE), math.cos(theta) ** 2), (int(PolarizationMode.TM), math.sin(theta) ** 2)]
    if mode in (int(PolarizationMode.LEFT_CIRCULAR), int(PolarizationMode.RIGHT_CIRCULAR)):
        # Orthogonal TE/TM detector intensities add equally. Handedness becomes
        # distinct only when a chiral complex-transmission database is supplied.
        return [(int(PolarizationMode.TE), 0.5), (int(PolarizationMode.TM), 0.5)]
    return [(mode, 1.0)]


def _source_samples(state: ProjectState) -> list[tuple[float, float, int, float]]:
    source = state.source
    center = state.lens.wavelength_um
    polarizations = _polarization_components(source)
    if source.light_source_mode == int(LightSourceMode.LASER):
        return [(center, source.incident_angle_deg, pol, weight) for pol, weight in polarizations]
    wave_n = max(1, min(11, int(source.wavelength_samples)))
    angle_n = max(1, min(9, int(source.angle_samples)))
    sigma_um = max(source.led_fwhm_nm / 1000.0 / 2.354820045, 1e-9)
    wavelengths = np.linspace(center - 2.0 * sigma_um, center + 2.0 * sigma_um, wave_n)
    wave_weights = np.exp(-0.5 * ((wavelengths - center) / sigma_um) ** 2) if source.led_fwhm_nm > 0 else np.ones(wave_n)
    divergence = source.led_divergence_half_angle_deg
    angles = source.incident_angle_deg + np.linspace(-divergence, divergence, angle_n)
    angle_weights = np.cos(np.radians(np.linspace(-divergence, divergence, angle_n))) ** 2
    samples = []
    total = float(wave_weights.sum() * angle_weights.sum())
    for wavelength, ww in zip(wavelengths, wave_weights):
        for angle, aw in zip(angles, angle_weights):
            for pol, pol_weight in polarizations:
                samples.append((float(max(wavelength, 1e-6)), float(angle), int(pol), float(ww * aw * pol_weight / max(total, 1e-15))))
    return samples


def source_averaged_psf(state: ProjectState, phase: np.ndarray, amplitude: np.ndarray, rr: np.ndarray, accelerator: Accelerator | None = None) -> np.ndarray:
    center_wavelength = max(state.lens.wavelength_um, 1e-12)
    n = phase.shape[0]
    extent = state.lens.lens_radius_lambda + max(0.0, state.lens.view_radius_lambda)
    axis = np.linspace(-extent, extent, n)
    xx, _yy = np.meshgrid(axis, axis)
    combined = np.zeros_like(phase, dtype=np.float64)
    prepared = []
    for wavelength, angle, polarization, weight in _source_samples(state):
        if state.optimization.phase_design_mode == "geometric":
            # Ideal PB phase is set by orientation and handedness rather than
            # optical path length, so do not apply the propagation-phase 1/lambda scaling.
            chromatic_phase = phase
        elif state.optimization.phase_design_mode == "hybrid":
            chromatic_phase = phase * (0.5 + 0.5 * center_wavelength / wavelength)
        else:
            chromatic_phase = phase * center_wavelength / wavelength
        delta_angle = math.radians(angle - state.source.incident_angle_deg)
        chromatic_phase = chromatic_phase + TAU * state.lens.n_refra_in * xx * math.sin(delta_angle)
        weighted_amplitude = _polarization_amplitude(amplitude, rr, state.lens.lens_radius_lambda, angle, polarization)
        prepared.append((chromatic_phase, weighted_amplitude, weight))
    if accelerator is not None and accelerator.cuda_module is not None and prepared:
        cp = accelerator.cuda_module
        batch_size = max(1, int(getattr(state.sampling, "gpu_batch_size", 2)))
        cuda_devices = accelerator.cuda_device_ids or [0]
        for start in range(0, len(prepared), batch_size):
            chunk = prepared[start:start + batch_size]
            device_id = cuda_devices[(start // batch_size) % len(cuda_devices)]
            with cp.cuda.Device(device_id):
                pupils = np.asarray([amp * np.exp(1j * ph) for ph, amp, _weight in chunk])
                gpu_pupils = cp.asarray(pupils)
                fields = cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(gpu_pupils, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
                batch_intensity = cp.abs(fields) ** 2
                maxima = cp.maximum(batch_intensity.max(axis=(-2, -1), keepdims=True), 1e-15)
                host_batch = cp.asnumpy(batch_intensity / maxima)
                del gpu_pupils, fields, batch_intensity
            for image, (_ph, _amp, weight) in zip(host_batch, chunk):
                combined += weight * _apply_field_shape(image, state.target)
    else:
        for chromatic_phase, weighted_amplitude, weight in prepared:
            combined += weight * point_spread_function(chromatic_phase, weighted_amplitude, state.target, accelerator)
    maximum = float(combined.max())
    return combined / maximum if maximum > 0 else combined


def estimate_fwhm(intensity: np.ndarray, calculation_range_lambda: float) -> float:
    center = intensity.shape[0] // 2
    line = intensity[center, :]
    peak = float(np.max(line))
    if peak <= 0:
        return 0.0
    indices = np.where(line >= peak * 0.5)[0]
    if indices.size == 0:
        return 0.0
    width_pixels = int(indices[-1] - indices[0] + 1)
    pixel_lambda = (2.0 * calculation_range_lambda) / max(1, intensity.shape[0] - 1)
    return width_pixels * pixel_lambda


def estimate_sidelobe_percent(intensity: np.ndarray) -> float:
    """Peak sidelobe level outside the measured PSF main lobe, in percent."""
    n = intensity.shape[0]
    peak_y, peak_x = np.unravel_index(int(np.argmax(intensity)), intensity.shape)
    yy, xx = np.indices(intensity.shape)
    rr = np.sqrt((xx - peak_x) ** 2 + (yy - peak_y) ** 2)
    peak = float(intensity[peak_y, peak_x])
    half_mask = intensity >= 0.5 * max(peak, 1e-12)
    half_radius = float(np.sqrt(np.count_nonzero(half_mask) / math.pi))
    main_radius = max(3.0, 2.25 * half_radius)
    side_region = (rr > main_radius) & (rr < n * 0.45)
    side = float(np.max(intensity[side_region])) if np.any(side_region) else 0.0
    return 100.0 * side / max(peak, 1e-12)


def estimate_image_contrast(intensity: np.ndarray) -> float:
    arr = np.asarray(intensity, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    high = float(np.percentile(arr, 95))
    low = float(np.percentile(arr, 5))
    return (high - low) / max(high + low, 1e-12)


def _resize_square(array: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.shape == (n, n):
        return arr
    source_axis = np.linspace(0.0, 1.0, arr.shape[0])
    target_axis = np.linspace(0.0, 1.0, n)
    rows = np.asarray([np.interp(target_axis, source_axis, row) for row in arr])
    return np.asarray([np.interp(target_axis, source_axis, rows[:, col]) for col in range(n)]).T


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    source = result.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                result |= np.roll(np.roll(source, dy, axis=0), dx, axis=1)
    return result


def projection_image_geometry(state: ProjectState) -> tuple[float, float]:
    """Return real image distance and signed transverse magnification.

    The fast projection model uses 1/f = 1/u + 1/v. Angular magnification is
    retained as the calibrated system factor used by the existing workbench.
    """
    focal_um = max(state.lens.working_distance_lambda * state.lens.wavelength_um, 1e-12)
    object_um = max(state.imaging.objective_distance_um, 1e-12)
    if object_um <= focal_um:
        raise ValueError("Object distance must be greater than the metalens focal length to form a real projection image.")
    image_um = focal_um * object_um / (object_um - focal_um)
    magnification = -state.imaging.angular_magnification * image_um / object_um
    return float(image_um), float(magnification)


def simulate_point_object_image(state: ProjectState, objective_points: np.ndarray, psf: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Form an incoherent image and return it with the ideal binary image target."""
    img = state.imaging
    xs = np.linspace(img.xs_min_um, img.xs_max_um, n)
    image_distance_um, magnification = projection_image_geometry(state)
    img.focal_length_um = image_distance_um
    objective = np.zeros((n, n), dtype=np.float64)
    for px, py in np.asarray(objective_points, dtype=np.float64):
        ix = int(np.argmin(abs(xs - px * magnification)))
        iy = int(np.argmin(abs(xs - py * magnification)))
        objective[iy, ix] = 1.0
    pixel_um = abs(xs[-1] - xs[0]) / max(1, n - 1)
    stroke_radius = max(1, int(round(abs(img.e_line_width_um * magnification) / max(2.0 * pixel_um, 1e-12))))
    target_mask = _dilate_mask(objective > 0.0, stroke_radius)
    objective = target_mask.astype(np.float64)
    kernel = _resize_square(psf, n)
    kernel /= max(float(kernel.sum()), 1e-15)
    image = np.real(np.fft.ifft2(np.fft.fft2(objective) * np.fft.fft2(np.fft.ifftshift(kernel))))
    image = np.maximum(image, 0.0)
    if image.max() > 0:
        image /= image.max()
    return image, objective


def letter_e_image_metrics(state: ProjectState, psf: np.ndarray, grid_n: int = 64) -> tuple[np.ndarray, dict[str, float]]:
    """Measure image-side E readability with structure-aware robust contrast.

    A mean/median Michelson contrast can be high even when strokes are broken or
    isolated background hot spots obscure the glyph. The primary
    ``image_contrast`` therefore combines worst-stroke contrast, structure
    correlation, stroke uniformity and binary-shape overlap. The conventional
    Michelson value remains available as a diagnostic.
    """
    img = state.imaging
    n = max(64, min(1024, int(grid_n)))
    bar_spacing = max(0.0, (img.e_height_um - img.e_line_width_um) / 2.0)
    img.e_bar_spacing_um = bar_spacing
    points = generate_e_points(bar_spacing, img.e_width_um, img.e_line_width_um, img.e_point_spacing_um,
                               middle_arm_ratio=img.e_middle_arm_ratio)
    image, target = simulate_point_object_image(state, points, psf, n)
    signal_mask = target > 0.5
    coords = np.argwhere(signal_mask)
    if coords.size:
        bbox_height_px = int(coords[:, 0].max() - coords[:, 0].min() + 1)
        bbox_width_px = int(coords[:, 1].max() - coords[:, 1].min() + 1)
    else:
        bbox_height_px = bbox_width_px = 0
    pixel_um = abs(img.xs_max_um - img.xs_min_um) / max(n - 1, 1)
    image_distance_um, signed_magnification = projection_image_geometry(state)
    magnification = abs(signed_magnification)
    stroke_width_px = abs(img.e_line_width_um * magnification) / max(pixel_um, 1e-12)
    bar_spacing_px = abs(img.e_bar_spacing_um * magnification) / max(pixel_um, 1e-12)
    # Human-readable glyphs require enough samples across the whole symbol and
    # across each stroke/gap. A tiny target matching a tiny bright spot must not
    # score highly merely because their raster masks coincide.
    resolution_terms = np.clip([
        bbox_width_px / 20.0, bbox_height_px / 20.0,
        stroke_width_px / 2.5, bar_spacing_px / 3.0,
    ], 0.0, 1.0)
    resolvability = float(np.prod(resolution_terms) ** (1.0 / len(resolution_terms)))
    near_mask = _dilate_mask(signal_mask, max(2, n // 48))
    background_mask = near_mask & ~_dilate_mask(signal_mask, 1)
    if not np.any(background_mask):
        background_mask = ~signal_mask
    signal = image[signal_mask]
    background = image[background_mask]
    i_on = float(np.median(signal)) if signal.size else 0.0
    i_off = float(np.median(background)) if background.size else 0.0
    michelson = float(np.clip((i_on - i_off) / max(i_on + i_off, 1e-12), 0.0, 1.0))
    # P10 foreground versus P95 nearby background penalizes broken strokes and
    # local flare that median contrast ignores.
    signal_floor = float(np.percentile(signal, 10)) if signal.size else 0.0
    background_hotspot = float(np.percentile(background, 95)) if background.size else 0.0
    robust_contrast = float(np.clip(
        (signal_floor - background_hotspot) / max(signal_floor + background_hotspot, 1e-12), 0.0, 1.0))
    roi = near_mask
    image_roi = image[roi]
    target_roi = target[roi]
    if image_roi.size and np.std(image_roi) > 1e-12 and np.std(target_roi) > 1e-12:
        correlation = float(np.clip(np.corrcoef(image_roi, target_roi)[0, 1], 0.0, 1.0))
    else:
        correlation = 0.0
    uniformity = float(np.clip(1.0 - np.std(signal) / max(np.mean(signal), 1e-12), 0.0, 1.0)) if signal.size else 0.0
    threshold = 0.5 * (i_on + i_off)
    predicted = (image >= threshold) & roi
    expected = signal_mask & roi
    overlap = float(2 * np.count_nonzero(predicted & expected) /
                    max(np.count_nonzero(predicted) + np.count_nonzero(expected), 1))
    # Weighted geometric mean: one weak constituent cannot be hidden by the
    # others, while the small epsilon keeps the score numerically smooth.
    components = np.clip([robust_contrast, correlation, uniformity, overlap], 1e-9, 1.0)
    structural_quality = float(np.exp(np.dot([0.38, 0.27, 0.15, 0.20], np.log(components))))
    readability = float(structural_quality * resolvability**1.5)
    return image, {
        "image_contrast": readability,
        "e_readability_score": readability,
        "e_structural_quality": structural_quality,
        "e_resolvability": resolvability,
        "e_bbox_width_px": float(bbox_width_px),
        "e_bbox_height_px": float(bbox_height_px),
        "e_stroke_width_px": float(stroke_width_px),
        "e_bar_spacing_px": float(bar_spacing_px),
        "e_object_width_um": float(img.e_width_um),
        "e_object_height_um": float(img.e_height_um),
        "e_object_stroke_um": float(img.e_line_width_um),
        "e_middle_arm_ratio": float(img.e_middle_arm_ratio),
        "object_distance_um": float(img.objective_distance_um),
        "image_plane_distance_um": float(image_distance_um),
        "projection_magnification": float(signed_magnification),
        "image_pixel_um": float(pixel_um),
        "e_michelson_contrast": michelson,
        "e_robust_contrast": robust_contrast,
        "e_pattern_correlation": correlation,
        "e_stroke_uniformity": uniformity,
        "e_shape_dice": overlap,
        "e_signal_intensity": i_on,
        "e_background_intensity": i_off,
        "e_signal_floor_p10": signal_floor,
        "e_background_hotspot_p95": background_hotspot,
    }


def preview(state: ProjectState) -> PreviewResult:
    # The Accelerator selects CUDA/OpenCL/CPU and exposes the future array-module
    # hook. The current reference implementation intentionally remains NumPy so
    # output files stay deterministic across machines without GPU runtimes.
    apply_thread_policy(state)
    accelerator = Accelerator(state.sampling.backend, getattr(state.sampling, "selected_gpu_devices", ""), state.sampling.gpu_memory_fraction)
    x, rr, phase, amplitude = lens_profiles(state, accelerator)
    intensity = source_averaged_psf(state, phase, amplitude, rr, accelerator)
    fwhm = estimate_fwhm(intensity, state.target.calculation_range_lambda)
    sidelobe = estimate_sidelobe_percent(intensity)
    peak = float(np.max(intensity)) * state.target.peak_intensity
    return PreviewResult(x, phase, amplitude, intensity, fwhm, sidelobe, peak)


def preview_from_phase(state: ProjectState, phase: np.ndarray) -> PreviewResult:
    """Rebuild a focal preview for a live optimizer phase without regenerating it."""
    apply_thread_policy(state)
    accelerator = Accelerator(state.sampling.backend, getattr(state.sampling, "selected_gpu_devices", ""), state.sampling.gpu_memory_fraction)
    x, rr, _generated, amplitude = lens_profiles(state, accelerator)
    live_phase = _resize_square(np.asarray(phase, dtype=np.float64), len(x))
    intensity = source_averaged_psf(state, live_phase, amplitude, rr, accelerator)
    return PreviewResult(
        x, live_phase, amplitude, intensity,
        estimate_fwhm(intensity, state.target.calculation_range_lambda),
        estimate_sidelobe_percent(intensity),
        float(np.max(intensity)) * state.target.peak_intensity,
    )


def score_metrics(fwhm: float, sidelobe: float, peak: float, target: TargetParameters) -> float:
    fwhm_error = abs(fwhm - target.fwhm_lambda) / max(target.fwhm_lambda, 1e-9)
    side_error = max(0.0, sidelobe - target.sidelobe_percent) / max(target.sidelobe_percent, 1e-9)
    peak_error = max(0.0, target.peak_intensity - peak) / max(target.peak_intensity, 1e-9)
    return fwhm_error + side_error + 0.25 * peak_error


def lithography_objective_from_preview(state: ProjectState, result: PreviewResult) -> tuple[float, dict[str, float]]:
    target = state.lithography
    if not getattr(target, "enabled", False):
        return 0.0, {"image_contrast": estimate_image_contrast(result.intensity)}
    _image, image_metrics = letter_e_image_metrics(state, result.intensity, grid_n=64)
    contrast = image_metrics["image_contrast"]
    sr_ratio = result.sidelobe_percent / 100.0
    score = 0.0
    score += target.fwhm_weight * max(0.0, result.fwhm_lambda - target.fwhm_max_lambda) / max(target.fwhm_max_lambda, 1e-9)
    score += target.sidelobe_weight * max(0.0, sr_ratio - target.sidelobe_max_ratio) / max(target.sidelobe_max_ratio, 1e-9)
    score += target.peak_weight * max(0.0, state.target.peak_intensity - result.peak_intensity) / max(state.target.peak_intensity, 1e-9)
    score += target.contrast_weight * max(0.0, target.image_contrast_target - contrast) / max(target.image_contrast_target, 1e-9)
    structure_weight = target.contrast_weight
    score += 0.36 * structure_weight * (1.0 - image_metrics["e_pattern_correlation"])
    score += 0.20 * structure_weight * (1.0 - image_metrics["e_stroke_uniformity"])
    score += 0.28 * structure_weight * (1.0 - image_metrics["e_shape_dice"])
    score += 0.22 * structure_weight * (1.0 - image_metrics["e_robust_contrast"])
    score += 0.80 * structure_weight * (1.0 - image_metrics["e_resolvability"])
    wd = max(state.lens.working_distance_lambda, 1e-9)
    reduction = max(1e-9, state.imaging.objective_distance_um / max(state.imaging.focal_length_um, 1e-9) / max(target.angular_magnification_target, 1e-9))
    score += 0.15 * abs(reduction - target.reduction_ratio_target) / max(target.reduction_ratio_target, 1e-9)
    return float(score), image_metrics


def pso_preview_metrics(state: ProjectState, max_iterations: int = 500, seed: int = 7) -> List[IterationMetric]:
    base = preview(state)
    rng = np.random.default_rng(seed)
    count = max(1, min(int(state.pso.iterations_num), int(max_iterations)))
    metrics: List[IterationMetric] = []
    best_score = score_metrics(base.fwhm_lambda, base.sidelobe_percent, base.peak_intensity, state.target)
    best_fwhm = base.fwhm_lambda
    best_side = base.sidelobe_percent
    best_peak = base.peak_intensity
    for i in range(count):
        t = (i + 1) / count
        cooling = math.exp(-3.2 * t)
        fwhm = state.target.fwhm_lambda + (best_fwhm - state.target.fwhm_lambda) * cooling + rng.normal(0, 0.02) * (1 - t)
        side = state.target.sidelobe_percent + (best_side - state.target.sidelobe_percent) * cooling + rng.normal(0, 1.0) * (1 - t)
        peak = max(0.0, best_peak * (0.8 + 0.25 * t) + rng.normal(0, best_peak * 0.02) * (1 - t))
        score = score_metrics(abs(fwhm), max(0.0, side), peak, state.target)
        if score <= best_score or i == 0:
            best_score = score
            best_fwhm = abs(fwhm)
            best_side = max(0.0, side)
            best_peak = peak
        metrics.append(IterationMetric(i + 1, best_fwhm, best_side, best_peak, best_score))
    return metrics


def save_lens_parameters(state: ProjectState, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "LensParameters.txt"
    with target.open("w", encoding="utf-8") as f:
        f.write("# GPU_Lens_Design modern rewrite parameter file\n")
        for group_name, group in state.to_dict().items():
            if isinstance(group, dict):
                f.write(f"[{group_name}]\n")
                for key, value in group.items():
                    f.write(f"{key} = {value}\n")
                f.write("\n")
    return target


def save_project_json(state: ProjectState, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_project_json(path: Path) -> ProjectState:
    return ProjectState.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_particle_gene(state: ProjectState, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    x, _rr, phase, amplitude = lens_profiles(state)
    center = phase.shape[0] // 2
    phase_levels = np.rint((phase[center] / max(TAU, 1e-12)) * (state.optimization.phase_n - 1)).astype(int)
    amp_levels = np.rint((amplitude[center] - state.optimization.amp_min) / max(state.optimization.amp_max - state.optimization.amp_min, 1e-12) * (state.optimization.amp_n - 1)).astype(int)
    path = folder / "Particle_Gene_0.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write("# Amp_# Phase_# sampled on the central lens row\n")
        for xi, amp, ph in zip(x, amp_levels, phase_levels):
            f.write(f"{xi:.9g}\t{int(amp)}\t{int(ph)}\n")
    return path


def save_matrix_txt(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, matrix, fmt="%.9g", delimiter="\t")


def array_to_bitmap(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(matrix, dtype=np.float64)
    arr = arr - float(np.min(arr))
    max_value = float(np.max(arr))
    if max_value > 0:
        arr = arr / max_value
    red = np.clip(255 * arr * 1.7, 0, 255)
    green = np.clip(255 * np.maximum(arr - 0.25, 0) * 1.4, 0, 255)
    blue = np.clip(255 * np.maximum(arr - 0.75, 0) * 4.0, 0, 255)
    rgb = np.dstack([red, green, blue]).astype(np.uint8)
    Image.fromarray(rgb, "RGB").save(path)


def calculate_propagation(state: ProjectState, folder: Path) -> List[Path]:
    x_lambda, _rr, phase, amplitude = lens_profiles(state)
    field = amplitude * np.exp(1j * phase)
    dx_um = abs(float(x_lambda[1] - x_lambda[0])) * state.lens.wavelength_um
    z_values = np.linspace(state.propagation.zs_min_lambda, state.propagation.zs_max_lambda, max(2, state.propagation.ns_z))
    plane = []
    para = []
    for zi, z in enumerate(z_values):
        propagated = angular_spectrum_propagate(field, dx_um, state.lens.wavelength_um, z * state.lens.wavelength_um)
        intensity = propagated.intensity
        intensity /= max(float(intensity.max()), 1e-15)
        line = intensity[intensity.shape[0] // 2]
        plane.append(line)
        axis_lambda = propagated.x_um / state.lens.wavelength_um
        axis_range = max(abs(float(axis_lambda[0])), abs(float(axis_lambda[-1])))
        para.append([z, float(np.max(line)), estimate_fwhm(np.tile(line, (len(line), 1)), axis_range), estimate_sidelobe_percent(intensity)])
    plane_arr = np.asarray(plane)
    para_arr = np.asarray(para)
    x = propagated.x_um / state.lens.wavelength_um
    paths = [
        folder / "IntensityOnPropagationPlane_0.txt",
        folder / "X.txt",
        folder / "Z.txt",
        folder / "Para_onZ_0.txt",
    ]
    save_matrix_txt(paths[0], plane_arr)
    save_matrix_txt(paths[1], x)
    save_matrix_txt(paths[2], z_values)
    save_matrix_txt(paths[3], para_arr)
    return paths


def generate_e_points(center_distance_um: float, width_um: float, line_width_um: float, point_spacing_um: float, offset_x_um: float = 0.0, offset_y_um: float = 0.0, middle_arm_ratio: float = 0.72) -> np.ndarray:
    spacing = max(point_spacing_um, 1e-6)
    x0 = -width_um / 2.0
    x1 = width_um / 2.0
    y_positions = np.array([-center_distance_um, 0.0, center_distance_um])
    points = []
    vertical_y = np.arange(y_positions[0], y_positions[-1] + spacing * 0.5, spacing)
    for y in vertical_y:
        points.append([x0, y])
    for idx, y in enumerate(y_positions):
        length = width_um if idx != 1 else width_um * float(np.clip(middle_arm_ratio, 0.1, 1.0))
        xs = np.arange(x0, x0 + length + spacing * 0.5, spacing)
        for x in xs:
            for dy in np.arange(-line_width_um / 2.0, line_width_um / 2.0 + spacing * 0.5, spacing):
                points.append([x, y + dy])
    arr = np.unique(np.round(np.asarray(points), decimals=9), axis=0)
    arr[:, 0] += offset_x_um
    arr[:, 1] += offset_y_um
    return arr


def generate_scalebar_points(center_distance_um: float, width_um: float, line_width_um: float, point_spacing_um: float, offset_x_um: float = 0.0, offset_y_um: float = 0.0) -> np.ndarray:
    spacing = max(point_spacing_um, 1e-6)
    xs = np.arange(-width_um / 2.0, width_um / 2.0 + spacing * 0.5, spacing)
    ys = np.arange(-line_width_um / 2.0, line_width_um / 2.0 + spacing * 0.5, spacing)
    points = np.asarray([[x + offset_x_um, y + offset_y_um] for x in xs for y in ys], dtype=np.float64)
    if center_distance_um > 0:
        tick_h = center_distance_um / 2.0
        ticks = []
        for x in (-width_um / 2.0, width_um / 2.0):
            for y in np.arange(-tick_h, tick_h + spacing * 0.5, spacing):
                ticks.append([x + offset_x_um, y + offset_y_um])
        points = np.vstack([points, np.asarray(ticks)])
    return np.unique(np.round(points, decimals=9), axis=0)


def save_objective_points(path: Path, points: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{len(points)}\t0\n")
        for x, y in points:
            f.write(f"{x:.9g}\t{y:.9g}\n")
    return path


def load_objective_points(path: Path) -> np.ndarray:
    rows: List[Tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        first = f.readline()
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                rows.append((float(parts[0]), float(parts[1])))
    return np.asarray(rows, dtype=np.float64)


def calculate_imaging(state: ProjectState, objective_points: np.ndarray, folder: Path, phase_override: np.ndarray | None = None) -> List[Path]:
    img = state.imaging
    n = max(64, min(1024, int(img.grid_n)))
    xs = np.linspace(img.xs_min_um, img.xs_max_um, n)
    xx, yy = np.meshgrid(xs, xs)
    image_distance_um, magnification = projection_image_geometry(state)
    img.focal_length_um = image_distance_um
    # The projection page has one physically determined best image plane. Axial
    # scans remain available in the dedicated propagation/research tools.
    z_values = np.asarray([image_distance_um], dtype=np.float64)
    out_paths: List[Path] = []
    if phase_override is None:
        psf = np.asarray(preview(state).intensity, dtype=np.float64)
    else:
        _x, rr, _phase, amplitude = lens_profiles(state)
        phase = np.asarray(phase_override, dtype=np.float64)
        if phase.shape != amplitude.shape:
            source_axis = np.linspace(0.0, 1.0, phase.shape[0])
            target_axis = np.linspace(0.0, 1.0, amplitude.shape[0])
            rows = np.asarray([np.interp(target_axis, source_axis, row) for row in phase])
            phase = np.asarray([np.interp(target_axis, source_axis, rows[:, col]) for col in range(amplitude.shape[0])]).T
        psf = source_averaged_psf(state, phase, amplitude, rr)
    psf = _resize_square(psf, n)
    psf /= max(float(psf.sum()), 1e-15)
    for i, z in enumerate(z_values):
        defocus = 1.0 + abs(z - img.focal_length_um) / max(abs(img.focal_length_um), 1.0)
        image, objective = simulate_point_object_image(state, objective_points, psf, n)
        transfer = np.fft.fft2(np.fft.ifftshift(psf))
        if defocus > 1.0:
            fy, fx = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n))
            transfer *= np.exp(-2.0 * math.pi**2 * (defocus - 1.0) * (fx * fx + fy * fy))
            image = np.real(np.fft.ifft2(np.fft.fft2(objective) * transfer))
            image = np.maximum(image, 0.0)
            if image.max() > 0:
                image /= image.max()
        txt = folder / f"ImageOnXY_Plane_Z{i}.txt"
        bmp = folder / f"ImageOnXY_Plane_Z{i}.bmp"
        save_matrix_txt(txt, image)
        array_to_bitmap(bmp, image)
        out_paths.extend([txt, bmp])
    save_matrix_txt(folder / "X.txt", xs)
    z_pairs = np.column_stack([np.full_like(z_values, img.objective_distance_um), z_values])
    save_matrix_txt(folder / "Z.txt", z_pairs)
    geometry = np.asarray([[img.objective_distance_um, image_distance_um, magnification,
                            img.e_width_um, img.e_height_um, img.e_line_width_um,
                            img.e_middle_arm_ratio, abs(xs[-1] - xs[0]) / max(1, n - 1)]])
    save_matrix_txt(folder / "ImagePlaneGeometry.txt", geometry)
    out_paths.extend([folder / "X.txt", folder / "Z.txt", folder / "ImagePlaneGeometry.txt"])
    return out_paths


def estimate_object_distance_limit(working_wavelength_um: float, focal_length_um: float, lens_radius_um: float, fwhm_focal_ratio: float, angular_magnification: float, center_distance_um: float) -> Tuple[float, float]:
    diameter = 2.0 * max(lens_radius_um, 1e-9)
    traditional_angle = 1.22 * working_wavelength_um / diameter
    objective_distance_theoretical = center_distance_um / max(traditional_angle, 1e-12)
    lens_angle = max(fwhm_focal_ratio / max(angular_magnification, 1e-12), 1e-12)
    objective_distance_lens = center_distance_um / lens_angle
    return objective_distance_theoretical, objective_distance_lens


def export_design_bundle(state: ProjectState, folder: Path) -> List[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    result = preview(state)
    paths = [
        save_lens_parameters(state, folder),
        save_particle_gene(state, folder),
    ]
    save_matrix_txt(folder / "Lens_Phase_Profile.txt", result.phase_rad)
    save_matrix_txt(folder / "Lens_Amplitude_Profile.txt", result.amplitude)
    save_matrix_txt(folder / "Preview_Intensity.txt", result.intensity)
    array_to_bitmap(folder / "Preview_Intensity.bmp", result.intensity)
    paths.extend([
        folder / "Lens_Phase_Profile.txt",
        folder / "Lens_Amplitude_Profile.txt",
        folder / "Preview_Intensity.txt",
        folder / "Preview_Intensity.bmp",
    ])
    return paths
