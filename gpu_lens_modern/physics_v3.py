from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class PropagationResult:
    x_um: np.ndarray
    y_um: np.ndarray
    field: np.ndarray
    intensity: np.ndarray


@dataclass
class QualityMetrics:
    fwhm_um: float
    sidelobe_ratio: float
    peak: float
    contrast: float
    strehl: float
    efficiency: float
    mtf50_cyc_per_um: float


def next_power_of_two(value: int) -> int:
    value = max(2, int(value))
    return 1 << (value - 1).bit_length()


def fft_coordinates(n: int, dx_um: float, wavelength_um: float, z_um: float) -> np.ndarray:
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=dx_um))
    return wavelength_um * z_um * freq


def fresnel_propagate(field: np.ndarray, dx_um: float, wavelength_um: float, z_um: float) -> PropagationResult:
    """Single-FFT Fresnel propagation with physically calibrated output axes."""
    field = np.asarray(field, dtype=np.complex128)
    if field.ndim != 2 or field.shape[0] != field.shape[1]:
        raise ValueError("Fresnel propagation requires a square 2-D field.")
    if dx_um <= 0 or wavelength_um <= 0 or z_um == 0:
        raise ValueError("dx_um and wavelength_um must be positive and z_um must be non-zero.")
    n = field.shape[0]
    k = 2.0 * math.pi / wavelength_um
    x0 = (np.arange(n) - n / 2) * dx_um
    xx0, yy0 = np.meshgrid(x0, x0)
    pre = np.exp(1j * k * (xx0 * xx0 + yy0 * yy0) / (2.0 * z_um))
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field * pre))) * dx_um * dx_um
    x1 = fft_coordinates(n, dx_um, wavelength_um, z_um)
    xx1, yy1 = np.meshgrid(x1, x1)
    post = np.exp(1j * k * z_um) * np.exp(1j * k * (xx1 * xx1 + yy1 * yy1) / (2.0 * z_um)) / (1j * wavelength_um * z_um)
    out = spectrum * post
    return PropagationResult(x1, x1.copy(), out, np.abs(out) ** 2)


def angular_spectrum_propagate(field: np.ndarray, dx_um: float, wavelength_um: float, z_um: float) -> PropagationResult:
    """Band-limited angular spectrum propagation, including evanescent filtering."""
    field = np.asarray(field, dtype=np.complex128)
    if field.ndim != 2 or field.shape[0] != field.shape[1]:
        raise ValueError("Angular-spectrum propagation requires a square 2-D field.")
    if dx_um <= 0 or wavelength_um <= 0:
        raise ValueError("dx_um and wavelength_um must be positive.")
    n = field.shape[0]
    freq = np.fft.fftfreq(n, d=dx_um)
    fx, fy = np.meshgrid(freq, freq)
    root = 1.0 / wavelength_um**2 - fx * fx - fy * fy
    propagating = root >= 0.0
    transfer = np.zeros_like(root, dtype=np.complex128)
    transfer[propagating] = np.exp(1j * 2.0 * math.pi * z_um * np.sqrt(root[propagating]))
    out = np.fft.ifft2(np.fft.fft2(field) * transfer)
    x = (np.arange(n) - n / 2) * dx_um
    return PropagationResult(x, x.copy(), out, np.abs(out) ** 2)


def debye_high_na_psf(phase: np.ndarray, amplitude: np.ndarray, na: float, refractive_index: float = 1.0) -> np.ndarray:
    """Vector-weighted high-NA approximation suitable for rapid design screening."""
    n = phase.shape[0]
    yy, xx = np.indices((n, n), dtype=np.float64)
    rr = np.sqrt((xx - (n - 1) / 2) ** 2 + (yy - (n - 1) / 2) ** 2) / max(n / 2, 1)
    sin_theta = np.clip(rr * na / max(refractive_index, 1e-9), 0.0, 0.999999)
    cos_theta = np.sqrt(1.0 - sin_theta * sin_theta)
    vector_weight = np.sqrt(cos_theta) * (1.0 + cos_theta) / 2.0
    pupil = amplitude * vector_weight * np.exp(1j * phase)
    image = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
    intensity = np.abs(image) ** 2
    return intensity / max(float(intensity.max()), 1e-15)


def _half_max_width(axis: np.ndarray, line: np.ndarray) -> float:
    peak_index = int(np.argmax(line))
    half = float(line[peak_index]) * 0.5
    above = np.flatnonzero(line >= half)
    if above.size < 1:
        return 0.0
    left, right = int(above[0]), int(above[-1])
    def crossing(i0: int, i1: int) -> float:
        if i0 == i1 or line[i1] == line[i0]:
            return float(axis[i1])
        return float(axis[i0] + (half-line[i0])*(axis[i1]-axis[i0])/(line[i1]-line[i0]))
    xl = crossing(max(0, left-1), left)
    xr = crossing(right, min(len(line)-1, right+1))
    return abs(xr-xl)


def calculate_metrics(intensity: np.ndarray, axis_um: np.ndarray, ideal_peak: float | None = None) -> QualityMetrics:
    arr = np.maximum(np.asarray(intensity, dtype=np.float64), 0.0)
    peak = float(arr.max())
    center = np.unravel_index(int(np.argmax(arr)), arr.shape)
    line = arr[center[0], :]
    fwhm = _half_max_width(axis_um, line)
    yy, xx = np.indices(arr.shape)
    radius = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
    main_radius = max(3.0, arr.shape[0] * 0.025)
    side = arr[radius > main_radius * 1.8]
    sidelobe = float(side.max() / max(peak, 1e-15)) if side.size else 0.0
    p95, p05 = np.percentile(arr, [95, 5])
    contrast = float((p95 - p05) / max(p95 + p05, 1e-15))
    total = float(arr.sum())
    efficiency = float(arr[radius <= main_radius * 1.5].sum() / max(total, 1e-15))
    strehl = float(min(1.0, peak / max(ideal_peak if ideal_peak is not None else peak, 1e-15)))
    normalized_line = line / max(float(line.sum()), 1e-15)
    mtf = np.abs(np.fft.rfft(normalized_line))
    mtf /= max(float(mtf[0]), 1e-15)
    below = np.flatnonzero(mtf <= 0.5)
    spacing = abs(float(axis_um[1] - axis_um[0])) if axis_um.size > 1 else 1.0
    mtf_axis = np.fft.rfftfreq(line.size, d=spacing)
    mtf50 = float(mtf_axis[below[0]]) if below.size else float(mtf_axis[-1])
    return QualityMetrics(fwhm, sidelobe, peak, contrast, strehl, efficiency, mtf50)


def polychromatic_average(fields: Iterable[np.ndarray], weights: Iterable[float] | None = None) -> np.ndarray:
    arrays = [np.asarray(x, dtype=np.float64) for x in fields]
    if not arrays:
        raise ValueError("At least one wavelength result is required.")
    w = np.asarray(list(weights) if weights is not None else np.ones(len(arrays)), dtype=np.float64)
    w /= max(float(w.sum()), 1e-15)
    return sum(weight * value for weight, value in zip(w, arrays))
