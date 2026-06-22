from __future__ import annotations

import json
import math
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from . import optics
from .models import ProjectState


@dataclass
class HistoricalDataSummary:
    consent: bool
    root: str
    files_seen: int = 0
    projects: int = 0
    metric_files: int = 0
    gene_files: int = 0
    usable_vectors: int = 0
    note: str = ""
    ml_profile: Dict[str, float] | None = None


@dataclass
class HybridOptimizationResult:
    iterations: int
    best_score: float
    best_vector: List[float]
    curve: List[float]
    method: str
    historical_vectors_used: int
    note: str
    metrics: List[Dict[str, float]]
    best_phase_profile: List[List[float]]
    stopped: bool = False
    ml_profile: Dict[str, float] | None = None


def _safe_float(value: str) -> float | None:
    try:
        return float(value.strip())
    except Exception:
        return None


def scan_historical_data(root: Path, consent: bool) -> tuple[HistoricalDataSummary, np.ndarray]:
    """Read old calculation data only after explicit user consent.

    The scanner is intentionally conservative: it only reads text/json files that
    match the original project/output names and extracts numeric vectors. This
    keeps the ML stage explainable and avoids silently indexing unrelated files.
    """

    summary = HistoricalDataSummary(consent=consent, root=str(root))
    if not consent:
        summary.note = "User consent was not granted; no historical files were read."
        return summary, np.empty((0, 8), dtype=np.float64)
    if not root.exists():
        summary.note = "Selected folder does not exist."
        return summary, np.empty((0, 8), dtype=np.float64)

    vectors: List[List[float]] = []
    allowed_names = {
        "LensParameters.txt",
        "Particle_Gene_0.txt",
        "Para_onZ_0.txt",
        "Preview_Intensity.txt",
        "Lens_Phase_Profile.txt",
        "Optimized_Project.json",
        "Optimization_Process.json",
        "Optimization_Metrics.tsv",
        "ObjectivePoints_Letter_E.txt",
        "gpu_lens_project.json",
        "metalens_workbench_project.json",
    }
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name not in allowed_names and not path.name.startswith(("Para_onZ_", "ImageOnXY_Plane_Z", "IntensityOnPropagationPlane_")):
            continue
        summary.files_seen += 1
        if path.name.endswith(".json"):
            summary.projects += 1
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
                lens = data.get("lens", {})
                target = data.get("target", {})
                vectors.append([
                    float(lens.get("wavelength_um", 0.6328)),
                    float(lens.get("working_distance_lambda", 10.0)),
                    float(lens.get("lens_radius_lambda", 50.0)),
                    float(target.get("fwhm_lambda", 0.36)),
                    float(target.get("sidelobe_percent", 25.0)),
                    float(target.get("peak_intensity", 1.0)),
                    0.0,
                    0.0,
                ])
            except Exception:
                pass
            continue
        if "Gene" in path.name:
            summary.gene_files += 1
        if "Para_onZ" in path.name or "Intensity" in path.name or "ImageOnXY" in path.name:
            summary.metric_files += 1
        try:
            values = []
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[:2000]:
                for token in line.replace(",", " ").split():
                    val = _safe_float(token)
                    if val is not None and math.isfinite(val):
                        values.append(val)
                if len(values) >= 128:
                    break
            if values:
                arr = np.asarray(values, dtype=np.float64)
                vectors.append([
                    float(np.mean(arr)),
                    float(np.std(arr)),
                    float(np.min(arr)),
                    float(np.max(arr)),
                    float(np.median(arr)),
                    float(np.percentile(arr, 25)),
                    float(np.percentile(arr, 75)),
                    float(len(arr)),
                ])
        except Exception:
            continue
    summary.usable_vectors = len(vectors)
    if vectors:
        arr = np.asarray(vectors, dtype=np.float64)
        finite = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        summary.ml_profile = {
            "samples": float(finite.shape[0]),
            "feature_mean": float(np.mean(finite)),
            "feature_std": float(np.std(finite)),
            "feature_min": float(np.min(finite)),
            "feature_max": float(np.max(finite)),
        }
        summary.note = f"Historical ML model ready. {summary.usable_vectors} numeric vectors were normalized into a surrogate prior."
        return summary, finite
    summary.note = "Historical scan complete, but no usable numeric vectors were found for ML re-optimization."
    return summary, np.empty((0, 8), dtype=np.float64)


def lithography_score(state: ProjectState, vector: np.ndarray) -> float:
    target = state.lithography
    if not target.enabled:
        return 0.0
    # The vector is normalized design DNA. These relationships are a transparent
    # surrogate until full electromagnetic/propagation kernels are available.
    angular_mag = 1.0 + 6.0 * vector[0]
    uniformity = 80.0 + 20.0 * (1.0 - abs(vector[1] - 0.5) * 2.0)
    telecentricity = 0.2 * abs(vector[2] - 0.5)
    distortion = 2.0 * abs(vector[3] - 0.5)
    overlay = 5.0 + 40.0 * abs(vector[4] - 0.5)
    feature = 200.0 + 800.0 * vector[5]
    contrast = 0.08 + 0.12 * (1.0 - abs(vector[6] - 0.5) * 2.0)
    fwhm = 0.35 + 0.45 * vector[7]
    sidelobe = 0.08 + 0.22 * abs(vector[1] - 0.4)

    score = 0.0
    score += target.angular_magnification_weight * abs(angular_mag - target.angular_magnification_target) / max(target.angular_magnification_target, 1e-9)
    score += max(0.0, target.field_uniformity_target_percent - uniformity) / 100.0
    score += max(0.0, telecentricity - target.telecentricity_error_max_deg) / max(target.telecentricity_error_max_deg, 1e-9)
    score += max(0.0, distortion - target.distortion_max_percent) / max(target.distortion_max_percent, 1e-9)
    score += max(0.0, overlay - target.overlay_tolerance_nm) / max(target.overlay_tolerance_nm, 1e-9)
    score += max(0.0, feature - target.min_feature_nm) / max(target.min_feature_nm, 1e-9)
    score += max(0.0, target.image_contrast_target - contrast) / max(target.image_contrast_target, 1e-9)
    score += max(0.0, fwhm - target.fwhm_max_lambda) / max(target.fwhm_max_lambda, 1e-9)
    score += max(0.0, sidelobe - target.sidelobe_max_ratio) / max(target.sidelobe_max_ratio, 1e-9)
    return float(score)


def _surrogate_bias(history: np.ndarray, dims: int) -> np.ndarray:
    if history.size == 0:
        return np.full(dims, 0.5, dtype=np.float64)
    base = np.mean(history, axis=0)
    base = base[:dims] if len(base) >= dims else np.pad(base, (0, dims - len(base)), constant_values=float(np.mean(base)))
    # Robust squashing maps arbitrary historical numeric ranges to design DNA.
    return 1.0 / (1.0 + np.exp(-np.nan_to_num(base, nan=0.0) / (np.std(base) + 1e-9)))


def historical_ml_profile(history: np.ndarray, dims: int = 8) -> Dict[str, Any]:
    if history.size == 0:
        return {
            "enabled": False,
            "samples": 0,
            "confidence": 0.0,
            "center": [0.5] * dims,
            "spread": [0.25] * dims,
            "note": "No historical vectors were available. The optimizer uses an unbiased search prior.",
        }
    arr = np.nan_to_num(np.asarray(history, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    center = _surrogate_bias(arr, dims)
    base = arr[:, :dims] if arr.shape[1] >= dims else np.pad(arr, ((0, 0), (0, dims - arr.shape[1])), constant_values=float(np.mean(arr)))
    spread_raw = np.std(base, axis=0)
    spread = 1.0 / (1.0 + np.exp(-spread_raw / (float(np.std(base)) + 1e-9)))
    confidence = min(0.85, 0.15 + 0.08 * math.log1p(arr.shape[0]))
    return {
        "enabled": True,
        "samples": int(arr.shape[0]),
        "confidence": float(confidence),
        "center": [float(x) for x in center],
        "spread": [float(x) for x in np.clip(spread, 0.05, 0.95)],
        "note": "Historical vectors are normalized into a surrogate prior that biases initialization and a decaying ML pull during optimization.",
    }


def state_from_design_vector(state: ProjectState, vector: np.ndarray) -> ProjectState:
    candidate = copy.deepcopy(state)
    x = np.asarray(vector, dtype=np.float64)
    if x.size < 8:
        x = np.pad(x, (0, 8 - x.size), constant_values=0.5)
    candidate.lens.working_distance_lambda = max(0.1, state.lens.working_distance_lambda * (0.75 + 0.5 * x[0]))
    candidate.lens.k_flat_field = 0.25 + 2.5 * x[1]
    candidate.lens.na_axicon = 0.45 * x[2]
    candidate.source.incident_angle_deg = (x[3] - 0.5) * 2.0 * max(0.1, state.lithography.fov_half_angle_deg)
    candidate.optimization.phase_max_pi = max(candidate.optimization.phase_min_pi + 0.1, 1.45 + 1.1 * x[4])
    candidate.optimization.amp_max = max(candidate.optimization.amp_min + 0.05, 0.72 + 0.28 * x[5])
    candidate.lens.integrated_lens_separation_um = state.lens.integrated_lens_separation_um + 2.0 * (x[6] - 0.5)
    candidate.target.calculation_range_lambda = max(2.0, state.target.calculation_range_lambda * (0.85 + 0.3 * x[7]))
    # Keep iterative optimization responsive while still evaluating real optics.
    candidate.sampling.preview_n = min(max(64, int(state.sampling.preview_n)), 128)
    return candidate


def evaluate_design_candidate(state: ProjectState, vector: np.ndarray) -> tuple[float, Dict[str, float], optics.PreviewResult]:
    candidate = state_from_design_vector(state, vector)
    result = optics.preview(candidate)
    base_score = optics.score_metrics(result.fwhm_lambda, result.sidelobe_percent, result.peak_intensity, candidate.target)
    litho_score, image_metrics = optics.lithography_objective_from_preview(candidate, result)
    contrast = image_metrics["image_contrast"]
    phase_line = result.phase_rad[result.phase_rad.shape[0] // 2, :]
    phase_smoothness = float(np.mean(np.diff(phase_line) ** 2)) / (math.pi * math.pi)
    intensity_reward = -0.05 * math.log10(max(result.peak_intensity, 1.0))
    focus_weight = 0.25 if candidate.lithography.enabled else 1.0
    score = focus_weight * base_score + litho_score + 0.02 * phase_smoothness + intensity_reward
    metrics = {
        "fwhm_lambda": float(result.fwhm_lambda),
        "sidelobe_percent": float(result.sidelobe_percent),
        "sidelobe_ratio": float(result.sidelobe_percent / 100.0),
        "peak_intensity": float(result.peak_intensity),
        "image_contrast": float(contrast),
        "base_score": float(base_score),
        "lithography_score": float(litho_score),
        "phase_smoothness": float(phase_smoothness),
        "score": float(score),
    }
    metrics.update({key: float(value) for key, value in image_metrics.items()})
    return float(score), metrics, result


def run_hybrid_optimizer(
    state: ProjectState,
    history: np.ndarray,
    iterations: int = 160,
    particles: int = 48,
    seed: int = 2026,
    progress: Callable[[Dict[str, float], List[float]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> HybridOptimizationResult:
    """Hybrid PSO + Marine Predator + lightweight ML surrogate.

    This is a production-shaped optimizer scaffold: PSO handles global social
    search, the marine-predator phases add Brownian/Levy-style exploration and
    exploitation, and historical data biases candidate proposals through a
    transparent surrogate vector.
    """

    dims = 8
    rng = np.random.default_rng(seed)
    particles = max(8, int(particles))
    iterations = max(8, int(iterations))
    ml_center = _surrogate_bias(history, dims)
    ml_profile = historical_ml_profile(history, dims)
    ml_confidence = float(ml_profile.get("confidence", 0.0))
    pos = np.clip(0.75 * rng.random((particles, dims)) + 0.25 * ml_center, 0.0, 1.0)
    if history.size:
        guided = max(1, particles // 3)
        pos[:guided] = np.clip(ml_center + rng.normal(0.0, max(0.035, 0.16 * (1.0 - ml_confidence)), size=(guided, dims)), 0.0, 1.0)
    vel = rng.normal(0.0, 0.08, size=(particles, dims))
    personal = pos.copy()

    cache: Dict[tuple[float, ...], tuple[float, Dict[str, float], optics.PreviewResult]] = {}

    def objective(x: np.ndarray) -> tuple[float, Dict[str, float], optics.PreviewResult]:
        key = tuple(np.round(x, 6))
        if key not in cache:
            score, metrics, result = evaluate_design_candidate(state, x)
            ml_penalty = float(np.mean((x - ml_center) ** 2)) if history.size else 0.0
            score += 0.08 * ml_penalty
            metrics["ml_penalty"] = ml_penalty
            metrics["score"] = float(score)
            cache[key] = (float(score), metrics, result)
        return cache[key]

    evaluated = [objective(x) for x in pos]
    scores = np.asarray([item[0] for item in evaluated])
    personal_scores = scores.copy()
    best_idx = int(np.argmin(scores))
    global_best = pos[best_idx].copy()
    global_score = float(scores[best_idx])
    global_metrics = dict(evaluated[best_idx][1])
    global_result = evaluated[best_idx][2]
    curve: List[float] = [global_score]
    metrics_history: List[Dict[str, float]] = []

    for i in range(iterations):
        if should_stop is not None and should_stop():
            break
        t = (i + 1) / iterations
        phase = 1 if t < 1 / 3 else 2 if t < 2 / 3 else 3
        inertia = 0.9 - 0.5 * t
        c1 = 1.6
        c2 = 1.8
        r1 = rng.random((particles, dims))
        r2 = rng.random((particles, dims))
        vel = inertia * vel + c1 * r1 * (personal - pos) + c2 * r2 * (global_best - pos)
        if phase == 1:
            predator = rng.normal(0.0, 0.12, size=(particles, dims))
        elif phase == 2:
            predator = rng.standard_cauchy(size=(particles, dims)) * 0.025
        else:
            predator = (global_best - pos) * rng.random((particles, dims)) * 0.35
        ml_pull = (ml_center - pos) * ((0.12 + 0.28 * ml_confidence) * (1.0 - t))
        pos = np.clip(pos + vel + predator + ml_pull, 0.0, 1.0)
        evaluated = [objective(x) for x in pos]
        scores = np.asarray([item[0] for item in evaluated])
        improved = scores < personal_scores
        personal[improved] = pos[improved]
        personal_scores[improved] = scores[improved]
        best_idx = int(np.argmin(personal_scores))
        if float(personal_scores[best_idx]) < global_score:
            global_score = float(personal_scores[best_idx])
            global_best = personal[best_idx].copy()
            # Re-evaluate the stored personal best so metrics correspond to the
            # design actually being kept as global best.
            global_score, global_metrics, global_result = objective(global_best)
        curve.append(global_score)
        row = dict(global_metrics)
        row["iteration"] = float(i + 1)
        row["best_score"] = float(global_score)
        row["particle_count"] = float(particles)
        for design_index, design_value in enumerate(global_best):
            row[f"design_{design_index}"] = float(design_value)
        metrics_history.append(row)
        if progress is not None:
            progress_row = dict(row)
            progress_row["phase_profile"] = global_result.phase_rad.tolist()
            progress(progress_row, [float(x) for x in curve])

    stopped = len(metrics_history) < iterations

    return HybridOptimizationResult(
        iterations=iterations,
        best_score=global_score,
        best_vector=[float(x) for x in global_best],
        curve=[float(x) for x in curve],
        method="Hybrid PSO + Marine Predator + Historical Surrogate ML",
        historical_vectors_used=int(history.shape[0]) if history.size else 0,
        note="Lower score is better. With projection lithography enabled, contrast is measured from the image-side Letter-E pattern, together with pattern correlation and stroke uniformity.",
        metrics=metrics_history,
        best_phase_profile=global_result.phase_rad.tolist(),
        stopped=stopped,
        ml_profile=ml_profile,
    )
