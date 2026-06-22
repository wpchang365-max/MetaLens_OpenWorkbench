from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .models import ProjectState


FEATURE_NAMES = ["wavelength", "focal_lambda", "radius_lambda", "fwhm", "sidelobe", "contrast"]
TARGET_NAMES = ["working_distance", "flat_field_k", "axicon_na", "incident_angle", "phase_max_pi", "amplitude_max", "separation", "range"]


@dataclass
class TrainingSummary:
    samples: int
    projects_seen: int
    validation_rmse: float
    confidence: float
    source: str


@dataclass
class MLDesignResult:
    vector: np.ndarray
    uncertainty: np.ndarray
    neighbors: int
    confidence: float
    summary: TrainingSummary


class HistoricalLensRegressor:
    """Dependency-free ensemble ridge regressor with uncertainty estimates."""

    def __init__(self, alpha: float = 0.08, ensembles: int = 12, seed: int = 3000):
        self.alpha = float(alpha)
        self.ensembles = int(ensembles)
        self.rng = np.random.default_rng(seed)
        self.x_mean = np.zeros(6)
        self.x_scale = np.ones(6)
        self.y_mean = np.zeros(8)
        self.y_scale = np.ones(8)
        self.coefs: list[np.ndarray] = []
        self.x_train = np.empty((0, 6))

    def fit(self, x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 2:
            raise ValueError("At least two historical optimized projects are required.")
        self.x_mean, self.x_scale = x.mean(0), np.maximum(x.std(0), 1e-9)
        self.y_mean, self.y_scale = y.mean(0), np.maximum(y.std(0), 1e-9)
        xn, yn = (x - self.x_mean) / self.x_scale, (y - self.y_mean) / self.y_scale
        design = np.column_stack([np.ones(len(xn)), xn, xn * xn])
        self.coefs.clear()
        predictions = []
        for _ in range(self.ensembles):
            idx = self.rng.integers(0, len(xn), len(xn))
            xb, yb = design[idx], yn[idx]
            regularizer = np.eye(xb.shape[1]) * self.alpha
            regularizer[0, 0] = 0.0
            coef = np.linalg.pinv(xb.T @ xb + regularizer) @ xb.T @ yb
            self.coefs.append(coef)
            predictions.append(design @ coef)
        mean = np.mean(predictions, axis=0) * self.y_scale + self.y_mean
        self.x_train = xn
        return float(np.sqrt(np.mean((mean - y) ** 2)))

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        if not self.coefs:
            raise RuntimeError("The historical model is not trained.")
        xn = (np.asarray(features) - self.x_mean) / self.x_scale
        design = np.r_[1.0, xn, xn * xn]
        outputs = np.asarray([design @ coef for coef in self.coefs]) * self.y_scale + self.y_mean
        distance = np.sqrt(np.sum((self.x_train - xn) ** 2, axis=1))
        neighbors = int(np.sum(distance <= max(1.5, float(np.percentile(distance, 35)))))
        return np.clip(outputs.mean(0), 0.0, 1.0), outputs.std(0), neighbors


def _state_features(state: ProjectState, metrics: dict[str, float] | None = None) -> np.ndarray:
    metrics = metrics or {}
    return np.asarray([
        state.lens.wavelength_um,
        state.lens.working_distance_lambda,
        state.lens.lens_radius_lambda,
        metrics.get("fwhm_lambda", state.target.fwhm_lambda),
        metrics.get("sidelobe_percent", state.target.sidelobe_percent),
        metrics.get("image_contrast", state.lithography.image_contrast_target),
    ], dtype=np.float64)


def load_training_set(root: Path, progress: Callable[[int], None] | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    features, targets = [], []
    files = list(root.rglob("Optimized_Project.json")) + list(root.rglob("metalens_workbench_project.json"))
    for index, path in enumerate(dict.fromkeys(files)):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            state = ProjectState.from_dict(data)
            process = path.with_name("Optimization_Process.json")
            payload = json.loads(process.read_text(encoding="utf-8-sig")) if process.exists() else {}
            metrics = payload.get("latest_metrics", {})
            vector = payload.get("best_vector")
            if vector is None or len(vector) < 8:
                continue
            features.append(_state_features(state, metrics))
            targets.append(np.clip(np.asarray(vector[:8], dtype=np.float64), 0.0, 1.0))
            # Optimization trajectories contain useful near-optimal and failed
            # examples. Keep a bounded, evenly sampled subset plus the final row.
            history_csv = path.with_name("Optimization_Metrics.csv")
            if history_csv.exists():
                with history_csv.open("r", encoding="utf-8-sig", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                stride = max(1, len(rows) // 40)
                selected_rows = rows[::stride]
                if rows and rows[-1] not in selected_rows:
                    selected_rows.append(rows[-1])
                for row in selected_rows:
                    values = [row.get(f"design_{i}", "") for i in range(8)]
                    if any(value == "" for value in values):
                        continue
                    row_metrics = {
                        key: float(row[key]) for key in ("fwhm_lambda", "sidelobe_percent", "image_contrast")
                        if row.get(key) not in (None, "")
                    }
                    features.append(_state_features(state, row_metrics))
                    targets.append(np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if progress:
            progress(index + 1)
    return np.asarray(features), np.asarray(targets), len(files)


def train_and_design(root: Path, requested_state: ProjectState, *, alpha: float = 0.08, ensembles: int = 12) -> MLDesignResult:
    x, y, projects = load_training_set(root)
    model = HistoricalLensRegressor(alpha=alpha, ensembles=ensembles)
    rmse = model.fit(x, y)
    vector, uncertainty, neighbors = model.predict(_state_features(requested_state))
    confidence = float(np.clip((1.0 - rmse) * (1.0 - float(uncertainty.mean())) * math.tanh(len(x) / 12), 0.0, 0.98))
    summary = TrainingSummary(len(x), projects, rmse, confidence, str(root))
    return MLDesignResult(vector, uncertainty, neighbors, confidence, summary)
