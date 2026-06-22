from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DatasetAudit:
    samples: int
    features: int
    duplicates: int
    outliers: int
    missing: int
    target_coverage: dict[str, tuple[float, float]]


class HistoricalDesignLab:
    """Audited historical surrogate with validation and uncertainty estimates."""

    def __init__(self):
        self.feature_names: list[str] = []
        self.target_names: list[str] = []
        self.x_mean = self.x_scale = self.y_mean = None
        self.weights: list[np.ndarray] = []
        self.validation_rmse: dict[str, float] = {}
        self.audit: DatasetAudit | None = None
        self._x = self._y = None

    def load_projects(self, folder: str | Path) -> DatasetAudit:
        rows = []
        for path in Path(folder).rglob("*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(value, dict) and ("design_vector" in value or "best_vector" in value):
                    rows.append(value)
            except (OSError, ValueError, TypeError):
                continue
        if not rows:
            raise ValueError("No compatible optimization JSON records were found.")
        self.feature_names = ["wavelength_um", "focal_lambda", "radius_lambda", "target_fwhm_lambda",
                              "max_sidelobe", "target_contrast", "field_angle_deg", "bandwidth_nm"]
        vectors = [r.get("design_vector", r.get("best_vector")) for r in rows]
        width = min(len(v) for v in vectors)
        self.target_names = [f"design_{i+1}" for i in range(width)]
        def lookup(row, key, default=0.0):
            pools = [row, row.get("targets", {}), row.get("project", {}).get("lens", {}),
                     row.get("project", {}).get("target", {}), row.get("project", {}).get("source", {})]
            for pool in pools:
                if key in pool: return float(pool[key])
            return default
        x = np.asarray([[lookup(r, k) for k in self.feature_names] for r in rows], dtype=float)
        y = np.asarray([[float(v) for v in vec[:width]] for vec in vectors], dtype=float)
        missing = int(np.count_nonzero(~np.isfinite(x)) + np.count_nonzero(~np.isfinite(y)))
        x = np.nan_to_num(x); y = np.nan_to_num(y)
        packed = np.round(np.c_[x, y], 10)
        duplicates = len(packed) - len(np.unique(packed, axis=0))
        median, mad = np.median(x, axis=0), np.median(abs(x-np.median(x, axis=0)), axis=0)
        rz = abs(x-median) / np.maximum(1.4826*mad, 1e-12)
        outliers = int(np.count_nonzero(np.any(rz > 5, axis=1)))
        keep = np.all(rz <= 8, axis=1)
        self._x, self._y = x[keep], y[keep]
        coverage = {name: (float(self._x[:, i].min()), float(self._x[:, i].max())) for i, name in enumerate(self.feature_names)}
        self.audit = DatasetAudit(len(self._x), self._x.shape[1], duplicates, outliers, missing, coverage)
        return self.audit

    def train(self, alpha: float = 0.05, folds: int = 5, ensembles: int = 9, seed: int = 11) -> dict:
        if self._x is None or len(self._x) < 4:
            raise ValueError("At least four audited historical samples are required.")
        x, y = self._x, self._y
        self.x_mean, self.x_scale = x.mean(0), np.maximum(x.std(0), 1e-12)
        self.y_mean = y.mean(0)
        xn, yn = (x-self.x_mean)/self.x_scale, y-self.y_mean
        design = np.c_[np.ones(len(xn)), xn, xn*xn]
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(x))
        predictions = np.zeros_like(y)
        for fold in range(min(folds, len(x))):
            test = indices[fold::min(folds, len(x))]
            train = np.setdiff1d(indices, test)
            gram = design[train].T @ design[train] + alpha*np.eye(design.shape[1])
            weight = np.linalg.solve(gram, design[train].T @ yn[train])
            predictions[test] = design[test] @ weight + self.y_mean
        rmse = np.sqrt(np.mean((predictions-y)**2, axis=0))
        self.validation_rmse = dict(zip(self.target_names, map(float, rmse)))
        self.weights = []
        for _ in range(max(1, ensembles)):
            boot = rng.integers(0, len(x), len(x))
            gram = design[boot].T @ design[boot] + alpha*np.eye(design.shape[1])
            self.weights.append(np.linalg.solve(gram, design[boot].T @ yn[boot]))
        return {"audit": self.audit.__dict__, "validation_rmse": self.validation_rmse,
                "ensemble_models": len(self.weights)}

    def predict(self, targets: dict[str, float]) -> dict:
        if not self.weights:
            raise ValueError("Train the historical model first.")
        raw = np.asarray([float(targets.get(k, 0.0)) for k in self.feature_names])
        xn = (raw-self.x_mean)/self.x_scale
        design = np.r_[1.0, xn, xn*xn]
        predictions = np.asarray([design @ w + self.y_mean for w in self.weights])
        nearest = np.argsort(np.linalg.norm((self._x-raw)/self.x_scale, axis=1))[:5]
        outside = {name: float(raw[i]) for i, name in enumerate(self.feature_names)
                   if raw[i] < self.audit.target_coverage[name][0] or raw[i] > self.audit.target_coverage[name][1]}
        return {"design": dict(zip(self.target_names, map(float, predictions.mean(0)))),
                "uncertainty": dict(zip(self.target_names, map(float, predictions.std(0)))),
                "nearest_cases": nearest.tolist(), "outside_training_domain": outside,
                "recommend_full_optimization": bool(outside or np.mean(predictions.std(0)) > np.mean(list(self.validation_rmse.values())))}

    def active_learning_candidates(self, bounds: dict[str, tuple[float, float]], count: int = 10, seed: int = 3) -> list[dict]:
        if self._x is None:
            raise ValueError("Load historical projects first.")
        rng = np.random.default_rng(seed)
        pool = np.asarray([[rng.uniform(*bounds.get(k, self.audit.target_coverage[k])) for k in self.feature_names] for _ in range(1000)])
        distances = np.min(np.linalg.norm((pool[:, None, :]-self._x[None, :, :])/self.x_scale, axis=2), axis=1)
        chosen = np.argsort(distances)[-count:][::-1]
        return [dict(zip(self.feature_names, map(float, pool[i]))) | {"novelty": float(distances[i])} for i in chosen]


def generate_html_report(path: str | Path, title: str, sections: dict[str, object]) -> Path:
    path = Path(path)
    cards = []
    for heading, value in sections.items():
        body = json.dumps(value, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
        cards.append(f"<section><h2>{heading}</h2><pre>{body}</pre></section>")
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>
<style>body{{font-family:'Segoe UI',sans-serif;background:#f5f7fa;color:#1b1b1f;margin:32px}}header,section{{background:white;border-radius:12px;padding:22px;margin:14px auto;max-width:1100px;box-shadow:0 1px 4px #ccd}}h1,h2{{color:#02529f}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style></head>
<body><header><h1>{title}</h1><p>MetaLens Open Workbench v6.1.1</p></header>{''.join(cards)}</body></html>"""
    path.write_text(html, encoding="utf-8")
    return path
