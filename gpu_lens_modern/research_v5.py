from __future__ import annotations

import csv
import json
import math
import sqlite3
import struct
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


def wrap_phase(value: np.ndarray | float) -> np.ndarray:
    return np.angle(np.exp(1j * np.asarray(value, dtype=float)))


@dataclass
class MetaAtomRecord:
    material: str
    geometry: str
    width_um: float
    length_um: float
    height_um: float
    wavelength_um: float
    angle_deg: float
    polarization: str
    t_real: float
    t_imag: float
    period_um: float = 0.0
    corner_radius_um: float = 0.0
    source: str = "imported"

    @property
    def transmission(self) -> complex:
        return complex(self.t_real, self.t_imag)

    @property
    def phase_rad(self) -> float:
        return float(np.angle(self.transmission))

    @property
    def efficiency(self) -> float:
        return float(abs(self.transmission) ** 2)


class MetaAtomDatabase:
    """Complex-transmission library populated by external RCWA/FDTD data."""

    REQUIRED = {"material", "geometry", "width_um", "length_um", "height_um",
                "wavelength_um", "angle_deg", "polarization"}

    def __init__(self, records: Iterable[MetaAtomRecord] = ()):
        self.records = list(records)

    @classmethod
    def load(cls, path: str | Path) -> "MetaAtomDatabase":
        path = Path(path)
        if path.suffix.lower() == ".json":
            rows = json.loads(path.read_text(encoding="utf-8"))
        else:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        records = []
        for row in rows:
            missing = cls.REQUIRED.difference(row)
            if missing:
                raise ValueError(f"Meta-atom database misses columns: {sorted(missing)}")
            if "t_real" in row and "t_imag" in row:
                tr, ti = float(row["t_real"]), float(row["t_imag"])
            elif "amplitude" in row and ("phase_rad" in row or "phase_deg" in row):
                phase = float(row.get("phase_rad") or math.radians(float(row["phase_deg"])))
                value = float(row["amplitude"]) * np.exp(1j * phase)
                tr, ti = float(value.real), float(value.imag)
            else:
                raise ValueError("Provide t_real/t_imag or amplitude with phase_rad/phase_deg.")
            records.append(MetaAtomRecord(
                material=str(row["material"]), geometry=str(row["geometry"]),
                width_um=float(row["width_um"]), length_um=float(row["length_um"]),
                height_um=float(row["height_um"]), wavelength_um=float(row["wavelength_um"]),
                angle_deg=float(row["angle_deg"]), polarization=str(row["polarization"]).upper(),
                t_real=tr, t_imag=ti, period_um=float(row.get("period_um") or 0),
                corner_radius_um=float(row.get("corner_radius_um") or 0),
                source=str(row.get("source") or path.name)))
        return cls(records)

    def filtered(self, wavelength_um: float, angle_deg: float, polarization: str,
                 wavelength_tol_um: float = 0.02, angle_tol_deg: float = 2.0) -> list[MetaAtomRecord]:
        pol = polarization.upper()
        candidates = [r for r in self.records if r.polarization in {pol, "ANY"}
                      and abs(r.wavelength_um - wavelength_um) <= wavelength_tol_um
                      and abs(r.angle_deg - angle_deg) <= angle_tol_deg]
        if not candidates:
            candidates = sorted(self.records, key=lambda r: abs(r.wavelength_um - wavelength_um)
                                + 0.01 * abs(r.angle_deg - angle_deg)
                                + (0 if r.polarization in {pol, "ANY"} else 10))[:256]
        return candidates

    def match(self, target_phase_rad: float, wavelength_um: float, angle_deg: float,
              polarization: str, min_efficiency: float = 0.0,
              efficiency_weight: float = 0.25) -> MetaAtomRecord:
        candidates = self.filtered(wavelength_um, angle_deg, polarization)
        candidates = [r for r in candidates if r.efficiency >= min_efficiency] or candidates
        if not candidates:
            raise ValueError("The meta-atom database is empty.")
        return min(candidates, key=lambda r: abs(float(wrap_phase(r.phase_rad - target_phase_rad)))
                   + efficiency_weight * (1.0 - min(r.efficiency, 1.0)))

    def synthesize(self, phase: np.ndarray, pitch_um: float, wavelength_um: float,
                   angle_deg: float, polarization: str) -> list[dict]:
        phase = np.asarray(phase, dtype=float)
        cy, cx = (np.asarray(phase.shape) - 1) / 2
        output = []
        for iy, ix in np.ndindex(phase.shape):
            rec = self.match(float(phase[iy, ix]), wavelength_um, angle_deg, polarization)
            output.append({"x_um": (ix - cx) * pitch_um, "y_um": (iy - cy) * pitch_um,
                           "target_phase_rad": float(phase[iy, ix]),
                           "phase_error_rad": float(wrap_phase(rec.phase_rad - phase[iy, ix])),
                           "efficiency": rec.efficiency, **asdict(rec)})
        return output


def _gds_record(record_type: int, data_type: int, payload: bytes = b"") -> bytes:
    if len(payload) % 2:
        payload += b"\0"
    return struct.pack(">HBB", len(payload) + 4, record_type, data_type) + payload


def export_gds_rectangles(layout: Sequence[dict], path: str | Path, layer: int = 1) -> Path:
    """Write a dependency-free GDSII library containing rectangular meta-atoms."""
    path = Path(path)
    now = time.localtime()[:6]
    dates = struct.pack(">12h", *(now + now))
    data = bytearray()
    data += _gds_record(0x00, 0x02, struct.pack(">h", 600))
    data += _gds_record(0x01, 0x02, dates)
    data += _gds_record(0x02, 0x06, b"OPEN_METALENS")
    # GDS real8 values: 1e-3 user unit in meters, 1e-9 database unit in meters.
    data += _gds_record(0x03, 0x05, bytes.fromhex("3E4189374BC6A7F0") + bytes.fromhex("3944B82FA09B5A54"))
    data += _gds_record(0x05, 0x02, dates)
    data += _gds_record(0x06, 0x06, b"METALENS")
    scale = 1000.0
    for item in layout:
        x, y = float(item["x_um"]), float(item["y_um"])
        w, h = float(item["width_um"]), float(item["length_um"])
        pts = [(x-w/2, y-h/2), (x+w/2, y-h/2), (x+w/2, y+h/2),
               (x-w/2, y+h/2), (x-w/2, y-h/2)]
        xy = b"".join(struct.pack(">ii", round(px*scale), round(py*scale)) for px, py in pts)
        data += _gds_record(0x08, 0x00)
        data += _gds_record(0x0D, 0x02, struct.pack(">h", layer))
        data += _gds_record(0x0E, 0x02, struct.pack(">h", 0))
        data += _gds_record(0x10, 0x03, xy)
        data += _gds_record(0x11, 0x00)
    data += _gds_record(0x07, 0x00)
    data += _gds_record(0x04, 0x00)
    path.write_bytes(data)
    return path


def export_dxf_rectangles(layout: Sequence[dict], path: str | Path, layer: str = "METALENS") -> Path:
    path = Path(path)
    lines = ["0", "SECTION", "2", "ENTITIES"]
    for item in layout:
        x, y = float(item["x_um"]), float(item["y_um"])
        w, h = float(item["width_um"]), float(item["length_um"])
        pts = [(x-w/2, y-h/2), (x+w/2, y-h/2), (x+w/2, y+h/2), (x-w/2, y+h/2)]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
            lines += ["0", "LINE", "8", layer, "10", str(x0), "20", str(y0), "30", "0",
                      "11", str(x1), "21", str(y1), "31", "0"]
    lines += ["0", "ENDSEC", "0", "EOF"]
    path.write_text("\n".join(lines), encoding="ascii")
    return path


def jones_to_mueller(jones: np.ndarray) -> np.ndarray:
    j = np.asarray(jones, dtype=complex).reshape(2, 2)
    sigma = [np.eye(2), np.array([[1, 0], [0, -1]]),
             np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]])]
    return np.array([[0.5 * np.trace(a @ j @ b @ j.conj().T).real for b in sigma] for a in sigma])


@dataclass
class AdjointResult:
    phase: np.ndarray
    loss: list[float]
    efficiency: list[float]


def fourier_adjoint_optimize(initial_phase: np.ndarray, target_field: np.ndarray,
                             aperture: np.ndarray | None = None, iterations: int = 100,
                             learning_rate: float = 0.08, phase_levels: int = 0,
                             smooth_weight: float = 0.0,
                             callback: Callable[[int, float, np.ndarray], None] | None = None) -> AdjointResult:
    """Adjoint gradient for scalar Fourier-plane complex-field matching."""
    phase = np.asarray(initial_phase, dtype=float).copy()
    target = np.asarray(target_field, dtype=complex)
    mask = np.ones_like(phase) if aperture is None else np.asarray(aperture, dtype=float)
    scale = math.sqrt(phase.size)
    losses, efficiencies = [], []
    target_power = max(float(np.sum(abs(target) ** 2)), 1e-15)
    for step in range(max(1, int(iterations))):
        pupil = mask * np.exp(1j * phase)
        field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil), norm="ortho"))
        error = field - target
        loss = float(np.mean(abs(error) ** 2))
        adjoint = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(error), norm="ortho"))
        grad = 2.0 * np.real(np.conj(adjoint) * (1j * pupil)) / scale
        if smooth_weight:
            lap = (np.roll(phase, 1, 0) + np.roll(phase, -1, 0)
                   + np.roll(phase, 1, 1) + np.roll(phase, -1, 1) - 4 * phase)
            grad -= smooth_weight * lap
        phase = np.mod(phase - learning_rate * grad, 2 * math.pi)
        if phase_levels > 1:
            phase = np.round(phase / (2 * math.pi) * phase_levels) / phase_levels * 2 * math.pi
        losses.append(loss)
        efficiencies.append(float(np.sum(abs(field) * abs(target)) ** 2 /
                                  max(np.sum(abs(field) ** 2) * target_power, 1e-15)))
        if callback:
            callback(step, loss, phase)
    return AdjointResult(phase, losses, efficiencies)


def pareto_front(records: Sequence[dict], objectives: dict[str, str]) -> list[dict]:
    """Return non-dominated rows. Objective direction is 'min' or 'max'."""
    front = []
    for i, row in enumerate(records):
        dominated = False
        for j, other in enumerate(records):
            if i == j:
                continue
            no_worse, better = True, False
            for key, direction in objectives.items():
                a, b = float(row[key]), float(other[key])
                no_worse &= b <= a if direction == "min" else b >= a
                better |= b < a if direction == "min" else b > a
            if no_worse and better:
                dominated = True
                break
        if not dominated:
            front.append(dict(row))
    return front


@dataclass
class ToleranceConfig:
    linewidth_sigma_nm: float = 5.0
    etch_depth_sigma_nm: float = 10.0
    refractive_index_sigma: float = 0.005
    corner_radius_sigma_nm: float = 3.0
    wavelength_sigma_nm: float = 0.5
    samples: int = 100
    seed: int = 7


def monte_carlo_tolerance(evaluator: Callable[[dict], dict], config: ToleranceConfig) -> dict:
    rng = np.random.default_rng(config.seed)
    rows = []
    for _ in range(max(1, config.samples)):
        perturbation = {
            "linewidth_nm": float(rng.normal(0, config.linewidth_sigma_nm)),
            "etch_depth_nm": float(rng.normal(0, config.etch_depth_sigma_nm)),
            "refractive_index": float(rng.normal(0, config.refractive_index_sigma)),
            "corner_radius_nm": abs(float(rng.normal(0, config.corner_radius_sigma_nm))),
            "wavelength_nm": float(rng.normal(0, config.wavelength_sigma_nm)),
        }
        rows.append({**perturbation, **evaluator(perturbation)})
    metric_keys = [k for k in rows[0] if k not in perturbation]
    summary = {}
    for key in metric_keys:
        values = np.asarray([float(r[key]) for r in rows])
        summary[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=0)),
                        "p05": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95)),
                        "worst_min": float(values.min()), "worst_max": float(values.max())}
    return {"configuration": asdict(config), "summary": summary, "samples": rows}


def critical_dimension(binary: np.ndarray, pixel_um: float) -> float:
    image = np.asarray(binary, dtype=bool)
    widths = []
    for row in image:
        padded = np.r_[False, row, False].astype(np.int8)
        edges = np.flatnonzero(np.diff(padded))
        widths.extend((edges[1::2] - edges[::2]) * pixel_um)
    return float(np.median(widths)) if widths else 0.0


def lithography_metrics(intensity: np.ndarray, pixel_um: float, threshold: float = 0.5) -> dict:
    arr = np.asarray(intensity, dtype=float)
    arr /= max(float(arr.max()), 1e-15)
    binary = arr >= threshold
    cd = critical_dimension(binary, pixel_um)
    gy, gx = np.gradient(arr, pixel_um)
    slope_map = np.hypot(gx, gy)
    edge_band = max(0.02, min(0.51, 1.5 * float(slope_map.max()) * pixel_um))
    edge = (slope_map > 1e-12) & (np.abs(arr - threshold) <= edge_band)
    slope = slope_map[edge]
    nils = float(cd * np.median(slope) / max(threshold, 1e-15)) if slope.size else 0.0
    thresholds = np.linspace(max(.05, threshold-.2), min(.95, threshold+.2), 9)
    cds = np.array([critical_dimension(arr >= t, pixel_um) for t in thresholds])
    valid = cds > 0
    exposure_window = float(np.ptp(thresholds[valid])) if np.count_nonzero(valid) > 1 else 0.0
    return {"threshold": threshold, "critical_dimension_um": cd, "nils": nils,
            "exposure_window_normalized": exposure_window, "printed_area_um2": float(binary.sum()*pixel_um**2)}


def depth_of_focus(stack: np.ndarray, z_um: np.ndarray, pixel_um: float,
                   threshold: float = 0.5, cd_tolerance: float = 0.1) -> dict:
    cds = np.asarray([lithography_metrics(x, pixel_um, threshold)["critical_dimension_um"] for x in stack])
    center = int(np.argmin(abs(np.asarray(z_um))))
    nominal = max(float(cds[center]), 1e-15)
    good = abs(cds / nominal - 1.0) <= cd_tolerance
    dof = float(np.ptp(np.asarray(z_um)[good])) if np.count_nonzero(good) > 1 else 0.0
    return {"z_um": np.asarray(z_um).tolist(), "cd_um": cds.tolist(), "depth_of_focus_um": dof}


def load_weighted_distribution(path: str | Path, axes: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    points = np.asarray([[float(row[a]) for a in axes] for row in rows])
    weights = np.asarray([float(row.get("weight", row.get("intensity", 1))) for row in rows])
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Distribution weights must be non-negative with a positive sum.")
    return points, weights / weights.sum()


def align_experiment(reference: np.ndarray, measurement: np.ndarray) -> dict:
    ref = np.asarray(reference, dtype=float)
    obs = np.asarray(measurement, dtype=float)
    if ref.shape != obs.shape:
        raise ValueError("Reference and measurement must have the same shape.")
    corr = np.fft.ifft2(np.fft.fft2(ref-ref.mean()) * np.conj(np.fft.fft2(obs-obs.mean()))).real
    shift = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dy, dx = (int(v if v <= n//2 else v-n) for v, n in zip(shift, ref.shape))
    aligned = np.roll(obs, (dy, dx), axis=(0, 1))
    design = np.c_[aligned.ravel(), np.ones(aligned.size)]
    gain, offset = np.linalg.lstsq(design, ref.ravel(), rcond=None)[0]
    calibrated = aligned * gain + offset
    rmse = float(np.sqrt(np.mean((calibrated-ref)**2)))
    correlation = float(np.corrcoef(ref.ravel(), calibrated.ravel())[0, 1])
    return {"shift_y_px": dy, "shift_x_px": dx, "gain": float(gain), "offset": float(offset),
            "rmse": rmse, "correlation": correlation, "aligned": calibrated}


class ProjectDatabase:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS versions (id INTEGER PRIMARY KEY, project TEXT, version INTEGER, created REAL, data TEXT, metrics TEXT, note TEXT)")
        self.connection.commit()

    def save(self, project: str, data: dict, metrics: dict | None = None, note: str = "") -> int:
        version = self.connection.execute("SELECT COALESCE(MAX(version),0)+1 FROM versions WHERE project=?", (project,)).fetchone()[0]
        cursor = self.connection.execute("INSERT INTO versions(project,version,created,data,metrics,note) VALUES(?,?,?,?,?,?)",
                                         (project, version, time.time(), json.dumps(data), json.dumps(metrics or {}), note))
        self.connection.commit()
        return int(cursor.lastrowid)

    def list_versions(self, project: str | None = None) -> list[dict]:
        sql = "SELECT id,project,version,created,metrics,note FROM versions"
        args = ()
        if project is not None:
            sql += " WHERE project=?"; args = (project,)
        sql += " ORDER BY created DESC"
        return [{"id": r[0], "project": r[1], "version": r[2], "created": r[3],
                 "metrics": json.loads(r[4]), "note": r[5]} for r in self.connection.execute(sql, args)]

    def compare(self, first_id: int, second_id: int) -> dict:
        rows = []
        for value in (first_id, second_id):
            row = self.connection.execute("SELECT data,metrics FROM versions WHERE id=?", (value,)).fetchone()
            if not row: raise KeyError(value)
            rows.append((json.loads(row[0]), json.loads(row[1])))
        def flatten(value, prefix=""):
            out = {}
            for key, item in value.items():
                path = f"{prefix}.{key}" if prefix else key
                out.update(flatten(item, path) if isinstance(item, dict) else {path: item})
            return out
        a, b = flatten(rows[0][0] | {"metrics": rows[0][1]}), flatten(rows[1][0] | {"metrics": rows[1][1]})
        return {key: {"first": a.get(key), "second": b.get(key)} for key in sorted(a.keys() | b.keys()) if a.get(key) != b.get(key)}

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
