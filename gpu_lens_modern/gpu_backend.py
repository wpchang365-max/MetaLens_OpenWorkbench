from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Tuple

import numpy as np

from .models import CalculationBackend


_DEVICE_CACHE: Tuple[float, List["GpuDevice"], List[str]] | None = None
_DEVICE_CACHE_SECONDS = 30.0


class BackendKind(str, Enum):
    CPU = "CPU"
    NVIDIA_CUDA = "NVIDIA CUDA"
    OPENCL = "OpenCL"


@dataclass
class GpuDevice:
    backend: BackendKind
    name: str
    vendor: str = ""
    memory_mb: Optional[int] = None
    platform: str = ""
    available: bool = True
    detail: str = ""

    def label(self) -> str:
        mem = f", {self.memory_mb} MB" if self.memory_mb else ""
        vendor = f"{self.vendor} " if self.vendor else ""
        platform = f" [{self.platform}]" if self.platform else ""
        return f"{self.backend.value}: {vendor}{self.name}{mem}{platform}".strip()


@dataclass
class BackendStatus:
    selected: BackendKind
    devices: List[GpuDevice]
    notes: List[str]
    using_gpu: bool

    @property
    def summary(self) -> str:
        if self.using_gpu and self.devices:
            return self.devices[0].label()
        if self.notes:
            return f"CPU fallback: {self.notes[0]}"
        return "CPU fallback"


class Accelerator:
    """Small backend facade.

    The current numerical implementation keeps a NumPy reference path so that
    the application is deterministic and always runnable. If CuPy or PyOpenCL
    is installed with matching drivers, this facade exposes the selected device
    and provides array-module hooks for future CUDA/OpenCL kernels.
    """

    def __init__(self, requested: int | CalculationBackend = CalculationBackend.AUTO, selected_device_labels: str = "", memory_fraction: float = 0.75):
        self.requested = CalculationBackend(int(requested))
        self.status = select_backend(self.requested, selected_device_labels)
        self.xp = np
        self.cuda_module: Any = None
        self.cuda_device_ids: List[int] = []
        self.opencl_module: Any = None
        self.opencl_context: Any = None
        self.opencl_queue: Any = None
        self.opencl_contexts: List[Any] = []
        self.opencl_queues: List[Any] = []
        self.opencl_programs: List[Any] = []
        self.opencl_device_labels: List[str] = []
        if self.status.selected == BackendKind.NVIDIA_CUDA:
            try:
                import cupy as cp  # type: ignore

                self.xp = cp
                self.cuda_module = cp
                selected = [x.strip() for x in selected_device_labels.split("||") if x.strip()]
                count = int(cp.cuda.runtime.getDeviceCount())
                for device_id in range(count):
                    props = cp.cuda.runtime.getDeviceProperties(device_id)
                    name = props.get("name", b"")
                    if isinstance(name, bytes):
                        name = name.decode(errors="replace")
                    matches = not selected or any(_cuda_label_matches(label, device_id, str(name)) for label in selected)
                    if matches:
                        self.cuda_device_ids.append(device_id)
                if not self.cuda_device_ids:
                    raise RuntimeError("None of the selected CUDA devices is available.")
                # CuPy's memory pool keeps FFT work buffers in VRAM. Limit the
                # pool so the desktop and other scientific applications retain
                # headroom instead of allowing an unbounded cache.
                for device_id in self.cuda_device_ids:
                    with cp.cuda.Device(device_id):
                        cp.get_default_memory_pool().set_limit(fraction=float(np.clip(memory_fraction, 0.1, 0.95)))
                if len(self.cuda_device_ids) > 1:
                    self.status.notes.append(f"CUDA source-sample scheduler active: {len(self.cuda_device_ids)} devices selected.")
            except Exception as exc:
                self.status.notes.append(f"CuPy import failed after CUDA selection: {exc}")
                self.status.selected = BackendKind.CPU
                self.status.using_gpu = False
        elif self.status.selected == BackendKind.OPENCL:
            try:
                import pyopencl as cl  # type: ignore

                selected = [x.strip() for x in selected_device_labels.split("||") if x.strip()]
                gpu_devices = []
                for platform in cl.get_platforms():
                    for device in platform.get_devices():
                        if device.type & cl.device_type.GPU:
                            label = _pyopencl_device_label(platform, device)
                            device_name = getattr(device, "name", "").strip()
                            vendor = getattr(device, "vendor", "").strip()
                            matched = (not selected) or any(
                                label == item
                                or (device_name and device_name in item and (not vendor or vendor in item))
                                for item in selected
                            )
                            if matched:
                                gpu_devices.append((platform, device, label))
                if not gpu_devices:
                    raise RuntimeError("No OpenCL GPU device is available.")
                self.opencl_module = cl
                for _platform, device, label in gpu_devices:
                    context = cl.Context([device])
                    queue = cl.CommandQueue(context)
                    self.opencl_contexts.append(context)
                    self.opencl_queues.append(queue)
                    self.opencl_device_labels.append(label)
                self.opencl_context = self.opencl_contexts[0]
                self.opencl_queue = self.opencl_queues[0]
                if len(self.opencl_contexts) > 1:
                    self.status.notes.append(f"OpenCL multi-device profile scheduler active: {len(self.opencl_contexts)} devices selected.")
            except Exception as exc:
                self.status.notes.append(f"OpenCL initialization failed after selection: {exc}")
                self.status.selected = BackendKind.CPU
                self.status.using_gpu = False

    def to_host(self, array: Any) -> np.ndarray:
        if self.cuda_module is not None:
            try:
                return self.cuda_module.asnumpy(array)
            except Exception:
                pass
        return np.asarray(array)

    def memory_summary(self) -> str:
        if self.cuda_module is None:
            return "VRAM is not active on the selected backend."
        try:
            summaries = []
            for device_id in self.cuda_device_ids or [0]:
                with self.cuda_module.cuda.Device(device_id):
                    free_bytes, total_bytes = self.cuda_module.cuda.runtime.memGetInfo()
                used = total_bytes - free_bytes
                summaries.append(f"GPU {device_id}: {used / 2**20:.0f}/{total_bytes / 2**20:.0f} MB")
            return "CUDA VRAM: " + "; ".join(summaries)
        except Exception as exc:
            return f"CUDA VRAM query unavailable: {exc}"


OPENCL_PROFILE_KERNEL = r"""
__kernel void lens_profile(
    __global const double *x,
    __global double *phase,
    __global double *amplitude,
    const int n,
    const int row_start,
    const int row_count,
    const double wavelength_um,
    const double working_distance_lambda,
    const double lens_radius_lambda,
    const double center_block_radius_lambda,
    const int center_blocked,
    const int gaussian_beam,
    const double waist_w0_lambda,
    const double n_refra_in,
    const double n_refra_out,
    const double incident_angle_rad,
    const int subtype2,
    const double k_flat_field,
    const int phase_n,
    const double phase_min,
    const double phase_max,
    const int amp_n,
    const double amp_min,
    const double amp_max)
{
    int gid = get_global_id(0);
    int total = n * row_count;
    if (gid >= total) return;
    int iy_local = gid / n;
    int iy = row_start + iy_local;
    int ix = gid - iy_local * n;
    double xx = x[ix];
    double yy = x[iy];
    double rr = sqrt(xx * xx + yy * yy);
    double amp = rr <= lens_radius_lambda ? 1.0 : 0.0;
    if (center_blocked && rr <= center_block_radius_lambda) amp = 0.0;
    if (gaussian_beam && waist_w0_lambda > 0.0) {
        amp *= exp(-(rr / waist_w0_lambda) * (rr / waist_w0_lambda));
    }
    double f_um = working_distance_lambda * wavelength_um;
    double r_um = rr * wavelength_um;
    double tau = 6.2831853071795864769;
    double ph = -(tau * n_refra_out / wavelength_um) * (sqrt(r_um * r_um + f_um * f_um) - f_um);
    if (fabs(incident_angle_rad) > 1.0e-12) {
        double sign = subtype2 == 1 ? -1.0 : 1.0;
        double displacement;
        if (subtype2 == 4) displacement = k_flat_field * working_distance_lambda * tan(incident_angle_rad);
        else if (subtype2 == 5) displacement = k_flat_field * working_distance_lambda * sin(incident_angle_rad);
        else displacement = k_flat_field * working_distance_lambda * incident_angle_rad;
        double tilt = sign * tau * n_refra_in * (xx * wavelength_um) * sin(incident_angle_rad) / wavelength_um;
        ph += tilt - displacement * 0.01;
    }
    ph = fmod(ph, tau);
    if (ph < 0.0) ph += tau;
    if (phase_max > phase_min && phase_n > 1) {
        double normalized = clamp((ph - phase_min) / (phase_max - phase_min), 0.0, 1.0);
        double idx = floor(normalized * (double)(phase_n - 1) + 0.5);
        ph = phase_min + idx * (phase_max - phase_min) / (double)(phase_n - 1);
    }
    if (amp_max >= amp_min && amp_n > 1) {
        double normalized_amp = clamp((amp - amp_min) / fmax(amp_max - amp_min, 1.0e-12), 0.0, 1.0);
        double idx_amp = floor(normalized_amp * (double)(amp_n - 1) + 0.5);
        amp = amp_min + idx_amp * (amp_max - amp_min) / (double)(amp_n - 1);
    }
    int out_idx = iy_local * n + ix;
    phase[out_idx] = ph;
    amplitude[out_idx] = amp;
}
"""


def opencl_lens_profiles(accelerator: Accelerator, state: Any) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if accelerator.opencl_module is None or not accelerator.opencl_contexts or not accelerator.opencl_queues:
        return None
    try:
        cl = accelerator.opencl_module
        n = int(max(64, min(1024, 2 ** round(np.log2(max(2, int(state.sampling.preview_n)))))))
        extent = state.lens.lens_radius_lambda + max(0.0, state.lens.view_radius_lambda)
        x = np.linspace(-extent, extent, n, dtype=np.float64)
        phase = np.empty((n, n), dtype=np.float64)
        amplitude = np.empty((n, n), dtype=np.float64)
        mf = cl.mem_flags
        device_count = len(accelerator.opencl_contexts)
        if len(accelerator.opencl_programs) != device_count:
            accelerator.opencl_programs = [cl.Program(context, OPENCL_PROFILE_KERNEL).build()
                                           for context in accelerator.opencl_contexts]
        splits = np.linspace(0, n, device_count + 1, dtype=int)
        for idx, (context, queue, program) in enumerate(zip(accelerator.opencl_contexts, accelerator.opencl_queues, accelerator.opencl_programs)):
            row_start = int(splits[idx])
            row_end = int(splits[idx + 1])
            row_count = max(0, row_end - row_start)
            if row_count == 0:
                continue
            phase_part = np.empty((row_count, n), dtype=np.float64)
            amp_part = np.empty((row_count, n), dtype=np.float64)
            x_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
            phase_buf = cl.Buffer(context, mf.WRITE_ONLY, phase_part.nbytes)
            amp_buf = cl.Buffer(context, mf.WRITE_ONLY, amp_part.nbytes)
            program.lens_profile(
                queue,
                (n * row_count,),
                None,
                x_buf,
                phase_buf,
                amp_buf,
                np.int32(n),
                np.int32(row_start),
                np.int32(row_count),
                np.float64(state.lens.wavelength_um),
                np.float64(state.lens.working_distance_lambda),
                np.float64(state.lens.lens_radius_lambda),
                np.float64(state.lens.center_block_radius_lambda),
                np.int32(1 if state.lens.lens_center_type == 1 else 0),
                np.int32(1 if state.source.beam_shape == 1 else 0),
                np.float64(state.source.waist_w0_lambda),
                np.float64(state.lens.n_refra_in),
                np.float64(state.lens.n_refra_out),
                np.float64(np.deg2rad(state.source.incident_angle_deg)),
                np.int32(state.lens.lens_sub_type2),
                np.float64(state.lens.k_flat_field),
                np.int32(state.optimization.phase_n),
                np.float64(state.optimization.phase_min_pi * np.pi),
                np.float64(state.optimization.phase_max_pi * np.pi),
                np.int32(state.optimization.amp_n),
                np.float64(state.optimization.amp_min),
                np.float64(state.optimization.amp_max),
            )
            cl.enqueue_copy(queue, phase_part, phase_buf)
            cl.enqueue_copy(queue, amp_part, amp_buf)
            queue.finish()
            phase[row_start:row_end, :] = phase_part
            amplitude[row_start:row_end, :] = amp_part
        xx, yy = np.meshgrid(x, x)
        rr = np.sqrt(xx * xx + yy * yy)
        return x, rr, phase, amplitude
    except Exception as exc:
        accelerator.status.notes.append(f"OpenCL profile kernel failed; CPU reference path will be used: {exc}")
        return None


def _run_short(command: List[str]) -> Tuple[int, str]:
    try:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        completed = subprocess.run(command, capture_output=True, text=True, timeout=4, check=False, creationflags=flags)
        return completed.returncode, (completed.stdout or completed.stderr or "").strip()
    except Exception as exc:
        return 1, str(exc)


def _pyopencl_device_label(platform: Any, device: Any) -> str:
    name = getattr(device, "name", "OpenCL GPU").strip()
    vendor = getattr(device, "vendor", "").strip()
    platform_name = getattr(platform, "name", "OpenCL").strip()
    return f"OpenCL | {vendor} | {name} | {platform_name}"


def _cuda_label_matches(label: str, device_id: int, device_name: str) -> bool:
    if "CUDA" not in label.upper():
        return False
    indexed = re.search(r"(?:^|\s)(\d+):\s", label)
    if indexed is not None:
        return int(indexed.group(1)) == device_id
    return bool(device_name and device_name in label)


def _device_matches_selection(device: GpuDevice, selected: List[str], device_index: int | None = None) -> bool:
    if not selected:
        return True
    if device_index is not None:
        prefix = f"{device.backend.value}#{device_index}|"
        backend_prefix = f"{device.backend.value}#"
        prefixed = [label for label in selected if label.startswith(backend_prefix)]
        if prefixed:
            return any(label.startswith(prefix) for label in prefixed)
    if device.backend == BackendKind.NVIDIA_CUDA:
        indexed = re.match(r"(\d+):\s*(.*)", device.name)
        if indexed is not None:
            return any(_cuda_label_matches(label, int(indexed.group(1)), indexed.group(2)) for label in selected)
        return any("CUDA" in label.upper() and device.name in label for label in selected)
    return any(
        "OPENCL" in label.upper()
        and device.name in label
        and (not device.vendor or device.vendor in label)
        for label in selected
    )


def _detect_nvidia_smi() -> List[GpuDevice]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    code, out = _run_short([
        exe,
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    devices: List[GpuDevice] = []
    if code != 0 or not out:
        return devices
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts:
            continue
        name = parts[0]
        memory = None
        if len(parts) > 1:
            try:
                memory = int(float(parts[1]))
            except ValueError:
                memory = None
        driver = f"driver {parts[2]}" if len(parts) > 2 else ""
        devices.append(GpuDevice(BackendKind.NVIDIA_CUDA, name=name, vendor="NVIDIA", memory_mb=memory, platform="nvidia-smi", detail=driver))
    return devices


def _cupy_devices() -> Tuple[List[GpuDevice], List[str]]:
    notes: List[str] = []
    try:
        import cupy as cp  # type: ignore
    except Exception as exc:
        return [], [f"CuPy is not installed or cannot load CUDA: {exc}"]
    devices: List[GpuDevice] = []
    try:
        count = cp.cuda.runtime.getDeviceCount()
        for idx in range(count):
            props = cp.cuda.runtime.getDeviceProperties(idx)
            name = props.get("name", b"CUDA GPU")
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            memory = int(props.get("totalGlobalMem", 0) / (1024 * 1024)) if props.get("totalGlobalMem") else None
            devices.append(GpuDevice(BackendKind.NVIDIA_CUDA, name=f"{idx}: {name}", vendor="NVIDIA", memory_mb=memory, platform="CuPy/CUDA"))
    except Exception as exc:
        notes.append(f"CUDA runtime query failed: {exc}")
    return devices, notes


def _opencl_devices() -> Tuple[List[GpuDevice], List[str]]:
    notes: List[str] = []
    devices: List[GpuDevice] = []
    try:
        import pyopencl as cl  # type: ignore
    except Exception as exc:
        notes.append(f"PyOpenCL is not installed or cannot load OpenCL: {exc}")
        return devices, notes
    try:
        for platform in cl.get_platforms():
            for device in platform.get_devices():
                dtype = getattr(device, "type", 0)
                if dtype & cl.device_type.GPU:
                    memory = int(getattr(device, "global_mem_size", 0) / (1024 * 1024)) if getattr(device, "global_mem_size", 0) else None
                    devices.append(
                        GpuDevice(
                            BackendKind.OPENCL,
                            name=getattr(device, "name", "OpenCL GPU").strip(),
                            vendor=getattr(device, "vendor", "").strip(),
                            memory_mb=memory,
                            platform=getattr(platform, "name", "OpenCL").strip(),
                            detail=getattr(device, "version", ""),
                        )
                    )
    except Exception as exc:
        notes.append(f"OpenCL device query failed: {exc}")
    return devices, notes


def detect_devices(force_refresh: bool = False) -> Tuple[List[GpuDevice], List[str]]:
    global _DEVICE_CACHE
    now = time.monotonic()
    if not force_refresh and _DEVICE_CACHE is not None:
        stamp, cached_devices, cached_notes = _DEVICE_CACHE
        if now - stamp < _DEVICE_CACHE_SECONDS:
            return list(cached_devices), list(cached_notes)

    devices: List[GpuDevice] = []
    notes: List[str] = []

    cuda_devices, cuda_notes = _cupy_devices()
    devices.extend(cuda_devices)
    notes.extend(cuda_notes)

    if not cuda_devices:
        smi_devices = _detect_nvidia_smi()
        if smi_devices:
            devices.extend(smi_devices)
            notes.append("NVIDIA driver is visible, but CuPy is unavailable; CUDA numerical kernels are not active in this build.")

    opencl_devices, opencl_notes = _opencl_devices()
    devices.extend(opencl_devices)
    notes.extend(opencl_notes)

    if not devices:
        notes.append("No usable CUDA/OpenCL GPU backend was found. Install NVIDIA CUDA+CuPy or an OpenCL runtime+PyOpenCL to enable GPU kernels.")
    _DEVICE_CACHE = (now, list(devices), list(notes))
    return devices, notes


def select_backend(requested: CalculationBackend, selected_device_labels: str = "") -> BackendStatus:
    devices, notes = detect_devices()
    selected = [item.strip() for item in selected_device_labels.split("||") if item.strip()]
    cuda = [d for idx, d in enumerate(devices) if d.backend == BackendKind.NVIDIA_CUDA and d.platform.startswith("CuPy") and _device_matches_selection(d, selected, idx)]
    opencl = [d for idx, d in enumerate(devices) if d.backend == BackendKind.OPENCL and _device_matches_selection(d, selected, idx)]

    if requested == CalculationBackend.CPU:
        return BackendStatus(BackendKind.CPU, devices, notes + ["CPU backend selected by user."], False)
    if requested == CalculationBackend.NVIDIA_CUDA:
        if cuda:
            return BackendStatus(BackendKind.NVIDIA_CUDA, cuda, notes, True)
        return BackendStatus(BackendKind.CPU, devices, notes + ["NVIDIA CUDA backend requested, but CuPy/CUDA is unavailable."], False)
    if requested == CalculationBackend.OPENCL:
        if opencl:
            return BackendStatus(BackendKind.OPENCL, opencl, notes, True)
        return BackendStatus(BackendKind.CPU, devices, notes + ["OpenCL backend requested, but PyOpenCL/OpenCL is unavailable."], False)
    if requested == CalculationBackend.MULTI_GPU:
        gpu_devices = cuda + opencl
        if gpu_devices:
            use_cuda = len(cuda) >= len(opencl)
            selected_kind = BackendKind.NVIDIA_CUDA if use_cuda else BackendKind.OPENCL
            compatible = cuda if use_cuda else opencl
            backend_note = f"{len(compatible)} selected {selected_kind.value} device(s) will be scheduled. CUDA and OpenCL devices cannot share one FFT batch."
            return BackendStatus(selected_kind, compatible, notes + [backend_note], True)
        return BackendStatus(BackendKind.CPU, devices, notes + ["Multi-GPU requested, but no selected GPU compute backend is available."], False)

    if cuda:
        return BackendStatus(BackendKind.NVIDIA_CUDA, cuda, notes, True)
    if opencl:
        return BackendStatus(BackendKind.OPENCL, opencl, notes, True)
    return BackendStatus(BackendKind.CPU, devices, notes, False)


def status_text(requested: int | CalculationBackend = CalculationBackend.AUTO) -> str:
    status = select_backend(CalculationBackend(int(requested)))
    lines = [f"Selected backend: {status.selected.value}", f"Using GPU: {'yes' if status.using_gpu else 'no'}"]
    if status.devices:
        lines.append("Detected devices:")
        lines.extend(f"- {device.label()}" for device in status.devices)
    if status.notes:
        lines.append("Notes:")
        lines.extend(f"- {note}" for note in status.notes)
    return "\n".join(lines)
