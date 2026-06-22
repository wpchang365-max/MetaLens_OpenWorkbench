from __future__ import annotations

import json
import math
import queue
import threading
import tkinter as tk
from dataclasses import fields, is_dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageTk

from . import optics
from .gpu_backend import status_text
from .models import (
    BeamShape,
    CalculationAccuracy,
    CalculationBackend,
    FieldShape,
    ImageType,
    IncidentPolarization,
    LensCenterType,
    LensSubType,
    LensSubType2,
    LensType,
    MetaArrangeType,
    OptimizationMethod,
    OptimizationSequence,
    OptimizationType,
    PolarPreference,
    ProjectState,
    ReloadStrategy,
    enum_from_option,
    enum_options,
)


BG = "#f6f8fb"
SURFACE = "#ffffff"
SURFACE_ALT = "#eef3fb"
TEXT = "#1f2937"
MUTED = "#64748b"
ACCENT = "#2563eb"
ACCENT_DARK = "#1d4ed8"
SUCCESS = "#16a34a"
WARNING = "#d97706"
ERROR = "#dc2626"
BORDER = "#d7dee9"


def _enum_pairs(enum_type):
    return enum_options(enum_type)


FIELD_LABELS = {
    "wavelength_um": "Wavelength λ (μm)",
    "working_distance_lambda": "Working Distance Z (λ)",
    "lens_radius_lambda": "Lens Radius R (λ)",
    "view_radius_lambda": "Additional View R (λ)",
    "pitch_um": "Meta-atom Pitch (μm)",
    "n_refra_in": "n incident",
    "n_refra_out": "n output",
    "n_refra_lens": "n lens substrate",
    "lens_center_type": "Lens Center",
    "center_block_radius_lambda": "Center Block Radius (λ)",
    "meta_arrange_type": "Meta Arrangement",
    "lens_type": "Lens Structure",
    "lens_thickness_um": "Lens Thickness (μm)",
    "lens_sub_type": "Functional Type I",
    "lens_sub_type2": "Functional Type II",
    "lens_sub_type3": "Image Type",
    "k_flat_field": "Flat-field K",
    "integrated_lens_separation_um": "Integrated Lens Gap (μm)",
    "na_axicon": "Axicon NA",
    "optimization_type": "Optimization Type",
    "amp_n": "Amplitude Levels",
    "amp_min": "Amp Min",
    "amp_max": "Amp Max",
    "phase_n": "Phase Levels",
    "phase_min_pi": "Phase Min (π)",
    "phase_max_pi": "Phase Max (π)",
    "radius_n": "Radius Levels",
    "radius_min_um": "Radius Min (μm)",
    "radius_max_um": "Radius Max (μm)",
    "polar_n": "Polar Levels",
    "polar_min_pi": "Polar Min (π)",
    "polar_max_pi": "Polar Max (π)",
    "optimize_lens_thickness": "Optimize Thickness",
    "lens_thickness_n": "Thickness Levels",
    "lens_thickness_min_um": "Thickness Min (μm)",
    "lens_thickness_max_um": "Thickness Max (μm)",
    "incident_polarization": "Incident Polarization",
    "theta_polar_angle_deg": "Polar Angle θ (deg)",
    "beam_shape": "Beam Shape",
    "waist_w0_lambda": "Gaussian Waist W0 (λ)",
    "incident_angle_deg": "Incident Angle (deg)",
    "incident_angle_n": "Incident Angle Count",
    "theta_min_deg": "θ Min (deg)",
    "theta_max_deg": "θ Max (deg)",
    "angle_groups": "Angle Groups",
    "incident_wavelength_n": "Wavelength Count",
    "wavelength_min_um": "λ Min (μm)",
    "wavelength_max_um": "λ Max (μm)",
    "diffraction_z_n": "Focal Point Count",
    "z_min_lambda": "Z Min (λ)",
    "z_max_lambda": "Z Max (λ)",
    "field_shape": "Field Shape",
    "polar_preference": "Polar Preference",
    "calculation_range_lambda": "Calculation Range (λ)",
    "fwhm_lambda": "Target FWHM (λ)",
    "sidelobe_percent": "Target Sidelobe (%)",
    "peak_intensity": "Target Peak Intensity",
    "nx": "NX",
    "ny": "NY",
    "fft_n": "FFT_N",
    "interpolate": "Interpolate",
    "dxs_lambda": "Interpolation dXs (λ)",
    "backend": "Backend",
    "multi_gpu_n": "Multi-GPU Count",
    "thread_count": "CPU Threads",
    "calculation_accuracy": "Accuracy",
    "preview_n": "Preview Grid",
    "particle_number": "Particles",
    "iterations_num": "Iterations",
    "optimization_sequence": "Optimization Sequence",
    "optimization_method": "Optimization Method",
    "reload_strategy": "Reload Strategy",
    "linear_weight": "Linear Weight",
    "fixed_weight": "Fixed Weight",
    "weight": "Weight",
    "weight_max": "Weight Max",
    "weight_min": "Weight Min",
    "gabpso": "GABPSO",
    "crossover": "Crossover",
    "cal_type": "Calculation Type",
    "ns_para": "Parameter Count",
    "zs_min_lambda": "Zs Min (λ)",
    "zs_max_lambda": "Zs Max (λ)",
    "ns_z": "Z Samples",
    "working_wavelength_um": "Working λ (μm)",
    "focal_length_um": "Focal Length (μm)",
    "lens_radius_um": "Lens Radius (μm)",
    "fwhm_focal_ratio": "FWHM / Focal Length",
    "angular_magnification": "Angular Magnification K",
    "center_distance_um": "Center Distance (μm)",
    "objective_distance_um": "Objective Distance (μm)",
    "xs_min_um": "Xs Min (μm)",
    "xs_max_um": "Xs Max (μm)",
    "grid_n": "Image Grid",
}


ENUM_FIELDS = {
    "lens_center_type": LensCenterType,
    "meta_arrange_type": MetaArrangeType,
    "lens_type": LensType,
    "lens_sub_type": LensSubType,
    "lens_sub_type2": LensSubType2,
    "lens_sub_type3": ImageType,
    "optimization_type": OptimizationType,
    "incident_polarization": IncidentPolarization,
    "beam_shape": BeamShape,
    "field_shape": FieldShape,
    "polar_preference": PolarPreference,
    "backend": CalculationBackend,
    "calculation_accuracy": CalculationAccuracy,
    "optimization_sequence": OptimizationSequence,
    "optimization_method": OptimizationMethod,
    "reload_strategy": ReloadStrategy,
}


class ScrollFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, bg=BG)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas, style="Root.TFrame")
        self.window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.content.bind("<Configure>", self._on_content_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_content_configure(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.window, width=event.width)

    def _on_mousewheel(self, event):
        widget = self.winfo_containing(event.x_root, event.y_root)
        while widget is not None and not isinstance(widget, ScrollFrame):
            widget = getattr(widget, "master", None)
        if widget is not self:
            return
        first, last = self.canvas.yview()
        if first <= 0.0 and last >= 1.0:
            return "break"
        steps = int(-event.delta / 120)
        if steps == 0 and event.delta:
            steps = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(steps, "units")
        return "break"


class ModernLensApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GPU Lens Design Modern")
        self.geometry("1480x920")
        self.minsize(1180, 760)
        self.configure(bg=BG)
        self.state_model = ProjectState()
        self.controls: Dict[str, Tuple[tk.Variable, Callable[[str], Any], Callable[[Any], Any]]] = {}
        self.preview_image_ref = None
        self.phase_image_ref = None
        self.metrics_queue: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.current_page = "Design"
        self._setup_style()
        self._build_shell()
        self._build_pages()
        self._show_page("Design")
        self._refresh_preview()
        self.after(100, self._pump_queue)

    def _setup_style(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("Root.TFrame", background=BG)
        self.style.configure("Surface.TFrame", background=SURFACE, relief="flat")
        self.style.configure("Alt.TFrame", background=SURFACE_ALT)
        self.style.configure("Card.TLabelframe", background=SURFACE, bordercolor=BORDER, relief="solid")
        self.style.configure("Card.TLabelframe.Label", background=SURFACE, foreground=TEXT, font=("Segoe UI Semibold", 11))
        self.style.configure("TLabel", background=SURFACE, foreground=TEXT, font=("Segoe UI", 9))
        self.style.configure("Muted.TLabel", background=SURFACE, foreground=MUTED, font=("Segoe UI", 9))
        self.style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI Semibold", 24))
        self.style.configure("Subtitle.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 10))
        self.style.configure("Metric.TLabel", background=SURFACE, foreground=TEXT, font=("Segoe UI Semibold", 18))
        self.style.configure("Nav.TButton", font=("Segoe UI Semibold", 10), padding=(14, 12), borderwidth=0)
        self.style.map("Nav.TButton", background=[("active", "#e8f0ff")], foreground=[("active", ACCENT)])
        self.style.configure("Accent.TButton", background=ACCENT, foreground="white", font=("Segoe UI Semibold", 10), padding=(14, 9), borderwidth=0)
        self.style.map("Accent.TButton", background=[("active", ACCENT_DARK)])
        self.style.configure("Ghost.TButton", background=SURFACE, foreground=TEXT, font=("Segoe UI", 10), padding=(12, 8), borderwidth=1)
        self.style.configure("TEntry", fieldbackground="#fbfdff", bordercolor=BORDER, padding=5)
        self.style.configure("TCombobox", fieldbackground="#fbfdff", bordercolor=BORDER, padding=5)
        self.style.configure("TCheckbutton", background=SURFACE, foreground=TEXT, font=("Segoe UI", 9))
        self.style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor="#e5eaf3")

    def _build_shell(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.nav = tk.Frame(self, bg="#101827", width=226)
        self.nav.grid(row=0, column=0, sticky="ns")
        self.nav.grid_propagate(False)
        tk.Label(self.nav, text="GPU Lens", bg="#101827", fg="white", font=("Segoe UI Semibold", 20)).pack(anchor="w", padx=22, pady=(28, 2))
        tk.Label(self.nav, text="Modern rewrite prototype", bg="#101827", fg="#9fb1c7", font=("Segoe UI", 9)).pack(anchor="w", padx=22, pady=(0, 22))
        self.nav_buttons: Dict[str, tk.Button] = {}
        for page in ["Design", "Optimization", "Propagation", "Imaging", "Export", "About"]:
            b = tk.Button(
                self.nav,
                text=page,
                command=lambda p=page: self._show_page(p),
                anchor="w",
                relief="flat",
                bd=0,
                padx=22,
                pady=12,
                bg="#101827",
                fg="#d9e5f5",
                activebackground="#1d2a44",
                activeforeground="white",
                font=("Segoe UI Semibold", 10),
            )
            b.pack(fill="x", padx=10, pady=2)
            self.nav_buttons[page] = b
        tk.Label(self.nav, text="Evidence-based rewrite\nMFC + CUDA 10.2 origin", bg="#101827", fg="#7f93ad", font=("Segoe UI", 8), justify="left").pack(side="bottom", anchor="w", padx=22, pady=22)

        self.main = ttk.Frame(self, style="Root.TFrame")
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.rowconfigure(1, weight=1)
        self.main.columnconfigure(0, weight=1)
        self.header = ttk.Frame(self.main, style="Root.TFrame")
        self.header.grid(row=0, column=0, sticky="ew", padx=28, pady=(24, 8))
        self.title_label = ttk.Label(self.header, text="", style="Title.TLabel")
        self.title_label.pack(anchor="w")
        self.subtitle_label = ttk.Label(self.header, text="", style="Subtitle.TLabel")
        self.subtitle_label.pack(anchor="w", pady=(4, 0))
        self.page_host = ttk.Frame(self.main, style="Root.TFrame")
        self.page_host.grid(row=1, column=0, sticky="nsew", padx=24, pady=(0, 22))
        self.page_host.rowconfigure(0, weight=1)
        self.page_host.columnconfigure(0, weight=1)
        self.pages: Dict[str, ttk.Frame] = {}

    def _build_pages(self):
        self.pages["Design"] = self._design_page()
        self.pages["Optimization"] = self._optimization_page()
        self.pages["Propagation"] = self._propagation_page()
        self.pages["Imaging"] = self._imaging_page()
        self.pages["Export"] = self._export_page()
        self.pages["About"] = self._about_page()
        for page in self.pages.values():
            page.grid(row=0, column=0, sticky="nsew")

    def _show_page(self, page: str):
        self.current_page = page
        self.title_label.configure(text=page)
        subtitles = {
            "Design": "Build lens, source, target, sampling, and inspect live optical previews.",
            "Optimization": "Run a responsive PSO-style preview loop and watch FWHM, sidelobe, intensity and score converge.",
            "Propagation": "Generate files compatible with the original propagation-plane workflow.",
            "Imaging": "Generate objective point files and simulate image-plane intensity.",
            "Export": "Save projects, lens parameters, genes, previews and migration bundles.",
            "About": "What was recovered from the executable and how this rewrite maps to the original software.",
        }
        self.subtitle_label.configure(text=subtitles.get(page, ""))
        for name, button in self.nav_buttons.items():
            if name == page:
                button.configure(bg="#243653", fg="white")
            else:
                button.configure(bg="#101827", fg="#d9e5f5")
        self.pages[page].tkraise()

    def _card(self, parent, title: str, row: int, col: int, colspan: int = 1):
        card = ttk.Labelframe(parent, text=title, style="Card.TLabelframe", padding=14)
        card.grid(row=row, column=col, columnspan=colspan, sticky="nsew", padx=8, pady=8)
        return card

    def _design_page(self):
        page = ttk.Frame(self.page_host, style="Root.TFrame")
        page.columnconfigure(0, weight=0)
        page.columnconfigure(1, weight=1)
        page.rowconfigure(0, weight=1)
        left_scroll = ScrollFrame(page)
        left_scroll.grid(row=0, column=0, sticky="nsw", padx=(0, 8))
        left_scroll.configure(width=600)
        left = left_scroll.content
        left.columnconfigure(0, weight=1)
        left.columnconfigure(1, weight=1)
        self._form_card(left, "Lens Basic Parameters", self.state_model.lens, 0, 0, [
            "wavelength_um", "working_distance_lambda", "lens_radius_lambda", "view_radius_lambda",
            "pitch_um", "n_refra_in", "n_refra_out", "n_refra_lens",
            "lens_center_type", "center_block_radius_lambda", "meta_arrange_type", "lens_type",
            "lens_thickness_um", "lens_sub_type", "lens_sub_type2", "lens_sub_type3",
            "k_flat_field", "integrated_lens_separation_um", "na_axicon",
        ])
        self._form_card(left, "Optimized Parameters", self.state_model.optimization, 1, 0, [
            "optimization_type", "amp_n", "amp_min", "amp_max", "phase_n", "phase_min_pi",
            "phase_max_pi", "radius_n", "radius_min_um", "radius_max_um", "polar_n",
            "polar_min_pi", "polar_max_pi", "optimize_lens_thickness", "lens_thickness_n",
            "lens_thickness_min_um", "lens_thickness_max_um",
        ])
        self._form_card(left, "Light Source", self.state_model.source, 2, 0, [
            "incident_polarization", "theta_polar_angle_deg", "beam_shape", "waist_w0_lambda",
            "incident_angle_deg", "incident_angle_n", "theta_min_deg", "theta_max_deg",
            "angle_groups", "incident_wavelength_n", "wavelength_min_um", "wavelength_max_um",
            "diffraction_z_n", "z_min_lambda", "z_max_lambda",
        ])
        self._form_card(left, "Target + Sampling", self.state_model.target, 3, 0, [
            "field_shape", "polar_preference", "calculation_range_lambda", "fwhm_lambda",
            "sidelobe_percent", "peak_intensity",
        ])
        self._form_card(left, "Diffraction Sampling + Hardware", self.state_model.sampling, 4, 0, [
            "nx", "ny", "fft_n", "interpolate", "dxs_lambda", "backend", "multi_gpu_n",
            "thread_count", "calculation_accuracy", "preview_n",
        ])
        controls = ttk.Frame(left, style="Root.TFrame")
        controls.grid(row=5, column=0, sticky="ew", padx=8, pady=8)
        ttk.Button(controls, text="Validate", style="Ghost.TButton", command=self._validate_state).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Refresh Preview", style="Accent.TButton", command=self._refresh_preview).pack(side="left")

        right = ttk.Frame(page, style="Root.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.columnconfigure(1, weight=1)
        right.rowconfigure(1, weight=1)
        metric_card = self._card(right, "Live Metrics", 0, 0, 2)
        metric_card.columnconfigure((0, 1, 2), weight=1)
        self.metric_fwhm = self._metric(metric_card, "FWHM", "0 lambda", 0)
        self.metric_sidelobe = self._metric(metric_card, "Sidelobe", "0 %", 1)
        self.metric_peak = self._metric(metric_card, "Peak Intensity", "0", 2)

        field_card = self._card(right, "Focal/PSF Preview", 1, 0)
        phase_card = self._card(right, "Lens Phase Profile", 1, 1)
        self.preview_canvas = tk.Canvas(field_card, width=420, height=420, bg="#0b1020", highlightthickness=0)
        self.preview_canvas.pack(fill="both", expand=True)
        self.phase_canvas = tk.Canvas(phase_card, width=420, height=420, bg="#0b1020", highlightthickness=0)
        self.phase_canvas.pack(fill="both", expand=True)
        self.gpu_status_label = ttk.Label(right, text="", style="Subtitle.TLabel")
        self.gpu_status_label.grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=(6, 0))
        note = ttk.Label(right, text="GPU policy: Auto selects CUDA/CuPy first, then OpenCL for AMD/NVIDIA/Intel, then CPU fallback. Current prototype keeps deterministic NumPy reference math while exposing the backend boundary.", style="Subtitle.TLabel")
        note.grid(row=3, column=0, columnspan=2, sticky="ew", padx=8, pady=(2, 6))
        return page

    def _optimization_page(self):
        page = ttk.Frame(self.page_host, style="Root.TFrame")
        page.columnconfigure(0, weight=0)
        page.columnconfigure(1, weight=1)
        page.rowconfigure(0, weight=1)
        left = ttk.Frame(page, style="Root.TFrame")
        left.grid(row=0, column=0, sticky="nsw")
        self._form_card(left, "PSO", self.state_model.pso, 0, 0, [
            "particle_number", "iterations_num", "optimization_sequence", "optimization_method",
            "reload_strategy", "linear_weight", "fixed_weight", "weight", "weight_max",
            "weight_min", "gabpso", "crossover",
        ])
        actions = ttk.Frame(left, style="Root.TFrame")
        actions.grid(row=1, column=0, sticky="ew", padx=8, pady=8)
        ttk.Button(actions, text="Start Iteration", style="Accent.TButton", command=self._start_iteration).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Stop", style="Ghost.TButton", command=self._stop_iteration).pack(side="left")
        self.progress = ttk.Progressbar(left, mode="determinate", maximum=100, style="Horizontal.TProgressbar")
        self.progress.grid(row=2, column=0, sticky="ew", padx=12, pady=(4, 12))

        right = ttk.Frame(page, style="Root.TFrame")
        right.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        right.rowconfigure((0, 1), weight=1)
        right.columnconfigure((0, 1), weight=1)
        self.chart_canvases = {}
        for idx, name in enumerate(["FWHM", "Sidelobe", "Peak Intensity", "Score"]):
            card = self._card(right, name, idx // 2, idx % 2)
            canvas = tk.Canvas(card, width=420, height=240, bg=SURFACE, highlightthickness=0)
            canvas.pack(fill="both", expand=True)
            self.chart_canvases[name] = canvas
        self.metrics_history: List[optics.IterationMetric] = []
        return page

    def _propagation_page(self):
        page = ttk.Frame(self.page_host, style="Root.TFrame")
        page.columnconfigure(0, weight=0)
        page.columnconfigure(1, weight=1)
        page.rowconfigure(0, weight=1)
        left = ttk.Frame(page, style="Root.TFrame")
        left.grid(row=0, column=0, sticky="nsw")
        self._form_card(left, "Propagation Plane / On-Z Parameters", self.state_model.propagation, 0, 0, [
            "cal_type", "ns_para", "zs_min_lambda", "zs_max_lambda", "ns_z",
            "theta_min_deg", "theta_max_deg", "wavelength_min_um", "wavelength_max_um",
        ])
        ttk.Button(left, text="Calculate Save", style="Accent.TButton", command=self._calculate_propagation).grid(row=1, column=0, sticky="w", padx=12, pady=8)
        info = self._card(page, "Output Contract", 0, 1)
        text = tk.Text(info, bg=SURFACE, fg=TEXT, relief="flat", wrap="word", height=16, font=("Segoe UI", 10))
        text.pack(fill="both", expand=True)
        text.insert("1.0", "Generates the original workflow files:\n\nIntensityOnPropagationPlane_0.txt\nX.txt\nZ.txt\nPara_onZ_0.txt\n\nThese files are tab-delimited and can be loaded by MATLAB, Python, Origin, or the original plotting snippets in the manual.")
        text.configure(state="disabled")
        return page

    def _imaging_page(self):
        page = ttk.Frame(self.page_host, style="Root.TFrame")
        page.columnconfigure(0, weight=0)
        page.columnconfigure(1, weight=1)
        page.rowconfigure(0, weight=1)
        left = ttk.Frame(page, style="Root.TFrame")
        left.grid(row=0, column=0, sticky="nsw")
        self._form_card(left, "Imaging Simulation", self.state_model.imaging, 0, 0, [
            "working_wavelength_um", "focal_length_um", "lens_radius_um", "fwhm_focal_ratio",
            "angular_magnification", "center_distance_um", "objective_distance_um",
            "ns_z", "zs_min_um", "zs_max_um", "xs_min_um", "xs_max_um", "grid_n",
        ])
        action_card = self._card(left, "Objective Point Files", 1, 0)
        ttk.Button(action_card, text="Generate E ObjectivePoints.txt", style="Accent.TButton", command=lambda: self._generate_objective("E")).pack(fill="x", pady=4)
        ttk.Button(action_card, text="Generate ScaleBar ObjectivePoints.txt", style="Ghost.TButton", command=lambda: self._generate_objective("ScaleBar")).pack(fill="x", pady=4)
        ttk.Button(action_card, text="Load Objective and Simulate", style="Ghost.TButton", command=self._simulate_imaging).pack(fill="x", pady=4)
        ttk.Button(action_card, text="Calculate Object Distance", style="Ghost.TButton", command=self._object_distance).pack(fill="x", pady=4)
        self.imaging_canvas = tk.Canvas(page, width=620, height=620, bg="#0b1020", highlightthickness=0)
        self.imaging_canvas.grid(row=0, column=1, sticky="nsew", padx=12, pady=8)
        self.imaging_image_ref = None
        return page

    def _export_page(self):
        page = ttk.Frame(self.page_host, style="Root.TFrame")
        page.columnconfigure(0, weight=1)
        output_card = self._card(page, "Output Folder", 0, 0)
        output_card.columnconfigure(0, weight=1)
        self.output_var = tk.StringVar(value=str(Path.cwd() / "outputs"))
        entry = ttk.Entry(output_card, textvariable=self.output_var)
        entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(output_card, text="Browse", style="Ghost.TButton", command=self._browse_output).grid(row=0, column=1)
        actions = self._card(page, "Export Actions", 1, 0)
        for label, command in [
            ("Save Project JSON", self._save_project),
            ("Load Project JSON", self._load_project),
            ("Export LensParameters + Particle_Gene + Previews", self._export_bundle),
            ("Validate Original-Compatible Files", self._validate_files),
        ]:
            ttk.Button(actions, text=label, style="Accent.TButton" if label.startswith("Export") else "Ghost.TButton", command=command).pack(anchor="w", pady=5)
        self.log_text = tk.Text(page, height=18, bg=SURFACE, fg=TEXT, relief="flat", wrap="word", font=("Consolas", 10))
        self.log_text.grid(row=2, column=0, sticky="nsew", padx=8, pady=8)
        page.rowconfigure(2, weight=1)
        return page

    def _about_page(self):
        page = ttk.Frame(self.page_host, style="Root.TFrame")
        card = self._card(page, "Recovered Evidence and Rewrite Scope", 0, 0)
        card.columnconfigure(0, weight=1)
        text = tk.Text(card, bg=SURFACE, fg=TEXT, relief="flat", wrap="word", font=("Segoe UI", 10), height=26)
        text.grid(row=0, column=0, sticky="nsew")
        content = """The original executable was identified as a 64-bit native Windows MFC application, not a .NET application.

Recovered evidence:
- File version: GPU_Lens_Design 1.0.0.1.
- Build timestamp: 2023-04-12 UTC.
- Dependencies: cudart64_102.dll, cufft64_10.dll, opencv_world340.dll, mfc140.dll, VCOMP140.dll.
- CUDA kernels include angular-spectrum propagation, transmission, FFT correction, intensity calculation, multi-lens transmission, multi-angle incident beams, and on-Z-axis calculations.
- The bundled manual defines lens structures, target metrics, PSO options, propagation-plane output files, imaging output files, and ObjectivePoints.txt format.

This rewrite implements the application shell, parameter model, file formats, validation, CPU reference calculations, preview visualizations, objective generation, propagation export, and a backend boundary where CUDA/cuFFT or OpenCL kernels can be connected.

GPU support policy:
- NVIDIA: CUDA path through CuPy when CuPy and a matching CUDA runtime/driver are installed.
- AMD/NVIDIA/Intel: OpenCL path through PyOpenCL when an OpenCL runtime and PyOpenCL are installed.
- CPU fallback: always available, so packaged software still runs on machines without GPU drivers.
- GPU drivers and vendor runtimes are not embedded in the exe; they must be installed on the workstation.

It intentionally avoids copying decompiled native code. The goal is a clean, maintainable scientific application whose behavior is derived from the manual, visible UI, file formats, and symbol evidence."""
        text.insert("1.0", content)
        text.configure(state="disabled")
        return page

    def _metric(self, parent, label, value, col):
        frame = ttk.Frame(parent, style="Surface.TFrame")
        frame.grid(row=0, column=col, sticky="ew", padx=8)
        ttk.Label(frame, text=label, style="Muted.TLabel").pack(anchor="w")
        value_label = ttk.Label(frame, text=value, style="Metric.TLabel")
        value_label.pack(anchor="w", pady=(2, 0))
        return value_label

    def _form_card(self, parent, title: str, obj: Any, row: int, col: int, field_names: List[str]):
        card = self._card(parent, title, row, col)
        card.columnconfigure(1, weight=1)
        for idx, name in enumerate(field_names):
            value = getattr(obj, name)
            label = FIELD_LABELS.get(name, name)
            ttk.Label(card, text=label).grid(row=idx, column=0, sticky="w", padx=(0, 10), pady=3)
            self._add_control(card, obj, name, value, idx)
        return card

    def _add_control(self, parent, obj: Any, name: str, value: Any, row: int):
        key = f"{obj.__class__.__name__}.{name}"
        if isinstance(value, bool):
            var = tk.BooleanVar(value=value)
            widget = ttk.Checkbutton(parent, variable=var, command=self._on_control_changed)
            widget.grid(row=row, column=1, sticky="w", pady=3)
            parser = lambda s: bool(s)
            setter = lambda v, o=obj, n=name: setattr(o, n, bool(v))
        elif name in ENUM_FIELDS:
            enum_type = ENUM_FIELDS[name]
            options = enum_options(enum_type)
            current = [o for o in options if o.startswith(f"{int(value)}-")]
            var = tk.StringVar(value=current[0] if current else options[0])
            widget = ttk.Combobox(parent, textvariable=var, values=options, state="readonly", width=32)
            widget.bind("<<ComboboxSelected>>", lambda _e: self._on_control_changed())
            widget.grid(row=row, column=1, sticky="ew", pady=3)
            parser = lambda s, e=enum_type: int(enum_from_option(e, s))
            setter = lambda v, o=obj, n=name: setattr(o, n, int(v))
        else:
            var = tk.StringVar(value=str(value))
            widget = ttk.Entry(parent, textvariable=var, width=18)
            widget.bind("<FocusOut>", lambda _e: self._on_control_changed())
            widget.bind("<Return>", lambda _e: self._on_control_changed())
            widget.grid(row=row, column=1, sticky="ew", pady=3)
            if isinstance(value, int):
                parser = lambda s: int(float(s))
            elif isinstance(value, float):
                parser = lambda s: float(s)
            else:
                parser = lambda s: s
            setter = lambda v, o=obj, n=name: setattr(o, n, v)
        self.controls[key] = (var, parser, setter)

    def _sync_controls_to_state(self) -> bool:
        try:
            for _key, (var, parser, setter) in self.controls.items():
                setter(parser(var.get()))
            self.state_model.output_path = self.output_var.get() if hasattr(self, "output_var") else self.state_model.output_path
            return True
        except Exception as exc:
            messagebox.showerror("Invalid value", f"Could not parse parameter value:\n{exc}")
            return False

    def _on_control_changed(self):
        if self._sync_controls_to_state() and self.current_page == "Design":
            self.after_idle(self._refresh_preview)

    def _validate_state(self):
        if not self._sync_controls_to_state():
            return
        errors = self.state_model.validate()
        if errors:
            messagebox.showwarning("Validation", "\n".join(errors[:12]))
        else:
            messagebox.showinfo("Validation", "All core parameters look valid.")

    def _refresh_preview(self):
        if not self._sync_controls_to_state():
            return
        try:
            result = optics.preview(self.state_model)
        except Exception as exc:
            self._draw_canvas_text(self.preview_canvas, f"Preview error:\n{exc}")
            return
        self.metric_fwhm.configure(text=f"{result.fwhm_lambda:.4g} lambda")
        self.metric_sidelobe.configure(text=f"{result.sidelobe_percent:.3g} %")
        self.metric_peak.configure(text=f"{result.peak_intensity:.3g}")
        if hasattr(self, "gpu_status_label"):
            try:
                self.gpu_status_label.configure(text=status_text(self.state_model.sampling.backend).splitlines()[0])
            except Exception as exc:
                self.gpu_status_label.configure(text=f"Backend status unavailable: {exc}")
        self._draw_heatmap(self.preview_canvas, result.intensity, "hot", "Intensity")
        self._draw_heatmap(self.phase_canvas, result.phase_rad, "phase", "Phase (rad)")

    def _draw_canvas_text(self, canvas: tk.Canvas, text: str):
        canvas.delete("all")
        w = max(canvas.winfo_width(), 300)
        h = max(canvas.winfo_height(), 220)
        canvas.create_text(w / 2, h / 2, text=text, fill="#dbeafe", font=("Segoe UI", 12), justify="center")

    def _draw_heatmap(self, canvas: tk.Canvas, matrix: np.ndarray, cmap: str, title: str):
        canvas.delete("all")
        w = max(canvas.winfo_width(), 360)
        h = max(canvas.winfo_height(), 360)
        arr = np.asarray(matrix, dtype=np.float64)
        arr = arr - float(np.min(arr))
        if float(np.max(arr)) > 0:
            arr = arr / float(np.max(arr))
        if cmap == "phase":
            r = (127.5 * (1 + np.sin(2 * math.pi * arr))).astype(np.uint8)
            g = (127.5 * (1 + np.sin(2 * math.pi * arr + 2.1))).astype(np.uint8)
            b = (127.5 * (1 + np.sin(2 * math.pi * arr + 4.2))).astype(np.uint8)
        else:
            r = np.clip(255 * arr * 1.7, 0, 255).astype(np.uint8)
            g = np.clip(255 * np.maximum(arr - 0.25, 0) * 1.4, 0, 255).astype(np.uint8)
            b = np.clip(255 * np.maximum(arr - 0.75, 0) * 4.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(np.dstack([r, g, b]), "RGB")
        size = min(w - 28, h - 48)
        img = img.resize((size, size), Image.Resampling.BILINEAR)
        photo = ImageTk.PhotoImage(img)
        if canvas is self.preview_canvas:
            self.preview_image_ref = photo
        elif canvas is self.phase_canvas:
            self.phase_image_ref = photo
        else:
            self.imaging_image_ref = photo
        canvas.create_image(w / 2, h / 2 + 8, image=photo, anchor="center")
        canvas.create_text(18, 16, text=title, fill="#e5eefc", anchor="w", font=("Segoe UI Semibold", 10))

    def _start_iteration(self):
        if self.worker and self.worker.is_alive():
            return
        if not self._sync_controls_to_state():
            return
        self.metrics_history = []
        self.progress.configure(value=0)
        stop_token = {"stop": False}
        self.stop_token = stop_token

        def run():
            metrics = optics.pso_preview_metrics(self.state_model, max_iterations=500)
            total = len(metrics)
            for i, metric in enumerate(metrics, 1):
                if stop_token["stop"]:
                    break
                self.metrics_queue.put(("metric", metric, 100.0 * i / total))
                if i % 10 == 0:
                    threading.Event().wait(0.01)
            self.metrics_queue.put(("done", None, None))

        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()

    def _stop_iteration(self):
        if hasattr(self, "stop_token"):
            self.stop_token["stop"] = True

    def _pump_queue(self):
        try:
            while True:
                kind, metric, progress = self.metrics_queue.get_nowait()
                if kind == "metric":
                    self.metrics_history.append(metric)
                    self.progress.configure(value=progress)
                    self._redraw_charts()
                elif kind == "done":
                    self.progress.configure(value=100)
        except queue.Empty:
            pass
        self.after(100, self._pump_queue)

    def _redraw_charts(self):
        if not self.metrics_history:
            return
        series = {
            "FWHM": [m.fwhm_lambda for m in self.metrics_history],
            "Sidelobe": [m.sidelobe_percent for m in self.metrics_history],
            "Peak Intensity": [m.peak_intensity for m in self.metrics_history],
            "Score": [m.score for m in self.metrics_history],
        }
        for name, values in series.items():
            self._draw_line_chart(self.chart_canvases[name], values, ACCENT if name != "Score" else SUCCESS)

    def _draw_line_chart(self, canvas: tk.Canvas, values: List[float], color: str):
        canvas.delete("all")
        w = max(canvas.winfo_width(), 320)
        h = max(canvas.winfo_height(), 200)
        pad = 28
        canvas.create_rectangle(pad, pad, w - pad, h - pad, outline=BORDER, fill="#fbfdff")
        if len(values) < 2:
            return
        vmin = min(values)
        vmax = max(values)
        if abs(vmax - vmin) < 1e-12:
            vmax += 1.0
            vmin -= 1.0
        points = []
        for i, value in enumerate(values):
            x = pad + (w - 2 * pad) * i / (len(values) - 1)
            y = h - pad - (h - 2 * pad) * (value - vmin) / (vmax - vmin)
            points.extend([x, y])
        canvas.create_line(*points, fill=color, width=2.2, smooth=True)
        canvas.create_text(pad, 14, text=f"{values[-1]:.4g}", fill=TEXT, anchor="w", font=("Segoe UI Semibold", 10))

    def _output_folder(self) -> Path:
        if hasattr(self, "output_var"):
            self.state_model.output_path = self.output_var.get()
        path = Path(self.state_model.output_path or (Path.cwd() / "outputs"))
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _browse_output(self):
        selected = filedialog.askdirectory()
        if selected:
            self.output_var.set(selected)
            self.state_model.output_path = selected

    def _save_project(self):
        if not self._sync_controls_to_state():
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", initialfile="gpu_lens_project.json", filetypes=[("JSON", "*.json")])
        if path:
            optics.save_project_json(self.state_model, Path(path))
            self._log(f"Saved project: {path}")

    def _load_project(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            self.state_model = optics.load_project_json(Path(path))
            messagebox.showinfo("Loaded", "Project loaded. Restart the app to fully rebind controls in this prototype.")
            self._log(f"Loaded project: {path}")
            self._refresh_preview()
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))

    def _export_bundle(self):
        if not self._sync_controls_to_state():
            return
        folder = self._output_folder()
        paths = optics.export_design_bundle(self.state_model, folder)
        self._log("Exported design bundle:\n" + "\n".join(str(p) for p in paths))
        messagebox.showinfo("Export complete", f"Exported {len(paths)} files to:\n{folder}")

    def _calculate_propagation(self):
        if not self._sync_controls_to_state():
            return
        folder = self._output_folder()
        paths = optics.calculate_propagation(self.state_model, folder)
        self._log("Propagation files:\n" + "\n".join(str(p) for p in paths))
        messagebox.showinfo("Propagation complete", f"Generated {len(paths)} files.")

    def _generate_objective(self, kind: str):
        if not self._sync_controls_to_state():
            return
        folder = self._output_folder()
        img = self.state_model.imaging
        if kind == "E":
            points = optics.generate_e_points(img.center_distance_um, img.center_distance_um * 2.4, img.center_distance_um * 0.25, max(0.1, img.center_distance_um / 8))
        else:
            points = optics.generate_scalebar_points(img.center_distance_um, img.center_distance_um * 4.0, img.center_distance_um * 0.25, max(0.1, img.center_distance_um / 8))
        path = optics.save_objective_points(folder / "ObjectivePoints.txt", points)
        self._log(f"Generated {kind} objective with {len(points)} points: {path}")
        self._draw_points(points)

    def _draw_points(self, points: np.ndarray):
        n = 512
        img = np.zeros((n, n), dtype=np.float64)
        if len(points) == 0:
            return
        max_abs = max(float(np.max(np.abs(points))), 1e-9)
        scale = (n * 0.42) / max_abs
        for x, y in points:
            ix = int(n / 2 + x * scale)
            iy = int(n / 2 - y * scale)
            if 0 <= ix < n and 0 <= iy < n:
                img[max(0, iy - 1):min(n, iy + 2), max(0, ix - 1):min(n, ix + 2)] = 1.0
        self._draw_heatmap(self.imaging_canvas, img, "hot", "Objective Points")

    def _simulate_imaging(self):
        if not self._sync_controls_to_state():
            return
        path = filedialog.askopenfilename(initialfile="ObjectivePoints.txt", filetypes=[("Objective points", "*.txt"), ("All files", "*.*")])
        if not path:
            default_path = self._output_folder() / "ObjectivePoints.txt"
            if default_path.exists():
                path = str(default_path)
            else:
                return
        points = optics.load_objective_points(Path(path))
        folder = self._output_folder()
        paths = optics.calculate_imaging(self.state_model, points, folder)
        first_txt = next((p for p in paths if p.name.startswith("ImageOnXY") and p.suffix == ".txt"), None)
        if first_txt:
            image = np.loadtxt(first_txt)
            self._draw_heatmap(self.imaging_canvas, image, "hot", "Image Plane")
        self._log("Imaging files:\n" + "\n".join(str(p) for p in paths))
        messagebox.showinfo("Imaging complete", f"Generated {len(paths)} files.")

    def _object_distance(self):
        if not self._sync_controls_to_state():
            return
        img = self.state_model.imaging
        traditional, lens_limit = optics.estimate_object_distance_limit(
            img.working_wavelength_um,
            img.focal_length_um,
            img.lens_radius_um,
            img.fwhm_focal_ratio,
            img.angular_magnification,
            img.center_distance_um,
        )
        messagebox.showinfo(
            "Object distance limit",
            f"ObjectiveDistanceTheoreticalLimit: {traditional:.4g} μm\nObjectiveDistanceLensLimit: {lens_limit:.4g} μm",
        )

    def _validate_files(self):
        folder = self._output_folder()
        required = ["LensParameters.txt", "Particle_Gene_0.txt"]
        missing = [name for name in required if not (folder / name).exists()]
        if missing:
            messagebox.showwarning("Missing files", "Missing:\n" + "\n".join(missing))
        else:
            messagebox.showinfo("Files OK", "Original-compatible design files are present.")

    def _log(self, message: str):
        if hasattr(self, "log_text"):
            self.log_text.insert("end", message + "\n\n")
            self.log_text.see("end")


def main():
    app = ModernLensApp()
    app.mainloop()
