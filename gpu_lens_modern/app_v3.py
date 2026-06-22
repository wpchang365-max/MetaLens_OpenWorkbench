from __future__ import annotations

import copy
import csv
import json
import math
import os
import queue
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from . import optics
from .brand import APP_CN_NAME, APP_COPYRIGHT, APP_EN_NAME, APP_PUBLISHER, APP_PUBLISHER_EN, APP_VERSION, PRIMARY_HEX, asset_path
from .exporters import export_array, export_image, export_manifest, export_records
from .gpu_backend import Accelerator, detect_devices, status_text
from .intelligence import run_hybrid_optimizer, state_from_design_vector
from .ml_design import train_and_design
from .models import CalculationBackend, ProjectState
from .physics_v3 import calculate_metrics
from .win11_widgets import FluentComboBox, FluentProgress, FluentScrollFrame, RoundedButton, ToggleCheck


ZH = {
    "title": "超透镜智能设计工作台 Pro", "design": "设计优化", "ml": "历史数据智能设计",
    "gpu": "计算设备", "imaging": "字母 E 成像", "data": "数据与导出", "help": "帮助", "about": "关于", "language": "English",
    "core": "核心参数", "source_settings": "光源与偏振", "smart": "智能优化", "litho": "投影光刻", "preview": "焦面预览",
    "phase": "相位与过程", "imaging": "字母 E 成像", "start": "开始优化", "stop": "停止",
    "refresh": "刷新预览", "export_data": "导出数据", "export_image": "导出图片",
    "export_all": "导出完整项目", "history_folder": "历史优化数据路径", "train_design": "训练并设计",
    "ml_help": "使用本机历史优化项目作为训练集，直接预测满足当前目标的超透镜设计。",
    "gpu_refresh": "重新检测", "benchmark": "运行基准测试", "apply": "应用设备",
    "gpu_select_all": "全选 GPU", "gpu_clear": "清空选择",
    "ready": "就绪", "running": "正在优化", "completed": "优化完成", "stopping": "正在安全停止",
    "dataset": "训练集状态", "prediction": "机器学习设计结果", "save_project": "保存项目",
    "load_project": "打开项目", "path": "数据路径", "browse": "浏览", "open_folder": "打开文件夹",
    "metric_help": "实时质量指标", "convergence": "收敛速度", "process": "优化状态",
    "simulate_e": "模拟字母 E", "export_points": "导出目标点", "phase_source": "相位数据来源",
    "phase_current": "当前设计相位", "phase_optimized": "最近优化相位", "phase_ml": "机器学习设计相位", "phase_imported": "导入的相位文件",
    "import_phase": "导入相位", "device_summary": "设备概览", "available_devices": "可用计算设备", "selected_devices": "已选择设备",
    "benchmark_result": "基准测试结果", "no_device": "未检测到可用 GPU，将稳定使用 CPU", "help_title": "使用帮助",
    "ready_detail": "系统就绪", "copyright": "版权声明", "enable_litho": "启用投影光刻",
    "backend_tile": "活动后端", "devices_tile": "已选 GPU", "speed_tile": "每次耗时 (秒)",
    "export_note": "所有数值结果均可导出为 TXT/CSV，所有渲染图像均可导出为 PNG/JPG/BMP/TIFF。",
    "changelog": "更新日志", "result_title": "优化完成", "export_result": "导出优化结果", "close": "关闭",
    "ml_targets": "机器学习设计目标", "ml_settings": "机器学习参数", "ensemble_size": "集成模型数量",
    "ridge_alpha": "正则化强度", "confidence": "模型置信度", "samples": "训练样本", "validation": "验证误差",
    "export_project": "项目与参数", "export_opt": "优化过程与指标", "export_phase": "相位数据", "export_focal": "焦面与成像",
    "export_ml": "机器学习报告", "export_description": "选择需要导出的内容，系统会建立结构清晰的结果目录。",
    "source_note": "LED 按波长、角度和偏振进行非相干积分；采样数越高，计算越慢。",
}

EN = {
    "title": "MetaLens Intelligent Design Workbench Pro", "design": "Design & Optimization",
    "ml": "Historical ML Designer", "gpu": "Compute Devices", "imaging": "Letter-E Imaging", "data": "Data & Export", "help": "Help", "about": "About",
    "language": "中文", "core": "Core Parameters", "source_settings": "Source & Polarization", "smart": "Intelligent Optimization",
    "litho": "Projection Lithography", "preview": "Focal Preview", "phase": "Phase & Process",
    "imaging": "Letter-E Imaging", "start": "Start Optimization", "stop": "Stop",
    "refresh": "Refresh Preview", "export_data": "Export Data", "export_image": "Export Image",
    "export_all": "Export Complete Project", "history_folder": "Historical optimization data",
    "train_design": "Train and Design", "ml_help": "Train on optimized projects stored on this computer and directly predict a design for the current target.",
    "gpu_refresh": "Detect Again", "benchmark": "Run Benchmark", "apply": "Apply Devices", "ready": "Ready",
    "gpu_select_all": "Select All GPUs", "gpu_clear": "Clear Selection",
    "running": "Optimization running", "completed": "Optimization complete", "stopping": "Stopping safely",
    "dataset": "Training Dataset", "prediction": "ML Design Result", "save_project": "Save Project",
    "load_project": "Open Project", "path": "Data Path", "browse": "Browse", "open_folder": "Open Folder",
    "metric_help": "Live quality metrics", "convergence": "Convergence Speed", "process": "Optimization Status",
    "simulate_e": "Simulate Letter E", "export_points": "Export Object Points", "phase_source": "Phase Data Source",
    "phase_current": "Current Design Phase", "phase_optimized": "Latest Optimized Phase", "phase_ml": "Machine-Learning Phase", "phase_imported": "Imported Phase File",
    "import_phase": "Import Phase", "device_summary": "Device Summary", "available_devices": "Available Compute Devices", "selected_devices": "Selected Devices",
    "benchmark_result": "Benchmark Result", "no_device": "No usable GPU detected; the stable CPU path will be used", "help_title": "User Guide",
    "ready_detail": "System ready", "copyright": "Copyright Notice", "enable_litho": "Enable projection lithography",
    "backend_tile": "Active Backend", "devices_tile": "Selected GPUs", "speed_tile": "Seconds / Run",
    "export_note": "All numerical results support TXT/CSV export; every rendered image supports PNG/JPG/BMP/TIFF export.",
    "changelog": "Changelog", "result_title": "Optimization Complete", "export_result": "Export Optimization Result", "close": "Close",
    "ml_targets": "Machine-Learning Design Targets", "ml_settings": "Machine-Learning Settings", "ensemble_size": "Ensemble Models",
    "ridge_alpha": "Regularization Strength", "confidence": "Model Confidence", "samples": "Training Samples", "validation": "Validation Error",
    "export_project": "Project & Parameters", "export_opt": "Optimization Process & Metrics", "export_phase": "Phase Data", "export_focal": "Focal & Imaging Data",
    "export_ml": "Machine-Learning Report", "export_description": "Choose export content; the application creates a clearly structured result folder.",
    "source_note": "LED uses incoherent wavelength, angle and polarization integration; more samples increase runtime.",
}
ZH.update({
    "focal_preview": "焦平面点扩散函数（PSF）",
    "aerial_preview": "投影像面字母 E 空中像",
    "e_geometry_note": "字母 E 几何：总宽度与总高度定义外包络；笔画宽度同时作用于竖画和三条横画；中横画比例定义中横画长度/总宽度。点间距仅控制目标离散精度，不是几何尺寸。像面距离由焦距和物距自动计算。",
})
EN.update({
    "focal_preview": "Focal-Plane Point Spread Function (PSF)",
    "aerial_preview": "Projection Image-Plane Letter-E Aerial Image",
    "e_geometry_note": "Letter-E geometry: overall width and height define the bounding box; stroke width applies to the vertical and all three arms; middle-arm ratio is middle-arm length / overall width. Point spacing controls numerical discretization, not geometry. Image distance is calculated from focal length and object distance.",
})
ZH.update({
    "design_output_path": "优化数据保存路径",
    "path_note": "设计优化产生的自动归档、相位库和批量队列会保存到此路径。",
})
EN.update({
    "design_output_path": "Optimization Data Path",
    "path_note": "Automatic optimization archives, phase libraries and batch queues are saved to this path.",
})
ZH.update({
    "laser_settings": "激光器参数：相干光源，主要设置偏振、光束形状、束腰和入射角。",
    "led_settings": "LED 参数：非相干光源，主要设置光谱宽度、波长采样、发散角和角度采样。",
    "grid_note": "预览/相位网格同时影响普通聚焦模式和投影光刻模式。网格越高，相位图和焦面细节越清晰，但计算更慢。",
    "gpu_scheduler_note": "多 GPU 会加速可拆分的光源采样或 OpenCL 相位 profile；单个不可拆分 FFT 通常仍在一张 GPU 上执行。",
})
EN.update({
    "laser_settings": "Laser parameters: coherent source with polarization, beam shape, waist and incident angle controls.",
    "led_settings": "LED parameters: incoherent source with spectrum width, wavelength samples, divergence and angle samples.",
    "grid_note": "The Preview / Phase Grid affects both normal focusing and projection lithography. Higher grids make phase and focal details clearer, but increase runtime.",
    "gpu_scheduler_note": "Multi-GPU accelerates splittable source samples or OpenCL phase profiles; a single indivisible FFT usually still runs on one GPU.",
})

FIELD_LABELS = {
    "zh": {
        "wavelength": "工作波长 (µm)", "focal": "焦距 (λ)", "radius": "透镜半径 (λ)", "pitch": "超原子间距 (µm)",
        "source_mode": "光源模式", "polarization_mode": "偏振模式", "beam_shape": "光束形状", "waist_w0": "高斯光束腰斑 (λ)",
        "linear_angle": "线偏振角度 (°)",
        "led_fwhm": "LED 光谱 FWHM (nm)", "wavelength_samples": "波长采样数", "led_divergence": "LED 发散半角 (°)", "angle_samples": "角度采样数",
        "preview_n": "预览采样点", "phase_levels": "相位量化级数", "iterations": "迭代次数", "particles": "粒子数",
        "seed": "随机种子", "live_every": "相位图刷新间隔", "target_fwhm": "目标 FWHM (λ)", "target_sr": "最大旁瓣比 (%)",
        "target_contrast": "目标对比度", "angular_mag": "角放大率", "reduction": "缩小倍率", "fov": "半视场角 (度)",
        "e_width": "字母 E 宽度 (µm)", "e_spacing": "横条间距 (µm)", "e_line": "线宽 (µm)", "e_point": "点间距 (µm)",
        "e_object_distance": "物距 (µm)", "e_image_distance": "像距 (µm)", "e_range": "成像范围 (µm)", "e_grid": "成像网格",
        "ml_wavelength": "目标波长 (µm)", "ml_focal": "目标焦距 (λ)", "ml_radius": "目标透镜半径 (λ)", "ml_fwhm": "目标 FWHM (λ)",
        "ml_sidelobe": "最大旁瓣比 (%)", "ml_contrast": "目标对比度", "ml_ensembles": "集成模型数量", "ml_alpha": "正则化强度",
    },
    "en": {
        "wavelength": "Wavelength (µm)", "focal": "Focal Distance (λ)", "radius": "Lens Radius (λ)", "pitch": "Meta-Atom Pitch (µm)",
        "source_mode": "Source Mode", "polarization_mode": "Polarization", "beam_shape": "Beam Shape", "waist_w0": "Gaussian Waist (λ)",
        "linear_angle": "Linear Polarization Angle (°)",
        "led_fwhm": "LED Spectral FWHM (nm)", "wavelength_samples": "Wavelength Samples", "led_divergence": "LED Divergence Half-Angle (°)", "angle_samples": "Angle Samples",
        "preview_n": "Preview Grid", "phase_levels": "Phase Levels", "iterations": "Iterations", "particles": "Particles",
        "seed": "Random Seed", "live_every": "Phase Refresh Interval", "target_fwhm": "Target FWHM (λ)", "target_sr": "Max Sidelobe (%)",
        "target_contrast": "Target Contrast", "angular_mag": "Angular Magnification", "reduction": "Reduction Ratio", "fov": "Half Field Angle (deg)",
        "e_width": "E Width (µm)", "e_spacing": "Bar Spacing (µm)", "e_line": "Line Width (µm)", "e_point": "Point Spacing (µm)",
        "e_object_distance": "Object Distance (µm)", "e_image_distance": "Image Distance (µm)", "e_range": "Image Range (µm)", "e_grid": "Image Grid",
        "ml_wavelength": "Target Wavelength (µm)", "ml_focal": "Target Focal Distance (λ)", "ml_radius": "Target Lens Radius (λ)", "ml_fwhm": "Target FWHM (λ)",
        "ml_sidelobe": "Max Sidelobe (%)", "ml_contrast": "Target Contrast", "ml_ensembles": "Ensemble Models", "ml_alpha": "Regularization Strength",
    },
}
FIELD_LABELS["zh"].update({
    "phase_mode": "相位实现方式", "geometric_efficiency": "几何相位转换效率",
    "contrast_weight": "E 对比度权重", "peak_weight": "最大强度权重",
    "sidelobe_weight": "PSF 旁瓣泄漏权重", "fwhm_weight": "FWHM 权重",
    "gpu_memory_fraction": "显存使用上限", "gpu_batch_size": "GPU 批处理数",
    "e_width": "E 总宽度 (µm)", "e_height": "E 总高度 (µm)",
    "e_line": "E 笔画宽度 (µm)", "e_middle_ratio": "中横画长度比例",
    "e_point": "目标离散点间距 (µm)", "e_object_distance": "E 到超透镜物距 (µm)",
    "e_range": "像面显示范围 (µm)", "e_grid": "像面采样网格",
})
FIELD_LABELS["en"].update({
    "phase_mode": "Phase Implementation", "geometric_efficiency": "Geometric Conversion Efficiency",
    "contrast_weight": "E Contrast Weight", "peak_weight": "Peak Weight",
    "sidelobe_weight": "PSF Sidelobe Leakage Weight", "fwhm_weight": "FWHM Weight",
    "gpu_memory_fraction": "VRAM Usage Limit", "gpu_batch_size": "GPU Batch Size",
    "e_width": "E Overall Width (µm)", "e_height": "E Overall Height (µm)",
    "e_line": "E Stroke Width (µm)", "e_middle_ratio": "Middle-Arm Length Ratio",
    "e_point": "Object Discretization Step (µm)", "e_object_distance": "E-to-Metalens Object Distance (µm)",
    "e_range": "Image-Plane Display Range (µm)", "e_grid": "Image-Plane Sampling Grid",
})

ZH.update({
    "manual_threads": "手动指定 CPU 线程数",
    "thread_auto_note": "未勾选时，软件会根据本机逻辑 CPU 数自动调度线程，并尽量保留系统响应能力。",
})
EN.update({
    "manual_threads": "Manually choose CPU thread count",
    "thread_auto_note": "When unchecked, the software automatically schedules threads from the logical CPU count while preserving system responsiveness.",
})
FIELD_LABELS["zh"].update({"thread_count": "CPU 线程数"})
FIELD_LABELS["en"].update({"thread_count": "CPU Threads"})


BG, CARD, TEXT, MUTED, BORDER, ACCENT = "#F5F7FA", "#FFFFFF", "#1B1B1F", "#5D6470", "#E1E5EA", "#0F6CBD"


class MetricPlot(tk.Canvas):
    COLORS = ["#0F6CBD", "#D83B01", "#107C10", "#8764B8"]

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=CARD, highlightthickness=0, **kwargs)
        self.series: dict[str, list[float]] = {}
        self.bind("<Configure>", lambda _e: self.redraw())

    def set_series(self, series: dict[str, list[float]]):
        self.series = series
        self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = max(180, self.winfo_width()), max(150, self.winfo_height())
        names = list(self.series)
        if not names:
            self.create_text(w / 2, h / 2, text="No live data", fill=MUTED, font=("Segoe UI", 11))
            return
        cols = 1 if w < 360 else 2
        rows = math.ceil(len(names) / cols)
        for index, name in enumerate(names):
            values = np.asarray(self.series[name], dtype=float)
            x0 = 24 + (index % cols) * w / cols
            y0 = 24 + (index // cols) * h / rows
            x1 = (index % cols + 1) * w / cols - 18
            y1 = (index // cols + 1) * h / rows - 22
            self.create_rectangle(x0, y0, x1, y1, outline=BORDER)
            self.create_text(x0 + 8, y0 + 9, text=name, anchor="nw", fill=TEXT, font=("Segoe UI", 9, "bold"))
            if values.size:
                lo, hi = float(values.min()), float(values.max())
                if math.isclose(lo, hi): hi = lo + 1.0
                pts = []
                for j, value in enumerate(values):
                    px = x0 + 10 + (x1 - x0 - 20) * j / max(1, len(values) - 1)
                    py = y1 - 10 - (y1 - y0 - 35) * (float(value) - lo) / (hi - lo)
                    pts.extend((px, py))
                if len(pts) >= 4:
                    self.create_line(*pts, fill=self.COLORS[index % len(self.COLORS)], width=2, smooth=True)
                self.create_text(x1 - 7, y0 + 9, text=f"{values[-1]:.4g}", anchor="ne", fill=self.COLORS[index % len(self.COLORS)], font=("Consolas", 9, "bold"))


class Heatmap(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg="#10151D", highlightthickness=0, **kwargs)
        self.data: np.ndarray | None = None
        self.image_ref = None
        self.title = ""
        self.display_min: float | None = None
        self.display_max: float | None = None
        self.zoom_factor = 1.0
        self.pan_x = self.pan_y = 0.0
        self._drag_start = None
        self.bind("<Configure>", lambda _e: self.redraw())
        self.bind("<Double-Button-1>", self._range_dialog)
        self.bind("<MouseWheel>", self._zoom_wheel)
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", lambda _e: setattr(self, "_drag_start", None))

    def set_data(self, data: np.ndarray, title: str):
        self.data, self.title = np.asarray(data), title
        self.zoom_factor = 1.0
        self.pan_x = self.pan_y = 0.0
        self.redraw()

    def zoom(self, factor: float):
        self.zoom_factor = float(np.clip(self.zoom_factor * factor, 1.0, 24.0))
        if self.zoom_factor == 1.0:
            self.pan_x = self.pan_y = 0.0
        self.redraw()

    def fit(self):
        self.zoom_factor = 1.0; self.pan_x = self.pan_y = 0.0; self.redraw()

    def focus_signal(self):
        if self.data is None: return
        arr = np.asarray(self.data, dtype=float)
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        mask = arr >= lo + 0.15 * max(hi-lo, 1e-15)
        coords = np.argwhere(mask)
        if not coords.size: return
        h = max(1, coords[:,0].max()-coords[:,0].min()+1)
        w = max(1, coords[:,1].max()-coords[:,1].min()+1)
        self.zoom_factor = float(np.clip(0.65 * min(arr.shape[0]/h, arr.shape[1]/w), 1.0, 24.0))
        center_y, center_x = coords.mean(axis=0)
        side = max(64, min(self.winfo_width() - 90, self.winfo_height() - 42))
        self.pan_x = (arr.shape[1]/2-center_x) * side/arr.shape[1] * self.zoom_factor
        self.pan_y = (arr.shape[0]/2-center_y) * side/arr.shape[0] * self.zoom_factor
        self.redraw()

    def _zoom_wheel(self, event):
        self.zoom(1.25 if event.delta > 0 else 0.8)
        return "break"

    def _press(self, event):
        if 8 <= event.x <= 121 and 32 <= event.y <= 62:
            if event.x < 43: self.zoom(1.25)
            elif event.x < 78: self.zoom(0.8)
            else: self.focus_signal()
            return "break"
        self._drag_start = (event.x, event.y, self.pan_x, self.pan_y)

    def _drag(self, event):
        if self._drag_start is None or self.zoom_factor <= 1.0: return
        x, y, px, py = self._drag_start
        self.pan_x, self.pan_y = px + event.x-x, py + event.y-y
        self.redraw()

    def set_range(self, minimum: float | None, maximum: float | None):
        if minimum is not None and maximum is not None and maximum <= minimum:
            raise ValueError("Colorbar maximum must be greater than minimum.")
        self.display_min, self.display_max = minimum, maximum
        self.redraw()

    @staticmethod
    def _colors(values: np.ndarray) -> np.ndarray:
        return np.dstack([np.clip(values * 420, 0, 255),
                          np.clip((values - .18) * 370, 0, 255),
                          np.clip((values - .65) * 730, 0, 255)]).astype(np.uint8)

    def _range_dialog(self, _event=None):
        if self.data is None:
            return
        dialog = tk.Toplevel(self)
        dialog.title("色标范围 / Colorbar Range")
        dialog.resizable(False, False)
        dialog.configure(bg="#F5F7FA")
        panel = tk.Frame(dialog, bg="#FFFFFF", padx=18, pady=16)
        panel.pack(fill="both", expand=True, padx=12, pady=12)
        actual_min, actual_max = float(np.nanmin(self.data)), float(np.nanmax(self.data))
        minimum = tk.StringVar(value=f"{self.display_min if self.display_min is not None else actual_min:.8g}")
        maximum = tk.StringVar(value=f"{self.display_max if self.display_max is not None else actual_max:.8g}")
        for label, variable in (("最小值 / Minimum", minimum), ("最大值 / Maximum", maximum)):
            row = tk.Frame(panel, bg="#FFFFFF"); row.pack(fill="x", pady=5)
            tk.Label(row, text=label, bg="#FFFFFF", fg=TEXT, font=("Segoe UI", 9)).pack(side="left")
            ttk.Entry(row, textvariable=variable, width=16).pack(side="right")
        actions = tk.Frame(panel, bg="#FFFFFF"); actions.pack(fill="x", pady=(12, 0))
        def apply():
            try:
                self.set_range(float(minimum.get()), float(maximum.get())); dialog.destroy()
            except ValueError as exc:
                messagebox.showerror("Colorbar", str(exc), parent=dialog)
        RoundedButton(actions, "自动 / Auto", lambda: (self.set_range(None, None), dialog.destroy()), width=110).pack(side="left")
        RoundedButton(actions, "应用 / Apply", apply, accent=True, width=110).pack(side="right")
        dialog.transient(self.winfo_toplevel()); dialog.grab_set()

    def redraw(self):
        self.delete("all")
        if self.data is None: return
        raw = self.data.astype(float)
        lo = float(np.nanmin(raw)) if self.display_min is None else float(self.display_min)
        hi = float(np.nanmax(raw)) if self.display_max is None else float(self.display_max)
        arr = np.clip((raw - lo) / max(hi - lo, 1e-15), 0.0, 1.0)
        rgb = self._colors(arr)
        image = Image.fromarray(rgb, "RGB")
        side = max(64, min(self.winfo_width() - 90, self.winfo_height() - 42))
        display_side = max(64, int(side * self.zoom_factor))
        resample = Image.Resampling.NEAREST if self.zoom_factor >= 8 else Image.Resampling.BICUBIC
        image = image.resize((display_side, display_side), resample)
        self.image_ref = ImageTk.PhotoImage(image)
        image_x = max(side / 2 + 10, (self.winfo_width() - 58) / 2)
        self.create_image(image_x + self.pan_x, self.winfo_height() / 2 + 10 + self.pan_y, image=self.image_ref)
        self.create_text(14, 12, text=self.title, anchor="nw", fill="#FFFFFF", font=("Segoe UI", 10, "bold"))
        for index, label in enumerate(("+", "−", "聚焦")):
            x0 = 8 + index*35; x1 = x0 + (34 if index < 2 else 43)
            self.create_rectangle(x0, 34, x1, 60, fill="#1E2937", outline="#536273")
            self.create_text((x0+x1)/2, 47, text=label, fill="#FFFFFF", font=("Microsoft YaHei UI", 9, "bold"))
        self.create_text(14, 67, text=f"{self.zoom_factor:.2f}x｜滚轮缩放，拖动平移", anchor="nw", fill="#AAB4C0", font=("Microsoft YaHei UI", 8))
        bar_x0, bar_x1 = self.winfo_width() - 42, self.winfo_width() - 24
        bar_y0, bar_y1 = 42, max(74, self.winfo_height() - 35)
        gradient = np.linspace(1, 0, max(2, int(bar_y1-bar_y0)))[:, None]
        bar = Image.fromarray(self._colors(gradient).reshape(len(gradient), 1, 3), "RGB").resize((bar_x1-bar_x0, int(bar_y1-bar_y0)))
        self.colorbar_ref = ImageTk.PhotoImage(bar)
        self.create_image(bar_x0, bar_y0, image=self.colorbar_ref, anchor="nw")
        self.create_rectangle(bar_x0, bar_y0, bar_x1, bar_y1, outline="#D6DAE0")
        self.create_text((bar_x0+bar_x1)/2, bar_y0-6, text=f"{hi:.3g}", anchor="s", fill="#FFFFFF", font=("Segoe UI", 8))
        self.create_text((bar_x0+bar_x1)/2, bar_y1+5, text=f"{lo:.3g}", anchor="n", fill="#FFFFFF", font=("Segoe UI", 8))
        self.create_text(self.winfo_width()-12, self.winfo_height()-8, text="双击调整", anchor="se", fill="#AAB4C0", font=("Microsoft YaHei UI", 7))


class ProApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.language = "zh"
        self.words = ZH
        self.state_model = ProjectState()
        self.state_model.sampling.preview_n = 256
        self.history: list[dict[str, float]] = []
        self.last_preview = None
        self.last_image: np.ndarray | None = None
        self.last_phase: np.ndarray | None = None
        self.last_design_display: np.ndarray | None = None
        self.last_lithography_image: np.ndarray | None = None
        self.last_lithography_metrics: dict[str, float] = {}
        self.queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.optimization_active = False
        self.controls: dict[str, tk.Variable] = {}
        self.control_entries: dict[str, ttk.Entry] = {}
        self.control_rows: dict[str, ttk.Frame] = {}
        self.pages: dict[str, ttk.Frame] = {}
        self.nav_buttons: dict[str, RoundedButton] = {}
        self.i18n_widgets: list[tuple[tk.Widget, str]] = []
        self.field_labels: dict[str, ttk.Label] = {}
        self.field_label_widgets: list[tuple[str, ttk.Label]] = []
        self.phase_library: dict[str, np.ndarray] = {}
        self.current_page = "design"
        self.title(f"{APP_CN_NAME} v{APP_VERSION}")
        self.geometry("1460x900")
        self.minsize(860, 650)
        self.configure(bg=BG)
        try:
            self.iconbitmap(default=str(asset_path("app_icon.ico")))
        except (OSError, tk.TclError):
            pass
        self.protocol("WM_DELETE_WINDOW", self._close)
        self._style()
        self._layout()
        self.bind("<Configure>", self._responsive_layout)
        self._show("design")
        self.after(150, self._refresh_preview)

    def t(self, key: str) -> str:
        return self.words.get(key, key)

    def _localized_app_name(self) -> str:
        return APP_EN_NAME if self.language == "en" else APP_CN_NAME

    def _register_text(self, widget, key: str):
        self.i18n_widgets.append((widget, key))
        return widget

    def _button(self, parent, key: str, command, *, accent=False, width=132, anchor="center") -> RoundedButton:
        return self._register_text(RoundedButton(parent, self.t(key), command, accent=accent, width=width, anchor=anchor), key)

    def _style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD, relief="flat")
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=CARD, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Title.TLabel", background=BG, foreground=TEXT, font=("Segoe UI Variable Display", 20, "bold"))
        style.configure("CardTitle.TLabel", background=CARD, foreground=TEXT, font=("Segoe UI", 12, "bold"))
        style.configure("TButton", padding=(12, 7), font=("Segoe UI", 9), relief="flat")
        style.configure("Accent.TButton", background=ACCENT, foreground="white")
        style.map("Accent.TButton", background=[("active", "#115EA3"), ("disabled", "#A8C7E6")])
        style.configure("Nav.TButton", anchor="w", padding=(16, 11), background="#EEF2F6")
        style.configure("TEntry", padding=7, fieldbackground="white", bordercolor=BORDER)
        style.configure("TCombobox", padding=6)

    def _layout(self):
        sidebar = tk.Frame(self, bg="#EEF2F6", width=245)
        self.sidebar = sidebar
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        self.sidebar_app_name = tk.Label(
            sidebar,
            text=self._localized_app_name(),
            justify="left",
            wraplength=205,
            bg="#EEF2F6",
            fg=TEXT,
            font=("Segoe UI Variable Display", 15, "bold"),
        )
        self.sidebar_app_name.pack(anchor="w", padx=18, pady=(22, 18))
        for key in ("design", "ml", "imaging", "gpu", "data", "help", "changelog", "about"):
            button = self._button(sidebar, key, lambda value=key: self._show(value), width=225, anchor="w")
            button.pack(fill="x", padx=8, pady=2)
            self.nav_buttons[key] = button
        self.language_button = self._button(sidebar, "language", self._toggle_language, width=215)
        self.language_button.pack(side="bottom", fill="x", padx=12, pady=16)

        shell = ttk.Frame(self)
        shell.pack(side="left", fill="both", expand=True)
        header = ttk.Frame(shell)
        header.pack(fill="x", padx=24, pady=(17, 10))
        self.header_title = ttk.Label(header, text=self.t("title"), style="Title.TLabel")
        self.header_title.pack(side="left")
        self.status_var = tk.StringVar(value=self.t("ready"))
        self.container = ttk.Frame(shell)
        self.container.pack(fill="both", expand=True, padx=20, pady=(0, 12))
        self._build_design()
        self._build_ml()
        self._build_imaging()
        self._build_gpu()
        self._build_data()
        self._build_help()
        self._build_changelog()
        self._build_about()
        footer = tk.Frame(shell, bg="#E9EDF2", height=36)
        footer.pack(fill="x")
        footer.pack_propagate(False)
        self.status_light = tk.Canvas(footer, width=16, height=16, bg="#E9EDF2", highlightthickness=0)
        self.status_light.pack(side="left", padx=(16, 5), pady=9)
        self.status_dot = self.status_light.create_oval(3, 3, 13, 13, fill="#107C10", outline="")
        self.footer_status_label = tk.Label(footer, textvariable=self.status_var, bg="#E9EDF2", fg=TEXT, font=("Segoe UI", 9), anchor="w")
        self.footer_status_label.pack(side="left", fill="x", expand=True, pady=7)
        self.footer_progress = FluentProgress(footer, height=7)
        self.footer_progress.configure(width=120)
        self.footer_progress.pack(side="left", padx=12, pady=14)

    def _page(self, key: str) -> ttk.Frame:
        page = ttk.Frame(self.container)
        self.pages[key] = page
        return page

    def _responsive_layout(self, event=None):
        if event is not None and event.widget is not self:
            return
        width = self.winfo_width()
        compact = width < 1080
        self.sidebar.configure(width=205 if compact else 245)
        self.sidebar_app_name.configure(wraplength=168 if compact else 205,
                                        font=("Segoe UI Variable Display", 12 if compact else 15, "bold"))
        for button in self.nav_buttons.values():
            button.configure(width=185 if compact else 225)
        self.language_button.configure(width=175 if compact else 215)
        if compact:
            self.footer_progress.configure(width=82)
            if hasattr(self, "design_result_cards"):
                for index, card in enumerate(self.design_result_cards):
                    card.grid_configure(row=index, column=0, padx=0, pady=5)
                self.design_right_scroll.content.columnconfigure(1, weight=0)
        else:
            self.footer_progress.configure(width=120)
            if hasattr(self, "design_result_cards"):
                positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
                for card, (row, column) in zip(self.design_result_cards, positions):
                    card.grid_configure(row=row, column=column, padx=6, pady=6)
                self.design_right_scroll.content.columnconfigure(1, weight=1)

    def _card(self, parent, title: str) -> ttk.Frame:
        card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        label = ttk.Label(card, text=title, style="CardTitle.TLabel")
        label.pack(anchor="w", pady=(0, 9))
        card.title_label = label
        for key, value in self.words.items():
            if value == title:
                self._register_text(label, key)
                break
        return card

    def _entry(self, parent, key: str, label: str, value, width=13):
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=3)
        self.control_rows[key] = row
        label_widget = ttk.Label(row, text=FIELD_LABELS[self.language].get(key, label), style="Card.TLabel")
        label_widget.pack(side="left")
        self.field_labels[key] = label_widget
        self.field_label_widgets.append((key, label_widget))
        var = self.controls.get(key)
        if var is None:
            var = tk.StringVar(value=str(value))
            self.controls[key] = var
        entry = ttk.Entry(row, textvariable=var, width=width)
        entry.pack(side="right")
        self.control_entries[key] = entry
        return var

    def _combo_row(self, parent, key: str, variable: tk.StringVar):
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=3)
        label = ttk.Label(row, text=FIELD_LABELS[self.language].get(key, key), style="Card.TLabel")
        label.pack(side="left")
        self.field_labels[key] = label
        self.field_label_widgets.append((key, label))
        combo = FluentComboBox(row, variable, width=168)
        combo.pack(side="right")
        return combo

    def _update_source_values(self):
        source_key = getattr(self, "source_mode_key", "laser")
        polarization_key = getattr(self, "polarization_mode_key", "te")
        beam_key = getattr(self, "beam_shape_key", "plane")
        source_values = {"laser": ("激光器", "Laser"), "led": ("LED 灯", "LED")}
        polarization_values = {"te": ("TE 波", "TE"), "tm": ("TM 波", "TM"), "linear": ("线偏振", "Linear"), "lcp": ("左旋圆偏振", "Left Circular"), "rcp": ("右旋圆偏振", "Right Circular"), "unpolarized": ("非偏振", "Unpolarized")}
        beam_values = {"plane": ("平面波", "Plane Wave"), "gaussian": ("高斯光束", "Gaussian Beam")}
        index = 0 if self.language == "zh" else 1
        self.source_mode_map = {labels[index]: key for key, labels in source_values.items()}
        self.polarization_mode_map = {labels[index]: key for key, labels in polarization_values.items()}
        self.beam_shape_map = {labels[index]: key for key, labels in beam_values.items()}
        self.source_mode_combo.configure(values=list(self.source_mode_map))
        self.polarization_mode_combo.configure(values=list(self.polarization_mode_map))
        self.beam_shape_combo.configure(values=list(self.beam_shape_map))
        self.source_mode_var.set(source_values[source_key][index])
        self.polarization_mode_var.set(polarization_values[polarization_key][index])
        self.beam_shape_var.set(beam_values[beam_key][index])
        self._refresh_source_parameter_visibility()

    def _refresh_source_parameter_visibility(self):
        if not hasattr(self, "source_mode_var") or not hasattr(self, "source_mode_map"):
            return
        source_key = self.source_mode_map.get(self.source_mode_var.get(), getattr(self, "source_mode_key", "laser"))
        self.source_mode_key = source_key
        laser_rows = {"beam_shape", "waist_w0", "linear_angle"}
        led_rows = {"led_fwhm", "wavelength_samples", "led_divergence", "angle_samples"}
        for key in laser_rows | led_rows:
            row = self.control_rows.get(key)
            if row is None:
                continue
            show = key in (led_rows if source_key == "led" else laser_rows)
            if show and not row.winfo_manager():
                row.pack(fill="x", pady=3)
            elif not show and row.winfo_manager():
                row.pack_forget()
        if hasattr(self, "source_mode_note"):
            self.source_mode_note.configure(text=self.t("led_settings" if source_key == "led" else "laser_settings"))

    def _update_phase_mode_values(self):
        if not hasattr(self, "phase_mode_combo"):
            return
        current = getattr(self, "phase_mode_key", "propagation")
        labels = {
            "propagation": ("传播相位", "Propagation phase"),
            "geometric": ("几何相位（PB）", "Geometric phase (PB)"),
            "hybrid": ("传播-几何混合相位", "Hybrid propagation-geometric"),
        }
        index = 0 if self.language == "zh" else 1
        self.phase_mode_map = {value[index]: key for key, value in labels.items()}
        self.phase_mode_combo.configure(values=list(self.phase_mode_map))
        self.phase_mode_var.set(labels[current][index])

    def _build_design(self):
        page = self._page("design")
        page.columnconfigure(0, weight=0, minsize=350)
        page.columnconfigure(1, weight=1)
        page.rowconfigure(0, weight=1)
        page.rowconfigure(1, weight=0)
        scroll = FluentScrollFrame(page, background=BG)
        self.design_left_scroll = scroll
        scroll.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        core = self._card(scroll.content, self.t("core")); core.pack(fill="x", pady=(0, 10))
        for key, label, value in [
            ("wavelength", "Wavelength (µm)", .6328), ("focal", "Focal distance (λ)", 10),
            ("radius", "Lens radius (λ)", 50), ("pitch", "Meta-atom pitch (µm)", .24),
            ("preview_n", "Preview grid", 256), ("phase_levels", "Phase levels", 32)]:
            self._entry(core, key, label, value)
        self.phase_mode_var = tk.StringVar(value="传播相位" if self.language == "zh" else "Propagation phase")
        self.phase_mode_combo = self._combo_row(core, "phase_mode", self.phase_mode_var)
        self._update_phase_mode_values()
        self._entry(core, "geometric_efficiency", "Geometric conversion efficiency", .9)
        source = self._card(scroll.content, self.t("source_settings")); source.pack(fill="x", pady=(0, 10))
        self.source_mode_var = tk.StringVar()
        self.polarization_mode_var = tk.StringVar()
        self.beam_shape_var = tk.StringVar()
        self.source_mode_combo = self._combo_row(source, "source_mode", self.source_mode_var)
        self.polarization_mode_combo = self._combo_row(source, "polarization_mode", self.polarization_mode_var)
        self.beam_shape_combo = self._combo_row(source, "beam_shape", self.beam_shape_var)
        for key, label, value in [("linear_angle", "Linear angle", 0), ("waist_w0", "Gaussian waist", 20), ("led_fwhm", "LED FWHM", 20), ("wavelength_samples", "Wavelength samples", 5), ("led_divergence", "LED divergence", 5), ("angle_samples", "Angle samples", 3)]:
            self._entry(source, key, label, value)
        try:
            self.source_mode_var.trace_add("write", lambda *_args: self._refresh_source_parameter_visibility())
        except AttributeError:
            self.source_mode_var.trace("w", lambda *_args: self._refresh_source_parameter_visibility())
        self.source_mode_note = ttk.Label(source, text="", style="Card.TLabel", wraplength=290, justify="left")
        self.source_mode_note.pack(anchor="w", pady=(8, 0), fill="x")
        note = self._register_text(ttk.Label(source, text=self.t("source_note"), style="Card.TLabel", wraplength=290), "source_note")
        note.pack(anchor="w", pady=(8, 0))
        self._update_source_values()
        grid_note = self._register_text(ttk.Label(core, text=self.t("grid_note"), style="Card.TLabel", wraplength=290, justify="left"), "grid_note")
        grid_note.pack(anchor="w", pady=(8, 0), fill="x")
        smart = self._card(scroll.content, self.t("smart")); smart.pack(fill="x", pady=(0, 10))
        for key, label, value in [("iterations", "Iterations", 80), ("particles", "Particles", 32), ("seed", "Random seed", 2026), ("live_every", "Phase refresh interval", 2)]:
            self._entry(smart, key, label, value)
        self.optimization_progress = FluentProgress(smart)
        self.optimization_progress.pack(fill="x", pady=(12, 4))
        self.optimization_text = tk.StringVar(value=self.t("ready"))
        ttk.Label(smart, textvariable=self.optimization_text, style="Card.TLabel", wraplength=300).pack(anchor="w")
        litho = self._card(scroll.content, self.t("litho")); litho.pack(fill="x", pady=(0, 10))
        self.litho_enabled = tk.BooleanVar(value=True)
        self.litho_toggle = ToggleCheck(litho, self.t("enable_litho"), self.litho_enabled)
        self._register_text(self.litho_toggle.label, "enable_litho")
        self.litho_toggle.pack(anchor="w", pady=(0, 8))
        geometry_note = self._register_text(ttk.Label(litho, text=self.t("e_geometry_note"), style="Card.TLabel", wraplength=300, justify="left"), "e_geometry_note")
        geometry_note.pack(fill="x", pady=(0, 8))
        for key, label, value in [
            ("contrast_weight", "E contrast weight", 3.0), ("peak_weight", "Peak weight", .35),
            ("sidelobe_weight", "PSF sidelobe weight", .55), ("fwhm_weight", "FWHM weight", .45),
            ("e_width", "E width", 5.0), ("e_height", "E height", 4.5),
            ("e_line", "Line width", .5), ("e_middle_ratio", "Middle arm ratio", .72),
            ("e_point", "Point spacing", .25), ("e_object_distance", "Object distance", 100),
            ("e_range", "Image range", 40), ("e_grid", "Image grid", 256),
        ]:
            self._entry(litho, key, label, value)
        for key, label, value in [("target_fwhm", "Target FWHM (λ)", .36), ("target_sr", "Max sidelobe (%)", 18), ("target_contrast", "Target contrast", .145), ("angular_mag", "Angular magnification", 1.8), ("reduction", "Reduction ratio", 4.57), ("fov", "Half field angle (deg)", 7)]:
            self._entry(litho, key, label, value)
        actions = ttk.Frame(page)
        actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        actions.columnconfigure(0, weight=1)
        self.design_action_bar = actions
        self._button(actions, "refresh", self._refresh_preview, width=132).pack(side="left", padx=(0, 8))
        self.start_button = self._button(actions, "start", self._start_optimization, accent=True)
        self.start_button.pack(side="right")
        self.stop_button = self._button(actions, "stop", self._stop_optimization)
        self.stop_button.configure(state="disabled")
        self.stop_button.pack(side="right", padx=6)

        self.design_right_scroll = FluentScrollFrame(page, background=BG); self.design_right_scroll.grid(row=0, column=1, sticky="nsew")
        right = self.design_right_scroll.content
        right.columnconfigure(0, weight=1); right.columnconfigure(1, weight=1); right.rowconfigure(0, weight=1); right.rowconfigure(1, weight=1)
        preview_card = self._card(right, self.t("aerial_preview")); preview_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        self.design_preview_card = preview_card
        self.preview_heatmap = Heatmap(preview_card, height=330); self.preview_heatmap.pack(fill="both", expand=True)
        self._export_buttons(preview_card, lambda: getattr(self, "last_design_display", None), "design_preview")
        metric_card = self._card(right, self.t("metric_help")); metric_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6))
        self.metric_plot = MetricPlot(metric_card, height=330); self.metric_plot.pack(fill="both", expand=True)
        convergence_card = self._card(right, self.t("convergence")); convergence_card.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(6, 0))
        self.convergence_plot = MetricPlot(convergence_card, height=260); self.convergence_plot.pack(fill="both", expand=True)
        phase_card = self._card(right, self.t("phase")); phase_card.grid(row=1, column=1, sticky="nsew", padx=(6, 0), pady=(6, 0))
        self.phase_heatmap = Heatmap(phase_card, height=250); self.phase_heatmap.pack(fill="both", expand=True)
        self._export_buttons(phase_card, lambda: self.last_phase, "phase_profile")
        self.design_result_cards = [preview_card, metric_card, convergence_card, phase_card]

    def _export_buttons(self, parent, provider, stem: str):
        row = ttk.Frame(parent, style="Card.TFrame"); row.pack(fill="x", pady=(8, 0))
        self._button(row, "export_data", lambda: self._export_matrix(provider(), stem)).pack(side="left")
        self._button(row, "export_image", lambda: self._export_picture(provider(), stem)).pack(side="left", padx=6)

    def _build_ml(self):
        page = self._page("ml")
        page.columnconfigure(0, weight=1); page.columnconfigure(1, weight=1); page.rowconfigure(0, weight=1)
        left_scroll = FluentScrollFrame(page, background=BG); left_scroll.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        left = self._card(left_scroll.content, self.t("dataset")); left.pack(fill="x", pady=(0, 8))
        self._register_text(ttk.Label(left, text=self.t("ml_help"), style="Card.TLabel", wraplength=280), "ml_help").pack(anchor="w", pady=(0, 12))
        self.history_path = tk.StringVar(value=str(self._data_dir()))
        path_row = ttk.Frame(left, style="Card.TFrame"); path_row.pack(fill="x")
        ttk.Entry(path_row, textvariable=self.history_path).pack(side="left", fill="x", expand=True)
        self._button(path_row, "browse", self._browse_history, width=90).pack(side="left", padx=(6, 0))
        self.ml_progress = FluentProgress(left); self.ml_progress.pack(fill="x", pady=14)

        targets = self._card(left_scroll.content, self.t("ml_targets")); targets.pack(fill="x", pady=8)
        for key, value in [("ml_wavelength", .6328), ("ml_focal", 10), ("ml_radius", 50), ("ml_fwhm", .36), ("ml_sidelobe", 18), ("ml_contrast", .145)]:
            self._entry(targets, key, key, value)
        settings = self._card(left_scroll.content, self.t("ml_settings")); settings.pack(fill="x", pady=8)
        self._entry(settings, "ml_ensembles", "ml_ensembles", 12)
        self._entry(settings, "ml_alpha", "ml_alpha", .08)
        self._button(left_scroll.content, "train_design", self._start_ml_design, accent=True, width=180).pack(anchor="e", pady=12)

        right = self._card(page, self.t("prediction")); right.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        ml_metrics = ttk.Frame(right, style="Card.TFrame"); ml_metrics.pack(fill="x", pady=(0, 8))
        self.ml_confidence_value = self._metric_tile(ml_metrics, "confidence", "-", 0)
        self.ml_samples_value = self._metric_tile(ml_metrics, "samples", "0", 1)
        self.ml_validation_value = self._metric_tile(ml_metrics, "validation", "-", 2)
        self.ml_summary_var = tk.StringVar(value=self.t("ready"))
        ttk.Label(right, textvariable=self.ml_summary_var, style="Card.TLabel", wraplength=280, justify="left").pack(fill="x", pady=8)
        self.ml_confidence_bar = FluentProgress(right); self.ml_confidence_bar.pack(fill="x", pady=(0, 8))
        self.ml_phase_heatmap = Heatmap(right, height=220); self.ml_phase_heatmap.pack(fill="both", expand=True, pady=(10, 5))
        self.ml_heatmap = Heatmap(right, height=260); self.ml_heatmap.pack(fill="both", expand=True, pady=(5, 0))
        self._export_buttons(right, lambda: self.last_preview.intensity if self.last_preview else None, "ml_designed_intensity")

    def _build_gpu(self):
        page = self._page("gpu")
        page.columnconfigure(0, weight=1); page.columnconfigure(1, weight=1); page.rowconfigure(0, weight=1)
        devices_card = self._card(page, self.t("available_devices")); devices_card.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        top = ttk.Frame(devices_card, style="Card.TFrame"); top.pack(fill="x")
        self.backend_var = tk.StringVar(value="0-Auto")
        self.backend_combo = FluentComboBox(top, self.backend_var,
            values=["0-Auto", "1-CPU", "2-NVIDIA CUDA", "3-OpenCL", "4-Multi GPU"], width=210)
        self.backend_combo.pack(side="left")
        self._button(top, "gpu_refresh", self._detect_gpu, width=110).pack(side="right")
        gpu_limits = ttk.Frame(devices_card, style="Card.TFrame"); gpu_limits.pack(fill="x", pady=(10, 0))
        self._entry(gpu_limits, "gpu_memory_fraction", "VRAM limit", .75)
        self._entry(gpu_limits, "gpu_batch_size", "GPU batch", 2)
        self.manual_thread_var = tk.BooleanVar(value=bool(getattr(self.state_model.sampling, "manual_thread_mode", False)))
        self.thread_toggle = ToggleCheck(gpu_limits, self.t("manual_threads"), self.manual_thread_var, command=self._update_thread_controls)
        self._register_text(self.thread_toggle.label, "manual_threads")
        self.thread_toggle.pack(anchor="w", pady=(8, 2))
        self._entry(gpu_limits, "thread_count", "CPU threads", str(optics.recommended_thread_count()))
        self.thread_policy_label = ttk.Label(gpu_limits, text="", style="Card.TLabel", wraplength=300, justify="left")
        self.thread_policy_label.pack(anchor="w", pady=(5, 0), fill="x")
        self._update_thread_controls()
        self._register_text(ttk.Label(gpu_limits, text=self.t("gpu_scheduler_note"), style="Card.TLabel", wraplength=300, justify="left"), "gpu_scheduler_note").pack(anchor="w", pady=(8, 0), fill="x")
        action = ttk.Frame(devices_card, style="Card.TFrame")
        self.gpu_action_frame = action
        action.pack(side="bottom", fill="x", pady=(10, 0))
        self.gpu_apply_button = self._button(action, "apply", self._apply_gpu)
        self.gpu_apply_button.pack(side="left")
        self.gpu_benchmark_button = self._button(action, "benchmark", self._benchmark_gpu, accent=True, width=155)
        self.gpu_benchmark_button.pack(side="right")
        selection_actions = ttk.Frame(devices_card, style="Card.TFrame")
        selection_actions.pack(side="bottom", fill="x", pady=(8, 0))
        self._button(selection_actions, "gpu_select_all", lambda: self._set_all_gpus(True), width=145).pack(side="left")
        self._button(selection_actions, "gpu_clear", lambda: self._set_all_gpus(False), width=135).pack(side="right")
        self.gpu_device_scroll = FluentScrollFrame(devices_card, background=CARD)
        self.gpu_device_scroll.pack(fill="both", expand=True, pady=(12, 0))
        self.gpu_device_frame = self.gpu_device_scroll.content
        self.gpu_vars: dict[str, tk.BooleanVar] = {}
        self.gpu_checks: dict[str, ToggleCheck] = {}
        self.gpu_display_labels: dict[str, str] = {}

        summary = self._card(page, self.t("device_summary")); summary.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        self.gpu_status_title = ttk.Label(summary, text=self.t("ready"), style="CardTitle.TLabel")
        self.gpu_status_title.pack(anchor="w")
        self.gpu_status_detail = ttk.Label(summary, text="", style="Card.TLabel", wraplength=280, justify="left")
        self.gpu_status_detail.pack(anchor="w", fill="x", pady=8)
        self.gpu_progress = FluentProgress(summary); self.gpu_progress.pack(fill="x", pady=12)
        metrics = ttk.Frame(summary, style="Card.TFrame"); metrics.pack(fill="x", pady=8)
        self.gpu_metric_backend = self._metric_tile(metrics, "backend_tile", "CPU", 0)
        self.gpu_metric_devices = self._metric_tile(metrics, "devices_tile", "0", 1)
        self.gpu_metric_speed = self._metric_tile(metrics, "speed_tile", "-", 2)
        self.gpu_notes_frame = ttk.Frame(summary, style="Card.TFrame")
        self.gpu_notes_frame.pack(fill="both", expand=True, pady=(12, 0))
        self.after(300, self._detect_gpu)

    def _metric_tile(self, parent, title_key: str, value: str, column: int):
        tile = tk.Frame(parent, bg="#F1F5F9", padx=12, pady=10)
        tile.grid(row=0, column=column, sticky="nsew", padx=4)
        parent.columnconfigure(column, weight=1)
        title = self._register_text(tk.Label(tile, text=self.t(title_key), bg="#F1F5F9", fg=MUTED, font=("Segoe UI", 8)), title_key)
        title.pack(anchor="w")
        label = tk.Label(tile, text=value, bg="#F1F5F9", fg=TEXT, font=("Segoe UI", 15, "bold"))
        label.pack(anchor="w")
        return label

    def _thread_policy_text(self) -> str:
        logical = os.cpu_count() or 1
        if hasattr(self, "manual_thread_var") and self.manual_thread_var.get():
            value = self.controls.get("thread_count", tk.StringVar(value="")).get()
            return (f"手动模式：将使用 {value} 个 CPU 线程。"
                    if self.language == "zh"
                    else f"Manual mode: {value} CPU thread(s) will be used.")
        recommended = optics.recommended_thread_count()
        return (f"自动模式：检测到 {logical} 个逻辑 CPU 线程，当前建议使用 {recommended} 个线程。"
                if self.language == "zh"
                else f"Auto mode: {logical} logical CPU thread(s) detected; {recommended} thread(s) recommended.")

    def _update_thread_controls(self):
        manual = bool(getattr(self, "manual_thread_var", tk.BooleanVar(value=False)).get())
        entry = self.control_entries.get("thread_count")
        if entry is not None:
            entry.configure(state="normal" if manual else "disabled")
        if hasattr(self, "thread_policy_label"):
            self.thread_policy_label.configure(text=self._thread_policy_text())

    def _build_imaging(self):
        page = self._page("imaging")
        page.columnconfigure(0, weight=0, minsize=330); page.columnconfigure(1, weight=1); page.rowconfigure(0, weight=1)
        controls = self._card(page, self.t("imaging")); controls.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        note = self._register_text(ttk.Label(controls, text=self.t("e_geometry_note"), style="Card.TLabel", wraplength=290, justify="left"), "e_geometry_note")
        note.pack(fill="x", pady=(0, 10))
        for key, label, value in [("e_width", "E width", 5.0), ("e_height", "E height", 4.5), ("e_line", "Line width", .5), ("e_middle_ratio", "Middle arm ratio", .72), ("e_point", "Point spacing", .25), ("e_object_distance", "Object distance", 100), ("e_range", "Image range", 40), ("e_grid", "Image grid", 256)]:
            self._entry(controls, key, label, value)
        source_label = self._register_text(ttk.Label(controls, text=self.t("phase_source"), style="Card.TLabel"), "phase_source")
        source_label.pack(anchor="w", pady=(14, 4))
        self.phase_source_var = tk.StringVar(value="current")
        self.phase_source_combo = FluentComboBox(controls, self.phase_source_var, width=270)
        self.phase_source_combo.pack(fill="x", pady=(0, 6))
        self._update_phase_source_values()
        source_actions = ttk.Frame(controls, style="Card.TFrame"); source_actions.pack(fill="x")
        self._button(source_actions, "import_phase", self._import_phase, width=130).pack(side="left")
        self._button(controls, "simulate_e", self._run_e_imaging, accent=True, width=260).pack(fill="x", pady=(16, 5))
        self._button(controls, "export_points", self._export_e_points, width=260).pack(fill="x", pady=5)
        result = self._card(page, self.t("imaging")); result.grid(row=0, column=1, sticky="nsew")
        self.e_heatmap = Heatmap(result, height=620); self.e_heatmap.pack(fill="both", expand=True)
        self._export_buttons(result, lambda: self.last_image, "letter_e_image")

    def _run_e_imaging(self):
        if not self._sync(): return
        try:
            width = float(self.controls["e_width"].get()); height = float(self.controls["e_height"].get())
            line = float(self.controls["e_line"].get()); point = float(self.controls["e_point"].get())
            middle_ratio = float(self.controls["e_middle_ratio"].get())
            spacing = max(0.0, (height - line) / 2.0)
            self.e_points = optics.generate_e_points(spacing, width, line, point, middle_arm_ratio=middle_ratio)
            img = self.state_model.imaging
            img.objective_distance_um = float(self.controls["e_object_distance"].get())
            img.focal_length_um, _magnification = optics.projection_image_geometry(self.state_model)
            image_range = abs(float(self.controls["e_range"].get()))
            img.xs_min_um, img.xs_max_um = -image_range / 2, image_range / 2
            img.grid_n = int(float(self.controls["e_grid"].get()))
            folder = self._data_dir() / "ImagingPreview"
            phase = self._selected_phase()
            paths = optics.calculate_imaging(self.state_model, self.e_points, folder, phase_override=phase)
            txt = next(path for path in paths if path.name == "ImageOnXY_Plane_Z0.txt")
            self.last_image = np.loadtxt(txt)
            self.e_heatmap.set_data(self.last_image, self.t("imaging"))
            self.e_heatmap.focus_signal()
            self.status_var.set("Letter-E imaging complete; use the wheel or Focus button to inspect details" if self.language == "en" else "字母 E 成像完成；可使用滚轮或“聚焦”按钮查看细节")
        except Exception as exc:
            messagebox.showerror(self._localized_app_name(), str(exc))

    def _update_phase_source_values(self):
        old_map = getattr(self, "phase_source_map", {})
        current_value = self.phase_source_var.get() if hasattr(self, "phase_source_var") else "current"
        current_key = old_map.get(current_value, current_value if current_value in {"current", "optimized", "ml", "imported"} else "current")
        values = [("current", self.t("phase_current")), ("optimized", self.t("phase_optimized")), ("ml", self.t("phase_ml")), ("imported", self.t("phase_imported"))]
        self.phase_source_map = {label: key for key, label in values}
        self.phase_source_labels = {key: label for key, label in values}
        if hasattr(self, "phase_source_combo"):
            self.phase_source_combo.configure(values=[label for _, label in values])
            self.phase_source_var.set(self.phase_source_labels.get(current_key, self.phase_source_labels["current"]))

    def _selected_phase(self) -> np.ndarray | None:
        value = self.phase_source_var.get()
        key = self.phase_source_map.get(value, value if value in {"current", "optimized", "ml", "imported"} else "current")
        if key == "current":
            return self.phase_library.get("current", self.last_phase)
        phase = self.phase_library.get(key)
        if phase is None:
            raise ValueError(self.t("phase_source") + ": " + self.phase_source_labels.get(key, key))
        return phase

    def _import_phase(self):
        path = filedialog.askopenfilename(filetypes=[("Phase matrix", "*.txt *.csv"), ("All files", "*.*")])
        if not path: return
        delimiter = "," if Path(path).suffix.lower() == ".csv" else None
        phase = np.loadtxt(path, delimiter=delimiter)
        if phase.ndim != 2 or phase.shape[0] != phase.shape[1]:
            raise ValueError("Phase data must be a square two-dimensional matrix.")
        self.phase_library["imported"] = phase
        self._save_phase_auto("imported", phase)
        self.phase_source_var.set(self.phase_source_labels["imported"])

    def _export_e_points(self):
        if not hasattr(self, "e_points"):
            self._run_e_imaging()
        if not hasattr(self, "e_points"): return
        path = filedialog.asksaveasfilename(initialfile="Letter_E_Object.csv", defaultextension=".csv", filetypes=[("CSV", "*.csv"), ("Text", "*.txt")])
        if path: export_array(Path(path), self.e_points, ["x_um", "y_um"])

    def _build_data(self):
        page = self._page("data")
        scroll = FluentScrollFrame(page, background=BG); scroll.pack(fill="both", expand=True)
        card = self._card(scroll.content, self.t("data")); card.pack(fill="x")
        self.data_path = tk.StringVar(value=str(self._data_dir()))
        self.optimization_data_path = self.data_path
        row = ttk.Frame(card, style="Card.TFrame"); row.pack(fill="x")
        self._register_text(ttk.Label(row, text=self.t("design_output_path"), style="Card.TLabel"), "design_output_path").pack(side="left")
        ttk.Entry(row, textvariable=self.data_path).pack(side="left", fill="x", expand=True, padx=8)
        self._button(row, "browse", self._browse_data, width=90).pack(side="left")
        self._register_text(ttk.Label(card, text=self.t("path_note"), style="Card.TLabel", wraplength=620, justify="left"), "path_note").pack(anchor="w", pady=(8, 0))
        buttons = ttk.Frame(card, style="Card.TFrame"); buttons.pack(fill="x", pady=14)
        self._button(buttons, "save_project", self._save_project).pack(side="left")
        self._button(buttons, "load_project", self._load_project).pack(side="left", padx=6)
        self._button(buttons, "open_folder", self._open_data).pack(side="right")
        export_card = self._card(scroll.content, self.t("export_all")); export_card.pack(fill="x", pady=10)
        self._register_text(ttk.Label(export_card, text=self.t("export_description"), style="Card.TLabel", wraplength=500), "export_description").pack(anchor="w", pady=(0, 10))
        self.export_vars = {}
        descriptions = {
            "export_project": ("JSON、透镜参数与设计清单", "JSON project, lens parameters and design manifest"),
            "export_opt": ("迭代指标、收敛曲线与优化摘要", "Iteration metrics, convergence data and optimization summary"),
            "export_phase": ("相位矩阵 TXT/CSV 与 PNG/JPG", "Phase matrices as TXT/CSV and PNG/JPG"),
            "export_focal": ("焦面强度、PSF 和字母 E 成像", "Focal intensity, PSF and Letter-E imaging"),
            "export_ml": ("训练样本、置信度、误差与预测向量", "Training samples, confidence, error and predicted vector"),
        }
        for key, texts in descriptions.items():
            item = tk.Frame(export_card, bg="#F7F9FB", padx=12, pady=8)
            item.pack(fill="x", pady=3)
            var = tk.BooleanVar(value=True); self.export_vars[key] = var
            toggle = ToggleCheck(item, self.t(key), var); toggle.pack(side="left")
            self._register_text(toggle.label, key)
            detail = tk.Label(item, text=texts[0], bg="#F7F9FB", fg=MUTED, font=("Segoe UI", 9), anchor="w")
            detail.pack(side="left", padx=16, fill="x", expand=True)
            detail._localized_values = texts
            self.localized_value_labels = getattr(self, "localized_value_labels", []) + [detail]
        self._button(export_card, "export_all", self._export_all, accent=True, width=190).pack(anchor="e", pady=(12, 0))
        self.export_status_var = tk.StringVar(value=self.t("ready"))
        status_card = self._card(scroll.content, self.t("process")); status_card.pack(fill="x", pady=(0, 15))
        self.export_status_light = tk.Canvas(status_card, width=18, height=18, bg=CARD, highlightthickness=0)
        self.export_status_light.pack(side="left")
        self.export_status_light.create_oval(3, 3, 15, 15, fill="#107C10", outline="")
        ttk.Label(status_card, textvariable=self.export_status_var, style="Card.TLabel", wraplength=480).pack(side="left", padx=8)

    def _build_about(self):
        page = self._page("about")
        card = self._card(page, APP_EN_NAME); card.pack(fill="both", expand=True)
        text = (f"Version {APP_VERSION}\n{APP_PUBLISHER}\n\n"
                f"{APP_COPYRIGHT}\n\n"
                "Open-source license notice / 开源许可声明：本源码包按 MIT License 授权用于研究、教学、内部工程验证、修改和再分发，"
                "需保留版权声明和许可证声明。第三方开源组件保留其各自许可证，包括 Python、NumPy、Pillow、Tk/Tcl 和可选 GPU 库。\n"
                "MIT License summary: permission is granted to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies, "
                "provided that copyright and license notices are preserved. The software is provided \"as is\", without warranty.\n\n"
                "Physics: calibrated Fourier optics, Fresnel and angular-spectrum propagation, high-NA vector weighting, polychromatic workflows.\n"
                "Optimization: hybrid PSO/Marine Predator with reproducible history and convergence monitoring.\n"
                "Historical ML: local ensemble ridge surrogate with validation error and uncertainty.\n"
                "Reliability: CPU fallback, background workers, atomic project exports and source-level portability.")
        ttk.Label(card, text=text, style="Card.TLabel", wraplength=520, justify="left").pack(anchor="nw")

    def _build_help(self):
        page = self._page("help")
        scroll = FluentScrollFrame(page, background=BG)
        scroll.pack(fill="both", expand=True)
        card = self._card(scroll.content, self.t("help_title")); card.pack(fill="both", expand=True)
        zh = """1. 普通聚焦模式：以焦斑 FWHM、PSF 旁瓣泄漏和峰值强度为主要目标，适合点聚焦与基础超透镜设计。\n\n2. 投影光刻模式：目标位于像方，优先优化字母 E 可辨识度、轮廓重合、笔画均匀性和可解析度；PSF 旁瓣是次级杂散光指标，不能代替 CD、NILS 和图形质量。可直接设置 E 几何、物距、像距、成像范围和网格。\n\n3. 相位方式：传播相位按光程差随波长缩放；几何相位采用 Pancharatnam-Berry 关系 φ=2σθ，适合圆偏振并考虑转换效率；混合模式组合两者。\n\n4. 历史数据智能设计：训练集同时使用最终结果和带设计向量的优化过程样本，并显示相位、焦面及结果弹窗。\n\n5. 网格与实时预览：更高网格改善细节采样，但增加 FFT、显存与运行时间。优化时相位与焦面同步更新。\n\n6. 计算设备：CUDA 使用有上限的显存池和批量 FFT，可设置显存占用与批处理数；失败时自动回退 CPU。\n\n7. 数据与导出：数值支持 TXT/CSV，图像支持 PNG/JPG/BMP/TIFF。"""
        en = """1. Normal focusing mode primarily optimizes focal FWHM, PSF sidelobe leakage and peak intensity.\n\n2. Projection lithography mode defines the target in the image plane and prioritizes Letter-E readability, shape overlap, stroke uniformity and resolvability. PSF sidelobe leakage is secondary and does not replace CD, NILS or pattern quality. E geometry, distances, range and grid are configurable.\n\n3. Phase implementation: propagation phase scales with optical path and wavelength; geometric phase uses the Pancharatnam-Berry relation phi=2 sigma theta and conversion efficiency; hybrid mode combines both.\n\n4. Historical ML uses final solutions and trajectory rows containing design vectors, and displays phase, focal and result views.\n\n5. Higher grids improve sampling but increase FFT time and VRAM. Phase and focal maps update together.\n\n6. CUDA uses bounded per-device VRAM pools and distributes independent source-sample batches across every selected CUDA GPU; OpenCL profile rows are split across all selected OpenCL GPUs. A single indivisible FFT runs on one device. Failures fall back to CPU.\n\n7. Numerical results support TXT/CSV and images support PNG/JPG/BMP/TIFF."""
        self.help_text = ttk.Label(card, text=zh, style="Card.TLabel", wraplength=520, justify="left")
        self.help_text.pack(anchor="nw", fill="x")
        self.help_texts = {"zh": zh, "en": en}

    def _build_changelog(self):
        page = self._page("changelog")
        scroll = FluentScrollFrame(page, background=BG); scroll.pack(fill="both", expand=True)
        card = self._card(scroll.content, self.t("changelog")); card.pack(fill="x")
        entries = [
            ("6.1.1", "新增 CPU 线程调度模式：默认按本机逻辑 CPU 自动分配计算线程，勾选后可手动指定线程数量，并同步到 NumPy/OpenMP/MKL 等计算后端。", "Added CPU thread scheduling: automatic mode chooses a sensible thread count from the local logical CPUs, while manual mode lets users set the exact count and applies it to NumPy/OpenMP/MKL-style compute backends."),
            ("6.1.0", "光源与偏振设置按激光器/LED 动态切换参数；普通聚焦模式明确加入预览/相位网格说明并改善相位热图插值；GPU 页面增加多卡调度说明、VRAM/后端诊断并减少 OpenCL 重复编译开销；关于页加入 MIT 许可证中文声明。", "Laser and LED source settings now switch dynamically; normal focusing mode clarifies the preview/phase grid and phase heatmap smoothing; the GPU page adds multi-device scheduling diagnostics, VRAM/backend details and reduced OpenCL rebuild overhead; the About page includes MIT license wording."),
            ("6.0.1", "设计优化页新增底部固定命令栏，开始优化、刷新预览和停止按钮不再随左侧参数滚动；优化自动归档改为写入用户设置的数据路径；版权声明页增加开源许可说明，并优化小窗口下按钮可见性。", "Added a fixed bottom command bar on the design page so Start, Refresh and Stop no longer scroll with parameters; automatic optimization archives now use the user-selected data path; the copyright page now includes an open-source license notice, with compact-window button visibility improved."),
            ("6.0.0", "新增物理模型审计、超原子数据库质量闭环、实验校准建议、多 GPU 可断点批量队列和更完整的科研报告汇总；进一步强化历史智能设计、工程交付、Windows 11 风格一致性与小窗口可用性。", "Added physics model audit, meta-atom database quality loop, experimental calibration advice, checkpointed multi-GPU batch queues and richer research reports; strengthened historical intelligence, engineering delivery, Windows 11 UI consistency and compact-window usability."),
            ("5.1.7", "修复 40 张同型号 GPU 场景下设备标签重复导致选择数量被压缩的问题：界面和后端均使用带序号的唯一设备 key，右侧计数、保存数量和实际后端筛选保持一致。", "Fixed large same-model GPU systems where duplicate display labels collapsed selections: the UI and backend now use indexed unique device keys, keeping the shown count, saved selection and backend filtering consistent."),
            ("5.1.6", "增强 Windows 10 下 GPU 全选稳定性：全选/清空后显式刷新每一个设备勾选控件，并兼容旧 Tcl/Tk 的 trace API，避免系统主题导致勾选状态不显示。", "Hardened GPU Select All on Windows 10 by explicitly refreshing every device checkbox after select-all/clear, with fallback support for older Tcl/Tk trace APIs."),
            ("5.1.5", "修复优化完成弹窗与主界面最终预览不一致的问题：弹窗和最终导出的字母 E 空中像现在复用主界面同一份当前网格结果，不再隐藏截断到 256 网格。", "Fixed final optimization result consistency: the completion popup and exported Letter-E aerial image now reuse the same current-grid result shown in the main preview, with no hidden 256-grid truncation."),
            ("5.1.4", "修复 GPU 全选按钮只更新数据、不刷新勾选控件显示的问题；自定义 ToggleCheck 现在会监听外部变量变化，并新增全选/清空视觉状态回归测试。", "Fixed GPU Select All updating data without repainting the visible checkbox state; ToggleCheck now listens to external variable changes, with regression coverage for visual select-all/clear states."),
            ("5.1.3", "计算设备不再假定最多两块 GPU：支持任意数量设备的全选、清空、自由组合和刷新后保留选择；CUDA 光源采样批次会轮转分配到所选 GPU；安装与卸载按钮改为紧凑直角矩形，并继续固定在小窗口底部。", "Removed the two-GPU assumption with select-all, clear, arbitrary device combinations and refresh persistence; distributed CUDA source batches across selected GPUs; changed installer/uninstaller actions to compact rectangular buttons fixed in view."),
            ("5.1.2", "修复安装器和卸载器自定义 Fluent 控件在 ttk.Frame 中读取背景色导致的启动崩溃，并加强安装程序真实窗口启动验证。", "Fixed installer/uninstaller startup crashes caused by Fluent controls reading background from ttk.Frame, and strengthened real installer-window launch verification."),
            ("5.1.1", "字母 E 几何参数改为总宽度、总高度、笔画宽度和中横画比例；点间距明确为离散精度；删除手动像距，按焦距和物距自动计算实像面；普通模式显示焦平面 PSF，光刻模式显示像面 E 空中像，并分别导出。", "Clarified Letter-E geometry as overall width, height, stroke width and middle-arm ratio; identified point spacing as discretization; removed manual image distance and derived the real image plane from focal/object distance; separated focal PSF and image-plane E aerial-image display and export."),
            ("5.1.0", "新增传播相位、几何相位与混合相位；光刻设计加入字母 E 几何、物距/像距、网格和目标权重；修复实时焦面不更新；历史机器学习纳入优化过程样本并显示相位与结果弹窗；CUDA 加入显存预算和批量 FFT；安装与卸载界面统一为 Fluent 风格。", "Added propagation, geometric and hybrid phase implementations; Letter-E geometry, distances, grids and objective weights in lithography; fixed live focal updates; included optimization trajectories in historical ML with phase/result views; added bounded batched CUDA VRAM use and Fluent installer/uninstaller UI."),
            ("5.0.2", "修复极小字母 E 被误判为高可辨识度的问题：加入像面包围盒、笔画宽度和横条间距的像素可解析度硬约束；投影角放大率同步到成像模型；全部热图支持滚轮缩放、按钮缩放、拖动平移和自动聚焦。", "Fixed tiny Letter-E patterns being misreported as highly readable by enforcing pixel resolvability for glyph bounds, stroke width and bar spacing; synchronized angular magnification into imaging; added wheel/button zoom, pan and auto-focus to every heatmap."),
            ("5.0.1", "修正字母 E 对比度：采用稳健前景/背景分离、结构相关性、笔画均匀性与轮廓重合度组成的可辨识度指标；全部色图加入可调色标；合并历史智能模块并统一 Windows 11 界面。", "Corrected Letter-E contrast with robust foreground/background separation, structural correlation, stroke uniformity and shape overlap; added adjustable colorbars to all heatmaps, merged historical intelligence and unified Windows 11 UI."),
            ("5.0.0", "新增超原子数据库、伴随逆向设计、制造鲁棒性、光刻计量、实验标定、项目版本与科研报告。", "Added meta-atom databases, adjoint inverse design, manufacturing robustness, lithography metrology, experimental calibration, project versioning and research reports."),
            ("3.5.0", "新增任意角度线偏振和左右旋圆偏振，统一光源 Fluent 选择控件及字母 E 对比度标识。", "Added arbitrary-angle linear and left/right circular polarization, Fluent source selectors and explicit Letter-E contrast labels."),
            ("3.4.0", "新增激光器/LED、TE/TM/非偏振及光谱和角度非相干积分。", "Added laser/LED, TE/TM/unpolarized illumination, and spectral/angular incoherent integration."),
            ("3.3.0", "投影光刻对比度改为像方字母 E 的 Michelson 对比度，并加入图案相关性与笔画均匀性约束。", "Projection-lithography contrast now uses image-side Letter-E Michelson contrast with pattern-correlation and stroke-uniformity constraints."),
            ("3.2.4", "设备列表独立滚动，GPU 数量较多时底部操作按钮保持可见。", "Independent device-list scrolling keeps bottom actions visible with many GPUs."),
            ("3.2.3", "修复并列滚动区域同时响应鼠标滚轮的问题。", "Fixed simultaneous wheel scrolling in adjacent panels."),
            ("3.2.2", "侧栏软件名称完整支持中英文即时切换，统一发布版本信息。", "Completed bilingual sidebar naming and unified release metadata."),
            ("3.2.0", "响应式小窗口布局、优化结果弹窗、机器学习目标、Fluent 进度条、分类导出中心。", "Responsive compact layout, result dialog, ML targets, Fluent progress and categorized exports."),
            ("3.1.0", "完整中英文切换、统一校徽、多 GPU 选择、相位来源、帮助页与底部状态栏。", "Complete bilingual switching, unified emblem, multi-GPU selection, phase sources, help and status bar."),
            ("3.0.0", "物理计算内核、历史机器学习、实时图形监控、Windows 安装器与统一导出。", "Physics engine, historical ML, live charts, Windows installer and unified exports."),
        ]
        for version, zh_note, en_note in entries:
            row = tk.Frame(card, bg="#F7F9FB", padx=14, pady=12)
            row.pack(fill="x", pady=5)
            tk.Label(row, text="v" + version, bg="#F7F9FB", fg=ACCENT, font=("Segoe UI", 12, "bold")).pack(anchor="w")
            note = tk.Label(row, text=zh_note, bg="#F7F9FB", fg=TEXT, font=("Segoe UI", 10), wraplength=520, justify="left")
            note.pack(anchor="w", pady=(4, 0))
            note._localized_values = (zh_note, en_note)
            self.localized_value_labels = getattr(self, "localized_value_labels", []) + [note]

    def _show(self, key: str):
        for page in self.pages.values(): page.pack_forget()
        self.pages[key].pack(fill="both", expand=True)
        self.header_title.configure(text=self.t(key))
        self.current_page = key

    def _set_status(self, text: str, *, kind: str = "ready", progress: float | None = None):
        self.status_var.set(text)
        colors = {"ready": "#107C10", "busy": "#0F6CBD", "warning": "#F5A623", "error": "#C50F1F"}
        if hasattr(self, "status_light"):
            self.status_light.itemconfigure(self.status_dot, fill=colors.get(kind, colors["ready"]))
        if hasattr(self, "footer_progress"):
            self.footer_progress["value"] = 0 if progress is None else max(0, min(100, progress))

    def _contrast_label(self) -> str:
        if self.state_model.lithography.enabled:
            return "字母 E 可辨识度" if self.language == "zh" else "E Readability"
        return "对比度" if self.language == "zh" else "Contrast"

    def _toggle_language(self):
        if hasattr(self, "source_mode_var"):
            self.source_mode_key = self.source_mode_map.get(self.source_mode_var.get(), "laser")
            self.polarization_mode_key = self.polarization_mode_map.get(self.polarization_mode_var.get(), "te")
            self.beam_shape_key = self.beam_shape_map.get(self.beam_shape_var.get(), "plane")
        if hasattr(self, "phase_mode_var"):
            self.phase_mode_key = self.phase_mode_map.get(self.phase_mode_var.get(), "propagation")
        self.language = "en" if self.language == "zh" else "zh"
        self.words = EN if self.language == "en" else ZH
        for widget, key in self.i18n_widgets:
            try: widget.configure(text=self.t(key))
            except tk.TclError: pass
        for key, label in self.field_label_widgets:
            label.configure(text=FIELD_LABELS[self.language].get(key, key))
        self.header_title.configure(text=self.t(self.current_page))
        self.sidebar_app_name.configure(text=self._localized_app_name())
        self.title(f"{self._localized_app_name()} v{APP_VERSION}")
        if hasattr(self, "help_text"):
            self.help_text.configure(text=self.help_texts[self.language])
        for label in getattr(self, "localized_value_labels", []):
            label.configure(text=label._localized_values[0 if self.language == "zh" else 1])
        self._update_source_values()
        self._update_phase_mode_values()
        self._update_phase_source_values()
        self._update_thread_controls()
        self._update_design_preview_title()
        self._set_status(self.t("ready"))

    def _update_design_preview_title(self):
        if hasattr(self, "design_preview_card"):
            key = "aerial_preview" if self.litho_enabled.get() else "focal_preview"
            self.design_preview_card.title_label.configure(text=self.t(key))

    def _sync(self) -> bool:
        try:
            s = self.state_model
            s.lens.wavelength_um = float(self.controls["wavelength"].get())
            s.lens.working_distance_lambda = float(self.controls["focal"].get())
            s.lens.lens_radius_lambda = float(self.controls["radius"].get())
            s.lens.pitch_um = float(self.controls["pitch"].get())
            self.source_mode_key = self.source_mode_map.get(self.source_mode_var.get(), "laser")
            self.polarization_mode_key = self.polarization_mode_map.get(self.polarization_mode_var.get(), "te")
            self.beam_shape_key = self.beam_shape_map.get(self.beam_shape_var.get(), "plane")
            s.source.light_source_mode = {"laser": 0, "led": 1}[self.source_mode_key]
            s.source.polarization_mode = {"te": 0, "tm": 1, "unpolarized": 2, "linear": 3, "lcp": 4, "rcp": 5}[self.polarization_mode_key]
            s.source.theta_polar_angle_deg = float(self.controls["linear_angle"].get())
            s.source.beam_shape = {"plane": 0, "gaussian": 1}[self.beam_shape_key]
            s.source.waist_w0_lambda = float(self.controls["waist_w0"].get())
            s.source.led_fwhm_nm = float(self.controls["led_fwhm"].get())
            s.source.wavelength_samples = int(float(self.controls["wavelength_samples"].get()))
            s.source.led_divergence_half_angle_deg = float(self.controls["led_divergence"].get())
            s.source.angle_samples = int(float(self.controls["angle_samples"].get()))
            s.sampling.preview_n = int(float(self.controls["preview_n"].get()))
            s.optimization.phase_n = int(float(self.controls["phase_levels"].get()))
            s.optimization.phase_design_mode = self.phase_mode_map.get(self.phase_mode_var.get(), "propagation")
            s.optimization.geometric_conversion_efficiency = float(self.controls["geometric_efficiency"].get())
            s.target.fwhm_lambda = float(self.controls["target_fwhm"].get())
            s.target.sidelobe_percent = float(self.controls["target_sr"].get())
            s.lithography.enabled = self.litho_enabled.get()
            s.lithography.image_contrast_target = float(self.controls["target_contrast"].get())
            s.lithography.angular_magnification_target = float(self.controls["angular_mag"].get())
            s.imaging.angular_magnification = s.lithography.angular_magnification_target
            s.lithography.reduction_ratio_target = float(self.controls["reduction"].get())
            s.lithography.fov_half_angle_deg = float(self.controls["fov"].get())
            s.lithography.contrast_weight = float(self.controls["contrast_weight"].get())
            s.lithography.peak_weight = float(self.controls["peak_weight"].get())
            s.lithography.sidelobe_weight = float(self.controls["sidelobe_weight"].get())
            s.lithography.fwhm_weight = float(self.controls["fwhm_weight"].get())
            s.sampling.gpu_memory_fraction = float(self.controls["gpu_memory_fraction"].get())
            s.sampling.gpu_batch_size = int(float(self.controls["gpu_batch_size"].get()))
            s.sampling.manual_thread_mode = bool(getattr(self, "manual_thread_var", tk.BooleanVar(value=False)).get())
            s.sampling.thread_count = int(float(self.controls["thread_count"].get()))
            s.imaging.e_width_um = float(self.controls["e_width"].get())
            s.imaging.e_height_um = float(self.controls["e_height"].get())
            s.imaging.e_line_width_um = float(self.controls["e_line"].get())
            s.imaging.e_middle_arm_ratio = float(self.controls["e_middle_ratio"].get())
            s.imaging.e_point_spacing_um = float(self.controls["e_point"].get())
            s.imaging.objective_distance_um = float(self.controls["e_object_distance"].get())
            s.imaging.e_bar_spacing_um = max(0.0, (s.imaging.e_height_um - s.imaging.e_line_width_um) / 2.0)
            if s.lithography.enabled:
                s.imaging.focal_length_um, _magnification = optics.projection_image_geometry(s)
            image_range = abs(float(self.controls["e_range"].get()))
            s.imaging.xs_min_um, s.imaging.xs_max_um = -image_range / 2, image_range / 2
            s.imaging.grid_n = int(float(self.controls["e_grid"].get()))
            s.output_path = self.data_path.get() if hasattr(self, "data_path") else str(self._data_dir())
            errors = s.validate()
            if errors: raise ValueError("\n".join(errors[:8]))
            thread_count, _thread_mode = optics.apply_thread_policy(s)
            if hasattr(self, "thread_policy_label"):
                self.thread_policy_label.configure(text=self._thread_policy_text())
            if hasattr(self, "gpu_metric_speed") and getattr(self, "current_page", "") == "gpu":
                self.gpu_status_detail.configure(text=(f"CPU threads: {thread_count}" if self.language == "en" else f"CPU 线程数：{thread_count}"))
            return True
        except (ValueError, TypeError) as exc:
            messagebox.showerror(self._localized_app_name(), str(exc))
            return False

    def _refresh_preview(self):
        if not self._sync(): return
        try:
            result = optics.preview(self.state_model)
            self.last_preview, self.last_phase = result, result.phase_rad
            self.phase_library["current"] = np.asarray(result.phase_rad)
            self._save_phase_auto("current", result.phase_rad)
            self.phase_heatmap.set_data(result.phase_rad, self.t("phase"))
            if self.state_model.lithography.enabled:
                e_image, image_metrics = optics.letter_e_image_metrics(self.state_model, result.intensity, grid_n=self.state_model.imaging.grid_n)
                contrast = image_metrics["image_contrast"]
                self.last_design_display = e_image
                self.last_image = e_image
                self.last_lithography_image = e_image
                self.last_lithography_metrics = image_metrics
                title = self.t("aerial_preview")
            else:
                contrast = optics.estimate_image_contrast(result.intensity)
                self.last_design_display = result.intensity
                self.last_lithography_image = None
                self.last_lithography_metrics = {}
                title = self.t("focal_preview")
            self.design_preview_card.title_label.configure(text=title)
            self.preview_heatmap.set_data(self.last_design_display, title)
            if self.state_model.lithography.enabled:
                self.preview_heatmap.focus_signal()
            self.last_display_contrast = contrast
            self.metric_plot.set_series({"FWHM (λ)": [result.fwhm_lambda], "SR (%)": [result.sidelobe_percent], "Peak": [result.peak_intensity], self._contrast_label(): [contrast]})
            self._set_status(self.t("ready"))
        except Exception as exc:
            messagebox.showerror(self._localized_app_name(), str(exc))

    def _start_optimization(self):
        if self.worker and self.worker.is_alive(): return
        if not self._sync(): return
        iterations = max(8, int(float(self.controls["iterations"].get())))
        particles = max(8, int(float(self.controls["particles"].get())))
        seed = int(float(self.controls["seed"].get()))
        live_every = max(1, int(float(self.controls["live_every"].get())))
        state = copy.deepcopy(self.state_model)
        self.history.clear(); self.stop_event.clear()
        self.optimization_active = True
        self.start_button.configure(state="disabled"); self.stop_button.configure(state="normal")
        self._set_status(self.t("running"), kind="busy", progress=0); self.optimization_text.set(self.t("running")); self.optimization_progress["value"] = 0

        def progress(row, curve):
            phase = row.pop("phase_profile", None) if int(row["iteration"]) % live_every == 0 else None
            design = [row.get(f"design_{i}", .5) for i in range(8)] if phase is not None else None
            self.queue.put(("progress", row, curve, phase, design, iterations))

        def run():
            try:
                result = run_hybrid_optimizer(state, np.empty((0, 8)), iterations, particles, seed, progress, self.stop_event.is_set)
                self.queue.put(("done", result, state))
            except Exception as exc:
                self.queue.put(("error", str(exc)))
        self.worker = threading.Thread(target=run, name="MetaLens-v3-Optimizer", daemon=False)
        self.worker.start(); self.after(80, self._poll)

    def _poll(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg[0] == "progress":
                    _, row, curve, phase, design, total = msg
                    self.history.append({k: float(v) for k, v in row.items() if isinstance(v, (int, float))})
                    pct = int(100 * row["iteration"] / total); self.optimization_progress["value"] = pct
                    self._set_status(f"{self.t('running')}: {int(row['iteration'])}/{total}", kind="busy", progress=pct)
                    self.optimization_text.set(f"{self.t('running')}: {int(row['iteration'])}/{total}, score={row['best_score']:.6g}")
                    self.metric_plot.set_series({"FWHM (λ)": [r["fwhm_lambda"] for r in self.history], "SR (%)": [r["sidelobe_percent"] for r in self.history], "Peak": [r["peak_intensity"] for r in self.history], self._contrast_label(): [r["image_contrast"] for r in self.history]})
                    self.convergence_plot.set_series({"Best score": list(curve), "Improvement / step": list(np.maximum(0, -np.diff(curve, prepend=curve[0])))})
                    if phase is not None:
                        live_state = state_from_design_vector(self.state_model, np.asarray(design, dtype=np.float64))
                        live_preview = optics.preview_from_phase(live_state, np.asarray(phase))
                        self.last_phase = live_preview.phase_rad; self.last_preview = live_preview
                        self.phase_heatmap.set_data(self.last_phase, self.t("phase"))
                        if live_state.lithography.enabled:
                            live_display, _live_metrics = optics.letter_e_image_metrics(live_state, live_preview.intensity, grid_n=live_state.imaging.grid_n)
                            live_title = self.t("aerial_preview")
                        else:
                            live_display, live_title = live_preview.intensity, self.t("focal_preview")
                        self.last_design_display = live_display
                        self.design_preview_card.title_label.configure(text=live_title)
                        self.preview_heatmap.set_data(live_display, live_title)
                elif msg[0] == "done":
                    _, result, state = msg
                    self.state_model = state_from_design_vector(state, np.asarray(result.best_vector))
                    self.phase_library["optimized"] = np.asarray(result.best_phase_profile, dtype=np.float64)
                    self._save_phase_auto("optimized", self.phase_library["optimized"])
                    self._finish_worker(self.t("completed")); self._refresh_preview(); self._save_run(result)
                    self._show_optimization_result(result)
                    return
                elif msg[0] == "error":
                    self._finish_worker(msg[1]); messagebox.showerror(self._localized_app_name(), msg[1]); return
        except queue.Empty:
            pass
        if self.optimization_active: self.after(80, self._poll)

    def _finish_worker(self, text: str):
        self.optimization_active = False
        self.start_button.configure(state="normal"); self.stop_button.configure(state="disabled")
        self.optimization_progress["value"] = 100; self.optimization_text.set(text); self._set_status(text, progress=100)

    def _stop_optimization(self):
        self.stop_event.set(); self.optimization_text.set(self.t("stopping")); self._set_status(self.t("stopping"), kind="warning"); self.stop_button.configure(state="disabled")

    def _start_ml_design(self):
        if not self._sync(): return
        try:
            self.state_model.lens.wavelength_um = float(self.controls["ml_wavelength"].get())
            self.state_model.lens.working_distance_lambda = float(self.controls["ml_focal"].get())
            self.state_model.lens.lens_radius_lambda = float(self.controls["ml_radius"].get())
            self.state_model.target.fwhm_lambda = float(self.controls["ml_fwhm"].get())
            self.state_model.target.sidelobe_percent = float(self.controls["ml_sidelobe"].get())
            self.state_model.lithography.image_contrast_target = float(self.controls["ml_contrast"].get())
            ensembles = max(3, int(float(self.controls["ml_ensembles"].get())))
            alpha = max(0.0, float(self.controls["ml_alpha"].get()))
        except ValueError as exc:
            messagebox.showerror(self._localized_app_name(), str(exc)); return
        self.ml_progress.start(30); self._set_status("Training historical ML model...", kind="busy")
        state, root = copy.deepcopy(self.state_model), Path(self.history_path.get())
        def run():
            try: self.queue.put(("ml_done", train_and_design(root, state, alpha=alpha, ensembles=ensembles), state))
            except Exception as exc: self.queue.put(("ml_error", str(exc)))
        threading.Thread(target=run, name="MetaLens-v3-ML", daemon=True).start(); self.after(100, self._poll_ml)

    def _poll_ml(self):
        try:
            msg = self.queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_ml); return
        self.ml_progress.stop()
        if msg[0] == "ml_error":
            self.status_var.set(self.t("ready")); messagebox.showerror(self._localized_app_name(), msg[1]); return
        if msg[0] != "ml_done":
            self.queue.put(msg); self.after(100, self._poll_ml); return
        _, result, base = msg
        self.state_model = state_from_design_vector(base, result.vector)
        preview = optics.preview(self.state_model); self.last_preview, self.last_phase = preview, preview.phase_rad
        self.phase_library["ml"] = np.asarray(preview.phase_rad)
        self._save_phase_auto("ml", preview.phase_rad)
        self.last_ml_result = result
        self.ml_confidence_value.configure(text=f"{result.confidence:.1%}")
        self.ml_samples_value.configure(text=str(result.summary.samples))
        self.ml_validation_value.configure(text=f"{result.summary.validation_rmse:.4g}")
        self.ml_confidence_bar["value"] = result.confidence * 100
        self.ml_summary_var.set((f"Local neighbors: {result.neighbors}. The predicted design is now the active project." if self.language == "en" else f"局部相似样本：{result.neighbors}。机器学习预测结果已成为当前设计。"))
        self.ml_phase_heatmap.set_data(preview.phase_rad, self.t("phase"))
        self.ml_heatmap.set_data(preview.intensity, "ML-designed focal intensity")
        self._show_ml_result(result, preview)
        self._set_status("Historical ML design complete")

    def _show_ml_result(self, result, preview):
        popup = tk.Toplevel(self); popup.title(self.t("prediction")); popup.geometry("960x660"); popup.minsize(720, 520)
        body = ttk.Frame(popup, padding=14); body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1); body.columnconfigure(1, weight=1); body.rowconfigure(1, weight=1)
        ttk.Label(body, style="Card.TLabel", text=f"Samples: {result.summary.samples} | Confidence: {result.confidence:.1%} | Validation RMSE: {result.summary.validation_rmse:.4g}").grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        phase_card = self._card(body, self.t("phase")); phase_card.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        phase_view = Heatmap(phase_card); phase_view.pack(fill="both", expand=True); phase_view.set_data(preview.phase_rad, self.t("phase"))
        focal_card = self._card(body, self.t("preview")); focal_card.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        focal_view = Heatmap(focal_card); focal_view.pack(fill="both", expand=True); focal_view.set_data(preview.intensity, self.t("preview"))
        actions = ttk.Frame(body); actions.grid(row=2, column=0, columnspan=2, sticky="e", pady=(10, 0))
        self._button(actions, "export_data", lambda: self._export_matrix(preview.phase_rad, "ml_phase")).pack(side="left", padx=5)
        self._button(actions, "export_image", lambda: self._export_picture(preview.intensity, "ml_focal")).pack(side="left", padx=5)
        self._button(actions, "close", popup.destroy, width=100).pack(side="left")

    def _save_phase_auto(self, source: str, phase: np.ndarray):
        folder = self._data_dir() / "PhaseLibrary"
        folder.mkdir(parents=True, exist_ok=True)
        export_array(folder / f"{source}_phase_latest.txt", np.asarray(phase))
        export_array(folder / f"{source}_phase_latest.csv", np.asarray(phase))

    def _show_optimization_result(self, result):
        if self.last_preview is None:
            return
        popup = tk.Toplevel(self)
        popup.title(self.t("result_title"))
        popup.geometry("1040x720")
        popup.minsize(760, 560)
        popup.configure(bg=BG)
        try: popup.iconbitmap(default=str(asset_path("app_icon.ico")))
        except tk.TclError: pass
        shell = ttk.Frame(popup, padding=14); shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1); shell.columnconfigure(1, weight=1); shell.rowconfigure(1, weight=1)
        metrics = ttk.Frame(shell, style="Card.TFrame", padding=10); metrics.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        latest = result.metrics[-1] if result.metrics else {}
        final_litho = dict(getattr(self, "last_lithography_metrics", {}) or {})
        final_display_contrast = final_litho.get("image_contrast", getattr(self, "last_display_contrast", optics.estimate_image_contrast(self.last_preview.intensity)))
        values = [
            ("FWHM (λ)", latest.get("fwhm_lambda", self.last_preview.fwhm_lambda)),
            ("SR (%)", latest.get("sidelobe_percent", self.last_preview.sidelobe_percent)),
            ("Peak", latest.get("peak_intensity", self.last_preview.peak_intensity)),
            (self._contrast_label(), final_display_contrast),
            ("Best Score", result.best_score),
        ]
        if self.state_model.lithography.enabled:
            values.extend([
                (("Michelson 对比度" if self.language == "zh" else "Michelson Contrast"), latest.get("e_michelson_contrast", 0.0)),
                (("稳健对比度" if self.language == "zh" else "Robust Contrast"), latest.get("e_robust_contrast", 0.0)),
                (("轮廓重合度" if self.language == "zh" else "Shape Dice"), latest.get("e_shape_dice", 0.0)),
                (("可解析度" if self.language == "zh" else "Resolvability"), latest.get("e_resolvability", 0.0)),
                (("E 宽度（像素）" if self.language == "zh" else "E Width (px)"), latest.get("e_bbox_width_px", 0.0)),
                (("笔画宽度（像素）" if self.language == "zh" else "Stroke Width (px)"), latest.get("e_stroke_width_px", 0.0)),
                (("自动像面距离 (µm)" if self.language == "zh" else "Auto Image Distance (µm)"), latest.get("image_plane_distance_um", self.state_model.imaging.focal_length_um)),
                (("投影倍率" if self.language == "zh" else "Projection Magnification"), latest.get("projection_magnification", 0.0)),
            ])
        for index, (name, value) in enumerate(values):
            tile = tk.Frame(metrics, bg="#F1F5F9", padx=10, pady=8)
            row, column = divmod(index, 4)
            tile.grid(row=row, column=column, sticky="ew", padx=4, pady=3); metrics.columnconfigure(column, weight=1)
            tk.Label(tile, text=name, bg="#F1F5F9", fg=MUTED, font=("Segoe UI", 8)).pack(anchor="w")
            tk.Label(tile, text=f"{float(value):.5g}", bg="#F1F5F9", fg=TEXT, font=("Segoe UI", 14, "bold")).pack(anchor="w")
        phase_card = self._card(shell, self.t("phase")); phase_card.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        phase_view = Heatmap(phase_card); phase_view.pack(fill="both", expand=True); phase_view.set_data(self.last_phase, self.t("phase"))
        display_image = self.last_preview.intensity
        display_title = self.t("preview")
        if self.state_model.lithography.enabled:
            display_image = self.last_lithography_image if self.last_lithography_image is not None else self.last_design_display
            if display_image is None:
                display_image, self.last_lithography_metrics = optics.letter_e_image_metrics(self.state_model, self.last_preview.intensity, grid_n=self.state_model.imaging.grid_n)
                self.last_lithography_image = display_image
            display_title = self.t("imaging")
        intensity_card = self._card(shell, display_title); intensity_card.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        intensity_view = Heatmap(intensity_card); intensity_view.pack(fill="both", expand=True); intensity_view.set_data(display_image, display_title)
        if self.state_model.lithography.enabled:
            intensity_view.focus_signal()
        if self.state_model.lithography.enabled and latest.get("e_resolvability", 1.0) < 0.65:
            warning = ("当前字母 E 在像面采样中尺寸过小；即使局部对比度较高，也不能视为可辨识成像。请缩小成像范围、增大角放大率或提高采样网格。"
                       if self.language == "zh" else
                       "The Letter E is undersampled in the image plane. High local contrast does not imply readable imaging; reduce image range, increase angular magnification, or raise the grid size.")
            tk.Label(intensity_card, text=warning, bg="#FFF4CE", fg="#7A4F01", font=("Segoe UI", 9), wraplength=430, justify="left", padx=10, pady=7).pack(fill="x", pady=(6, 0))
        curve = MetricPlot(shell, height=150); curve.grid(row=2, column=0, columnspan=2, sticky="ew", pady=10)
        curve.set_series({"Best score": list(result.curve), "Improvement": list(np.maximum(0, -np.diff(result.curve, prepend=result.curve[0])))})
        actions = ttk.Frame(shell); actions.grid(row=3, column=0, columnspan=2, sticky="e")
        self._button(actions, "export_result", lambda: self._export_optimization_result(result), accent=True, width=190).pack(side="left", padx=5)
        self._button(actions, "close", popup.destroy, width=100).pack(side="left")

    def _export_optimization_result(self, result):
        folder = filedialog.askdirectory(parent=self)
        if not folder: return
        target = Path(folder) / f"Optimization_Result_{time.strftime('%Y%m%d_%H%M%S')}"; target.mkdir(parents=True, exist_ok=True)
        latest = result.metrics[-1] if result.metrics else {}
        export_array(target / "Final_Phase.txt", self.last_phase)
        export_array(target / "Final_Phase.csv", self.last_phase)
        export_image(target / "Final_Phase.png", self.last_phase)
        export_array(target / "Final_Focal_PSF.csv", self.last_preview.intensity)
        export_image(target / "Final_Focal_PSF.png", self.last_preview.intensity)
        if self.state_model.lithography.enabled:
            letter_e_image = self.last_lithography_image
            letter_e_metrics = dict(getattr(self, "last_lithography_metrics", {}) or {})
            if letter_e_image is None:
                letter_e_image, letter_e_metrics = optics.letter_e_image_metrics(self.state_model, self.last_preview.intensity, grid_n=self.state_model.imaging.grid_n)
            export_array(target / "Final_Letter_E_Aerial_Image.txt", letter_e_image)
            export_array(target / "Final_Letter_E_Aerial_Image.csv", letter_e_image)
            export_image(target / "Final_Letter_E_Aerial_Image.png", letter_e_image)
            export_records(target / "Final_Letter_E_ImagePlane_Metrics.csv", [letter_e_metrics])
        export_records(target / "Final_Metrics.csv", [latest | {"best_score": result.best_score}])
        export_records(target / "Optimization_History.csv", result.metrics)
        phase_img = Image.open(target / "Final_Phase.png").resize((420, 420))
        intensity_img = Image.open(target / "Final_Focal_PSF.png").resize((420, 420))
        report = Image.new("RGB", (900, 540), "white"); report.paste(phase_img, (20, 80)); report.paste(intensity_img, (460, 80))
        draw = ImageDraw.Draw(report); draw.text((20, 18), f"MetaLens Optimization Result v{APP_VERSION}", fill="#1B1B1F")
        draw.text((20, 48), " | ".join(f"{key}={float(value):.5g}" for key, value in (latest | {"best_score": result.best_score}).items() if isinstance(value, (int, float))), fill="#44505E")
        report.save(target / "Optimization_Summary.png")
        self._set_status((f"Result exported to {target}" if self.language == "en" else f"优化结果已导出到 {target}"), progress=100)

    def _detect_gpu(self):
        previous = {label for label, var in self.gpu_vars.items() if var.get()}
        if not previous:
            previous = {item for item in self.state_model.sampling.selected_gpu_devices.split("||") if item}
        devices, notes = detect_devices(force_refresh=True)
        self.detected_gpu_devices = devices
        for child in self.gpu_device_frame.winfo_children(): child.destroy()
        self.gpu_vars.clear()
        self.gpu_checks.clear()
        self.gpu_display_labels.clear()
        if not devices:
            ttk.Label(self.gpu_device_frame, text=self.t("no_device"), style="Card.TLabel", wraplength=260).pack(anchor="w", pady=18)
        for index, device in enumerate(devices):
            card = tk.Frame(self.gpu_device_frame, bg="#F1F5F9", padx=12, pady=10, highlightbackground=BORDER, highlightthickness=1)
            card.pack(fill="x", pady=5)
            label = device.label()
            key = f"{device.backend.value}#{index}|{label}"
            var = tk.BooleanVar(value=(key in previous or label in previous) if previous else index == 0)
            try:
                var.trace_add("write", lambda *_args: self._update_gpu_selection_count())
            except AttributeError:
                var.trace("w", lambda *_args: self._update_gpu_selection_count())
            self.gpu_vars[key] = var
            check = ToggleCheck(card, device.name, var)
            self.gpu_checks[key] = check
            self.gpu_display_labels[key] = label
            check.pack(anchor="w")
            detail = f"{device.backend.value}  |  {device.vendor or 'Unknown vendor'}  |  {device.memory_mb or '?'} MB  |  {device.platform}"
            tk.Label(card, text=detail, bg="#F1F5F9", fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w", padx=(31, 0), pady=(4, 0))
        for child in self.gpu_notes_frame.winfo_children(): child.destroy()
        translations = {
            "CuPy is not installed or cannot load CUDA: No module named 'cupy'": "未安装 CuPy，或 CuPy 无法加载 CUDA。",
            "NVIDIA driver is visible, but CuPy is unavailable; CUDA numerical kernels are not active in this build.": "已检测到 NVIDIA 驱动，但 CuPy 不可用；当前版本不会启用 CUDA 数值内核。",
            "No usable CUDA/OpenCL GPU backend was found. Install NVIDIA CUDA+CuPy or an OpenCL runtime+PyOpenCL to enable GPU kernels.": "未发现可用的 CUDA/OpenCL 计算后端。安装 CUDA+CuPy 或 OpenCL+PyOpenCL 后可启用 GPU 内核。",
        }
        for note in notes[:6]:
            shown = note if self.language == "en" else translations.get(note, note)
            ttk.Label(self.gpu_notes_frame, text="• " + shown, style="Card.TLabel", wraplength=260).pack(anchor="w", pady=2)
        self.gpu_status_title.configure(text=f"{len(devices)} GPU(s) detected" if self.language == "en" else f"检测到 {len(devices)} 个 GPU")
        self.gpu_status_detail.configure(text=self.t("no_device") if not devices else ("Select one or more devices, then apply the selection." if self.language == "en" else "可选择一个或多个设备，然后应用选择。"))
        self.gpu_metric_devices.configure(text=str(sum(var.get() for var in self.gpu_vars.values())))

    def _update_gpu_selection_count(self):
        if hasattr(self, "gpu_metric_devices"):
            self.gpu_metric_devices.configure(text=str(sum(var.get() for var in self.gpu_vars.values())))

    def _set_all_gpus(self, selected: bool):
        for label, var in self.gpu_vars.items():
            var.set(selected)
            if label in self.gpu_checks:
                self.gpu_checks[label].refresh()
        self._update_gpu_selection_count()
        self.gpu_device_frame.update_idletasks()

    def _apply_gpu(self):
        self.state_model.sampling.backend = int(self.backend_var.get().split("-", 1)[0])
        selected = [label for label, var in self.gpu_vars.items() if var.get()]
        self.state_model.sampling.selected_gpu_devices = "||".join(selected)
        self.state_model.sampling.multi_gpu_n = len(selected)
        self.gpu_metric_backend.configure(text=self.backend_var.get().split("-", 1)[1])
        self.gpu_metric_devices.configure(text=str(len(selected)))
        self.gpu_status_title.configure(text=self.t("selected_devices"))
        self.gpu_status_detail.configure(text="\n".join(selected) if selected else ("CPU fallback" if self.language == "en" else "使用 CPU 稳定后备路径"))
        display = [self.gpu_display_labels.get(label, label.split("|", 1)[-1]) for label in selected]
        if display:
            self.gpu_status_detail.configure(text="\n".join(display))
        self._set_status(self.t("ready"))

    def _benchmark_gpu(self):
        self._apply_gpu(); self.gpu_progress["value"] = 10
        state = copy.deepcopy(self.state_model); state.sampling.preview_n = 512
        start = time.perf_counter()
        try:
            for index in range(3): optics.preview(state); self.gpu_progress["value"] = 30 + index * 30; self.update_idletasks()
            elapsed = time.perf_counter() - start
            self.gpu_metric_speed.configure(text=f"{elapsed / 3:.3f}")
            self.gpu_status_title.configure(text=self.t("benchmark_result"))
            accelerator = Accelerator(state.sampling.backend, getattr(state.sampling, "selected_gpu_devices", ""), state.sampling.gpu_memory_fraction)
            detail = (f"3 × 512² previews completed in {elapsed:.3f} seconds. Active backend: {self.gpu_metric_backend.cget('text')}."
                      if self.language == "en" else
                      f"3 次 512² 预览在 {elapsed:.3f} 秒内完成。当前后端：{self.gpu_metric_backend.cget('text')}。")
            detail += "\n" + accelerator.memory_summary()
            if accelerator.status.notes:
                detail += "\n" + "\n".join(accelerator.status.notes[-3:])
            self.gpu_status_detail.configure(text=detail)
        except Exception as exc:
            self.gpu_status_title.configure(text="Benchmark failed" if self.language == "en" else "基准测试失败")
            self.gpu_status_detail.configure(text=str(exc))
        self.gpu_progress["value"] = 100

    def _data_dir(self) -> Path:
        return Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "Open.MetaLensWorkbench" / "Data"

    def _browse_history(self):
        value = filedialog.askdirectory();
        if value: self.history_path.set(value)

    def _browse_data(self):
        value = filedialog.askdirectory();
        if value: self.data_path.set(value)

    def _open_data(self):
        path = Path(self.data_path.get()); path.mkdir(parents=True, exist_ok=True); os.startfile(path)

    def _save_project(self):
        if not self._sync(): return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("MetaLens Project", "*.json")])
        if path: optics.save_project_json(self.state_model, Path(path))

    def _load_project(self):
        path = filedialog.askopenfilename(filetypes=[("MetaLens Project", "*.json")])
        if path:
            self.state_model = optics.load_project_json(Path(path)); self._reflect_state(); self._refresh_preview()

    def _reflect_state(self):
        values = {"wavelength": self.state_model.lens.wavelength_um, "focal": self.state_model.lens.working_distance_lambda, "radius": self.state_model.lens.lens_radius_lambda, "pitch": self.state_model.lens.pitch_um, "preview_n": self.state_model.sampling.preview_n, "phase_levels": self.state_model.optimization.phase_n, "target_fwhm": self.state_model.target.fwhm_lambda, "target_sr": self.state_model.target.sidelobe_percent, "linear_angle": self.state_model.source.theta_polar_angle_deg, "waist_w0": self.state_model.source.waist_w0_lambda, "led_fwhm": self.state_model.source.led_fwhm_nm, "wavelength_samples": self.state_model.source.wavelength_samples, "led_divergence": self.state_model.source.led_divergence_half_angle_deg, "angle_samples": self.state_model.source.angle_samples, "e_width": self.state_model.imaging.e_width_um, "e_height": self.state_model.imaging.e_height_um, "e_line": self.state_model.imaging.e_line_width_um, "e_middle_ratio": self.state_model.imaging.e_middle_arm_ratio, "e_point": self.state_model.imaging.e_point_spacing_um, "e_object_distance": self.state_model.imaging.objective_distance_um, "e_range": abs(self.state_model.imaging.xs_max_um - self.state_model.imaging.xs_min_um), "e_grid": self.state_model.imaging.grid_n}
        for key, value in values.items(): self.controls[key].set(str(value))
        self.source_mode_key = "led" if self.state_model.source.light_source_mode == 1 else "laser"
        self.polarization_mode_key = {0: "te", 1: "tm", 2: "unpolarized", 3: "linear", 4: "lcp", 5: "rcp"}.get(self.state_model.source.polarization_mode, "te")
        self.beam_shape_key = "gaussian" if self.state_model.source.beam_shape == 1 else "plane"
        self._update_source_values()

    def _export_matrix(self, data, stem):
        if data is None: return
        path = filedialog.asksaveasfilename(initialfile=stem + ".csv", defaultextension=".csv", filetypes=[("CSV", "*.csv"), ("Text", "*.txt")])
        if path: export_array(Path(path), np.asarray(data))

    def _export_picture(self, data, stem):
        if data is None: return
        path = filedialog.asksaveasfilename(initialfile=stem + ".png", defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Bitmap", "*.bmp"), ("TIFF", "*.tiff")])
        if path: export_image(Path(path), np.asarray(data))

    def _save_run(self, result):
        base = Path(self.data_path.get()) if hasattr(self, "data_path") and self.data_path.get().strip() else self._data_dir()
        folder = base / time.strftime("Run_%Y%m%d_%H%M%S"); folder.mkdir(parents=True, exist_ok=True)
        optics.save_project_json(self.state_model, folder / "Optimized_Project.json")
        export_manifest(folder / "Optimization_Process.json", {"version": APP_VERSION, "best_vector": result.best_vector, "best_score": result.best_score, "latest_metrics": result.metrics[-1] if result.metrics else {}, "method": result.method})
        export_records(folder / "Optimization_Metrics.csv", result.metrics)
        if self.last_preview:
            export_array(folder / "Focal_PSF.txt", self.last_preview.intensity); export_image(folder / "Focal_PSF.png", self.last_preview.intensity)
            if self.state_model.lithography.enabled:
                aerial, aerial_metrics = optics.letter_e_image_metrics(self.state_model, self.last_preview.intensity, grid_n=self.state_model.imaging.grid_n)
                export_array(folder / "Letter_E_Aerial_Image.txt", aerial); export_image(folder / "Letter_E_Aerial_Image.png", aerial)
                export_records(folder / "Letter_E_ImagePlane_Metrics.csv", [aerial_metrics])
        if self.last_phase is not None: export_array(folder / "Phase_Profile.txt", self.last_phase); export_image(folder / "Phase_Profile.png", self.last_phase)

    def _export_all(self):
        if not self._sync(): return
        folder = filedialog.askdirectory()
        if not folder: return
        target = Path(folder) / f"MetaLens_Project_{time.strftime('%Y%m%d_%H%M%S')}"; target.mkdir(parents=True, exist_ok=True)
        exported = []
        selected = lambda key: not hasattr(self, "export_vars") or self.export_vars[key].get()
        if selected("export_project"):
            exported.extend(optics.export_design_bundle(self.state_model, target / "Project"))
            exported.append(optics.save_project_json(self.state_model, target / "Project" / "metalens_workbench_project.json"))
        if selected("export_opt") and self.history:
            exported.append(export_records(target / "Optimization" / "Optimization_Metrics.csv", self.history))
        if selected("export_phase") and self.last_phase is not None:
            exported.append(export_array(target / "Phase" / "Phase_Profile.txt", self.last_phase))
            exported.append(export_array(target / "Phase" / "Phase_Profile.csv", self.last_phase))
            exported.append(export_image(target / "Phase" / "Phase_Profile.png", self.last_phase))
            exported.append(export_image(target / "Phase" / "Phase_Profile.jpg", self.last_phase))
        if selected("export_focal") and self.last_preview:
            exported.append(export_array(target / "Imaging" / "Focal_PSF.csv", self.last_preview.intensity))
            exported.append(export_image(target / "Imaging" / "Focal_PSF.png", self.last_preview.intensity))
            if self.last_image is not None:
                exported.append(export_array(target / "Imaging" / "Letter_E_Aerial_Image.csv", self.last_image))
                exported.append(export_image(target / "Imaging" / "Letter_E_Aerial_Image.png", self.last_image))
        if selected("export_ml") and hasattr(self, "last_ml_result"):
            result = self.last_ml_result
            exported.append(export_manifest(target / "MachineLearning" / "ML_Report.json", {"confidence": result.confidence, "neighbors": result.neighbors, "summary": result.summary.__dict__, "vector": result.vector.tolist(), "uncertainty": result.uncertainty.tolist()}))
        message = (f"Exported {len(exported)} files to {target}" if self.language == "en" else f"已导出 {len(exported)} 个文件到 {target}")
        self.export_status_var.set(message)
        self._set_status(message, progress=100)

    def _close(self):
        if self.worker and self.worker.is_alive():
            self.stop_event.set(); self.status_var.set(self.t("stopping")); self.after(100, self._wait_close)
        else: self.destroy()

    def _wait_close(self):
        if self.worker and self.worker.is_alive(): self.after(100, self._wait_close)
        else: self.destroy()


def main():
    ProApp().mainloop()


if __name__ == "__main__":
    main()
