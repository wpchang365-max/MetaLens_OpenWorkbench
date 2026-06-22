from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import numpy as np
from PIL import Image

from . import app_v3
from .app_v3 import ACCENT, BG, CARD, MUTED, TEXT, Heatmap
from .brand import APP_VERSION
from .engineering_v5 import BatchJob, CheckpointBatch
from .intelligence_v5 import HistoricalDesignLab, generate_html_report
from .physics_v3 import calculate_metrics, debye_high_na_psf
from .research_v5 import (MetaAtomDatabase, ProjectDatabase, ToleranceConfig,
                          align_experiment, export_dxf_rectangles,
                          export_gds_rectangles, fourier_adjoint_optimize,
                          jones_to_mueller, lithography_metrics,
                          monte_carlo_tolerance)
from .win11_widgets import FluentComboBox, FluentLanguageSwitch, FluentProgress, FluentScrollFrame


V5_ZH = {
    "research": "科研设计套件", "meta_db": "超原子数据库", "browse_db": "导入数据库",
    "export_gds": "导出 GDSII", "export_dxf": "导出 DXF", "inverse_design": "逆向设计",
    "run_adjoint": "运行伴随设计", "vector_source": "矢量光源与多目标设计", "show_mueller": "显示 Mueller 矩阵",
    "robustness": "制造鲁棒性", "run_tolerance": "运行容差分析", "litho_metrology": "光刻计量",
    "calculate_litho": "计算 CD、NILS 与曝光窗口", "experiment_calibration": "实验标定",
    "import_experiment": "导入 PSF 或字母 E 图像", "research_engineering": "科研工程",
    "save_version": "保存项目版本", "generate_report": "生成科研报告", "advanced_history": "高级历史智能",
    "audit_train": "审计并训练", "predict_design": "预测设计", "active_learning": "推荐补充样本",
    "dataset_quality": "数据集质量与模型可信度", "central_archive": "集中归档与可复现实验",
    "save_snapshot": "保存版本快照", "report_center": "生成完整 HTML 报告", "checksums": "生成校验清单",
    "audit_samples": "有效样本", "audit_duplicates": "重复记录", "audit_outliers": "异常记录", "audit_rmse": "验证误差",
}
V5_EN = {
    "research": "Research Design Suite", "meta_db": "Meta-Atom Database", "browse_db": "Import Database",
    "export_gds": "Export GDSII", "export_dxf": "Export DXF", "inverse_design": "Inverse Design",
    "run_adjoint": "Run Adjoint Design", "vector_source": "Vector Source & Multi-Target", "show_mueller": "Show Mueller Matrix",
    "robustness": "Manufacturing Robustness", "run_tolerance": "Run Tolerance Analysis", "litho_metrology": "Lithography Metrology",
    "calculate_litho": "Calculate CD, NILS & Window", "experiment_calibration": "Experiment Calibration",
    "import_experiment": "Import PSF or Letter-E Image", "research_engineering": "Research Engineering",
    "save_version": "Save Project Version", "generate_report": "Generate Research Report", "advanced_history": "Advanced Historical Intelligence",
    "audit_train": "Audit & Train", "predict_design": "Predict Design", "active_learning": "Recommend New Samples",
    "dataset_quality": "Dataset Quality & Model Confidence", "central_archive": "Central Archive & Reproducibility",
    "save_snapshot": "Save Version Snapshot", "report_center": "Generate Full HTML Report", "checksums": "Generate Checksums",
    "audit_samples": "Valid Samples", "audit_duplicates": "Duplicates", "audit_outliers": "Outliers", "audit_rmse": "Validation RMSE",
}
app_v3.ZH.update(V5_ZH); app_v3.EN.update(V5_EN)
app_v3.ZH.update({
    "physics_audit": "物理模型审计",
    "run_physics_audit": "运行模型审计",
    "db_quality": "数据库质量闭环",
    "audit_database": "审计数据库",
    "batch_center": "多 GPU 批量队列",
    "run_batch_plan": "生成队列预演",
    "calibration_advice": "生成校准建议",
})
app_v3.EN.update({
    "physics_audit": "Physics Model Audit",
    "run_physics_audit": "Run Model Audit",
    "db_quality": "Database Quality Loop",
    "audit_database": "Audit Database",
    "batch_center": "Multi-GPU Batch Queue",
    "run_batch_plan": "Generate Queue Preview",
    "calibration_advice": "Generate Calibration Advice",
})


class ProApp(app_v3.ProApp):
    def _layout(self):
        super()._layout()
        self.meta_db = MetaAtomDatabase()
        self.history_lab = HistoricalDesignLab()
        self.robustness_result = None
        self.experiment_result = None
        self.physics_audit_result = {}
        self.database_quality_result = {}
        self.batch_plan_result = {}
        self._build_research_v5()
        self._merge_history_lab_v5()
        button = self._button(self.sidebar, "research", lambda: self._show("research"), width=225, anchor="w")
        button.pack(fill="x", padx=8, pady=2); self.nav_buttons["research"] = button
        self._rebuild_navigation_footer()
        self._enrich_data_center()
        self._update_backend_language()

    def _rebuild_navigation_footer(self):
        self.language_button.pack_forget()
        for key in ("help", "changelog", "about"):
            self.nav_buttons[key].pack_forget()
        self.language_switch = FluentLanguageSwitch(self.sidebar, self.language, self._select_language, width=215)
        self.language_switch.pack(side="bottom", padx=12, pady=(8, 14))
        for key in ("about", "changelog", "help"):
            self.nav_buttons[key].pack(side="bottom", fill="x", padx=8, pady=2)
        self.nav_separator = tk.Frame(self.sidebar, bg="#D7DCE3", height=1)
        self.nav_separator.pack(side="bottom", fill="x", padx=16, pady=(10, 5))

    def _select_language(self, language):
        if language != self.language:
            self._toggle_language()
        self.language_switch.set_language(self.language)

    def _responsive_layout(self, event=None):
        super()._responsive_layout(event)
        if event is not None and event.widget is not self:
            return
        compact = self.winfo_width() < 1080
        if hasattr(self, "language_switch"):
            self.language_switch.configure(width=175 if compact else 215)
        if hasattr(self, "research_left"):
            if compact:
                self.research_left.grid(row=0, column=0, sticky="nsew", padx=0)
                self.research_right.grid(row=1, column=0, sticky="nsew", padx=0, pady=(8, 0))
                self.research_body.columnconfigure(1, weight=0)
            else:
                self.research_left.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=0)
                self.research_right.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=0)
                self.research_body.columnconfigure(1, weight=1)
        if hasattr(self, "history_action_buttons"):
            for button in self.history_action_buttons:
                button.pack_forget()
            for index, button in enumerate(self.history_action_buttons):
                button.pack(side="top" if compact else "left", fill="x" if compact else "none",
                            pady=2 if compact else 0, padx=0 if index == 0 or compact else 5)

    def _toggle_language(self):
        if hasattr(self, "target_mode_combo"):
            old_values = list(self.target_mode_combo.cget("values"))
            if self.target_mode.get() in old_values:
                self.target_mode_index = old_values.index(self.target_mode.get())
        super()._toggle_language()
        if hasattr(self, "language_switch"):
            self.language_switch.set_language(self.language)
        self._localize_v5_values()
        self._update_backend_language()

    def _update_backend_language(self):
        values = (["0-自动", "1-CPU", "2-NVIDIA CUDA", "3-OpenCL", "4-多 GPU"] if self.language == "zh"
                  else ["0-Auto", "1-CPU", "2-NVIDIA CUDA", "3-OpenCL", "4-Multi GPU"])
        try: index = int(self.backend_var.get().split("-", 1)[0])
        except (ValueError, IndexError): index = 0
        self.backend_combo.configure(values=values); self.backend_var.set(values[index])

    def _v5_local(self, widget, zh, en, option="text"):
        widget._v5_localized = (zh, en, option)
        self.v5_localized = getattr(self, "v5_localized", []) + [widget]
        try: widget.configure(**{option: zh if self.language == "zh" else en})
        except tk.TclError: pass
        return widget

    def _localize_v5_values(self):
        for widget in getattr(self, "v5_localized", []):
            zh, en, option = widget._v5_localized
            try: widget.configure(**{option: zh if self.language == "zh" else en})
            except tk.TclError: pass
        if hasattr(self, "target_mode_combo"):
            values = ["单焦点", "消色差", "多焦点", "宽视场", "字母 E"] if self.language == "zh" else ["Single focus", "Achromatic", "Multi-focus", "Wide field", "Letter E"]
            index = max(0, getattr(self, "target_mode_index", 0)); self.target_mode_combo.configure(values=values); self.target_mode.set(values[index])

    def _v5_entry(self, parent, label, value, label_en=None):
        row = ttk.Frame(parent, style="Card.TFrame"); row.pack(fill="x", pady=3)
        label_widget = ttk.Label(row, text=label, style="Card.TLabel"); label_widget.pack(side="left")
        if label_en is not None: self._v5_local(label_widget, label, label_en)
        var = tk.StringVar(value=str(value)); ttk.Entry(row, textvariable=var, width=18).pack(side="right")
        return var

    def _build_research_v5(self):
        page = self._page("research")
        scroll = FluentScrollFrame(page, background=BG); scroll.pack(fill="both", expand=True)
        body = scroll.content
        self.research_body = body
        body.columnconfigure(0, weight=1); body.columnconfigure(1, weight=1)
        left = ttk.Frame(body); right = ttk.Frame(body)
        self.research_left, self.research_right = left, right
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6)); right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        physics = self._card(left, self.t("physics_audit")); physics.pack(fill="x", pady=(0, 10))
        self.physics_model_var = tk.StringVar(value="Auto audit")
        self.physics_model_combo = FluentComboBox(physics, self.physics_model_var,
            values=["Auto audit", "Scalar Fourier", "Fresnel", "Angular spectrum", "Debye-Wolf high NA"], width=260)
        self.physics_model_combo.pack(anchor="w")
        self._button(physics, "run_physics_audit", self._run_physics_model_audit, accent=True, width=190).pack(anchor="w", pady=6)
        self.physics_audit_label = ttk.Label(physics, style="Card.TLabel", wraplength=420, justify="left")
        self._v5_local(self.physics_audit_label,
            "审计标量、Fresnel、角谱和 Debye-Wolf 高 NA 模型的适用范围。",
            "Audit scalar, Fresnel, angular-spectrum and Debye-Wolf high-NA model applicability.")
        self.physics_audit_label.pack(fill="x")

        device = self._card(left, self.t("meta_db")); device.pack(fill="x", pady=(0, 10))
        self.db_path_var = self._v5_entry(device, "RCWA/FDTD 数据文件", "", "RCWA/FDTD Data File")
        row = ttk.Frame(device, style="Card.TFrame"); row.pack(fill="x", pady=5)
        self._button(row, "browse_db", self._load_meta_database, width=120).pack(side="left")
        self._button(row, "export_gds", lambda: self._export_layout("gds"), width=130).pack(side="left", padx=6)
        self._button(row, "export_dxf", lambda: self._export_layout("dxf"), width=120).pack(side="left")
        self._button(row, "audit_database", self._audit_meta_database, width=135).pack(side="left", padx=6)
        self.db_status = ttk.Label(device, style="Card.TLabel", foreground=MUTED)
        self._v5_local(self.db_status, "尚未导入全波数据库", "No full-wave database loaded")
        self.db_status.pack(anchor="w", pady=5)
        self.db_quality_label = ttk.Label(device, style="Card.TLabel", wraplength=420, justify="left")
        self._v5_local(self.db_quality_label,
            "数据库审计会检查相位覆盖、效率分布、几何范围和制造约束。",
            "Database audit checks phase coverage, efficiency distribution, geometry ranges and fabrication constraints.")
        self.db_quality_label.pack(fill="x", pady=(3, 0))

        inverse = self._card(left, self.t("inverse_design")); inverse.pack(fill="x", pady=(0, 10))
        self.inverse_mode = tk.StringVar(value="傅里叶伴随" if self.language == "zh" else "Fourier adjoint")
        self.inverse_combo = FluentComboBox(inverse, self.inverse_mode, values=["傅里叶伴随", "历史代理模型", "PSO + 伴随混合"], width=260); self.inverse_combo.pack(anchor="w")
        self._v5_local(self.inverse_combo, ["傅里叶伴随", "历史代理模型", "PSO + 伴随混合"], ["Fourier adjoint", "Historical surrogate", "Hybrid PSO + adjoint"], "values")
        self.adjoint_iterations = self._v5_entry(inverse, "伴随迭代次数", 100, "Adjoint Iterations")
        self.adjoint_lr = self._v5_entry(inverse, "学习率", 0.08, "Learning Rate")
        self.phase_levels_v5 = self._v5_entry(inverse, "制造相位级数", 32, "Fabrication Phase Levels")
        self._button(inverse, "run_adjoint", self._run_adjoint, accent=True, width=190).pack(anchor="w", pady=(8, 2))
        self.adjoint_progress = FluentProgress(inverse); self.adjoint_progress.pack(fill="x", pady=6)

        source = self._card(left, self.t("vector_source")); source.pack(fill="x", pady=(0, 10))
        self.target_mode_index = 0; self.target_mode = tk.StringVar(value="单焦点" if self.language == "zh" else "Single focus")
        target_values = ["单焦点", "消色差", "多焦点", "宽视场", "字母 E"] if self.language == "zh" else ["Single focus", "Achromatic", "Multi-focus", "Wide field", "Letter E"]
        self.target_mode_combo = FluentComboBox(source, self.target_mode, values=target_values, width=260); self.target_mode_combo.pack(anchor="w")
        self.multi_waves = self._v5_entry(source, "波长（μm，逗号分隔）", "0.532,0.633", "Wavelengths (um, comma-separated)")
        self.field_angles = self._v5_entry(source, "视场角（度）", "0,3,5", "Field Angles (deg)")
        self.jones_ex = self._v5_entry(source, "Jones Ex 振幅", 1.0, "Jones Ex Amplitude")
        self.jones_ey = self._v5_entry(source, "Jones Ey 振幅", 0.0, "Jones Ey Amplitude")
        self.jones_phase = self._v5_entry(source, "Ey 相位（度）", 90.0, "Ey Phase (deg)")
        self._button(source, "show_mueller", self._show_mueller, width=180).pack(anchor="w", pady=6)

        robust = self._card(right, self.t("robustness")); robust.pack(fill="x", pady=(0, 10))
        self.robust_samples = self._v5_entry(robust, "蒙特卡洛样本数", 100, "Monte Carlo Samples")
        self.linewidth_sigma = self._v5_entry(robust, "线宽标准差（nm）", 5, "Linewidth Sigma (nm)")
        self.etch_sigma = self._v5_entry(robust, "刻蚀深度标准差（nm）", 10, "Etch-Depth Sigma (nm)")
        self.index_sigma = self._v5_entry(robust, "折射率标准差", 0.005, "Index Sigma")
        self._button(robust, "run_tolerance", self._run_robustness, accent=True, width=190).pack(anchor="w", pady=7)
        self.robust_text = tk.Text(robust, height=7, bg="#F7F9FB", relief="flat", font=("Consolas", 9)); self.robust_text.pack(fill="x")

        litho = self._card(right, self.t("litho_metrology")); litho.pack(fill="x", pady=(0, 10))
        self.resist_threshold = self._v5_entry(litho, "光刻胶阈值", 0.5, "Resist Threshold")
        self.pixel_um = self._v5_entry(litho, "像素尺寸（μm）", 0.05, "Image Pixel Size (um)")
        self._button(litho, "calculate_litho", self._run_lithography_metrics, width=230).pack(anchor="w", pady=6)
        self.litho_result = ttk.Label(litho, style="Card.TLabel", wraplength=420)
        self._v5_local(self.litho_result, "请先生成成像数据，再计算 CD、NILS 和曝光窗口。", "Generate image data before calculating CD, NILS and exposure window.")
        self.litho_result.pack(anchor="w")

        experiment = self._card(right, self.t("experiment_calibration")); experiment.pack(fill="x", pady=(0, 10))
        self._button(experiment, "import_experiment", self._import_experiment, width=190).pack(anchor="w")
        self._button(experiment, "calibration_advice", self._generate_calibration_advice, width=190).pack(anchor="w", pady=(5, 0))
        self.experiment_label = ttk.Label(experiment, style="Card.TLabel", wraplength=420)
        self._v5_local(self.experiment_label, "支持 CSV/TXT/NPY/PNG/JPG；执行 FFT 配准、增益与背景标定。", "Supports CSV/TXT/NPY/PNG/JPG; FFT registration plus gain/background calibration.")
        self.experiment_label.pack(anchor="w", pady=6)

        project = self._card(right, self.t("research_engineering")); project.pack(fill="x")
        note = ttk.Label(project, style="Card.TLabel", wraplength=420); note.pack(fill="x")
        self._button(project, "run_batch_plan", self._generate_batch_plan, width=190).pack(anchor="w", pady=7)
        self.batch_plan_label = ttk.Label(project, style="Card.TLabel", wraplength=420, justify="left")
        self._v5_local(self.batch_plan_label,
            "队列预演会按已选 GPU 生成可断点续算的任务计划。",
            "Queue preview creates a checkpoint-ready task plan across selected GPUs.")
        self.batch_plan_label.pack(fill="x")
        self._v5_local(note, "批处理支持断点续算和多 GPU 分配；版本快照、完整报告与校验清单统一放在“数据与导出”页面。", "Batch jobs support checkpoints and multi-GPU assignment. Version snapshots, full reports and checksums are centralized in Data & Export.")

    def _merge_history_lab_v5(self):
        page = self.pages["ml"]
        left_scroll = next(child for child in page.winfo_children() if isinstance(child, FluentScrollFrame))
        advanced = self._card(left_scroll.content, self.t("advanced_history")); advanced.pack(fill="x", pady=8)
        note = ttk.Label(advanced, style="Card.TLabel", wraplength=300, justify="left"); note.pack(fill="x", pady=(0, 8))
        self._v5_local(note,
            "在原有快速预测之外，执行去重、稳健异常值检测、交叉验证、集成不确定度、相似案例检索和主动学习分析。",
            "Adds deduplication, robust outlier detection, cross-validation, ensemble uncertainty, nearest-case retrieval and active learning to the fast predictor.")
        self.history_field_angle = self._v5_entry(advanced, "目标视场角（度）", 0, "Target Field Angle (deg)")
        self.history_bandwidth = self._v5_entry(advanced, "目标带宽（nm）", 0, "Target Bandwidth (nm)")
        actions = ttk.Frame(advanced, style="Card.TFrame"); actions.pack(fill="x", pady=7)
        b1 = self._button(actions, "audit_train", self._train_history_v5, accent=True, width=130); b1.pack(side="left")
        b2 = self._button(actions, "predict_design", self._predict_history_v5, width=125); b2.pack(side="left", padx=5)
        b3 = self._button(actions, "active_learning", self._active_learning_v5, width=145); b3.pack(side="left")
        self.history_action_buttons = [b1, b2, b3]
        quality = self._card(left_scroll.content, self.t("dataset_quality")); quality.pack(fill="x", pady=(0, 12))
        tiles = ttk.Frame(quality, style="Card.TFrame"); tiles.pack(fill="x")
        self.audit_samples_value = self._metric_tile(tiles, "audit_samples", "0", 0)
        self.audit_duplicates_value = self._metric_tile(tiles, "audit_duplicates", "0", 1)
        self.audit_outliers_value = self._metric_tile(tiles, "audit_outliers", "0", 2)
        self.history_advanced_progress = FluentProgress(quality); self.history_advanced_progress.pack(fill="x", pady=8)
        self.history_advanced_var = tk.StringVar(value=self.t("ready"))
        ttk.Label(quality, textvariable=self.history_advanced_var, style="Card.TLabel", wraplength=300, justify="left").pack(fill="x")

    def _enrich_data_center(self):
        page = self.pages["data"]
        scroll = next(child for child in page.winfo_children() if isinstance(child, FluentScrollFrame))
        card = self._card(scroll.content, self.t("central_archive")); card.pack(fill="x", pady=(0, 15))
        note = ttk.Label(card, style="Card.TLabel", wraplength=680, justify="left"); note.pack(fill="x", pady=(0, 10))
        self._v5_local(note,
            "本页负责跨模块集中交付：项目快照、全部结果包、可复现报告和 SHA-256 校验清单。单幅图像或单个数组仍在其产生页面就近导出。",
            "This page handles cross-module delivery: project snapshots, complete result bundles, reproducible reports and SHA-256 manifests. Individual plots and arrays remain exportable where they are produced.")
        actions = ttk.Frame(card, style="Card.TFrame"); actions.pack(fill="x")
        self._button(actions, "save_snapshot", self._save_project_version, width=170).pack(side="left")
        self._button(actions, "report_center", self._generate_report, width=190).pack(side="left", padx=6)
        self._button(actions, "checksums", self._generate_checksums, width=150).pack(side="left")

    def _generate_checksums(self):
        folder = filedialog.askdirectory()
        if not folder: return
        root = Path(folder); files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
        target = root / "SHA256SUMS.txt"
        with target.open("w", encoding="utf-8") as handle:
            for path in files:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                handle.write(f"{digest}  {path.relative_to(root).as_posix()}\n")
        self.export_status_var.set((f"已生成 {len(files)} 个文件的 SHA-256 校验清单" if self.language == "zh" else f"Generated SHA-256 manifest for {len(files)} files"))

    def _build_history_lab_v5(self):
        page = self._page("history_lab")
        scroll = FluentScrollFrame(page, background=BG); scroll.pack(fill="both", expand=True)
        intro = self._card(scroll.content, "Audited Historical Design / 经审计的历史智能设计"); intro.pack(fill="x", pady=(0, 10))
        ttk.Label(intro, text="Dataset audit, deduplication, robust outlier detection, cross-validation, ensemble uncertainty, nearest-case retrieval and active-learning recommendations.", style="Card.TLabel", wraplength=760).pack(anchor="w")
        row = ttk.Frame(intro, style="Card.TFrame"); row.pack(fill="x", pady=8)
        self.history_path = tk.StringVar()
        ttk.Entry(row, textvariable=self.history_path).pack(side="left", fill="x", expand=True)
        self._button(row, "Browse", self._browse_history, width=100).pack(side="left", padx=6)
        self._button(row, "Audit & Train", self._train_history_v5, accent=True, width=140).pack(side="left")
        target = self._card(scroll.content, "Design Targets & Confidence / 设计目标与可信度"); target.pack(fill="x", pady=(0, 10))
        self.history_targets = {}
        defaults = [("wavelength_um", .6328), ("focal_lambda", 10), ("radius_lambda", 50),
                    ("target_fwhm_lambda", .36), ("max_sidelobe", .25), ("target_contrast", .8),
                    ("field_angle_deg", 0), ("bandwidth_nm", 0)]
        for name, value in defaults: self.history_targets[name] = self._v5_entry(target, name, value)
        self._button(target, "Predict Design", self._predict_history_v5, width=150).pack(anchor="w", pady=7)
        self.history_output = tk.Text(scroll.content, height=18, bg=CARD, relief="flat", font=("Consolas", 9)); self.history_output.pack(fill="both", expand=True)

    def _run_physics_model_audit(self):
        try:
            if self.last_preview is None or self.last_phase is None:
                self._refresh_preview()
            preview = self.last_preview
            phase = np.asarray(self.last_phase if self.last_phase is not None else preview.phase_rad, dtype=float)
            amplitude = np.asarray(preview.amplitude, dtype=float)
            radius = max(float(self.state_model.lens.lens_radius_lambda), 1e-12)
            focal = max(float(self.state_model.lens.working_distance_lambda), 1e-12)
            na = float(radius / math.sqrt(radius * radius + focal * focal))
            debye = debye_high_na_psf(phase, amplitude, na, max(float(self.state_model.lens.n_refra_out), 1e-9))
            axis = np.linspace(-self.state_model.target.calculation_range_lambda,
                               self.state_model.target.calculation_range_lambda,
                               debye.shape[0])
            metrics = calculate_metrics(debye, axis)
            warnings = []
            if na > 0.7:
                warnings.append("High NA: prefer Debye-Wolf/vector validation.")
            source_mode = getattr(self.state_model.source, "light_source_mode", "")
            if source_mode == 1 or str(source_mode).lower() in {"led", "extended", "incoherent"}:
                warnings.append("Extended/LED source: use incoherent wavelength-angle averaging.")
            if getattr(self.state_model.optimization, "phase_design_mode", "") == "geometric":
                warnings.append("PB/geometric phase requires circular-polarization conversion-efficiency validation.")
            if self.meta_db.records:
                warnings.append("Meta-atom database is loaded; verify phase-to-geometry residuals before layout export.")
            if not warnings:
                warnings.append("Current parameters are within the fast scalar screening range.")
            self.physics_audit_result = {
                "selected_model": self.physics_model_var.get(),
                "estimated_na": na,
                "debye_fwhm_lambda": metrics.fwhm_um,
                "debye_mtf50_cyc_per_lambda": metrics.mtf50_cyc_per_um,
                "debye_efficiency": metrics.efficiency,
                "warnings": warnings,
            }
            text = (f"NA≈{na:.3f} | Debye FWHM={metrics.fwhm_um:.4g} λ | "
                    f"MTF50={metrics.mtf50_cyc_per_um:.4g} cyc/λ | Efficiency={metrics.efficiency:.2%}\n"
                    + "\n".join(f"- {item}" for item in warnings))
            self.physics_audit_label.configure(text=text)
        except Exception as exc:
            messagebox.showerror("Physics Audit", str(exc), parent=self)

    def _audit_meta_database(self):
        try:
            records = self.meta_db.records
            if not records:
                raise ValueError("Load an RCWA/FDTD meta-atom database first.")
            phases = np.unwrap([r.phase_rad for r in records])
            wrapped = np.sort(np.mod([r.phase_rad for r in records], 2 * math.pi))
            gaps = np.diff(np.r_[wrapped, wrapped[0] + 2 * math.pi]) if len(wrapped) > 1 else np.array([2 * math.pi])
            eff = np.asarray([r.efficiency for r in records], dtype=float)
            widths = np.asarray([r.width_um for r in records], dtype=float)
            lengths = np.asarray([r.length_um for r in records], dtype=float)
            heights = np.asarray([r.height_um for r in records], dtype=float)
            flags = []
            if eff.mean() < 0.5:
                flags.append("low mean efficiency")
            if min(widths.min(), lengths.min(), heights.min()) <= 0:
                flags.append("non-positive geometry")
            if float(gaps.max()) > math.pi / 6:
                flags.append("phase sampling gap > 30 deg")
            self.database_quality_result = {
                "records": len(records),
                "materials": sorted({r.material for r in records}),
                "polarizations": sorted({r.polarization for r in records}),
                "phase_coverage_cycles": float(np.ptp(phases) / (2 * math.pi)) if len(phases) > 1 else 0.0,
                "max_phase_gap_rad": float(gaps.max()),
                "efficiency_min": float(eff.min()),
                "efficiency_mean": float(eff.mean()),
                "efficiency_max": float(eff.max()),
                "width_range_um": [float(widths.min()), float(widths.max())],
                "length_range_um": [float(lengths.min()), float(lengths.max())],
                "height_range_um": [float(heights.min()), float(heights.max())],
                "manufacturing_flags": flags,
            }
            text = (f"{len(records)} records | phase coverage={self.database_quality_result['phase_coverage_cycles']:.2f} cycles | "
                    f"efficiency={eff.min():.2f}/{eff.mean():.2f}/{eff.max():.2f} | "
                    f"max phase gap={gaps.max():.3f} rad\n"
                    + ("Flags: " + ", ".join(flags) if flags else "Database quality looks usable for layout synthesis."))
            self.db_quality_label.configure(text=text)
        except Exception as exc:
            messagebox.showerror("Database Quality", str(exc), parent=self)

    def _generate_calibration_advice(self):
        if not self.experiment_result:
            self.experiment_label.configure(text="Import an experimental PSF or Letter-E image first.")
            return
        result = self.experiment_result
        advice = []
        if abs(result.get("shift_x_px", 0)) + abs(result.get("shift_y_px", 0)) > 2:
            advice.append("Registration shift is non-zero: check optical axis, object placement and crop origin.")
        if result.get("gain", 1.0) < 0.8 or result.get("gain", 1.0) > 1.2:
            advice.append("Gain differs from unity: calibrate exposure, source power or camera response.")
        if result.get("rmse", 0.0) > 0.15:
            advice.append("High RMSE: consider phase error, fabrication bias, defocus or background subtraction.")
        if result.get("correlation", 1.0) < 0.85:
            advice.append("Low correlation: run a full vector model or update the meta-atom database.")
        if not advice:
            advice.append("Experiment agrees with the current model; keep this run as a calibration reference.")
        result["advice"] = advice
        self.experiment_label.configure(text="Calibration advice:\n" + "\n".join(f"- {item}" for item in advice))

    def _generate_batch_plan(self):
        try:
            selected = [item for item in getattr(self.state_model.sampling, "selected_gpu_devices", "").split("||") if item]
            devices = selected or ["AUTO"]
            waves = [float(item.strip()) for item in self.multi_waves.get().split(",") if item.strip()]
            angles = [float(item.strip()) for item in self.field_angles.get().split(",") if item.strip()]
            jobs = []
            for wave in waves or [float(self.state_model.lens.wavelength_um)]:
                for angle in angles or [float(self.state_model.source.incident_angle_deg)]:
                    jobs.append(BatchJob(
                        name=f"lambda_{wave:.4g}_angle_{angle:.4g}",
                        parameters={"wavelength_um": wave, "field_angle_deg": angle},
                    ))
            checkpoint = self._data_dir() / "BatchQueue" / "v6_batch_preview.json"
            batch = CheckpointBatch(checkpoint, jobs)
            for index, job in enumerate(batch.jobs):
                job.gpu = devices[index % len(devices)]
            batch.save()
            self.batch_plan_result = {
                "checkpoint": str(checkpoint),
                "devices": devices,
                "jobs": [job.__dict__ for job in batch.jobs],
            }
            self.batch_plan_label.configure(text=f"{len(jobs)} jobs | {len(devices)} GPU target(s)\nCheckpoint: {checkpoint}")
        except Exception as exc:
            messagebox.showerror("Batch Queue", str(exc), parent=self)

    def _load_meta_database(self):
        path = filedialog.askopenfilename(filetypes=[("Meta-atom database", "*.csv *.json")])
        if not path: return
        try:
            self.meta_db = MetaAtomDatabase.load(path); self.db_path_var.set(path)
            efficiencies = [r.efficiency for r in self.meta_db.records]
            value = (f"已导入 {len(efficiencies)} 条记录｜效率 {min(efficiencies):.3f}...{max(efficiencies):.3f}" if self.language == "zh" else f"{len(efficiencies)} records | efficiency {min(efficiencies):.3f}...{max(efficiencies):.3f}")
            self.db_status.configure(text=value, foreground="#107C10")
        except Exception as exc: messagebox.showerror("数据库" if self.language == "zh" else "Database", str(exc), parent=self)

    def _run_adjoint(self):
        try:
            n = min(256, max(32, int(float(self.controls.get("preview_n", tk.StringVar(value=128)).get()))))
            yy, xx = np.indices((n, n)); rr = np.hypot(xx-(n-1)/2, yy-(n-1)/2)
            aperture = rr <= n*.46
            sigma = max(1.2, n*.025)
            target = np.exp(-(rr/sigma)**2).astype(complex)
            if self.target_mode_index == 4 or self.target_mode.get() in {"Letter E", "字母 E"}:
                target[:] = 0; target[n//3:2*n//3, n//3:n//3+4] = 1
                target[n//3:n//3+4, n//3:2*n//3] = 1; target[n//2-2:n//2+2, n//3:n//2+8] = 1; target[2*n//3-4:2*n//3, n//3:2*n//3] = 1
            initial = np.zeros((n, n)) if self.last_phase is None or self.last_phase.shape != (n,n) else self.last_phase
            result = fourier_adjoint_optimize(initial, target, aperture, int(self.adjoint_iterations.get()),
                                              float(self.adjoint_lr.get()), int(self.phase_levels_v5.get()), .002)
            self.last_phase = result.phase; self.phase_library["optimized"] = result.phase
            self.adjoint_progress["value"] = 100
            value = (f"已完成 {len(result.loss)} 次伴随迭代。\n最终损失：{result.loss[-1]:.6g}\n重合效率：{result.efficiency[-1]:.4f}" if self.language == "zh" else f"Completed {len(result.loss)} adjoint steps.\nFinal loss: {result.loss[-1]:.6g}\nOverlap efficiency: {result.efficiency[-1]:.4f}")
            messagebox.showinfo("逆向设计" if self.language == "zh" else "Inverse Design", value, parent=self)
        except Exception as exc: messagebox.showerror("逆向设计" if self.language == "zh" else "Inverse Design", str(exc), parent=self)

    def _export_layout(self, kind):
        if not self.meta_db.records: messagebox.showwarning("版图" if self.language == "zh" else "Layout", "请先导入 RCWA/FDTD 超原子数据库。" if self.language == "zh" else "Load an RCWA/FDTD meta-atom database first.", parent=self); return
        if self.last_phase is None: messagebox.showwarning("版图" if self.language == "zh" else "Layout", "请先运行或导入相位设计。" if self.language == "zh" else "Run or import a phase design first.", parent=self); return
        suffix = ".gds" if kind == "gds" else ".dxf"
        path = filedialog.asksaveasfilename(defaultextension=suffix, filetypes=[(kind.upper(), "*"+suffix)])
        if not path: return
        phase = self.last_phase
        stride = max(1, math.ceil(max(phase.shape)/96)); phase = phase[::stride, ::stride]
        layout = self.meta_db.synthesize(phase, self.state_model.lens.pitch_um*stride, self.state_model.lens.wavelength_um, 0, "TE")
        (export_gds_rectangles if kind == "gds" else export_dxf_rectangles)(layout, path)
        csv_path = Path(path).with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=layout[0].keys()); writer.writeheader(); writer.writerows(layout)
        messagebox.showinfo("版图" if self.language == "zh" else "Layout", (f"已导出 {len(layout)} 个超原子及映射 CSV。" if self.language == "zh" else f"Exported {len(layout)} atoms and mapping CSV."), parent=self)

    def _run_robustness(self):
        try:
            base = self.history[-1] if self.history else {"FWHM": .36, "SR": .15, "Peak": 1, "Contrast": .8}
            def evaluator(p):
                size = abs(p["linewidth_nm"])*.002 + abs(p["etch_depth_nm"])*.0005 + abs(p["refractive_index"])*2
                return {"fwhm_lambda": float(base.get("FWHM", .36))*(1+size),
                        "contrast": max(0., float(base.get("Contrast", .8))*(1-size)),
                        "efficiency": max(0., 0.85-1.5*size)}
            cfg = ToleranceConfig(float(self.linewidth_sigma.get()), float(self.etch_sigma.get()), float(self.index_sigma.get()), 3, .5, int(self.robust_samples.get()))
            self.robustness_result = monte_carlo_tolerance(evaluator, cfg)
            self.robust_text.delete("1.0", "end"); self.robust_text.insert("1.0", json.dumps(self.robustness_result["summary"], indent=2))
        except Exception as exc: messagebox.showerror("Robustness", str(exc), parent=self)

    def _run_lithography_metrics(self):
        image = self.last_image
        if image is None and self.last_preview is not None: image = self.last_preview.intensity
        if image is None: messagebox.showwarning("光刻计量" if self.language == "zh" else "Lithography", "请先生成焦面或字母 E 成像数据。" if self.language == "zh" else "Generate focal or Letter-E image data first.", parent=self); return
        result = lithography_metrics(image, float(self.pixel_um.get()), float(self.resist_threshold.get()))
        self.litho_result.configure(text=" | ".join(f"{k}: {v:.5g}" for k,v in result.items()))

    def _import_experiment(self):
        path = filedialog.askopenfilename(filetypes=[("Array or image", "*.csv *.txt *.npy *.png *.jpg *.jpeg *.tif *.tiff")])
        if not path: return
        try:
            suffix = Path(path).suffix.lower()
            obs = np.load(path) if suffix == ".npy" else (np.loadtxt(path, delimiter="," if suffix == ".csv" else None) if suffix in {".csv", ".txt"} else np.asarray(Image.open(path).convert("L"), dtype=float))
            ref = self.last_image if self.last_image is not None else (self.last_preview.intensity if self.last_preview is not None else None)
            if ref is None: raise ValueError("Generate a simulated focal or Letter-E image first.")
            if obs.shape != ref.shape: obs = np.asarray(Image.fromarray(obs.astype(np.float32), mode="F").resize((ref.shape[1],ref.shape[0]), Image.Resampling.BILINEAR))
            self.experiment_result = align_experiment(ref, obs)
            value = (f"位移 ({self.experiment_result['shift_x_px']}, {self.experiment_result['shift_y_px']}) px｜RMSE {self.experiment_result['rmse']:.5g}｜相关系数 {self.experiment_result['correlation']:.5f}" if self.language == "zh" else f"Shift ({self.experiment_result['shift_x_px']}, {self.experiment_result['shift_y_px']}) px | RMSE {self.experiment_result['rmse']:.5g} | correlation {self.experiment_result['correlation']:.5f}")
            self.experiment_label.configure(text=value)
        except Exception as exc: messagebox.showerror("Experiment", str(exc), parent=self)

    def _show_mueller(self):
        ex, ey, phase = float(self.jones_ex.get()), float(self.jones_ey.get()), math.radians(float(self.jones_phase.get()))
        j = np.array([[ex, 0], [0, ey*np.exp(1j*phase)]], complex)
        messagebox.showinfo("Mueller Matrix", np.array2string(jones_to_mueller(j), precision=5), parent=self)

    def _save_project_version(self):
        path = filedialog.asksaveasfilename(defaultextension=".sqlite", filetypes=[("Project database", "*.sqlite")])
        if not path: return
        with ProjectDatabase(path) as db:
            row = db.save("MetaLens Open Workbench", self.state_model.to_dict(), self.history[-1] if self.history else {}, "v6 workbench snapshot")
        messagebox.showinfo("项目版本库" if self.language == "zh" else "Project Database", (f"已保存不可变项目版本，记录 ID {row}。" if self.language == "zh" else f"Saved immutable project version, row ID {row}."), parent=self)

    def _generate_report(self):
        path = filedialog.asksaveasfilename(defaultextension=".html", filetypes=[("HTML report", "*.html")])
        if not path: return
        sections = {
            "Project": self.state_model.to_dict(),
            "Physics Model Audit": self.physics_audit_result,
            "Meta-Atom Database Quality": self.database_quality_result,
            "Optimization": self.history,
            "Robustness": self.robustness_result,
            "Experiment": {k: v for k, v in (self.experiment_result or {}).items() if k != "aligned"},
            "Multi-GPU Batch Queue": self.batch_plan_result,
        }
        generate_html_report(path, "MetaLens Research Report v6.0.0", sections)
        messagebox.showinfo("科研报告" if self.language == "zh" else "Report", "已生成独立完整的科研报告。" if self.language == "zh" else "The self-contained research report was generated.", parent=self)

    def _browse_history(self):
        value = filedialog.askdirectory();
        if value: self.history_path.set(value)

    def _train_history_v5(self):
        try:
            self.history_advanced_progress["value"] = 15
            audit = self.history_lab.load_projects(self.history_path.get()); report = self.history_lab.train()
            self.audit_samples_value.configure(text=str(audit.samples))
            self.audit_duplicates_value.configure(text=str(audit.duplicates))
            self.audit_outliers_value.configure(text=str(audit.outliers))
            rmse = float(np.mean(list(report["validation_rmse"].values())))
            self.history_advanced_progress["value"] = 100
            self.history_advanced_var.set((
                f"训练完成。平均交叉验证 RMSE={rmse:.5g}；已建立 {report['ensemble_models']} 个集成模型。"
                if self.language == "zh" else
                f"Training complete. Mean cross-validation RMSE={rmse:.5g}; {report['ensemble_models']} ensemble models fitted."))
        except Exception as exc: messagebox.showerror("Historical Intelligence", str(exc), parent=self)

    def _predict_history_v5(self):
        try:
            targets = {
                "wavelength_um": float(self.controls["ml_wavelength"].get()),
                "focal_lambda": float(self.controls["ml_focal"].get()),
                "radius_lambda": float(self.controls["ml_radius"].get()),
                "target_fwhm_lambda": float(self.controls["ml_fwhm"].get()),
                "max_sidelobe": float(self.controls["ml_sidelobe"].get()) / 100.0,
                "target_contrast": float(self.controls["ml_contrast"].get()),
                "field_angle_deg": float(self.history_field_angle.get()),
                "bandwidth_nm": float(self.history_bandwidth.get()),
            }
            result = self.history_lab.predict(targets)
            uncertainty = float(np.mean(list(result["uncertainty"].values())))
            confidence = max(0.0, min(1.0, 1.0-uncertainty))
            self.ml_confidence_value.configure(text=f"{confidence:.1%}")
            self.ml_confidence_bar["value"] = confidence*100
            domain = result["outside_training_domain"]
            self.history_advanced_var.set((
                f"预测完成；平均不确定度 {uncertainty:.5g}。" + (f" 超出训练范围：{', '.join(domain)}。建议回到设计优化生成新样本。" if domain else " 当前目标位于历史数据覆盖范围内。")
                if self.language == "zh" else
                f"Prediction complete; mean uncertainty {uncertainty:.5g}. " + (f"Outside training domain: {', '.join(domain)}. Run a full optimization to add a sample." if domain else "Target is within historical coverage.")))
        except Exception as exc: messagebox.showerror("Historical Intelligence", str(exc), parent=self)

    def _active_learning_v5(self):
        try:
            candidates = self.history_lab.active_learning_candidates({}, count=5)
            lines = []
            for index, row in enumerate(candidates, 1):
                lines.append(f"{index}. λ={row['wavelength_um']:.4g} μm, f={row['focal_lambda']:.4g} λ, NA-range radius={row['radius_lambda']:.4g} λ, novelty={row['novelty']:.3f}")
            self.history_advanced_var.set(("建议优先补充以下高信息量设计：\n" if self.language == "zh" else "Recommended high-information designs:\n") + "\n".join(lines))
        except Exception as exc: messagebox.showerror("Historical Intelligence", str(exc), parent=self)


def main():
    app = ProApp()
    app.mainloop()
