from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum
import os
from pathlib import Path
from typing import Any, Dict, List


class LensCenterType(IntEnum):
    CENTER_TRANSPARENT = 0
    CENTER_BLOCKED = 1


class MetaArrangeType(IntEnum):
    META_CENTER = 0
    META_CORNER = 1
    RING_BELT_CENTER = 2
    RING_BELT_CORNER = 3


class LensType(IntEnum):
    SINGLET_NO_THICKNESS = 0
    SINGLET_THICKNESS = 1
    DOUBLETS = 2
    LENS_WITH_APERTURE = 3
    MULTI_LAYER_LENSES = 4


class LensSubType(IntEnum):
    NORMAL = 0
    INTEGRATED = 1
    FLAT_FIELD = 2
    ACHROMATIC_MULTI_WAVELENGTH = 3
    NON_DIFFRACTION_MULTI_POINT = 4
    NON_DIFFRACTION_AXICON_MODIFIED = 5
    SINGLE_POINT = 6
    NORMAL_ON_Z_INFO = 7


class LensSubType2(IntEnum):
    NORMAL = 0
    ANTI_NORMAL = 1
    RANDOM = 2
    FLAT_K_F_CETA = 3
    FLAT_K_F_TAN = 4
    FLAT_K_F_SIN = 5


class ImageType(IntEnum):
    REAL_IMAGE = 1
    VIRTUAL_IMAGE = 2


class OptimizationType(IntEnum):
    AMPLITUDE_ONLY = 0
    PHASE_ONLY = 1
    AMPLITUDE_AND_PHASE = 2
    PHASE_AND_RADIUS = 3
    LOAD_META_ATOM_PARAMETERS = 4
    AMP_PHASE_POLARIZATION = 5


class IncidentPolarization(IntEnum):
    X_LINEAR = 0
    AZIMUTHAL = 1
    RADIAL = 2
    CIRCULAR = 3


class BeamShape(IntEnum):
    PLANE_WAVE = 0
    GAUSSIAN = 1


class LightSourceMode(IntEnum):
    LASER = 0
    LED = 1


class PolarizationMode(IntEnum):
    TE = 0
    TM = 1
    UNPOLARIZED = 2
    LINEAR = 3
    LEFT_CIRCULAR = 4
    RIGHT_CIRCULAR = 5


class FieldShape(IntEnum):
    SOLID_SPOT = 0
    HOLLOW_RING = 1
    HOLLOW_SPOT = 2


class PolarPreference(IntEnum):
    X_POLAR = 0
    Y_POLAR = 1
    TRANSVERSE_POLAR = 2
    LONGITUDINAL_POLAR = 3
    ENTIRE_FIELDS = 4


class CalculationBackend(IntEnum):
    AUTO = 0
    CPU = 1
    NVIDIA_CUDA = 2
    OPENCL = 3
    MULTI_GPU = 4


class CalculationAccuracy(IntEnum):
    DOUBLE = 0
    FLOAT = 1


class OptimizationSequence(IntEnum):
    SR_FWHM_INTENSITY = 0
    FWHM_SR_INTENSITY = 1


class OptimizationMethod(IntEnum):
    BINARY_PSO = 0
    BLACK_HOLE_PSO = 1
    MARINE_PREDATOR = 2
    MODIFIED_MARINE_PREDATOR = 3


class ReloadStrategy(IntEnum):
    GENERATE_NEW_PARTICLES = 0
    RELOAD_ALL_PARTICLES = 1
    RELOAD_GLOBAL_BEST_ONLY = 2


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def enum_options(enum_type: type[IntEnum]) -> List[str]:
    return [f"{item.value}-{item.name.replace('_', ' ').title()}" for item in enum_type]


def enum_from_option(enum_type: type[IntEnum], option: str | int) -> IntEnum:
    if isinstance(option, int):
        return enum_type(option)
    if "-" in option:
        return enum_type(int(option.split("-", 1)[0]))
    return enum_type[option]


@dataclass
class LensBasicParameters:
    wavelength_um: float = 0.6328
    working_distance_lambda: float = 10.0
    lens_radius_lambda: float = 50.0
    view_radius_lambda: float = 0.0
    pitch_um: float = 0.24
    n_refra_in: float = 1.0
    n_refra_out: float = 1.0
    n_refra_lens: float = 1.4
    lens_center_type: int = int(LensCenterType.CENTER_TRANSPARENT)
    center_block_radius_lambda: float = 0.0
    meta_arrange_type: int = int(MetaArrangeType.META_CENTER)
    lens_type: int = int(LensType.SINGLET_NO_THICKNESS)
    lens_thickness_um: float = 175.0
    lens_sub_type: int = int(LensSubType.NORMAL)
    lens_sub_type2: int = int(LensSubType2.NORMAL)
    lens_sub_type3: int = int(ImageType.REAL_IMAGE)
    k_flat_field: float = 1.0
    integrated_lens_separation_um: float = 0.0
    na_axicon: float = 0.0


@dataclass
class OptimizationParameters:
    phase_design_mode: str = "propagation"
    geometric_conversion_efficiency: float = 0.9
    geometric_handedness: int = 1
    optimization_type: int = int(OptimizationType.PHASE_ONLY)
    amp_n: int = 32
    amp_min: float = 0.0
    amp_max: float = 1.0
    phase_n: int = 32
    phase_min_pi: float = 0.0
    phase_max_pi: float = 2.0
    radius_n: int = 32
    radius_min_um: float = 0.0
    radius_max_um: float = 1.0
    polar_n: int = 32
    polar_min_pi: float = 0.0
    polar_max_pi: float = 1.0
    optimize_lens_thickness: bool = False
    lens_thickness_n: int = 32
    lens_thickness_min_um: float = 50.0
    lens_thickness_max_um: float = 500.0


@dataclass
class SourceParameters:
    incident_polarization: int = int(IncidentPolarization.X_LINEAR)
    theta_polar_angle_deg: float = 0.0
    beam_shape: int = int(BeamShape.PLANE_WAVE)
    waist_w0_lambda: float = 0.0
    incident_angle_deg: float = 0.0
    incident_angle_n: int = 1
    theta_min_deg: float = 0.0
    theta_max_deg: float = 0.0
    angle_groups: int = 1
    incident_wavelength_n: int = 1
    wavelength_min_um: float = 0.6328
    wavelength_max_um: float = 0.6328
    diffraction_z_n: int = 1
    z_min_lambda: float = 10.0
    z_max_lambda: float = 10.0
    light_source_mode: int = int(LightSourceMode.LASER)
    polarization_mode: int = int(PolarizationMode.TE)
    led_fwhm_nm: float = 20.0
    wavelength_samples: int = 5
    led_divergence_half_angle_deg: float = 5.0
    angle_samples: int = 3
    spectrum_file: str = ""
    angular_distribution_file: str = ""
    jones_ex: float = 1.0
    jones_ey: float = 0.0
    jones_phase_deg: float = 0.0


@dataclass
class TargetParameters:
    field_shape: int = int(FieldShape.SOLID_SPOT)
    polar_preference: int = int(PolarPreference.X_POLAR)
    calculation_range_lambda: float = 20.0
    fwhm_lambda: float = 0.36
    sidelobe_percent: float = 25.0
    peak_intensity: float = 1_000_000.0


@dataclass
class SamplingParameters:
    nx: int = 4096
    ny: int = 4096
    fft_n: int = 0
    interpolate: bool = False
    dxs_lambda: float = 0.1
    backend: int = int(CalculationBackend.AUTO)
    multi_gpu_n: int = 0
    manual_thread_mode: bool = False
    thread_count: int = 4
    calculation_accuracy: int = int(CalculationAccuracy.DOUBLE)
    preview_n: int = 256
    selected_gpu_devices: str = ""
    gpu_memory_fraction: float = 0.75
    gpu_batch_size: int = 2


@dataclass
class LithographyTargetParameters:
    enabled: bool = True
    optional: bool = True
    angular_magnification_target: float = 1.8
    angular_magnification_weight: float = 1.0
    field_uniformity_target_percent: float = 95.0
    telecentricity_error_max_deg: float = 0.05
    distortion_max_percent: float = 0.5
    exposure_window_um: float = 1.0
    overlay_tolerance_nm: float = 20.0
    min_feature_nm: float = 500.0
    working_field_mm: float = 5.0
    image_contrast_target: float = 0.145
    image_contrast_min: float = 0.105
    fwhm_max_lambda: float = 0.51
    sidelobe_max_ratio: float = 0.18
    fov_half_angle_deg: float = 7.0
    focal_depth_lambda: float = 1.0
    reduction_ratio_target: float = 4.57
    projection_mode: str = "angular_magnification_metalens"
    resist_threshold: float = 0.5
    cd_tolerance_percent: float = 10.0
    target_nils: float = 2.0
    contrast_weight: float = 3.0
    peak_weight: float = 0.35
    sidelobe_weight: float = 0.55
    fwhm_weight: float = 0.45


@dataclass
class ResearchParameters:
    meta_atom_database: str = ""
    design_mode: str = "Fourier adjoint"
    target_mode: str = "Single focus"
    target_wavelengths_um: str = "0.6328"
    focal_planes_lambda: str = "10"
    field_angles_deg: str = "0"
    adjoint_iterations: int = 100
    learning_rate: float = 0.08
    smooth_weight: float = 0.002
    robust_samples: int = 100
    linewidth_sigma_nm: float = 5.0
    etch_depth_sigma_nm: float = 10.0
    index_sigma: float = 0.005
    corner_sigma_nm: float = 3.0
    experiment_file: str = ""
    project_database: str = ""


@dataclass
class PsoParameters:
    particle_number: int = 60
    iterations_num: int = 100_000
    optimization_sequence: int = int(OptimizationSequence.SR_FWHM_INTENSITY)
    optimization_method: int = int(OptimizationMethod.BINARY_PSO)
    reload_strategy: int = int(ReloadStrategy.GENERATE_NEW_PARTICLES)
    linear_weight: bool = True
    fixed_weight: bool = False
    weight: float = 0.4
    weight_max: float = 1.0
    weight_min: float = 0.0
    gabpso: bool = True
    crossover: float = 0.5


@dataclass
class PropagationParameters:
    cal_type: int = 0
    ns_para: int = 1
    zs_min_lambda: float = -20.0
    zs_max_lambda: float = 20.0
    ns_z: int = 41
    theta_min_deg: float = 0.0
    theta_max_deg: float = 0.0
    wavelength_min_um: float = 0.6328
    wavelength_max_um: float = 0.6328


@dataclass
class ImagingParameters:
    working_wavelength_um: float = 0.6328
    focal_length_um: float = 10.0
    lens_radius_um: float = 31.64
    fwhm_focal_ratio: float = 0.01
    angular_magnification: float = 1.0
    center_distance_um: float = 2.0
    objective_distance_um: float = 100.0
    ns_z: int = 1
    zs_min_um: float = 10.0
    zs_max_um: float = 10.0
    xs_min_um: float = -20.0
    xs_max_um: float = 20.0
    grid_n: int = 256
    e_width_um: float = 5.0
    e_height_um: float = 4.5
    e_bar_spacing_um: float = 2.0
    e_line_width_um: float = 0.5
    e_middle_arm_ratio: float = 0.72
    e_point_spacing_um: float = 0.25


@dataclass
class ProjectState:
    lens: LensBasicParameters = field(default_factory=LensBasicParameters)
    optimization: OptimizationParameters = field(default_factory=OptimizationParameters)
    source: SourceParameters = field(default_factory=SourceParameters)
    target: TargetParameters = field(default_factory=TargetParameters)
    sampling: SamplingParameters = field(default_factory=SamplingParameters)
    pso: PsoParameters = field(default_factory=PsoParameters)
    propagation: PropagationParameters = field(default_factory=PropagationParameters)
    imaging: ImagingParameters = field(default_factory=ImagingParameters)
    lithography: LithographyTargetParameters = field(default_factory=LithographyTargetParameters)
    research: ResearchParameters = field(default_factory=ResearchParameters)
    output_path: str = ""
    data_path: str = ""

    def validate(self) -> List[str]:
        errors: List[str] = []
        for name, value in {
            "wavelength_um": self.lens.wavelength_um,
            "working_distance_lambda": self.lens.working_distance_lambda,
            "lens_radius_lambda": self.lens.lens_radius_lambda,
            "pitch_um": self.lens.pitch_um,
            "n_refra_in": self.lens.n_refra_in,
            "n_refra_out": self.lens.n_refra_out,
            "n_refra_lens": self.lens.n_refra_lens,
        }.items():
            if value <= 0:
                errors.append(f"{name} must be positive.")

        for name, value in {
            "amp_n": self.optimization.amp_n,
            "phase_n": self.optimization.phase_n,
            "radius_n": self.optimization.radius_n,
            "polar_n": self.optimization.polar_n,
            "lens_thickness_n": self.optimization.lens_thickness_n,
            "nx": self.sampling.nx,
            "ny": self.sampling.ny,
            "preview_n": self.sampling.preview_n,
        }.items():
            if not is_power_of_two(int(value)):
                errors.append(f"{name} should be a power of two.")

        if self.optimization.phase_max_pi <= self.optimization.phase_min_pi:
            errors.append("phase_max_pi must be greater than phase_min_pi.")
        if self.optimization.amp_max < self.optimization.amp_min:
            errors.append("amp_max must be greater than or equal to amp_min.")
        if self.pso.particle_number <= 0 or self.pso.iterations_num <= 0:
            errors.append("PSO particle_number and iterations_num must be positive.")
        if self.source.led_fwhm_nm < 0 or self.source.led_divergence_half_angle_deg < 0:
            errors.append("LED bandwidth and divergence must be non-negative.")
        if self.source.wavelength_samples <= 0 or self.source.angle_samples <= 0:
            errors.append("Source wavelength and angle sample counts must be positive.")
        if self.imaging.e_width_um <= 0 or self.imaging.e_height_um <= 0 or self.imaging.e_line_width_um <= 0:
            errors.append("Letter-E width, height and stroke width must be positive.")
        if not 0.1 <= self.imaging.e_middle_arm_ratio <= 1.0:
            errors.append("Letter-E middle-arm ratio must be between 0.1 and 1.0.")
        focal_um = self.lens.working_distance_lambda * self.lens.wavelength_um
        if self.lithography.enabled and self.imaging.objective_distance_um <= focal_um:
            errors.append("Projection lithography requires object distance greater than focal length for a real image.")
        if not 0.0 < self.lithography.resist_threshold < 1.0:
            errors.append("Lithography resist_threshold must be between 0 and 1.")
        if self.optimization.phase_design_mode not in {"propagation", "geometric", "hybrid"}:
            errors.append("phase_design_mode must be propagation, geometric or hybrid.")
        if not 0.0 < self.optimization.geometric_conversion_efficiency <= 1.0:
            errors.append("geometric_conversion_efficiency must be in (0, 1].")
        if not 0.1 <= self.sampling.gpu_memory_fraction <= 0.95:
            errors.append("gpu_memory_fraction must be between 0.1 and 0.95.")
        if self.sampling.gpu_batch_size <= 0:
            errors.append("gpu_batch_size must be positive.")
        logical_cpus = os.cpu_count() or 1
        if self.sampling.thread_count <= 0:
            errors.append("thread_count must be positive.")
        if self.sampling.thread_count > max(1, logical_cpus * 2):
            errors.append(f"thread_count is too high for this computer ({logical_cpus} logical CPU threads detected).")
        for name, value in {
            "contrast_weight": self.lithography.contrast_weight,
            "peak_weight": self.lithography.peak_weight,
            "sidelobe_weight": self.lithography.sidelobe_weight,
            "fwhm_weight": self.lithography.fwhm_weight,
        }.items():
            if value < 0:
                errors.append(f"{name} must be non-negative.")
        if self.research.adjoint_iterations <= 0 or self.research.robust_samples <= 0:
            errors.append("Research iteration and robustness sample counts must be positive.")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectState":
        return cls(
            lens=LensBasicParameters(**data.get("lens", {})),
            optimization=OptimizationParameters(**data.get("optimization", {})),
            source=SourceParameters(**data.get("source", {})),
            target=TargetParameters(**data.get("target", {})),
            sampling=SamplingParameters(**data.get("sampling", {})),
            pso=PsoParameters(**data.get("pso", {})),
            propagation=PropagationParameters(**data.get("propagation", {})),
            imaging=ImagingParameters(**data.get("imaging", {})),
            lithography=LithographyTargetParameters(**data.get("lithography", {})),
            research=ResearchParameters(**data.get("research", {})),
            output_path=data.get("output_path", ""),
            data_path=data.get("data_path", ""),
        )

    def resolved_output_path(self) -> Path:
        return Path(self.output_path).expanduser() if self.output_path else Path.cwd()
