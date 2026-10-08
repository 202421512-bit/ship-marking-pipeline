"""Central configuration for the PAC Mission 3 marking-analysis research prototype.

Every tunable number used by the pipeline lives here so that no module contains
hidden magic numbers. All thresholds are DEMO values for a research prototype;
they are NOT industrial safety criteria.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Paths (relative to the project root, never hard-coded absolute paths)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
INPUT_DIR: Path = DATA_DIR / "input"
DEMO_DIR: Path = DATA_DIR / "demo"
PROTOTYPE_DIR: Path = DATA_DIR / "prototypes"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
ENV_FILE: Path = PROJECT_ROOT / ".env"

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")

PROJECT_TITLE: str = "PAC MISSION 3"
PROJECT_SUBTITLE: str = "SHIP BLOCK MARKING ANALYSIS"
PROTOTYPE_LABEL: str = "Research Prototype / Proof of Concept / Demo"


def ensure_directories() -> None:
    """Create every working directory the pipeline needs (idempotent)."""
    for directory in (INPUT_DIR, DEMO_DIR, PROTOTYPE_DIR, RESULTS_DIR):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - filesystem specific
            print(f"[WARN] Could not create directory {directory}: {exc}")


# ---------------------------------------------------------------------------
# Classes (add new labels here; prototypes are rebuilt automatically)
# ---------------------------------------------------------------------------
CLASSES: List[str] = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F",
]


@dataclass(frozen=True)
class PreprocessingConfig:
    """Parameters of the image preprocessing stage."""

    normalized_size: int = 64            # side of the square analysis canvas
    glyph_box: int = 48                  # longest glyph side inside the canvas
    illumination_sigma_ratio: float = 0.25  # background blur sigma / max(H, W)
    stretch_low_percentile: float = 1.0
    stretch_high_percentile: float = 99.0
    clahe_mode: str = "auto"             # "auto" (only for low contrast) | "on" | "off"
    clahe_low_contrast_range: float = 60.0  # p95 - p5 grey levels below this = low light
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 4
    median_kernel: int = 3
    denoise_strength_per_sigma: float = 1.2  # NL-means h = factor * estimated noise sigma
    denoise_strength_min: float = 7.0
    denoise_strength_max: float = 30.0
    denoise_template_window: int = 7
    denoise_search_window: int = 21
    threshold_method: str = "otsu"       # "otsu" | "adaptive"
    adaptive_block_ratio: float = 0.15   # block size / min(H, W)
    adaptive_c: float = 5.0
    border_margin_ratio: float = 0.04    # width of border strip for polarity test
    min_component_ratio: float = 0.02    # component area / largest component area
    min_component_pixels: int = 12
    use_closing: bool = False            # restoration handles closing explicitly
    closing_kernel: int = 3
    resize_binarize_threshold: float = 0.5


@dataclass(frozen=True)
class TopologyConfig:
    """Parameters of the topology analysis stage."""

    min_hole_area: int = 4               # px on the normalized canvas
    spur_prune_length: int = 4           # skeleton spurs shorter than this are pruned
    use_persistent_homology: bool = True
    persistence_min_lifetime: float = 1.5   # px of signed distance
    persistence_max_gap_birth: float = 3.0  # H1 born above this is not a "near hole"


@dataclass(frozen=True)
class PrototypeConfig:
    """Parameters of the prototype statistical reference model."""

    fonts: Tuple[str, ...] = (
        "arial.ttf", "arialbd.ttf", "calibri.ttf", "consola.ttf",
        "verdana.ttf", "tahoma.ttf", "times.ttf", "segoeui.ttf",
    )
    render_canvas: int = 160
    render_font_size: int = 110
    augmentations_per_font: int = 8
    rotation_deg: float = 7.0
    scale_range: Tuple[float, float] = (0.85, 1.10)
    translation_px: int = 6
    morph_probability: float = 0.35      # chance of dilation / erosion
    blur_sigma_range: Tuple[float, float] = (0.0, 2.0)
    threshold_range: Tuple[int, int] = (100, 160)
    seed: int = 42
    cache_file: str = "prototype_stats.json"
    template_file: str = "prototype_templates.npz"
    cache_version: int = 3


@dataclass(frozen=True)
class RestorationConfig:
    """Weights of the topology-constrained restoration objective J."""

    lambda_data: float = 0.30
    lambda_topology: float = 0.30
    lambda_geometry: float = 0.25
    lambda_change: float = 0.15
    w_beta0: float = 1.0
    w_beta1: float = 1.0
    w_endpoints: float = 0.15
    w_branch_points: float = 0.15
    geometry_z_clip: float = 5.0
    top_k_reference: int = 3
    operations: Tuple[str, ...] = (
        "original", "closing_3x3", "closing_5x5", "closing_7x7",
        "opening_3x3", "opening_5x5", "dilation_3x3", "erosion_3x3",
        "hole_filling", "small_component_removal",
    )
    small_component_ratio: float = 0.10  # relative to largest component


@dataclass(frozen=True)
class BayesianConfig:
    """Parameters of the Gaussian Bayesian shape inference."""

    topology_features: Tuple[str, ...] = (
        "beta_0", "beta_1", "euler_characteristic",
        "skeleton_length_norm", "endpoints", "branch_points",
    )
    geometry_features: Tuple[str, ...] = (
        "aspect_ratio", "area_ratio", "circularity", "solidity", "eccentricity",
        "hu_1", "hu_2", "hu_3", "hu_4",
        "centroid_x_norm", "centroid_y_norm",
        "quadrant_tl", "quadrant_tr", "quadrant_bl", "quadrant_br",
    )
    topology_weight: float = 1.0
    geometry_weight: float = 1.0
    # "sum": Σ feature log-lik per group (synthetic demo model, unchanged)
    # "mean": per-group mean log-lik, so group size does not decide influence (real experiment)
    aggregation: str = "sum"
    # Tempering divides the summed log-likelihood; naive feature independence
    # otherwise produces over-confident posteriors.
    likelihood_temperature: float = 4.0
    variance_epsilon: float = 1e-6
    sigma_floor_default: float = 0.04
    sigma_floor: Dict[str, float] = field(default_factory=lambda: {
        "beta_0": 0.35, "beta_1": 0.35, "euler_characteristic": 0.35,
        "endpoints": 0.6, "branch_points": 0.6, "skeleton_length_norm": 0.25,
    })
    priors: Dict[str, float] = field(default_factory=dict)  # empty = uniform
    top_n: int = 5


@dataclass(frozen=True)
class VLMConfig:
    """Parameters of the optional VLM visual verification."""

    default_model: str = "gpt-4o-mini"
    timeout_sec: float = 45.0
    max_tokens: int = 500
    temperature: float = 0.0
    image_side: int = 256


@dataclass(frozen=True)
class FusionConfig:
    """Parameters of heuristic evidence fusion."""

    alpha: float = 0.70                  # exponent of Bayesian evidence
    vlm_evidence_floor: float = 0.01     # evidence for labels not named by the VLM


@dataclass(frozen=True)
class DecisionConfig:
    """DEMO decision thresholds - NOT industrial safety criteria."""

    high_fusion_score: float = 0.80
    high_bayes_posterior: float = 0.70
    high_vlm_confidence: float = 0.70
    max_normalized_entropy_high: float = 0.35
    ambiguous_normalized_entropy: float = 0.70
    ambiguous_margin: float = 0.15
    geometry_consistency_max_z: float = 1.5
    require_vlm_for_high_confidence: bool = True


@dataclass(frozen=True)
class DemoConfig:
    """Parameters of synthetic field degradation for --demo."""

    target: str = "8"
    seed: int = 42
    canvas: int = 160
    font: str = "arialbd.ttf"
    font_size: int = 120
    rotation_deg: float = 6.0
    gap_width_px: int = 6                # stroke removal across upper loop
    partial_break_px: int = 4            # partial break on lower loop
    blur_sigma: float = 1.6
    speckle_std: float = 18.0
    noise_blobs: int = 9
    illumination_gradient: float = 70.0
    foreground_level: int = 55
    background_level: int = 175


SYMBOLS_DIR: Path = (INPUT_DIR / "Symbols") if (INPUT_DIR / "Symbols").exists() else (PROJECT_ROOT / "Symbols")
REAL_RESULTS_DIR: Path = RESULTS_DIR / "real_symbols"
CHAR_RESULTS_DIR: Path = RESULTS_DIR / "character_experiment"
DEGRADATION_RESULTS_DIR: Path = RESULTS_DIR / "degradation_experiment"
UNCERTAINTY_RESULTS_DIR: Path = RESULTS_DIR / "uncertainty_validation"
MARKING_RESULTS_DIR: Path = RESULTS_DIR / "real_marking_recognition"
DIRECTION_RESULTS_DIR: Path = RESULTS_DIR / "direction_aware_experiment"
TEXT_DETECTION_RESULTS_DIR: Path = RESULTS_DIR / "text_detection_integration"
MODELS_DIR: Path = DATA_DIR / "models"
SCENE_RESULTS_DIR: Path = RESULTS_DIR / "full_scene_stabilization"
SYMBOL_RESULTS_DIR: Path = RESULTS_DIR / "symbol_expansion"
DEMO_RESULTS_DIR: Path = RESULTS_DIR / "demo_synthetic"     # --demo (synthetic glyph) outputs only
SINGLE_IMAGE_RESULTS_DIR: Path = RESULTS_DIR / "single_image"  # --input / GUI single-character outputs
MANIFEST_PATH: Path = DATA_DIR / "dataset_manifest.csv"
TRANSCRIPTIONS_PATH: Path = DATA_DIR / "transcriptions.csv"
WELDING_MANIFEST_PATH: Path = DATA_DIR / "welding_symbol_manifest.csv"
WELDING_RULES_PATH: Path = DATA_DIR / "welding_rules.json"
TRAINING_FEATURES_PATH: Path = DATA_DIR / "training_features.csv"
WELDING_FEATURES_PATH: Path = DATA_DIR / "welding_symbol_features.csv"
DATA_FITTED_MODEL_PATH: Path = PROTOTYPE_DIR / "data_fitted_model.json"
DATA_FITTED_TEMPLATES_PATH: Path = PROTOTYPE_DIR / "data_fitted_templates.npz"


@dataclass(frozen=True)
class DatasetConfig:
    """Real-dataset (Symbols/) manifest, label validation and character segmentation."""

    formal_dir: str = "Formal Text"
    informal_dir: str = "Informal Text"
    welding_dir: str = "Welding Symbol"
    formal_pattern: str = r"^formal_(\d+)_(\d+)$"        # group id, variant
    informal_pattern: str = r"^informal_(\d+)_(\d+)$"
    scene_pattern: str = r"^scene_(\d+)$"
    welding_pattern: str = r"^([a-z_]+?)_(\d+)$"          # symbol class, sample index
    # Comparison policy: the original transcription is kept verbatim; only the
    # normalized copy is compared with the segmentation count.
    space_policy: str = "ignore"            # "ignore": whitespace is not a character | "count"
    # Characters removed from the comparison copy. Empty by default: glyph arrows
    # ("← F6") are real segments; underline arrows are removed geometrically and
    # should simply not be typed.
    ignore_chars: str = ""
    # Segmentation (full-resolution binary mask of a marking string)
    segment_min_component_ratio: float = 0.004   # keep dots ('.', ':'): small relative to largest component
    segment_min_pixels: int = 6
    segment_merge_overlap: float = 0.5     # x-overlap / narrower width -> same character ('=', ':', '%', 'i')
    line_min_aspect: float = 4.0           # wide thin components (underlines / drawn arrows) ...
    line_min_width_ratio: float = 1.6      # ... wider than this x median character width are excluded
    background_kernel_ratio: float = 0.35  # median-blur background kernel / min(H, W)
    peak_percentile: float = 99.7          # peak stroke response used for polarity + threshold
    stroke_threshold_ratio: float = 0.45   # ink = response > ratio x peak
    segment_contrast_ratio: float = 0.5   # component ink contrast must be >= ratio x strongest (rust is soft)
    segment_ring_px: int = 3               # background ring width for the contrast measure
    segment_band_height_ratio: float = 0.5  # components this tall (vs tallest) define the text-line band
    min_samples_per_class: int = 3         # classes with fewer real samples are left out of the model
    preview_max_per_run: int = 400


@dataclass(frozen=True)
class RealExperimentConfig:
    """REAL SYMBOLS experiment. All values fixed BEFORE evaluation (baseline; never tuned on test accuracy)."""

    # Welding drawing ROI (content/position rule identical for every image; never uses the label)
    welding_ink_threshold: int = 60          # drawing ink ~25, caption text ~70-120, warp border fill ~115-135
    caption_band_ratio: float = 0.22         # components entirely above this fraction of H are caption text
    welding_canvas: int = 128                # line drawings need a larger canvas than characters
    welding_box: int = 120
    # Bayesian (per-fold reference statistics only)
    w_topology: float = 0.5
    w_geometry: float = 0.5
    variance_shrinkage_lambda: float = 2.0   # pseudo-count toward pooled within-class variance
    sigma_floor_relative: float = 0.05       # x reference-fold std of the feature
    sigma_floor_absolute: float = 1e-3       # for zero-variance features
    top_k: int = 3
    # OOD probe
    ood_percentile: float = 95.0             # of out-of-fold welding distances
    # Character segment diagnostics
    touching_width_ratio: float = 1.5        # segment width > ratio x median height -> possible touching chars
    small_height_ratio: float = 0.4          # segment height < ratio x median height -> punctuation / dot
    # Scene candidate extraction (qualitative; no ground truth)
    scene_background_kernel: int = 31
    scene_threshold_ratio: float = 0.35
    scene_line_aspect: float = 6.0           # thin long components = clutter lines
    scene_merge_kernel: Tuple[int, int] = (21, 7)
    scene_min_area_ratio: float = 0.0004
    scene_max_area_ratio: float = 0.08


@dataclass(frozen=True)
class CharacterExperimentConfig:
    """Character experiment protocol - PRE-DECLARED before any result was seen; never tuned on test accuracy."""

    w_topology: float = 0.5                  # combined model: group-mean log-likelihood weights
    w_geometry: float = 0.5
    min_train_samples_per_class: int = 1     # a test sample is evaluable iff its class occurs in the training fold
    bootstrap_iterations: int = 2000         # group-level (paired) bootstrap for 95% CI
    bootstrap_seed: int = 42
    ci_unstable_min_groups: int = 20         # fewer test groups -> CI flagged as unstable
    large_difference: float = 0.20           # per-class "large" difference (fraction) ...
    large_difference_min_n: int = 5          # ... only reported for classes with >= this many test samples


@dataclass(frozen=True)
class DegradationExperimentConfig:
    """Degradation robustness protocol. ALL parameters fixed before the first run; never tuned on results.

    Severity index 0..4; level 0 is always the identity (clean). Sizes are relative to the character's
    own foreground (stroke width / bounding box / contrast), never to its class.
    """

    seed: int = 42
    replicates: int = 3                       # per sample x stochastic degradation x severity
    blur_replicates: int = 1                  # Gaussian blur is deterministic
    pad_ratio: float = 0.6                    # canvas padding = ratio x foreground diagonal (rotation/perspective)
    read_noise_sigma: float = 2.0             # camera read noise (grey levels) for low light / contrast, s>=1
    # severity parameters (index = severity)
    blur_sigma_x_stroke: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.8, 1.2)       # sigma = k * stroke width
    low_light_factor: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.12, 0.06)        # exposure scale
    noise_sigma_x_contrast: Tuple[float, ...] = (0.0, 0.15, 0.3, 0.5, 0.8)    # sigma = k * |ink - bg|
    contrast_factor: Tuple[float, ...] = (1.0, 0.5, 0.3, 0.15, 0.08)          # compression around mean
    stroke_cuts: Tuple[int, ...] = (0, 1, 2, 3, 4)                            # number of stroke gaps
    stroke_cut_radius_x_stroke: Tuple[float, ...] = (0.0, 0.6, 0.8, 1.0, 1.2)  # gap radius = k * stroke width
    occlusion_area_fraction: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.30)  # of foreground bbox area
    rotation_deg: Tuple[float, ...] = (0.0, 5.0, 10.0, 20.0, 30.0)            # sign drawn from seed
    perspective_corner_shift: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.22)  # x foreground bbox size
    failure_drop: float = 0.20                # topology "failure point": first severity with drop >= 20 pp
    example_severity: int = 2                 # qualitative examples are taken at this severity
    bootstrap_iterations: int = 2000
    workers: int = 0                          # 0 = auto (cpu_count - 1, max 8)


@dataclass(frozen=True)
class UncertaintyValidationConfig:
    """Uncertainty / calibration validation protocol - fixed before the first run."""

    seed: int = 42
    outer_folds: int = 5                     # final-test folds (group-level)
    calibration_fraction: float = 0.25       # of outer-train groups -> calibration partition
    domain_shift_formal_parts: int = 4       # formal groups split into parts; one part = calibration (C)
    ece_bins: int = 15                       # primary ECE: equal-width bins on max posterior
    ece_bins_sensitivity: Tuple[int, ...] = (5, 10, 20)
    target_risk: float = 0.05                # selective classification target (calibration-chosen threshold)
    temperature_bounds: Tuple[float, float] = (0.05, 50.0)
    bootstrap_iterations: int = 1000         # group-level CIs on final-test metrics


@dataclass(frozen=True)
class MarkingRecognitionConfig:
    """Whole-marking (string) recognition check - fixed before the first run."""

    primary_model: str = "combined"          # existing 0.5/0.5 topology+geometry model, unchanged
    space_gap_ratio: float = 0.6             # gap > ratio x median char height -> POSSIBLE space (flag only)
    line_overlap_min: float = 0.3            # vertical overlap with the text-line band below this -> order flag
    thin_stroke_px: float = 2.0              # median stroke width <= this -> "thin stroke" audit category
    slant_deg: float = 12.0                  # |median segment slant| >= this -> "slanted" audit category
    examples_per_kind: int = 3               # deterministic example selection (lowest image path first)
    audit_images_per_category: int = 12


@dataclass(frozen=True)
class DirectionAwareConfig:
    """Exploratory direction / line-relative feature experiment - feature lists and weights fixed a priori."""

    direction_features: Tuple[str, ...] = (
        "hole_y", "hole_x", "endpoint_y_mean", "endpoint_x_mean", "branch_y_mean",
        "proj_row_top", "proj_row_bottom", "proj_col_left", "proj_col_right",
        "orient_h", "orient_d45", "orient_v", "orient_d135",
    )
    line_features: Tuple[str, ...] = (
        "rel_height", "rel_width", "baseline_offset", "vcenter_offset", "gap_left", "gap_right",
    )
    undefined_position: float = 0.5          # no hole / endpoint / branch point -> box centre
    bootstrap_iterations: int = 2000
    seed: int = 42


@dataclass(frozen=True)
class TextDetectionConfig:
    """Text detection integration - all values fixed before the first run (OpenCV-zoo defaults for the models)."""

    detector_file: str = "text_detection_en_ppocrv3_2023may.onnx"     # PaddleOCR PP-OCRv3 DBNet (ONNX, OpenCV zoo)
    recognizer_file: str = "text_recognition_CRNN_EN_2021sep.onnx"    # CRNN-CTC, charset 0-9a-z (OpenCV zoo)
    recognizer_charset: str = "0123456789abcdefghijklmnopqrstuvwxyz"
    det_max_side: int = 736
    det_binary_threshold: float = 0.3
    det_polygon_threshold: float = 0.5
    det_max_candidates: int = 200
    det_unclip_ratio: float = 2.0
    det_mean: Tuple[float, float, float] = (122.67891434, 116.66876762, 104.00698793)
    rec_size: Tuple[int, int] = (100, 32)
    crop_margin_ratio: float = 0.5          # B: axis-aligned region box expanded by this x box height per side
    same_line_overlap: float = 0.5          # regions on one line if vertical overlap > this x smaller height
    coverage_center_inside: bool = True     # proxy recall: validated char-box centre inside a detected region


@dataclass(frozen=True)
class SceneStabilizationConfig:
    """Crop stabilization - fixed in this implementation. NOTE: pad_ratio 2.0 was suggested by the post-hoc
    margin diagnostic on the same crops, so crop-level gains are NOT independent validation."""

    pad_ratio: float = 2.0                  # padding per side = ratio x text height (rotated-rect short side)
    tight_ratio: float = 0.5                # previous pre-declared margin, recomputed as the "before" condition
    rotation_threshold_deg: float = 2.0     # |region angle| >= this -> deskew around the region before cropping
    slant_handwriting_deg: float = 12.0     # characteristic flag for topology-only-correct analysis
    labels_file: str = "scene_labels.csv"   # data/<file>: manual scene transcriptions (user-entered only)
    examples_per_kind: int = 4


@dataclass(frozen=True)
class AttachConfig:
    """Generic 'mark above a stroke' attachment for components the text-band filter would drop.
    Fixed before evaluation; not specific to any character and never uses ground truth."""

    max_area_ratio: float = 0.25        # candidate area <= ratio x median area of kept components
    min_x_overlap: float = 0.3          # horizontal overlap >= ratio x candidate width
    max_gap_ratio: float = 0.6          # host top - candidate bottom <= ratio x median char height
    above_tolerance_ratio: float = 0.15  # candidate bottom must be <= host top + ratio x median height


@dataclass(frozen=True)
class SymbolExpansionConfig:
    """Symbol registry, new-data collection and independent-split rules - fixed a priori."""

    new_classes: Tuple[str, ...] = ("+", "←", "→", "↑", "↓")
    incoming_dir: str = "new_symbols/incoming"          # under data/
    manifest_file: str = "new_symbol_dataset_manifest.csv"  # under data/
    min_train_groups: int = 2           # a new class is trained only with >= this many TRAIN source groups
    min_train_samples: int = 3
    split_fractions: Tuple[float, float, float] = (0.6, 0.2, 0.2)   # auto split by source_group (train/dev/test)
    dhash_near_duplicate_bits: int = 20  # ink-pattern dHash (256 bits): Hamming <= this -> content reuse suspected
    seed: int = 42


@dataclass(frozen=True)
class GUIConfig:
    """Parameters of the PyQt6 GUI."""

    preview_max_side: int = 2048         # large drawings are previewed downscaled
    zoom_step: float = 1.15
    zoom_min: float = 0.05
    zoom_max: float = 60.0
    window_size: Tuple[int, int] = (1600, 950)


@dataclass(frozen=True)
class AppConfig:
    """Aggregate configuration object passed through the pipeline."""

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    prototypes: PrototypeConfig = field(default_factory=PrototypeConfig)
    restoration: RestorationConfig = field(default_factory=RestorationConfig)
    bayesian: BayesianConfig = field(default_factory=BayesianConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    real: RealExperimentConfig = field(default_factory=RealExperimentConfig)
    character: CharacterExperimentConfig = field(default_factory=CharacterExperimentConfig)
    degradation: DegradationExperimentConfig = field(default_factory=DegradationExperimentConfig)
    uncertainty: UncertaintyValidationConfig = field(default_factory=UncertaintyValidationConfig)
    marking: MarkingRecognitionConfig = field(default_factory=MarkingRecognitionConfig)
    direction: DirectionAwareConfig = field(default_factory=DirectionAwareConfig)
    text_detection: TextDetectionConfig = field(default_factory=TextDetectionConfig)
    scene: SceneStabilizationConfig = field(default_factory=SceneStabilizationConfig)
    attach: AttachConfig = field(default_factory=AttachConfig)
    symbols: SymbolExpansionConfig = field(default_factory=SymbolExpansionConfig)
    reference_model: str = "auto"   # "auto" (data-fitted if available) | "synthetic" | "data"
    classes: Tuple[str, ...] = tuple(CLASSES)


# Korean-capable font families searched (in order) for figures and the GUI.
# If none is installed, figures replace unsupported glyphs and the GUI uses English.
KOREAN_FONT_CANDIDATES: Tuple[str, ...] = (
    "Malgun Gothic", "NanumGothic", "NanumBarunGothic", "Gulim", "Dotum", "Batang",
    "AppleGothic", "Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Sans KR", "UnDotum",
)


def font_search_dirs() -> List[Path]:
    """Return directories that may contain TrueType fonts (Windows first)."""
    dirs: List[Path] = []
    windir = os.environ.get("WINDIR")
    if windir:
        dirs.append(Path(windir) / "Fonts")
    local = os.environ.get("LOCALAPPDATA")
    if local:
        dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
    try:
        import matplotlib

        dirs.append(Path(matplotlib.get_data_path()) / "fonts" / "ttf")
    except Exception:  # pragma: no cover - matplotlib is a requirement
        pass
    dirs.extend([Path("/usr/share/fonts"), Path("/Library/Fonts")])
    return [d for d in dirs if d.exists()]


DEFAULT_CONFIG = AppConfig()
