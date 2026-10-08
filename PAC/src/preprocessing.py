"""Image preprocessing: grayscale -> normalization -> CLAHE -> denoise ->
threshold -> polarity -> noise removal -> (closing) -> size normalization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .config import PreprocessingConfig


@dataclass
class PreprocessingResult:
    """All intermediate images and metadata of the preprocessing stage."""

    stages: Dict[str, np.ndarray]           # name -> image (for visualization)
    mask: np.ndarray                         # bool, normalized canvas, True = ink
    soft: np.ndarray                         # float32 [0,1] ink evidence, normalized canvas
    gray_normalized: np.ndarray              # uint8 grayscale crop on normalized canvas
    info: Dict[str, object] = field(default_factory=dict)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR / grayscale input to single-channel uint8."""
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_intensity(gray: np.ndarray, cfg: PreprocessingConfig) -> np.ndarray:
    """Correct uneven illumination by background division, then percentile stretch."""
    sigma = max(2.0, cfg.illumination_sigma_ratio * max(gray.shape))
    background = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma) + 1.0
    flat = gray.astype(np.float32) / background
    lo, hi = np.percentile(flat, [cfg.stretch_low_percentile, cfg.stretch_high_percentile])
    stretched = (flat - lo) / max(float(hi - lo), 1e-6) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """Robust noise std estimate (MAD of the Laplacian response)."""
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)
    return float(np.median(np.abs(lap - np.median(lap))) * 1.4826 / np.sqrt(20.0))


def contrast_is_low(gray: np.ndarray, cfg: PreprocessingConfig) -> bool:
    """Low-light / low-contrast test used by the 'auto' CLAHE mode."""
    lo, hi = np.percentile(gray, [5, 95])
    return float(hi - lo) < cfg.clahe_low_contrast_range


def apply_clahe(gray: np.ndarray, cfg: PreprocessingConfig) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization (optional)."""
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip_limit,
                            tileGridSize=(cfg.clahe_tile_grid, cfg.clahe_tile_grid))
    return clahe.apply(gray)


def denoise(gray: np.ndarray, cfg: PreprocessingConfig) -> Tuple[np.ndarray, float]:
    """Median + non-local means denoising with noise-adaptive strength."""
    med = cv2.medianBlur(gray, cfg.median_kernel)
    sigma = estimate_noise_sigma(gray)
    h = float(np.clip(cfg.denoise_strength_per_sigma * sigma, cfg.denoise_strength_min, cfg.denoise_strength_max))
    out = cv2.fastNlMeansDenoising(med, None, h=h,
                                   templateWindowSize=cfg.denoise_template_window,
                                   searchWindowSize=cfg.denoise_search_window)
    return out, h


def threshold(gray: np.ndarray, cfg: PreprocessingConfig) -> Tuple[np.ndarray, float]:
    """Binarize with Otsu or adaptive threshold. Returns (binary 0/255, threshold)."""
    if cfg.threshold_method == "adaptive":
        block = int(cfg.adaptive_block_ratio * min(gray.shape)) | 1
        block = max(block, 3)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block, cfg.adaptive_c)
        return binary, float("nan")
    thr, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, float(thr)


def decide_polarity(binary: np.ndarray, cfg: PreprocessingConfig) -> Tuple[np.ndarray, str]:
    """Return a bool ink mask, choosing polarity so the border is background."""
    h, w = binary.shape
    m = max(1, int(cfg.border_margin_ratio * min(h, w)))
    border = np.concatenate([binary[:m].ravel(), binary[-m:].ravel(),
                             binary[:, :m].ravel(), binary[:, -m:].ravel()])
    white_border = float(np.mean(border > 0))
    if white_border >= 0.5:
        return binary == 0, "dark_on_light"
    return binary > 0, "light_on_dark"


def remove_small_components(mask: np.ndarray, ratio: float, min_pixels: int) -> Tuple[np.ndarray, int]:
    """Drop 8-connected components smaller than ratio*largest or min_pixels."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask.copy(), 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    limit = max(min_pixels, ratio * float(areas.max()))
    keep = np.zeros(n, dtype=bool)
    keep[1:] = areas >= limit
    return keep[labels], int(np.sum(~keep[1:]))


def normalize_size(mask: np.ndarray, soft: np.ndarray, gray: np.ndarray,
                   cfg: PreprocessingConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Crop to the ink bounding box and fit it, aspect preserved, into a square canvas."""
    size, box = cfg.normalized_size, cfg.glyph_box
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("No foreground found after preprocessing.")
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    h, w = y1 - y0, x1 - x0
    scale = box / float(max(h, w))
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    oy, ox = (size - nh) // 2, (size - nw) // 2

    def place(img: np.ndarray, fill: float, interp: int) -> np.ndarray:
        crop = cv2.resize(img[y0:y1, x0:x1].astype(np.float32), (nw, nh), interpolation=interp)
        canvas = np.full((size, size), fill, dtype=np.float32)
        canvas[oy:oy + nh, ox:ox + nw] = crop
        return canvas

    mask_n = place(mask, 0.0, cv2.INTER_AREA) >= cfg.resize_binarize_threshold
    soft_n = np.clip(place(soft, 0.0, cv2.INTER_AREA), 0.0, 1.0)
    gray_n = np.clip(place(gray, float(np.median(gray)), cv2.INTER_AREA), 0, 255).astype(np.uint8)
    info = {"bbox_xyxy": [x0, y0, x1, y1], "bbox_width": w, "bbox_height": h,
            "scale": round(scale, 5), "canvas": size}
    return mask_n, soft_n, gray_n, info


def ink_evidence(gray: np.ndarray, thr: float, polarity: str) -> np.ndarray:
    """Soft ink probability map from the denoised grayscale (sigmoid around threshold)."""
    g = gray.astype(np.float32)
    if np.isnan(thr):
        thr = float(np.mean(g))
    spread = max(4.0, float(np.std(g)) * 0.25)
    z = (thr - g) / spread if polarity == "dark_on_light" else (g - thr) / spread
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


@dataclass
class BinarizationResult:
    """Full-resolution binarization (steps 1-7), shared by single-image and dataset analysis."""

    gray: np.ndarray
    enhanced: np.ndarray        # grayscale used for thresholding
    mask: np.ndarray            # bool ink mask after noise removal (full resolution)
    threshold: float
    polarity: str
    removed_components: int
    h_used: float
    clahe_applied: bool
    stages: Dict[str, np.ndarray]


def binarize(image: np.ndarray, cfg: PreprocessingConfig,
             min_component_ratio: Optional[float] = None) -> BinarizationResult:
    """grayscale -> denoise -> illumination normalization -> CLAHE -> threshold -> polarity -> noise removal."""
    stages: Dict[str, np.ndarray] = {}
    gray = to_grayscale(image)
    stages["1_grayscale"] = gray
    denoised, h_used = denoise(gray, cfg)
    stages["2_denoised"] = denoised
    norm = normalize_intensity(denoised, cfg)
    stages["3_normalized"] = norm
    use_clahe = cfg.clahe_mode == "on" or (cfg.clahe_mode == "auto" and contrast_is_low(gray, cfg))
    den = apply_clahe(norm, cfg) if use_clahe else norm
    stages["4_clahe" if use_clahe else "4_clahe_skipped"] = den
    binary, thr = threshold(den, cfg)
    stages["5_threshold"] = binary
    mask, polarity = decide_polarity(binary, cfg)
    stages["6_foreground"] = mask.astype(np.uint8) * 255
    ratio = cfg.min_component_ratio if min_component_ratio is None else min_component_ratio
    cleaned, removed = remove_small_components(mask, ratio, cfg.min_component_pixels)
    stages["7_noise_removed"] = cleaned.astype(np.uint8) * 255
    return BinarizationResult(gray=gray, enhanced=den, mask=cleaned, threshold=thr, polarity=polarity,
                              removed_components=removed, h_used=h_used, clahe_applied=use_clahe,
                              stages=stages)


def preprocess(image: np.ndarray, cfg: PreprocessingConfig) -> PreprocessingResult:
    """Run the full preprocessing chain on an untouched input image.

    Args:
        image: Original BGR or grayscale uint8 image (not modified).
        cfg: Preprocessing parameters.

    Returns:
        PreprocessingResult with intermediate stages and the normalized mask.
    """
    b = binarize(image, cfg)
    stages, gray, den, thr, polarity = b.stages, b.gray, b.enhanced, b.threshold, b.polarity
    cleaned, removed, h_used, use_clahe = b.mask, b.removed_components, b.h_used, b.clahe_applied
    if cfg.use_closing:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.closing_kernel, cfg.closing_kernel))
        cleaned = cv2.morphologyEx(cleaned.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
        stages["8_closing"] = cleaned.astype(np.uint8) * 255
    soft = ink_evidence(den, thr, polarity)
    mask_n, soft_n, gray_n, size_info = normalize_size(cleaned, soft, den, cfg)
    stages["9_size_normalized"] = mask_n.astype(np.uint8) * 255

    info: Dict[str, object] = {
        "input_shape": list(image.shape),
        "threshold_method": cfg.threshold_method,
        "otsu_threshold": None if np.isnan(thr) else round(thr, 2),
        "polarity": polarity,
        "clahe_mode": cfg.clahe_mode,
        "clahe_applied": use_clahe,
        "noise_sigma_estimate": round(estimate_noise_sigma(gray), 3),
        "nlmeans_h": round(h_used, 3),
        "order": "grayscale -> denoise -> illumination normalization -> CLAHE(optional) -> threshold",
        "closing": cfg.use_closing,
        "removed_noise_components": removed,
        "foreground_pixels_normalized": int(mask_n.sum()),
        "size_normalization": size_info,
        "steps": list(stages.keys()),
    }
    return PreprocessingResult(stages=stages, mask=mask_n, soft=soft_n,
                               gray_normalized=gray_n, info=info)


def mask_from_glyph(gray: np.ndarray, threshold_value: int, cfg: PreprocessingConfig) -> np.ndarray:
    """Fast path for clean prototype glyphs: threshold + size normalization only."""
    mask = gray < threshold_value
    if not mask.any():
        raise ValueError("Empty glyph.")
    mask, _ = remove_small_components(mask, cfg.min_component_ratio, cfg.min_component_pixels)
    soft = mask.astype(np.float32)
    mask_n, _, _, _ = normalize_size(mask, soft, gray, cfg)
    return mask_n
