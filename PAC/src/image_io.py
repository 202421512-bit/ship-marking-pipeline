"""Image I/O, glyph rendering and synthetic demo generation.

Images are read with ``np.fromfile`` + ``cv2.imdecode`` so that Windows paths
containing spaces or non-ASCII characters work. Original images are never
modified; all derived images are written to separate files.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import (
    DEMO_DIR,
    SUPPORTED_EXTENSIONS,
    DemoConfig,
    font_search_dirs,
)


class ImageLoadError(RuntimeError):
    """Raised when an input image cannot be read."""


@dataclass
class LoadedImage:
    """An input image together with traceability metadata."""

    path: Path
    image: np.ndarray            # BGR or grayscale uint8, untouched original
    sha256: str
    width: int
    height: int
    channels: int


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file (for traceability)."""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ImageLoadError(f"Cannot hash file {path}: {exc}") from exc
    return digest.hexdigest()


def load_image(path: Path) -> LoadedImage:
    """Load an image file without modifying it.

    Args:
        path: Path to a .png/.jpg/.jpeg/.bmp file.

    Returns:
        LoadedImage with the decoded pixels and metadata.
    """
    path = Path(path)
    if not path.exists():
        raise ImageLoadError(f"Input image not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(
            f"Unsupported extension '{path.suffix}'. Supported: {SUPPORTED_EXTENSIONS}"
        )
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    except Exception as exc:
        raise ImageLoadError(f"Failed to decode {path}: {exc}") from exc
    if image is None:
        raise ImageLoadError(f"Failed to decode {path}")
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    channels = 1 if image.ndim == 2 else int(image.shape[2])
    return LoadedImage(
        path=path.resolve(),
        image=image,
        sha256=file_sha256(path),
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        channels=channels,
    )


def save_image(path: Path, image: np.ndarray) -> Path:
    """Save an image using ``cv2.imencode`` (unicode-safe on Windows)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.dtype == bool:
        image = image.astype(np.uint8) * 255
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise IOError(f"Failed to encode image for {path}")
    encoded.tofile(str(path))
    return path


def find_font(name: str) -> Optional[Path]:
    """Locate a TrueType font file by name in known font directories."""
    for directory in font_search_dirs():
        for candidate in (directory / name, directory / name.upper(), directory / name.lower()):
            if candidate.exists():
                return candidate
    return None


def fallback_font() -> Optional[Path]:
    """Return a portable fallback font (DejaVu Sans shipped with matplotlib)."""
    return find_font("DejaVuSans-Bold.ttf") or find_font("DejaVuSans.ttf")


def render_glyph(label: str, font_path: Optional[Path], canvas: int, font_size: int) -> np.ndarray:
    """Render a label as dark-on-white grayscale, centered on a square canvas.

    Args:
        label: Character(s) to draw.
        font_path: TrueType font file; ``None`` uses Pillow's default font.
        canvas: Side length of the output image in pixels.
        font_size: Font size in points.

    Returns:
        uint8 grayscale image (0 = ink, 255 = background).
    """
    try:
        font = (
            ImageFont.truetype(str(font_path), font_size)
            if font_path is not None
            else ImageFont.load_default()
        )
    except OSError:
        font = ImageFont.load_default()
    img = Image.new("L", (canvas, canvas), 255)
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    x = (canvas - (right - left)) / 2 - left
    y = (canvas - (bottom - top)) / 2 - top
    draw.text((x, y), label, fill=0, font=font)
    return np.asarray(img, dtype=np.uint8).copy()


def _rotate(gray: np.ndarray, angle: float, border: int) -> np.ndarray:
    """Rotate a grayscale image about its center with a constant border."""
    h, w = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def generate_demo_image(cfg: DemoConfig) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Create a clean prototype glyph and a field-like degraded version.

    Degradations (seeded, reproducible): stroke removal, partial break,
    slight rotation, Gaussian blur, uneven illumination, intensity variation,
    speckle noise and small noise components.

    Returns:
        (clean_gray, degraded_gray, description_dict)
    """
    rng = np.random.default_rng(cfg.seed)
    font_path = find_font(cfg.font) or fallback_font()
    clean = render_glyph(cfg.target, font_path, cfg.canvas, cfg.font_size)
    ink = clean < 128

    # Locate the stroke geometry so the damage lands on real strokes.
    ys, xs = np.nonzero(ink)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy = (y0 + y1) // 2
    damaged = ink.copy()
    # 1) Stroke removal: vertical gap through the right side of the upper loop.
    upper_row = y0 + (cy - y0) // 2
    damaged[upper_row - cfg.gap_width_px // 2: upper_row + (cfg.gap_width_px + 1) // 2,
            (x0 + x1) // 2: x1 + 1] = False
    # 2) Partial break: narrower gap through the left side of the lower loop.
    lower_row = cy + (y1 - cy) // 2
    damaged[lower_row - cfg.partial_break_px // 2: lower_row + (cfg.partial_break_px + 1) // 2,
            x0: (x0 + x1) // 2 - 6] = False
    # 3) Erode a few random bites from the outer contour (rust / wear).
    for _ in range(4):
        by = int(rng.integers(y0, y1))
        bx = int(rng.integers(x0, x1))
        cv2.circle(damaged.view(np.uint8), (bx, by), int(rng.integers(2, 4)), 0, -1)

    fg, bg = cfg.foreground_level, cfg.background_level
    gray = np.where(damaged, fg, bg).astype(np.float32)
    gray = _rotate(gray, cfg.rotation_deg, bg)
    gray = cv2.GaussianBlur(gray, (0, 0), cfg.blur_sigma)

    # Uneven illumination (darker top-left -> brighter bottom-right) + intensity variation.
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gradient = (xx / w + yy / h - 1.0) * cfg.illumination_gradient * 0.5
    low_freq = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 18)
    low_freq = low_freq / (np.abs(low_freq).max() + 1e-6) * 18.0
    gray = gray + gradient + low_freq

    # Small noise components (dirt / weld spatter).
    for _ in range(cfg.noise_blobs):
        cx, cyy = int(rng.integers(5, w - 5)), int(rng.integers(5, h - 5))
        radius = int(rng.integers(1, 3))
        cv2.circle(gray, (cx, cyy), radius, float(fg + rng.integers(-10, 25)), -1)

    # Speckle noise (multiplicative + additive).
    speckle = gray * rng.normal(0, 0.06, gray.shape) + rng.normal(0, cfg.speckle_std * 0.5, gray.shape)
    degraded = np.clip(gray + speckle, 0, 255).astype(np.uint8)

    description = {
        "target": cfg.target,
        "seed": cfg.seed,
        "font": font_path.name if font_path else "PIL default",
        "degradations": [
            f"stroke removal (gap {cfg.gap_width_px}px, upper loop)",
            f"partial break (gap {cfg.partial_break_px}px, lower loop)",
            "random contour bites",
            f"rotation {cfg.rotation_deg} deg",
            f"Gaussian blur sigma={cfg.blur_sigma}",
            f"uneven illumination +/-{cfg.illumination_gradient / 2:.0f}",
            "low-frequency intensity variation",
            f"{cfg.noise_blobs} small noise components",
            f"speckle noise std~{cfg.speckle_std}",
        ],
    }
    return clean, degraded, description


def write_demo_files(cfg: DemoConfig) -> Tuple[Path, dict]:
    """Generate the synthetic demo images under ``data/demo/`` only and return the degraded image path.

    Demo images are never written into ``data/input/`` (real input folder), so a synthetic
    glyph can never be mistaken for real data.
    """
    clean, degraded, description = generate_demo_image(cfg)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    clean_path = save_image(DEMO_DIR / f"demo_{cfg.target}_clean.png", clean)
    degraded_path = save_image(DEMO_DIR / f"demo_{cfg.target}_degraded.png", degraded)
    description["clean_path"] = str(clean_path)
    description["degraded_path"] = str(degraded_path)
    return degraded_path, description
