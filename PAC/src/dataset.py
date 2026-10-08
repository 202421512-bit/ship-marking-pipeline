"""Real dataset (Symbols/) audit, manifest, label validation and character segmentation.

Label policy (no guessing):
  * FORMAL / INFORMAL images carry NO class label in folder or filename. Their
    multi-character text must be typed by the user (``transcription``).
  * A transcription is accepted for training/evaluation only if the character
    segmentation of the image yields exactly as many characters as the
    normalized transcription (label_source=manual, validation_status=COUNT_MATCH).
  * Everything else stays in the manifest but is excluded automatically.
  * ``scene_*.png`` are full scenes (not crops): style=INFORMAL_SCENE, never used
    for training/evaluation; kept for future detection -> crop -> analysis demos.
  * Welding Symbol files are a separate labeled dataset (class from filename).

User input lives in two places; both are preserved on every re-run:
  * ``data/transcriptions.csv``  - one row per image group (type the text once).
  * ``data/dataset_manifest.csv`` column ``transcription_override`` - optional
    per-image override (takes precedence over the group value).
The manifest column ``transcription`` is OUTPUT ONLY (effective value); editing it has no effect.
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import (
    MANIFEST_PATH,
    PROJECT_ROOT,
    RESULTS_DIR,
    SYMBOLS_DIR,
    TRANSCRIPTIONS_PATH,
    WELDING_MANIFEST_PATH,
    WELDING_RULES_PATH,
    AppConfig,
    DatasetConfig,
    PreprocessingConfig,
)
from .image_io import ImageLoadError, load_image, save_image
from .preprocessing import denoise, remove_small_components, to_grayscale

FORMAL, INFORMAL, INFORMAL_SCENE, UNKNOWN = "FORMAL", "INFORMAL", "INFORMAL_SCENE", "UNKNOWN"

# validation_status values
NO_TRANSCRIPTION = "NO_TRANSCRIPTION"
COUNT_MATCH = "COUNT_MATCH"
COUNT_MISMATCH = "COUNT_MISMATCH"
SEGMENTATION_FAILED = "SEGMENTATION_FAILED"
EXCLUDED_SCENE = "EXCLUDED_SCENE"
UNRECOGNIZED_FILENAME = "UNRECOGNIZED_FILENAME"
IMAGE_READ_ERROR = "IMAGE_READ_ERROR"

MANIFEST_COLUMNS = [
    "image_path", "style", "group_id", "variant", "transcription_override", "transcription",
    "normalized_transcription",
    "label_source", "valid_for_training", "valid_for_evaluation", "segmentation_count",
    "expected_character_count", "validation_status", "transcription_origin",
    "excluded_line_components", "preview_path", "notes",
]
TRANSCRIPTION_COLUMNS = ["group_id", "style", "image_count", "example_image", "transcription", "notes"]
WELDING_COLUMNS = ["image_path", "dataset", "symbol_class", "sample_index", "label_source",
                   "valid_for_training", "valid_for_evaluation", "validation_status", "notes"]


# ---------------------------------------------------------------- CSV helpers
def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read a CSV written by this tool or re-saved by Excel (UTF-8 with/without BOM, or CP949)."""
    if not Path(path).exists():
        return []
    for encoding in ("utf-8-sig", "cp949"):
        try:
            with Path(path).open("r", encoding=encoding, newline="") as handle:
                return [dict(r) for r in csv.DictReader(handle)]
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode {path} (expected UTF-8 or CP949).")


def write_csv_rows(path: Path, columns: Sequence[str], rows: Sequence[Dict[str, object]]) -> Path:
    """Write CSV as UTF-8 with BOM so Excel shows Korean / symbols correctly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def rel(path: Path) -> str:
    """Project-relative POSIX-style path (stable manifest key across machines)."""
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).as_posix()


def image_key(path: str) -> str:
    """'<style folder>/<file name>' - stable when the Symbols folder is moved."""
    parts = Path(str(path).replace("\\", "/")).parts
    return "/".join(parts[-2:])


# ---------------------------------------------------------------- scanning
@dataclass
class ImageEntry:
    """One text image found under Symbols/."""

    path: Path
    style: str
    group_id: str
    variant: str
    status_hint: str = ""


def scan_text_images(symbols_dir: Path, cfg: DatasetConfig) -> List[ImageEntry]:
    """List Formal / Informal / scene images with style and group from the real filenames."""
    entries: List[ImageEntry] = []
    for folder, pattern, style, prefix in (
        (cfg.formal_dir, cfg.formal_pattern, FORMAL, "formal"),
        (cfg.informal_dir, cfg.informal_pattern, INFORMAL, "informal"),
    ):
        directory = symbols_dir / folder
        if not directory.exists():
            continue
        for path in sorted(p for p in directory.iterdir() if p.is_file()):
            stem = path.stem
            m = re.match(pattern, stem)
            s = re.match(cfg.scene_pattern, stem)
            if m:
                entries.append(ImageEntry(path, style, f"{prefix}_{m.group(1)}", m.group(2)))
            elif s and style == INFORMAL:
                entries.append(ImageEntry(path, INFORMAL_SCENE, f"scene_{s.group(1)}", "1"))
            else:
                entries.append(ImageEntry(path, UNKNOWN, stem, "", UNRECOGNIZED_FILENAME))
    return entries


# ---------------------------------------------------------------- transcription
def normalize_transcription(text: str, cfg: DatasetConfig) -> str:
    """Comparison copy of a transcription (the original is never modified).

    NFC-normalizes, removes ``ignore_chars`` (drawn arrows) and, with
    space_policy='ignore', all whitespace (spaces are gaps, not ink).
    """
    t = unicodedata.normalize("NFC", text or "").strip()
    t = "".join(c for c in t if c not in cfg.ignore_chars)
    if cfg.space_policy == "ignore":
        t = "".join(c for c in t if not c.isspace())
    return t


# ---------------------------------------------------------------- segmentation
@dataclass
class Segmentation:
    """Character segmentation of one marking-string image."""

    boxes: List[Tuple[int, int, int, int]]          # x0, y0, x1, y1 per character, left -> right
    char_masks: List[np.ndarray] = field(repr=False)  # bool crop per character
    excluded_lines: List[Tuple[int, int, int, int]]
    mask: np.ndarray = field(repr=False)
    polarity: str = ""

    @property
    def count(self) -> int:
        """Number of character segments."""
        return len(self.boxes)


def binarize_for_segmentation(image: np.ndarray, pcfg: PreprocessingConfig,
                              dcfg: DatasetConfig) -> Tuple[np.ndarray, np.ndarray, str]:
    """Background-subtracted stroke mask for marking strings (no CLAHE).

    CLAHE / border-based polarity fail on small or light text under strong
    illumination gradients. Here: denoise -> subtract a large median background ->
    polarity = sign with the stronger peak stroke response (crisp paint) ->
    threshold at a fraction of that peak.
    """
    gray = to_grayscale(image)
    den, _ = denoise(gray, pcfg)
    k = max(3, int(dcfg.background_kernel_ratio * min(den.shape)) | 1)
    background = cv2.medianBlur(den, k).astype(np.float32)
    residual = den.astype(np.float32) - background
    peak_light = float(np.percentile(residual, dcfg.peak_percentile))
    peak_dark = float(np.percentile(-residual, dcfg.peak_percentile))
    if peak_light > peak_dark:
        response, peak, polarity = residual, peak_light, "light_on_dark"
    else:
        response, peak, polarity = -residual, peak_dark, "dark_on_light"
    mask = response > dcfg.stroke_threshold_ratio * max(peak, 1e-6)
    mask, _ = remove_small_components(mask, dcfg.segment_min_component_ratio, dcfg.segment_min_pixels)
    return mask, den, polarity


def _filter_text_components(comps: List[tuple], labels: np.ndarray, gray: np.ndarray,
                            dcfg: DatasetConfig, return_band_dropped: bool = False):
    """Drop non-text blobs (rust, stains, background regions).

    1. Contrast: |mean ink grey - median grey of a thin surrounding ring|; painted
       strokes are crisp, rust / stains are soft -> keep >= ratio x strongest.
    2. Text-line band: tall components define the line's vertical span; components
       not overlapping that band (blobs above / below the text) are removed.
    """
    h_img, w_img = gray.shape
    g = gray.astype(np.float32)
    kernel = np.ones((2 * dcfg.segment_ring_px + 1,) * 2, np.uint8)
    contrast = []
    for lab, x, y, w, h in comps:
        r = dcfg.segment_ring_px
        y0, y1, x0, x1 = max(0, y - r), min(h_img, y + h + r), max(0, x - r), min(w_img, x + w + r)
        ink = labels[y0:y1, x0:x1] == lab
        ring = (cv2.dilate(ink.astype(np.uint8), kernel) > 0) & (labels[y0:y1, x0:x1] == 0)
        patch = g[y0:y1, x0:x1]
        contrast.append(abs(float(patch[ink].mean()) - float(np.median(patch[ring]))) if ring.any() else 0.0)
    strongest = max(contrast) if contrast else 0.0
    kept = [c for c, k in zip(comps, contrast) if k >= dcfg.segment_contrast_ratio * strongest]
    if not kept:
        return ([], []) if return_band_dropped else []
    tallest = max(c[4] for c in kept)
    tall = [c for c in kept if c[4] >= dcfg.segment_band_height_ratio * tallest]
    band_top = float(np.median([c[2] for c in tall]))
    band_bottom = float(np.median([c[2] + c[4] for c in tall]))
    in_band = [c for c in kept if c[2] < band_bottom and c[2] + c[4] > band_top]
    if return_band_dropped:
        return in_band, [c for c in kept if c not in in_band]
    return in_band


def _attach_marks_above(dropped: List[tuple], kept: List[tuple], areas: Dict[int, int],
                        acfg) -> Dict[int, int]:
    """Generic rule: a SMALL component dropped by the text-band filter that lies just ABOVE a kept
    component and overlaps it horizontally is attached to it (e.g. dots / accents above a stem).
    Returns {candidate label: host label}. Marks on or below the band (decimal points, hyphens) are never
    dropped by the band filter and so are never touched; no new standalone character is ever created."""
    if not kept or not dropped:
        return {}
    med_area = float(np.median([areas[c[0]] for c in kept]))
    med_h = float(np.median([c[4] for c in kept]))
    out = {}
    for lab, x, y, w, h in dropped:
        if areas[lab] > acfg.max_area_ratio * med_area:
            continue
        best, best_gap = None, None
        for hl, hx, hy, hw, hh in kept:
            overlap = min(x + w, hx + hw) - max(x, hx)
            if overlap < acfg.min_x_overlap * max(1, w):
                continue
            bottom = y + h
            if bottom > hy + acfg.above_tolerance_ratio * med_h:      # must lie above the host's top
                continue
            gap = hy - bottom
            if gap > acfg.max_gap_ratio * med_h:
                continue
            if best is None or gap < best_gap:
                best, best_gap = hl, gap
        if best is not None:
            out[lab] = best
    return out


def segment_characters(image: np.ndarray, pcfg: PreprocessingConfig, dcfg: DatasetConfig,
                       attach=None) -> Segmentation:
    """Split a marking string into characters with connected components.

    1. Same binarization as the single-image pipeline (lower noise ratio so dots survive).
    2. Wide, thin components (underlines / drawn arrows) are excluded.
    3. Components overlapping horizontally are merged into one character ('=', ':', '%', 'i').
    attach (AttachConfig or None): opt-in repair - small marks dropped by the text-band filter that
    sit just above a kept stroke are attached to it. None = baseline behaviour (unchanged).
    """
    mask, den, polarity = binarize_for_segmentation(image, pcfg, dcfg)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    comps = [(i, int(stats[i, 0]), int(stats[i, 1]), int(stats[i, 2]), int(stats[i, 3]))
             for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= dcfg.segment_min_pixels]
    if not comps:
        return Segmentation([], [], [], mask, polarity)
    forced: Dict[int, int] = {}
    if attach is None:
        comps = _filter_text_components(comps, labels, den, dcfg)
    else:
        kept, dropped = _filter_text_components(comps, labels, den, dcfg, return_band_dropped=True)
        forced = _attach_marks_above(dropped, kept, {i: int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)}, attach)
        comps = kept + [c for c in dropped if c[0] in forced]
    if not comps:
        return Segmentation([], [], [], mask, polarity)
    widths = [w for _, _, _, w, h in comps if w / max(h, 1) < dcfg.line_min_aspect] or [c[3] for c in comps]
    median_w = float(np.median(widths))
    chars, lines = [], []
    for c in comps:
        _, x, y, w, h = c
        if w / max(h, 1) >= dcfg.line_min_aspect and w >= dcfg.line_min_width_ratio * median_w:
            lines.append((x, y, x + w, y + h))
        else:
            chars.append(c)
    # union-find over horizontal overlap
    parent = list(range(len(chars)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(chars)):
        for j in range(i + 1, len(chars)):
            _, xi, _, wi, _ = chars[i]
            _, xj, _, wj, _ = chars[j]
            overlap = min(xi + wi, xj + wj) - max(xi, xj)
            if overlap > 0 and overlap / max(1, min(wi, wj)) >= dcfg.segment_merge_overlap:
                parent[find(i)] = find(j)
    if forced:  # attached marks join their host character
        idx = {c[0]: k for k, c in enumerate(chars)}
        for cand, host in forced.items():
            if cand in idx and host in idx:
                parent[find(idx[cand])] = find(idx[host])
    groups: Dict[int, List[tuple]] = defaultdict(list)
    for i, c in enumerate(chars):
        groups[find(i)].append(c)
    boxes, masks = [], []
    for members in sorted(groups.values(), key=lambda g: min(m[1] for m in g)):
        x0 = min(m[1] for m in members)
        y0 = min(m[2] for m in members)
        x1 = max(m[1] + m[3] for m in members)
        y1 = max(m[2] + m[4] for m in members)
        crop = np.isin(labels[y0:y1, x0:x1], [m[0] for m in members])
        boxes.append((x0, y0, x1, y1))
        masks.append(crop)
    return Segmentation(boxes, masks, lines, mask, polarity)


def save_segmentation_preview(image: np.ndarray, seg: Segmentation, path: Path, text: str = "") -> Path:
    """Draw numbered character boxes (green) and excluded lines (red) for manual checking."""
    vis = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for k, (x0, y0, x1, y1) in enumerate(seg.boxes):
        cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (40, 200, 40), 2)
        cv2.putText(vis, str(k + 1), (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 200, 40), 1)
    for x0, y0, x1, y1 in seg.excluded_lines:
        cv2.rectangle(vis, (x0, y0), (x1 - 1, y1 - 1), (40, 40, 220), 1)
    banner = np.full((26, vis.shape[1], 3), 255, np.uint8)
    label = f"segments={seg.count}  excluded_lines={len(seg.excluded_lines)}"
    if text:
        label += f"  expected={len(text)}"
    cv2.putText(banner, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return save_image(path, np.vstack([banner, vis]))


# ---------------------------------------------------------------- manifest
@dataclass
class AuditResult:
    """Everything produced by one dataset audit."""

    manifest: List[Dict[str, object]]
    welding: List[Dict[str, object]]
    summary: Dict[str, object]
    segmentations: Dict[str, Segmentation] = field(default_factory=dict, repr=False)


def _load_user_transcriptions() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Per-image overrides (manifest), per-group values and group notes (transcriptions.csv)."""
    # Only the user-owned column counts as a per-image override. The 'transcription'
    # column is tool output (override or group value); reading it back would turn
    # copied group values into permanent overrides. Legacy manifests without the
    # column therefore contribute no overrides.
    per_image = {image_key(r.get("image_path", "")): (r.get("transcription_override") or "")
                 for r in read_csv_rows(MANIFEST_PATH) if (r.get("transcription_override") or "").strip()}
    group_rows = read_csv_rows(TRANSCRIPTIONS_PATH)
    per_group = {r.get("group_id", ""): (r.get("transcription") or "")
                 for r in group_rows if (r.get("transcription") or "").strip()}
    notes = {r.get("group_id", ""): (r.get("notes") or "") for r in group_rows}
    return per_image, per_group, notes


def _validate(entry: ImageEntry, transcription: str, origin: str, seg: Optional[Segmentation],
              error: str, cfg: DatasetConfig) -> Dict[str, object]:
    """Compute label_source / validity / status for one manifest row."""
    norm = normalize_transcription(transcription, cfg)
    row: Dict[str, object] = {
        "image_path": rel(entry.path), "style": entry.style, "group_id": entry.group_id,
        "variant": entry.variant, "transcription": transcription, "normalized_transcription": norm,
        "label_source": "manual" if transcription.strip() else "none",
        "valid_for_training": False, "valid_for_evaluation": False,
        "segmentation_count": "" if seg is None else seg.count,
        "expected_character_count": len(norm) if transcription.strip() else "",
        "transcription_origin": origin,
        "excluded_line_components": "" if seg is None else len(seg.excluded_lines),
        "notes": "",
    }
    if entry.style == INFORMAL_SCENE:
        row["validation_status"] = EXCLUDED_SCENE
        row["notes"] = "full 900x600 scene, not a crop; kept for future detection->crop->analysis demos"
    elif entry.status_hint == UNRECOGNIZED_FILENAME:
        row["validation_status"] = UNRECOGNIZED_FILENAME
    elif error:
        row["validation_status"] = IMAGE_READ_ERROR if "load" in error.lower() else SEGMENTATION_FAILED
        row["notes"] = error[:200]
    elif not transcription.strip():
        row["validation_status"] = NO_TRANSCRIPTION
    elif seg is None or seg.count == 0:
        row["validation_status"] = SEGMENTATION_FAILED
    elif seg.count == len(norm):
        row["validation_status"] = COUNT_MATCH
        row["valid_for_training"] = row["valid_for_evaluation"] = True
    else:
        row["validation_status"] = COUNT_MISMATCH
        row["notes"] = f"segments={seg.count} != characters={len(norm)}; excluded (check preview)"
    return row


def build_welding_dataset(symbols_dir: Path, cfg: DatasetConfig) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Separate labeled dataset: class from filename ``<class>_<index>.png``; rules stay unparsed."""
    directory = symbols_dir / cfg.welding_dir
    rows: List[Dict[str, object]] = []
    if directory.exists():
        for path in sorted(p for p in directory.iterdir() if p.is_file()):
            m = re.match(cfg.welding_pattern, path.stem)
            ok = bool(m) and path.suffix.lower() == ".png"
            rows.append({
                "image_path": rel(path), "dataset": "welding_symbol",
                "symbol_class": m.group(1) if m else "", "sample_index": m.group(2) if m else "",
                "label_source": "filename" if ok else "none",
                "valid_for_training": ok, "valid_for_evaluation": ok,
                "validation_status": "FILENAME_LABEL" if ok else UNRECOGNIZED_FILENAME,
                "notes": "drawing contains a caption with the class name (label leakage risk for image models)"
                         if ok else "filename does not match <class>_<index>",
            })
    by_class: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        if r["valid_for_training"]:
            by_class[str(r["symbol_class"])].append(str(r["image_path"]))
    rules = {
        "status": "unparsed",
        "reason": ("Welding Symbol folder contains only PNG symbol drawings; no regulation / rule text "
                   "(PDF, TXT, DOCX, CSV) was found. Meanings and rules are NOT generated."),
        "symbols": [{
            "symbol": cls, "display_name": cls.replace("_", " "),
            "meaning": None, "rule": None, "status": "unparsed",
            "source_files": files, "source_location": "filename (class label only)",
        } for cls, files in sorted(by_class.items())],
    }
    return rows, rules


def run_dataset_audit(cfg: AppConfig, symbols_dir: Path = SYMBOLS_DIR, write: bool = True,
                      previews: bool = True, verbose: bool = True) -> AuditResult:
    """Scan Symbols/, segment every text image, validate user transcriptions and write outputs.

    Safe to run any number of times: user transcriptions are read back first and preserved.
    """
    d = cfg.dataset
    per_image, per_group, group_notes = _load_user_transcriptions()
    entries = scan_text_images(symbols_dir, d)
    preview_dir = RESULTS_DIR / "segmentation_preview"
    manifest: List[Dict[str, object]] = []
    segs: Dict[str, Segmentation] = {}
    for e in entries:
        key = rel(e.path)
        if image_key(key) in per_image:
            text, origin = per_image[image_key(key)], "image"
        elif e.group_id in per_group and e.style in (FORMAL, INFORMAL):
            text, origin = per_group[e.group_id], "group"
        else:
            text, origin = "", ""
        seg, error = None, ""
        if e.style in (FORMAL, INFORMAL):
            try:
                seg = segment_characters(load_image(e.path).image, cfg.preprocessing, d)
                segs[key] = seg
            except ImageLoadError as exc:
                error = f"load: {exc}"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        row = _validate(e, text, origin, seg, error, d)
        row["transcription_override"] = per_image.get(image_key(key), "")
        if previews and seg is not None:
            try:
                p = save_segmentation_preview(load_image(e.path).image, seg,
                                              preview_dir / f"{e.group_id}_{e.variant}.png",
                                              str(row["normalized_transcription"]))
                row["preview_path"] = rel(p)
            except Exception as exc:
                row["notes"] = f"{row['notes']} preview failed: {exc}".strip()
        manifest.append(row)

    welding_rows, rules = build_welding_dataset(symbols_dir, d)
    summary = summarize(manifest, welding_rows, symbols_dir, cfg)
    if write:
        write_csv_rows(MANIFEST_PATH, MANIFEST_COLUMNS, manifest)
        _write_transcription_template(entries, per_group, group_notes)
        write_csv_rows(WELDING_MANIFEST_PATH, WELDING_COLUMNS, welding_rows)
        WELDING_RULES_PATH.write_text(json.dumps(rules, indent=2, ensure_ascii=False), encoding="utf-8")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "dataset_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if verbose:
        print(format_summary(summary))
    return AuditResult(manifest=manifest, welding=welding_rows, summary=summary, segmentations=segs)


def _write_transcription_template(entries: Sequence[ImageEntry], per_group: Dict[str, str],
                                  notes: Dict[str, str]) -> None:
    """One row per FORMAL / INFORMAL group; existing user text and notes are kept."""
    groups: Dict[str, List[ImageEntry]] = defaultdict(list)
    for e in entries:
        if e.style in (FORMAL, INFORMAL):
            groups[e.group_id].append(e)
    rows = [{
        "group_id": gid, "style": members[0].style, "image_count": len(members),
        "example_image": rel(members[0].path), "transcription": per_group.get(gid, ""),
        "notes": notes.get(gid, ""),
    } for gid, members in sorted(groups.items())]
    write_csv_rows(TRANSCRIPTIONS_PATH, TRANSCRIPTION_COLUMNS, rows)


def summarize(manifest: Sequence[Dict[str, object]], welding: Sequence[Dict[str, object]],
              symbols_dir: Path, cfg: AppConfig) -> Dict[str, object]:
    """Counts for dataset_summary.json and the terminal report."""
    all_files = [p for p in symbols_dir.rglob("*") if p.is_file()] if symbols_dir.exists() else []
    styles = Counter(str(r["style"]) for r in manifest)
    status = Counter(str(r["validation_status"]) for r in manifest)
    labeled = [r for r in manifest if r["label_source"] == "manual"]
    trainable = [r for r in manifest if r["valid_for_training"] is True]
    groups = defaultdict(set)
    for r in manifest:
        groups[str(r["style"])].add(str(r["group_id"]))
    char_counts = Counter(c for r in trainable for c in str(r["normalized_transcription"]))
    weld_classes = Counter(str(r["symbol_class"]) for r in welding if r["valid_for_training"])
    return {
        "symbols_dir": rel(symbols_dir),
        "total_files": len(all_files),
        "extensions": dict(Counter(p.suffix.lower() for p in all_files)),
        "text_images": {
            "FORMAL": styles.get(FORMAL, 0), "INFORMAL": styles.get(INFORMAL, 0),
            "INFORMAL_SCENE": styles.get(INFORMAL_SCENE, 0), "UNKNOWN": styles.get(UNKNOWN, 0),
        },
        "text_groups": {k: len(v) for k, v in groups.items()},
        "label_policy": {
            "class_from_folder": False, "class_from_filename": False,
            "source": "manual transcription (data/transcriptions.csv or manifest per-image override)",
            "space_policy": cfg.dataset.space_policy, "ignore_chars": cfg.dataset.ignore_chars,
        },
        "labeled_images": len(labeled),
        "trainable_images": len(trainable),
        "trainable_by_style": dict(Counter(str(r["style"]) for r in trainable)),
        "trainable_characters": sum(char_counts.values()),
        "trainable_character_classes": dict(sorted(char_counts.items())),
        "validation_status": dict(status),
        "welding_symbol": {
            "images": len(welding), "labeled_images": sum(1 for r in welding if r["valid_for_training"]),
            "classes": len(weld_classes), "per_class": dict(sorted(weld_classes.items())),
            "label_source": "filename", "rules_status": "unparsed (no rule text in source files)",
        },
        "outputs": {"manifest": rel(MANIFEST_PATH), "transcriptions": rel(TRANSCRIPTIONS_PATH),
                    "welding_manifest": rel(WELDING_MANIFEST_PATH), "welding_rules": rel(WELDING_RULES_PATH),
                    "segmentation_previews": rel(RESULTS_DIR / "segmentation_preview")},
    }


def format_summary(s: Dict[str, object]) -> str:
    """Human-readable dataset audit report."""
    t = s["text_images"]
    w = s["welding_symbol"]
    lines = [
        "=" * 50, "DATASET AUDIT (Real Dataset: Symbols/)", "=" * 50,
        f"Symbols dir            = {s['symbols_dir']}",
        f"Total files            = {s['total_files']}  {s['extensions']}",
        f"Formal images          = {t['FORMAL']}  (groups {s['text_groups'].get('FORMAL', 0)})",
        f"Informal images        = {t['INFORMAL']}  (groups {s['text_groups'].get('INFORMAL', 0)})",
        f"Informal scene images  = {t['INFORMAL_SCENE']}  (excluded from training/evaluation)",
        f"Unrecognized filenames = {t['UNKNOWN']}",
        f"Welding symbol images  = {w['images']}  ({w['classes']} classes, label from filename)",
        "",
        f"Labeled text images    = {s['labeled_images']}  (manual transcription)",
        f"Trainable text images  = {s['trainable_images']}  {s['trainable_by_style']}",
        f"Trainable characters   = {s['trainable_characters']}  in {len(s['trainable_character_classes'])} classes",
        f"Validation status      = {s['validation_status']}",
        f"Welding labeled        = {w['labeled_images']}  rules: {w['rules_status']}",
        "",
        f"Edit transcriptions in : {s['outputs']['transcriptions']}  "
        f"(or per image: column transcription_override in {s['outputs']['manifest']})",
        f"Check segmentation in  : {s['outputs']['segmentation_previews']}",
        "Then re-run            : python Main.py --train-reference",
        "=" * 50,
    ]
    if s["trainable_images"] == 0:
        lines.insert(-1, "NOTE: no trainable text data yet -> reference model stays SYNTHETIC (font prototypes).")
    return "\n".join(lines)
