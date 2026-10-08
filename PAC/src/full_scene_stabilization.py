"""FULL-SCENE RECOGNITION STABILIZATION.

Image -> PP-OCRv3 DB detection (polygon kept) -> rotation-aware PADDED crop (pad proportional to text
height, clipped to the image, original coordinates kept) -> existing segmentation -> existing
topology + geometry + 13 direction features -> Gaussian group-mean classifier -> string.

* "before" = tight crop (previous margin 0.5), "after" = padded crop (2.0 x text height). The padding
  value was suggested by an earlier POST-HOC diagnostic on the same crops: crop-level improvement is
  NOT independent validation. Only newly labelled scenes can give an independent check.
* Scene ground truth comes ONLY from data/scene_labels.csv (entered by a person via --label-scenes);
  it is never passed to detection, OCR or the classifier, and nothing is auto-labelled.
* OCR (CRNN) and topology outputs are compared, never fused.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import DATA_DIR, PROJECT_ROOT, SCENE_RESULTS_DIR, AppConfig
from .dataset import FORMAL, INFORMAL, INFORMAL_SCENE, normalize_transcription, read_csv_rows, run_dataset_audit
from .direction_aware_experiment import fit_group_model
from .image_io import load_image
from .real_experiment import GEO_FEATURES, TOPO_FEATURES, write_csv
from .real_marking_recognition import edit_ops
from .text_detection_integration import (Detector, Recognizer, build_pool, casefold_alnum, order_regions,
                                         segment_features)

CRNN_SUPPORTED = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")


# ====================================================================== padded crop (rotation aware)
@dataclass
class Crop:
    """A crop of the original image plus the affine map back to original coordinates."""

    image: np.ndarray = field(repr=False)
    to_original: np.ndarray = field(repr=False)      # 2x3: crop (x, y, 1) -> original (x, y)
    angle: float
    text_height: float
    pad_px: float
    clipped: bool                                    # padding reduced by the image border
    rotated: bool


def region_geometry(poly: np.ndarray) -> Tuple[Tuple[float, float], float, float, float]:
    """(centre, width, height, angle) of the region's minimum-area rectangle, width >= height, |angle| <= 45."""
    (cx, cy), (w, h), a = cv2.minAreaRect(poly.astype(np.float32))
    if w < h:
        w, h, a = h, w, a + 90.0
    a = ((a + 90.0) % 180.0) - 90.0
    if a > 45:
        a -= 90
    elif a < -45:
        a += 90
    return (cx, cy), w, h, a


def padded_crop(img: np.ndarray, poly: np.ndarray, ratio: float, rot_thr: float) -> Crop:
    """Crop the region with padding = ratio x text height on every side; deskew rotated regions first."""
    H, W = img.shape[:2]
    (cx, cy), w, h, a = region_geometry(poly)
    pad = ratio * max(1.0, h)
    rotated = abs(a) >= rot_thr
    if rotated:
        M = cv2.getRotationMatrix2D((cx, cy), a, 1.0)          # rotate so the text line is horizontal
        src = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        x0, x1 = cx - w / 2 - pad, cx + w / 2 + pad
        y0, y1 = cy - h / 2 - pad, cy + h / 2 + pad
        Minv = cv2.invertAffineTransform(M)
    else:
        src = img
        x0, y0 = poly[:, 0].min() - pad, poly[:, 1].min() - pad
        x1, y1 = poly[:, 0].max() + pad, poly[:, 1].max() + pad
        Minv = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    cx0, cy0 = int(max(0, math.floor(x0))), int(max(0, math.floor(y0)))
    cx1, cy1 = int(min(W, math.ceil(x1))), int(min(H, math.ceil(y1)))
    clipped = cx0 > x0 + 0.5 or cy0 > y0 + 0.5 or cx1 < x1 - 0.5 or cy1 < y1 - 0.5
    # crop pixel (u, v) -> rotated-frame (u + cx0, v + cy0) -> original via Minv
    off = np.array([[1, 0, cx0], [0, 1, cy0], [0, 0, 1]], dtype=np.float64)
    to_orig = (np.vstack([Minv, [0, 0, 1]]) @ off)[:2]
    return Crop(src[cy0:cy1, cx0:cx1].copy(), to_orig, a, h, pad, bool(clipped), rotated)


def to_original_poly(box: Tuple[int, int, int, int], M: np.ndarray) -> List[Tuple[float, float]]:
    """Map a crop-space box to an original-image polygon."""
    x0, y0, x1, y1 = box
    pts = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]], dtype=np.float64)
    return [tuple(map(float, p)) for p in (pts @ M.T)]


# ====================================================================== records
@dataclass
class RegionRun:
    """One region under one crop condition."""

    condition: str
    text_b: str
    n_segments: int
    candidates: List[str]
    char_polys: List[List[Tuple[float, float]]]
    border_touching_segments: int
    small_segments: int
    excluded_lines: int
    other_regions_in_crop: int
    clipped: bool
    rotated: bool
    angle: float
    crop_path: str = ""


@dataclass
class SceneImage:
    """Detection, both crop conditions and both recognizers for one image."""

    path: str
    kind: str
    group_id: str
    status: str
    gt: str                       # normalized ground truth ('' if none)
    gt_raw: str
    label_flags: Dict[str, object]
    regions: List[Dict[str, object]]
    runs: Dict[str, List[RegionRun]]
    text_a: str
    text_b: Dict[str, str]
    reading_order: str
    marking_flags: Dict[str, object] = field(default_factory=dict)
    failure: Dict[str, str] = field(default_factory=dict)


def load_scene_labels(cfg: AppConfig) -> Dict[str, Dict[str, object]]:
    """User-entered scene transcriptions (never generated by the program)."""
    rows = read_csv_rows(DATA_DIR / cfg.scene.labels_file)
    out = {}
    for r in rows:
        out[r.get("image_path", "")] = {"transcription": r.get("transcription", "") or "",
                                        "unreadable": str(r.get("unreadable", "")).lower() in ("1", "true", "yes"),
                                        "uncertain": str(r.get("uncertain", "")).lower() in ("1", "true", "yes")}
    return out


def _run_condition(img, ir_regions, ratio, model, feats, cfg, out_dir, stem, cond) -> List[RegionRun]:
    runs = []
    for g in ir_regions:
        cr = padded_crop(img, g["poly"], ratio, cfg.scene.rotation_threshold_deg)
        crop_file = out_dir / "crops" / f"{stem}_r{g['index']}_{cond}.png"
        cv2.imencode(".png", cr.image)[1].tofile(str(crop_file))
        # other detected regions whose centre falls inside this padded crop (contamination record)
        others = 0
        inv = cv2.invertAffineTransform(cr.to_original)
        for o in ir_regions:
            if o["index"] == g["index"]:
                continue
            c = o["poly"].mean(axis=0)
            u, v = inv @ np.array([c[0], c[1], 1.0])
            if 0 <= u < cr.image.shape[1] and 0 <= v < cr.image.shape[0]:
                others += 1
        try:
            seg, fs = segment_features(cr.image, cfg)
        except Exception:
            runs.append(RegionRun(cond, "", 0, [], [], 0, 0, 0, others, cr.clipped, cr.rotated, cr.angle,
                                  str(crop_file)))
            continue
        cands, text = [], ""
        for f in fs:
            if f is None:
                cands.append("FEATURE_FAILED")
                text += "?"
                continue
            lp = model.log_posterior(np.array([[f[k] for k in feats]]))[0]
            order = np.argsort(-lp)[:3]
            cands.append(" ".join(f"{model.classes[j]}:{np.exp(lp[j]):.2f}" for j in order))
            text += model.classes[order[0]]
        hh, ww = cr.image.shape[:2]
        heights = [b[3] - b[1] for b in seg.boxes]
        med = float(np.median(heights)) if heights else 0.0
        runs.append(RegionRun(
            cond, text, seg.count, cands, [to_original_poly(b, cr.to_original) for b in seg.boxes],
            border_touching_segments=sum(1 for b in seg.boxes if b[0] <= 0 or b[1] <= 0 or b[2] >= ww or b[3] >= hh),
            small_segments=sum(1 for b in seg.boxes if med and (b[3] - b[1]) < 0.4 * med),
            excluded_lines=len(seg.excluded_lines), other_regions_in_crop=others, clipped=cr.clipped,
            rotated=cr.rotated, angle=cr.angle, crop_path=str(crop_file)))
    return runs


def attribute(si: SceneImage, cond: str) -> str:
    """Pre-declared attribution: detection -> cropping -> segmentation -> classification -> correct."""
    if not si.gt:
        return "NOT_EVALUABLE"
    if not si.regions:
        return "DETECTION_MISS"
    runs = si.runs[cond]
    if any(r.border_touching_segments for r in runs):
        return "CROPPING (segment touches crop border: stroke may be cut)"
    if sum(r.n_segments for r in runs) != len(si.gt):
        return "SEGMENTATION"
    if si.text_b[cond] != si.gt:
        return "CLASSIFICATION"
    return "CORRECT"


def run_stabilization(cfg: AppConfig, out_dir: Path = SCENE_RESULTS_DIR,
                      say: Callable[[str], None] = print) -> Dict[str, object]:
    """Both crop conditions on every Formal / Informal crop and every scene."""
    sc = cfg.scene
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "crops").mkdir(exist_ok=True)
    det, rec = Detector(cfg), Recognizer(cfg)
    items, pool = build_pool(cfg, say)
    flags_by_path = {it.path: it.flags for it in items}
    feats = list(TOPO_FEATURES) + list(GEO_FEATURES) + list(cfg.direction.direction_features)
    groups = {"topology": list(TOPO_FEATURES), "geometry": list(GEO_FEATURES),
              "direction": list(cfg.direction.direction_features)}
    models: Dict[str, object] = {}

    def model_for(group: Optional[str]):
        key = group or "__all__"
        if key not in models:
            train = [p for p in pool if p[0] != group]
            if group is not None and any(p[0] == group for p in train):
                raise RuntimeError("leakage")
            X = np.array([[p[2][f] for f in feats] for p in train])
            models[key] = fit_group_model(X, [p[1] for p in train], feats, groups, {g: 1 / 3 for g in groups}, cfg)
        return models[key]

    labels = load_scene_labels(cfg)
    audit = run_dataset_audit(cfg, write=False, previews=False, verbose=False)
    entries = sorted([m for m in audit.manifest if m["style"] in (FORMAL, INFORMAL, INFORMAL_SCENE)],
                     key=lambda m: str(m["image_path"]))
    images: List[SceneImage] = []
    for n, m in enumerate(entries):
        path, kind = str(m["image_path"]), str(m["style"])
        img = load_image(PROJECT_ROOT / path).image
        img = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if kind == INFORMAL_SCENE:
            lab = labels.get(path, {})
            raw = "" if lab.get("unreadable") else str(lab.get("transcription", ""))
            label_flags = {"labeled": bool(raw.strip()), "unreadable": bool(lab.get("unreadable")),
                           "uncertain": bool(lab.get("uncertain"))}
        else:
            raw = str(m.get("transcription", "") or "")
            label_flags = {"labeled": bool(raw.strip())}
        gt = normalize_transcription(raw, cfg.dataset) if raw.strip() else ""
        polys, confs = det.detect(img)
        regions = [{"index": k, "poly": p, "score": c,
                    "bbox": (float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max()))}
                   for k, (p, c) in enumerate(zip(polys, confs))]

        class _R:  # adapter for order_regions (needs .bbox / .line)
            def __init__(self, d):
                self.d, self.bbox, self.line = d, d["bbox"], 0
        ordered, note = order_regions([_R(d) for d in regions], cfg.text_detection.same_line_overlap)
        regions = [o.d for o in ordered]
        model = model_for(str(m["group_id"]) if kind in (FORMAL, INFORMAL) else None)
        text_a = "".join(rec.recognize(rec.rectify(img, g["poly"])) for g in regions)
        stem = Path(path).stem
        runs = {cond: _run_condition(img, regions, ratio, model, feats, cfg, out_dir, stem, cond)
                for cond, ratio in (("tight", sc.tight_ratio), ("padded", sc.pad_ratio))}
        si = SceneImage(path, kind, str(m["group_id"]), str(m["validation_status"]), gt, raw, label_flags, regions,
                        runs, text_a, {c: "".join(r.text_b for r in rs) for c, rs in runs.items()}, note,
                        flags_by_path.get(path, {}))
        si.failure = {c: attribute(si, c) for c in runs}
        images.append(si)
        if (n + 1) % 100 == 0:
            say(f"[SCENE] {n + 1}/{len(entries)} images")
    return {"images": images, "pool_characters": len(pool), "models": len(models)}


# ====================================================================== evaluation
def _metrics(sel: Sequence[SceneImage], cond: str) -> Dict[str, object]:
    n = sum(len(i.gt) for i in sel)
    return {"images": len(sel), "cer": sum(edit_ops(i.text_b[cond], i.gt)[0] for i in sel) / n if n else float("nan"),
            "exact": float(np.mean([i.text_b[cond] == i.gt for i in sel])) if sel else float("nan"),
            "failures": dict(Counter(i.failure[cond].split(" ")[0] for i in sel))}


def characteristics(i: SceneImage, slant_thr: float) -> Dict[str, bool]:
    """Observable traits of an image (from GT text, style and measured flags; no manual annotation)."""
    g = i.gt_raw
    return {"handwriting (Informal)": i.kind in (INFORMAL, INFORMAL_SCENE),
            "has uppercase": any(c.isupper() for c in g), "has digit": any(c.isdigit() for c in g),
            "has symbol (outside CRNN charset)": any((not c.isspace()) and c not in CRNN_SUPPORTED for c in g),
            "slanted (|median slant| >= thr)": abs(float(i.marking_flags.get("median_slant_deg", 0) or 0)) >= slant_thr,
            "touching flagged": bool(i.marking_flags.get("possible_touching", 0))}


def evaluate(res: Dict[str, object], cfg: AppConfig) -> Dict[str, object]:
    imgs: List[SceneImage] = res["images"]
    crops = [i for i in imgs if i.kind in (FORMAL, INFORMAL) and i.gt]
    scenes = [i for i in imgs if i.kind == INFORMAL_SCENE]
    labeled = [i for i in scenes if i.gt]
    out = {"crops": {c: _metrics(crops, c) for c in ("tight", "padded")},
           "crops_by_style": {s: {c: _metrics([i for i in crops if i.kind == s], c) for c in ("tight", "padded")}
                              for s in (FORMAL, INFORMAL)},
           "scenes": {"total": len(scenes), "detected": sum(1 for i in scenes if i.regions),
                      "missed": [Path(i.path).stem for i in scenes if not i.regions],
                      "labeled": len(labeled), "unlabeled": sum(1 for i in scenes if not i.label_flags.get("labeled")
                                                                and not i.label_flags.get("unreadable")),
                      "unreadable": sum(1 for i in scenes if i.label_flags.get("unreadable")),
                      "uncertain": sum(1 for i in scenes if i.label_flags.get("uncertain")),
                      "evaluation": ({c: _metrics(labeled, c) for c in ("tight", "padded")} if labeled
                                     else "NOT EVALUABLE (no manual scene transcriptions yet - run --label-scenes)")}}
    # segmentation compatibility (padded vs tight, crops with GT)
    comp = Counter()
    for i in crops:
        for c in ("tight", "padded"):
            nseg = sum(r.n_segments for r in i.runs[c])
            comp[(c, "over_segmented")] += nseg > len(i.gt)
            comp[(c, "under_segmented")] += nseg < len(i.gt)
            comp[(c, "border_touching (cut stroke risk)")] += any(r.border_touching_segments for r in i.runs[c])
            comp[(c, "decimal_lost")] += ("." in i.gt and nseg < len(i.gt))
            comp[(c, "small_symbol_lost")] += (any(ch in i.gt for ch in "-./=+") and nseg < len(i.gt))
            comp[(c, "touching_merged")] += (bool(i.marking_flags.get("possible_touching")) and nseg < len(i.gt))
            comp[(c, "neighbour_region_in_crop")] += any(r.other_regions_in_crop for r in i.runs[c])
            comp[(c, "crop_clipped_by_border")] += any(r.clipped for r in i.runs[c])
    out["segmentation_compatibility"] = {f"{c}|{k}": v for (c, k), v in comp.items()}
    # OCR vs topology (padded), case-folded alphanumerics; scope = crops with GT (+ labelled scenes)
    target = crops + labeled
    cats = defaultdict(list)
    for i in target:
        a_ok = casefold_alnum(i.text_a) == casefold_alnum(i.gt)
        b_ok = casefold_alnum(i.text_b["padded"]) == casefold_alnum(i.gt)
        cats["both_correct" if a_ok and b_ok else "ocr_only_correct" if a_ok else
             "topology_only_correct" if b_ok else "both_wrong"].append(i)
    out["ocr_vs_topology"] = {k: len(v) for k, v in cats.items()}
    supported = [i for i in target if all((c.isspace() or c in CRNN_SUPPORTED) for c in i.gt_raw)]
    out["ocr_vs_topology_supported_only"] = {
        "images": len(supported),
        "A_exact_casefold": float(np.mean([casefold_alnum(i.text_a) == casefold_alnum(i.gt) for i in supported])) if supported else float("nan"),
        "B_exact_casefold": float(np.mean([casefold_alnum(i.text_b["padded"]) == casefold_alnum(i.gt) for i in supported])) if supported else float("nan")}
    out["unsupported_by_crnn_images"] = len(target) - len(supported)
    trait_rows = []
    for k, v in cats.items():
        if not v:
            continue
        ch = [characteristics(i, cfg.scene.slant_handwriting_deg) for i in v]
        trait_rows.append({"category": k, "images": len(v),
                           **{t: float(np.mean([c[t] for c in ch])) for t in ch[0]}})
    out["category_traits"] = trait_rows
    out["_cats"] = cats
    return out
