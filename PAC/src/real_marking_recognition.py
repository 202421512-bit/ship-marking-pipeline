"""REAL MARKING RECOGNITION: end-to-end check on whole marking images (not a single-digit demo).

Original image -> preprocessing/binarization -> character segmentation -> topology + geometry ->
Bayesian character classification -> left-to-right string reconstruction -> recognized text.

Evaluation (transcribed images only; ground truth never used in inference):
* Each image is recognized by models fitted on COUNT_MATCH characters of OTHER groups (leave-one-group-out).
* String metrics (exact match, CER via edit distance) include images whose segmentation failed or whose
  segment count mismatches - missing / extra characters count as errors, never silently dropped.
* Segmented-character classification accuracy is reported separately (COUNT_MATCH images only).
* Spaces and line breaks are NOT inserted: possible spaces / reading-order doubts are flagged.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from .config import MARKING_RESULTS_DIR, AppConfig
from .dataset import FORMAL, INFORMAL, normalize_transcription, run_dataset_audit, segment_characters
from .image_io import load_image
from .prototypes import extract_features
from .real_experiment import GEO_FEATURES, TOPO_FEATURES, char_canvas, feature_dict, fit_reference, rel, write_csv

MODELS = ("geometry", "topology", "combined")


# ====================================================================== data structures
@dataclass
class Segment:
    """One character segment with features and (later) predictions."""

    index: int
    box: Tuple[int, int, int, int]
    mask: np.ndarray = field(repr=False)
    x: Optional[Dict[str, float]] = field(default=None, repr=False)
    feature_error: str = ""
    topo: Dict[str, float] = field(default_factory=dict)
    preds: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)   # model -> top-3 (label, posterior)


@dataclass
class MarkingImage:
    """One Formal / Informal marking image and its recognition result."""

    path: str
    image_id: str
    group_id: str
    style: str
    status: str
    transcription: str
    gt: str                      # normalized transcription ('' if none)
    segments: List[Segment]
    excluded_lines: int
    flags: Dict[str, object] = field(default_factory=dict)
    text: Dict[str, str] = field(default_factory=dict)          # model -> reconstructed text

    @property
    def transcribed(self) -> bool:
        return bool(self.transcription.strip())


# ====================================================================== reconstruction
def reconstruct(segments: Sequence[Segment], model: str, cfg: AppConfig) -> Tuple[str, Dict[str, object]]:
    """Left-to-right single-line string + uncertainty flags (spaces / order are flagged, not decided)."""
    mc = cfg.marking
    if not segments:
        return "", {"possible_space_after": [], "off_line_segments": [], "reading_order": "no segments"}
    heights = np.array([b[3] - b[1] for b in (s.box for s in segments)], dtype=float)
    med_h = float(np.median(heights))
    tops = np.array([s.box[1] for s in segments], float)
    bots = np.array([s.box[3] for s in segments], float)
    band_t, band_b = float(np.median(tops)), float(np.median(bots))
    off_line = []
    for s, t, b in zip(segments, tops, bots):
        ov = max(0.0, min(b, band_b) - max(t, band_t)) / max(1.0, min(b - t, band_b - band_t))
        if ov < mc.line_overlap_min:
            off_line.append(s.index)
    spaces = []
    for a, b in zip(segments[:-1], segments[1:]):
        gap = b.box[0] - a.box[2]
        if gap > mc.space_gap_ratio * med_h:
            spaces.append(a.index)
    text = "".join(s.preds[model][0][0] if s.preds.get(model) else "?" for s in segments)
    return text, {"possible_space_after": spaces, "off_line_segments": off_line,
                  "reading_order": "single line, left-to-right" + (" (UNCERTAIN: off-line segments)" if off_line else "")}


def edit_ops(pred: str, gt: str) -> Tuple[int, int, int, int]:
    """Levenshtein distance with (substitutions, deletions=missed GT chars, insertions=extra chars)."""
    n, m = len(gt), len(pred)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (gt[i - 1] != pred[j - 1]))
    i, j, s, dl, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (gt[i - 1] != pred[j - 1]):
            s += gt[i - 1] != pred[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            dl += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return int(d[n, m]), s, dl, ins


# ====================================================================== feature / audit helpers
def _stroke_width(mask: np.ndarray) -> float:
    skel = skeletonize(mask)
    return 2.0 * float(np.median(ndimage.distance_transform_edt(mask)[skel])) if skel.any() else 0.0


def _slant_deg(mask: np.ndarray) -> Optional[float]:
    """Major-axis angle from vertical (deg) for elongated segments; None if not elongated."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return None
    cov = np.cov(np.vstack([xs - xs.mean(), ys - ys.mean()]))
    w, v = np.linalg.eigh(cov)
    if w[1] < 2.5 * max(w[0], 1e-9):           # not elongated enough to define a slant
        return None
    vx, vy = v[:, 1]
    ang = math.degrees(math.atan2(vx, vy))       # 0 = vertical
    return ((ang + 90) % 180) - 90


def load_markings(cfg: AppConfig, say: Callable[[str], None]) -> List[MarkingImage]:
    """Segment every Formal / Informal crop and extract features of every segment (no labels used)."""
    audit = run_dataset_audit(cfg, write=False, previews=False, verbose=False)
    items: List[MarkingImage] = []
    rc = cfg.real
    for m in sorted(audit.manifest, key=lambda r: str(r["image_path"])):
        if m["style"] not in (FORMAL, INFORMAL):
            continue
        seg = audit.segmentations.get(str(m["image_path"]))
        segs: List[Segment] = []
        if seg is not None:
            for k, (box, cm) in enumerate(zip(seg.boxes, seg.char_masks)):
                s = Segment(k, tuple(int(v) for v in box), cm)
                try:
                    b = extract_features(char_canvas(cm, cfg), cfg, with_persistence=True)
                    f = feature_dict(b, with_ph=True)
                    s.x = {kk: float(f[kk]) for kk in TOPO_FEATURES + GEO_FEATURES}
                    s.topo = {"beta_0": f["beta_0"], "beta_1": f["beta_1"], "endpoints": f["endpoints"],
                              "branch_points": f["branch_points"]}
                except Exception as exc:  # recorded; the segment is still part of the string
                    s.feature_error = f"{type(exc).__name__}: {exc}"
                segs.append(s)
        trans = str(m.get("transcription", "") or "")
        mi = MarkingImage(str(m["image_path"]), Path(str(m["image_path"])).stem, str(m["group_id"]), str(m["style"]),
                          str(m["validation_status"]), trans,
                          normalize_transcription(trans, cfg.dataset) if trans.strip() else "", segs,
                          int(m.get("excluded_line_components") or 0))
        heights = [b[3] - b[1] for b in (s.box for s in segs)]
        med_h = float(np.median(heights)) if heights else 0.0
        slants = [a for a in (_slant_deg(s.mask) for s in segs) if a is not None]
        ink = seg.mask if seg is not None else None
        mi.flags = {
            "possible_touching": sum(1 for s in segs if med_h and (s.box[2] - s.box[0]) > rc.touching_width_ratio * med_h),
            "small_segments": sum(1 for s in segs if med_h and (s.box[3] - s.box[1]) < rc.small_height_ratio * med_h),
            "median_stroke_width_px": _stroke_width(ink) if ink is not None and ink.any() else 0.0,
            "median_slant_deg": float(np.median(slants)) if slants else 0.0,
        }
        items.append(mi)
    say(f"[MARK] {len(items)} Formal/Informal marking images, {sum(len(i.segments) for i in items)} segments")
    return items


# ====================================================================== recognition + evaluation
def recognize_all(items: Sequence[MarkingImage], cfg: AppConfig) -> Dict[str, object]:
    """Leave-one-group-out models from COUNT_MATCH characters; predict every segment of every image."""
    c = cfg.character
    weights = {"geometry": (0.0, 1.0), "topology": (1.0, 0.0), "combined": (c.w_topology, c.w_geometry)}
    pool: List[Tuple[str, str, Dict[str, float]]] = []          # (group, label, x)
    for it in items:
        if it.status == "COUNT_MATCH" and len(it.segments) == len(it.gt):
            for s, ch in zip(it.segments, it.gt):
                if s.x is not None:
                    pool.append((it.group_id, ch, s.x))
    cache: Dict[str, Dict[str, object]] = {}
    checks = Counter()
    for it in items:
        if it.group_id not in cache:
            fit_groups = [g for g, _, _ in pool if g != it.group_id]
            if it.group_id in set(fit_groups):
                raise RuntimeError("leakage: test group inside training pool")
            fit = [(lab, x) for g, lab, x in pool if g != it.group_id]
            cache[it.group_id] = {m: fit_reference([x for _, x in fit], [lab for lab, _ in fit], TOPO_FEATURES,
                                                   GEO_FEATURES, w, cfg) for m, w in weights.items()}
            checks["models_fitted"] += 1
        models = cache[it.group_id]
        for s in it.segments:
            if s.x is None:
                continue
            for m, ref in models.items():
                res = ref.clf.predict(dict(s.x))
                s.preds[m] = list(res.posteriors.items())[:3]
        for m in MODELS:
            it.text[m], fl = reconstruct(it.segments, m, cfg)
            if m == cfg.marking.primary_model:
                it.flags.update(fl)
                it.flags["train_classes"] = len(models[m].classes)
                it.flags["gt_classes_unseen_in_training"] = sorted({ch for ch in it.gt if ch not in models[m].classes})
    return {"pool_characters": len(pool), "pool_classes": len({lab for _, lab, _ in pool}),
            "pool_groups": len({g for g, _, _ in pool}), **checks}


def evaluate(items: Sequence[MarkingImage]) -> Dict[str, object]:
    """String-level and character-level metrics on transcribed images."""
    tr = [i for i in items if i.transcribed]
    out: Dict[str, object] = {"transcribed_images": len(tr),
                              "status_counts": dict(Counter(i.status for i in tr)),
                              "not_transcribed_images": sum(1 for i in items if not i.transcribed)}
    for m in MODELS:
        block = {}
        for scope, sel in (("all", tr), ("FORMAL", [i for i in tr if i.style == FORMAL]),
                           ("INFORMAL", [i for i in tr if i.style == INFORMAL])):
            ed = [edit_ops(i.text[m], i.gt) for i in sel]
            n_gt = sum(len(i.gt) for i in sel)
            cm = [i for i in sel if i.status == "COUNT_MATCH"]
            seg_correct = [s.preds[m][0][0] == ch for i in cm for s, ch in zip(i.segments, i.gt) if s.preds.get(m)]
            block[scope] = {
                "images": len(sel), "gt_characters": n_gt,
                "string_exact_match": float(np.mean([i.text[m] == i.gt for i in sel])) if sel else float("nan"),
                "cer": sum(e[0] for e in ed) / n_gt if n_gt else float("nan"),
                "substitutions": sum(e[1] for e in ed), "deletions_missed_chars": sum(e[2] for e in ed),
                "insertions_extra_chars": sum(e[3] for e in ed),
                "end_to_end_char_accuracy_1_minus_cer": 1 - sum(e[0] for e in ed) / n_gt if n_gt else float("nan"),
                "count_match_images": len(cm),
                "segmented_char_accuracy_count_match": float(np.mean(seg_correct)) if seg_correct else float("nan"),
                "segmented_chars_evaluated": len(seg_correct),
                "string_exact_match_count_match_only": float(np.mean([i.text[m] == i.gt for i in cm])) if cm else float("nan"),
            }
        out[m] = block
    out["segmentation_success_rate_count_match"] = (
        sum(1 for i in tr if i.status == "COUNT_MATCH") / len(tr) if tr else float("nan"))
    out["segmentation_failures"] = {k: v for k, v in out["status_counts"].items() if k != "COUNT_MATCH"}
    return out


def per_class(items: Sequence[MarkingImage], model: str) -> List[Dict[str, object]]:
    """Per-class accuracy of segmented characters (COUNT_MATCH images)."""
    by = defaultdict(list)
    conf = Counter()
    for i in items:
        if i.status != "COUNT_MATCH":
            continue
        for s, ch in zip(i.segments, i.gt):
            if s.preds.get(model):
                p = s.preds[model][0][0]
                by[ch].append((p == ch, i.style, i.group_id))
                if p != ch:
                    conf[(ch, p)] += 1
    rows = []
    for ch in sorted(by):
        v = by[ch]
        top_conf = [f"{p}x{n}" for (g, p), n in conf.most_common() if g == ch][:3]
        rows.append({"class": ch, "n": len(v), "accuracy": float(np.mean([x[0] for x in v])),
                     "formal_n": sum(x[1] == FORMAL for x in v), "informal_n": sum(x[1] == INFORMAL for x in v),
                     "source_groups": len({x[2] for x in v}), "most_common_confusions": " ".join(top_conf)})
    return rows


def class_coverage(items: Sequence[MarkingImage], cfg: AppConfig) -> List[Dict[str, object]]:
    """Which characters are trainable / insufficient / only in failed segmentations / welding (separate)."""
    trained = defaultdict(lambda: {"n": 0, "groups": set(), "formal": 0, "informal": 0})
    failed = Counter()
    for i in items:
        if not i.transcribed:
            continue
        if i.status == "COUNT_MATCH":
            for ch in i.gt:
                t = trained[ch]
                t["n"] += 1
                t["groups"].add(i.group_id)
                t["formal" if i.style == FORMAL else "informal"] += 1
        else:
            for ch in i.gt:
                failed[ch] += 1
    rows = []
    for ch in sorted(set(trained) | set(failed)):
        t = trained.get(ch)
        n, g = (t["n"], len(t["groups"])) if t else (0, 0)
        if n == 0:
            cat = "NOT_TRAINED (appears only in COUNT_MISMATCH / SEGMENTATION_FAILED images)"
        elif g < 2:
            cat = "INSUFFICIENT (single source group: trainable but never testable under group splits)"
        elif n < cfg.dataset.min_samples_per_class:
            cat = "INSUFFICIENT (fewer than min_samples_per_class)"
        else:
            cat = "TRAINABLE"
        rows.append({"class": ch, "category": cat, "training_samples": n, "source_groups": g,
                     "formal_samples": t["formal"] if t else 0, "informal_samples": t["informal"] if t else 0,
                     "chars_in_failed_segmentation_images": failed.get(ch, 0)})
    return rows


@dataclass
class MarkingResult:
    """In-memory results for figures and reports."""

    items: List[MarkingImage]
    fit_info: Dict[str, object]
    metrics: Dict[str, object]
    per_class: List[Dict[str, object]]
    coverage: List[Dict[str, object]]
    audit_rows: List[Dict[str, object]]
    report: Dict[str, object] = field(default_factory=dict)
    out_dir: Path = MARKING_RESULTS_DIR


AUDIT_CATEGORIES = {
    "touching": "possible touching characters (segment width > 1.5 x median height)",
    "decimal_point": "transcription contains '.'",
    "hyphen": "transcription contains '-'",
    "plus": "transcription contains '+'",
    "thin_stroke": "median stroke width <= thin_stroke_px",
    "slanted": "|median segment slant| >= slant_deg",
    "background_line": "line-like components were excluded (scratches / guide lines)",
}


def audit_membership(i: MarkingImage, cfg: AppConfig) -> List[str]:
    """Pre-declared segmentation-audit categories of one image (image content + transcription text only)."""
    mc = cfg.marking
    cats = []
    if i.flags.get("possible_touching", 0) > 0:
        cats.append("touching")
    if "." in i.transcription:
        cats.append("decimal_point")
    if "-" in i.transcription:
        cats.append("hyphen")
    if "+" in i.transcription:
        cats.append("plus")
    if 0 < i.flags.get("median_stroke_width_px", 0) <= mc.thin_stroke_px:
        cats.append("thin_stroke")
    if abs(i.flags.get("median_slant_deg", 0)) >= mc.slant_deg:
        cats.append("slanted")
    if i.excluded_lines > 0:
        cats.append("background_line")
    return cats


def run_marking_recognition(cfg: AppConfig, out_dir: Path = MARKING_RESULTS_DIR,
                            say: Callable[[str], None] = print) -> MarkingResult:
    """Full check; CSV outputs written to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    items = load_markings(cfg, say)
    fit_info = recognize_all(items, cfg)
    metrics = evaluate(items)
    pc = per_class(items, cfg.marking.primary_model)
    cov = class_coverage(items, cfg)
    pm = cfg.marking.primary_model
    img_rows, char_rows, audit_rows = [], [], []
    for i in items:
        cats = audit_membership(i, cfg)
        ed = edit_ops(i.text[pm], i.gt) if i.transcribed else None
        img_rows.append({
            "image_path": i.path, "style": i.style, "group_id": i.group_id, "validation_status": i.status,
            "ground_truth": i.transcription if i.transcribed else "UNKNOWN (no transcription)",
            "ground_truth_normalized": i.gt, "recognized_text": i.text[pm],
            "recognized_geometry": i.text["geometry"], "recognized_topology": i.text["topology"],
            "exact_match": (i.text[pm] == i.gt) if i.transcribed else "NOT_EVALUABLE",
            "edit_distance": ed[0] if ed else "", "cer": (ed[0] / len(i.gt)) if ed and i.gt else "",
            "segment_count": len(i.segments), "expected_characters": len(i.gt) if i.transcribed else "",
            "possible_space_after_segment": "|".join(map(str, i.flags.get("possible_space_after", []))),
            "reading_order": i.flags.get("reading_order", ""),
            "gt_classes_unseen_in_training": "".join(i.flags.get("gt_classes_unseen_in_training", [])),
            "audit_categories": "|".join(cats), **{k: i.flags.get(k) for k in
                                                   ("possible_touching", "small_segments", "median_stroke_width_px",
                                                    "median_slant_deg")},
            "excluded_line_components": i.excluded_lines})
        for s in i.segments:
            gt_ch = i.gt[s.index] if (i.status == "COUNT_MATCH" and s.index < len(i.gt)) else ""
            row = {"image_path": i.path, "style": i.style, "segment_index": s.index,
                   "box": f"{s.box[0]},{s.box[1]},{s.box[2]},{s.box[3]}",
                   "ground_truth_aligned": gt_ch or ("UNALIGNED" if i.transcribed else "UNKNOWN"),
                   "feature_error": s.feature_error, **s.topo,
                   "posterior_type": "UNCALIBRATED MODEL POSTERIOR"}
            for m in MODELS:
                cand = s.preds.get(m, [])
                row[f"{m}_candidates"] = " ".join(f"{c}:{p:.3f}" for c, p in cand)
                row[f"{m}_correct"] = (cand[0][0] == gt_ch) if (cand and gt_ch) else ""
            char_rows.append(row)
    for cat, desc in AUDIT_CATEGORIES.items():
        mem = [i for i in items if cat in audit_membership(i, cfg)]
        tr = [i for i in mem if i.transcribed]
        audit_rows.append({"category": cat, "definition": desc, "images": len(mem), "transcribed": len(tr),
                           "count_match_rate": (sum(i.status == "COUNT_MATCH" for i in tr) / len(tr)) if tr else "",
                           "string_exact_match_rate": (sum(i.text[pm] == i.gt for i in tr) / len(tr)) if tr else "",
                           "note": "COUNT_MATCH is necessary, not sufficient, for correct segmentation; see sheet"})
    write_csv(out_dir / "image_recognition.csv", img_rows)
    write_csv(out_dir / "character_predictions.csv", char_rows)
    write_csv(out_dir / "per_class_recognition.csv", pc)
    write_csv(out_dir / "class_coverage.csv", cov)
    write_csv(out_dir / "segmentation_audit.csv", audit_rows)
    return MarkingResult(items, fit_info, metrics, pc, cov, audit_rows, out_dir=out_dir)
