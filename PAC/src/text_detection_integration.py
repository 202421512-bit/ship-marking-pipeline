"""TEXT DETECTION INTEGRATION: full image -> text regions -> segmentation -> topology-based recognition -> string.

* Detector: PaddleOCR PP-OCRv3 DBNet exported to ONNX (OpenCV model zoo), run with OpenCV's built-in
  cv2.dnn_TextDetectionModel_DB - no new Python packages. It only proposes regions (never reads characters).
* A (pretrained OCR): OpenCV-zoo CRNN-EN (CTC, charset 0-9a-z) on each rectified region.
* B (ours): region box expanded -> existing segmentation -> existing topology + geometry + the 13 fixed
  direction features -> Gaussian group-mean classifier (1/3 per group), leave-one-group-out for crops,
  all 989 validated characters for scenes (scenes are never in training). No line-relative features.
* A and B are reported separately; their scores are never combined. Posteriors are not used for acceptance.
* No detection ground truth exists: detection recall is NOT EVALUABLE; a labelled PROXY (coverage of
  validated character boxes) is reported for COUNT_MATCH crops only.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import MODELS_DIR, PROJECT_ROOT, TEXT_DETECTION_RESULTS_DIR, AppConfig
from .dataset import FORMAL, INFORMAL, INFORMAL_SCENE, normalize_transcription, run_dataset_audit, segment_characters
from .direction_aware_experiment import direction_features, fit_group_model
from .image_io import load_image
from .prototypes import extract_features
from .real_experiment import GEO_FEATURES, TOPO_FEATURES, char_canvas, feature_dict, rel
from .real_marking_recognition import edit_ops, load_markings


# ====================================================================== pretrained models (OpenCV DNN)
class Detector:
    """PP-OCRv3 DBNet via cv2.dnn_TextDetectionModel_DB (aspect-preserving input, multiples of 32)."""

    def __init__(self, cfg: AppConfig) -> None:
        t = cfg.text_detection
        path = MODELS_DIR / t.detector_file
        if not path.exists():
            raise FileNotFoundError(f"detector weights missing: {path}")
        self.t = t
        self.model = cv2.dnn_TextDetectionModel_DB(str(path))
        self.model.setBinaryThreshold(t.det_binary_threshold)
        self.model.setPolygonThreshold(t.det_polygon_threshold)
        self.model.setMaxCandidates(t.det_max_candidates)
        self.model.setUnclipRatio(t.det_unclip_ratio)

    def detect(self, img_bgr: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        h, w = img_bgr.shape[:2]
        s = self.t.det_max_side / max(h, w)
        iw, ih = max(32, int(round(w * s / 32)) * 32), max(32, int(round(h * s / 32)) * 32)
        self.model.setInputParams(scale=1.0 / 255.0, size=(iw, ih), mean=self.t.det_mean)
        polys, confs = self.model.detect(img_bgr)
        return [np.asarray(p, dtype=np.float32).reshape(4, 2) for p in polys], [float(c) for c in confs]


class Recognizer:
    """CRNN-EN (CTC greedy) via cv2.dnn_TextRecognitionModel on a 100x32 rectified region."""

    def __init__(self, cfg: AppConfig) -> None:
        t = cfg.text_detection
        path = MODELS_DIR / t.recognizer_file
        if not path.exists():
            raise FileNotFoundError(f"recognizer weights missing: {path}")
        self.t = t
        self.model = cv2.dnn_TextRecognitionModel(str(path))
        self.model.setDecodeType("CTC-greedy")
        self.model.setVocabulary(list(t.recognizer_charset))
        self.model.setInputParams(scale=1.0 / 127.5, size=t.rec_size, mean=(127.5, 127.5, 127.5))
        self.gray = True

    def rectify(self, img_bgr: np.ndarray, poly: np.ndarray) -> np.ndarray:
        """Perspective-rectify a DB quadrilateral (bottom-left, top-left, top-right, bottom-right)."""
        w, h = self.t.rec_size
        dst = np.float32([[0, h - 1], [0, 0], [w - 1, 0], [w - 1, h - 1]])
        return cv2.warpPerspective(img_bgr, cv2.getPerspectiveTransform(poly.astype(np.float32), dst), (w, h))

    def recognize(self, rect_bgr: np.ndarray) -> str:
        if self.gray:
            try:
                return self.model.recognize(cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY))
            except cv2.error:
                self.gray = False
        return self.model.recognize(rect_bgr)


# ====================================================================== data structures
@dataclass
class Region:
    """One detected text region and both recognitions."""

    index: int
    poly: np.ndarray = field(repr=False)
    score: float
    bbox: Tuple[int, int, int, int]
    line: int = 0
    text_a: str = ""
    text_b: str = ""
    segments: List[Dict[str, object]] = field(default_factory=list)
    spurious: Optional[bool] = None      # COUNT_MATCH crops: contains no validated character centre


@dataclass
class ImageResult:
    """Detection + recognition result of one image."""

    path: str
    kind: str
    group_id: str
    status: str
    transcription: str
    gt: str
    regions: List[Region]
    text_a: str = ""
    text_b: str = ""
    reading_order: str = ""
    proxy_coverage: Optional[float] = None
    failure: str = ""
    flags: Dict[str, object] = field(default_factory=dict)


# ====================================================================== helpers
def order_regions(regions: List[Region], overlap: float) -> Tuple[List[Region], str]:
    """Group regions into lines by vertical overlap (top to bottom), left-to-right within a line."""
    regs = sorted(regions, key=lambda r: (r.bbox[1] + r.bbox[3]) / 2)
    lines: List[List[Region]] = []
    for r in regs:
        placed = False
        for ln in lines:
            ref = ln[0]
            ov = min(r.bbox[3], ref.bbox[3]) - max(r.bbox[1], ref.bbox[1])
            if ov > overlap * min(r.bbox[3] - r.bbox[1], ref.bbox[3] - ref.bbox[1]):
                ln.append(r)
                placed = True
                break
        if not placed:
            lines.append([r])
    out = []
    for k, ln in enumerate(lines):
        for r in sorted(ln, key=lambda x: x.bbox[0]):
            r.line = k
            out.append(r)
    note = "single line" if len(lines) <= 1 else f"UNCERTAIN: {len(lines)} lines (top-to-bottom order assumed)"
    return out, note


def casefold_alnum(s: str) -> str:
    """Comparison form shared by A and B: lower-case alphanumerics only (CRNN-EN charset)."""
    return "".join(c for c in s.lower() if c.isascii() and c.isalnum())


def build_pool(cfg: AppConfig, say: Callable[[str], None]):
    """Validated characters (COUNT_MATCH) with existing + direction features, and the marking metadata."""
    items = load_markings(cfg, say)
    pool = []          # (group, label, feature dict)
    for it in items:
        if it.status == "COUNT_MATCH" and len(it.segments) == len(it.gt):
            for s, ch in zip(it.segments, it.gt):
                if s.x is None:
                    continue
                f = dict(s.x)
                f.update(direction_features(char_canvas(s.mask, cfg), cfg))
                pool.append((it.group_id, ch, f))
    return items, pool


def segment_features(crop_bgr: np.ndarray, cfg: AppConfig,
                     attach=None) -> Tuple[list, List[Optional[Dict[str, float]]]]:
    """Existing segmentation (+ optional mark attachment) + existing and direction features for one crop."""
    seg = segment_characters(crop_bgr, cfg.preprocessing, cfg.dataset, attach=attach)
    feats = []
    for cm in seg.char_masks:
        try:
            f = feature_dict(extract_features(char_canvas(cm, cfg), cfg, with_persistence=True), with_ph=True)
            x = {k: float(f[k]) for k in TOPO_FEATURES + GEO_FEATURES}
            x.update(direction_features(char_canvas(cm, cfg), cfg))
            feats.append(x)
        except Exception:
            feats.append(None)
    return seg, feats


# ====================================================================== main run
@dataclass
class TDResult:
    """All results."""

    images: List[ImageResult]
    info: Dict[str, object]
    report: Dict[str, object] = field(default_factory=dict)
    out_dir: Path = TEXT_DETECTION_RESULTS_DIR


def run_text_detection(cfg: AppConfig, out_dir: Path = TEXT_DETECTION_RESULTS_DIR,
                       say: Callable[[str], None] = print) -> TDResult:
    """Detect, recognise (A and B) and evaluate every Formal / Informal crop and every scene."""
    t = cfg.text_detection
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "detected_regions").mkdir(exist_ok=True)
    t0 = time.perf_counter()
    det, rec = Detector(cfg), Recognizer(cfg)
    items, pool = build_pool(cfg, say)
    by_path = {it.path: it for it in items}
    feats = list(TOPO_FEATURES) + list(GEO_FEATURES) + list(cfg.direction.direction_features)
    groups = {"topology": list(TOPO_FEATURES), "geometry": list(GEO_FEATURES),
              "direction": list(cfg.direction.direction_features)}
    weights = {g: 1.0 / 3 for g in groups}
    models: Dict[str, object] = {}

    def model_for(group: Optional[str]):
        key = group or "__all__"
        if key not in models:
            train = [p for p in pool if p[0] != group]
            if group is not None and any(p[0] == group for p in train):
                raise RuntimeError("leakage: test group in training")
            X = np.array([[p[2][f] for f in feats] for p in train])
            models[key] = fit_group_model(X, [p[1] for p in train], feats, groups, weights, cfg)
        return models[key]

    audit = run_dataset_audit(cfg, write=False, previews=False, verbose=False)
    entries = sorted([m for m in audit.manifest if m["style"] in (FORMAL, INFORMAL, INFORMAL_SCENE)],
                     key=lambda m: str(m["image_path"]))
    results: List[ImageResult] = []
    for n, m in enumerate(entries):
        path = str(m["image_path"])
        img = load_image(PROJECT_ROOT / path).image
        img = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        trans = str(m.get("transcription", "") or "")
        ir = ImageResult(path, str(m["style"]), str(m["group_id"]), str(m["validation_status"]), trans,
                         normalize_transcription(trans, cfg.dataset) if trans.strip() else "", [])
        polys, confs = det.detect(img)
        H, W = img.shape[:2]
        for k, (p, c) in enumerate(zip(polys, confs)):
            x0, y0 = np.floor(p.min(axis=0)).astype(int)
            x1, y1 = np.ceil(p.max(axis=0)).astype(int)
            ir.regions.append(Region(k, p, c, (max(0, x0), max(0, y0), min(W, x1), min(H, y1))))
        ir.regions, ir.reading_order = order_regions(ir.regions, t.same_line_overlap)
        # scenes and untranscribed crops: never in training -> model on all validated characters
        group = ir.group_id if ir.kind in (FORMAL, INFORMAL) else None
        model = model_for(group)
        for r in ir.regions:
            rect = rec.rectify(img, r.poly)
            r.text_a = rec.recognize(rect)
            cv2.imencode(".png", rect)[1].tofile(str(out_dir / "detected_regions" / f"{Path(path).stem}_r{r.index}.png"))
            x0, y0, x1, y1 = r.bbox
            mg = int(round(t.crop_margin_ratio * max(1, y1 - y0)))
            cx0, cy0, cx1, cy1 = max(0, x0 - mg), max(0, y0 - mg), min(W, x1 + mg), min(H, y1 + mg)
            try:
                seg, fs = segment_features(img[cy0:cy1, cx0:cx1], cfg)
            except Exception as exc:
                r.segments = [{"error": f"segmentation: {exc}"}]
                continue
            for (bx0, by0, bx1, by1), f in zip(seg.boxes, fs):
                entry = {"box": (bx0 + cx0, by0 + cy0, bx1 + cx0, by1 + cy0)}
                if f is None:
                    entry["candidates"] = []
                    entry["status"] = "FEATURE_FAILED"
                else:
                    lp = model.log_posterior(np.array([[f[k] for k in feats]]))[0]
                    order = np.argsort(-lp)[:3]
                    entry["candidates"] = [(model.classes[j], float(np.exp(lp[j]))) for j in order]
                    entry["status"] = "OK"
                r.segments.append(entry)
            r.text_b = "".join(e["candidates"][0][0] if e.get("candidates") else "?" for e in r.segments)
        ir.text_a = "".join(r.text_a for r in ir.regions)
        ir.text_b = "".join(r.text_b for r in ir.regions)
        mk = by_path.get(path)
        if ir.status == "COUNT_MATCH" and mk is not None and ir.gt:
            centers = [((s.box[0] + s.box[2]) / 2, (s.box[1] + s.box[3]) / 2) for s in mk.segments]
            covered = [any(cv2.pointPolygonTest(r.poly, (float(cx), float(cy)), False) >= 0 for r in ir.regions)
                       for cx, cy in centers]
            ir.proxy_coverage = float(np.mean(covered)) if covered else 0.0
            for r in ir.regions:
                r.spurious = not any(cv2.pointPolygonTest(r.poly, (float(cx), float(cy)), False) >= 0
                                     for cx, cy in centers)
        if mk is not None:
            ir.flags = {"touching": mk.flags.get("possible_touching", 0) > 0, "decimal_point": "." in trans}
        ir.failure = attribute_failure(ir)
        results.append(ir)
        if (n + 1) % 50 == 0:
            say(f"[TD] {n + 1}/{len(entries)} images ({time.perf_counter() - t0:.0f} s)")
    info = {"runtime_s": round(time.perf_counter() - t0, 1), "pool_characters": len(pool),
            "pool_groups": len({p[0] for p in pool}), "models_fitted": len(models),
            "recognizer_input": "gray" if rec.gray else "bgr"}
    return TDResult(results, info, out_dir=out_dir)


def attribute_failure(ir: ImageResult) -> str:
    """Pre-declared failure attribution for transcribed images (B pipeline)."""
    if not ir.gt:
        return "NOT_EVALUABLE (no ground truth)"
    if not ir.regions:
        return "DETECTION_MISS (no region)"
    if ir.proxy_coverage is not None and ir.proxy_coverage < 1.0:
        return "DETECTION_MISS (validated character not covered)"
    n_seg = sum(len(r.segments) for r in ir.regions)
    if ir.proxy_coverage is not None and any(r.spurious and r.segments for r in ir.regions):
        return "DETECTION_FALSE_POSITIVE (extra region produced segments)"
    if n_seg != len(ir.gt):
        return "SEGMENTATION (segment count != characters)"
    if ir.text_b != ir.gt:
        return "CLASSIFICATION"
    return "CORRECT"


def evaluate(r: TDResult) -> Dict[str, object]:
    """CER / exact match for A and B on transcribed crops + failure counts + proxy detection coverage."""
    tr = [i for i in r.images if i.gt]
    out: Dict[str, object] = {"transcribed_images": len(tr)}
    for scope, sel in (("all", tr), (FORMAL, [i for i in tr if i.kind == FORMAL]),
                       (INFORMAL, [i for i in tr if i.kind == INFORMAL])):
        blk = {"images": len(sel)}
        n = sum(len(i.gt) for i in sel)
        blk["B_cer_raw"] = sum(edit_ops(i.text_b, i.gt)[0] for i in sel) / n if n else float("nan")
        blk["B_exact_raw"] = float(np.mean([i.text_b == i.gt for i in sel])) if sel else float("nan")
        blk["A_cer_raw"] = sum(edit_ops(i.text_a, i.gt)[0] for i in sel) / n if n else float("nan")
        blk["A_exact_raw"] = float(np.mean([i.text_a == i.gt for i in sel])) if sel else float("nan")
        cf = [(casefold_alnum(i.text_a), casefold_alnum(i.text_b), casefold_alnum(i.gt)) for i in sel]
        nc = sum(len(g) for _, _, g in cf)
        blk["A_cer_casefold_alnum"] = sum(edit_ops(a, g)[0] for a, _, g in cf) / nc if nc else float("nan")
        blk["B_cer_casefold_alnum"] = sum(edit_ops(b, g)[0] for _, b, g in cf) / nc if nc else float("nan")
        blk["A_exact_casefold_alnum"] = float(np.mean([a == g for a, _, g in cf])) if cf else float("nan")
        blk["B_exact_casefold_alnum"] = float(np.mean([b == g for _, b, g in cf])) if cf else float("nan")
        out[scope] = blk
    cm = [i for i in tr if i.proxy_coverage is not None]
    out["proxy_detection_coverage"] = {
        "definition": "fraction of validated character-box centres (COUNT_MATCH crops) inside a detected region; "
                      "PROXY, not detection recall (no detection ground truth exists)",
        "images": len(cm), "mean": float(np.mean([i.proxy_coverage for i in cm])) if cm else float("nan"),
        "images_fully_covered": sum(i.proxy_coverage == 1.0 for i in cm)}
    out["failure_counts"] = dict(Counter(i.failure.split(" ")[0] for i in tr))
    cf_pairs = [(casefold_alnum(i.text_a) == casefold_alnum(i.gt), casefold_alnum(i.text_b) == casefold_alnum(i.gt), i)
                for i in tr]
    out["A_vs_B_casefold"] = {"both_correct": sum(a and b for a, b, _ in cf_pairs),
                              "only_A_correct": sum(a and not b for a, b, _ in cf_pairs),
                              "only_B_correct": sum(b and not a for a, b, _ in cf_pairs),
                              "both_wrong": sum((not a) and (not b) for a, b, _ in cf_pairs)}
    for cat in ("touching", "decimal_point"):
        sel = [i for i in tr if i.flags.get(cat)]
        n = sum(len(i.gt) for i in sel)
        out[f"category_{cat}"] = {"images": len(sel),
                                  "B_exact": float(np.mean([i.text_b == i.gt for i in sel])) if sel else float("nan"),
                                  "B_cer": sum(edit_ops(i.text_b, i.gt)[0] for i in sel) / n if n else float("nan"),
                                  "failures": dict(Counter(i.failure.split(" ")[0] for i in sel))}
    sc = [i for i in r.images if i.kind == INFORMAL_SCENE]
    out["scenes"] = {"total": len(sc), "with_regions": sum(1 for i in sc if i.regions),
                     "regions": sum(len(i.regions) for i in sc),
                     "regions_without_B_segments": sum(1 for i in sc for g in i.regions if not g.segments),
                     "regions_with_empty_A_text": sum(1 for i in sc for g in i.regions if not g.text_a),
                     "multi_line_scenes": sum(1 for i in sc if i.reading_order.startswith("UNCERTAIN")),
                     "detection_accuracy": "NOT EVALUABLE (no scene detection ground truth)"}
    return out
