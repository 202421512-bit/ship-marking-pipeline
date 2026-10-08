"""SCENE ERROR AUDIT: character-level root causes of the failed full-scene readings.

Nothing in the system is changed: the frozen pipeline (PP-OCRv3 DB, padding 2.0, existing segmentation,
topology + geometry + 13 direction features, 1/3 group-mean Gaussian classifier trained on the 989
validated characters) is re-run on the 30 scenes only, and its strings are asserted to equal the stored
results/full_scene_stabilization outputs before any analysis.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import DATA_DIR, PROJECT_ROOT, SCENE_RESULTS_DIR, AppConfig
from .dataset import FORMAL, INFORMAL, INFORMAL_SCENE, normalize_transcription, read_csv_rows, run_dataset_audit
from .direction_aware_experiment import fit_group_model
from .full_scene_stabilization import CRNN_SUPPORTED, load_scene_labels, padded_crop
from .image_io import load_image
from .real_experiment import GEO_FEATURES, TOPO_FEATURES, write_csv
from .text_detection_integration import Detector, Recognizer, build_pool, casefold_alnum, order_regions, segment_features
from .dataset import segment_characters, rel

AUDIT_DIR = SCENE_RESULTS_DIR.parent / "scene_error_audit"
CAUSES = ["DETECTION_MISS", "CROP_DAMAGE", "SEGMENTATION_SPLIT", "SEGMENTATION_MERGE", "UNKNOWN_CLASS",
          "CHARACTER_CONFUSION", "READING_ORDER", "PREPROCESSING_ARTIFACT", "UNDETERMINED"]
CHECK_SYMBOLS = ["→", "←", "↑", "↓", "+", "-", ".", "/"]


# ====================================================================== alignment
def align(pred: str, gt: str) -> List[Tuple[str, Optional[int], Optional[int]]]:
    """Levenshtein alignment: list of (op, gt_index, pred_index); op in MATCH/SUB/DEL/INS."""
    n, m = len(gt), len(pred)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (gt[i - 1] != pred[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (gt[i - 1] != pred[j - 1]):
            ops.append(("MATCH" if gt[i - 1] == pred[j - 1] else "SUB", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("DEL", i - 1, None))
            i -= 1
        else:
            ops.append(("INS", None, j - 1))
            j -= 1
    return ops[::-1]


# ====================================================================== re-run (frozen) on scenes
@dataclass
class SceneRun:
    """Frozen-pipeline reading of one scene with per-segment detail."""

    path: str
    gt_raw: str
    gt: str
    regions: List[Dict[str, object]]
    crops: List[object] = field(repr=False)
    segs: List[object] = field(repr=False)              # per region Segmentation (crop coords)
    seg_feats: List[List[Optional[Dict[str, float]]]] = field(repr=False)
    pred_segments: List[Dict[str, object]] = field(default_factory=list)   # flattened, reading order
    text_b: str = ""
    text_a: str = ""


def rerun_scenes(cfg: AppConfig, say=print):
    det, rec = Detector(cfg), Recognizer(cfg)
    items, pool = build_pool(cfg, say)
    feats = list(TOPO_FEATURES) + list(GEO_FEATURES) + list(cfg.direction.direction_features)
    groups = {"topology": list(TOPO_FEATURES), "geometry": list(GEO_FEATURES),
              "direction": list(cfg.direction.direction_features)}
    X = np.array([[p[2][f] for f in feats] for p in pool])
    model = fit_group_model(X, [p[1] for p in pool], feats, groups, {g: 1 / 3 for g in groups}, cfg)
    labels = load_scene_labels(cfg)
    audit = run_dataset_audit(cfg, write=False, previews=False, verbose=False)
    scenes = sorted(str(m["image_path"]) for m in audit.manifest if m["style"] == INFORMAL_SCENE)
    runs = []
    for path in scenes:
        img = load_image(PROJECT_ROOT / path).image
        img = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        raw = str(labels.get(path, {}).get("transcription", ""))
        polys, confs = det.detect(img)
        regions = [{"index": k, "poly": p, "score": c,
                    "bbox": (float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max()))}
                   for k, (p, c) in enumerate(zip(polys, confs))]

        class _R:
            def __init__(self, d):
                self.d, self.bbox, self.line = d, d["bbox"], 0
        ordered, _ = order_regions([_R(d) for d in regions], cfg.text_detection.same_line_overlap)
        regions = [o.d for o in ordered]
        sr = SceneRun(path, raw, normalize_transcription(raw, cfg.dataset) if raw.strip() else "", regions, [], [], [])
        sr.text_a = "".join(rec.recognize(rec.rectify(img, g["poly"])) for g in regions)
        for g in regions:
            cr = padded_crop(img, g["poly"], cfg.scene.pad_ratio, cfg.scene.rotation_threshold_deg)
            seg, fs = segment_features(cr.image, cfg)
            sr.crops.append(cr)
            sr.segs.append(seg)
            sr.seg_feats.append(fs)
            for k, (box, cm, f) in enumerate(zip(seg.boxes, seg.char_masks, fs)):
                entry = {"region": g["index"], "k": k, "box": box, "mask": cm, "x": f}
                if f is None:
                    entry.update({"pred": "?", "top3": []})
                else:
                    lp = model.log_posterior(np.array([[f[q] for q in feats]]))[0]
                    order = np.argsort(-lp)[:3]
                    entry.update({"pred": model.classes[order[0]],
                                  "top3": [(model.classes[j], float(np.exp(lp[j]))) for j in order]})
                sr.pred_segments.append(entry)
        sr.text_b = "".join(e["pred"] for e in sr.pred_segments)
        runs.append(sr)
    return runs, model, feats, groups, pool, items


def verify_against_stored(runs: Sequence[SceneRun]) -> Dict[str, object]:
    """The audit must examine exactly the stored predictions."""
    stored = {r["image_path"]: r for r in read_csv_rows(SCENE_RESULTS_DIR / "scene_recognition_results.csv")}
    mism = [(Path(s.path).name, s.text_b, stored[s.path]["B_padded"], s.text_a, stored[s.path]["A_crnn"])
            for s in runs if s.text_b != stored[s.path]["B_padded"] or s.text_a != stored[s.path]["A_crnn"]]
    if mism:
        raise RuntimeError(f"frozen re-run does not reproduce stored scene results: {mism}")
    return {"scenes_checked": len(runs), "mismatches": 0}


# ====================================================================== cause attribution
def classify_errors(s: SceneRun, classes: set) -> List[Dict[str, object]]:
    """Pre-declared rule-based cause per character error, with evidence (manual review may override)."""
    rows = []
    if not s.regions:
        for gi, ch in enumerate(s.gt):
            rows.append({"op": "DEL", "gt_index": gi, "gt_char": ch, "pred_char": "", "segment": "",
                         "cause": "DETECTION_MISS", "evidence": "detector returned no region for the scene"})
        return rows
    ops = align(s.text_b, s.gt)
    heights = [e["box"][3] - e["box"][1] for e in s.pred_segments]
    med_h = float(np.median(heights)) if heights else 0.0
    for op, gi, pi in ops:
        if op == "MATCH":
            continue
        gch = s.gt[gi] if gi is not None else ""
        e = s.pred_segments[pi] if pi is not None else None
        pch = e["pred"] if e else ""
        cause, ev = "UNDETERMINED", ""
        if op == "SUB":
            w = e["box"][2] - e["box"][0]
            if gch not in classes:
                cause, ev = "UNKNOWN_CLASS", f"'{gch}' is not one of the trained classes; the closed-set classifier must output a trained class ('{pch}')"
            elif med_h and w > 1.5 * med_h:
                cause, ev = "SEGMENTATION_MERGE", f"segment width {w}px > 1.5 x median height {med_h:.0f}px"
            elif e["x"] is None:
                cause, ev = "PREPROCESSING_ARTIFACT", "no features could be extracted from the segment"
            else:
                cause, ev = "CHARACTER_CONFUSION", (f"one segment aligned 1:1 with '{gch}'; top-3 "
                                                    + ", ".join(f"{c}:{p:.2f}" for c, p in e["top3"]))
        elif op == "DEL":
            cause = "UNKNOWN_CLASS" if gch not in classes else "UNDETERMINED"
            ev = "ground-truth character has no aligned segment"
        elif op == "INS":
            cause, ev = "SEGMENTATION_SPLIT", f"extra segment '{pch}' with no ground-truth counterpart"
        rows.append({"op": op, "gt_index": gi if gi is not None else "", "gt_char": gch, "pred_char": pch,
                     "segment": f"r{e['region']}k{e['k']}" if e else "", "cause": cause, "evidence": ev})
    return rows


# ====================================================================== topology analysis
def topology_analysis(s: SceneRun, err: Dict[str, object], model, feats, groups, cfg) -> Optional[Dict[str, object]]:
    """Per-group log-likelihood of the TRUE vs PREDICTED class for one confusion error."""
    if err["cause"] != "CHARACTER_CONFUSION":
        return None
    e = next(x for x in s.pred_segments if f"r{x['region']}k{x['k']}" == err["segment"])
    x = np.array([e["x"][f] for f in feats])
    ci = {c: k for k, c in enumerate(model.classes)}
    g, p = err["gt_char"], err["pred_char"]
    row = {"scene": Path(s.path).name, "segment": err["segment"], "gt_char": g, "pred_char": p,
           "gt_in_top3": g in [c for c, _ in e["top3"]], "top3": " ".join(f"{c}:{q:.2f}" for c, q in e["top3"])}
    favour = {}
    for gn, fl in groups.items():
        idx = [feats.index(f) for f in fl]
        def ll(c):
            mu, sd = model.mean[ci[c], idx], model.sigma[ci[c], idx]
            return float((-0.5 * (((x[idx] - mu) / sd) ** 2 + np.log(sd ** 2))).mean())
        d = ll(g) - ll(p)
        row[f"{gn}_loglik_true_minus_pred"] = d
        favour[gn] = "TRUE" if d > 0 else "PRED"
    for f in ("beta_0", "beta_1", "euler_characteristic", "endpoints", "branch_points"):
        j = feats.index(f)
        row[f"{f}_sample"] = float(x[j])
        row[f"{f}_mean_true"] = float(model.mean[ci[g], j])
        row[f"{f}_mean_pred"] = float(model.mean[ci[p], j])
    topo_idx = [feats.index(f) for f in TOPO_FEATURES]
    pooled = np.sqrt((model.sigma[ci[g], topo_idx] ** 2 + model.sigma[ci[p], topo_idx] ** 2) / 2)
    sep = np.abs(model.mean[ci[g], topo_idx] - model.mean[ci[p], topo_idx]) / pooled
    row["max_topology_class_separation_sigma"] = float(sep.max())
    row["topology_can_distinguish_classes"] = bool(sep.max() >= 0.5)
    row["groups_favouring_true"] = ",".join(k for k, v in favour.items() if v == "TRUE") or "none"
    row["case_pair"] = g.lower() == p.lower() and g != p
    row["recoverability"] = (
        "needs size / line-relative information (case pair; shapes are scale-normalised)" if row["case_pair"] else
        "potentially recoverable: at least one feature group already favours the true class" if favour and any(v == "TRUE" for v in favour.values()) else
        "not recoverable with current features (no group favours the true class)")
    row["n_train_true"] = int(sum(1 for c in [model.classes[ci[g]]]))
    return row


# ====================================================================== independence audit
def independence_audit(runs: Sequence[SceneRun], items, cfg, inlier_min: int = 20) -> List[Dict[str, object]]:
    """Filename/metadata + transcription overlap + ORB geometric matching against same-text crops."""
    orb = cv2.ORB_create(1500)
    by_text = defaultdict(list)
    for it in items:
        if it.transcribed:
            by_text[it.gt].append(it.path)
    rows = []
    for s in runs:
        cands = by_text.get(s.gt, [])
        best, best_path = 0, ""
        scene = cv2.cvtColor(load_image(PROJECT_ROOT / s.path).image, cv2.COLOR_BGR2GRAY)
        ks, ds = orb.detectAndCompute(scene, None)
        for cp in cands:
            crop = load_image(PROJECT_ROOT / cp).image
            crop = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            kc, dc = orb.detectAndCompute(crop, None)
            if ds is None or dc is None or len(kc) < 8:
                continue
            m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(dc, ds)
            if len(m) < 8:
                continue
            src = np.float32([kc[x.queryIdx].pt for x in m]).reshape(-1, 1, 2)
            dst = np.float32([ks[x.trainIdx].pt for x in m]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
            inl = int(mask.sum()) if mask is not None else 0
            if inl > best:
                best, best_path = inl, cp
        rows.append({"scene": Path(s.path).name, "scene_gt": s.gt_raw,
                     "filename_link_to_crop": "none (scene_NNN vs formal_/informal_NNN_k; no source metadata)",
                     "crops_with_identical_transcription": len(cands),
                     "example_crops": " | ".join(Path(c).name for c in cands[:4]),
                     "max_orb_ransac_inliers": best, "best_matching_crop": Path(best_path).name if best_path else ""})
    # negative control: same scenes vs crops with DIFFERENT text (fixed seed) -> what inlier counts mean
    import random
    rng = random.Random(cfg.uncertainty.seed)
    texts = [(it.path, it.gt) for it in items if it.transcribed]
    neg = []
    for s in runs:
        scene = cv2.cvtColor(load_image(PROJECT_ROOT / s.path).image, cv2.COLOR_BGR2GRAY)
        ks, ds = orb.detectAndCompute(scene, None)
        for cp in rng.sample([p for p, t in texts if t != s.gt], 5):
            crop = load_image(PROJECT_ROOT / cp).image
            crop = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            kc, dc = orb.detectAndCompute(crop, None)
            if ds is None or dc is None or len(kc) < 8:
                neg.append(0)
                continue
            m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(dc, ds)
            if len(m) < 8:
                neg.append(0)
                continue
            H, mk = cv2.findHomography(np.float32([kc[x.queryIdx].pt for x in m]).reshape(-1, 1, 2),
                                       np.float32([ks[x.trainIdx].pt for x in m]).reshape(-1, 1, 2), cv2.RANSAC, 4.0)
            neg.append(int(mk.sum()) if mk is not None else 0)
    neg = np.array(neg)
    p95, mx = float(np.percentile(neg, 95)), float(neg.max())
    for r_ in rows:
        b = r_["max_orb_ransac_inliers"]
        r_["negative_control_p95"], r_["negative_control_max"] = p95, mx
        r_["verdict"] = ("LIKELY SAME RENDERING (exceeds every different-text control)" if b > mx else
                         "PROBABLE SAME RENDERING (exceeds different-text control 95th percentile)" if b > p95 else
                         "same text content, image match not above control" if r_["crops_with_identical_transcription"]
                         else "no crop with this text")
    rows.append({"scene": "_NEGATIVE_CONTROL_", "scene_gt": f"{len(neg)} scene-vs-different-text pairs",
                 "max_orb_ransac_inliers": mx, "negative_control_p95": p95, "negative_control_max": mx,
                 "verdict": f"median {np.median(neg):.0f}, fraction >= {inlier_min}: {(neg >= inlier_min).mean():.2f}"})
    return rows


@dataclass
class AuditResult:
    runs: List[SceneRun]
    failed: List[SceneRun]
    errors: List[Dict[str, object]]
    topo: List[Dict[str, object]]
    inventory: List[Dict[str, object]]
    unsupported: List[Dict[str, object]]
    independence: List[Dict[str, object]]
    ocr: Dict[str, object]
    verification: Dict[str, object]
    classes: List[str]
    report: Dict[str, object] = field(default_factory=dict)


def run_audit(cfg: AppConfig, say=print) -> AuditResult:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    runs, model, feats, groups, pool, items = rerun_scenes(cfg, say)
    verification = verify_against_stored(runs)
    say(f"[AUDIT] frozen re-run reproduces stored results for {verification['scenes_checked']} scenes")
    classes = sorted(set(model.classes))
    cls_set = set(classes)
    failed = [s for s in runs if s.text_b != s.gt]
    errors, topo = [], []
    for s in failed:
        for e in classify_errors(s, cls_set):
            e.update({"scene": Path(s.path).name, "gt": s.gt_raw, "pred": s.text_b, "ocr": s.text_a})
            errors.append(e)
            t = topology_analysis(s, e, model, feats, groups, cfg)
            if t:
                topo.append(t)
    counts = Counter(p[1] for p in pool)
    pool_groups = defaultdict(set)
    for g, lab, _ in pool:
        pool_groups[lab].add(g)
    inventory = [{"class": c, "samples": counts[c], "source_groups": len(pool_groups[c]),
                  "type": ("digit" if c.isdigit() else "uppercase" if c.isupper() else "lowercase" if c.islower()
                           else "symbol")} for c in classes]
    for sym in CHECK_SYMBOLS:
        if sym not in cls_set:
            inventory.append({"class": sym, "samples": 0, "source_groups": 0, "type": "MISSING (not trained)"})
    all_gt_chars = Counter(ch for s in runs for ch in s.gt)
    unsupported = [{"char": ch, "occurrences_in_scene_gt": n, "in_training_classes": ch in cls_set,
                    "in_crnn_charset": ch in CRNN_SUPPORTED,
                    "scenes": " ".join(sorted({Path(s.path).stem for s in runs if ch in s.gt}))}
                   for ch, n in sorted(all_gt_chars.items()) if ch not in cls_set or ch not in CRNN_SUPPORTED]
    ocr = Counter()
    ocr_rows = []
    for s in runs:
        a_ok = casefold_alnum(s.text_a) == casefold_alnum(s.gt)
        b_ok = s.text_b == s.gt
        outside = any(ch not in CRNN_SUPPORTED for ch in s.gt)
        cat = "both_correct" if a_ok and b_ok else "ocr_only_correct" if a_ok else "topology_only_correct" if b_ok else "both_wrong"
        ocr[cat] += 1
        ocr["gt_outside_crnn_charset"] += outside
        ocr_rows.append({"scene": Path(s.path).name, "gt": s.gt_raw, "A_crnn": s.text_a, "B_topology": s.text_b,
                         "A_correct_casefold_alnum": a_ok, "B_correct_exact": b_ok, "category": cat,
                         "gt_has_chars_outside_crnn_charset": outside})
    independence = independence_audit(runs, items, cfg)
    return AuditResult(runs, failed, errors, topo, inventory, unsupported, independence,
                       {"counts": dict(ocr), "rows": ocr_rows}, verification, classes)
