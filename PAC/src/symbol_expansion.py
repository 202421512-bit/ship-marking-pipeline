"""SYMBOL EXPANSION & SEGMENTATION REPAIR.

* Segmentation repair: opt-in generic attachment of small marks dropped by the text-band filter
  (dataset.segment_characters(attach=AttachConfig)). Baseline segmentation is byte-identical.
  Regression checks on the EXISTING crops / scenes are EXPLORATORY (scene_011 / scene_015 were the
  debugging cases).
* Class registry: 46 trained classes kept; + ← → ↑ ↓ REGISTERED. A registered class is never reported
  as TRAINABLE without real samples from >= min_train_groups source groups.
* New-data collection (--label-new-data) + group-aware train / dev / final-holdout split audit; images
  that reuse existing crop / scene content are never used as independent test data.
* Experimental classifier (baseline pool + new TRAIN samples) only when new classes have enough data;
  otherwise every new-symbol metric is NOT EVALUABLE. No synthetic training data is generated.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import DATA_DIR, PROJECT_ROOT, SCENE_RESULTS_DIR, SYMBOL_RESULTS_DIR, AppConfig
from .dataset import (FORMAL, INFORMAL, INFORMAL_SCENE, binarize_for_segmentation, normalize_transcription,
                      read_csv_rows, run_dataset_audit, segment_characters, write_csv_rows)
from .direction_aware_experiment import direction_features, fit_group_model
from .full_scene_stabilization import load_scene_labels, padded_crop
from .image_io import load_image
from .prototypes import extract_features
from .real_experiment import GEO_FEATURES, TOPO_FEATURES, char_canvas, feature_dict, write_csv
from .real_marking_recognition import edit_ops
from .text_detection_integration import Detector, build_pool, order_regions, segment_features

MANIFEST_COLUMNS = ["image_path", "transcription", "source_group", "acquisition_source", "capture_session",
                    "split", "readable", "uncertain", "registered_at", "note"]


# ====================================================================== shared model / features
def feature_names(cfg: AppConfig) -> Tuple[List[str], Dict[str, List[str]]]:
    feats = list(TOPO_FEATURES) + list(GEO_FEATURES) + list(cfg.direction.direction_features)
    groups = {"topology": list(TOPO_FEATURES), "geometry": list(GEO_FEATURES),
              "direction": list(cfg.direction.direction_features)}
    return feats, groups


def char_features(mask: np.ndarray, cfg: AppConfig) -> Dict[str, float]:
    f = feature_dict(extract_features(char_canvas(mask, cfg), cfg, with_persistence=True), with_ph=True)
    x = {k: float(f[k]) for k in TOPO_FEATURES + GEO_FEATURES}
    x.update(direction_features(char_canvas(mask, cfg), cfg))
    return x


# ====================================================================== 1. segmentation regression
@dataclass
class ImgRun:
    path: str
    kind: str
    gt: str
    gt_raw: str
    regions: int
    text: Dict[str, str]
    nseg: Dict[str, int]
    status: Dict[str, str]
    segs: Dict[str, list] = field(default_factory=dict, repr=False)
    crops: list = field(default_factory=list, repr=False)


def run_regression(cfg: AppConfig, say=print):
    """Baseline vs repaired segmentation through the frozen detection -> padded crop -> classifier pipeline."""
    det = Detector(cfg)
    items, pool = build_pool(cfg, say)
    feats, groups = feature_names(cfg)
    models: Dict[str, object] = {}

    def model_for(group):
        key = group or "__all__"
        if key not in models:
            tr = [p for p in pool if p[0] != group]
            X = np.array([[p[2][f] for f in feats] for p in tr])
            models[key] = fit_group_model(X, [p[1] for p in tr], feats, groups, {g: 1 / 3 for g in groups}, cfg)
        return models[key]

    labels = load_scene_labels(cfg)
    audit = run_dataset_audit(cfg, write=False, previews=False, verbose=False)
    out = []
    for m in sorted(audit.manifest, key=lambda r: str(r["image_path"])):
        if m["style"] not in (FORMAL, INFORMAL, INFORMAL_SCENE):
            continue
        path, kind = str(m["image_path"]), str(m["style"])
        img = load_image(PROJECT_ROOT / path).image
        img = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        raw = str(labels.get(path, {}).get("transcription", "")) if kind == INFORMAL_SCENE else str(m.get("transcription", "") or "")
        gt = normalize_transcription(raw, cfg.dataset) if raw.strip() else ""
        polys, confs = det.detect(img)
        regions = [{"index": k, "poly": p, "bbox": (float(p[:, 0].min()), float(p[:, 1].min()),
                                                     float(p[:, 0].max()), float(p[:, 1].max()))}
                   for k, p in enumerate(polys)]

        class _R:
            def __init__(self, d):
                self.d, self.bbox, self.line = d, d["bbox"], 0
        regions = [o.d for o in order_regions([_R(d) for d in regions], cfg.text_detection.same_line_overlap)[0]]
        model = model_for(str(m["group_id"]) if kind in (FORMAL, INFORMAL) else None)
        ir = ImgRun(path, kind, gt, raw, len(regions), {}, {}, {})
        crops = [padded_crop(img, g["poly"], cfg.scene.pad_ratio, cfg.scene.rotation_threshold_deg) for g in regions]
        ir.crops = crops
        for cond, att in (("baseline", None), ("repaired", cfg.attach)):
            text, n, segs = "", 0, []
            for cr in crops:
                seg, fs = segment_features(cr.image, cfg, attach=att)
                segs.append(seg)
                for f in fs:
                    n += 1
                    if f is None:
                        text += "?"
                        continue
                    lp = model.log_posterior(np.array([[f[k] for k in feats]]))[0]
                    text += model.classes[int(np.argmax(lp))]
            ir.text[cond], ir.nseg[cond], ir.segs[cond] = text, n, segs
            ir.status[cond] = output_state(ir, cond, set(model.classes))
        out.append(ir)
    return out, models, pool, items


def output_state(ir: ImgRun, cond: str, trained: set) -> str:
    """Per-image output state (evaluation-side; the model itself only ever outputs trained classes)."""
    if ir.regions == 0:
        return "NO_DETECTION"
    if not ir.gt:
        return "RECOGNIZED_CANDIDATE (not evaluated: no ground truth)"
    if any(c not in trained for c in ir.gt):
        return "NOT_EVALUATED_UNSUPPORTED (ground truth contains untrained class)"
    if ir.nseg[cond] != len(ir.gt):
        return "SEGMENTATION_FAILURE"
    return "RECOGNIZED_CANDIDATE"


def verify_baseline(runs: Sequence[ImgRun]) -> Dict[str, int]:
    stored = {r["image_path"]: r["B_padded"] for r in read_csv_rows(SCENE_RESULTS_DIR / "scene_recognition_results.csv")}
    bad = [r.path for r in runs if stored.get(r.path) != r.text["baseline"]]
    if bad:
        raise RuntimeError(f"baseline arm does not reproduce stored full_scene_stabilization strings: {bad[:5]}")
    return {"images_checked": len(runs), "mismatches": 0}


def regression_metrics(runs: Sequence[ImgRun]) -> Dict[str, object]:
    def block(sel):
        n = sum(len(r.gt) for r in sel)
        return {c: {"images": len(sel), "cer": sum(edit_ops(r.text[c], r.gt)[0] for r in sel) / n if n else float("nan"),
                    "exact": float(np.mean([r.text[c] == r.gt for r in sel])) if sel else float("nan"),
                    "segment_count_ok": int(sum(r.nseg[c] == len(r.gt) for r in sel))}
                for c in ("baseline", "repaired")}
    crops = [r for r in runs if r.kind in (FORMAL, INFORMAL) and r.gt]
    scenes = [r for r in runs if r.kind == INFORMAL_SCENE and r.gt]
    dec = [r for r in runs if r.gt and "." in r.gt]
    changed = [r for r in runs if r.text["baseline"] != r.text["repaired"] or r.nseg["baseline"] != r.nseg["repaired"]]
    return {"crops": block(crops), "scenes": block(scenes), "decimal_point_images": block(dec),
            "changed_images": len(changed),
            "changed": [{"image": Path(r.path).name, "gt": r.gt_raw, "baseline": r.text["baseline"],
                         "repaired": r.text["repaired"], "segments": f"{r.nseg['baseline']}->{r.nseg['repaired']}"}
                        for r in changed]}


def touching_metrics(runs: Sequence[ImgRun], items) -> Dict[str, object]:
    flag = {it.path: bool(it.flags.get("possible_touching")) for it in items}
    sel = [r for r in runs if r.gt and flag.get(r.path)]
    n = sum(len(r.gt) for r in sel)
    return {c: {"images": len(sel), "exact": float(np.mean([r.text[c] == r.gt for r in sel])) if sel else float("nan"),
                "cer": sum(edit_ops(r.text[c], r.gt)[0] for r in sel) / n if n else float("nan"),
                "segment_count_ok": int(sum(r.nseg[c] == len(r.gt) for r in sel))} for c in ("baseline", "repaired")}


def dot_check(runs: Sequence[ImgRun], cfg: AppConfig, scenes=("scene_011", "scene_015")) -> List[Dict[str, object]]:
    """Debugging cases (NOT independent): is the mark above the stem now part of the character?"""
    rows = []
    for r in runs:
        if Path(r.path).stem not in scenes:
            continue
        for cond in ("baseline", "repaired"):
            seg = r.segs[cond][0]
            masks = seg.char_masks
            k = r.gt.find("i")
            b0 = None
            if 0 <= k < len(masks):
                n, _ = cv2.connectedComponents(masks[k].astype(np.uint8), connectivity=8)
                b0 = n - 1
            rows.append({"scene": Path(r.path).name, "condition": cond, "gt": r.gt_raw, "prediction": r.text[cond],
                         "i_segment_components(beta0)": b0, "dot_preserved": b0 == 2,
                         "exact": r.text[cond] == r.gt, "note": "debugging case - not independent evidence"})
    return rows


# ====================================================================== 2. class registry + unicode
def class_inventory(pool, cfg: AppConfig, new_counts: Dict[str, Dict[str, int]]) -> List[Dict[str, object]]:
    cnt = Counter(p[1] for p in pool)
    grp = defaultdict(set)
    for g, lab, _ in pool:
        grp[lab].add(g)
    rows = [{"class": c, "codepoint": " ".join(f"U+{ord(ch):04X}" for ch in c), "origin": "existing (46)",
             "samples": cnt[c], "source_groups": len(grp[c]),
             "status": "TRAINED (baseline pool)" if len(grp[c]) >= 2 else "TRAINED (single source group: never testable)"}
            for c in sorted(cnt)]
    s = cfg.symbols
    for c in s.new_classes:
        nc = new_counts.get(c, {})
        tg, tn = nc.get("train_groups", 0), nc.get("train_samples", 0)
        status = ("TRAINABLE (new data)" if tg >= s.min_train_groups and tn >= s.min_train_samples else
                  "REGISTERED - INSUFFICIENT DATA" if tn else "REGISTERED - NO DATA (cannot be output by any model)")
        rows.append({"class": c, "codepoint": " ".join(f"U+{ord(ch):04X}" for ch in c), "origin": "registered (new)",
                     "samples": tn, "source_groups": tg, "status": status,
                     "dev_groups": nc.get("dev_groups", 0), "test_groups": nc.get("test_groups", 0)})
    return rows


def unicode_checks(cfg: AppConfig, out: Path) -> Dict[str, object]:
    """Round-trip the arrow symbols through every storage / display path used by the project."""
    syms = "".join(cfg.symbols.new_classes)
    res = {}
    p = out / "_unicode_check.csv"
    write_csv_rows(p, ["text"], [{"text": syms}])
    res["csv_utf8_bom_roundtrip"] = read_csv_rows(p)[0]["text"] == syms
    p.unlink()
    res["json_roundtrip"] = json.loads(json.dumps({"t": syms}, ensure_ascii=False))["t"] == syms
    res["transcription_normalization_keeps_symbols"] = normalize_transcription(f"F8 {syms}", cfg.dataset) == f"F8{syms}"
    res["string_reconstruction_concat"] = "".join(list(syms)) == syms
    import logging
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    msgs = []
    h = logging.Handler()
    h.emit = lambda rec: msgs.append(rec.getMessage())
    logging.getLogger("matplotlib").addHandler(h)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fig = plt.figure()
        fig.text(0.1, 0.5, syms)
        fig.savefig(out / "_glyph_check.png")
        plt.close(fig)
    logging.getLogger("matplotlib").removeHandler(h)
    (out / "_glyph_check.png").unlink()
    res["matplotlib_glyphs_render"] = not any("missing" in m.lower() for m in msgs + [str(x.message) for x in w])
    res["classifier_label_encoding"] = "labels are plain Python str keys (any Unicode); no integer re-encoding"
    return res


# ====================================================================== 3. new data manifest + split audit
def load_manifest(cfg: AppConfig) -> List[Dict[str, str]]:
    return read_csv_rows(DATA_DIR / cfg.symbols.manifest_file)


def auto_split(group: str, cfg: AppConfig) -> str:
    """Deterministic group-level split (hash of source_group) when the user left split empty."""
    h = int(hashlib.sha256(f"{cfg.symbols.seed}:{group}".encode()).hexdigest(), 16) % 10_000 / 10_000
    a, b, _ = cfg.symbols.split_fractions
    return "train" if h < a else "dev" if h < a + b else "test"


def dhash(img: np.ndarray, cfg: AppConfig, size: int = 16) -> int:
    """Difference hash of the binarized INK pattern (ink bounding box), not of the background.

    A whole-image hash is dominated by plain painted backgrounds and flags unrelated images as duplicates."""
    mask, _, _ = binarize_for_segmentation(img, cfg.preprocessing, cfg.dataset)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    s = cv2.resize(crop, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


class SplitLeakageError(RuntimeError):
    """A source group appears in more than one split."""


def split_audit(cfg: AppConfig) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Resolve splits, check group isolation and content reuse against existing crops / scenes."""
    rows = load_manifest(cfg)
    if not rows:
        return [], []
    existing_hash = {}
    existing_text = set()
    for p in list((PROJECT_ROOT / "data/input/Symbols").rglob("*.png")):
        existing_hash[str(p)] = dhash(load_image(p).image, cfg)
    for m in read_csv_rows(DATA_DIR / "dataset_manifest.csv"):
        if m.get("normalized_transcription"):
            existing_text.add(m["normalized_transcription"])
    for r in load_scene_labels(cfg).values():
        if r.get("transcription"):
            existing_text.add(normalize_transcription(r["transcription"], cfg.dataset))
    resolved, groups = [], defaultdict(set)
    for r in rows:
        g = (r.get("source_group") or "").strip()
        if not g:
            raise SplitLeakageError(f"{r['image_path']}: source_group is required")
        split = (r.get("split") or "").strip().lower() or auto_split(g, cfg)
        groups[g].add(split)
        img = load_image(PROJECT_ROOT / r["image_path"]).image
        hd = dhash(img, cfg)
        near = min((bin(hd ^ v).count("1"), k) for k, v in existing_hash.items())
        text = normalize_transcription(r.get("transcription", ""), cfg.dataset)
        reuse = near[0] <= cfg.symbols.dhash_near_duplicate_bits or text in existing_text
        resolved.append({**r, "split_resolved": split, "transcription_normalized": text,
                         "nearest_existing_image": Path(near[1]).name, "dhash_distance": near[0],
                         "text_matches_existing": text in existing_text, "content_reuse_suspected": reuse,
                         "eligible_for_final_holdout": split == "test" and not reuse})
    bad = {g: s for g, s in groups.items() if len(s) > 1}
    if bad:
        raise SplitLeakageError(f"source groups in more than one split: {bad}")
    audit_rows = [{"source_group": g, "splits": "|".join(sorted(s)),
                   "images": sum(1 for r in resolved if r["source_group"].strip() == g),
                   "content_reuse_images": sum(1 for r in resolved if r["source_group"].strip() == g and r["content_reuse_suspected"])}
                  for g, s in sorted(groups.items())]
    return resolved, audit_rows


# ====================================================================== 4. symbol features
def symbol_feature_analysis(runs: Sequence[ImgRun], new_rows, cfg: AppConfig, model) -> List[Dict[str, object]]:
    """Features of every available instance of the registered symbols vs existing class prototypes."""
    feats, groups = feature_names(cfg)
    ci = {c: k for k, c in enumerate(model.classes)}
    syms = set(cfg.symbols.new_classes)
    out = []

    def describe(mask, label, source, kind):
        x = char_features(mask, cfg)
        v = np.array([x[f] for f in feats])
        z = np.abs((v[None, :] - model.mean) / model.sigma).mean(axis=1)
        order = np.argsort(z)[:3]
        lp = model.log_posterior(v[None, :])[0]
        return {"symbol": label, "source": source, "evidence_type": kind,
                "beta_0": x["beta_0"], "beta_1": x["beta_1"], "euler": x["euler_characteristic"],
                "endpoints": x["endpoints"], "branch_points": x["branch_points"],
                "endpoint_x_mean": x["endpoint_x_mean"], "endpoint_y_mean": x["endpoint_y_mean"],
                "left_right_asymmetry": x["proj_col_left"] - x["proj_col_right"],
                "top_bottom_asymmetry": x["proj_row_top"] - x["proj_row_bottom"],
                "orient_h": x["orient_h"], "orient_v": x["orient_v"], "orient_d45": x["orient_d45"],
                "orient_d135": x["orient_d135"],
                "nearest_existing_classes(mean|z|)": " ".join(f"{model.classes[j]}:{z[j]:.2f}" for j in order),
                "baseline_classifier_output": model.classes[int(np.argmax(lp))]}

    for r in runs:
        if r.kind != INFORMAL_SCENE or not r.gt or not any(c in syms for c in r.gt):
            continue
        masks = [m for s in r.segs["repaired"] for m in s.char_masks]
        if len(masks) != len(r.gt):
            out.append({"symbol": "".join(c for c in r.gt if c in syms), "source": Path(r.path).name,
                        "evidence_type": "segmentation count mismatch - not analysed"})
            continue
        for ch, mk in zip(r.gt, masks):
            if ch in syms:
                out.append(describe(mk, ch, Path(r.path).name, "existing scene (debug instance, NOT training data)"))
    for r in new_rows:
        if not r.get("transcription_normalized"):
            continue
        img = load_image(PROJECT_ROOT / r["image_path"]).image
        seg = segment_characters(img, cfg.preprocessing, cfg.dataset, attach=cfg.attach)
        t = r["transcription_normalized"]
        if seg.count != len(t):
            continue
        for ch, mk in zip(t, seg.char_masks):
            if ch in syms:
                out.append(describe(mk, ch, Path(r["image_path"]).name, f"new data ({r['split_resolved']})"))
    return out


# ====================================================================== 5. experimental classifier
def experimental_evaluation(new_rows, pool, cfg: AppConfig) -> Dict[str, object]:
    """Train baseline pool + new TRAIN characters; evaluate dev and eligible final-holdout images."""
    s = cfg.symbols
    feats, groups = feature_names(cfg)
    if not new_rows:
        return {"status": "NOT EVALUABLE (no new images registered)", "model_trained": False}
    samples = defaultdict(list)          # split -> [(group, image, gt, segment masks)]
    for r in new_rows:
        if str(r.get("readable", "True")).lower() in ("false", "0", "no") or not r["transcription_normalized"]:
            continue
        img = load_image(PROJECT_ROOT / r["image_path"]).image
        seg = segment_characters(img, cfg.preprocessing, cfg.dataset, attach=cfg.attach)
        samples[r["split_resolved"]].append((r["source_group"], r["image_path"], r["transcription_normalized"], seg,
                                              r["eligible_for_final_holdout"]))
    train_chars = [(g, ch, char_features(mk, cfg)) for g, _, t, seg, _ in samples["train"]
                   if seg.count == len(t) for ch, mk in zip(t, seg.char_masks)]
    per_class = defaultdict(lambda: {"train_samples": 0, "train_groups": set(), "dev_groups": set(), "test_groups": set()})
    for g, ch, _ in train_chars:
        per_class[ch]["train_samples"] += 1
        per_class[ch]["train_groups"].add(g)
    for split in ("dev", "test"):
        for g, _, t, _, _ in samples[split]:
            for ch in set(t):
                per_class[ch][f"{split}_groups"].add(g)
    new_counts = {c: {"train_samples": v["train_samples"], "train_groups": len(v["train_groups"]),
                      "dev_groups": len(v["dev_groups"]), "test_groups": len(v["test_groups"])} for c, v in per_class.items()}
    trainable = [c for c in s.new_classes if new_counts.get(c, {}).get("train_groups", 0) >= s.min_train_groups
                 and new_counts.get(c, {}).get("train_samples", 0) >= s.min_train_samples]
    res = {"new_counts": new_counts, "trainable_new_classes": trainable,
           "train_images": len(samples["train"]), "dev_images": len(samples["dev"]), "test_images": len(samples["test"])}
    if not trainable:
        res.update({"status": "NOT EVALUABLE (no registered class reaches the training threshold)", "model_trained": False})
        return res
    use = [(g, ch, x) for g, ch, x in train_chars if ch in trainable or ch not in s.new_classes]
    tr = [(p[0], p[1], p[2]) for p in pool] + use
    X = np.array([[p[2][f] for f in feats] for p in tr])
    model = fit_group_model(X, [p[1] for p in tr], feats, groups, {g: 1 / 3 for g in groups}, cfg)
    train_groups = {g for g, _, _ in use}

    def evaluate(split):
        rows = [x for x in samples[split] if split != "test" or x[4]]
        if any(g in train_groups for g, *_ in rows):
            raise SplitLeakageError(f"{split} group also in training")
        out, conf = [], Counter()
        for g, path, t, seg, _ in rows:
            pred = "".join(model.classes[int(np.argmax(model.log_posterior(np.array([[char_features(mk, cfg)[f]
                           for f in feats]]))[0]))] for mk in seg.char_masks)
            out.append((t, pred))
            if seg.count == len(t):
                for a, b in zip(t, pred):
                    conf[(a, b)] += 1
        n = sum(len(t) for t, _ in out)
        return {"images": len(out), "exact": float(np.mean([t == p for t, p in out])) if out else "NOT EVALUABLE",
                "cer": sum(edit_ops(p, t)[0] for t, p in out) / n if n else "NOT EVALUABLE",
                "symbol_confusion": {f"{a}->{b}": v for (a, b), v in conf.items() if a in s.new_classes}}
    res.update({"status": "TRAINED (experimental)", "model_trained": True, "dev": evaluate("dev"),
                "final_holdout": evaluate("test"),
                "per_class_generalization": {c: ("EVALUABLE" if new_counts.get(c, {}).get("test_groups", 0) >= 1 and c in trainable
                                                 else "NOT EVALUABLE (no independent train AND test source groups)")
                                             for c in s.new_classes}})
    return res
