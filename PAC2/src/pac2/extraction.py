"""Phase 4: handwritten-stroke extraction with the Phase 3 shape models (no OCR, no symbol classes).

image -> stroke-scale candidate mask (multi-scale top/black-hat + thinness filter; the Phase 2 >= 2/4 consensus is kept
for comparison only) -> connected components -> proximity groups
-> 7 shape features per group (same domain-matched normalization as training, H_t from the checkpoint)
-> P(handwritten) from a checkpoint (S1 / S3 / S4 logistic or Random Forest) -> keep / reject by an operating threshold.
Only real foreground stroke pixels of accepted groups enter the final mask (never filled bounding boxes).

Caveats carried into every output: the models were trained on WHOLE crops (one marking per image), so groups found in a
scene are a shifted input distribution (OOD flags are recorded); probabilities are not calibrated on steel plates.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import joblib
import numpy as np

from . import ROOT
from .candidates import candidates, read_bgr
from .features import FEATURES, components, extract
from .normalize import normalize_mask

CKPT_DIR = ROOT / "models" / "phase4"
MODELS = ("S1_basic_unweighted", "S3_inner_cv_logistic", "S4_hard_example_reweighted", "RF_random_forest")


class TestSetAccessError(RuntimeError):
    __test__ = False          # not a pytest test class


def locked_test_paths() -> set:
    """Resolved paths of the locked Test 30 images (from the frozen split manifest)."""
    man = ROOT / "data" / "splits" / "split_manifest.csv"
    with open(man, encoding="utf-8-sig") as f:
        return {str((ROOT / r["path"]).resolve()).lower() for r in csv.DictReader(f) if r["set"] == "test"}


def guard_not_test(path: Path) -> None:
    if str(Path(path).resolve()).lower() in locked_test_paths():
        raise TestSetAccessError(f"{path} belongs to the locked Test 30 set")


# ---------------------------------------------------------------- checkpoints
def save_checkpoint(name: str, ck: dict, model_obj=None, folder: Path = CKPT_DIR) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    if model_obj is not None:
        joblib.dump(model_obj, folder / f"{name}.joblib")
        ck["model_file"] = f"{name}.joblib"
    p = folder / f"{name}.json"
    p.write_text(json.dumps(ck, indent=1), encoding="utf-8")
    return p


def load_checkpoint(name: str, folder: Path = CKPT_DIR) -> dict:
    ck = json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))
    required = {"model", "kind", "features", "candidate_method", "normalization", "H_t", "impute_median", "mu", "sigma", "thresholds",
                "trained_on"}
    miss = required - set(ck)
    if miss:
        raise ValueError(f"checkpoint {name} incomplete: {sorted(miss)}")
    if ck["features"] != FEATURES:
        raise ValueError(f"checkpoint feature order {ck['features']} != current {FEATURES}")
    if ck["trained_on"].get("test_images", 0) != 0:
        raise ValueError("checkpoint was trained with test images")
    if ck["kind"] == "random_forest":
        ck["_model"] = joblib.load(folder / ck["model_file"])
    return ck


def predict_proba(ck: dict, X_raw: np.ndarray) -> np.ndarray:
    med, mu, sd = (np.array(ck[k], float) for k in ("impute_median", "mu", "sigma"))
    Z = (np.where(np.isfinite(X_raw), X_raw, med) - mu) / sd
    if ck["kind"] == "random_forest":
        return ck["_model"].predict_proba(Z)[:, 1]
    return 1.0 / (1.0 + np.exp(-(Z @ np.array(ck["w"]) + ck["b"])))


def zscores(ck: dict, X_raw: np.ndarray) -> np.ndarray:
    med, mu, sd = (np.array(ck[k], float) for k in ("impute_median", "mu", "sigma"))
    return (np.where(np.isfinite(X_raw), X_raw, med) - mu) / sd


# ---------------------------------------------------------------- candidates and groups
STROKE_CFG = {"hat_kernels": (9, 15, 25, 41), "thinness_max": 0.35, "max_area_frac": 0.50, "min_area_px": 12,
              "noise_k": 5.0, "min_response": 10.0}
# changed after inspecting the first runs (no labels exist; disclosed in the Phase 4 notes):
#  * max_area_frac 0.10 -> 0.50: in 43 px font crops a single glyph exceeds 10 % of the image and was deleted (F8 -> 0 px)
#  * grouping: components longer than 40 % of the image side no longer link other components (long chalk / plate lines
#    merged a whole scene into one group in weld_marking_W79)
# changed once after the first Phase 4 run: threshold max(Otsu, mean + 3 SD) -> max(10, median + 5 x 1.4826 x MAD), because
# mean + 3 SD exceeded the stroke response on development crops where ink covers a large share of the image (0 candidates).


def stroke_candidates(bgr: np.ndarray, scfg: dict = STROKE_CFG) -> Dict[str, np.ndarray]:
    """Stroke-scale candidates for scenes (Phase 4 addition; designed without labels).

    1. multi-scale top-hat (bright ink) or black-hat (dark ink; polarity by the existing ink_polarity rule): the maximum
       response over kernels 9-41 px only keeps structures NARROWER than the kernel, so large rust patches / lit areas
       do not respond as wholes.
    2. threshold = max(10, median + 5 x 1.4826 x MAD) of the response (robust noise level).
    3. component filter: area >= min_area_px, area <= 10 % of the image, thinness = median stroke width (2 x distance
       transform on the skeleton) / max(bbox side) <= 0.35 (blobs are rejected, strokes and lines are kept).
    """
    from skimage.morphology import skeletonize

    from .candidates import ink_polarity
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    op = cv2.MORPH_BLACKHAT if ink_polarity(gray) == "dark" else cv2.MORPH_TOPHAT
    resp = np.zeros(gray.shape, np.float32)
    for k in scfg["hat_kernels"]:
        if k >= min(gray.shape):
            continue
        resp = np.maximum(resp, cv2.morphologyEx(gray, op, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32))
    r8 = np.clip(resp, 0, 255).astype(np.uint8)
    med = float(np.median(resp))
    mad = float(np.median(np.abs(resp - med)))
    thr = max(scfg["min_response"], med + scfg["noise_k"] * 1.4826 * mad)     # robust noise level: unaffected by a large ink share
    raw = (resp > thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    dt = cv2.distanceTransform(raw, cv2.DIST_L2, 5)
    sk = skeletonize(raw > 0)
    keep = np.zeros(n, bool)
    rejected_blob = np.zeros(raw.shape, bool)
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a < scfg["min_area_px"]:
            continue
        sel = lab == i
        if a > scfg["max_area_frac"] * raw.size:
            rejected_blob |= sel
            continue
        sw = 2.0 * float(np.median(dt[sel & sk])) if (sel & sk).any() else 2.0 * float(dt[sel].max())
        if sw / max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]) <= scfg["thinness_max"]:
            keep[i] = True
        else:
            rejected_blob |= sel
    stroke = (keep[lab] * 255).astype(np.uint8)
    return {"stroke_response": r8, "stroke_raw": raw * 255, "stroke": stroke, "rejected_blob": (rejected_blob * 255).astype(np.uint8),
            "polarity": "dark" if op == cv2.MORPH_BLACKHAT else "bright", "threshold": thr}


def candidate_mask(bgr: np.ndarray, cfg: dict, min_votes: int = 2) -> Dict[str, np.ndarray]:
    """Phase 2 candidates + their >= 2/4 consensus (kept for comparison) and the Phase 4 stroke candidates.
    The 'final' candidate used for grouping is the stroke-scale mask."""
    c = candidates(bgr, dict(cfg["candidates"], methods=["gray_otsu", "clahe_adaptive", "lab_bg_distance", "morph_hat"]))
    votes = sum((m > 0).astype(np.uint8) for m in c.values())
    c["consensus"] = ((votes >= min_votes) * 255).astype(np.uint8)
    s = stroke_candidates(bgr)
    c.update({k: v for k, v in s.items() if isinstance(v, np.ndarray)})
    c["final"] = s["stroke"]
    c["_meta"] = {"polarity": s["polarity"], "stroke_threshold": round(s["threshold"], 2), **{k: list(v) if isinstance(v, tuple) else v for k, v in STROKE_CFG.items()}}
    return c


def group_components(mask: np.ndarray, link_ratio: float = 0.6, min_area: int = 8, long_frac: float = 0.4) -> List[dict]:
    """Proximity grouping: components whose bounding boxes are closer than link_ratio x (median height of the non-long
    components) are linked (union-find). Long line-shaped components (bbox side > long_frac x image side AND aspect >= 4,
    e.g. plate edges or long chalk lines) stay separate groups and never link others. Groups keep member labels in
    ORIGINAL image coordinates."""
    m = (mask > 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    idx = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_area]
    if not idx:
        return []
    Hm, Wm = m.shape
    def line_like(i):
        w_, h_ = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        return (w_ > long_frac * Wm or h_ > long_frac * Hm) and max(w_, h_) / max(1, min(w_, h_)) >= 4.0
    is_long = {i: line_like(i) for i in idx}       # long AND line-shaped (characters in crops are long but not line-shaped)
    if len(idx) == 1:
        is_long = {i: False for i in idx}          # a single component (e.g. a crop) is never isolated as 'long'
    short = [i for i in idx if not is_long[i]] or idx
    hs = np.array([st[i, cv2.CC_STAT_HEIGHT] for i in short])
    gap = max(2.0, link_ratio * float(np.median(hs)))
    parent = {i: i for i in idx}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    boxes = {i: (st[i, 0], st[i, 1], st[i, 0] + st[i, 2], st[i, 1] + st[i, 3]) for i in idx}
    for a_i, a in enumerate(idx):
        ax0, ay0, ax1, ay1 = boxes[a]
        if is_long[a]:
            continue
        for b in idx[a_i + 1:]:
            if is_long[b]:
                continue
            bx0, by0, bx1, by1 = boxes[b]
            dx = max(0, max(bx0 - ax1, ax0 - bx1))
            dy = max(0, max(by0 - ay1, ay0 - by1))
            if dx <= gap and dy <= gap:
                parent[find(a)] = find(b)
    groups: Dict[int, List[int]] = {}
    for i in idx:
        groups.setdefault(find(i), []).append(i)
    out = []
    for gid, members in enumerate(sorted(groups.values(), key=lambda g: min(boxes[i][0] for i in g))):
        gm = np.isin(lab, members)
        ys, xs = np.nonzero(gm)
        out.append({"group_id": gid, "members": members, "n_cc": len(members), "pixels": int(gm.sum()),
                    "long_structure": bool(len(members) == 1 and is_long[members[0]]),
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], "mask": gm})
    return out


# ---------------------------------------------------------------- extraction
def score_groups(groups: List[dict], cks: Dict[str, dict], cfg: dict, dev_ranges: dict) -> List[dict]:
    rows = []
    any_ck = next(iter(cks.values()))
    for g in groups:
        x0, y0, x1, y1 = g["bbox"]
        crop = (g["mask"][y0:y1, x0:x1] * 255).astype(np.uint8)
        feats = {}
        by_h = {}
        for ck in cks.values():
            h = ck["H_t"]
            if h not in by_h:
                by_h[h] = extract(normalize_mask(crop, h), cfg["features"])
        r = {"group_id": g["group_id"], "bbox_x0y0x1y1": g["bbox"], "n_cc": g["n_cc"], "pixels": g["pixels"],
             "height_px": y1 - y0, "width_px": x1 - x0, "long_structure": int(g.get("long_structure", False))}
        f0 = by_h[any_ck["H_t"]]
        for k in FEATURES:
            r[k] = f0[k]
            r[f"{k}_missing"] = f0[f"{k}_missing"]
        r["n_missing"] = int(sum(f0[f"{k}_missing"] for k in FEATURES))
        ood = []
        if not dev_ranges["n_cc"][0] <= g["n_cc"] <= dev_ranges["n_cc"][1]:
            ood.append(f"n_cc {g['n_cc']} outside dev range {dev_ranges['n_cc']}")
        asp = (x1 - x0) / max(1, y1 - y0)
        if not dev_ranges["aspect"][0] <= asp <= dev_ranges["aspect"][1]:
            ood.append(f"aspect {asp:.2f} outside dev range [{dev_ranges['aspect'][0]:.2f}, {dev_ranges['aspect'][1]:.2f}]")
        for name, ck in cks.items():
            fx = by_h[ck["H_t"]]
            X = np.array([[fx[k] for k in FEATURES]], float)
            r[f"p_{name}"] = round(float(predict_proba(ck, X)[0]), 5)
            if name == "S3_inner_cv_logistic":
                z = zscores(ck, X)[0]
                big = [f"{FEATURES[j]} z={z[j]:+.1f}" for j in range(7) if abs(z[j]) > 4]
                if big:
                    ood.append("feature |z|>4: " + ", ".join(big))
                for j, k in enumerate(FEATURES):
                    r[f"z_{k}"] = round(float(z[j]), 4)
        r["ood_flags"] = "; ".join(ood)
        r["reliability"] = "LOW" if (ood or r["n_missing"] >= 3) else "MEDIUM" if r["n_missing"] else "OK"
        rows.append(r)
    return rows


def extract_image(path: Path, cks: Dict[str, dict], cfg: dict, dev_ranges: dict, primary: str, modes: Dict[str, float],
                  out_dir: Optional[Path] = None) -> dict:
    guard_not_test(path)
    t0 = time.perf_counter()
    bgr = read_bgr(path)
    cand = candidate_mask(bgr, cfg)
    cmask = cand["final"]
    groups = group_components(cmask)
    rows = score_groups(groups, cks, cfg, dev_ranges)
    H, W = cmask.shape
    masks = {}
    for mode, thr in modes.items():
        acc = np.zeros((H, W), bool)
        for g, r in zip(groups, rows):
            keep = r[f"p_{primary}"] >= thr
            r[f"keep_{mode}"] = int(keep)
            if keep:
                acc |= g["mask"]
        masks[mode] = acc
    for name, ck in cks.items():           # per-model decisions at each model's own recall-priority threshold
        for r in rows:
            r[f"keep_{name}@recall"] = int(r[f"p_{name}"] >= ck["thresholds"]["recall_priority"])
    elapsed = time.perf_counter() - t0
    res = {"path": str(path), "bgr": bgr, "candidates": cand, "candidate_mask": cmask, "groups": groups, "rows": rows,
           "masks": masks, "seconds": elapsed}
    if out_dir is not None:
        save_outputs(res, out_dir, cfg, primary, modes)
    return res


def save_outputs(res: dict, out_dir: Path, cfg: dict, primary: str, modes: Dict[str, float]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    bgr, cmask = res["bgr"], res["candidate_mask"]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    pre = cv2.createCLAHE(cfg["candidates"]["clahe_clip"], (cfg["candidates"]["clahe_tile"],) * 2).apply(gray)
    main_mode = "recall_priority"
    acc = res["masks"][main_mode]
    rej = (cmask > 0) & ~acc
    w = lambda name, im: cv2.imencode(".png", im)[1].tofile(str(out_dir / name))  # noqa: E731
    w("original.png", bgr)
    w("preprocessed.png", pre)
    w("candidate_mask.png", cmask)
    w("candidate_consensus_phase2.png", res["candidates"]["consensus"])
    w("candidate_rejected_blobs.png", res["candidates"]["rejected_blob"])
    for mode in modes:
        w(f"accepted_mask_{mode}.png", (res["masks"][mode] * 255).astype(np.uint8))
    w("accepted_mask.png", (acc * 255).astype(np.uint8))
    w("rejected_mask.png", (rej * 255).astype(np.uint8))
    w("final_mask.png", (acc * 255).astype(np.uint8))
    ov = bgr.copy()
    ov[rej] = (0.5 * ov[rej] + 0.5 * np.array([40, 40, 214])).astype(np.uint8)
    ov[acc] = (0, 200, 0)
    w("overlay.png", ov)
    rgba = np.dstack([bgr, (acc * 255).astype(np.uint8)])
    w("handwritten_only.png", rgba)
    cols = list(dict.fromkeys(k for r in res["rows"] for k in r))
    with open(out_dir / "candidate_scores.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(res["rows"])
    (out_dir / "run_info.json").write_text(json.dumps({
        "source": res["path"], "seconds": round(res["seconds"], 3), "primary_model": primary, "modes": modes,
        "main_mode_for_final_mask": main_mode, "candidate_rule": "multi-scale top/black-hat + thinness filter (Phase 4)",
        "candidate_meta": res["candidates"]["_meta"],
        "n_groups": len(res["groups"]), "image_size": list(cmask.shape)}, indent=1), encoding="utf-8")
