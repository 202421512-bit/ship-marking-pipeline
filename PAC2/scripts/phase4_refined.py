"""Phase 4 refined: group decision (unchanged checkpoints, RF primary, recall-priority threshold) + stroke-only refinement.

No training; Test 30 never read (guard in extract_image). Field images with full-dev checkpoints; one development
image (V8, handwritten_013) with the fold model that excluded it (for the 'short neat mark' failure case).
Raw outputs: reports/phase4_refined/inference/<name>/ ; presentation: results/phase4_refined/{figures,examples}/
Run: .venv\\Scripts\\python.exe scripts\\phase4_refined.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pac2 import ROOT, load_config  # noqa: E402
from pac2.extraction import CKPT_DIR, MODELS, extract_image, load_checkpoint  # noqa: E402
from pac2.refine import FINAL_METHOD, REFINE_CFG, proxies, refine_methods, select  # noqa: E402
from pac2.visualization import ERR, FINAL, plt  # noqa: E402

RAW = ROOT / "reports" / "phase4_refined" / "inference"
RES = ROOT / "results" / "phase4_refined"
FIELD = ROOT / "data" / "field_test" / "images"


def disp(n: str) -> str:
    return n.replace("예시", "example")


def wr(p: Path, im) -> None:
    cv2.imencode(".png", im)[1].tofile(str(p))


def overlay(bgr, keep, rej):
    ov = bgr.copy()
    ov[rej] = (0.5 * ov[rej] + 0.5 * np.array([40, 40, 214])).astype(np.uint8)
    ov[keep] = (0, 200, 0)
    return ov


def main() -> int:
    cfg = load_config()
    sel = json.loads((CKPT_DIR / "selection.json").read_text(encoding="utf-8"))
    primary, ranges = sel["primary_model"], sel["dev_ranges"]
    full = {m: load_checkpoint(m) for m in MODELS}
    with open(ROOT / "data" / "splits" / "split_manifest.csv", encoding="utf-8-sig") as f:
        man = {r["image_id"]: r for r in csv.DictReader(f)}
    v8 = man["handwritten/handwritten_013.png"]
    jobs = [(f"field__{p.stem}", p, full) for p in sorted(FIELD.glob("*.png"))]
    jobs.append(("dev__handwritten_013_V8", ROOT / v8["path"], {m: load_checkpoint(m, CKPT_DIR / f"fold{v8['fold']}") for m in MODELS}))
    (RES / "figures").mkdir(parents=True, exist_ok=True)
    (RES / "examples").mkdir(parents=True, exist_ok=True)
    summary = []
    for name, path, cks in jobs:
        t0 = time.perf_counter()
        thr = min(cks[primary]["thresholds"]["recall_priority"], cks[primary]["thresholds"]["precision_priority"])
        res = extract_image(path, cks, cfg, ranges, primary, {"recall_priority": thr, "precision_priority": max(thr, 0.99)})
        bgr = res["bgr"]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        cand = res["candidate_mask"] > 0
        acc = res["masks"]["recall_priority"]                         # accepted groups, BEFORE refinement
        rej = cand & ~acc
        ms = refine_methods(gray, cand, res["candidates"]["_meta"]["polarity"])
        w = ms.pop("_w")
        ms = {k: v & acc for k, v in ms.items()}                       # refinement only inside accepted groups
        prox = {k: proxies(acc, v, w) for k, v in ms.items()}
        rule_choice = select(prox)
        final = ms[FINAL_METHOD]
        assert final.shape == cand.shape and not (final & ~cand).any()  # subset of candidate pixels only
        secs = time.perf_counter() - t0
        d = RAW / name
        (d / "methods").mkdir(parents=True, exist_ok=True)
        u8 = lambda m: (m * 255).astype(np.uint8)  # noqa: E731
        wr(d / "original.png", bgr); wr(d / "candidate_mask.png", u8(cand)); wr(d / "accepted_group_mask.png", u8(acc))
        wr(d / "refined_stroke_mask.png", u8(final)); wr(d / "rejected_mask.png", u8(rej)); wr(d / "final_mask.png", u8(final))
        wr(d / "overlay_before_refinement.png", overlay(bgr, acc, rej)); wr(d / "overlay.png", overlay(bgr, final, cand & ~final))
        wr(d / "handwritten_only.png", np.dstack([bgr, u8(final)]))
        for k, v in ms.items():
            wr(d / "methods" / f"{k}.png", u8(v))
        pd.DataFrame([{"method": k, **p, "adopted": k == FINAL_METHOD, "a_priori_rule_choice": k == rule_choice} for k, p in prox.items()]).to_csv(
            d / "refine_stats.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(res["rows"]).to_csv(d / "candidate_scores.csv", index=False, encoding="utf-8-sig")
        conf = {"refine_cfg": {k: list(v) if isinstance(v, tuple) else v for k, v in REFINE_CFG.items()}, "final_method": FINAL_METHOD,
                "primary_model": primary, "threshold": thr, "stroke_width_px": round(w, 2)}
        (d / "run_info.json").write_text(json.dumps({**conf, "source": str(path), "seconds": round(secs, 3)}, indent=1), encoding="utf-8")
        p = prox[FINAL_METHOD]
        summary.append({"image": name, "groups": len(res["rows"]), "accepted_groups": int(sum(r["keep_recall_priority"] for r in res["rows"])),
                        "candidate_px": int(cand.sum()), "accepted_group_px": int(acc.sum()), "refined_px": int(final.sum()),
                        "removed_by_refinement_px": int(acc.sum() - final.sum()),
                        "broad_fraction_before": round(prox[FINAL_METHOD]["candidate_broad_fraction"], 4), "broad_fraction_after": round(p["broad_fraction"], 4),
                        "thin_retention": round(p["thin_retention"], 3), "a_priori_rule_choice": rule_choice, "seconds": round(secs, 3)})
        # per-image comparison (examples)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
        for ax, (im, t) in zip(axes, ((u8(cand), f"raw candidates {int(cand.sum())} px"), (u8(acc), f"accepted groups (before) {int(acc.sum())} px"),
                                      (u8(final), f"refined stroke-only (after) {int(final.sum())} px"),
                                      (overlay(bgr, final, cand & ~final)[..., ::-1], "final overlay (green kept, red removed)"))):
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None, interpolation="nearest"); ax.set_title(t, fontsize=9); ax.axis("off")
        fig.suptitle(f"{disp(name)} - refinement {FINAL_METHOD}, stroke width ~{w:.1f} px. No ground truth: pixel counts are not accuracy.", fontsize=10)
        fig.savefig(RES / "examples" / f"{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"{name:32s} groups {len(res['rows']):3d} cand {int(cand.sum()):6d} acc {int(acc.sum()):6d} refined {int(final.sum()):6d} "
              f"broad {summary[-1]['broad_fraction_before']:.3f}->{summary[-1]['broad_fraction_after']:.3f} retention {p['thin_retention']:.2f} rule->{rule_choice}")
    s = pd.DataFrame(summary)
    s.to_csv(RAW.parent / "refined_summary.csv", index=False, encoding="utf-8-sig")
    snap = RAW.parent / "snapshots" / hashlib.sha256(json.dumps(conf, sort_keys=True, default=str).encode()).hexdigest()[:12]
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "config.json").write_text(json.dumps(conf, indent=1), encoding="utf-8")
    s.to_csv(snap / "refined_summary.csv", index=False, encoding="utf-8-sig")

    # ================= H5 before vs after (>= 3 field images)
    picks = [n for n in ("field__예시1", "field__weld_marking_W79", "field__예시2", "field__rusted_stencil_GBO") if (RAW / n).exists()]
    fig, axes = plt.subplots(len(picks), 6, figsize=(13.333, 7.5))
    rd = lambda n, f, fl=cv2.IMREAD_UNCHANGED: cv2.imdecode(np.fromfile(str(RAW / n / f), np.uint8), fl)  # noqa: E731
    rows = []
    for row, n in zip(axes, picks):
        o = rd(n, "original.png", cv2.IMREAD_COLOR)
        c, a, r_ = rd(n, "candidate_mask.png", 0), rd(n, "accepted_group_mask.png", 0), rd(n, "refined_stroke_mask.png", 0)
        panels = [(o[..., ::-1], "1 original"), (c, f"2 raw candidates\n{int((c > 0).sum())} px"), (a, f"3 accepted groups\n(before) {int((a > 0).sum())} px"),
                  (r_, f"4 refined stroke-only\n(after) {int((r_ > 0).sum())} px"),
                  (rd(n, "overlay_before_refinement.png", cv2.IMREAD_COLOR)[..., ::-1], "5 overlay BEFORE"),
                  (rd(n, "overlay.png", cv2.IMREAD_COLOR)[..., ::-1], "6 overlay AFTER")]
        for ax, (im, t) in zip(row, panels):
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None, interpolation="nearest"); ax.set_title(t, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        row[0].set_ylabel(disp(n.split("__")[1]), fontsize=9, color=ERR if "stencil" in n else "black")
        rows.append(s[s.image == n].iloc[0].to_dict())
    fig.suptitle(f"H5  Extraction before vs after stroke-only refinement ({FINAL_METHOD}; same RF group decisions). Green = kept, red = removed. "
                 "No ground truth -> no accuracy claim.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RES / "figures" / "H5_extraction_before_after.png", dpi=300, bbox_inches="tight")
    fig.savefig(RES / "figures" / "H5_extraction_before_after.svg", bbox_inches="tight")
    pd.DataFrame(rows).to_csv(RES / "figures" / "H5_extraction_before_after.csv", index=False, encoding="utf-8-sig")
    plt.close(fig)
    print(s.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
