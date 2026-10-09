"""Phase 4 recovery run: frozen baseline (Phase 4 refined, M5) vs recovery pipeline, R1-R5 figures, GT metrics if masks exist.

Per image (reports/phase4_recovery/inference/<name>/): bright / dark / low-contrast / combined candidates, structure and
other removed masks, accepted groups, final mask, overlay, handwritten-only, components.csv, groups.csv.
The 7-feature RF is an AUXILIARY check only: it can reject a group (p < recall-priority threshold) but never accepts
anything that the stroke / structure tests removed. OOD groups are recorded.
No training, Test 30 never read. Run: .venv\\Scripts\\python.exe scripts\\phase4_recovery.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pac2 import ROOT, load_config  # noqa: E402
from pac2.candidates import read_bgr  # noqa: E402
from pac2.extraction import CKPT_DIR, MODELS, group_components, guard_not_test, load_checkpoint, score_groups  # noqa: E402
from pac2.recovery import RECOVERY_CFG, recover  # noqa: E402
from pac2.visualization import ERR, plt  # noqa: E402

RAW = ROOT / "reports" / "phase4_recovery" / "inference"
BASE = ROOT / "reports" / "phase4_refined" / "inference"
OUT = ROOT / "results" / "phase4_recovery"
FIELD = ROOT / "data" / "field_test" / "images"
MASKS = ROOT / "data" / "field_test" / "masks"
u8 = lambda m: (m > 0).astype(np.uint8) * 255  # noqa: E731


def wr(p, im):
    cv2.imencode(".png", im)[1].tofile(str(p))


def rd(p, fl=cv2.IMREAD_UNCHANGED):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), fl) if Path(p).exists() else None


def overlay(bgr, keep, rem):
    ov = bgr.copy()
    ov[rem] = (0.5 * ov[rem] + 0.5 * np.array([40, 40, 214])).astype(np.uint8)
    ov[keep] = (0, 200, 0)
    return ov


def gt_metrics(pred, cand, gt, ign):
    v = ~ign
    g = gt & v
    tp = int((pred & g).sum()); fp = int((pred & ~gt & v).sum()); fn = int((~pred & g).sum())
    return {"gt_px": int(g.sum()), "candidate_recall": float((cand & g).sum() / max(1, g.sum())), "final_recall": tp / max(1, tp + fn),
            "precision": tp / max(1, tp + fp), "dice": 2 * tp / max(1, 2 * tp + fp + fn), "iou": tp / max(1, tp + fp + fn), "fp_px": fp}


def main() -> int:
    cfg = load_config()
    sel = json.loads((CKPT_DIR / "selection.json").read_text(encoding="utf-8"))
    primary = sel["primary_model"]
    with open(ROOT / "data" / "splits" / "split_manifest.csv", encoding="utf-8-sig") as f:
        man = {r["image_id"]: r for r in csv.DictReader(f)}
    v8 = man["handwritten/handwritten_013.png"]
    jobs = [(f"field__{p.stem}", p, {m: load_checkpoint(m) for m in MODELS}) for p in sorted(FIELD.glob("*.png"))]
    jobs.append(("dev__handwritten_013_V8", ROOT / v8["path"], {m: load_checkpoint(m, CKPT_DIR / f"fold{v8['fold']}") for m in MODELS}))
    OUT.mkdir(parents=True, exist_ok=True)
    summary, store = [], {}
    for name, path, cks in jobs:
        guard_not_test(path)
        bgr = read_bgr(path)
        r = recover(bgr)
        ch = r["channels"]
        thr = min(cks[primary]["thresholds"]["recall_priority"], cks[primary]["thresholds"]["precision_priority"])
        groups = group_components(u8(r["combined"]))
        rows = score_groups(groups, cks, cfg, sel["dev_ranges"]) if groups else []
        final = np.zeros_like(r["combined"])
        rf_rej = np.zeros_like(final)
        for g, row in zip(groups, rows):
            row["rf_accept"] = int(row[f"p_{primary}"] >= thr)
            (final if row["rf_accept"] else rf_rej)[g["mask"]] = True
        cand_all = r["bright_candidate"] | r["dark_candidate"]
        removed_all = cand_all & ~final
        d = RAW / name
        d.mkdir(parents=True, exist_ok=True)
        for k, m in (("bright_candidate", ch["bright"]), ("dark_candidate", ch["dark"]),
                     ("lowcontrast_candidate", ch["bright_lowcontrast"] | ch["dark_lowcontrast"]), ("combined_candidate", cand_all),
                     ("accepted_groups", r["combined"]), ("final_mask", final), ("rf_rejected", rf_rej),
                     ("structure_edges", r["structure_edges"]), *[(f"removed_{k2}", v) for k2, v in r["removed"].items()]):
            wr(d / f"{k}.png", u8(m))
        wr(d / "original.png", bgr)
        wr(d / "overlay.png", overlay(bgr, final, removed_all))
        wr(d / "handwritten_only.png", np.dstack([bgr, u8(final)]))
        pd.DataFrame(r["components"]).to_csv(d / "components.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(r["segments"]).to_csv(d / "structural_segments.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(rows).to_csv(d / "groups.csv", index=False, encoding="utf-8-sig")
        (d / "run_info.json").write_text(json.dumps({"source": str(path), "recovery_cfg": {k: list(v) if isinstance(v, tuple) else v for k, v in RECOVERY_CFG.items()},
                                                    "primary_model": primary, "rf_threshold": thr}, indent=1), encoding="utf-8")
        b = BASE / name
        base_final = rd(b / "final_mask.png", 0)
        base_cand = rd(b / "candidate_mask.png", 0)
        rec = {"image": name, "baseline_candidate_px": int((base_cand > 0).sum()) if base_cand is not None else "",
               "baseline_final_px": int((base_final > 0).sum()) if base_final is not None else "",
               "bright_px": int(ch["bright"].sum()), "dark_px": int(ch["dark"].sum()),
               "lowcontrast_px": int((ch["bright_lowcontrast"] | ch["dark_lowcontrast"]).sum()), "combined_candidate_px": int(cand_all.sum()),
               **{f"removed_{k}_px": int(v.sum()) for k, v in r["removed"].items()},
               "groups": len(groups), "rf_rejected_groups": int(sum(1 - x["rf_accept"] for x in rows)), "ood_groups": int(sum(bool(x["ood_flags"]) for x in rows)),
               "final_px": int(final.sum())}
        stem = name.split("__")[1]
        gtp = MASKS / f"{stem}_gt.png"
        if gtp.exists():
            gt = rd(gtp, 0) > 0
            ign = (rd(MASKS / f"{stem}_ignore.png", 0) > 0) if (MASKS / f"{stem}_ignore.png").exists() else np.zeros_like(gt)
            for tag, pred, cnd in (("new", final, cand_all), ("baseline", base_final > 0, base_cand > 0)):
                rec.update({f"{tag}_{k}": round(v, 4) if isinstance(v, float) else v for k, v in gt_metrics(pred, cnd, gt, ign).items()})
        summary.append(rec)
        store[name] = {"bgr": bgr, "r": r, "final": final, "cand": cand_all, "base_final": base_final, "base_cand": base_cand}
        print(f"{name:30s} cand {int(cand_all.sum()):6d} final {int(final.sum()):6d} (baseline final {rec['baseline_final_px']}) groups {len(groups)} rf-rejected {rec['rf_rejected_groups']}")
    s = pd.DataFrame(summary)
    s.to_csv(RAW.parent / "recovery_summary.csv", index=False, encoding="utf-8-sig")
    figures(store, s)
    return 0


def _show(ax, im, t, red=False):
    ax.imshow(im if im.ndim == 3 else im, cmap=None if im.ndim == 3 else "gray", interpolation="nearest")
    ax.set_title(t, fontsize=8, color=ERR if red else "black")
    ax.set_xticks([]); ax.set_yticks([])


def _save(fig, name, data):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    data.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    plt.close(fig)


def figures(st, s):
    rgb = lambda x: x[..., ::-1]  # noqa: E731
    disp = lambda n: n.split("__")[1].replace("예시", "example")  # noqa: E731
    row = lambda n: s[s.image == n].iloc[0]  # noqa: E731

    # R1 example1 background leakage
    n = "field__예시1"; d = st[n]; r = d["r"]
    fig, ax = plt.subplots(1, 6, figsize=(13.333, 3.4))
    for a, im, t in zip(ax, (rgb(d["bgr"]), d["base_final"], u8(r["channels"]["dark"]), u8(r["channels"]["bright"]), u8(d["final"]),
                             rgb(overlay(d["bgr"], d["final"], d["cand"] & ~d["final"]))),
                        ("original", f"baseline final (M5)\n{row(n).baseline_final_px} px", "new: DARK candidate\n(old pipeline used only this)",
                         "new: BRIGHT candidate\n(chalk)", f"new final\n{row(n).final_px} px", "new overlay (green kept, red removed)")):
        _show(a, im, t)
    fig.suptitle("R1  example1 background leakage. Cause: the old generator chose ONE polarity per image ('dark' here), so it extracted the darker plate "
                 "between the bright chalk strokes. New: bright and dark channels separately; the seam is removed as a structural segment.", fontsize=8.5)
    _save(fig, "R1_example1_background_leakage", s[s.image == n])

    # R2 faint chalk recovery (field 2, crop of the chalk area)
    n = "field__2"; d = st[n]; r = d["r"]; y0, y1, x0, x1 = 90, 240, 40, 240
    cr = lambda m: m[y0:y1, x0:x1]  # noqa: E731
    fig, ax = plt.subplots(1, 5, figsize=(13.333, 3.4))
    for a, im, t in zip(ax, (rgb(cr(d["bgr"])), cr(d["base_cand"]), cr(u8(r["channels"]["bright_lowcontrast"])),
                             cr(u8(r["channels"]["bright"] | r["channels"]["bright_lowcontrast"])), cr(u8(d["final"]))),
                        ("original (crop)", "baseline candidates", "new low-contrast channel", "new bright candidates", "new final mask")):
        _show(a, im, t)
    fig.suptitle("R2  Faint chalk recovery (field 2, crop x 40-240, y 90-240). Cause: 41 px top-hat kernels responded to large bright structures, "
                 "raising the robust threshold to 120.8 so chalk was cut at the candidate stage. Some strokes ('Angle' letters) remain incomplete.", fontsize=8.5)
    _save(fig, "R2_faint_chalk_recovery", s[s.image == n])

    # R3 structural false positives (field 2 full + field 1)
    fig, axes = plt.subplots(2, 4, figsize=(13.333, 6.2))
    for axr, n in zip(axes, ("field__2", "field__1")):
        d = st[n]; r = d["r"]
        struct = r["removed"]["structure_segment"] | r["removed"]["structure_edge"] | r["removed"]["structure_large"]
        sv = np.zeros((*struct.shape, 3), np.uint8); sv[struct] = (230, 30, 30); sv[r["removed"]["speck"] | r["removed"]["no_ridge"]] = (120, 120, 120)
        for a, im, t in zip(axr, (rgb(d["bgr"]), rgb(overlay(d["bgr"], d["base_final"] > 0, np.zeros_like(d["final"]))), sv,
                                  rgb(overlay(d["bgr"], d["final"], np.zeros_like(d["final"])))),
                            ("original", f"baseline kept (green) {row(n).baseline_final_px} px", "removed: structure (red), specks / no ridge (grey)",
                             f"new kept (green) {row(n).final_px} px")):
            _show(a, im, t)
    fig.suptitle("R3  Structural false-positive suppression. Long straight border/seam segments and components on edges of large objects (found after "
                 "inpainting all strokes) are removed. FAILURE kept: machinery interior lines at the top of field 2 and plate corners in field 1 survive.", fontsize=8.5)
    _save(fig, "R3_structure_suppression", s[s.image.isin(["field__2", "field__1"])])

    # R4 dual polarity: field 1 (dark marker + bright chalk) and example3
    fig, axes = plt.subplots(2, 4, figsize=(13.333, 6.2))
    for axr, n in zip(axes, ("field__1", "field__예시3")):
        d = st[n]; r = d["r"]
        both = np.zeros((*d["final"].shape, 3), np.uint8)
        both[d["final"] & r["dark_candidate"]] = (255, 140, 0); both[d["final"] & r["bright_candidate"]] = (255, 255, 255)
        for a, im, t in zip(axr, (rgb(d["bgr"]), d["base_final"], both, rgb(overlay(d["bgr"], d["final"], np.zeros_like(d["final"])))),
                            ("original", "baseline final (one polarity)", "new final: white = bright ink, orange = dark ink", "new overlay")):
            _show(a, im, t)
        axr[0].set_ylabel(disp(n), fontsize=9)
    fig.suptitle("R4  Dual-polarity detection: black marker ('do5', '835', cross) and bright chalk in the same image are both extracted.", fontsize=9)
    _save(fig, "R4_dual_polarity", s[s.image.isin(["field__1", "field__예시3"])])

    # R5 stroke preservation V8 + W79
    fig, axes = plt.subplots(2, 4, figsize=(13.333, 6.2))
    for axr, n in zip(axes, ("dev__handwritten_013_V8", "field__weld_marking_W79")):
        d = st[n]
        for a, im, t in zip(axr, (rgb(d["bgr"]), d["base_final"], u8(d["final"]), rgb(overlay(d["bgr"], d["final"], d["cand"] & ~d["final"]))),
                            ("original", f"baseline final (M5) {row(n).baseline_final_px} px", f"new final {row(n).final_px} px", "new overlay")):
            _show(a, im, t)
        axr[0].set_ylabel(disp(n), fontsize=9)
    fig.suptitle("R5  Stroke preservation. Thin chalk (W79) is kept; the thick marker V8 (stroke wider than the 15 px stroke-scale kernels) is only "
                 "partly captured by the new generator = FAILURE for thick strokes.", fontsize=9)
    _save(fig, "R5_stroke_preservation", s[s.image.isin(["dev__handwritten_013_V8", "field__weld_marking_W79"])])


if __name__ == "__main__":
    sys.exit(main())
