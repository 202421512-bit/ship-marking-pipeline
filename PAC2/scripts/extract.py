"""Version-selectable extraction + regression comparison.

  .venv\\Scripts\\python.exe scripts\\extract.py --version stable        re-run frozen stable; verifies masks == frozen snapshot
  .venv\\Scripts\\python.exe scripts\\extract.py --version experimental  stable + EXPERIMENTAL changes (separate outputs)
  .venv\\Scripts\\python.exe scripts\\extract.py --compare               both + per-image regression figures and verdicts
Outputs: reports/versions/<version>/<image>/ ; comparison: results/final_delivery/regression_comparison/
No training, no Test 30 access. Without ground truth every verdict is UNDETERMINED unless a protected handwriting area is
damaged (then REGRESSION) - fewer pixels is never counted as an improvement.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pac2 import ROOT, load_config  # noqa: E402
from pac2.candidates import read_bgr  # noqa: E402
from pac2.experimental_recovery import EXP_CFG, experimental  # noqa: E402
from pac2.extraction import CKPT_DIR, MODELS, group_components, guard_not_test, load_checkpoint, score_groups  # noqa: E402
from pac2.recovery import recover  # noqa: E402
from pac2.visualization import plt  # noqa: E402

STABLE = ROOT / "versions" / "stable_baseline"
OUT = ROOT / "reports" / "versions"
CMP = ROOT / "results" / "final_delivery" / "regression_comparison"
PROTECTED = {"field__예시1": "recovered chalk, no inter-character background", "field__2": "faint chalk letters / numbers",
             "field__1": "black marker text", "field__weld_marking_W79": "thin strokes and numbers", "dev__handwritten_013_V8": "thick strokes and junctions"}
u8 = lambda m: (m > 0).astype(np.uint8) * 255  # noqa: E731


def jobs():
    with open(ROOT / "data" / "splits" / "split_manifest.csv", encoding="utf-8-sig") as f:
        v8 = next(r for r in csv.DictReader(f) if r["image_id"] == "handwritten/handwritten_013.png")
    full = {m: load_checkpoint(m) for m in MODELS}
    out = [(f"field__{p.stem}", p, full) for p in sorted((ROOT / "data" / "field_test" / "images").glob("*.png"))]
    out.append(("dev__handwritten_013_V8", ROOT / v8["path"], {m: load_checkpoint(m, CKPT_DIR / f"fold{v8['fold']}") for m in MODELS}))
    return out


def stable_run(path, cks, cfg, sel):
    """Exactly the frozen stable logic (phase4_recovery.py): recover -> groups -> RF auxiliary reject."""
    bgr = read_bgr(path)
    r = recover(bgr)
    p = sel["primary_model"]
    thr = min(cks[p]["thresholds"]["recall_priority"], cks[p]["thresholds"]["precision_priority"])
    groups = group_components(u8(r["combined"]))
    rows = score_groups(groups, cks, cfg, sel["dev_ranges"]) if groups else []
    final = np.zeros_like(r["combined"])
    for g, row in zip(groups, rows):
        if row[f"p_{p}"] >= thr:
            final |= g["mask"]
    return bgr, r, final


def run(version: str):
    cfg = load_config()
    sel = json.loads((CKPT_DIR / "selection.json").read_text(encoding="utf-8"))
    res = {}
    for name, path, cks in jobs():
        guard_not_test(path)
        bgr, r, final = stable_run(path, cks, cfg, sel)
        d = OUT / version / name
        d.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".png", bgr)[1].tofile(str(d / "original.png"))
        if version == "stable":
            frozen = cv2.imdecode(np.fromfile(str(STABLE / "outputs" / name / "final_mask.png"), np.uint8), 0)
            same = bool(np.array_equal(frozen > 0, final))
            print(f"stable {name:28s} reproduces frozen mask: {same}")
            cv2.imencode(".png", u8(final))[1].tofile(str(d / "final_mask.png"))
            res[name] = {"bgr": bgr, "final": final, "reproduced": same}
        else:
            e = experimental(r, final)
            for k, m in e.items():
                cv2.imencode(".png", u8(m))[1].tofile(str(d / f"{k}.png"))
            cv2.imencode(".png", u8(e["final"]))[1].tofile(str(d / "final_mask.png"))
            (d / "run_info.json").write_text(json.dumps({"version": "experimental", "exp_cfg": EXP_CFG, "base": "stable_baseline"}, indent=1), encoding="utf-8")
            res[name] = {"bgr": bgr, "final": e["final"], "stable_final": final, **e}
    return res


def stable_removed_structure(name):
    m = None
    for k in ("structure_segment", "structure_large", "structure_edge"):
        x = cv2.imdecode(np.fromfile(str(STABLE / "outputs" / name / f"removed_{k}.png"), np.uint8), 0) > 0
        m = x if m is None else m | x
    return cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)


def compare():
    s = run("stable")
    e = run("experimental")
    CMP.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in e:
        st, ex, bgr = s[name]["final"], e[name]["final"], e[name]["bgr"]
        added, removed = ex & ~st, st & ~ex
        prot = e[name]["protect"]
        removed_prot = int((removed & prot).sum())
        lost_frac = removed_prot / max(1, int(prot.sum()))
        sr = stable_removed_structure(name)
        readd = int((added & sr).sum())
        readd_frac = readd / max(1, int(added.sum()))
        if name in PROTECTED and lost_frac > 0.02:
            verdict = "REGRESSION (protected handwriting area lost pixels)"
        elif readd_frac > 0.05:
            verdict = "REGRESSION (re-adds structure removed by stable)"
        elif added.sum() == 0 and removed.sum() == 0:
            verdict = "UNCHANGED"
        else:
            verdict = "UNDETERMINED (no ground truth)"
        rows.append({"image": name, "protected_case": PROTECTED.get(name, ""), "stable_px": int(st.sum()), "experimental_px": int(ex.sum()),
                     "added_px": int(added.sum()), "removed_px": int(removed.sum()), "structure_removed_px": int(e[name]["structure_removed"].sum()),
                     "thick_added_px": int(e[name]["thick_added"].sum()), "protected_zone_px": int(prot.sum()),
                     "protected_zone_removed_px": removed_prot, "readded_structure_px": readd, "stable_reproduced": s[name]["reproduced"], "verdict": verdict})
        diff = np.zeros((*st.shape, 3), np.uint8)
        diff[st & ex] = (180, 180, 180); diff[added] = (0, 200, 0); diff[removed] = (220, 30, 30)
        fig, ax = plt.subplots(1, 4, figsize=(13.333, 3.9))
        for a, im, t in zip(ax, (bgr[..., ::-1], u8(st), u8(ex), diff),
                            ("original", f"stable baseline {int(st.sum())} px", f"experimental {int(ex.sum())} px",
                             f"difference: green +{int(added.sum())} added, red -{int(removed.sum())} removed, grey common")):
            a.imshow(im, cmap="gray" if im.ndim == 2 else None, interpolation="nearest"); a.set_title(t, fontsize=8); a.axis("off")
        nm = name.replace("예시", "example")
        fig.suptitle(f"{nm}  |  verdict: {verdict}" + (f"  |  protected: {PROTECTED[name]}" if name in PROTECTED else ""), fontsize=9,
                     color="#d62728" if verdict.startswith("REGRESSION") else "black")
        fig.savefig(CMP / f"{nm}.png", dpi=300, bbox_inches="tight"); fig.savefig(CMP / f"{nm}.svg", bbox_inches="tight")
        plt.close(fig)
    t = pd.DataFrame(rows)
    t.to_csv(CMP / "regression_summary.csv", index=False, encoding="utf-8-sig")
    n_reg = int(t.verdict.str.startswith("REGRESSION").sum())
    decision = ("KEEP STABLE: experimental damages protected handwriting in " + ", ".join(t[t.verdict.str.startswith("REGRESSION")].image)
                if n_reg else "KEEP STABLE as default: experimental not proven better without ground truth (selectable with --version experimental)")
    (CMP / "DECISION.md").write_text(f"# Version decision\n\n{decision}\n\nDefault version: **stable**. Experimental settings were chosen while viewing these "
                                     "images (demo only, not generalization).\n\n" + "\n".join(["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)] + ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)]), encoding="utf-8")
    print(t[["image", "stable_px", "experimental_px", "added_px", "removed_px", "protected_zone_removed_px", "readded_structure_px", "verdict"]].to_string(index=False))
    print(decision)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["stable", "experimental"])
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.compare or not a.version:
        compare()
    else:
        run(a.version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
