"""Recover the 8 user-selected Phase 4 recovery results (copy only, SHA-256 verified, nothing regenerated or overwritten).

Mapping (from scripts/phase4_recovery.py figures(), the run of 2026-10-10 03:46, files unchanged since):
  P01 R1 example1  'new: BRIGHT candidate (chalk)'  = u8(channels['bright'])            -> inference/field__예시1/bright_candidate.png
  P02 R2 field 2   'new final mask'                 = final (figure shows crop 40:240, 90:240) -> inference/field__2/final_mask.png
  P03 R3 field 2   'new kept (green) 9015 px'       = overlay(bgr, final, 0)            -> inference/field__2/final_mask.png (+ original)
  P04 R3 field 1   'new kept (green) 6426 px'       = overlay(bgr, final, 0)            -> inference/field__1/final_mask.png (+ original)
  P05 R4 field 1   'new overlay'                    = overlay(bgr, final, 0)            -> inference/field__1/final_mask.png (+ original)
  P06 R4 example3  'baseline final (one polarity)'  = Phase 4 refined (M5) final        -> reports/phase4_refined/inference/field__예시3/final_mask.png
  P07 R5 V8        'new overlay'                    = overlay(bgr, final, cand & ~final) -> inference/dev__handwritten_013_V8/overlay.png (identical formula)
  P08 R5 W79       'new overlay'                    = overlay(bgr, final, cand & ~final) -> inference/field__weld_marking_W79/overlay.png
P03-P05 overlays (green only, no red) were drawn in memory and never saved as files: they are RE-RENDERED here from the
recovered original + final mask with the same overlay() formula (marked 'rerendered_from_recovered_files', not an original file).
Run: .venv\\Scripts\\python.exe scripts\\recover_favorites.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "RECOVERED_FAVORITES"
REC = ROOT / "reports" / "phase4_recovery" / "inference"
REF = ROOT / "reports" / "phase4_refined" / "inference"
FIG = ROOT / "results" / "phase4_recovery"
STABLE = ROOT / "versions" / "stable_baseline"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
rd = lambda p, f=cv2.IMREAD_UNCHANGED: cv2.imdecode(np.fromfile(str(p), np.uint8), f)  # noqa: E731
V_REC = "phase4_recovery (= stable_baseline, frozen)"
V_REF = "phase4_refined M5 (baseline used in R-figures)"

ITEMS = [
    ("P01", "example1 (field_test/예시1.png)", "R1", "new: BRIGHT candidate (chalk)", "candidate mask", REC / "field__예시1" / "bright_candidate.png", "raw_masks", V_REC, None),
    ("P02", "field 2 (field_test/2.png)", "R2", "new final mask (figure shows crop x40-240, y90-240)", "final mask", REC / "field__2" / "final_mask.png", "raw_masks", V_REC, "R2"),
    ("P03", "field 2 (field_test/2.png)", "R3", "new kept (green) 9015 px", "final mask (overlay)", REC / "field__2" / "final_mask.png", "raw_masks", V_REC, 9015),
    ("P04", "field 1 (field_test/1.png)", "R3", "new kept (green) 6426 px", "final mask (overlay)", REC / "field__1" / "final_mask.png", "raw_masks", V_REC, 6426),
    ("P05", "field 1 (field_test/1.png)", "R4", "new overlay", "final mask (overlay)", REC / "field__1" / "final_mask.png", "raw_masks", V_REC, None),
    ("P06", "example3 (field_test/예시3.png)", "R4", "baseline final (one polarity)", "final mask of BASELINE", REF / "field__예시3" / "final_mask.png", "raw_masks", V_REF, None),
    ("P07", "handwritten_013 V8 (dev image, fold model)", "R5", "new overlay", "final overlay", REC / "dev__handwritten_013_V8" / "overlay.png", "overlays", V_REC, None),
    ("P08", "weld_marking_W79 (field_test)", "R5", "new overlay", "final overlay", REC / "field__weld_marking_W79" / "overlay.png", "overlays", V_REC, None),
]
FIGS = {"R1": "R1_example1_background_leakage", "R2": "R2_faint_chalk_recovery", "R3": "R3_structure_suppression",
        "R4": "R4_dual_polarity", "R5": "R5_stroke_preservation"}


def copy(src: Path, dst: Path) -> tuple:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if sha(dst) != sha(src):
            return "CONFLICT (existing copy differs, not overwritten)", sha(dst)
        return "UNCHANGED", sha(dst)
    shutil.copy2(src, dst)
    return "COPIED", sha(dst)


def overlay_green(bgr, keep):
    ov = bgr.copy()
    ov[keep] = (0, 200, 0)
    return ov


def main() -> int:
    for sub in ("original_figures", "selected_panels", "raw_masks", "overlays", "comparisons"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    for fid, base in FIGS.items():
        for ext in ("png", "svg"):
            src = FIG / f"{base}.{ext}"
            st, h = copy(src, OUT / "original_figures" / src.name)
            rows.append({"preferred_id": "-", "image_name": "-", "figure_id": fid, "selected_stage": "whole figure", "source_path": str(src.relative_to(ROOT)),
                         "recovered_path": str((OUT / "original_figures" / src.name).relative_to(ROOT)), "source_sha256": sha(src), "recovered_sha256": h,
                         "artifact_type": "original figure", "pipeline_version": V_REC, "notes": st})
    frozen = json.loads((STABLE / "VERSION.json").read_text(encoding="utf-8"))["output_masks_sha256"]
    panels = []
    for pid, img, fig, stage, kind, src, sub, ver, extra in ITEMS:
        folder = src.parent
        tag = f"{pid}_{folder.name.replace('예시', 'example')}_{src.stem}"
        dst = OUT / sub / f"{tag}.png"
        notes = []
        if not src.exists():
            rows.append({"preferred_id": pid, "image_name": img, "figure_id": fig, "selected_stage": stage, "source_path": str(src.relative_to(ROOT)),
                         "recovered_path": "", "source_sha256": "", "recovered_sha256": "", "artifact_type": kind, "pipeline_version": ver,
                         "notes": "MISSING - not regenerated"})
            continue
        st, h = copy(src, dst)
        notes.append(st)
        m = rd(src)
        if src.name == "final_mask.png" and ver == V_REC:
            key = f"outputs/{folder.name}/final_mask.png"
            notes.append("matches frozen stable snapshot" if frozen.get(key) == sha(src) else "DIFFERS from frozen stable snapshot")
        if isinstance(extra, int):
            notes.append(f"pixel count {int((m > 0).sum())} (figure title {extra}) {'OK' if int((m > 0).sum()) == extra else 'MISMATCH'}")
        rows.append({"preferred_id": pid, "image_name": img, "figure_id": fig, "selected_stage": stage, "source_path": str(src.relative_to(ROOT)),
                     "recovered_path": str(dst.relative_to(ROOT)), "source_sha256": sha(src), "recovered_sha256": h, "artifact_type": kind,
                     "pipeline_version": ver, "notes": "; ".join(notes)})
        # the original image the stage was computed from (copied as-is)
        osrc = folder / "original.png"
        odst = OUT / "raw_masks" / f"{pid}_{folder.name.replace('예시', 'example')}_original.png"
        ost, oh = copy(osrc, odst)
        rows.append({"preferred_id": pid, "image_name": img, "figure_id": fig, "selected_stage": "input image of the stage", "source_path": str(osrc.relative_to(ROOT)),
                     "recovered_path": str(odst.relative_to(ROOT)), "source_sha256": sha(osrc), "recovered_sha256": oh, "artifact_type": "original image",
                     "pipeline_version": ver, "notes": ost})
        bgr = rd(osrc, cv2.IMREAD_COLOR)
        # presentation panel exactly as drawn in the figure
        if pid in ("P03", "P04", "P05"):
            panel = overlay_green(bgr, m > 0)
            pdst = OUT / "overlays" / f"{pid}_{folder.name.replace('예시', 'example')}_overlay_green_rerendered.png"
            if not pdst.exists():
                cv2.imencode(".png", panel)[1].tofile(str(pdst))
            rows.append({"preferred_id": pid, "image_name": img, "figure_id": fig, "selected_stage": stage + " (panel)", "source_path": "re-rendered: original.png + final_mask.png",
                         "recovered_path": str(pdst.relative_to(ROOT)), "source_sha256": "", "recovered_sha256": sha(pdst), "artifact_type": "overlay panel",
                         "pipeline_version": ver, "notes": "rerendered_from_recovered_files with the figure's overlay(bgr, final, 0) formula; overlay never saved as a file originally"})
        elif pid == "P02":
            panel = m[90:240, 40:240]
        else:
            panel = m
        cv2.imencode(".png", panel)[1].tofile(str(OUT / "selected_panels" / f"{tag}_panel.png")) if not (OUT / "selected_panels" / f"{tag}_panel.png").exists() else None
        panels.append((pid, img, fig, stage, kind, ver, bgr if pid != "P02" else bgr[90:240, 40:240], panel, "OK" if st != "CONFLICT" else st))
    with open(OUT / "recovery_manifest.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    contact(panels)
    ok = [r for r in rows if r["preferred_id"].startswith("P") and r["selected_stage"] not in ("input image of the stage",) and "(panel)" not in r["selected_stage"]]
    for r in ok:
        print(r["preferred_id"], r["source_path"], "->", r["recovered_path"], "| hash match:", r["source_sha256"] == r["recovered_sha256"], "|", r["notes"])
    return 0


def contact(panels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 4, figsize=(16, 18), gridspec_kw={"hspace": 0.45})
    for k, (pid, img, fig_id, stage, kind, ver, bgr, panel, status) in enumerate(panels):
        a1, a2 = axes[k // 2, (k % 2) * 2], axes[k // 2, (k % 2) * 2 + 1]
        a1.imshow(bgr[..., ::-1]); a1.set_title("original", fontsize=8)
        a2.imshow(panel[..., ::-1] if panel.ndim == 3 else panel, cmap=None if panel.ndim == 3 else "gray", interpolation="nearest")
        a2.set_title(f"{pid} | {fig_id} | {stage}\n{img.replace(chr(50696) + chr(49884), 'example')}\n{kind} | {ver}\nrecovered: {status}", fontsize=7)
        for a in (a1, a2):
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("PAC2 — User Selected Recovery Results (Curated Demo: stages and versions differ per image; not one automated model)", fontsize=13)
    fig.savefig(OUT / "comparisons" / "selected_8_results.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
