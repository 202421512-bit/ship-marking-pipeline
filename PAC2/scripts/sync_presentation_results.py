"""Copy presentation material from reports/ to results/ (copy only, never move/delete), verify SHA-256, keep a manifest.

* identical file already at the destination -> skipped (no duplicate copies, safe to run repeatedly)
* different file already at the destination  -> NOT overwritten, reported as CONFLICT
* missing source                             -> reported as MISSING (nothing is fabricated)
Run: .venv\\Scripts\\python.exe scripts\\sync_presentation_results.py [--phase 3|4|all]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANIFEST = RESULTS / "manifest.csv"
TREE = ["phase3/figures", "phase3/tables", "phase4/figures", "phase4/extraction_examples", "phase4/tables", "final_presentation"]
G3 = ["G1_loss_convergence", "G2_feature_weight_evolution", "G3_before_after_optimization", "G4_model_comparison",
      "G5_probability_distribution", "G6_domain_bias", "G7_confusion_matrix", "G8_extraction_comparison", "G9_reweighting_process"]
G4 = ["G10_extraction_pipeline", "G11_probability_map", "G12_seven_feature_comparison", "G13_model_threshold_comparison",
      "G14_extraction_statistics", "G15_failure_analysis"]
P3_TABLES = ["model_comparison.csv", "coefficient_stability.csv", "fold_metrics.csv", "oof_summary.csv", "reweighting_history.csv",
             "adaboost_history.csv", "feature_weights.csv", "experiment_meta.json", "PHASE3_REPORT.md", "failures_and_warnings.txt"]
FINAL = {  # final_presentation selection: (source in results, presentation message)
    "G2_feature_weight_evolution.png": ("phase3/figures", "7개 특징 가중치 학습 과정과 최종값(같은 표준화 기준 비교)"),
    "G1_loss_convergence.png": ("phase3/figures", "가중 BCE + L2 목적함수 수렴"),
    "G3_before_after_optimization.png": ("phase3/figures", "S1→S4 보정 전후 성능, 수기 recall 감소 상충관계"),
    "G7_confusion_matrix.png": ("phase3/figures", "기본 vs 최종 모델 클래스별 오류"),
    "G6_domain_bias.png": ("phase3/figures", "촬영 도메인만으로도 완전 분리 → 형상 학습 증거의 한계"),
    "G10_extraction_pipeline.png": ("phase4/figures", "원본 → 후보 → 보존/제거 → 최종 마스크"),
    "G11_probability_map.png": ("phase4/figures", "후보 그룹별 수기 확률(강판 미보정)"),
    "G13_model_threshold_comparison.png": ("phase4/figures", "모델·임계값별 보존 획 차이"),
    "G15_failure_analysis.png": ("phase4/figures", "S6/S16/V8 등 실패 사례와 한계"),
}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    if not MANIFEST.exists():
        return {}
    with open(MANIFEST, encoding="utf-8-sig") as f:
        return {r["output_path"]: r for r in csv.DictReader(f)}


def save_manifest(rows: dict) -> None:
    cols = ["file_name", "source_path", "output_path", "sha256", "bytes", "status", "synced_at"]
    with open(MANIFEST, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for k in sorted(rows):
            w.writerow({c: rows[k].get(c, "") for c in cols})


def copy_one(src: Path, dst: Path, manifest: dict, report: list) -> str:
    rel_dst = str(dst.relative_to(ROOT)).replace("\\", "/")
    rel_src = str(src.relative_to(ROOT)).replace("\\", "/")
    if not src.exists():
        report.append(("MISSING", rel_src, rel_dst))
        return "MISSING"
    s_hash = sha(src)
    if dst.exists():
        d_hash = sha(dst)
        status = "UNCHANGED" if d_hash == s_hash else "CONFLICT"
        if status == "CONFLICT":
            report.append(("CONFLICT (destination differs; not overwritten)", rel_src, rel_dst))
            return status
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if sha(dst) != s_hash:
            report.append(("HASH MISMATCH AFTER COPY", rel_src, rel_dst))
            return "HASH_MISMATCH"
        status = "COPIED"
    old = manifest.get(rel_dst, {})
    manifest[rel_dst] = {"file_name": dst.name, "source_path": rel_src, "output_path": rel_dst, "sha256": s_hash,
                         "bytes": src.stat().st_size, "status": "VERIFIED",
                         "synced_at": old.get("synced_at") if status == "UNCHANGED" and old else dt.datetime.now().isoformat(timespec="seconds")}
    report.append((status, rel_src, rel_dst))
    return status


def ensure_tree() -> None:
    for t in TREE:
        (RESULTS / t).mkdir(parents=True, exist_ok=True)


def sync_phase3(manifest, report):
    src_fig = ROOT / "reports" / "phase3" / "figures"
    for g in G3:
        for ext in ("png", "svg"):
            copy_one(src_fig / f"{g}.{ext}", RESULTS / "phase3" / "figures" / f"{g}.{ext}", manifest, report)
        copy_one(src_fig / f"{g}.csv", RESULTS / "phase3" / "tables" / f"{g}.csv", manifest, report)
    for t in P3_TABLES:
        copy_one(ROOT / "reports" / "phase3" / t, RESULTS / "phase3" / "tables" / t, manifest, report)
    copy_one(ROOT / "reports" / "phase3" / "presentation_notes.md", RESULTS / "phase3" / "presentation_notes.md", manifest, report)


def sync_phase4(manifest, report):
    src_fig = ROOT / "reports" / "phase4" / "figure_data"
    for g in G4:
        copy_one(src_fig / f"{g}.csv", RESULTS / "phase4" / "tables" / f"{g}.csv", manifest, report)


def sync_final(manifest, report):
    for name, (sub, _) in FINAL.items():
        src = RESULTS / sub / name
        if src.exists():
            copy_one(src, RESULTS / "final_presentation" / name, manifest, report)
    lines = ["# Final presentation index", "", "| 파일 | 그림 | 원본 경로 | 발표 설명 |", "|---|---|---|---|"]
    for name, (sub, msg) in FINAL.items():
        ok = (RESULTS / "final_presentation" / name).exists()
        lines.append(f"| {name if ok else name + ' (MISSING)'} | {name.split('_')[0]} | results/{sub}/{name} | {msg} |")
    lines += ["", "Phase 3 수치 = 개발 90장 5-fold OOF. Phase 4 = 개발 이미지 시연 + data/field_test 실제 이미지 추론(정답 마스크 없음 → 정확도 미계산).",
              "Test 30장은 사용하지 않았음."]
    (RESULTS / "final_presentation" / "presentation_index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["3", "4", "all"])
    a = ap.parse_args()
    ensure_tree()
    manifest, report = load_manifest(), []
    if a.phase in ("3", "all"):
        sync_phase3(manifest, report)
    if a.phase in ("4", "all"):
        sync_phase4(manifest, report)
        sync_final(manifest, report)
    save_manifest(manifest)
    counts = {}
    for st, *_ in report:
        counts[st] = counts.get(st, 0) + 1
    print("sync:", counts)
    for st, s, d in report:
        if st not in ("COPIED", "UNCHANGED"):
            print(f"  {st}: {s} -> {d}")
    return 1 if any(st.startswith(("CONFLICT", "HASH")) for st, *_ in report) else 0


if __name__ == "__main__":
    sys.exit(main())
