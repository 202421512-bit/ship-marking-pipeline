"""G10-G15 + per-image extraction examples + Phase 4 notes, regenerated from saved inference outputs only (no training).

Reads reports/phase4/inference/*/ and reports/phase4/extraction_summary.csv, models/phase4/selection.json + checkpoints
(thresholds only). Writes PNG (300 dpi) + SVG to results/phase4/figures, plot data CSV to reports/phase4/figure_data
(synced to results/phase4/tables), comparison images to results/phase4/extraction_examples, notes to results/phase4.
Run: .venv\\Scripts\\python.exe scripts\\phase4_visualize.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pac2 import ROOT  # noqa: E402
from pac2.features import FEATURES  # noqa: E402
from pac2.visualization import BASE, ERR, FIGSIZE, FINAL, MID, plt, save  # noqa: E402

INF = ROOT / "reports" / "phase4" / "inference"
DATA = ROOT / "reports" / "phase4" / "figure_data"
FIG = ROOT / "results" / "phase4" / "figures"
EXA = ROOT / "results" / "phase4" / "extraction_examples"
MODELS = ["S1_basic_unweighted", "S3_inner_cv_logistic", "S4_hard_example_reweighted", "RF_random_forest"]
MCOL = {"S1_basic_unweighted": BASE, "S3_inner_cv_logistic": MID, "S4_hard_example_reweighted": "#17becf", "RF_random_forest": FINAL}
FLAB = ["1 CV_w", "2 CV_h", "4 CV_g", "5 S_theta", "6 B", "10 CV_A", "11 R"]


def disp(name: str) -> str:
    """ASCII label for figures (the plotting font has no Hangul glyphs); file names are unchanged."""
    return name.replace("예시", "example")


def rd(p: Path, flag=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), flag)


def load_run(d: Path) -> dict:
    sc = d / "candidate_scores.csv"
    rows = pd.read_csv(sc) if sc.exists() and sc.stat().st_size > 5 else pd.DataFrame()
    return {"dir": d, "info": json.loads((d / "run_info.json").read_text(encoding="utf-8")), "rows": rows,
            **{k: rd(d / f"{k}.png", cv2.IMREAD_UNCHANGED) for k in ("original", "preprocessed", "candidate_mask", "accepted_mask",
                                                                       "rejected_mask", "final_mask", "overlay")}}


def stage_panels(r: dict):
    return [(r["original"][..., ::-1], "1 original"), (r["preprocessed"], "2 preprocessed (CLAHE, display)"),
            (r["candidate_mask"], "3 candidate strokes"), (r["accepted_mask"], "4 accepted (handwriting)"),
            (r["rejected_mask"], "5 rejected candidates"), (r["overlay"][..., ::-1], "6 final overlay (green kept, red removed)")]


def main() -> int:
    sel = json.loads((ROOT / "models" / "phase4" / "selection.json").read_text(encoding="utf-8"))
    primary = sel["primary_model"]
    ck = {m: json.loads((ROOT / "models" / "phase4" / f"{m}.json").read_text(encoding="utf-8")) for m in MODELS}
    thr_rec = {m: min(ck[m]["thresholds"]["recall_priority"], ck[m]["thresholds"]["precision_priority"]) for m in MODELS}
    summ = pd.read_csv(ROOT / "reports" / "phase4" / "extraction_summary.csv")
    runs = {d.name: load_run(d) for d in sorted(INF.iterdir()) if d.is_dir()}
    hw = json.loads(summ.hardware.iloc[0])
    notes = {}
    EXA.mkdir(parents=True, exist_ok=True)

    # per-image extraction examples (same coordinates, original aspect)
    for name, r in runs.items():
        fig, axes = plt.subplots(1, 6, figsize=(20, 3.8))
        for ax, (im, t) in zip(axes, stage_panels(r)):
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{disp(name)}  ({r['original'].shape[1]}x{r['original'].shape[0]} px, {r['info']['n_groups']} groups, primary {primary}, "
                     f"recall-priority threshold {r['info']['modes']['recall_priority']:.2f}). No ground truth -> no accuracy.", fontsize=10)
        fig.savefig(EXA / f"{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ---------------- G10 pipeline (two field images)
    picks = [n for n in ("field__예시1", "field__rusted_stencil_GBO") if n in runs]
    fig, axes = plt.subplots(len(picks), 6, figsize=FIGSIZE)
    rows = []
    for row, n in zip(np.atleast_2d(axes), picks):
        r = runs[n]
        for ax, (im, t) in zip(row, stage_panels(r)):
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(t, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        row[0].set_ylabel(disp(n.replace("field__", "")), fontsize=10)
        rows.append({"image": n, "candidate_px": int((r["candidate_mask"] > 0).sum()), "accepted_px": int((r["accepted_mask"] > 0).sum()),
                     "rejected_px": int((r["rejected_mask"] > 0).sum()), "groups": r["info"]["n_groups"]})
    fig.suptitle(f"G10  Extraction pipeline on real field images (same coordinates per row). Primary {primary}, recall-priority mode. "
                 "No ground-truth masks.", fontsize=11)
    save_both(fig, "G10_extraction_pipeline", pd.DataFrame(rows))
    notes["G10"] = dict(t="G10 Extraction Pipeline", p="한 이미지 안에서 후보 획 → 보존/제거 → 최종 마스크 과정을 같은 좌표로 보여줌",
                        m="멀티스케일 top/black-hat + 얇음 필터로 후보 획, 근접 그룹화, 그룹별 7특징(도메인 정합 H_t), Phase 3 모델 확률, OOF 기반 recall-priority 임계값",
                        d="data/field_test/images (실제 사진, 정답 마스크 없음)",
                        q="; ".join(f"{x['image']}: 후보 {x['candidate_px']} px → 보존 {x['accepted_px']} px, 제거 {x['rejected_px']} px, 그룹 {x['groups']}" for x in rows),
                        i="예시1은 4개 그룹 중 1개(작은 조각)만 제거, 녹슨 스텐실은 RF가 5개 그룹 모두 보존(스텐실은 비수기이므로 오보존).",
                        l="후보 생성기가 녹 경계·조명 가장자리를 포함하고 희미한 분필 획 일부를 놓침. 정답이 없어 픽셀 정확도 미계산.")

    # ---------------- G11 probability map
    picks = [n for n in ("field__예시1", "field__예시2", "field__rusted_stencil_GBO") if n in runs]
    fig, axes = plt.subplots(1, len(picks), figsize=FIGSIZE)
    cmap = plt.get_cmap("RdYlGn")
    rows = []
    for ax, n in zip(np.atleast_1d(axes), picks):
        r = runs[n]
        base = cv2.cvtColor(cv2.cvtColor(r["original"], cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB).astype(float) / 255 * 0.6
        lab_mask = r["candidate_mask"] > 0
        for _, g in r["rows"].iterrows():
            x0, y0, x1, y1 = json.loads(g.bbox_x0y0x1y1)
            p = g[f"p_{primary}"]
            sub = lab_mask[y0:y1, x0:x1]
            base[y0:y1, x0:x1][sub] = cmap(p)[:3]
            ax.text(x0, max(0, y0 - 3), f"{p:.2f}", color="white", fontsize=8, bbox=dict(fc="black", alpha=0.6, pad=1))
            rows.append({"image": n, "group_id": g.group_id, "p": p, "threshold": thr_rec[primary], "kept": int(p >= thr_rec[primary]),
                         "ood": g.ood_flags if isinstance(g.ood_flags, str) else ""})
        ax.imshow(base, interpolation="nearest")
        ax.set_title(disp(n.replace("field__", "")), fontsize=11); ax.set_xticks([]); ax.set_yticks([])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    fig.colorbar(sm, ax=list(np.atleast_1d(axes)), fraction=0.02, label=f"P(handwritten), {primary} (not calibrated on steel)")
    fig.suptitle(f"G11  Candidate-group handwriting probability (stroke pixels coloured by group score); threshold {thr_rec[primary]:.2f} "
                 "(recall priority, from development OOF)", fontsize=11)
    save_both(fig, "G11_probability_map", pd.DataFrame(rows))
    pr = pd.DataFrame(rows)
    notes["G11"] = dict(t="G11 Handwriting Probability Map", p="후보 그룹별 수기 적합도를 색으로 표시",
                        m=f"P = 모델 출력({primary}), 임계값 {thr_rec[primary]:.2f}는 개발 데이터 OOF에서 수기 recall ≥ 0.97을 만족하는 최대값",
                        d="field_test 3장", q=f"그룹 {len(pr)}개, 보존 {int(pr.kept.sum())}, 제거 {int((1 - pr.kept).sum())}, 점수 범위 {pr.p.min():.2f}–{pr.p.max():.2f}, "
                                              f"OOD 표시 그룹 {int((pr.ood != '').sum())}",
                        i="대부분 그룹이 높은 점수 → 7특징 모델이 강판의 불규칙한 비수기 구조(녹 조각, 스텐실)도 '불규칙 = 수기'로 판단하는 경향.",
                        l="확률은 크롭 학습 분포 기준이며 강판에서 보정되지 않았음. OOD 그룹의 점수는 신뢰도 낮음.")

    # ---------------- G12 seven feature comparison (accepted vs rejected groups, all runs, primary model)
    allg = pd.concat([r["rows"].assign(run=n) for n, r in runs.items() if len(r["rows"])], ignore_index=True)
    allg["kept"] = (allg[f"p_{primary}"] >= allg.run.map(lambda n: runs[n]["info"]["modes"]["recall_priority"])).astype(int)
    fig, axes = plt.subplots(2, 7, figsize=FIGSIZE)
    rows = []
    for j, (f, lab) in enumerate(zip(FEATURES, FLAB)):
        for row_i, (col, kind) in enumerate(((f, "raw value"), (f"z_{f}", "z (S3 standardization)"))):
            ax = axes[row_i, j]
            for k, (kv, color, name) in enumerate(((1, FINAL, "accepted"), (0, ERR, "rejected"))):
                sub = allg[allg.kept == kv]
                miss = sub[f"{f}_missing"] == 1
                v = sub.loc[~miss, col].astype(float)
                ax.scatter(np.full(len(v), k) + np.random.default_rng(j).uniform(-0.15, 0.15, len(v)), v, s=14, color=color, alpha=0.7,
                           marker="o" if kv else "x", label=name if j == 0 and row_i == 0 else None)
                ax.text(k, ax.get_ylim()[1] if len(v) else 0, f"miss {int(miss.sum())}", ha="center", fontsize=7, color="black")
                rows.append({"feature": f, "scale": kind, "group": name, "n": int(len(sub)), "n_missing": int(miss.sum()),
                             "median": float(v.median()) if len(v) else np.nan})
            ax.set_xticks([0, 1], ["acc.", "rej."], fontsize=8)
            if row_i == 0:
                ax.set_title(lab, fontsize=10)
            ax.set_ylabel(kind if j == 0 else "", fontsize=9)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"G12  Seven shape features of accepted vs rejected candidate groups ({len(allg)} groups, {primary}, recall priority). "
                 "Top: raw (different units); bottom: z-scores. Missing values counted, not plotted as 0.", fontsize=10)
    save_both(fig, "G12_seven_feature_comparison", pd.DataFrame(rows))
    notes["G12"] = dict(t="G12 Seven Feature Comparison", p="보존/제외 후보의 7개 형상 특징 차이",
                        m="원시값(특징마다 단위·척도 다름)과 S3 학습 기준 z-score를 분리 표시, 결측은 개수로만 표시",
                        d=f"field + dev 시연 이미지의 후보 그룹 {len(allg)}개",
                        q=f"보존 {int(allg.kept.sum())}개 / 제외 {int((1 - allg.kept).sum())}개, 결측 특징이 하나 이상인 그룹 {int((allg.n_missing > 0).sum())}개",
                        i="제외된 그룹 수가 매우 적어 특징 차이를 통계적으로 비교할 수 없음. 단일 연결요소 그룹은 CV_g·B 등이 결측되어 학습 중앙값으로 대체됨.",
                        l="그룹 단위 특징은 학습(이미지 단위)과 분포가 다름. 결측 대체값이 점수에 영향.")

    # ---------------- G13 model and threshold comparison on the same images
    picks = [n for n in ("field__rusted_stencil_GBO", "field__예시1", "dev__formal_002_F8") if n in runs]
    fig, axes = plt.subplots(len(picks), 5, figsize=FIGSIZE)
    rows = []
    for row, n in zip(np.atleast_2d(axes), picks):
        r = runs[n]
        row[0].imshow(r["original"][..., ::-1]); row[0].set_title("original", fontsize=9)
        cm = r["candidate_mask"] > 0
        n_lab, lab = cv2.connectedComponents(cm.astype(np.uint8), connectivity=8)
        for ax, m in zip(row[1:], MODELS):
            keep = np.zeros(cm.shape, bool)
            kept_n = 0
            for _, g in r["rows"].iterrows():
                x0, y0, x1, y1 = json.loads(g.bbox_x0y0x1y1)
                if g[f"p_{m}"] >= thr_rec[m]:
                    keep[y0:y1, x0:x1] |= cm[y0:y1, x0:x1]
                    kept_n += 1
            vis = np.zeros((*cm.shape, 3), np.uint8)
            vis[cm] = (214, 40, 40)
            vis[keep] = (44, 160, 44)
            ax.imshow(vis, interpolation="nearest")
            ax.set_title(f"{m.split('_')[0]} thr {thr_rec[m]:.2f}: keep {kept_n}/{len(r['rows'])}\n{int(keep.sum())} px kept", fontsize=8,
                         color=MCOL[m] if m != "S1_basic_unweighted" else "black")
            rows.append({"image": n, "model": m, "threshold": thr_rec[m], "groups": len(r["rows"]), "kept_groups": kept_n,
                         "removed_groups": len(r["rows"]) - kept_n, "kept_px": int(keep.sum()), "candidate_px": int(cm.sum()),
                         "mean_p": float(r["rows"][f"p_{m}"].mean()) if len(r["rows"]) else np.nan})
        for ax in row:
            ax.set_xticks([]); ax.set_yticks([])
        row[0].set_ylabel(disp(n.split("__")[1]), fontsize=9)
    fig.suptitle("G13  Same images, four models at their own recall-priority thresholds (green kept, red removed). "
                 "Bounding boxes of groups only select candidate pixels; statistics, not accuracy.", fontsize=10)
    save_both(fig, "G13_model_threshold_comparison", pd.DataFrame(rows))
    t13 = pd.DataFrame(rows)
    notes["G13"] = dict(t="G13 Model and Threshold Comparison", p="같은 이미지에서 S1/S3/S4/RF가 보존·제거한 획의 차이",
                        m="각 모델의 OOF 기반 recall-priority 임계값 적용",
                        d="녹슨 스텐실, 예시1(field), formal F8(dev, 해당 fold 제외 모델)",
                        q="; ".join(f"{x.image.split('__')[1]} {x.model.split('_')[0]}: {x.kept_groups}/{x.groups} 보존, {x.kept_px}/{x.candidate_px} px, 평균 p {x.mean_p:.2f}"
                                    for x in t13.itertuples()),
                        i="스텐실(비수기)은 S3·S4가 대부분 제거, RF는 전부 보존. formal F8은 이미지 단위(Phase 3)에서는 정형으로 맞췄으나 Phase 4 후보 생성 방식에서는 RF가 보존(오보존).",
                        l="정답이 없어 어느 모델이 옳은지 픽셀 단위로 판정 불가. 스텐실은 이 프로젝트 정의상 비수기이므로 정성적으로만 해석.")

    # ---------------- G14 extraction statistics
    s = summ.copy()
    s["label"] = s.kind + ":" + s.image.map(disp)
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
    x = np.arange(len(s))
    axes[0, 0].bar(x - 0.2, s.candidate_groups, 0.4, color=BASE, label="candidate groups")
    axes[0, 0].bar(x + 0.2, s.accepted_recall, 0.4, color=FINAL, hatch="//", label="accepted (recall mode)")
    axes[0, 0].set(ylabel="groups (count)", title="Candidate vs accepted groups")
    axes[0, 1].bar(x - 0.2, s.candidate_pixels, 0.4, color=BASE, label="candidate px")
    axes[0, 1].bar(x + 0.2, s.retained_pixels_recall, 0.4, color=FINAL, hatch="//", label="retained px")
    axes[0, 1].set(ylabel="pixels (count)", title="Candidate vs retained stroke pixels")
    axes[1, 0].bar(x, s.removed_candidate_pixel_ratio_recall, color=ERR, alpha=0.8, label="removed candidate pixel ratio")
    axes[1, 0].plot(x, s.mean_score_primary.astype(float), color=MID, marker="o", label=f"mean P(handwritten) {primary}")
    axes[1, 0].set(ylabel="ratio / probability (0-1)", ylim=(0, 1.05), title="Removed ratio and mean score")
    axes[1, 1].bar(x, s.seconds, color=MID)
    axes[1, 1].set(ylabel="seconds per image", title="Processing time")
    for ax in axes.ravel():
        ax.set_xticks(x, s.label, rotation=60, ha="right", fontsize=7)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.suptitle(f"G14  Extraction statistics per image (n={len(s)}; {hw['cpu']}, {hw['logical_cpus']} logical CPUs, CPU only, Python {hw['python']})", fontsize=10)
    save_both(fig, "G14_extraction_statistics", s.drop(columns=["hardware"]))
    notes["G14"] = dict(t="G14 Extraction Statistics", p="이미지별 후보·보존·제거 규모와 처리 시간",
                        m="후보 그룹 수, 보존 그룹 수, 후보/보존 픽셀, 제거 비율 = 1 − 보존/후보, 평균 점수, 처리 시간(후보 생성+특징+4개 모델)",
                        d=f"field {int((s.kind == 'field').sum())}장 + dev 시연 {int((s.kind == 'dev').sum())}장",
                        q="; ".join(f"{x.label}: 그룹 {x.candidate_groups}→{x.accepted_recall}, px {x.candidate_pixels}→{x.retained_pixels_recall}, {x.seconds:.2f}s" for x in s.itertuples()),
                        i="대부분 이미지에서 제거 비율이 0에 가까움 → 현재 모델은 후보를 거의 걸러내지 못함(후보 생성 단계가 사실상 결과를 결정).",
                        l=f"처리 시간은 {hw['cpu']} CPU 단일 프로세스 기준. 픽셀 정확도는 정답 없음으로 미계산.")

    # ---------------- G15 failure analysis (S6 / S16 / V8 + F8)
    cases = [("dev__handwritten_002", "S6"), ("dev__handwritten_009", "S16"), ("dev__handwritten_013", "V8"), ("dev__formal_002_F8", "F8 (font)")]
    cases = [c for c in cases if c[0] in runs]
    fig, axes = plt.subplots(len(cases), 4, figsize=FIGSIZE, gridspec_kw={"width_ratios": [1, 1, 1, 1.6]})
    rows = []
    for row, (n, lab) in zip(np.atleast_2d(axes), cases):
        r = runs[n]
        sr = summ[(summ.kind + "__" + summ.image) == n].iloc[0]
        p3 = json.loads(sr.phase3_image_level_oof)
        row[0].imshow(r["original"][..., ::-1]); row[0].set_title(f"{lab}: original", fontsize=9)
        row[1].imshow(r["candidate_mask"], cmap="gray"); row[1].set_title("Phase 4 candidates", fontsize=9)
        row[2].imshow(r["overlay"][..., ::-1]); row[2].set_title(f"{primary.split('_')[0]} decision (recall mode)", fontsize=9)
        g = r["rows"]
        miss = {f: int(g[f"{f}_missing"].sum()) for f in FEATURES} if len(g) else {}
        p4 = {m.split("_")[0]: [round(float(v), 2) for v in g[f"p_{m}"]] for m in MODELS} if len(g) else {}
        kept4 = {m.split("_")[0]: int((g[f"p_{m}"] >= thr_rec[m]).sum()) for m in MODELS} if len(g) else {}
        txt = (f"Phase 3 image-level OOF p: S1 {p3['S1']}, S3 {p3['S3']}, S4 {p3['S4']}, RF {p3['RF']}\n"
               f"Phase 4: {len(g)} group(s), components per group {g.n_cc.tolist() if len(g) else []}\n"
               f"group p: " + ", ".join(f"{k} {v}" for k, v in p4.items()) + "\n"
               f"kept groups (own recall thr): " + ", ".join(f"{k} {v}/{len(g)}" for k, v in kept4.items()) + "\n"
               f"missing per feature: CV_g {miss.get('CV_g', '-')}, B {miss.get('B', '-')}, S_theta {miss.get('S_theta', '-')}")
        row[3].text(0, 0.5, txt, fontsize=8, va="center", family="monospace")
        row[3].axis("off")
        for ax in row[:3]:
            ax.set_xticks([]); ax.set_yticks([])
        rows.append({"case": lab, "run": n, "phase3_oof": json.dumps(p3), "groups": len(g), "phase4_p": json.dumps(p4),
                     "kept_groups_recall_thr": json.dumps(kept4), "missing": json.dumps(miss)})
    fig.suptitle("G15  Failure analysis: short neat handwriting misclassified in Phase 3 (image level) vs Phase 4 group level; "
                 "dev images scored by the fold model that excluded them", fontsize=10)
    save_both(fig, "G15_failure_analysis", pd.DataFrame(rows))
    t15 = pd.DataFrame(rows)
    notes["G15"] = dict(t="G15 Failure Analysis", p="Phase 3에서 정형으로 오분류된 짧은 수기(S6, S16, V8)와 font F8이 Phase 4에서 어떻게 처리되는지",
                        m="해당 이미지를 제외한 fold 모델로 Phase 4 그룹 단위 추론, 각 모델의 recall-priority 임계값",
                        d="dev handwritten_002(S6), 009(S16), 013(V8), formal_002_F8",
                        q="; ".join(f"{x.case}: Phase3 {x.phase3_oof} → Phase4 kept {x.kept_groups_recall_thr}, 결측 {x.missing}" for x in t15.itertuples()),
                        i="recall 우선 임계값에서: " + "; ".join(f"{x.case} → " + ", ".join(f"{k} {'보존' if v else '제거'}" for k, v in json.loads(x.kept_groups_recall_thr).items())
                                                               for x in t15.itertuples()) +
                          ". 수기를 모두 보존하는 모델은 font F8도 보존(정형 오보존) → 짧은 수기와 정형 2글자를 7특징으로 구별하지 못함. 2글자 표기는 CV_g·B 결측(학습 중앙값 대체).",
                        l="개선은 임계값 이동 효과이며 특징 자체가 짧은 수기를 구별하게 된 것은 아님. 후보 생성 방식 변경(gray Otsu → stroke)으로 특징 분포가 이동.")

    order = ["G10", "G11", "G12", "G13", "G14", "G15"]
    files = {"G10": "G10_extraction_pipeline", "G11": "G11_probability_map", "G12": "G12_seven_feature_comparison",
             "G13": "G13_model_threshold_comparison", "G14": "G14_extraction_statistics", "G15": "G15_failure_analysis"}
    md = ["# PAC2 Phase 4 발표 노트 (수기 획 추출)", "",
          f"- primary 모델: `{primary}` (규칙: 개발 OOF에서 수기 recall ≥ 0.97 임계값 중 formal FPR 최저, 동률 시 BA 최고)",
          "- 체크포인트: Phase 3에서 모델 파일이 저장되지 않아 개발 90장만으로 재적합(full-dev). 개발 시연 이미지는 그 이미지를 제외한 fold 모델 사용.",
          "- field_test 7장: 실제 사진, **정답 마스크 없음 → 픽셀 정확도(Dice/IoU) 미계산**. Test 30장 미사용.",
          "- 첫 실행 후 변경(정답 없이 시각 점검으로 결정, 공개): ① 후보 생성을 Phase 2 합의 마스크 → 멀티스케일 stroke 후보로 교체(녹·조명 덩어리 포함, 분필 누락), "
          "② 임계값 규칙 mean+3SD → median+5·MAD, ③ 면적 상한 10%→50%(폰트 글자 삭제), ④ 긴 선형 성분은 그룹 연결에서 제외.", ""]
    for g in order:
        n = notes[g]
        md += [f"## {n['t']}", f"![{g}](figures/{files[g]}.png)", "", f"1. **목적**: {n['p']}", f"2. **방법**: {n['m']}", f"3. **데이터**: {n['d']}",
               f"4. **정량 변화**: {n['q']}", f"5. **해석**: {n['i']}", f"6. **실패 원인·한계**: {n['l']}", ""]
    (ROOT / "results" / "phase4" / "presentation_notes.md").write_text("\n".join(md), encoding="utf-8")
    print("primary:", primary)
    for g in order:
        print(g, "|", notes[g]["q"][:260])
    return 0


def save_both(fig, name, data):
    DATA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    data.to_csv(DATA / f"{name}.csv", index=False, encoding="utf-8-sig")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
