"""Regenerate every presentation figure + notes from the saved Phase 3 logs only (no retraining).

Inputs (reports/phase3/): training_history.csv, feature_weights.csv, model_comparison.csv, fold_metrics.csv,
oof_predictions.csv, reweighting_history.csv, adaboost_history.csv, experiment_meta.json, extraction_masks/
Outputs: reports/phase3/figures/G*.png|svg|csv, reports/phase3/presentation_notes.md
Run: .venv\\Scripts\\python.exe scripts\\generate_presentation_figures.py
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
from pac2.visualization import (BASE, ERR, FEATURE_COLORS, FEATURE_LABELS, FEATURE_STYLES, FIGSIZE, FINAL, MID,  # noqa: E402
                                bar_with_err, confusion_panel, mean_sd_by_epoch, plt, save)

P3 = ROOT / "reports" / "phase3"
FIG = P3 / "figures"
C_LR = "C_shape_domain_matched|train_all|torch_logistic"
STAGES = [("S1_basic_unweighted", "S1 basic\n(unweighted, lambda=0.01, thr 0.5)"),
          ("S2_class_weighted", "S2 + class weights\n(N/2N_c, thr 0.5)"),
          (C_LR, "S3 + inner-CV lambda, H_t,\nthreshold"),
          ("S4_hard_example_reweighted", "S4 + hard-example\nreweighting")]


def load():
    d = {n: pd.read_csv(P3 / f"{n}.csv") for n in ("training_history", "feature_weights", "model_comparison", "fold_metrics",
                                                   "oof_predictions", "reweighting_history", "adaboost_history")}
    d["meta"] = json.loads((P3 / "experiment_meta.json").read_text(encoding="utf-8"))
    return d


def fold_mean_sd(fm, model, metric):
    v = fm.loc[fm.model == model, metric].astype(float)
    return float(v.mean()), float(v.std(ddof=0)), len(v)


def main() -> int:
    d = load()
    fm, mc, oof, th, fw = d["fold_metrics"], d["model_comparison"], d["oof_predictions"], d["training_history"], d["feature_weights"]
    exp_id = d["meta"]["experiment_id"]
    n_img = int(mc.n_images.iloc[0])
    logistic_family = [s for s, _ in STAGES]
    final = max(logistic_family, key=lambda m: float(mc.loc[mc.model == m, "oof_balanced_accuracy"].iloc[0]))
    baseline = "S1_basic_unweighted"
    notes = {}
    tag = f"{exp_id} | dev n={n_img}, 5-fold OOF"

    # ---------------- G1 loss convergence (Adam log of the S3 model, z = train_all)
    h = th[(th.model == C_LR) & (th.optimizer == "adam")]
    curves = mean_sd_by_epoch(h, "train_loss").merge(mean_sd_by_epoch(h, "val_loss"), on="epoch")
    for c in ("weighted_bce", "l2_loss"):
        curves = curves.merge(mean_sd_by_epoch(h, c), on="epoch")
    best_ep = int(curves.loc[curves.val_loss_mean.idxmin(), "epoch"])
    stops = h.groupby("fold").early_stop_epoch.first()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIGSIZE)
    for col, color, ls, lab in (("train_loss", MID, "-", "training loss (weighted BCE + L2)"), ("val_loss", FINAL, "--", "validation loss")):
        a1.plot(curves.epoch, curves[f"{col}_mean"], color=color, ls=ls, lw=2, label=f"{lab}, mean of 5 folds")
        a1.fill_between(curves.epoch, curves[f"{col}_mean"] - curves[f"{col}_sd"], curves[f"{col}_mean"] + curves[f"{col}_sd"], color=color, alpha=0.15)
    a1.axvline(best_ep, color=ERR, ls=":", lw=1.5, label=f"min mean val loss at epoch {best_ep} (monitor only)")
    a1.set(xlabel="epoch (Adam, lr = 0.05)", ylabel="loss (class-weighted BCE, nats)", title="G1a  Training vs validation loss (S3 logistic)")
    a1.legend(loc="upper right")
    a2.plot(curves.epoch, curves.weighted_bce_mean, color=MID, lw=2, label="weighted BCE term (train)")
    a2.plot(curves.epoch, curves.l2_loss_mean, color=BASE, lw=2, ls="--", label="L2 term  lambda*||w||^2 (train)")
    a2.set(xlabel="epoch", ylabel="loss term value", title="G1b  Loss components (mean of 5 folds)")
    a2.legend()
    fig.suptitle(f"Loss convergence - {tag}. Early stop (training-loss plateau) epochs per fold: "
                 f"{', '.join(str(int(v)) if v == v else 'none' for v in stops)}; stopped folds carry their final value forward", fontsize=11)
    save(fig, FIG, "G1_loss_convergence", curves)
    last = curves.iloc[-1]
    gap_end = last.val_loss_mean - last.train_loss_mean
    notes["G1"] = dict(
        title="G1 Loss Convergence", purpose="7-특징 로지스틱의 목적함수(가중 BCE + L2)가 실제로 수렴하는지, 과적합이 생기는지 확인",
        method="p = sigmoid(b + Σ w_j z_j), L = Σ c_i·BCE_i / Σ c_i + λ‖w‖², c_i = N/(2N_class) (fold별), Adam lr 0.05 최대 600 epoch, "
               "훈련 손실 정체 시 조기 종료. 검증 손실 최저 epoch은 기록만 하고 모델 선택에 쓰지 않음.",
        numbers=f"epoch 0 훈련 손실 {curves.train_loss_mean.iloc[0]:.3f} → 마지막 {last.train_loss_mean:.3f}; 검증 손실 "
                f"{curves.val_loss_mean.iloc[0]:.3f} → 최저 {curves.val_loss_mean.min():.3f}(epoch {best_ep}) → 마지막 {last.val_loss_mean:.3f}; "
                f"마지막 epoch 검증-훈련 차이 {gap_end:+.3f}. L2 항 마지막 {last.l2_loss_mean:.4f}.",
        change=f"훈련 손실 {curves.train_loss_mean.iloc[0] - last.train_loss_mean:.3f} 감소, 검증 손실 {curves.val_loss_mean.iloc[0] - last.val_loss_mean:.3f} 감소.",
        interpretation=("검증 손실이 최저점 이후 다시 증가 → 약한 과적합 신호." if last.val_loss_mean > curves.val_loss_mean.min() + 0.02
                        else "검증 손실이 최저점 근처에서 유지 → 뚜렷한 과적합 없음.") + " 손실 감소는 형상 분류의 개선이며 픽셀 추출 성능과는 별개.",
        limits="formal 검증 샘플이 fold당 3장이라 검증 손실 변동이 큼. 최종 OOF 모델은 LBFGS 해이고 Adam 곡선은 최적화 과정 설명용.")

    # ---------------- G2 feature weight evolution (same standardization only: train_all z-space)
    wcols = [f"w{j}" for j in range(7)]
    evo = pd.concat([mean_sd_by_epoch(h, c).set_index("epoch") for c in wcols], axis=1).reset_index()
    fin = fw[(fw.model == C_LR) & (fw.optimizer == "lbfgs") & (fw["stage"] == "final")]
    fin_ms = fin.groupby("feature").weight.agg(["mean", "std"]).reindex(["CV_w", "CV_h", "CV_g", "S_theta", "B", "CV_A", "R"])
    fin_ms["std"] = fin.groupby("feature").weight.std(ddof=0).reindex(fin_ms.index)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"width_ratios": [1.3, 1]})
    for j, (lab, col, ls) in enumerate(zip(FEATURE_LABELS, FEATURE_COLORS, FEATURE_STYLES)):
        a1.plot(evo.epoch, evo[f"w{j}_mean"], color=col, ls=ls, lw=2, label=lab)
        a1.fill_between(evo.epoch, evo[f"w{j}_mean"] - evo[f"w{j}_sd"], evo[f"w{j}_mean"] + evo[f"w{j}_sd"], color=col, alpha=0.08)
    a1.axhline(0, color="black", lw=0.8)
    a1.set(xlabel="epoch (Adam)", ylabel="weight w_j (per 1 SD of feature, train-fold z)", title="G2a  Weight trajectories (mean ± SD, 5 folds)")
    a1.legend(ncol=2, fontsize=10)
    x = np.arange(7)
    a2.bar(x - 0.2, np.zeros(7), 0.4, color=BASE, edgecolor="black", label="initial (zero init)")
    bar_with_err(a2, x + 0.2, fin_ms["mean"].values, fin_ms["std"].values, FINAL, "final (LBFGS, mean ± SD of 5 folds)", width=0.4)
    a2.axhline(0, color="black", lw=0.8)
    a2.set_xticks(x, FEATURE_LABELS, rotation=30)
    a2.set(ylabel="weight (z-space)", title="G2b  Initial vs final weights")
    a2.legend(fontsize=10)
    fig.suptitle(f"Feature weights - S3 logistic, train-fold standardization only ({tag}). Formal-reference z weights are on a "
                 "different scale and are NOT plotted here. |w| is not causal importance.", fontsize=11)
    save(fig, FIG, "G2_feature_weight_evolution",
         pd.concat([evo, fin_ms.reset_index().rename(columns={"mean": "final_mean_lbfgs", "std": "final_sd_lbfgs"})], axis=1))
    sign = fin.groupby("feature").weight.apply(lambda v: bool((v > 0).all() or (v < 0).all()))
    notes["G2"] = dict(
        title="G2 Feature Weight Evolution", purpose="7개 PDF 특징의 가중치가 학습 중 어떻게 변하고 최종적으로 어떤 값/안정성을 갖는지",
        method="학습 fold 기준 z = (x-μ_train)/σ_train 공간의 계수. 모든 가중치 0에서 시작(초기 막대 = 0). 최종값은 LBFGS 해의 5-fold 평균±SD.",
        numbers="; ".join(f"{f} {fin_ms.loc[f, 'mean']:+.2f}±{fin_ms.loc[f, 'std']:.2f} ({'부호 일관' if sign[f] else '부호 불일치'})" for f in fin_ms.index),
        change="초기 0 → 최종값(위 수치). 양수 = 값이 클수록 수기 쪽.",
        interpretation="같은 표준화 기준 안에서만 비교 가능. 계수 크기는 다른 특징과의 상관·분리 가능성(거의 완전 분리 → 계수 불안정)에 좌우되므로 "
                       "인과적 중요도나 실제 마킹 추출 개선 효과로 해석하지 않음.",
        limits="formal_ref 표준화 모델 계수는 척도가 달라 이 그래프에서 제외(비교 불가). fold 간 SD가 평균의 ~50%로 크다.")

    # ---------------- G3 before vs after optimization (identical C features, folds)
    metrics = [("balanced_accuracy", "Balanced accuracy"), ("precision", "Handwritten precision"), ("recall", "Handwritten recall"), ("f1", "F1")]
    rows = []
    for s, lab in STAGES:
        r = {"stage": s, "label": lab.replace("\n", " ")}
        for m, _ in metrics + [("log_loss", "")]:
            r[f"{m}_mean"], r[f"{m}_sd"], r["n_folds"] = fold_mean_sd(fm, s, m)
            r[f"oof_{m}"] = float(mc.loc[mc.model == s, f"oof_{m}"].iloc[0])
        rows.append(r)
    g3 = pd.DataFrame(rows)
    colors = [BASE if s == baseline else FINAL if s == final else MID for s, _ in STAGES]
    hatches = [None, "//", "\\\\", None]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"width_ratios": [3, 1]})
    width = 0.2
    for i, (s, lab) in enumerate(STAGES):
        xs = np.arange(len(metrics)) + (i - 1.5) * width
        bar_with_err(a1, xs, g3.loc[i, [f"{m}_mean" for m, _ in metrics]].values, g3.loc[i, [f"{m}_sd" for m, _ in metrics]].values,
                     colors[i], lab.replace("\n", " "), hatch=hatches[i], width=width, rotate=True)
    a1.set_xticks(np.arange(len(metrics)), [l for _, l in metrics])
    a1.set_ylim(0, 1.2)
    a1.set(ylabel="score (fold mean ± SD, 5 folds)", title="G3a  Classification metrics by optimization stage")
    a1.legend(fontsize=9, loc="lower right")
    bar_with_err(a2, np.arange(4), g3.log_loss_mean.values, g3.log_loss_sd.values, colors, None, width=0.6)
    a2.set_xticks(np.arange(4), ["S1", "S2", "S3", "S4"])
    a2.set(ylabel="log loss (fold mean ± SD; lower = better)", title="G3b  Log loss")
    d_ba = [(g3.balanced_accuracy_mean[i] - g3.balanced_accuracy_mean[0]) * 100 for i in range(4)]
    d_rec = [(g3.recall_mean[i] - g3.recall_mean[0]) * 100 for i in range(4)]
    fig.suptitle(f"Before vs after optimization - same C features and folds ({tag}); final = {final} (highest OOF BA in logistic family)\n"
                 "BA change vs S1: " + ", ".join(f"S{i + 1} {v:+.1f} pp" for i, v in enumerate(d_ba)) +
                 "   |   handwritten recall change vs S1: " + ", ".join(f"S{i + 1} {v:+.1f} pp" for i, v in enumerate(d_rec)) +
                 ("  (recall DECREASED)" if min(d_rec[1:]) < 0 else ""), fontsize=11, color="black")
    save(fig, FIG, "G3_before_after_optimization", g3)
    deltas = {m: [(g3[f"{m}_mean"][i] - g3[f"{m}_mean"][i - 1]) * 100 for i in range(1, 4)] for m, _ in metrics}
    notes["G3"] = dict(
        title="G3 Before vs After Optimization", purpose="불균형 가중치, 정규화/임계값 최적화, hard-example 재가중이 각각 성능을 바꾸는지",
        method="동일한 C(도메인 정합) 특징·동일 fold. S1 무가중 BCE(λ=0.01, 임계 0.5) → S2 클래스 가중 → S3 내부 3-fold CV로 λ·H_t·임계값 선택 → "
               "S4 S3 + AdaBoost-inspired hard-example reweighting(10 rounds, α=clip(exp(EMA 상대손실-1),0.5,3), 훈련 샘플만). AdaBoost 알고리즘 아님.",
        numbers="; ".join(f"{r.stage}: BA {r.balanced_accuracy_mean:.3f}±{r.balanced_accuracy_sd:.3f}, P {r.precision_mean:.3f}, R {r.recall_mean:.3f}, "
                          f"F1 {r.f1_mean:.3f}, log loss {r.log_loss_mean:.3f}" for r in g3.itertuples()),
        change="단계별 변화(pp, 직전 대비): " + "; ".join(f"{l}: " + ", ".join(f"{v:+.1f}" for v in deltas[m]) for m, l in metrics),
        interpretation=f"최종 = {final}. BA는 S1 대비 {d_ba[3]:+.1f} pp. 단계별 이득의 대부분은 formal(소수 클래스) 오류 감소에서 나오며, "
                       "handwritten recall은 클래스 가중 후 오히려 낮아질 수 있음(위 수치 그대로).",
        limits="formal 15장 → formal 1장 = BA 약 3.3 pp. S3→S4 차이는 1~2장 수준으로 통계적으로 구분되지 않음. 새 데이터에서 재확인 필요.")

    # ---------------- G4 model comparison (C features; fair metrics only)
    comp = [(C_LR, "Logistic S3", MID), ("S4_hard_example_reweighted", "Logistic S4 (reweighted)", FINAL if final == "S4_hard_example_reweighted" else MID),
            ("C_shape_domain_matched|train_all|random_forest", "Random Forest", BASE), ("C_shape_domain_matched|train_all|adaboost", "AdaBoost", BASE),
            ("C_shape_domain_matched|train_all|gp_classifier", "GP classifier", BASE), ("C_shape_domain_matched|train_all|one_class_svm", "One-Class SVM", BASE)]
    mets = metrics
    rows = []
    for m_, lab, _ in comp:
        r = {"model": m_, "label": lab}
        for k, _ in mets:
            r[f"{k}_mean"], r[f"{k}_sd"], r["n_folds"] = fold_mean_sd(fm, m_, k)
        rows.append(r)
    g4 = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    width = 0.13
    hatch = [None, None, "//", "\\\\", "xx", ".."]
    for i, (m_, lab, col) in enumerate(comp):
        xs = np.arange(len(mets)) + (i - 2.5) * width
        bar_with_err(ax, xs, g4.loc[i, [f"{k}_mean" for k, _ in mets]].values, g4.loc[i, [f"{k}_sd" for k, _ in mets]].values, col, lab,
                     hatch=hatch[i], width=width)
    ax.set_xticks(np.arange(len(mets)), [l for _, l in mets])
    ax.set_ylim(0, 1.1)
    ax.set(ylabel="score (fold mean ± SD, 5 folds)", title="G4  Model comparison on identical 7 domain-matched shape features")
    ax.legend(ncol=3, fontsize=10, loc="lower right")
    fig.suptitle(f"{tag}. Thresholds: logistic / RF / AdaBoost / GP from inner CV; One-Class SVM decision >= 0 (nu = 0.1, trained on handwritten only). "
                 "AUC / log loss omitted: One-Class SVM gives no probability.", fontsize=10)
    save(fig, FIG, "G4_model_comparison", g4)
    notes["G4"] = dict(
        title="G4 Model Comparison", purpose="같은 7개 특징·fold에서 선형/비선형/커널/단일 클래스 모델 비교",
        method="Logistic(PyTorch), RandomForest(300 trees, balanced), AdaBoost(SAMME stumps 50, 클래스 가중 초기화), GP classifier(RBF, length scale 0.1~10 제한), "
               "One-Class SVM(handwritten만 학습, nu=0.1 사전 고정). 공통 지표 BA/F1/Precision/Recall만 표시.",
        numbers="; ".join(f"{r.label}: BA {r.balanced_accuracy_mean:.3f}±{r.balanced_accuracy_sd:.3f}, F1 {r.f1_mean:.3f}, P {r.precision_mean:.3f}, R {r.recall_mean:.3f}"
                          for r in g4.itertuples()),
        change="모델 간 BA 차이(최고-최저): " + f"{(g4.balanced_accuracy_mean.max() - g4.balanced_accuracy_mean.min()) * 100:.1f} pp",
        interpretation="RF가 BA 최고이나 2~3장 차이. One-Class SVM은 수기만으로 허용 범위를 학습하면 수기 1/3을 밀어냄 → 7개 특징만으로는 '수기 영역'이 촘촘히 정의되지 않음.",
        limits="모델마다 목적·점수 의미가 다름(OCSVM은 확률 아님). GP는 길이척도 발산 후 경계 제한을 추가한 결과. 개발 데이터 OOF이며 test 아님.")

    # ---------------- G5 probability distribution (OOF only)
    fo = oof[oof.model == final]
    oc = oof[oof.model == "C_shape_domain_matched|train_all|one_class_svm"]
    thr = [json.loads(s)["threshold"] if s.startswith("{") else float(s.split("thr=")[1].split()[0]) for s in fm.loc[fm.model == final, "selected"]]
    bins = np.linspace(0, 1, 21)
    hf, _ = np.histogram(fo.score[fo.label == 0], bins)
    hh, _ = np.histogram(fo.score[fo.label == 1], bins)
    overlap = float(np.minimum(hf / max(1, hf.sum()), hh / max(1, hh.sum())).sum())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"width_ratios": [1.4, 1]})
    a1.hist(fo.score[fo.label == 1], bins, color=FINAL, alpha=0.6, edgecolor="black", label=f"actual handwritten (n={int((fo.label == 1).sum())})")
    a1.hist(fo.score[fo.label == 0], bins, color=BASE, alpha=0.8, edgecolor="black", hatch="//", label=f"actual formal (n={int((fo.label == 0).sum())})")
    for t in thr:
        a1.axvline(t, color=ERR, ls=":", lw=1)
    a1.axvline(float(np.median(thr)), color=ERR, ls="--", lw=2, label=f"fold thresholds (dotted), median {np.median(thr):.2f}")
    a1.set(xlabel="out-of-fold P(handwritten)", ylabel="number of images", title=f"G5a  OOF probabilities - {final}")
    a1.legend(fontsize=10)
    a2.hist(oc.score[oc.label == 1], 20, color=FINAL, alpha=0.6, edgecolor="black", label="actual handwritten")
    a2.hist(oc.score[oc.label == 0], 20, color=BASE, alpha=0.8, edgecolor="black", hatch="//", label="actual formal")
    a2.axvline(0, color=ERR, ls="--", lw=2, label="decision = 0")
    a2.set(xlabel="One-Class SVM decision score (NOT a probability)", ylabel="number of images", title="G5b  One-Class SVM scores (own scale)")
    a2.legend(fontsize=10)
    fig.suptitle(f"Prediction distributions, out-of-fold only ({tag}). Histogram overlap (sum of min of normalized counts) = {overlap:.2f}", fontsize=11)
    save(fig, FIG, "G5_probability_distribution", pd.concat([fo.assign(panel="final_probability"), oc.assign(panel="ocsvm_decision")]))
    notes["G5"] = dict(
        title="G5 Prediction Probability Distribution", purpose="수기/정형의 OOF 예측 확률이 얼마나 분리·중첩되는지",
        method="각 이미지는 자기 fold가 검증일 때의 예측만 사용(학습 데이터 직접 예측 아님). 임계값 = 각 fold 학습 데이터 내부 CV에서 BA 최대.",
        numbers=f"fold 임계값 {', '.join(f'{t:.2f}' for t in thr)}; 정규화 히스토그램 중첩 {overlap:.2f}; formal 평균 p {fo.score[fo.label == 0].mean():.3f}, "
                f"handwritten 평균 p {fo.score[fo.label == 1].mean():.3f}; handwritten 중 p<0.5 {int(((fo.label == 1) & (fo.score < 0.5)).sum())}장.",
        change="—", interpretation="formal은 0 부근에 몰리고 handwritten은 1 부근이나, 짧고 단정한 수기(2글자)가 낮은 확률 쪽 꼬리를 형성.",
        limits="formal 15장의 분포라 히스토그램이 거칠다. OCSVM 점수는 확률과 다른 척도(별도 패널).")

    # ---------------- G6 domain bias
    dom = [("A_shape_original|train_all|torch_logistic", "A: 7 shape features\n(original masks)", MID),
           ("B_domain_only|train_all|torch_logistic", "B: imaging-domain only\n(control, not deployable)", ERR),
           (C_LR, "C: 7 shape features\n(domain-matched masks)", FINAL)]
    rows = []
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, (m_, lab, col) in enumerate(dom):
        v = fm.loc[fm.model == m_, "balanced_accuracy"].astype(float).values
        ax.bar(i, v.mean(), 0.6, yerr=v.std(), color=col, edgecolor="black", capsize=5, alpha=0.8, label=lab.replace("\n", " "),
               hatch=[None, "xx", "//"][i])
        ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, color="black", zorder=3, s=25)
        ax.annotate(f"{v.mean():.3f} ± {v.std():.3f}", (i, v.mean()), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=12)
        rows += [{"model": m_, "fold": f, "balanced_accuracy": x} for f, x in enumerate(v)]
    ax.set_xticks(range(3), [l for _, l, _ in dom])
    ax.set_ylim(0, 1.1)
    ax.set(ylabel="balanced accuracy (per fold dots, bar = mean ± SD)", title="G6  Domain-bias control: shape vs imaging-domain features (logistic)")
    ax.legend(fontsize=10, loc="lower right")
    fig.suptitle(f"{tag}. B separates the classes perfectly: acquisition differences alone suffice. C removes colour / polarity / resolution / sharpness only.", fontsize=10)
    save(fig, FIG, "G6_domain_bias", pd.DataFrame(rows))
    va = {m_: fold_mean_sd(fm, m_, "balanced_accuracy") for m_, _, _ in dom}
    notes["G6"] = dict(
        title="G6 Domain Bias Comparison", purpose="성능이 손글씨 형상 때문인지 촬영 조건 차이 때문인지 점검",
        method="같은 fold의 로지스틱: A 원본 마스크 형상 7특징, B 채도·선명도·크기·밝기·극성만, C 마스크 단계 도메인 정합(크롭+균일 리스케일 H_t+재이진화) 후 형상 7특징.",
        numbers="; ".join(f"{l.splitlines()[0]} BA {va[m_][0]:.3f}±{va[m_][1]:.3f}" for m_, l, _ in dom),
        change=f"A→C BA {(va[C_LR][0] - va[dom[0][0]][0]) * 100:+.1f} pp",
        interpretation="B가 완벽 → 데이터만으로는 '형상 학습'을 증명할 수 없음. C에서 성능이 거의 유지 → 제거한 도메인 단서 없이도 형상 특징이 분리력을 가짐.",
        limits="C도 '단일 컴퓨터 폰트 vs 손글씨'라는 근본 교란은 제거 못함. 편향이 완전히 제거되었다고 주장하지 않음.")

    # ---------------- G7 confusion matrices
    def cm(model):
        r = mc.loc[mc.model == model].iloc[0]
        o = oof[oof.model == model]
        tp = int(((o.label == 1) & (o.pred == 1)).sum()); fn = int(((o.label == 1) & (o.pred == 0)).sum())
        tn = int(((o.label == 0) & (o.pred == 0)).sum()); fp = int(((o.label == 0) & (o.pred == 1)).sum())
        return tn, fp, fn, tp
    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIGSIZE)
    c1, c2 = cm(baseline), cm(final)
    confusion_panel(a1, *c1, f"Baseline: {baseline}", BASE)
    confusion_panel(a2, *c2, f"Final: {final}", FINAL)
    fig.suptitle(f"G7  Confusion matrices, pooled out-of-fold ({tag}); red = errors. Formal has only 15 images: 1 error = 6.7 % of the row.", fontsize=11)
    save(fig, FIG, "G7_confusion_matrix", pd.DataFrame([dict(model=baseline, tn=c1[0], fp=c1[1], fn=c1[2], tp=c1[3]),
                                                        dict(model=final, tn=c2[0], fp=c2[1], fn=c2[2], tp=c2[3])]))
    notes["G7"] = dict(
        title="G7 Confusion Matrix", purpose="클래스별 오류(특히 소수 formal)를 기본 모델과 최종 모델에서 비교",
        method="pooled OOF 예측(각 fold 임계값 적용). 행 = 실제 클래스, 비율 = 행 기준.",
        numbers=f"{baseline}: 수기→수기 {c1[3]}, 수기→정형 {c1[2]}, 정형→정형 {c1[0]}, 정형→수기 {c1[1]}; "
                f"{final}: 수기→수기 {c2[3]}, 수기→정형 {c2[2]}, 정형→정형 {c2[0]}, 정형→수기 {c2[1]}",
        change=f"정형→수기 오류 {c1[1]} → {c2[1]}, 수기→정형 오류 {c1[2]} → {c2[2]}",
        interpretation="기본 모델은 다수 클래스(수기) 쪽으로 기울어 formal을 수기로 판정. 최종 모델은 formal 오류를 줄였지만 수기를 정형으로 버리는 오류가 늘 수 있음(추출 단계에서 수기 획 손실에 해당).",
        limits="formal 15장, 정확도 단일값 대신 행 비율로 해석. test 결과 아님.")

    # ---------------- G8 extraction comparison (dev images, OOF decisions, no GT -> no Dice / IoU)
    fo = oof[oof.model == final].set_index("image_id")
    def pick(cond, n):
        return list(fo[cond].sort_values("score").index[:n])
    ok_h = list(fo[(fo.label == 1) & (fo.pred == 1)].sort_values("score", ascending=False).index[:2])
    ok_f = pick((fo.label == 0) & (fo.pred == 0), 1)
    fail = list(fo[fo.pred != fo.label].index[:3])
    sel = [(i, "correct") for i in ok_h + ok_f] + [(i, "FAILURE") for i in fail]
    fig, axes = plt.subplots(len(sel), 4, figsize=(13.333, 1.25 * len(sel) + 1.2))
    rows = []
    for row, (iid, kind) in zip(np.atleast_2d(axes), sel):
        img = cv2.imdecode(np.fromfile(str(ROOT / "Source" / iid), np.uint8), cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.fromfile(str(P3 / "extraction_masks" / iid.replace("/", "__")), np.uint8), cv2.IMREAD_GRAYSCALE)
        keep = int(fo.loc[iid, "pred"]) == 1
        kept = mask if keep else np.zeros_like(mask)
        ov = img.copy()
        ov[mask > 0] = (0, 160, 0) if keep else (40, 40, 214)
        for ax, im, t in zip(row, (img[..., ::-1], mask, kept, ov[..., ::-1]),
                             ("1 original", "2 OpenCV candidate (gray Otsu)", "3 kept by final model", "4 binary mask overlay")):
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(t, fontsize=9)
        row[0].set_ylabel(f"{iid.split('/')[1][:-4]}\nlabel {int(fo.loc[iid, 'label'])}, p={fo.loc[iid, 'score']:.2f}\n{kind}",
                          fontsize=8, color=ERR if kind == "FAILURE" else "black", rotation=0, ha="right", va="center")
        rows.append({"image_id": iid, "label": int(fo.loc[iid, "label"]), "oof_p": float(fo.loc[iid, "score"]), "kept": keep, "case": kind})
    fig.suptitle("G8  Extraction on DEVELOPMENT crops with out-of-fold decisions (green = kept strokes, red = removed). "
                 "No ground-truth stroke masks -> no Dice / IoU. Not steel-plate images.", fontsize=10)
    fig.tight_layout()
    save(fig, FIG, "G8_extraction_comparison", pd.DataFrame(rows))
    notes["G8"] = dict(
        title="G8 Final Extraction Comparison", purpose="후보 생성 → 최종 모델 보존 → 이진 마스크 과정을 같은 이미지·같은 해상도로 보여줌(성공/실패 포함)",
        method="후보 = gray Otsu 마스크(획 픽셀). 이미지 단위 OOF 판정 p ≥ fold 임계값이면 후보 획 전체 보존, 아니면 제거. Bounding box 채우기 없음.",
        numbers="; ".join(f"{r['image_id']}: label {r['label']}, p {r['oof_p']:.2f}, {'보존' if r['kept'] else '제거'} ({r['case']})" for r in rows),
        change="—", interpretation="실패 사례는 짧고 단정한 수기가 정형으로 판정되어 수기 획 전체가 제거된 경우. 현재 판정은 이미지(ROI) 단위라 한 이미지 안의 수기/비수기 획을 나누지 못함.",
        limits="정답 획 마스크가 없어 Dice/IoU 미계산. 강판·녹·스크래치 이미지가 아님(개발 크롭).")

    # ---------------- G9 reweighting / AdaBoost process (logged iterations)
    rw, ad = d["reweighting_history"], d["adaboost_history"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIGSIZE)
    g_rw = rw.groupby("round")[["alpha_mean_formal", "alpha_mean_handwritten", "alpha_max", "train_misclassified", "train_mean_bce"]].mean().reset_index()
    a1.plot(g_rw["round"], g_rw.alpha_mean_formal, color=BASE, ls="--", marker="s", label="mean alpha, formal")
    a1.plot(g_rw["round"], g_rw.alpha_mean_handwritten, color=FINAL, marker="o", label="mean alpha, handwritten")
    a1.plot(g_rw["round"], g_rw.alpha_max, color=ERR, ls=":", marker="^", label="max alpha (clip 3.0)")
    a1b = a1.twinx()
    a1b.bar(g_rw["round"], g_rw.train_misclassified, color=MID, alpha=0.25, label="train misclassified (mean of folds)")
    a1b.set_ylabel("misclassified training images")
    a1.set(xlabel="reweighting round", ylabel="sample multiplier alpha", title="G9a  Hard-example reweighting (PyTorch logistic, 5 folds)")
    a1.legend(loc="upper left", fontsize=9)
    g_ad = ad.groupby("iteration")[["estimator_error_sklearn", "weight_share_formal", "ensemble_train_misclassified", "val_balanced_accuracy"]].mean().reset_index()
    a2.plot(g_ad.iteration, g_ad.estimator_error_sklearn, color=ERR, ls="--", label="weak-learner weighted error")
    a2.plot(g_ad.iteration, g_ad.weight_share_formal, color=BASE, label="share of sample weight on formal")
    a2.plot(g_ad.iteration, g_ad.val_balanced_accuracy, color=FINAL, ls="-.", label="validation BA (staged, thr 0.5)")
    a2.set(xlabel="AdaBoost iteration", ylabel="value (0-1)", title="G9b  sklearn AdaBoost (SAMME stumps) - separate algorithm")
    a2.legend(fontsize=9)
    recon = float((ad.estimator_error_sklearn - ad.estimator_error_reconstructed).abs().max())
    fig.suptitle(f"Reweighting processes ({tag}). AdaBoost sample weights reconstructed with the SAMME rule; max |error diff| vs sklearn = {recon:.1e}", fontsize=10)
    save(fig, FIG, "G9_reweighting_process", pd.concat([g_rw.assign(process="hard_example_reweighting"), g_ad.assign(process="adaboost")]))
    notes["G9"] = dict(
        title="G9 Reweighting Process (보조)", purpose="hard-example reweighting과 AdaBoost가 반복마다 샘플 가중치를 어떻게 바꾸는지(서로 다른 알고리즘)",
        method="PyTorch: 라운드마다 LBFGS 재학습, α_i = clip(exp(EMA 상대손실-1), 0.5, 3), 평균 1 정규화. AdaBoost: SAMME, w_i ← w_i·exp(α_m·[오분류]).",
        numbers=f"재가중 라운드 1→10: formal 평균 α {g_rw.alpha_mean_formal.iloc[0]:.2f}→{g_rw.alpha_mean_formal.iloc[-1]:.2f}, 수기 평균 α "
                f"{g_rw.alpha_mean_handwritten.iloc[0]:.2f}→{g_rw.alpha_mean_handwritten.iloc[-1]:.2f}, 훈련 오분류 {g_rw.train_misclassified.iloc[0]:.1f}→{g_rw.train_misclassified.iloc[-1]:.1f}; "
                f"AdaBoost formal 가중치 비중 {g_ad.weight_share_formal.iloc[0]:.2f}→{g_ad.weight_share_formal.iloc[-1]:.2f}, 검증 BA {g_ad.val_balanced_accuracy.iloc[0]:.3f}→{g_ad.val_balanced_accuracy.iloc[-1]:.3f}; "
                f"SAMME 재구성 오차 {recon:.1e}.",
        change="위 수치의 시작→끝 변화", interpretation="두 방법 모두 어려운(오분류) 샘플에 가중치를 몰아줌. 재가중은 검증 데이터에 적용하지 않음.",
        limits="라운드 수·η·clip은 사전 고정(검증 튜닝 안 함). 개선 폭은 1~2장 수준.")

    # ---------------- presentation notes
    order = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"]
    files = {"G1": "G1_loss_convergence", "G2": "G2_feature_weight_evolution", "G3": "G3_before_after_optimization", "G4": "G4_model_comparison",
             "G5": "G5_probability_distribution", "G6": "G6_domain_bias", "G7": "G7_confusion_matrix", "G8": "G8_extraction_comparison",
             "G9": "G9_reweighting_process"}
    md = [f"# PAC2 Phase 3 발표 노트", "", f"- 실험 ID: `{exp_id}` (seed {d['meta']['seed']})",
          f"- 데이터: **개발 세트 {n_img}장 5-fold 교차검증 OOF 결과**. 최종 Test 30장은 아직 사용하지 않음(이 노트의 어떤 수치도 test 결과가 아님).",
          f"- 최종 모델 규칙: 로지스틱 계열 단계 중 OOF BA 최고 → `{final}`. 기본 모델: `{baseline}`.",
          "- 색상: 기본 = 회색, 보정/중간 = 파랑, 최종 = 초록, 오류/저하 = 빨강. 모든 그림은 `figures/`에 PNG(300 dpi)·SVG·CSV로 저장.", ""]
    for g in order:
        n = notes[g]
        md += [f"## {n['title']}", f"![{g}](figures/{files[g]}.png)", "", f"1. **실험 목적**: {n['purpose']}", f"2. **수학적 방법**: {n['method']}",
               f"3. **관찰 수치**: {n['numbers']}", f"4. **보정 전후 변화량**: {n['change']}", f"5. **해석**: {n['interpretation']}",
               f"6. **한계 및 추가 검증**: {n['limits']}", ""]
    (P3 / "presentation_notes.md").write_text("\n".join(md), encoding="utf-8")
    print(f"final model = {final}; figures -> {FIG}")
    for g in order:
        print(f"{g}: {notes[g]['numbers'][:220]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
