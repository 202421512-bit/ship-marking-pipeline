"""G16-G20 + GIF: how the logged models evolved (no training; reads saved logs only).

Fold standardization (median, mu, sigma) was not logged in Phase 3. It is recomputed deterministically from the same
development features and fold split (FoldPrep on the training fold only) and then VERIFIED: with the logged weights/bias
of every epoch it must reproduce the logged training loss, validation loss and per-class mean validation probability.
If the maximum deviation exceeds 1e-6 the figures are not drawn.
Fold rule (fixed before plotting): the lowest-index fold whose Adam run was not early-stopped.
Run: .venv\\Scripts\\python.exe scripts\\progress_evolution.py
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
from pac2.candidates import candidates, read_bgr  # noqa: E402
from pac2.features import FEATURES, extract  # noqa: E402
from pac2.logistic import class_weights  # noqa: E402
from pac2.normalize import normalize_mask  # noqa: E402
from pac2.prep import FoldPrep  # noqa: E402
from pac2.visualization import BASE, ERR, FIGSIZE, FINAL, MID, plt  # noqa: E402

OUT = ROOT / "results" / "progress_evolution"
MODEL = "C_shape_domain_matched|train_all|torch_logistic"
HW, FO = FINAL, "#555555"           # handwritten = green, formal = dark grey (consistent in all panels)
IX, IY = FEATURES.index("CV_h"), FEATURES.index("S_theta")


def save(fig, name, data):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    data.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    plt.close(fig)


def sig(z):
    return 1 / (1 + np.exp(-z))


def wbce(p, y, cw):
    w = np.array([cw[int(v)] for v in y])
    return float(np.sum(w * -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))) / w.sum())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    hist = pd.read_csv(ROOT / "reports" / "phase3" / "training_history.csv")
    h = hist[(hist.model == MODEL) & (hist.optimizer == "adam")]
    stopped = h.groupby("fold").early_stop_epoch.first()
    fold = int(min(f for f, s in stopped.items() if not np.isfinite(s)))
    hf = h[h.fold == fold].sort_values("epoch").reset_index(drop=True)
    st = json.loads(hf.settings.iloc[0])
    H_t, lam, thr = int(st["H_t"]), float(st["lambda"]), float(st["threshold"])

    # ---- deterministic feature + fold-standardization reconstruction (dev only)
    with open(ROOT / "data" / "splits" / "split_manifest.csv", encoding="utf-8-sig") as f:
        dev = sorted([r for r in csv.DictReader(f) if r["set"] == "dev"], key=lambda r: r["image_id"])
    y = np.array([int(r["label"]) for r in dev])
    folds = np.array([int(r["fold"]) for r in dev])
    ccfg = dict(cfg["candidates"], methods=["gray_otsu"])
    X = np.array([[v for v in (lambda fx: [fx[k] for k in FEATURES])(extract(normalize_mask(candidates(read_bgr(ROOT / r["path"]), ccfg)["gray_otsu"], H_t),
                                                                                 cfg["features"]))] for r in dev], float)
    tr, va = folds != fold, folds == fold
    prep = FoldPrep("train_all").fit(X[tr], y[tr])
    Ztr, Zva = prep.transform(X[tr]), prep.transform(X[va])
    cw = class_weights(y[tr])
    W = hf[[f"w{j}" for j in range(7)]].to_numpy(float)
    B = hf.bias.to_numpy(float)
    Ptr, Pva = sig(Ztr @ W.T + B), sig(Zva @ W.T + B)                      # (n, epochs)
    rec_train = np.array([wbce(Ptr[:, e], y[tr], cw) + lam * (W[e] ** 2).sum() for e in range(len(hf))])
    rec_val = np.array([wbce(Pva[:, e], y[va], cw) for e in range(len(hf))])
    dev_ = {"train_loss": float(np.abs(rec_train - hf.train_loss).max()), "val_weighted_bce": float(np.abs(rec_val - hf.val_weighted_bce).max()),
            "val_mean_p_formal": float(np.abs(Pva[y[va] == 0].mean(0) - hf.val_mean_p_formal).max()),
            "val_mean_p_handwritten": float(np.abs(Pva[y[va] == 1].mean(0) - hf.val_mean_p_handwritten).max())}
    ok = max(dev_.values()) < 1e-6
    verify = pd.DataFrame([{"check": k, "max_abs_diff_vs_log": v, "pass": v < 1e-6} for k, v in dev_.items()] +
                          [{"check": "test_images_used", "max_abs_diff_vs_log": 0, "pass": True},
                           {"check": "validation_images_in_standardization_fit", "max_abs_diff_vs_log": int((tr & va).sum()), "pass": not (tr & va).any()}])
    verify.to_csv(OUT / "verification.csv", index=False, encoding="utf-8-sig")
    print(f"fold {fold} (H_t {H_t}, lambda {lam}, thr {thr}); reconstruction check:", dev_, "PASS" if ok else "FAIL")
    if not ok:
        print("Standardization could not be reconstructed exactly - figures G16-G18 NOT drawn.")
        return 1
    final_ep = int(hf.epoch.max())
    eps = [e for e in (10, 100, final_ep) if e in set(hf.epoch)]
    erow = {e: int(np.nonzero(hf.epoch.to_numpy() == e)[0][0]) for e in [0] + eps}

    # ---- 2-D conditional slice helpers (other 5 features at the training-fold median)
    med = prep.median
    xs = X[:, IX]; ys = X[:, IY]
    xg = np.linspace(np.nanmin(xs) - 0.05 * (np.nanmax(xs) - np.nanmin(xs)), np.nanmax(xs) + 0.05 * (np.nanmax(xs) - np.nanmin(xs)), 220)
    yg = np.linspace(np.nanmin(ys) - 0.05 * (np.nanmax(ys) - np.nanmin(ys)), np.nanmax(ys) + 0.05 * (np.nanmax(ys) - np.nanmin(ys)), 220)
    GX, GY = np.meshgrid(xg, yg)
    grid = np.tile(med, (GX.size, 1)); grid[:, IX] = GX.ravel(); grid[:, IY] = GY.ravel()
    Zg = (grid - prep.mu) / prep.sigma
    plot_ok = np.isfinite(xs) & np.isfinite(ys)

    def scatter(ax):
        for mask_, mk, edge, lab in ((tr, "o", "none", "train"), (va, "^", "black", "validation")):
            for c, col, name in ((1, HW, "handwritten"), (0, FO, "formal")):
                s = mask_ & (y == c) & plot_ok
                ax.scatter(xs[s], ys[s], c=col, marker=mk, s=28 if mk == "o" else 60, edgecolors=edge, linewidths=0.8,
                           alpha=0.85, label=f"{name} ({lab}, n={int(s.sum())})", zorder=3)

    # ================= G16 boundary evolution
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)
    rows = []
    panels = [("Raw data (no model)", None)] + [(f"Epoch {e}" + (" (final, Adam)" if e == final_ep else ""), e) for e in eps]
    for ax, (title, e) in zip(axes.ravel(), panels):
        if e is not None:
            r = erow[e]
            P = sig(Zg @ W[r] + B[r]).reshape(GX.shape)
            cf = ax.contourf(GX, GY, P, levels=np.linspace(0, 1, 11), cmap="RdYlGn", vmin=0, vmax=1, alpha=0.55)
            cs = ax.contour(GX, GY, P, levels=[thr], colors=ERR, linewidths=2.2)
            ax.contour(GX, GY, P, levels=[0.5], colors="black", linewidths=1, linestyles="--")
            title += f"   train loss {hf.train_loss[r]:.3f} | val loss {hf.val_loss[r]:.3f}"
            rows += [{"epoch": e, "cv_h": a, "s_theta": b, "p": p} for a, b, p in zip(GX.ravel()[::97], GY.ravel()[::97], P.ravel()[::97])]
        scatter(ax)
        ax.set_title(title, fontsize=11)
    for ax in axes[1]:
        ax.set_xlabel("CV_h (raw)")
    for ax in axes[:, 0]:
        ax.set_ylabel("S_theta (deg)")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.colorbar(cf, ax=axes.ravel().tolist(), fraction=0.025, label="P(handwritten) on the slice")
    fig.suptitle(f"G16  Decision boundary evolution - fold {fold} only (train n={int(tr.sum())}, validation n={int(va.sum())}), logged Adam weights.\n"
                 f"Conditional 2-D slice: the other 5 features fixed at the training-fold median (NOT a projection of the 7-D model). "
                 f"Red = fold threshold {thr} (inner CV), dashed = 0.5. {int((~plot_ok).sum())} images with missing CV_h/S_theta not plotted.", fontsize=10)
    save(fig, "G16_epoch_boundary_evolution", pd.DataFrame(rows))

    # ================= G17 probability curve evolution
    xc = np.linspace(np.nanmin(xs), np.nanmax(xs), 300)
    gc = np.tile(med, (len(xc), 1)); gc[:, IX] = xc
    Zc = (gc - prep.mu) / prep.sigma
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)
    rows = []
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.04, 0.04, len(y))
    for ax, e in zip(axes.ravel(), [0] + eps):
        r = erow[e]
        p = sig(Zc @ W[r] + B[r])
        for mask_, mk, lab in ((tr, "o", "train"), (va, "^", "validation")):
            for c, col in ((1, HW), (0, FO)):
                s = mask_ & (y == c) & np.isfinite(xs)
                ax.scatter(xs[s], c + jit[s], c=col, marker=mk, s=22 if mk == "o" else 55, edgecolors="black" if mk == "^" else "none",
                           alpha=0.8, label=f"{'handwritten' if c else 'formal'} label {c} ({lab})")
        ax.plot(xc, p, color=ERR, lw=2.5, label="P(handwritten) = sigmoid(b + w.z)")
        ax.axhline(thr, color=MID, ls=":", lw=1.5, label=f"threshold {thr}")
        ax.set_title(f"{'Initial (epoch 0, w = 0)' if e == 0 else f'Epoch {e}'}: train loss {hf.train_loss[r]:.3f}, val loss {hf.val_loss[r]:.3f}", fontsize=11)
        ax.set_ylim(-0.1, 1.1)
        rows += [{"epoch": e, "cv_h": a, "p": b} for a, b in zip(xc, p)]
    axes[0, 0].legend(fontsize=8, loc="center right")
    for ax in axes[1]:
        ax.set_xlabel("CV_h  (raw)")
    for ax in axes[:, 0]:
        ax.set_ylabel("P(hw) / label")
    fig.suptitle(f"G17  Handwriting-probability curve along CV_h - fold {fold}, logged Adam weights. Conditional slice: the other 6 features at the "
                 "training-fold median; points are placed by their real CV_h and label (their full 7-D probability differs).", fontsize=10)
    save(fig, "G17_probability_curve_evolution", pd.DataFrame(rows))

    # ================= G18 loss + validation predictions (full 7-D)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    a1.plot(hf.epoch, hf.train_loss, color=MID, lw=2, label="training loss (logged)")
    a1.plot(hf.epoch, hf.val_loss, color=FINAL, lw=2, ls="--", label="validation loss (logged)")
    a1.set(ylabel="loss (weighted BCE + L2)", title=f"G18  Loss and validation predictions, fold {fold}")
    a1.legend()
    ids = [r["image_id"] for r, v in zip(dev, va) if v]
    yv = y[va]
    for i in range(len(yv)):
        a2.plot(hf.epoch, Pva[i], color=HW if yv[i] else FO, lw=1.2 if yv[i] else 2, ls="-" if yv[i] else "--", alpha=0.8)
    a2.axhline(thr, color=ERR, ls=":", lw=1.5)
    a2.plot([], [], color=HW, label=f"validation handwritten (n={int(yv.sum())})")
    a2.plot([], [], color=FO, ls="--", label=f"validation formal (n={int((yv == 0).sum())})")
    a2.plot([], [], color=ERR, ls=":", label=f"threshold {thr}")
    errs = ((Pva >= thr).astype(int) != yv[:, None]).sum(0)
    a2b = a2.twinx()
    a2b.step(hf.epoch, errs, color=ERR, where="post", alpha=0.6)
    a2b.set_ylabel("validation errors at threshold", color=ERR)
    for ax in (a1, a2):
        for e in [0] + eps:
            ax.axvline(e, color="black", lw=0.8, ls="-.")
    a2.set(xlabel="epoch (Adam lr 0.05; dash-dot lines = G16/G17 panels)", ylabel="P(handwritten), full 7-D model", ylim=(0, 1))
    a2.legend(fontsize=9, loc="center right")
    fig.suptitle(f"Validation probabilities recomputed with the logged weights and the fold-{fold} standardization (verified against the log, "
                 f"max |diff| {max(dev_.values()):.1e}). Validation images were never used for fitting.", fontsize=10)
    save(fig, "G18_loss_vs_prediction", pd.DataFrame({"epoch": hf.epoch, "train_loss": hf.train_loss, "val_loss": hf.val_loss,
                                                      "val_errors_at_thr": errs, **{f"p_{iid}": Pva[i] for i, iid in enumerate(ids)}}))
    g18 = {"start_err": int(errs[0]), "final_err": int(errs[-1]), "min_err": int(errs.min())}

    # ================= G19 stages S1-S4 (OOF)
    oof = pd.read_csv(ROOT / "reports" / "phase3" / "oof_predictions.csv")
    stages = [("S1_basic_unweighted", "S1 basic", BASE), ("S2_class_weighted", "S2 + class weights", MID),
              (MODEL, "S3 + inner-CV tuning", MID), ("S4_hard_example_reweighted", "S4 + reweighting", FINAL)]
    rows = []
    for m, lab, _ in stages:
        o = oof[oof.model == m]
        tp = int(((o.label == 1) & (o.pred == 1)).sum()); fn = int(((o.label == 1) & (o.pred == 0)).sum())
        tn = int(((o.label == 0) & (o.pred == 0)).sum()); fp = int(((o.label == 0) & (o.pred == 1)).sum())
        rows.append({"stage": lab, "model": m, "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2, "handwritten_recall": tp / (tp + fn),
                     "formal_recall": tn / (tn + fp), "FP_formal_kept_as_handwritten": fp, "FN_handwriting_missed": fn})
    g19 = pd.DataFrame(rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(4)
    for k, (col, name, c, hatch) in enumerate((("balanced_accuracy", "Balanced accuracy", MID, None),
                                              ("handwritten_recall", "Handwritten recall", HW, "//"),
                                              ("formal_recall", "Formal recall", BASE, "\\\\"))):
        b = a1.bar(x + (k - 1) * 0.27, g19[col], 0.27, color=c, hatch=hatch, edgecolor="black", label=name)
        for bb, v in zip(b, g19[col]):
            a1.text(bb.get_x() + bb.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8, rotation=90)
    a1.set_xticks(x, ["S1\nbasic", "S2\n+class wt", "S3\n+inner CV", "S4\n+reweight"]); a1.set_ylim(0, 1.15)
    a1.set(ylabel="score (pooled OOF, dev n=90)", title="Metrics by optimization stage")
    a1.legend(fontsize=9, loc="lower right")
    b1 = a2.bar(x - 0.2, g19.FP_formal_kept_as_handwritten, 0.4, color=BASE, edgecolor="black", label="FP: formal -> handwritten (of 15)")
    b2 = a2.bar(x + 0.2, g19.FN_handwriting_missed, 0.4, color=ERR, edgecolor="black", hatch="xx", label="FN: handwriting missed (of 75)")
    for bb, v in list(zip(b1, g19.FP_formal_kept_as_handwritten)) + list(zip(b2, g19.FN_handwriting_missed)):
        a2.text(bb.get_x() + bb.get_width() / 2, v + 0.15, str(int(v)), ha="center", fontsize=11, fontweight="bold")
    a2.set_xticks(x, ["S1", "S2", "S3", "S4"])
    a2.set(ylabel="images (count)", title="Errors: improvement and side effect")
    a2.set_ylim(0, max(g19.FN_handwriting_missed.max(), g19.FP_formal_kept_as_handwritten.max()) * 1.35)
    a2.legend(fontsize=9, loc="upper left")
    fig.suptitle(f"G19  S1 -> S4: BA {g19.balanced_accuracy[0]:.3f} -> {g19.balanced_accuracy[3]:.3f}, but missed handwriting "
                 f"{g19.FN_handwriting_missed[0]} -> {g19.FN_handwriting_missed[3]} images. Higher BA does not mean better stroke preservation.", fontsize=11)
    save(fig, "G19_optimization_stage_comparison", g19)

    # ================= G20 extraction evolution (saved inference outputs only)
    inf = ROOT / "reports" / "phase4" / "inference"
    sel = json.loads((ROOT / "models" / "phase4" / "selection.json").read_text(encoding="utf-8"))
    primary = sel["primary_model"]
    cases = [("field__weld_marking_W79", "weld_marking_W79"), ("field__예시1", "example1"), ("field__rusted_stencil_GBO", "rusted_stencil_GBO")]
    rd = lambda p, f=cv2.IMREAD_UNCHANGED: cv2.imdecode(np.fromfile(str(p), np.uint8), f)  # noqa: E731
    fig, axes = plt.subplots(3, 6, figsize=FIGSIZE)
    rows = []
    for row, (d, lab) in zip(axes, cases):
        d = inf / d
        info = json.loads((d / "run_info.json").read_text(encoding="utf-8"))
        o = rd(d / "original.png", cv2.IMREAD_COLOR)
        init = rd(d / "candidate_consensus_phase2.png", 0) > 0
        cand = rd(d / "candidate_mask.png", 0) > 0
        acc = rd(d / "accepted_mask.png", 0) > 0
        rej = rd(d / "rejected_mask.png", 0) > 0
        selv = np.zeros((*cand.shape, 3), np.uint8); selv[rej] = (214, 40, 40); selv[acc] = (44, 160, 44)
        ov = rd(d / "overlay.png", cv2.IMREAD_COLOR)
        ims = [(o[..., ::-1], "1 original"), (init, f"2 initial candidates\n(Phase 2, >=2/4 methods)\n{int(init.sum())} px"),
               (cand, f"3 corrected candidates\n(stroke-scale, final)\n{int(cand.sum())} px"),
               (selv, f"4 group selection ({primary.split('_')[0]})\nkept {int(acc.sum())} px\nremoved {int(rej.sum())} px"),
               (acc, f"5 final binary mask\n{int(acc.sum())} px"), (ov[..., ::-1], "6 final overlay")]
        for ax, (im, t) in zip(row, ims):
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None, interpolation="nearest")
            ax.set_title(t, fontsize=7.5); ax.set_xticks([]); ax.set_yticks([])
        row[0].set_ylabel(lab, fontsize=9, color=ERR if "stencil" in lab else "black")
        rows.append({"image": lab, "initial_candidate_px": int(init.sum()), "corrected_candidate_px": int(cand.sum()), "kept_px": int(acc.sum()),
                     "removed_px": int(rej.sum()), "groups": info["n_groups"], "threshold": info["modes"]["recall_priority"],
                     "intermediate_versions_v1_v3": "NOT REPRODUCED (snapshots not saved)"})
    axes[2, 2].text(0.5, -0.12, "FAILURE: stencil letters G B O are not candidates (too wide for the stroke filter);\n"
                    "only rust fragments remain and RF keeps them", transform=axes[2, 2].transAxes, ha="center", va="top", fontsize=8, color=ERR)
    fig.suptitle("G20  Extraction evolution on real field images: initial (Phase 2) vs corrected (final) candidate generation. The 3 intermediate "
                 "configurations were not snapshotted and are not reproduced. No ground truth: pixel counts are NOT accuracy.", fontsize=9.5)
    save(fig, "G20_extraction_evolution", pd.DataFrame(rows))
    g20 = pd.DataFrame(rows)

    # ================= GIF (G17 curve over logged epochs)
    from PIL import Image
    frames = []
    ep_list = sorted(set([0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 500, final_ep]) & set(hf.epoch))
    for e in ep_list:
        r = erow.get(e, int(np.nonzero(hf.epoch.to_numpy() == e)[0][0]))
        f2, ax = plt.subplots(figsize=(8, 4.5))
        for c, col in ((1, HW), (0, FO)):
            s = (y == c) & np.isfinite(xs)
            ax.scatter(xs[s], c + jit[s], c=col, s=18, alpha=0.8, label=f"label {c}")
        ax.plot(xc, sig(Zc @ W[r] + B[r]), color=ERR, lw=2.5)
        ax.axhline(thr, color=MID, ls=":")
        ax.set(xlabel="CV_h (raw)", ylabel="P(handwritten)", ylim=(-0.1, 1.1),
               title=f"fold {fold}  epoch {e}  train loss {hf.train_loss[r]:.3f}  val loss {hf.val_loss[r]:.3f}")
        f2.text(0.01, 0.01, "conditional slice: other 6 features at training-fold median; logged Adam weights", fontsize=7)
        f2.canvas.draw()
        frames.append(Image.fromarray(np.asarray(f2.canvas.buffer_rgba())[..., :3].copy()))
        plt.close(f2)
    frames[0].save(OUT / "training_evolution.gif", save_all=True, append_images=frames[1:], duration=600, loop=0)

    # ================= notes
    r_ = {e: erow[e] for e in [0] + eps}
    md = [f"# PAC2 학습·보정·추출 발전 과정 (G16–G20)", "",
          f"- 사용 로그: `reports/phase3/training_history.csv` (모델 `{MODEL}`, Adam), OOF 예측, Phase 4 추론 결과. **새 학습 없음, Test 30장 미사용.**",
          f"- fold 선택 규칙(그리기 전 고정): 조기 종료되지 않은 fold 중 가장 낮은 번호 → **fold {fold}** (H_t {H_t}, λ {lam}, 임계값 {thr}, train {int(tr.sum())} / val {int(va.sum())}).",
          f"- fold 표준화 통계(중앙값·μ·σ)는 Phase 3에서 기록되지 않아 같은 개발 특징·같은 fold로 결정론적으로 재계산. **검증**: 기록된 모든 epoch의 가중치로 "
          f"훈련 손실·검증 손실·클래스별 평균 검증 확률을 재현, 최대 오차 {max(dev_.values()):.1e} (`verification.csv`). 검증 이미지는 표준화 적합에 쓰이지 않음.",
          f"- Adam 곡선은 최적화 과정 설명용이며 OOF 성능에 쓰인 최종 모델은 같은 목적함수의 LBFGS 해다.", "",
          "## G16 Epoch별 분류 경계 (2×2)", "![G16](G16_epoch_boundary_evolution.png)", "",
          f"1. 목적: 학습이 진행되며 CV_h–S_theta 평면에서 수기/정형 경계가 어떻게 형성되는지.",
          f"2. 방법: 기록된 epoch {', '.join(map(str, eps))}의 w, b로 p = sigmoid(b + Σw_j z_j) 계산. 나머지 5개 특징은 fold 학습 중앙값으로 고정한 **조건부 단면**(7차원 투영 아님).",
          f"3. 수치: " + "; ".join(f"epoch {e}: train {hf.train_loss[r_[e]]:.3f}, val {hf.val_loss[r_[e]]:.3f}" for e in eps) + f". CV_h·S_theta 결측 {int((~plot_ok).sum())}장은 미표시.",
          "4. 해석: 초기에는 경계가 화면 밖/평탄 → epoch이 진행되며 CV_h·S_theta가 큰 쪽(불규칙)을 수기로 가르는 경계가 형성.",
          "5. 한계: 점의 위치는 실제 두 특징값이지만 그 점의 실제 7차원 확률은 단면 색과 다를 수 있음. fold 하나의 그림.", "",
          "## G17 수기 확률 곡선 변화", "![G17](G17_probability_curve_evolution.png)", "",
          "1. 목적: CV_h에 따른 P(handwritten) 곡선이 학습으로 평탄(0.5) → S자 곡선으로 바뀌는 과정.",
          "2. 방법: 실제 기록 가중치의 시그모이드, 나머지 6특징은 학습 중앙값 고정. 점은 실제 CV_h·정답 라벨(세로 jitter ±0.04만 적용).",
          "3. 수치: " + "; ".join(f"epoch {e}: train {hf.train_loss[r_[e]]:.3f}, val {hf.val_loss[r_[e]]:.3f}" for e in [0] + eps),
          "4. 해석: epoch 0은 w=0이라 모든 곳에서 0.5. 학습 후 CV_h가 큰 쪽에서 확률이 1로 수렴.",
          "5. 한계: 조건부 단면 곡선이며 다항 회귀/임의 곡선 아님. `training_evolution.gif`는 같은 곡선을 기록된 epoch 프레임으로 애니메이션.", "",
          "## G18 Loss와 검증 예측의 동시 변화", "![G18](G18_loss_vs_prediction.png)", "",
          "1. 목적: 손실 감소와 검증 이미지 예측 변화의 연결.",
          "2. 방법: 상단 = 로그의 손실. 하단 = 검증 이미지별 **7차원 전체** 예측 확률(로그 가중치 + 재구성 표준화, 로그와 일치 검증).",
          f"3. 수치: 임계값 {thr}에서 검증 오류 epoch 0 {g18['start_err']}장 → 최종 {g18['final_err']}장 (최소 {g18['min_err']}). 검증 손실 {hf.val_loss.iloc[0]:.3f} → {hf.val_loss.iloc[-1]:.3f}.",
          "4. 해석: 손실이 크게 떨어지는 초기 구간에서 대부분 검증 이미지의 확률이 정답 쪽으로 분리.",
          "5. 한계: fold 하나(val 18장). 미기록 epoch 없음(Adam은 매 epoch 기록).", "",
          "## G19 보정 단계별 성능과 부작용", "![G19](G19_optimization_stage_comparison.png)", "",
          "1. 목적: S1→S4 개선과 부작용을 함께.",
          "2. 방법: 개발 90장 pooled OOF, 각 fold 임계값 적용.",
          "3. 수치: " + "; ".join(f"{r.stage}: BA {r.balanced_accuracy:.3f}, 수기 recall {r.handwritten_recall:.3f}, 정형 recall {r.formal_recall:.3f}, FP {r.FP_formal_kept_as_handwritten}, FN {r.FN_handwriting_missed}" for r in g19.itertuples()),
          f"4. 해석: BA 상승은 정형 오류(FP) 감소에서 오며, 수기 누락(FN)은 {g19.FN_handwriting_missed[0]} → {g19.FN_handwriting_missed[3]}장으로 증가. 수기 획 보존 관점에서는 악화.",
          "5. 한계: formal 15장(1장 ≈ BA 3.3 pp).", "",
          "## G20 실제 강판 추출 과정 발전", "![G20](G20_extraction_evolution.png)", "",
          "1. 목적: 같은 원본에서 초기 후보 → 수정 후보 → 선택 → 최종 마스크.",
          "2. 방법: 저장된 추론 결과 파일만 사용(재추론 없음). 초기 = Phase 2 합의 마스크(코드 불변, 추론 시 함께 저장됨), 수정 = 최종 stroke 후보.",
          "3. 수치: " + "; ".join(f"{r.image}: 초기 {r.initial_candidate_px} px → 수정 {r.corrected_candidate_px} px → 보존 {r.kept_px} / 제거 {r.removed_px}" for r in g20.itertuples()),
          "4. 해석: 초기 후보는 녹 덩어리·조명 영역을 대량 포함, 수정 후보는 얇은 획 위주로 축소. 녹슨 스텐실은 GBO 글자가 후보에서 빠짐(스텐실은 비수기라 최종적으로 제외되어야 하지만, "
          "그것이 모델 판단이 아니라 후보 단계 누락 때문) → 남은 녹 조각은 RF가 보존 = 실패 사례.",
          "5. 한계: **중간 3개 설정(임계 규칙·면적 상한·선 분리)의 스냅샷이 저장되지 않아 재현하지 않음.** 이후부터 `reports/phase4/snapshots/<config_hash>/`에 설정과 결과 요약을 저장. "
          "정답 마스크가 없어 픽셀 수는 정확도·노이즈 제거율이 아님."]
    (OUT / "presentation_notes.md").write_text("\n".join(md), encoding="utf-8")
    print(g19.round(3).to_string(index=False)); print(g20.to_string(index=False)); print("G18", g18, "| GIF frames", len(frames))
    return 0


if __name__ == "__main__":
    sys.exit(main())
