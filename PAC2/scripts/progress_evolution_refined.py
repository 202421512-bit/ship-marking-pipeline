"""H1-H4, H6 presentation figures (new set; G1-G20 untouched). Reads logs only; no training; Test 30 never read.

H1  PCA plane (training fold only) with the logged 7-D logistic evaluated EXACTLY on inverse-mapped plane points
    (z = mean + u*pc1 + v*pc2, orthogonal residual held at the training mean). Fold standardization is reconstructed and
    re-verified against the log exactly as for G16-G18.
H2  probability curves for CV_h, S_theta, R (top |weight| features of the S3 model), other features at training median.
H3  loss decrease (5-fold mean, logged) + BA S1->S4 + missed handwriting S1->S4 (pooled OOF) in one figure.
H4  domain gap: thumbnails + distributions of non-shape descriptors (dev 90 + field 7; test excluded).
H6  failure cases (refined Phase 4 outputs) with crops and short explanations.
Run: .venv\\Scripts\\python.exe scripts\\progress_evolution_refined.py
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
from pac2.domain import domain_features  # noqa: E402
from pac2.features import FEATURES, extract  # noqa: E402
from pac2.logistic import class_weights  # noqa: E402
from pac2.normalize import normalize_mask  # noqa: E402
from pac2.prep import FoldPrep  # noqa: E402
from pac2.visualization import BASE, ERR, FIGSIZE, FINAL, MID, plt  # noqa: E402

OUT = ROOT / "results" / "progress_evolution_refined"
MODEL = "C_shape_domain_matched|train_all|torch_logistic"
HW, FO = FINAL, "#555555"
sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731


def save(fig, name, data):
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    data.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    plt.close(fig)


def wbce(p, y, cw):
    w = np.array([cw[int(v)] for v in y])
    return float(np.sum(w * -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))) / w.sum())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    h = pd.read_csv(ROOT / "reports" / "phase3" / "training_history.csv")
    h = h[(h.model == MODEL) & (h.optimizer == "adam")]
    fold = int(min(f for f, s in h.groupby("fold").early_stop_epoch.first().items() if not np.isfinite(s)))
    hf = h[h.fold == fold].sort_values("epoch").reset_index(drop=True)
    st = json.loads(hf.settings.iloc[0])
    H_t, lam, thr = int(st["H_t"]), float(st["lambda"]), float(st["threshold"])
    with open(ROOT / "data" / "splits" / "split_manifest.csv", encoding="utf-8-sig") as f:
        man = list(csv.DictReader(f))
    dev = sorted([r for r in man if r["set"] == "dev"], key=lambda r: r["image_id"])
    y = np.array([int(r["label"]) for r in dev]); folds = np.array([int(r["fold"]) for r in dev])
    ccfg = dict(cfg["candidates"], methods=["gray_otsu"])
    imgs = {r["image_id"]: read_bgr(ROOT / r["path"]) for r in dev}
    X = np.array([[extract(normalize_mask(candidates(imgs[r["image_id"]], ccfg)["gray_otsu"], H_t), cfg["features"])[k] for k in FEATURES] for r in dev], float)
    tr, va = folds != fold, folds == fold
    prep = FoldPrep("train_all").fit(X[tr], y[tr])
    Z = prep.transform(X)
    W = hf[[f"w{j}" for j in range(7)]].to_numpy(float); B = hf.bias.to_numpy(float)
    cw = class_weights(y[tr])
    dev_tr = max(abs(wbce(sig(Z[tr] @ W[e] + B[e]), y[tr], cw) + lam * (W[e] ** 2).sum() - hf.train_loss[e]) for e in range(len(hf)))
    dev_va = max(abs(wbce(sig(Z[va] @ W[e] + B[e]), y[va], cw) - hf.val_weighted_bce[e]) for e in range(len(hf)))
    if max(dev_tr, dev_va) > 1e-6:
        print("standardization reconstruction failed - H1/H2 not drawn", dev_tr, dev_va)
        return 1
    final_ep = int(hf.epoch.max())
    eps = [0, 10, 100, final_ep]
    row = {e: int(np.nonzero(hf.epoch.to_numpy() == e)[0][0]) for e in eps}
    verify = {"fold": fold, "train_loss_max_diff": dev_tr, "val_bce_max_diff": dev_va}

    # ================= H1 PCA plane
    mu = Z[tr].mean(0)
    U, S, Vt = np.linalg.svd(Z[tr] - mu, full_matrices=False)
    pcs = Vt[:2]
    evr = (S[:2] ** 2) / (S ** 2).sum()
    P2 = (Z - mu) @ pcs.T
    pad = 0.6
    ug = np.linspace(P2[:, 0].min() - pad, P2[:, 0].max() + pad, 250); vg = np.linspace(P2[:, 1].min() - pad, P2[:, 1].max() + pad, 250)
    GU, GV = np.meshgrid(ug, vg)
    Zg = mu + GU.reshape(-1, 1) * pcs[0] + GV.reshape(-1, 1) * pcs[1]
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True)
    rows = []
    for ax, e in zip(axes.ravel(), eps):
        r = row[e]
        Pg = sig(Zg @ W[r] + B[r]).reshape(GU.shape)
        cf = ax.contourf(GU, GV, Pg, levels=np.linspace(0, 1, 21), cmap="RdYlGn", vmin=0, vmax=1, alpha=0.6)
        if Pg.min() < thr < Pg.max():
            ax.contour(GU, GV, Pg, levels=[thr], colors=ERR, linewidths=2.5)
        p_true = sig(Z @ W[r] + B[r])
        wrong = (p_true >= thr).astype(int) != y
        for m_, mk, lab in ((tr, "o", "train"), (va, "^", "validation")):
            for c, col in ((1, HW), (0, FO)):
                s = m_ & (y == c)
                ax.scatter(P2[s, 0], P2[s, 1], c=col, marker=mk, s=30 if mk == "o" else 70, edgecolors="black" if mk == "^" else "white",
                           linewidths=0.6, label=f"{'handwritten' if c else 'formal'} ({lab})", zorder=3)
        ax.scatter(P2[wrong, 0], P2[wrong, 1], facecolors="none", edgecolors=ERR, s=160, linewidths=1.8, label="misclassified (full 7-D model)", zorder=4)
        ax.set_title(f"{'Epoch 0 (w = 0, p = 0.5 everywhere)' if e == 0 else f'Epoch {e}'}   train loss {hf.train_loss[r]:.3f} | val loss {hf.val_loss[r]:.3f}"
                     f" | errors {int(wrong.sum())}/90", fontsize=10.5)
        rows.append({"epoch": e, "train_loss": hf.train_loss[r], "val_loss": hf.val_loss[r], "errors_full_model_all90": int(wrong.sum()),
                     "errors_validation": int(wrong[va].sum())})
    ylo, yhi = np.percentile(P2[:, 1], [0, 100])
    out_hi = P2[:, 1] > np.percentile(P2[:, 1], 99)
    ytop = float(np.max(P2[~out_hi, 1])) + 1.0
    for ax in axes.ravel():
        ax.set_ylim(ug.min() * 0 + vg.min(), ytop)
        for i_ in np.nonzero(out_hi)[0]:
            ax.annotate(f"outside view: PC2 = {P2[i_, 1]:.1f}", (P2[i_, 0], ytop), xytext=(0, -12), textcoords="offset points", fontsize=7, ha="center", color="black")
    axes[0, 0].legend(fontsize=7, loc="lower right", framealpha=0.9)
    for ax in axes[1]:
        ax.set_xlabel(f"PC1 of the 7 standardized features ({evr[0]:.0%} of variance)")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"PC2 ({evr[1]:.0%})")
    fig.colorbar(cf, ax=axes.ravel().tolist(), fraction=0.025, label="P(handwritten) of the 7-D model on the PCA plane")
    fig.suptitle(f"H1  The model learns the separation - fold {fold} (train 72 / validation 18), logged Adam weights. 2-D view of a 7-D model: background = exact "
                 f"model output on the PCA plane\n(residual off the plane held at the training mean); points are projections, red circles = errors of the FULL model. "
                 f"Red line = threshold {thr}. PCA fitted on the training fold only.", fontsize=9.5)
    save(fig, "H1_feature_space_evolution", pd.DataFrame(rows))

    # ================= H2 top-feature probability curves
    feats = ["CV_h", "S_theta", "R"]
    med = prep.median
    fig, axes = plt.subplots(3, 4, figsize=FIGSIZE, sharey=True)
    rows = []
    jit = np.random.default_rng(0).uniform(-0.05, 0.05, len(y))
    for i, fname in enumerate(feats):
        j = FEATURES.index(fname)
        xs = X[:, j]
        ok = np.isfinite(xs)
        xc = np.linspace(np.nanpercentile(xs, 1), np.nanpercentile(xs, 99), 250)
        g = np.tile(med, (len(xc), 1)); g[:, j] = xc
        Zc = (g - prep.mu) / prep.sigma
        for k, e in enumerate(eps):
            ax = axes[i, k]
            r = row[e]
            p = sig(Zc @ W[r] + B[r])
            for c, col in ((1, HW), (0, FO)):
                s = ok & (y == c)
                ax.scatter(xs[s], c + jit[s], s=10, c=col, alpha=0.55)
            ax.plot(xc, p, color=ERR, lw=2.4)
            ax.axhline(thr, color=MID, ls=":", lw=1)
            ax.set_ylim(-0.12, 1.12)
            if i == 0:
                ax.set_title(f"epoch {e}  (train loss {hf.train_loss[r]:.3f})", fontsize=10)
            if k == 0:
                ax.set_ylabel(f"{fname}\nP(hw) / label", fontsize=10)
            if i == 2:
                ax.set_xlabel("feature value (raw)", fontsize=9)
            ax.tick_params(labelsize=8)
            rows += [{"feature": fname, "epoch": e, "x": a, "p": b} for a, b in zip(xc[::5], p[::5])]
    fig.suptitle(f"H2  Probability curves for three presentation features CV_h, S_theta, R (S3 model, fold {fold}, logged weights; CV_w has a larger weight but is not shown). Each row varies one feature; "
                 "the other six are fixed at the training-fold median (conditional slice).\nGreen = handwritten (label 1), grey = formal (label 0), jitter ±0.05; "
                 f"blue dotted = threshold {thr}. Curves are the model's sigmoid, not fitted regressions.", fontsize=9.5)
    save(fig, "H2_top_feature_probability_evolution", pd.DataFrame(rows))

    # ================= H3 optimization progress summary
    g1 = pd.read_csv(ROOT / "results" / "phase3" / "tables" / "G1_loss_convergence.csv")
    oof = pd.read_csv(ROOT / "reports" / "phase3" / "oof_predictions.csv")
    stg = [("S1_basic_unweighted", "S1"), ("S2_class_weighted", "S2"), (MODEL, "S3"), ("S4_hard_example_reweighted", "S4")]
    srow = []
    for m, lab in stg:
        o = oof[oof.model == m]
        tp = int(((o.label == 1) & (o.pred == 1)).sum()); fn = int(((o.label == 1) & (o.pred == 0)).sum())
        tn = int(((o.label == 0) & (o.pred == 0)).sum()); fp = int(((o.label == 0) & (o.pred == 1)).sum())
        srow.append({"stage": lab, "BA": (tp / (tp + fn) + tn / (tn + fp)) / 2, "FN_missed_handwriting": fn, "FP_formal_kept": fp})
    sd = pd.DataFrame(srow)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=FIGSIZE)
    a1.plot(g1.epoch, g1.train_loss_mean, color=MID, lw=2.5, label="training")
    a1.plot(g1.epoch, g1.val_loss_mean, color=FINAL, lw=2.5, ls="--", label="validation")
    a1.set(xlabel="epoch", ylabel="loss (5-fold mean)", title=f"1  Loss {g1.train_loss_mean.iloc[0]:.2f} -> {g1.train_loss_mean.iloc[-1]:.2f} (train)")
    a1.legend()
    a2.plot(sd.stage, sd.BA, color=FINAL, marker="o", ms=10, lw=3)
    for x_, v in zip(sd.stage, sd.BA):
        a2.annotate(f"{v:.3f}", (x_, v), xytext=(0, -22 if v < 0.85 else 10), textcoords="offset points", ha="center", fontsize=12, fontweight="bold")
    a2.set(ylim=(0.5, 1.05), ylabel="balanced accuracy (pooled OOF)", title="2  Discrimination improves")
    b = a3.bar(sd.stage, sd.FN_missed_handwriting, color=ERR, edgecolor="black", hatch="xx", label="handwriting removed (FN, of 75)")
    a3.plot(sd.stage, sd.FP_formal_kept, color=BASE, marker="s", ms=9, lw=2, label="formal kept (FP, of 15)")
    for bb, v in zip(b, sd.FN_missed_handwriting):
        a3.text(bb.get_x() + bb.get_width() / 2, v + 0.2, str(v), ha="center", fontsize=13, fontweight="bold", color=ERR)
    a3.set(ylabel="images (count)", ylim=(0, sd.FN_missed_handwriting.max() + 3), title="3  ...but more handwriting is removed")
    a3.legend(fontsize=9)
    fig.suptitle(f"H3  Optimization improved overall discrimination (BA {sd.BA[0]:.3f} -> {sd.BA[3]:.3f}) at the cost of more handwriting removed "
                 f"(FN {sd.FN_missed_handwriting[0]} -> {sd.FN_missed_handwriting[3]}).  Development data, 5-fold out-of-fold, n = 90.", fontsize=11)
    save(fig, "H3_optimization_progress_summary", sd)

    # ================= H4 domain gap
    field = sorted((ROOT / "data" / "field_test" / "images").glob("*.png"))
    drows = []
    for r in dev:
        drows.append({"group": r["class"], "image": r["image_id"], **domain_features(imgs[r["image_id"]])})
    for p in field:
        drows.append({"group": "field (steel)", "image": p.name, **domain_features(read_bgr(p))})
    dd = pd.DataFrame(drows)
    dd["log10_height"] = np.log10(dd["height"])
    groups = [("formal", BASE), ("handwritten", HW), ("field (steel)", "#b5651d")]
    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, 9, height_ratios=[1, 1.15])
    thumbs = {"formal": [r["image_id"] for r in dev if r["class"] == "formal"][:3],
              "handwritten": [r["image_id"] for r in dev if r["class"] == "handwritten"][::25][:3], "field (steel)": [p.name for p in field][2:5]}
    k = 0
    for gname, col in groups:
        for iid in thumbs[gname]:
            ax = fig.add_subplot(gs[0, k]); k += 1
            im = imgs[iid] if gname != "field (steel)" else read_bgr(ROOT / "data" / "field_test" / "images" / iid)
            ax.imshow(im[..., ::-1]); ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_color(col); s_.set_linewidth(3)
            ax.set_title(gname, fontsize=9, color=col if gname != "formal" else "black")
    metrics = [("saturation_mean", "mean saturation (0-255)"), ("log_sharpness", "log(1 + Laplacian variance)"), ("gray_mean", "mean intensity (0-255)"),
               ("log10_height", "log10 image height (px)")]
    for i, (m, lab) in enumerate(metrics):
        ax = fig.add_subplot(gs[1, i * 2 + (1 if i else 0): i * 2 + 2 + (1 if i else 0)] if False else gs[1, [0, 2, 4, 6][i]:[0, 2, 4, 6][i] + 2])
        for j, (gname, col) in enumerate(groups):
            v = dd.loc[dd.group == gname, m].to_numpy(float)
            ax.boxplot(v, positions=[j], widths=0.5, patch_artist=True, boxprops=dict(facecolor=col, alpha=0.5), medianprops=dict(color="black"), showfliers=False)
            ax.scatter(np.full(len(v), j) + np.random.default_rng(j).uniform(-0.15, 0.15, len(v)), v, s=8, color=col, zorder=3)
        ax.set_xticks(range(3), ["formal\n(n=15)", "handwr.\n(n=75)", f"field\n(n={len(field)})"], fontsize=8)
        ax.set_title(lab, fontsize=9)
    fig.suptitle("H4  Why the domain gap matters: training classes differ in acquisition (rendered font vs photographed marker on paper-like crops), and real steel "
                 "photos differ again\n(development 90 + field 7 images; test 30 excluded). A model trained on the two left groups is applied to the right group.", fontsize=10)
    save(fig, "H4_domain_gap", dd.groupby("group")[[m for m, _ in metrics]].median().reset_index())

    # ================= H6 failure cases (refined outputs)
    RAW = ROOT / "reports" / "phase4_refined" / "inference"
    rd = lambda n, f, fl=cv2.IMREAD_UNCHANGED: cv2.imdecode(np.fromfile(str(RAW / n / f), np.uint8), fl)  # noqa: E731
    cases = [("field__예시1", None, "Too broad", "glow next to the hole merged with the strokes; refinement removes most of it (red) but part of the fan remains"),
             ("field__2", (40, 100, 240, 240), "Chalk missed", "chalk text 'Angle face ...' is fainter than the stroke threshold: few candidates, so it cannot be kept"),
             ("field__2", (0, 0, 343, 110), "Non-hw kept", "machine structure / edges are thin and bright: accepted by RF and kept by the stroke refinement"),
             ("dev__handwritten_013_V8", None, "Short neat mark", "V8: RF keeps the group (S1/S3/S4 would reject it); refinement also removes the thick marker junction/end (red) - readability loss")]
    fig, axes = plt.subplots(4, 4, figsize=FIGSIZE, gridspec_kw={"width_ratios": [1, 1, 1, 1.5]})
    rows = []
    for row_, (n, box, title, why) in zip(axes, cases):
        o, c, f_ = rd(n, "original.png", cv2.IMREAD_COLOR), rd(n, "candidate_mask.png", 0), rd(n, "overlay.png", cv2.IMREAD_COLOR)
        if box:
            x0, y0, x1, y1 = box
            o, c, f_ = o[y0:y1, x0:x1], c[y0:y1, x0:x1], f_[y0:y1, x0:x1]
        for ax, im, t in zip(row_[:3], (o[..., ::-1], c, f_[..., ::-1]), ("original crop", "candidate mask", "refined result")):
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(t, fontsize=8)
        row_[0].set_ylabel(title, fontsize=9, color=ERR)
        row_[3].text(0, 0.5, why, fontsize=9, va="center", wrap=True); row_[3].axis("off")
        rows.append({"case": title, "image": n, "crop": box, "explanation": why})
    fig.suptitle("H6  Remaining failure cases after stroke-only refinement (green = kept, red = removed by refinement or rejected).", fontsize=11)
    save(fig, "H6_failure_cases", pd.DataFrame(rows))
    (OUT / "verification.json").write_text(json.dumps({**verify, "pca_explained_variance": evr.tolist(), "test_images_used": 0}, indent=1), encoding="utf-8")
    print("verify", verify, "| PCA var", np.round(evr, 3))
    print(pd.read_csv(OUT / "H1_feature_space_evolution.csv").to_string(index=False))
    print(sd.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
