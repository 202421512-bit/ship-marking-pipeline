"""PyTorch logistic model p = sigmoid(b + sum_j w_j z_j) with class-weighted BCE + lambda * ||w||^2 (bias not penalized).

Class weights are recomputed from each training fold: w_c = N / (2 N_c) (no double correction); class_weighted=False
gives the unweighted baseline. Optional per-sample multipliers alpha_i (hard-example reweighting) scale the weights.
Divergence guard: non-finite loss / parameters or |w| > 1e3 stop training and record a failure reason.

Logging (Adam): every epoch - total / weighted-BCE / L2 loss, lr, weights, bias, gradient norm and, if a validation
set is given, validation loss / BA / precision / recall / F1 / mean probability per class. The best epoch by validation
loss is RECORDED for monitoring only; it never selects the returned model (that would leak the validation fold).
Early stopping uses the TRAINING loss plateau only.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

torch.set_default_dtype(torch.float64)


def class_weights(y: np.ndarray) -> Dict[int, float]:
    n = len(y)
    return {c: n / (2.0 * max(1, int((y == c).sum()))) for c in (0, 1)}


def _bce(p, y, eps=1e-12):
    return -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))


class TorchLogistic:
    def __init__(self, lam: float = 1e-2, optimizer: str = "lbfgs", epochs: int = 600, lr: float = 0.05, seed: int = 42,
                 class_weighted: bool = True, patience: int = 100, min_delta: float = 1e-7, log_every: int = 1):
        self.lam, self.optimizer, self.epochs, self.lr, self.seed = lam, optimizer, epochs, lr, seed
        self.class_weighted, self.patience, self.min_delta, self.log_every = class_weighted, patience, min_delta, log_every
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.history: List[dict] = []
        self.failure: str = ""
        self.early_stop_epoch: Optional[int] = None
        self.best_val_epoch: Optional[int] = None

    def _terms(self, w, b, X, y, sw):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, y, reduction="none")
        wbce = (sw * bce).sum() / sw.sum()
        l2 = self.lam * (w ** 2).sum()
        return wbce + l2, wbce, l2

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            sample_alpha: Optional[np.ndarray] = None):
        torch.manual_seed(self.seed)
        cw = class_weights(y) if self.class_weighted else {0: 1.0, 1: 1.0}
        self.class_weight = cw
        base = np.array([cw[int(v)] for v in y]) * (sample_alpha if sample_alpha is not None else 1.0)
        Xt, yt, sw = torch.tensor(X), torch.tensor(y, dtype=torch.float64), torch.tensor(base)
        w = torch.zeros(X.shape[1], requires_grad=True)
        b = torch.zeros((), requires_grad=True)
        self.initial_w = np.zeros(X.shape[1])

        def log(step, total, wbce, l2, lr):
            gn = float(torch.cat([w.grad.flatten(), b.grad.view(1)]).norm()) if w.grad is not None else float("nan")
            rec = {"epoch": step, "train_loss": float(total), "weighted_bce": float(wbce), "l2_loss": float(l2), "lr": lr,
                   "grad_norm": gn, "bias": float(b.detach()), **{f"w{j}": float(v) for j, v in enumerate(w.detach().numpy())}}
            if X_val is not None:
                from .metrics import binary_metrics
                p = 1 / (1 + np.exp(-(X_val @ w.detach().numpy() + float(b.detach()))))
                vw = np.array([cw[int(v)] for v in y_val])
                rec["val_weighted_bce"] = float(np.sum(vw * _bce(p, y_val)) / vw.sum())
                rec["val_loss"] = rec["val_weighted_bce"] + float(l2)
                m = binary_metrics(y_val, (p >= 0.5).astype(int))
                rec.update(val_balanced_accuracy=m["balanced_accuracy"], val_precision=m["precision"], val_recall=m["recall"],
                           val_f1=m["f1"], val_mean_p_formal=float(p[y_val == 0].mean()), val_mean_p_handwritten=float(p[y_val == 1].mean()))
            self.history.append(rec)

        def broken(loss) -> str:
            if not torch.isfinite(loss):
                return "loss became NaN/Inf"
            if not torch.isfinite(w).all() or not torch.isfinite(b):
                return "parameters became NaN/Inf"
            if w.detach().abs().max() > 1e3:
                return f"weights diverged (max |w| = {float(w.detach().abs().max()):.1f})"
            return ""

        if self.optimizer == "lbfgs":
            opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=200, line_search_fn="strong_wolfe", tolerance_grad=1e-9)
            step = [0]

            def closure():
                opt.zero_grad()
                total, wbce, l2 = self._terms(w, b, Xt, yt, sw)
                total.backward()
                step[0] += 1
                if step[0] % 5 == 1:
                    log(step[0], total.detach(), wbce.detach(), l2.detach(), 1.0)
                return total

            opt.step(closure)
            total, wbce, l2 = self._terms(w, b, Xt, yt, sw)
            total.backward()
            log(step[0], total.detach(), wbce.detach(), l2.detach(), 1.0)
            self.failure = broken(total.detach())
        else:
            opt = torch.optim.Adam([w, b], lr=self.lr)
            best_train, stall = float("inf"), 0
            for ep in range(self.epochs):
                opt.zero_grad()
                total, wbce, l2 = self._terms(w, b, Xt, yt, sw)
                total.backward()
                why = broken(total.detach())
                if why:
                    self.failure = f"epoch {ep}: {why}"
                    break
                if ep % self.log_every == 0 or ep == self.epochs - 1:
                    log(ep, total.detach(), wbce.detach(), l2.detach(), opt.param_groups[0]["lr"])
                opt.step()
                cur = float(total.detach())
                if best_train - cur > self.min_delta:
                    best_train, stall = cur, 0
                else:
                    stall += 1
                    if stall >= self.patience:
                        self.early_stop_epoch = ep
                        break
            if X_val is not None and self.history:
                self.best_val_epoch = int(min(self.history, key=lambda r: r["val_loss"])["epoch"])
        self.w, self.b = w.detach().numpy().copy(), float(b.detach())
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(X @ self.w + self.b)))


def fit_hard_example_reweighted(X, y, lam, rounds: int = 10, warmup: int = 1, eta: float = 1.0, ema: float = 0.5,
                                clip=(0.5, 3.0), X_val=None, y_val=None, seed: int = 42):
    """AdaBoost-INSPIRED hard-example reweighting (NOT the AdaBoost algorithm): repeated LBFGS fits where each training
    sample's multiplier alpha_i grows with its EMA-smoothed relative loss, alpha = clip(exp(eta*(e_i - 1)), 0.5, 3)
    normalized to mean 1. Round 1..warmup uses alpha = 1. Validation data are only logged, never reweighted."""
    alpha = np.ones(len(y))
    e = np.ones(len(y))
    hist, prev_val_p, model = [], None, None
    for r in range(1, rounds + 1):
        model = TorchLogistic(lam, "lbfgs", seed=seed).fit(X, y, sample_alpha=alpha)
        p = model.predict_proba(X)
        loss_i = _bce(p, y)
        rec = {"round": r, "alpha_min": alpha.min(), "alpha_p25": np.percentile(alpha, 25), "alpha_median": np.median(alpha),
               "alpha_p75": np.percentile(alpha, 75), "alpha_max": alpha.max(),
               "alpha_mean_formal": alpha[y == 0].mean(), "alpha_mean_handwritten": alpha[y == 1].mean(),
               "train_misclassified": int(((p >= 0.5).astype(int) != y).sum()), "train_mean_bce": float(loss_i.mean()),
               "train_recall_formal": float(((p < 0.5) & (y == 0)).sum() / max(1, (y == 0).sum())),
               "train_recall_handwritten": float(((p >= 0.5) & (y == 1)).sum() / max(1, (y == 1).sum())),
               "failure": model.failure}
        if X_val is not None:
            pv = model.predict_proba(X_val)
            rec.update(val_mean_p_formal=float(pv[y_val == 0].mean()), val_mean_p_handwritten=float(pv[y_val == 1].mean()),
                       val_mean_abs_dp_vs_prev=float(np.abs(pv - prev_val_p).mean()) if prev_val_p is not None else 0.0)
            prev_val_p = pv
        hist.append(rec)
        if r >= warmup:
            e = ema * e + (1 - ema) * (loss_i / max(loss_i.mean(), 1e-12))
            alpha = np.clip(np.exp(eta * (e - 1.0)), *clip)
            alpha = np.clip(alpha / alpha.mean(), *clip)
    return model, hist
