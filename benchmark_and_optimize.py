"""
통합 벤치마크 + 데이터 기반 가중치 산출.

  1. 합성 데이터셋 (인쇄체 n, 수기 n) 생성 -> 8개 특징 일괄 추출 (features.csv 로 캐시)
  2. 지표별 분포 통계(평균·표준편차·중앙값·결측률), ROC-AUC, Cohen's d, 인쇄체 종류별 AUC
  3. 지표 간 상관 행렬 (Pearson, Spearman) + 히트맵
  4. 학습/평가 분할 (층화 70/30). 학습 세트에서 L2 로지스틱 회귀(표준화 특징, λ 는 5-겹 교차검증)
     -> 가중치: 원래 φ 척도 계수 c_i/σ_i 의 양수 부분을 합 1 로 정규화
        (요청 방식 |c_i| 정규화도 비교용으로 보고. 음의 계수에 절댓값을 쓰면 '수기일수록 낮은' 지표를
         수기 쪽으로 더하게 되어 점수가 거꾸로 움직인다)
  5. 학습 세트에서 판정 임계값(확정 구간의 오판율 ≤ 3%) -> 평가 세트에서 AUC·확정률·확정 정확도
  6. objective_weights.json 저장 (objective_function.py 가 기본으로 읽음), benchmark_results/ 에 리포트

로지스틱 회귀는 NumPy 로 구현 (이 PC 에서는 scipy.optimize 가 정책으로 막혀 있고 scikit-learn 은 없음).
합성 데이터는 특징을 만들 때 쓴 생성기와 같은 계열이라 결과가 낙관적이다. 현장 라벨 사진으로 다시 돌릴 것:
  python benchmark_and_optimize.py --features 현장_features.csv
"""
import argparse
import csv
import json
import math
import os
import time
from datetime import date

import cv2
import numpy as np

from objective_function import FEATURE_NAMES, FEATURES, WEIGHTS_PATH, combine, extract_features
from synthetic_dataset import make_dataset

OUT_DIR = "benchmark_results"


# ---------------------------------------------------------------------------
# 통계 도구
# ---------------------------------------------------------------------------
def roc_auc(neg, pos):
    """순위 기반 AUC = P(수기 점수 > 인쇄체 점수) (동점 0.5)."""
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    allv = np.concatenate([neg, pos])
    ranks = np.empty(len(allv))
    order = np.argsort(allv, kind="mergesort")
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):                       # 동점은 평균 순위
        idx = allv == v
        ranks[idx] = ranks[idx].mean()
    return float((ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def cohens_d(neg, pos):
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    if len(neg) < 2 or len(pos) < 2:
        return float("nan")
    sp = math.sqrt(((len(neg) - 1) * neg.var(ddof=1) + (len(pos) - 1) * pos.var(ddof=1)) / (len(neg) + len(pos) - 2))
    return float((pos.mean() - neg.mean()) / sp) if sp > 0 else float("nan")


def _rank(v):
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    r[order] = np.arange(len(v))
    return r


def correlation(X, method="pearson"):
    """결측(NaN) 은 쌍별 제외."""
    d = X.shape[1]
    out = np.full((d, d), np.nan)
    for i in range(d):
        for j in range(d):
            ok = np.isfinite(X[:, i]) & np.isfinite(X[:, j])
            if ok.sum() < 3:
                continue
            a, b = X[ok, i], X[ok, j]
            if method == "spearman":
                a, b = _rank(a), _rank(b)
            if a.std() > 0 and b.std() > 0:
                out[i, j] = float(np.corrcoef(a, b)[0, 1])
    return out


# ---------------------------------------------------------------------------
# L2 로지스틱 회귀 (Newton/IRLS)
# ---------------------------------------------------------------------------
def fit_logistic(X, y, lam=1.0, iters=100):
    """min Σ logloss + (λ/2)||w||² (절편 제외). (w, b) 반환."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    theta = np.zeros(d + 1)
    reg = np.r_[np.full(d, lam), 0.0]
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ theta, -30, 30)))
        g = Xb.T @ (p - y) + reg * theta
        H = Xb.T @ (Xb * (p * (1 - p))[:, None]) + np.diag(reg + 1e-9)
        step = np.linalg.solve(H, g)
        theta -= step
        if np.abs(step).max() < 1e-9:
            break
    return theta[:-1], float(theta[-1])


def stratified_folds(y, k, rng):
    folds = [[] for _ in range(k)]
    for cls in (0, 1):
        idx = rng.permutation(np.flatnonzero(y == cls))
        for i, j in enumerate(idx):
            folds[i % k].append(j)
    return [np.array(sorted(f)) for f in folds]


class Preprocessor:
    """결측은 학습 세트 중앙값으로 채우고 표준화."""

    def fit(self, X):
        self.median = np.nanmedian(X, axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.5)
        Xf = self.fill(X)
        self.mean, self.std = Xf.mean(axis=0), Xf.std(axis=0)
        self.std = np.where(self.std > 1e-9, self.std, 1.0)
        return self

    def fill(self, X):
        return np.where(np.isfinite(X), X, self.median)

    def transform(self, X):
        return (self.fill(X) - self.mean) / self.std


def choose_lambda(X, y, grid=(0.01, 0.1, 1.0, 10.0), k=5, seed=0):
    rng = np.random.default_rng(seed)
    folds = stratified_folds(y, k, rng)
    scores = {}
    for lam in grid:
        aucs = []
        for f in folds:
            tr = np.setdiff1d(np.arange(len(y)), f)
            pp = Preprocessor().fit(X[tr])
            w, b = fit_logistic(pp.transform(X[tr]), y[tr], lam)
            z = pp.transform(X[f]) @ w + b
            aucs.append(roc_auc(z[y[f] == 0], z[y[f] == 1]))
        scores[lam] = float(np.mean(aucs))
    return max(scores, key=scores.get), scores


def weights_from_coefficients(w_std, std, mode="positive"):
    """표준화 계수 -> 원래 φ 척도 가중치 (합 1).
    mode="positive": max(c_i/σ_i, 0)   (권장)
    mode="abs"     : |c_i|              (요청 방식, 비교용)"""
    raw = np.maximum(w_std / std, 0) if mode == "positive" else np.abs(w_std)
    if raw.sum() <= 0:
        raw = np.ones_like(raw)
    return raw / raw.sum()


def score_matrix(X, weights):
    """동적 재분배 가중 합 (objective_function.combine 과 같음)."""
    wd = dict(zip(FEATURES, weights))
    out = []
    for row in X:
        phis = {k: (None if not np.isfinite(v) else float(v)) for k, v in zip(FEATURES, row)}
        s, _, _ = combine(phis, wd)
        out.append(np.nan if s is None else s)
    return np.array(out)


def choose_thresholds(s, y, max_error=0.03):
    """확정 구간 오판율 ≤ max_error 가 되는 가장 넓은 확정 구간.
    인쇄체 확정: s ≤ t_p 중 수기 비율 ≤ max_error 인 가장 큰 t_p / 수기 확정: s ≥ t_h 중 인쇄체 비율 ≤ ...."""
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    cand = np.unique(s)
    t_p = float(cand.min()) - 1e-9
    for t in cand:
        sel = s <= t
        if sel.any() and y[sel].mean() <= max_error:
            t_p = float(t)
    t_h = float(cand.max()) + 1e-9
    for t in cand[::-1]:
        sel = s >= t
        if sel.any() and (1 - y[sel]).mean() <= max_error:
            t_h = float(t)
    if t_p >= t_h:                          # 겹치면 가운데로
        t_p = t_h = (t_p + t_h) / 2
    return {"printed": t_p, "handwritten": t_h}


def decision_metrics(s, y, th):
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    hand = s >= th["handwritten"]
    prin = s <= th["printed"]
    decided = hand | prin
    correct = (hand & (y == 1)) | (prin & (y == 0))
    return {"decided_rate": float(decided.mean()),
            "accuracy_on_decided": float(correct[decided].mean()) if decided.any() else float("nan"),
            "review_rate": float(1 - decided.mean())}


# ---------------------------------------------------------------------------
# 데이터 / 캐시
# ---------------------------------------------------------------------------
def extract_all(n_per_class, seed, cache):
    if cache and os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"특징 캐시 사용: {cache} ({len(rows)}개)")
    else:
        data = make_dataset(n_per_class, seed)
        rows, t0 = [], time.perf_counter()
        for i, (img, meta) in enumerate(data):
            phis, _ = extract_features(img)
            rows.append({"label": meta["label"], "kind": meta["kind"] + ("_rot" if "angle" in meta else ""),
                         "text": meta["text"], **{k: ("" if v is None else v) for k, v in phis.items()}})
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(data)} 추출 ({(time.perf_counter() - t0) / (i + 1) * 1000:.0f} ms/장)")
        if cache:
            os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
            with open(cache, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    X = np.array([[float(r[k]) if r[k] not in ("", None) else np.nan for k in FEATURES] for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    kinds = np.array([r["kind"] for r in rows])
    return X, y, kinds


# ---------------------------------------------------------------------------
# 그림 (OpenCV 로 그림: matplotlib 의존성 없이)
# ---------------------------------------------------------------------------
def draw_heatmap(C, labels, title):
    d = len(labels)
    cell, left, top = 62, 130, 60
    img = np.full((top + d * cell + 20, left + d * cell + 20, 3), 255, np.uint8)
    cv2.putText(img, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)
    for i in range(d):
        cv2.putText(img, labels[i], (6, top + i * cell + cell // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1,
                    cv2.LINE_AA)
        cv2.putText(img, labels[i][:4], (left + i * cell + 12, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1,
                    cv2.LINE_AA)
        for j in range(d):
            v = C[i, j]
            if np.isfinite(v):   # 파랑(-1) - 흰색(0) - 빨강(+1)
                color = (255, int(255 * (1 + v)), int(255 * (1 + v))) if v < 0 else (int(255 * (1 - v)), int(255 * (1 - v)), 255)
            else:
                color = (230, 230, 230)
            x0, y0 = left + j * cell, top + i * cell
            cv2.rectangle(img, (x0, y0), (x0 + cell - 2, y0 + cell - 2), color, -1)
            if np.isfinite(v):
                cv2.putText(img, f"{v:+.2f}", (x0 + 6, y0 + cell // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (0, 0, 0), 1, cv2.LINE_AA)
    return img


def draw_distributions(X, y, aucs):
    rows = []
    for i, k in enumerate(FEATURES):
        panel = np.full((70, 620, 3), 255, np.uint8)
        cv2.putText(panel, f"{k} {FEATURE_NAMES[k]}  AUC {aucs[k]:.3f}", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (40, 40, 40), 1, cv2.LINE_AA)
        bins = np.linspace(0, 1, 41)
        for cls, color, sign in ((0, (60, 160, 60), -1), (1, (60, 60, 220), 1)):
            v = X[y == cls, i]
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            hist, _ = np.histogram(np.clip(v, 0, 1), bins)
            hist = hist / max(hist.max(), 1)
            for b, hv in enumerate(hist):
                x0 = 10 + b * 15
                if sign < 0:
                    cv2.rectangle(panel, (x0, 45), (x0 + 13, 45 - int(hv * 22)), color, -1)
                else:
                    cv2.rectangle(panel, (x0, 47), (x0 + 13, 47 + int(hv * 22)), color, -1)
        cv2.line(panel, (10, 46), (610, 46), (120, 120, 120), 1)
        rows.append(panel)
    legend = np.full((26, 620, 3), 255, np.uint8)
    cv2.putText(legend, "phi 0 -> 1 (left to right).  green (up) = printed,  red (down) = handwritten", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
    return np.vstack([legend] + rows)


def _save(path, image):
    ok, buf = cv2.imencode(".png", image)
    buf.tofile(path)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="φ₁~φ₈ 벤치마크 + 최적 가중치")
    parser.add_argument("--n", type=int, default=100, help="클래스당 장수")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--features", default=os.path.join(OUT_DIR, "features.csv"),
                        help="특징 CSV 캐시 (있으면 재사용, 현장 데이터 CSV 를 넣어도 됨)")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-save", action="store_true", help="objective_weights.json 을 쓰지 않음")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    X, y, kinds = extract_all(args.n, args.seed, None if args.no_cache else args.features)
    lines = [f"# 벤치마크 리포트 ({date.today()})", "",
             f"표본: 인쇄체 {int((y == 0).sum())}, 수기 {int((y == 1).sum())} (합성, seed={args.seed})", ""]

    # 1) 분포 통계 · AUC · Cohen's d
    lines += ["## 1. 지표별 분포와 분리도 (전체 표본)", "",
              "| 지표 | 인쇄체 평균±표준편차 (중앙값) | 수기 평균±표준편차 (중앙값) | 결측률 | AUC | Cohen's d |",
              "|---|---|---|---|---|---|"]
    aucs = {}
    print(f"\n{'지표':14s} {'인쇄체 mean±sd (med)':>24s} {'수기 mean±sd (med)':>24s} {'결측':>5s} {'AUC':>6s} {'d':>6s}")
    for i, k in enumerate(FEATURES):
        p, h = X[y == 0, i], X[y == 1, i]
        miss = float(np.mean(~np.isfinite(X[:, i])))
        p, h = p[np.isfinite(p)], h[np.isfinite(h)]
        aucs[k], d = roc_auc(p, h), cohens_d(p, h)
        ps = f"{p.mean():.3f}±{p.std():.3f} ({np.median(p):.3f})" if len(p) else "-"
        hs = f"{h.mean():.3f}±{h.std():.3f} ({np.median(h):.3f})" if len(h) else "-"
        print(f"{k} {FEATURE_NAMES[k]:9s} {ps:>24s} {hs:>24s} {miss:5.0%} {aucs[k]:6.3f} {d:6.2f}")
        lines.append(f"| {k} {FEATURE_NAMES[k]} | {ps} | {hs} | {miss:.0%} | {aucs[k]:.3f} | {d:.2f} |")

    kinds_p = sorted(set(kinds[y == 0]))
    lines += ["", "### 인쇄체 종류별 AUC (해당 인쇄체 vs 전체 수기)", "",
              "| 지표 | " + " | ".join(kinds_p) + " |", "|---|" + "---|" * len(kinds_p)]
    for i, k in enumerate(FEATURES):
        h = X[y == 1, i]
        h = h[np.isfinite(h)]
        cells = []
        for kd in kinds_p:
            p = X[(y == 0) & (kinds == kd), i]
            cells.append(f"{roc_auc(p[np.isfinite(p)], h):.3f}")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")

    # 2) 상관 행렬
    pear, spear = correlation(X, "pearson"), correlation(X, "spearman")
    labels = [f"{k[-1]} {FEATURE_NAMES[k][:8]}" for k in FEATURES]
    _save(os.path.join(OUT_DIR, "correlation_pearson.png"), draw_heatmap(pear, labels, "Pearson correlation (phi_i, phi_j)"))
    _save(os.path.join(OUT_DIR, "correlation_spearman.png"), draw_heatmap(spear, labels, "Spearman correlation"))
    _save(os.path.join(OUT_DIR, "distributions.png"), draw_distributions(X, y, aucs))
    lines += ["", "## 2. 지표 간 상관 (Spearman, |ρ| ≥ 0.5 인 쌍)", ""]
    pairs = [(FEATURES[i], FEATURES[j], spear[i, j]) for i in range(8) for j in range(i + 1, 8)
             if np.isfinite(spear[i, j]) and abs(spear[i, j]) >= 0.5]
    lines += [f"- {a} ↔ {b}: {r:+.2f}" for a, b, r in sorted(pairs, key=lambda t: -abs(t[2]))] or ["- 없음"]
    lines += ["", "그림: correlation_pearson.png, correlation_spearman.png, distributions.png"]

    # 3) 학습/평가 분할 + 로지스틱 회귀
    rng = np.random.default_rng(args.seed + 1)
    test_idx = np.concatenate([rng.permutation(np.flatnonzero(y == c))[: int(round(0.3 * (y == c).sum()))] for c in (0, 1)])
    train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
    Xtr, ytr, Xte, yte = X[train_idx], y[train_idx], X[test_idx], y[test_idx]
    lam, cv_scores = choose_lambda(Xtr, ytr, seed=args.seed)
    pp = Preprocessor().fit(Xtr)
    coef, bias = fit_logistic(pp.transform(Xtr), ytr, lam)
    w_pos = weights_from_coefficients(coef, pp.std, "positive")
    w_abs = weights_from_coefficients(coef, pp.std, "abs")
    w_eq = np.full(len(FEATURES), 1 / len(FEATURES))

    s_tr = score_matrix(Xtr, w_pos)
    th = choose_thresholds(s_tr, ytr)
    print(f"\nλ = {lam} (5-겹 CV AUC {cv_scores})")
    print(f"{'지표':14s} {'표준화 계수':>10s} {'w (권장)':>9s} {'|c| 정규화':>10s}")
    lines += ["", "## 3. 가중치 (학습 세트 70%, L2 로지스틱 회귀)", "",
              f"λ = {lam} (5-겹 교차검증 AUC: " + ", ".join(f"{k}: {v:.3f}" for k, v in cv_scores.items()) + ")", "",
              "| 지표 | 표준화 계수 c | w* (권장: max(c/σ,0) 정규화) | |c| 정규화 (요청 방식, 비교용) |",
              "|---|---|---|---|"]
    for k, c, a, b in zip(FEATURES, coef, w_pos, w_abs):
        print(f"{k} {FEATURE_NAMES[k]:9s} {c:+10.3f} {a:9.3f} {b:10.3f}")
        lines.append(f"| {k} {FEATURE_NAMES[k]} | {c:+.3f} | {a:.3f} | {b:.3f} |")
    neg = [k for k, c in zip(FEATURES, coef) if c < 0]
    if neg:
        lines += ["", f"음의 계수: {', '.join(neg)} - 다른 지표와 함께 쓸 때 수기일수록 오히려 낮게 기여. "
                       "|c| 정규화는 이들을 수기 쪽으로 더해 버리므로 권장 가중치에서는 0."]

    # 4) 평가 세트 성능
    res = {}
    for name, w in (("w* (권장)", w_pos), ("|c| 정규화", w_abs), ("균등 가중치", w_eq)):
        s = score_matrix(Xte, w)
        res[name] = roc_auc(s[(yte == 0) & np.isfinite(s)], s[(yte == 1) & np.isfinite(s)])
    z = pp.transform(Xte) @ coef + bias
    res["로지스틱 확률 (참고)"] = roc_auc(z[yte == 0], z[yte == 1])
    best_single = max(FEATURES, key=lambda k: aucs[k])
    i_best = FEATURES.index(best_single)
    v = Xte[:, i_best]
    res[f"단일 최고 지표 {best_single}"] = roc_auc(v[(yte == 0) & np.isfinite(v)], v[(yte == 1) & np.isfinite(v)])
    dm_tr = decision_metrics(s_tr, ytr, th)
    dm_te = decision_metrics(score_matrix(Xte, w_pos), yte, th)
    print("\n평가 세트 AUC: " + "  ".join(f"{k} {v:.3f}" for k, v in res.items()))
    print(f"임계값 (학습 세트, 확정 오판 ≤3%): 인쇄체 ≤ {th['printed']:.3f}, 수기 ≥ {th['handwritten']:.3f}")
    print(f"평가 세트: 확정 {dm_te['decided_rate']:.0%}, 확정 정확도 {dm_te['accuracy_on_decided']:.1%}, "
          f"검토 필요 {dm_te['review_rate']:.0%}")
    lines += ["", "## 4. 평가 세트 30% 성능", "", "| 점수 | AUC |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in res.items()]
    lines += ["", f"판정 임계값 (학습 세트에서 확정 구간 오판율 ≤ 3%): 인쇄체 확정 S ≤ {th['printed']:.3f}, "
                  f"수기 확정 S ≥ {th['handwritten']:.3f}", "",
              "| 세트 | 확정 비율 | 확정 정확도 | 검토 필요 |", "|---|---|---|---|",
              f"| 학습 | {dm_tr['decided_rate']:.0%} | {dm_tr['accuracy_on_decided']:.1%} | {dm_tr['review_rate']:.0%} |",
              f"| 평가 | {dm_te['decided_rate']:.0%} | {dm_te['accuracy_on_decided']:.1%} | {dm_te['review_rate']:.0%} |",
              "", "주의: 합성 데이터는 특징을 설계할 때 쓴 생성기 계열이라 실제보다 낙관적이다. "
                  "현장 라벨 사진으로 특징 CSV 를 만들어 다시 돌려야 한다."]

    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n리포트: {os.path.join(OUT_DIR, 'report.md')}")
    if not args.no_save:
        payload = {"weights": dict(zip(FEATURES, map(float, w_pos))), "thresholds": th,
                   "meta": {"date": str(date.today()), "source": "synthetic", "n_per_class": args.n, "seed": args.seed,
                            "lambda": lam, "test_auc": res["w* (권장)"],
                            "note": "합성 데이터 기준. 현장 데이터로 재산출 필요"}}
        with open(WEIGHTS_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"가중치 저장: {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
