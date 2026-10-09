"""
수기(Handwritten) vs 기계식 마킹(Printed) 판별 특징 - φ₄ 기준선 흔들림 (Baseline Fluctuation)

정의
  글자 k = 1..M (x 정렬) 의 가로 중심 x_k, 바닥 좌표 y_base,k
  기준선 y = a·x + b  (강건 회귀)
  RMSE_base = sqrt( (1/M) Σ (y_base,k − (a·x_k + b))² )
  φ₄ = tanh(α₄ · RMSE_base / H̄_char)      H̄_char: 유효 글자 평균 높이 (크기·거리 불변)

구현상 결정 (이유)
  1. 바닥점은 '줄 방향 좌표계' 에서 구한다. 줄이 기울면 이미지 y 기준 최하단은 글자 모서리가 되어
     글자 폭마다 다른 오차가 생긴다. 글자 중심점으로 줄 방향을 먼저 추정하고, 그 방향으로 세운
     좌표에서 열(column)마다 가장 아래 잉크를 찾은 뒤 상위 분위수(기본 80%)를 바닥으로 쓴다.
     -> 몇 개 열만 아래로 처진 흘러내림(drip)·긁힘 잔여는 무시되고, 'V', '7' 의 뾰족한 끝은 잡힌다.
  2. 회귀선은 강건하게(Theil-Sen 기본, RANSAC/Huber/OLS 선택) 맞추지만 RMSE 는 모든 유효 글자로 계산한다.
     이상치를 RMSE 에서 빼면 수기의 들쭉날쭉함 자체가 사라진다. 대신 잡음 한 점이 점수를
     지배하지 않게 잔차를 ±residual_cap·H̄ 로 자른다.
  3. 문장부호(높이 < 중앙값의 절반)와 디센더 후보(키가 중앙값보다 25% 이상 크고 바닥이 기준선보다
     0.2·H̄ 이상 아래)는 피팅과 RMSE 에서 뺀다. 크롭 위/아래 경계에 잘린 글자도 뺀다.
  4. 이 PC 에서는 SciPy 의 stats/optimize DLL 이 애플리케이션 제어 정책에 막혀 있어서 회귀는 NumPy 로 구현.
"""
import math

import cv2
import numpy as np

from orientation_feature import (_cut_thin_lines, _merge_fragments, _remove_specks, add_plate_noise,
                                 binarize, load_image, render_handwritten, render_printed)

ALPHA4 = 25.0             # tanh 기울기: φ₄=0.5 ≈ RMSE/H̄ 0.022. 합성 데이터 인쇄체 ≤0.014, 수기(±3px 흔들림) ≥0.025. 현장 재보정 필요
BASE_QUANTILE = 0.8       # 열별 바닥 위치의 상위 분위수 = 글자 바닥점 (상위 20% 열보다 좁은 흘러내림은 무시)
MERGE_OVERLAP = 0.5       # 진행 방향으로 짧은 쪽 폭의 50% 이상 겹치면 한 글자 (끊긴 도트/스텐실 글자)
PUNCT_HEIGHT_RATIO = 0.5  # 높이가 중앙값의 절반 미만이면 문장부호
DESCENDER_HEIGHT = 1.25   # 키가 중앙값의 이 배수 이상이고
DESCENDER_DROP = 0.2      # 바닥이 기준선보다 H̄ 의 이 비율 이상 아래면 디센더 후보
RESIDUAL_CAP = 0.5        # RMSE 계산 시 잔차 상한 (× H̄)
MAX_WIDTH_RATIO = 4.0     # 폭이 중앙 높이의 이 배수보다 넓으면 밑줄/긁힘 -> 제외


# ---------------------------------------------------------------------------
# 강건 직선 회귀 (y = a·x + b)
# ---------------------------------------------------------------------------
def fit_ols(x, y):
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def fit_theil_sen(x, y):
    """Theil-Sen: 모든 점 쌍 기울기의 중앙값, 절편은 y − a·x 의 중앙값. 붕괴점 약 29%."""
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    ok = np.abs(dx) > 1e-9
    a = float(np.median((y[j] - y[i])[ok] / dx[ok])) if ok.any() else 0.0
    return a, float(np.median(y - a * x))


def fit_huber(x, y, delta=1.345, iters=50):
    """Huber 손실 IRLS. 잔차 척도는 MAD 로 추정, |r| > δ·s 인 점은 가중치 δ·s/|r|."""
    a, b = fit_ols(x, y)
    for _ in range(iters):
        r = y - (a * x + b)
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        w = np.minimum(1.0, delta * s / np.maximum(np.abs(r), 1e-12))
        sw = np.sqrt(w)
        A = np.column_stack([x * sw, sw])
        (na, nb), *_ = np.linalg.lstsq(A, y * sw, rcond=None)
        if abs(na - a) < 1e-9 and abs(nb - b) < 1e-9:
            break
        a, b = float(na), float(nb)
    return a, b


def fit_ransac(x, y, threshold):
    """RANSAC: 글자 수가 적으므로 모든 점 쌍을 후보 직선으로 보고 inlier 가 가장 많은 것 -> inlier 로 OLS."""
    best, best_cost = None, None
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if abs(x[j] - x[i]) < 1e-9:
                continue
            a = (y[j] - y[i]) / (x[j] - x[i])
            r = np.abs(y - (a * x + y[i] - a * x[i]))
            inl = r < threshold
            cost = (-inl.sum(), r[inl].sum())          # inlier 많고, 같으면 오차 합이 작은 것
            if best_cost is None or cost < best_cost:
                best, best_cost = inl, cost
    if best is None or best.sum() < 2:
        return fit_ols(x, y)
    return fit_ols(x[best], y[best])


FITTERS = {"theil_sen": fit_theil_sen, "huber": fit_huber, "ols": fit_ols, "ransac": fit_ransac}


# ---------------------------------------------------------------------------
# 글자 바닥점
# ---------------------------------------------------------------------------
def _line_direction(cx, cy, fitter):
    """글자 중심점들로 줄 방향 단위벡터 (u: 진행 방향, n: 아래쪽 법선)."""
    a, _ = fit_theil_sen(cx, cy) if fitter != "ols" else fit_ols(cx, cy)
    u = np.array([1.0, a]) / math.hypot(1.0, a)
    return u, np.array([-u[1], u[0]])        # x 가 오른쪽, y 가 아래인 좌표에서 n 은 아래쪽


def _char_base(xs, ys, u, n, quantile):
    """줄 방향 좌표계에서 열마다 가장 아래 잉크의 위치 -> 상위 분위수 = 바닥. (along, across, 높이) 반환."""
    along = xs * u[0] + ys * u[1]
    across = xs * n[0] + ys * n[1]
    col = np.floor(along - along.min()).astype(int)
    bottoms = np.full(col.max() + 1, -np.inf)
    np.maximum.at(bottoms, col, across)
    bottoms = bottoms[np.isfinite(bottoms)]
    base = float(np.quantile(bottoms, quantile))
    return float(along.mean()), base, float(across.max() - across.min() + 1)


# ---------------------------------------------------------------------------
# φ₄
# ---------------------------------------------------------------------------
def _mark(group, reason):
    """글자 묶음과 그 구성 성분에 제외 사유를 기록 (시각화는 성분 단위로 그림)."""
    group["reason"] = reason
    for m in group["members"]:
        m["reason"] = reason


def calculate_phi_4_baseline(image_path_or_array, alpha=ALPHA4, fitter="theil_sen",
                             base_quantile=BASE_QUANTILE, residual_cap=RESIDUAL_CAP,
                             binarization="sauvola") -> dict:
    """단일 크롭 이미지의 φ₄ (기준선 흔들림).

    fitter  "theil_sen"(기본) | "ransac" | "huber" | "ols"

    반환 dict
      phi_4            정규화 점수 [0, 1) (유효 글자 2개 미만이면 None)
      rmse_px          기준선 수직 오차 RMSE (px, 잔차 상한 적용 후)
      rmse_norm        rmse_px / H̄_char
      slope_a, intercept_b   기준선 y = a·x + b (이미지 좌표)
      base_points      유효 글자 바닥점 [(x, y), ...] (x 정렬)
      residuals_px     각 바닥점의 수직 오차 (+ 면 기준선보다 아래)
      num_valid_chars  M
      mean_char_height H̄_char (px)
      components       성분별 상세 (시각화/디버깅용)
      reason           phi_4 가 None 인 이유
    """
    img = load_image(image_path_or_array)
    mask = _remove_specks(binarize(img, binarization))
    mask, _ = _merge_fragments(mask)
    mask = _cut_thin_lines(mask)
    n_lab, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    H, W = mask.shape

    comps = []
    for i in range(1, n_lab):
        x, y, w, h, area = (int(v) for v in stats[i])
        comps.append({"id": i, "box": (x, y, w, h), "area": area, "centroid": tuple(map(float, centroids[i])),
                      "used": False, "reason": ""})
    result = {"phi_4": None, "rmse_px": None, "rmse_norm": None, "slope_a": None, "intercept_b": None,
              "base_points": [], "residuals_px": [], "num_valid_chars": 0, "mean_char_height": None,
              "components": comps, "fitter": fitter, "reason": ""}

    # 1차 거르기: 경계에 잘린 글자, 밑줄/긁힘처럼 납작하고 긴 성분
    # (문장부호 판정은 끊긴 글자를 합친 뒤에 한다 - 조각 하나하나는 작아서 문장부호로 오인됨)
    if comps:
        median_h = np.median([c["box"][3] for c in comps])
        for c in comps:
            x, y, w, h = c["box"]
            if y == 0 or y + h >= H:
                c["reason"] = "border"
            elif w > MAX_WIDTH_RATIO * median_h:
                c["reason"] = "noise:line"
    chars = sorted([c for c in comps if not c["reason"]], key=lambda c: c["centroid"][0])
    if len(chars) < 2:
        result["reason"] = f"유효 글자 {len(chars)}개 (2개 이상 필요)"
        return result

    # 줄 방향 좌표계에서 글자별 바닥점
    cx = np.array([c["centroid"][0] for c in chars])
    cy = np.array([c["centroid"][1] for c in chars])
    u, nvec = _line_direction(cx, cy, fitter)

    # 진행 방향으로 많이 겹치는 성분은 한 글자로 합친다 (끊긴 도트/스텐실 글자, 위아래로 떨어진 조각,
    # 글자 바로 아래 떨어진 흘러내림). 조각마다 바닥을 재면 위쪽 조각이 기준선 위 이상치가 된다.
    for c in chars:
        ys, xs = np.nonzero(labels == c["id"])
        c["pts"] = (xs.astype(float), ys.astype(float))
        a_ = xs * u[0] + ys * u[1]
        c["span"] = (float(a_.min()), float(a_.max()))
    chars.sort(key=lambda c: c["span"][0])
    groups = []
    for c in chars:
        if groups:
            g = groups[-1]
            overlap = min(g["span"][1], c["span"][1]) - max(g["span"][0], c["span"][0])
            if overlap >= MERGE_OVERLAP * min(g["span"][1] - g["span"][0], c["span"][1] - c["span"][0]):
                g["span"] = (min(g["span"][0], c["span"][0]), max(g["span"][1], c["span"][1]))
                g["members"].append(c)
                continue
        groups.append({"span": c["span"], "members": [c], "reason": ""})
    for g in groups:
        xs = np.concatenate([m["pts"][0] for m in g["members"]])
        ys = np.concatenate([m["pts"][1] for m in g["members"]])
        along, across, height = _char_base(xs, ys, u, nvec, base_quantile)
        px, py = along * u + across * nvec                 # 줄 좌표 -> 이미지 좌표
        g.update(base=(float(px), float(py)), height=height, top=across - height + 1, bottom=across,
                 centroid=(float(xs.mean()), float(ys.mean())))
    chars = groups

    # 문장부호: 합친 글자 높이가 중앙값의 절반 미만 ('-', '.', ',')
    median_h = np.median([c["height"] for c in chars])
    for c in chars:
        if c["height"] < PUNCT_HEIGHT_RATIO * median_h:
            _mark(c, "punctuation")
    chars = [c for c in chars if not c["reason"]]
    if len(chars) < 2:
        result["reason"] = f"유효 글자 {len(chars)}개 (2개 이상 필요)"
        return result

    # 글자 몸통 띠(중심선) 밖에만 있는 작은 성분: 떨어져 나간 흘러내림(drip), 위쪽 얼룩
    centre = np.median([(c["top"] + c["bottom"]) / 2 for c in chars])
    median_h = np.median([c["height"] for c in chars])
    for c in chars:
        if c["height"] < 0.8 * median_h and (c["top"] > centre or c["bottom"] < centre):
            _mark(c, "noise:off_line")
    chars = [c for c in chars if not c["reason"]]
    if len(chars) < 2:
        result["reason"] = f"유효 글자 {len(chars)}개 (2개 이상 필요)"
        return result

    def fit_and_residuals(sel):
        bx = np.array([c["base"][0] for c in sel])
        by = np.array([c["base"][1] for c in sel])
        hbar = float(np.mean([c["height"] for c in sel]))
        if fitter == "ransac":
            a, b = fit_ransac(bx, by, threshold=0.1 * hbar)
        else:
            a, b = FITTERS[fitter](bx, by)
        return a, b, by - (a * bx + b), hbar

    # 디센더 후보 제외 (키가 크고 바닥이 기준선보다 확실히 아래) 후 다시 맞춤
    a, b, res, hbar = fit_and_residuals(chars)
    median_h = np.median([c["height"] for c in chars])
    for c, r in zip(chars, res):
        if c["height"] > DESCENDER_HEIGHT * median_h and r > DESCENDER_DROP * hbar:
            _mark(c, "descender")
    chars = [c for c in chars if not c["reason"]]
    if len(chars) < 2:
        result["reason"] = f"유효 글자 {len(chars)}개 (2개 이상 필요)"
        return result
    a, b, res, hbar = fit_and_residuals(chars)

    capped = np.clip(res, -residual_cap * hbar, residual_cap * hbar)
    rmse = float(np.sqrt(np.mean(capped ** 2)))
    for c, r in zip(chars, res):
        c["residual"] = float(r)
        for m in c["members"]:
            m["used"] = True
    result.update(phi_4=float(math.tanh(alpha * rmse / hbar)), rmse_px=rmse, rmse_norm=rmse / hbar,
                  slope_a=float(a), intercept_b=float(b), base_points=[c["base"] for c in chars],
                  residuals_px=[float(r) for r in res], num_valid_chars=len(chars), mean_char_height=hbar)
    return result


# ---------------------------------------------------------------------------
# 디버깅 시각화
# ---------------------------------------------------------------------------
REASON_COLORS = {"punctuation": (255, 160, 0), "descender": (200, 0, 200), "border": (160, 160, 160),
                 "noise:line": (0, 0, 255), "noise:off_line": (0, 140, 255)}


def draw_baseline_debug(image_path_or_array, result, scale=None):
    """원본 위에 기준선(파랑), 글자 바닥점(초록 점), 수직 오차(빨강 선), 제외된 성분(사유별 색 박스)."""
    img = load_image(image_path_or_array)
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    if scale is None:
        scale = max(1.0, 600 / max(canvas.shape[:2]))
    if scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    t = max(1, int(round(scale)))
    for c in result["components"]:
        if c["reason"] in REASON_COLORS:
            x, y, w, h = (int(v * scale) for v in c["box"])
            cv2.rectangle(canvas, (x, y), (x + w, y + h), REASON_COLORS[c["reason"]], t)
    if result["slope_a"] is not None:
        a, b = result["slope_a"], result["intercept_b"]
        w = canvas.shape[1] / scale
        p0 = (0, int(b * scale))
        p1 = (canvas.shape[1] - 1, int((a * w + b) * scale))
        cv2.line(canvas, p0, p1, (255, 120, 0), t, cv2.LINE_AA)
        for (x, y), r in zip(result["base_points"], result["residuals_px"]):
            on_line = (int(x * scale), int((a * x + b) * scale))
            pt = (int(x * scale), int(y * scale))
            cv2.line(canvas, on_line, pt, (0, 0, 255), t + 1, cv2.LINE_AA)
            cv2.circle(canvas, pt, t + 2, (0, 200, 0), -1, cv2.LINE_AA)
            cv2.putText(canvas, f"{r:+.1f}", (pt[0] + 4, pt[1] + 14 * t), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4 * t, (0, 0, 255), max(1, t // 2), cv2.LINE_AA)
    phi = result["phi_4"]
    text = (f"phi4={phi:.3f}  rmse={result['rmse_px']:.2f}px ({result['rmse_norm']:.3f}H)  "
            f"M={result['num_valid_chars']}  [{result['fitter']}]"
            if phi is not None else f"phi4=None: {result['reason']}")
    cv2.rectangle(canvas, (0, 0), (12 + 8 * len(text), 24), (0, 0, 0), -1)
    cv2.putText(canvas, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# 합성 시험 데이터
# ---------------------------------------------------------------------------
def add_drip(img, x, y_from, length=25, width=3, value=40):
    """글자 아래로 흘러내린 잉크 줄기."""
    out = img.copy()
    cv2.line(out, (x, y_from), (x, y_from + length), value, width, cv2.LINE_AA)
    return out


def _save(path, image):
    ok, buf = cv2.imencode(".png", image)
    buf.tofile(path)          # 한글 경로 저장


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="φ₄ 기준선 흔들림 계산 + 디버그 시각화")
    parser.add_argument("image", nargs="?", help="크롭 이미지 (없으면 합성 예시 4종)")
    parser.add_argument("-o", "--output", default="phi4_debug.png")
    parser.add_argument("--fitter", default="theil_sen", choices=sorted(FITTERS))
    args = parser.parse_args()

    if args.image:
        samples = [(args.image, args.image)]
    else:
        printed = render_printed("B12-SP3 4500")
        samples = [("printed", add_plate_noise(printed)),
                   ("printed 12deg", add_plate_noise(render_printed("B12-SP3 4500", angle=12))),
                   ("printed + drip", add_plate_noise(add_drip(printed, 64, 100))),
                   ("handwritten", add_plate_noise(render_handwritten("B12-SP3 4500", seed=3, tilt_std=5)))]
    panels = []
    for name, src in samples:
        res = calculate_phi_4_baseline(src, fitter=args.fitter)
        if res["phi_4"] is None:
            print(f"{name:16s} phi_4=None ({res['reason']})")
        else:
            print(f"{name:16s} phi_4={res['phi_4']:.3f}  rmse={res['rmse_px']:.2f}px  rmse/H={res['rmse_norm']:.4f}  "
                  f"a={res['slope_a']:+.4f} b={res['intercept_b']:.1f}  M={res['num_valid_chars']}")
        panels.append(draw_baseline_debug(src, res, scale=1.0))
    width = max(p.shape[1] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, 4, 0, width - p.shape[1], cv2.BORDER_CONSTANT) for p in panels]
    _save(args.output, np.vstack(panels))
    print(f"시각화 저장: {args.output}")
