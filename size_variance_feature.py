"""
수기(Handwritten) vs 기계식 마킹(Printed) 판별 특징 - φ₆ 연결요소 크기·종횡비 분산

정의
  유효 글자 k = 1..M 의 픽셀 면적 A_k, 종횡비 AR_k = W_k / H_k
  CV_A  = σ_A / μ_A,   CV_AR = σ_AR / μ_AR        (σ 는 모집단 표준편차, 1/M)
  φ₆    = ½ [ tanh(α_A·CV_A) + tanh(α_AR·CV_AR) ]

구현상 결정 (이유)
  1. 글자 단위로 묶는다: 도트/스텐실처럼 끊긴 글자의 조각이 각각 '작은 글자' 로 잡히면 면적 분산이
     부풀어 인쇄체가 수기처럼 보인다. 줄 방향으로 50% 이상 겹치는 조각은 한 글자로 합친다 (φ₄ 와 같은 규칙).
  2. W, H 는 줄 방향 좌표계의 폭·높이다. 이미지 축 기준 상자는 줄이 기울면 글자마다 다르게 늘어난다.
  3. 노이즈: 점 잡음(면적 하한), 긁힘(글자보다 훨씬 넓거나 큰 성분), 크롭 경계에 잘린 글자는 제외.
     문장부호(높이 < 중앙값의 절반: '.', '-')도 제외. '/' 처럼 키는 크지만 면적이 작은 기호는
     robust 옵션(IQR 울타리 / Huber 형 윈저화)으로 다룬다.
  4. 인쇄체도 글자 모양마다 면적·종횡비가 다르다 ('1' 은 좁고 '0', 'M' 은 넓음). 그래서 CV_A, CV_AR 은
     인쇄체에서도 0 이 아니다. 대문자·숫자는 글꼴 규격상 높이가 같으므로 높이 변동계수 cv_height 를
     함께 계산해 돌려준다 (score="height" 로 φ₆ 에 쓸 수 있음).
"""
import math

import cv2
import numpy as np

from baseline_feature import fit_theil_sen
from orientation_feature import _cut_thin_lines, _merge_fragments, _remove_specks, binarize, load_image

ALPHA_A = 3.5             # tanh(α_A·CV_A): 0.5 ≈ CV_A 0.157 (합성 인쇄체 0.11, 수기 0.19 사이)
ALPHA_AR = 3.0            # tanh(α_AR·CV_AR). 합성 실험에서 종횡비 분산은 구분력이 거의 없음 (AUC 0.47~0.68)
ALPHA_H = 18.0            # score="height": tanh(α_H·CV_H), 0.5 ≈ CV_H 0.03 (합성 인쇄체 0.01, 수기 0.07~0.08)
PUNCT_HEIGHT_RATIO = 0.5  # 높이가 중앙값의 절반 미만이면 문장부호
MAX_WIDTH_RATIO = 4.0     # 폭이 중앙 높이의 이 배수보다 넓으면 긁힘/밑줄
MAX_AREA_RATIO = 5.0      # 면적이 중앙 면적의 이 배수보다 크면 얼룩/긁힘 덩어리
MERGE_OVERLAP = 0.5       # 진행 방향으로 짧은 쪽 폭의 50% 이상 겹치면 한 글자
IQR_K = 1.5               # robust="iqr": Q1 − k·IQR ~ Q3 + k·IQR 밖은 제외
HUBER_K = 2.0             # robust="huber": 중앙값 ± k·1.4826·MAD 로 값 자르기 (윈저화)


# ---------------------------------------------------------------------------
# 글자 분리
# ---------------------------------------------------------------------------
def extract_characters(img, binarization="sauvola"):
    """글자 단위 성분 목록과 줄 방향 (u, n). 각 글자: area, width, height(줄 좌표), 꼭짓점 4개, members."""
    mask = _remove_specks(binarize(img, binarization))
    mask, _ = _merge_fragments(mask)
    mask = _cut_thin_lines(mask)
    n_lab, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    H, W = mask.shape
    comps = []
    for i in range(1, n_lab):
        x, y, w, h, area = (int(v) for v in stats[i])
        comps.append({"id": i, "box": (x, y, w, h), "area": area,
                      "centroid": tuple(map(float, centroids[i])), "reason": ""})
    if not comps:
        return comps, [], None

    median_h = np.median([c["box"][3] for c in comps])
    median_a = np.median([c["area"] for c in comps])
    for c in comps:
        x, y, w, h = c["box"]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            c["reason"] = "border"
        elif w > MAX_WIDTH_RATIO * median_h or c["area"] > MAX_AREA_RATIO * median_a:
            c["reason"] = "noise:scratch"
    cand = [c for c in comps if not c["reason"]]
    if len(cand) < 2:
        return comps, [], None

    # 줄 방향 (글자 중심점 Theil-Sen)
    cx = np.array([c["centroid"][0] for c in cand])
    cy = np.array([c["centroid"][1] for c in cand])
    a, _ = fit_theil_sen(cx, cy)
    u = np.array([1.0, a]) / math.hypot(1.0, a)
    nvec = np.array([-u[1], u[0]])

    for c in cand:
        ys, xs = np.nonzero(labels == c["id"])
        along, across = xs * u[0] + ys * u[1], xs * nvec[0] + ys * nvec[1]
        c["span"] = (float(along.min()), float(along.max()))
        c["across"] = (float(across.min()), float(across.max()))

    # 글자 띠(위·아래 경계의 중앙값) 와 거의 겹치지 않는 성분은 병합 전에 뺀다.
    # 같은 세로 열의 줄 위 긁힘 조각·얼룩이 글자와 합쳐지면 그 글자 높이가 크게 부풀기 때문.
    band = (np.median([c["across"][0] for c in cand]), np.median([c["across"][1] for c in cand]))
    for c in cand:
        h = c["across"][1] - c["across"][0] + 1
        overlap = min(c["across"][1], band[1]) - max(c["across"][0], band[0])
        if overlap < 0.5 * min(h, band[1] - band[0] + 1):
            c["reason"] = "noise:off_line"
    cand = [c for c in cand if not c["reason"]]
    if len(cand) < 2:
        return comps, [], (u, nvec)
    cand.sort(key=lambda c: c["span"][0])
    chars = []
    for c in cand:
        if chars:
            g = chars[-1]
            ov = min(g["span"][1], c["span"][1]) - max(g["span"][0], c["span"][0])
            if ov >= MERGE_OVERLAP * min(g["span"][1] - g["span"][0], c["span"][1] - c["span"][0]):
                g["span"] = (min(g["span"][0], c["span"][0]), max(g["span"][1], c["span"][1]))
                g["across"] = (min(g["across"][0], c["across"][0]), max(g["across"][1], c["across"][1]))
                g["area"] += c["area"]
                g["members"].append(c)
                continue
        chars.append({"span": c["span"], "across": c["across"], "area": c["area"], "members": [c], "reason": ""})
    for g in chars:
        g["width"] = g["span"][1] - g["span"][0] + 1
        g["height"] = g["across"][1] - g["across"][0] + 1
        g["aspect_ratio"] = g["width"] / g["height"]
        (a0, a1), (n0, n1) = g["span"], g["across"]
        g["corners"] = [tuple(p * u + q * nvec) for p, q in [(a0, n0), (a1, n0), (a1, n1), (a0, n1)]]
    return comps, chars, (u, nvec)


# ---------------------------------------------------------------------------
# 통계
# ---------------------------------------------------------------------------
def coefficient_of_variation(values):
    v = np.asarray(values, np.float64)
    mu = v.mean()
    return float(v.std() / mu) if mu > 0 else float("nan")


def _iqr_keep(values):
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    return (values >= q1 - IQR_K * iqr) & (values <= q3 + IQR_K * iqr)


def _winsorize(values):
    med = np.median(values)
    dev = np.abs(values - med)
    scale = 1.4826 * np.median(dev)
    if scale == 0:                       # 절반 넘게 같은 값이면 MAD=0 -> 평균 절대 편차로 대신
        scale = 1.2533 * dev.mean()
    return np.clip(values, med - HUBER_K * scale, med + HUBER_K * scale) if scale > 0 else values


def calculate_phi_6_size_variance(image_path_or_array, alpha_area=ALPHA_A, alpha_ar=ALPHA_AR,
                                  robust="iqr", score="spec", alpha_height=ALPHA_H,
                                  binarization="sauvola") -> dict:
    """단일 크롭 이미지의 φ₆ (크기·종횡비 분산).

    robust  "iqr"(기본, 면적·종횡비 중 하나라도 IQR 울타리 밖인 글자 제외) | "none"(정의 그대로)
            | "huber"(중앙값 ± 2·MAD 로 값을 잘라 평균·표준편차 계산)
    score   "spec"   : ½[tanh(α_A·CV_A) + tanh(α_AR·CV_AR)]  (정의)
            "height" : tanh(α_H·CV_H) - 글자 높이 변동만 사용 (모듈 docstring 4 참고)

    반환 dict: phi_6, cv_area, cv_aspect_ratio, cv_height, areas, aspect_ratios, heights,
              num_valid_chars, chars(시각화용), components, reason
    """
    img = load_image(image_path_or_array)
    comps, chars, frame = extract_characters(img, binarization)
    result = {"phi_6": None, "cv_area": None, "cv_aspect_ratio": None, "cv_height": None,
              "areas": [], "aspect_ratios": [], "heights": [], "num_valid_chars": 0,
              "chars": chars, "components": comps, "robust": robust, "score": score, "reason": ""}

    def mark(g, reason):
        g["reason"] = reason
        for m in g["members"]:
            m["reason"] = reason

    if chars:
        median_h = np.median([g["height"] for g in chars])
        for g in chars:
            if g["height"] < PUNCT_HEIGHT_RATIO * median_h:
                mark(g, "punctuation")
    valid = [g for g in chars if not g["reason"]]
    if robust == "iqr" and len(valid) >= 4:
        a = np.array([g["area"] for g in valid], float)
        r = np.array([g["aspect_ratio"] for g in valid], float)
        keep = _iqr_keep(a) & _iqr_keep(r)
        for g, k in zip(valid, keep):
            if not k:
                mark(g, "outlier:iqr")
        valid = [g for g in valid if not g["reason"]]
    if len(valid) < 2:
        result["reason"] = f"유효 글자 {len(valid)}개 (2개 이상 필요)"
        return result

    areas = np.array([g["area"] for g in valid], float)
    ars = np.array([g["aspect_ratio"] for g in valid], float)
    heights = np.array([g["height"] for g in valid], float)
    stats_in = (_winsorize(areas), _winsorize(ars), _winsorize(heights)) if robust == "huber" else (areas, ars, heights)
    cv_a, cv_ar, cv_h = (coefficient_of_variation(v) for v in stats_in)
    if score == "height":
        phi = math.tanh(alpha_height * cv_h)
    else:
        phi = 0.5 * (math.tanh(alpha_area * cv_a) + math.tanh(alpha_ar * cv_ar))

    mu_a, mu_ar = areas.mean(), ars.mean()
    for g in valid:
        g["used"] = True
        g["deviation"] = max(abs(g["area"] / mu_a - 1), abs(g["aspect_ratio"] / mu_ar - 1))
    result.update(phi_6=float(phi), cv_area=cv_a, cv_aspect_ratio=cv_ar, cv_height=cv_h,
                  areas=areas.tolist(), aspect_ratios=ars.tolist(), heights=heights.tolist(),
                  num_valid_chars=len(valid))
    return result


# ---------------------------------------------------------------------------
# 디버깅 시각화
# ---------------------------------------------------------------------------
REASON_COLORS = {"punctuation": (255, 160, 0), "border": (160, 160, 160), "noise:scratch": (0, 0, 255),
                 "outlier:iqr": (200, 0, 200), "noise:off_line": (0, 140, 255)}


def _deviation_color(d, ok=0.1, full=0.5):
    """편차 ok 이하 -> 초록, 중간 -> 노랑, full 이상 -> 빨강 (BGR)."""
    t = min(1.0, max(0.0, (d - ok) / (full - ok)))
    return (0, int(255 * min(1.0, 2 * (1 - t))), int(255 * min(1.0, 2 * t)))


def draw_size_variance_debug(image_path_or_array, result, scale=None):
    """글자별 (줄 방향) 상자를 평균 대비 편차로 칠한다: 초록=±10% 이내, 빨강=면적 또는 종횡비가 ±50% 이상.
    상자 위 숫자는 면적/평균, 아래는 종횡비. 제외된 성분은 사유별 색의 가는 상자."""
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
            cv2.rectangle(canvas, (x, y), (x + w, y + h), REASON_COLORS[c["reason"]], 1)
    mu_a = np.mean(result["areas"]) if result["areas"] else None
    for g in result["chars"]:
        if not g.get("used"):
            continue
        quad = (np.array(g["corners"]) * scale).astype(np.int32)
        color = _deviation_color(g["deviation"])
        cv2.polylines(canvas, [quad], True, color, t + 1, cv2.LINE_AA)
        top = quad[np.argmin(quad[:, 1])]
        bottom = quad[np.argmax(quad[:, 1])]
        cv2.putText(canvas, f"{g['area'] / mu_a:.2f}", (int(top[0]), int(top[1]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38 * t, color, t, cv2.LINE_AA)
        cv2.putText(canvas, f"{g['aspect_ratio']:.2f}", (int(bottom[0]), int(bottom[1]) + 12 * t),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38 * t, color, t, cv2.LINE_AA)
    phi = result["phi_6"]
    text = (f"phi6={phi:.3f}  CV_A={result['cv_area']:.3f}  CV_AR={result['cv_aspect_ratio']:.3f}  "
            f"CV_H={result['cv_height']:.3f}  M={result['num_valid_chars']}"
            if phi is not None else f"phi6=None: {result['reason']}")
    cv2.rectangle(canvas, (0, 0), (12 + 8 * len(text), 24), (0, 0, 0), -1)
    cv2.putText(canvas, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def _save(path, image):
    ok, buf = cv2.imencode(".png", image)
    buf.tofile(path)


if __name__ == "__main__":
    import argparse

    from orientation_feature import add_plate_noise, render_handwritten, render_printed

    parser = argparse.ArgumentParser(description="φ₆ 크기·종횡비 분산 계산 + 디버그 시각화")
    parser.add_argument("image", nargs="?", help="크롭 이미지 (없으면 합성 예시)")
    parser.add_argument("-o", "--output", default="phi6_debug.png")
    parser.add_argument("--robust", default="iqr", choices=["none", "iqr", "huber"])
    parser.add_argument("--score", default="spec", choices=["spec", "height"])
    args = parser.parse_args()

    if args.image:
        samples = [(args.image, args.image)]
    else:
        samples = [("printed", add_plate_noise(render_printed("B12-SP3 4500"))),
                   ("printed 10deg", add_plate_noise(render_printed("HK357 B12", angle=10))),
                   ("printed dots", add_plate_noise(render_printed("A2-05 E8K", dots=True))),
                   ("handwritten", add_plate_noise(render_handwritten("B12-SP3 4500", seed=3, tilt_std=5)))]
    panels = []
    for name, src in samples:
        res = calculate_phi_6_size_variance(src, robust=args.robust, score=args.score)
        if res["phi_6"] is None:
            print(f"{name:14s} phi_6=None ({res['reason']})")
        else:
            print(f"{name:14s} phi_6={res['phi_6']:.3f}  CV_A={res['cv_area']:.3f}  CV_AR={res['cv_aspect_ratio']:.3f}"
                  f"  CV_H={res['cv_height']:.3f}  M={res['num_valid_chars']}")
        panels.append(draw_size_variance_debug(src, res, scale=1.0))
    width = max(p.shape[1] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, 4, 0, width - p.shape[1], cv2.BORDER_CONSTANT) for p in panels]
    _save(args.output, np.vstack(panels))
    print(f"시각화 저장: {args.output}")
