"""
수기(Handwritten) vs 기계식 마킹(Printed) 판별 특징 - φ₇ 외곽선 불규칙도 / 테두리 거칠기

배경
  숏블라스트로 거칠어진 강판 위를 손으로 눌러 칠한 획은 잉크 번짐과 표면 마찰로 테두리가 잘게 들쭉날쭉하다.
  CNC·잉크젯·레이저 같은 기계식 마킹은 벡터 경로나 정형 액적이라 테두리가 매끈하다.

정의 (요청 수식)
  외곽선 k 의 둘레 P_k, 면적 A_k 로 등주비  R_k = P_k² / (4π A_k)       (원 = 1)
  R̄ = 면적 가중 평균,   φ₇ = 1 − exp(−γ₇ · max(0, R̄ − R₀))

구현상 결정 (이유)
  1. R_k 는 테두리 거칠기보다 글자 모양에 훨씬 크게 좌우된다. 매끈한 인쇄체라도 가는 'I' 획 하나가 R≈3,
     구멍 있는 'B' 나 가는 획일수록 R 이 급격히 커져서, 획이 가늘기만 해도 '거칠다' 로 나온다.
     그래서 점수에는 같은 외곽선을 획 폭 정도로 매끈하게 다듬은 형상의 등주비와의 비
         ρ_k = R_k / R_k^smooth     (≥ 1, 형상 영향이 분모에서 상쇄되고 테두리 요철만큼만 1 을 넘음)
     를 기본으로 쓴다 (score="relative"). 요청 수식 그대로의 R̄ 는 score="isoperimetric" 로 쓸 수 있고,
     R_k 값들은 항상 결과에 들어 있다.
  2. 화소 계단: 8-이웃 체인 둘레는 비스듬한 직선에서 최대 8% 가량 길게 나온다. 외곽선 좌표를 σ=1px 로
     먼저 다듬어 계단만 지운 뒤 둘레를 잰다 (2px 이상 크기의 요철은 남음).
  3. 형상 기준선(σ_s)은 테두리 요철(1~3px)보다 조금 크고 글자 모서리보다 훨씬 작아야 한다. 너무 크면
     모서리가 깎여 둘레가 줄어드는 '형상 변화' 가 거칠기로 잡힌다 (σ_s = 0.75×획 폭에서 매끈한 인쇄체가
     ρ≈1.3). 기본 0.15 × 획 폭, 하한 1.5px.
  4. 점 잡음(녹 반점, 찌꺼기)은 P²/A 가 폭발하므로 면적이 (2 × 획 폭)² 미만인 외곽선은 뺀다.
  5. 글자 하나만 있어도 동작한다 (외곽선 하나면 됨).
  6. 박스 카운팅 프랙탈 차원은 넣지 않았다: 화소 격자에서 잴 수 있는 2~16px 범위로는 거친 테두리와
     매끈한 테두리를 오히려 거꾸로 구분했다 (합성 실험 AUC 0.19~0.66).
"""
import math

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from orientation_feature import _cut_thin_lines, _merge_fragments, _remove_specks, binarize, load_image

GAMMA7 = 90.0              # φ₇ = 1 − exp(−γ·max(0, ρ̄ − R₀)): ρ̄ = R₀ + 0.0077 에서 0.5. 합성 기준, 현장 재보정 필요
R0_RELATIVE = 1.02         # score="relative" 기준값: 합성 인쇄체 ρ̄ 중앙 1.020 (p90 1.025), 거친 테두리 ≥1.028
R0_ISO = 3.0               # score="isoperimetric" 의 기준값 (인쇄체 글자 등주비 수준)
GAMMA_ISO = 0.3
PIXEL_SIGMA = 1.0          # 화소 계단 제거용 외곽선 스무딩 (px)
SHAPE_SIGMA_RATIO = 0.15   # 형상 기준선 스무딩 σ_s = 이 비율 × 획 폭 (합성 실험: 0.75 는 글자 모서리까지
                           # 깎아 매끈한 인쇄체도 ρ≈1.3, AUC 0.57 / 0.15 + 하한 1.5px 에서 AUC 0.98~1.00)
MIN_SHAPE_SIGMA = 1.5      # σ_s 하한 (px). 미세 스무딩(PIXEL_SIGMA) 보다는 커야 함
MIN_AREA_RATIO = 4.0       # 최소 면적 = 이 값 × 획 폭²  (= (2 × 획 폭)²)
LOCAL_WINDOW_RATIO = 1.0   # 국소 거칠기 창 = 이 비율 × 획 폭 (시각화용)


# ---------------------------------------------------------------------------
# 외곽선 기하
# ---------------------------------------------------------------------------
def smooth_closed(points, sigma):
    """닫힌 외곽선 좌표를 호 길이 1px 간격으로 다시 표본화한 뒤 가우시안 스무딩 (순환 경계)."""
    pts = np.vstack([points, points[:1]]).astype(np.float64)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(8, int(round(s[-1])))
    t = np.linspace(0, s[-1], n, endpoint=False)
    res = np.column_stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])])
    if sigma > 0:
        res = np.column_stack([gaussian_filter1d(res[:, i], sigma, mode="wrap") for i in range(2)])
    return res


def perimeter(points):
    return float(np.hypot(*np.diff(np.vstack([points, points[:1]]), axis=0).T).sum())


def polygon_area(points):
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def isoperimetric(points):
    a = polygon_area(points)
    return perimeter(points) ** 2 / (4 * math.pi * a) if a > 0 else float("inf"), a


def local_roughness(fine, shape, window):
    """창 길이마다 (미세 외곽선 길이 / 형상 외곽선 길이). fine, shape 는 같은 표본 수(같은 매개변수)."""
    seg_f = np.hypot(*np.diff(np.vstack([fine, fine[:1]]), axis=0).T)
    seg_s = np.hypot(*np.diff(np.vstack([shape, shape[:1]]), axis=0).T)
    k = np.ones(max(3, window))
    wrap = lambda v: np.concatenate([v[-len(k):], v, v[:len(k)]])
    lf = np.convolve(wrap(seg_f), k, "same")[len(k):-len(k)]
    ls = np.convolve(wrap(seg_s), k, "same")[len(k):-len(k)]
    return lf / np.maximum(ls, 1e-9)


# ---------------------------------------------------------------------------
# φ₇
# ---------------------------------------------------------------------------
def calculate_phi_7_roughness(image_path_or_array, score="relative", gamma=None, r0=None,
                              binarization="sauvola") -> dict:
    """단일 크롭 이미지의 φ₇ (테두리 거칠기).

    score  "relative"(기본): ρ_k = R_k / R_k^smooth 의 면적 가중 평균 (형상 영향 제거)
           "isoperimetric"  : 요청 수식 그대로 R̄ = 면적 가중 평균 R_k

    반환 dict
      phi_7, raw_roughness(점수에 쓴 가중 평균: ρ̄ 또는 R̄), isoperimetric_mean(R̄), relative_mean(ρ̄)
      perimeters, areas, roughness_values(R_k), relative_values(ρ_k), num_valid_contours
      stroke_width, contours(시각화용), reason
    """
    if gamma is None:
        gamma = GAMMA7 if score == "relative" else GAMMA_ISO
    if r0 is None:
        r0 = R0_RELATIVE if score == "relative" else R0_ISO
    img = load_image(image_path_or_array)
    mask = _remove_specks(binarize(img, binarization))
    mask, _ = _merge_fragments(mask)
    mask = _cut_thin_lines(mask)
    result = {"phi_7": None, "raw_roughness": None, "isoperimetric_mean": None, "relative_mean": None,
              "perimeters": [], "areas": [], "roughness_values": [], "relative_values": [],
              "num_valid_contours": 0, "stroke_width": None,
              "contours": [], "score": score, "reason": ""}
    if not mask.any():
        result["reason"] = "획 픽셀 없음"
        return result

    dist = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    stroke = 2 * float(np.percentile(dist[dist > 0], 90))       # 획 중심부 거리 ≈ 반폭
    result["stroke_width"] = stroke
    sigma_shape = max(MIN_SHAPE_SIGMA, SHAPE_SIGMA_RATIO * stroke)
    min_area = MIN_AREA_RATIO * stroke * stroke
    window = max(3, int(round(LOCAL_WINDOW_RATIO * stroke)))

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    H, W = mask.shape
    for c in contours:
        pts = c[:, 0, :].astype(np.float64)
        x, y, w, h = cv2.boundingRect(c)
        info = {"points": None, "valid": False, "reason": ""}
        if len(pts) < 8 or abs(cv2.contourArea(c)) < min_area:
            info["reason"] = "small"
        elif x == 0 or y == 0 or x + w >= W or y + h >= H:
            info["reason"] = "border"               # 크롭 경계에 잘린 외곽선은 직선 절단면이 섞임
        if info["reason"]:
            info["points"] = pts
            result["contours"].append(info)
            continue
        fine = smooth_closed(pts, PIXEL_SIGMA)
        shape = smooth_closed(pts, sigma_shape)
        r_fine, a_fine = isoperimetric(fine)
        r_shape, _ = isoperimetric(shape)
        rel = r_fine / r_shape
        info.update(points=fine, valid=True, perimeter=perimeter(fine), area=a_fine, R=r_fine, rho=rel,
                    local=local_roughness(fine, shape, window))
        result["contours"].append(info)

    valid = [c for c in result["contours"] if c["valid"]]
    if not valid:
        result["reason"] = f"면적 {min_area:.0f}px² 이상인 유효 외곽선 없음"
        return result
    w = np.array([c["area"] for c in valid])
    r_vals = np.array([c["R"] for c in valid])
    rho_vals = np.array([c["rho"] for c in valid])
    iso_mean = float(np.sum(w * r_vals) / w.sum())
    rel_mean = float(np.sum(w * rho_vals) / w.sum())
    raw = rel_mean if score == "relative" else iso_mean
    result.update(phi_7=float(1 - math.exp(-gamma * max(0.0, raw - r0))), raw_roughness=raw,
                  isoperimetric_mean=iso_mean, relative_mean=rel_mean,
                  perimeters=[c["perimeter"] for c in valid], areas=[c["area"] for c in valid],
                  roughness_values=r_vals.tolist(), relative_values=rho_vals.tolist(),
                  num_valid_contours=len(valid))
    return result


# ---------------------------------------------------------------------------
# 디버깅 시각화
# ---------------------------------------------------------------------------
def _local_color(v, lo=1.01, hi=1.08):
    """국소 길이 비 lo 이하 -> 초록, 중간 -> 노랑, hi 이상 -> 빨강 (BGR)."""
    t = min(1.0, max(0.0, (v - lo) / (hi - lo)))
    return (0, int(255 * min(1.0, 2 * (1 - t))), int(255 * min(1.0, 2 * t)))


def draw_roughness_debug(image_path_or_array, result, scale=None):
    """외곽선을 국소 거칠기(획 폭 길이 창의 미세/형상 둘레 비)로 칠한다: 초록=매끈, 노랑~빨강=들쭉날쭉.
    회색: 작아서/경계에 잘려서 제외된 외곽선."""
    img = load_image(image_path_or_array)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    canvas = cv2.cvtColor((gray * 0.55 + 100).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if scale is None:
        scale = max(1.0, 600 / max(canvas.shape[:2]))
    if scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    t = max(2, int(round(1.5 * scale)))
    for c in result["contours"]:
        pts = (np.asarray(c["points"]) * scale).astype(np.int32)
        if not c["valid"]:
            cv2.polylines(canvas, [pts], True, (150, 150, 150), 1, cv2.LINE_AA)
            continue
        loop = np.vstack([pts, pts[:1]])
        for i in range(len(pts)):
            cv2.line(canvas, tuple(loop[i]), tuple(loop[i + 1]), _local_color(c["local"][i]), t, cv2.LINE_AA)
    phi = result["phi_7"]
    text = (f"phi7={phi:.3f}  rho={result['relative_mean']:.3f}  Riso={result['isoperimetric_mean']:.2f}  "
            f"n={result['num_valid_contours']}  [{result['score']}]"
            if phi is not None else f"phi7=None: {result['reason']}")
    cv2.rectangle(canvas, (0, 0), (12 + 8 * len(text), 24), (0, 0, 0), -1)
    cv2.putText(canvas, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# 합성 시험 데이터
# ---------------------------------------------------------------------------
def roughen_edges(img, amount=0.35, grain=0.8, bleed=1.2, seed=0, background=200, ink=40):
    """거친 표면 위 번짐 흉내: 획 마스크를 흐린 뒤 공간 상관 잡음을 더해 다시 문턱 -> 테두리가 들쭉날쭉.
    amount: 잡음 세기(0~1), grain: 잡음 알갱이 크기(px), bleed: 번짐 폭(px)."""
    rng = np.random.default_rng(seed)
    ink_mask = (img < (background + ink) / 2).astype(np.float32)
    soft = cv2.GaussianBlur(ink_mask, (0, 0), bleed)
    noise = cv2.GaussianBlur(rng.normal(size=img.shape).astype(np.float32), (0, 0), grain)
    noise /= noise.std() + 1e-9
    # 요철은 테두리 근처 띠에서만 (바탕 전체에 잉크 점이 생기지 않게)
    band = cv2.dilate(ink_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    rough = (soft + amount * 0.5 * noise * band) > 0.5
    return np.where(rough, ink, background).astype(np.uint8)


def _save(path, image):
    ok, buf = cv2.imencode(".png", image)
    buf.tofile(path)


if __name__ == "__main__":
    import argparse

    from orientation_feature import add_plate_noise, render_printed

    parser = argparse.ArgumentParser(description="φ₇ 테두리 거칠기 계산 + 디버그 시각화")
    parser.add_argument("image", nargs="?", help="크롭 이미지 (없으면 합성 예시)")
    parser.add_argument("-o", "--output", default="phi7_debug.png")
    parser.add_argument("--score", default="relative", choices=["relative", "isoperimetric"])
    args = parser.parse_args()

    if args.image:
        samples = [(args.image, args.image)]
    else:
        printed = render_printed("B12-SP3 4500", thickness=8)
        samples = [("printed", add_plate_noise(printed)),
                   ("printed 7deg", add_plate_noise(render_printed("B12-SP3 4500", angle=7, thickness=8))),
                   ("rough 0.3", add_plate_noise(roughen_edges(printed, 0.3, seed=1))),
                   ("rough 0.6", add_plate_noise(roughen_edges(printed, 0.6, seed=2)))]
    panels = []
    for name, src in samples:
        res = calculate_phi_7_roughness(src, score=args.score)
        if res["phi_7"] is None:
            print(f"{name:13s} phi_7=None ({res['reason']})")
        else:
            print(f"{name:13s} phi_7={res['phi_7']:.3f}  rho={res['relative_mean']:.4f}  R_iso={res['isoperimetric_mean']:.3f}"
                  f"  contours={res['num_valid_contours']}  stroke={res['stroke_width']:.1f}px")
        panels.append(draw_roughness_debug(src, res, scale=1.0))
    width = max(p.shape[1] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, 4, 0, width - p.shape[1], cv2.BORDER_CONSTANT) for p in panels]
    _save(args.output, np.vstack(panels))
    print(f"시각화 저장: {args.output}")
