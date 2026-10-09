"""
기계식 마킹 사전 판별 (Early-exit): 사람이 손으로 만들기 어려운 규칙 패턴을 찾으면 즉시 인쇄체로 확정한다.

벤치마크에서 도트·스텐실 인쇄체는 획 기반 지표(φ₁·φ₅·φ₆·φ₇·φ₈)를 거꾸로 움직였다 (도트 테두리는 원의
연속이라 '거칠고', 스텐실 다리는 획 굵기·크기를 들쭉날쭉하게 만듦). 그래서 φ 계산 전에 걸러 낸다.

1. 도트 매트릭스 (DOD)
   - 도트 중심 = 거리 변환 국소 최대 (도트끼리 닿아 글자 하나가 한 덩어리여도 찾음)
   - 이웃 중심 사이에 '목' 이 있어야 함: 거리 변환이 접점에서 크게 줄어듦 (이어진 획의 능선은 평평)
   - 도트가 잉크의 대부분을 차지하며 (녹 반점은 작고 잉크 비중이 작음)
   - 도트 반지름이 고르고 (변동계수 작음)
   - 최근접 이웃 간격(피치)이 고르며, 이웃 방향이 격자처럼 두 축에 모임 (4θ 원형 통계)
   -> 반지름 R, 피치, 격자 방향을 함께 돌려준다.
2. 스텐실
   - 획 폭 정도의 닫힘 연산으로 메워지는 좁은 끊김(다리) 중, 크기가 비슷한 두 조각을 잇는 것만 센다
   - 다리가 여러 조각에 걸쳐 반복되고 (다리 수 / 조각 수), 다리 폭이 고르며 (변동계수 작음),
     방향이 글자 축과 평행/수직으로 정렬 (4θ 원형 통계)
   펜을 뗀 수기 끊김은 폭과 방향이 제각각이고 대부분 글자는 한 덩어리라 걸리지 않는다.
"""
import math

import cv2
import numpy as np

from orientation_feature import binarize, load_image

# 도트
DOT_MIN_COUNT = 15
DOT_MIN_RADIUS = 2.0        # px. 녹/요철 점(반지름 1~2px)과 구분
DOT_NECK_MAX = 0.8          # 이웃 중심 사이 거리 변환 최솟값 / 반지름 < 이 값이면 '도트' (이어진 획은 ≈1)
DOT_RATIO_MIN = 0.5         # 후보 중심 중 목이 있는 것의 비율 하한
DOT_INK_FRACTION = 0.4      # 도트 원 면적 합 / 잉크 면적 하한 (녹 반점은 작고 비중이 작음)
DOT_RADIUS_CV = 0.3
DOT_PITCH_CV = 0.25         # 피치 퍼짐 = 1.4826·MAD / 중앙값 (떨어진 도트 몇 개에 둔감)
DOT_PITCH_RANGE = (0.7, 2.6)  # 피치 / 지름 (도트가 조금씩 겹치면 1 보다 작음)
DOT_GRID_R = 0.5            # 이웃 방향 4θ 합성벡터 길이 하한 (격자 정렬)
# 스텐실
STENCIL_MIN_BRIDGES = 3
STENCIL_BRIDGE_RATIO = 0.2  # 다리 수 / 큰 조각 수 하한 (스텐실 글자는 2~4 조각에 다리 1~3 개)
STENCIL_WIDTH_CV = 0.4
STENCIL_AXIS_R = 0.5        # 다리 방향 4θ 합성벡터 길이 하한
STENCIL_PIECE_RATIO = 0.15  # 다리로 이어지는 두 조각 중 작은 쪽 면적 / 큰 쪽 면적 하한 (털·점 제외)


def _ink_mask(img):
    """도트·다리를 보존하는 이진화 (조각 병합·긁힘 열림 없이). 아주 작은 점만 제거."""
    mask = binarize(img)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= 6
    keep[0] = False
    return keep[lab]


def _axis_concentration(angles, weights=None):
    """각도(라디안)의 4θ 합성벡터 길이: 서로 직교하는 두 축에 모일수록 1."""
    a = np.asarray(angles, float)
    w = np.ones_like(a) if weights is None else np.asarray(weights, float)
    if len(a) == 0:
        return 0.0, 0.0
    z = np.sum(w * np.exp(4j * a)) / np.sum(w)
    return float(abs(z)), float(np.angle(z) / 4)


# ---------------------------------------------------------------------------
# 도트 매트릭스
# ---------------------------------------------------------------------------
def _peaks(dist, min_value, min_sep):
    """거리 변환의 국소 최대 (값 큰 순으로 min_sep 이내는 하나만 남김)."""
    is_max = (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8))) & (dist >= min_value)
    ys, xs = np.nonzero(is_max)
    order = np.argsort(-dist[ys, xs])
    kept, taken = [], np.zeros(dist.shape, bool)
    r = int(math.ceil(min_sep))
    for i in order:
        y, x = ys[i], xs[i]
        if taken[y, x]:
            continue
        kept.append((float(x), float(y), float(dist[y, x])))
        taken[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = True
    return np.array(kept) if kept else np.zeros((0, 3))


def _neck(dist, p, q, samples=12):
    """두 중심 사이 선분 위 거리 변환 최솟값 / 두 반지름 중 작은 것. 도트 사슬은 접점에서 크게 줄고(<0.8),
    이어진 획의 능선은 평평해서 1 에 가깝다. 떨어진 도트는 선분이 배경을 지나 0."""
    t = np.linspace(0, 1, samples)
    xs = np.clip(np.round(p[0] + t * (q[0] - p[0])).astype(int), 0, dist.shape[1] - 1)
    ys = np.clip(np.round(p[1] + t * (q[1] - p[1])).astype(int), 0, dist.shape[0] - 1)
    return float(dist[ys, xs].min() / max(min(p[2], q[2]), 1e-6))


def detect_dot_matrix(mask):
    """도트 중심 = 거리 변환 국소 최대 (도트끼리 닿아 한 덩어리여도 찾음).
    도트 판정은 이웃 중심 사이의 '목'(거리 변환이 접점에서 줄어드는 정도)으로 한다."""
    info = {"is_dot_matrix": False, "score": 0.0, "num_dots": 0, "radius": None, "pitch": None,
            "grid_angle_deg": None, "reason": "", "dot_centers": []}
    if not mask.any():
        info["reason"] = "잉크 없음"
        return info
    dist = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    rough = _peaks(dist, DOT_MIN_RADIUS, 1.0)
    if len(rough) < DOT_MIN_COUNT:
        info["reason"] = f"후보 중심 {len(rough)}개 < {DOT_MIN_COUNT}"
        return info
    r_est = float(np.median(rough[:, 2]))
    peaks = _peaks(dist, max(DOT_MIN_RADIUS, 0.6 * r_est), 1.2 * r_est)
    if len(peaks) < DOT_MIN_COUNT:
        info["reason"] = f"후보 중심 {len(peaks)}개 < {DOT_MIN_COUNT}"
        return info
    pts = peaks[:, :2]
    d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    np.fill_diagonal(d, np.inf)
    nn = d.argmin(axis=1)
    nn_d = d[np.arange(len(pts)), nn]
    necks = np.array([_neck(dist, peaks[i], peaks[j]) for i, j in enumerate(nn)])
    dotlike = necks < DOT_NECK_MAX
    dot_ratio = float(dotlike.mean())
    sel = peaks[dotlike]
    info["num_dots"] = int(dotlike.sum())
    if len(sel) < DOT_MIN_COUNT:
        info["reason"] = f"목이 있는 도트 {len(sel)}개 < {DOT_MIN_COUNT} (도트 비율 {dot_ratio:.2f})"
        return info
    radii = sel[:, 2]
    radius_cv = float(radii.std() / radii.mean())
    pitch_vals = nn_d[dotlike]
    pitch = float(np.median(pitch_vals))
    pitch_cv = float(1.4826 * np.median(np.abs(pitch_vals - pitch)) / pitch)
    vec = pts[nn[dotlike]] - pts[dotlike]
    grid_r, grid_angle = _axis_concentration(np.arctan2(vec[:, 1], vec[:, 0]))
    diameter = 2 * float(np.median(radii))
    ratio = pitch / diameter
    ink_fraction = float(min(1.0, len(sel) * math.pi * np.median(radii) ** 2 / mask.sum()))
    checks = {
        "dot_ratio": dot_ratio >= DOT_RATIO_MIN,
        "ink_fraction": ink_fraction >= DOT_INK_FRACTION,
        "radius_cv": radius_cv <= DOT_RADIUS_CV,
        "pitch_cv": pitch_cv <= DOT_PITCH_CV,
        "pitch_ratio": DOT_PITCH_RANGE[0] <= ratio <= DOT_PITCH_RANGE[1],
        "grid": grid_r >= DOT_GRID_R,
    }
    soft = [min(1, dot_ratio / DOT_RATIO_MIN), min(1, ink_fraction / DOT_INK_FRACTION),
            min(1, DOT_RADIUS_CV / max(radius_cv, 1e-6)), min(1, DOT_PITCH_CV / max(pitch_cv, 1e-6)),
            min(1, grid_r / DOT_GRID_R)]
    info.update(radius=float(np.median(radii)), pitch=pitch, grid_angle_deg=math.degrees(grid_angle),
                dot_ratio=dot_ratio, ink_fraction=ink_fraction, radius_cv=radius_cv, pitch_cv=pitch_cv,
                pitch_ratio=ratio, grid_concentration=grid_r, checks=checks, dot_centers=sel[:, :2].tolist(),
                score=float(np.prod(soft)) if all(checks.values()) else 0.0)
    info["is_dot_matrix"] = all(checks.values())
    if not info["is_dot_matrix"]:
        info["reason"] = "조건 불충족: " + ", ".join(k for k, ok in checks.items() if not ok)
    return info


# ---------------------------------------------------------------------------
# 스텐실
# ---------------------------------------------------------------------------
def detect_stencil(mask):
    """획 폭 크기의 닫힘으로 메워지는 끊김 중 '획 폭만큼 짧고 좁은 띠' 이면서 비슷한 크기의 두 조각을 잇는 것 = 다리.
    (글자 사이 간격도 닫힘으로 메워지지만 길이가 글자 높이만큼 길어서 걸러짐)"""
    info = {"is_stencil": False, "score": 0.0, "num_bridges": 0, "bridge_width": None, "reason": "", "bridges": []}
    if not mask.any():
        info["reason"] = "잉크 없음"
        return info
    m8 = mask.astype(np.uint8)
    dist = cv2.distanceTransform(np.pad(m8, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    stroke = 2 * float(np.percentile(dist[dist > 0], 90))
    k = max(3, int(round(1.1 * stroke)) | 1)
    # 원형 커널: 사각 커널은 기울어진 줄의 비스듬한 다리를 덜 메운다
    closed = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n0, lab0, st0, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    big = np.zeros(n0, bool)
    if n0 > 1:
        big[1:] = st0[1:, cv2.CC_STAT_AREA] >= 0.15 * np.median(st0[1:, cv2.CC_STAT_AREA])
    gaps = (closed > 0) & ~mask
    ng, labg, stg, _ = cv2.connectedComponentsWithStats(gaps.astype(np.uint8), connectivity=8)
    widths, angles = [], []
    for g in range(1, ng):
        x, y, w, h, area = stg[g]
        if area < 3:
            continue
        y0, y1 = max(0, y - 2), min(mask.shape[0], y + h + 2)
        x0, x1 = max(0, x - 2), min(mask.shape[1], x + w + 2)
        win = cv2.dilate((labg[y0:y1, x0:x1] == g).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2) > 0
        touch = [t for t in set(np.unique(lab0[y0:y1, x0:x1][win])) - {0} if big[t]]
        if len(touch) != 2:
            continue                                   # 다리는 정확히 두 조각을 잇는다
        a, b = sorted((int(st0[t, cv2.CC_STAT_AREA]) for t in touch), reverse=True)
        if b < STENCIL_PIECE_RATIO * a:
            continue
        pts = np.column_stack(np.nonzero(labg[y:y + h, x:x + w] == g))[:, ::-1].astype(np.float32)
        if len(pts) < 3:
            continue
        (_, _), (rw, rh), ang = cv2.minAreaRect(pts)
        width, length = min(rw, rh) + 1, max(rw, rh) + 1
        if width > 0.8 * stroke or length > 2.2 * stroke or length < 1.3 * width:
            continue                                   # 다리: 획 폭보다 좁고, 획 폭 정도로 짧은 띠
        long_angle = math.radians(ang if rw >= rh else ang + 90)
        widths.append(width)
        angles.append(long_angle)
        info["bridges"].append({"box": (int(x), int(y), int(w), int(h)), "width": float(width),
                                "angle_deg": math.degrees(long_angle)})
    info["num_bridges"] = len(widths)
    if len(widths) < STENCIL_MIN_BRIDGES:
        info["reason"] = f"다리 {len(widths)}개 < {STENCIL_MIN_BRIDGES}"
        return info
    pieces = [i for i in range(1, n0) if big[i]]
    bridge_ratio = len(widths) / max(1, len(pieces))
    w = np.array(widths)
    width_cv = float(w.std() / w.mean())
    axis_r, _ = _axis_concentration(angles)
    checks = {"bridge_ratio": bridge_ratio >= STENCIL_BRIDGE_RATIO, "width_cv": width_cv <= STENCIL_WIDTH_CV,
              "axis": axis_r >= STENCIL_AXIS_R}
    soft = [min(1, bridge_ratio / STENCIL_BRIDGE_RATIO), min(1, STENCIL_WIDTH_CV / max(width_cv, 1e-6)),
            min(1, axis_r / STENCIL_AXIS_R)]
    info.update(bridge_width=float(np.median(w)), width_cv=width_cv, axis_concentration=axis_r,
                bridge_ratio=bridge_ratio, stroke_width=stroke, checks=checks,
                score=float(np.prod(soft)) if all(checks.values()) else 0.0)
    info["is_stencil"] = all(checks.values())
    if not info["is_stencil"]:
        info["reason"] = "조건 불충족: " + ", ".join(k for k, ok in checks.items() if not ok)
    return info


def detect_mechanical_pattern(image_path_or_array, mask=None) -> dict:
    """도트/스텐실 검출. is_mechanical 이면 인쇄체 조기 확정 대상."""
    if mask is None:
        mask = _ink_mask(load_image(image_path_or_array))
    dot = detect_dot_matrix(mask)
    sten = detect_stencil(mask)
    kind = "dot_matrix" if dot["is_dot_matrix"] else ("stencil" if sten["is_stencil"] else None)
    return {"is_mechanical": kind is not None, "kind": kind, "is_dot_matrix": dot["is_dot_matrix"],
            "is_stencil": sten["is_stencil"], "score": max(dot["score"], sten["score"]),
            "dot": dot, "stencil": sten}


def draw_mechanical_debug(image_path_or_array, result):
    img = load_image(image_path_or_array)
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    dot = result["dot"]
    if dot.get("dot_centers"):
        r = max(2, int(round(dot["radius"])))
        for x, y in dot["dot_centers"]:
            cv2.circle(canvas, (int(x), int(y)), r, (0, 160, 255), 1, cv2.LINE_AA)
    for b in result["stencil"]["bridges"]:
        x, y, w, h = b["box"]
        cv2.rectangle(canvas, (x - 1, y - 1), (x + w, y + h), (255, 0, 200), 1)
    text = (f"MECHANICAL: {result['kind']} (score {result['score']:.2f})" if result["is_mechanical"]
            else f"no mechanical pattern  dots={dot['num_dots']}  bridges={result['stencil']['num_bridges']}")
    cv2.rectangle(canvas, (0, 0), (12 + 8 * len(text), 24), (0, 0, 0), -1)
    cv2.putText(canvas, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="도트/스텐실 기계 패턴 검출")
    parser.add_argument("image", nargs="?")
    parser.add_argument("-o", "--output", default="mechanical_debug.png")
    args = parser.parse_args()
    if args.image:
        samples = [(args.image, args.image)]
    else:
        import synthetic_dataset as sd
        rng = np.random.default_rng(3)
        samples = [(k, sd.make_sample(0, rng, k)[0]) for k in ("dots", "stencil", "sans")]
        samples.append(("handwritten", sd.make_sample(1, rng)[0]))
    panels = []
    for name, src in samples:
        res = detect_mechanical_pattern(src)
        d, s = res["dot"], res["stencil"]
        print(f"{name:12s} mechanical={res['kind']}  dots={d['num_dots']} ({d['reason'] or 'ok'})  "
              f"bridges={s['num_bridges']} ({s['reason'] or 'ok'})")
        panels.append(draw_mechanical_debug(src, res))
    width = max(p.shape[1] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, 4, 0, width - p.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
              for p in panels]
    ok, buf = cv2.imencode(".png", np.vstack(panels))
    buf.tofile(args.output)
    print(f"시각화 저장: {args.output}")
