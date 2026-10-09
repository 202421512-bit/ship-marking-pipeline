"""
수기(Handwritten) vs 기계식 마킹(Printed) 판별 특징 - φ₅ 곡률 변화 에너지 (Curvature Variation Energy)

배경
  손으로 쓴 궤적은 생리적 떨림(tremor)과 불완전한 운동 제어로 곡률이 잘게 출렁인다
  (Plamondon 의 운동학 이론). 기계 마킹은 직선(κ=0)이나 일정한 원호(κ=상수)라 dκ/ds 가 작다.

정의
  획 궤적 C(s) = (x(s), y(s)), s 는 호 길이
  κ(s)  = (x'y'' − y'x'') / (x'² + y'²)^(3/2)
  E_κ   = (1/L) ∫ (dκ/ds)² ds          획마다 계산 후 길이 L 로 가중 평균
  φ₅    = E_κ / (E_κ + β₅)

구현상 결정 (이유)
  1. 궤적은 '외곽선' 을 기본으로 쓴다. 합성 실험(정밀 원호 글자 vs 법선 방향 떨림)에서 골격선보다 구분력이
     높고(AUC 0.97~1.00 vs 0.81~1.00) 3배 빠르다. 단 외곽선은 획 굵기 변화도 곡률로 잡아 φ₁ 과 상관이 생긴다.
     펜 궤적만 보려면 curve="skeleton" (골격을 교차점에서 끊어 가지마다 순서대로 따라감).
  2. 스케일 불변: E_κ 의 단위는 길이⁻⁴ 이라 사진을 2배 키우면 1/16 이 된다. 좌표를 글자 높이 H 로
     나눈 무차원 좌표에서 계산한다.
  3. 화소 계단 현상과 표면 요철 잡음: 1px 간격으로 다시 표본화한 뒤 Savitzky-Golay(3차) 로 미분한다.
     창 길이는 글자 높이에 비례 (기본 0.25·H, 최소 9 표본) - 손떨림 파장(대략 0.2~0.35·H)은 남고
     1~2px 계단은 지워지는 크기.
  4. 'L', 'E' 같은 직각 모서리: κ 가 매우 커서 dκ/ds 가 발산하므로 |κ| 가 임계값을 넘는 구간과 그
     주변(창 절반)을 빼고, 남은 dκ/ds 도 상한으로 자른다. 교차점·끝점 근처는 골격이 휘므로 잘라 낸다.
"""
import math

import cv2
import numpy as np
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize

from orientation_feature import _cut_thin_lines, _merge_fragments, _remove_specks, binarize, load_image
from stroke_features import _prune_spurs, _skeleton_special_points

BETA5 = 1.3e3              # φ₅ = E/(E+β): φ₅=0.5 이 되는 E (무차원). 합성 외곽선 기준 인쇄체 p90 ≈1.2e3, 떨림 1px ≥2.2e3. 현장 재보정 필요
SMOOTH_RATIO = 0.25        # Savitzky-Golay 창 길이 = 이 비율 × 글자 높이 (px). 합성 실험에서 0.15 는 화소 계단이 남고 0.35 는 떨림까지 지움
MIN_WINDOW = 9             # 창 최소 표본 수 (홀수)
MIN_LENGTH_RATIO = 0.4     # 유효 획 최소 길이 L_min = 이 비율 × 글자 높이 (끝 자르기 후)
KAPPA_CORNER = 12.0        # |κ|·H 가 이보다 크면 모서리 (반경 < H/12) -> 주변 제외
DKDS_CLIP = 3.0e3          # |dκ/ds| 상한 (무차원). 남은 모서리 꼬리·잡음 한 점이 에너지를 지배하지 않게


# ---------------------------------------------------------------------------
# 곡선 추출
# ---------------------------------------------------------------------------
_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # 4-이웃 먼저


def trace_skeleton(skel):
    """골격을 교차점에서 끊고 가지마다 순서 있는 좌표열로. [(N, 2) (x, y), closed] 목록."""
    _, junctions = _skeleton_special_points(skel)
    near_j = cv2.dilate(junctions.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    branches = skel & ~near_j
    n, lab = cv2.connectedComponents(branches.astype(np.uint8), connectivity=8)
    ys, xs = np.nonzero(lab)
    order = np.argsort(lab[ys, xs], kind="stable")
    splits = np.flatnonzero(np.diff(lab[ys, xs][order])) + 1
    paths = []
    for idx in np.split(order, splits):
        pts = set(zip(ys[idx].tolist(), xs[idx].tolist()))
        if len(pts) < 3:
            continue
        deg = {p: sum((p[0] + dy, p[1] + dx) in pts for dy, dx in _NEIGHBORS) for p in pts}
        ends = [p for p, d in deg.items() if d == 1]
        closed = not ends
        cur = ends[0] if ends else min(pts)
        path, seen = [cur], {cur}
        while True:
            nxt = [(cur[0] + dy, cur[1] + dx) for dy, dx in _NEIGHBORS
                   if (cur[0] + dy, cur[1] + dx) in pts and (cur[0] + dy, cur[1] + dx) not in seen]
            if not nxt:
                break
            cur = min(nxt, key=lambda q: deg[q])   # 갈림이 생기면 덜 이어진 쪽(계단 모서리)을 먼저
            path.append(cur)
            seen.add(cur)
        arr = np.array(path, np.float64)[:, ::-1]  # (y, x) -> (x, y)
        paths.append((arr, closed and len(path) == len(pts)))
    return paths


def trace_contours(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return [(c[:, 0, :].astype(np.float64), True) for c in contours if len(c) >= 10]


def resample(points, closed, step=1.0):
    """호 길이 간격 step(px) 으로 다시 표본화 (8-이웃 대각선 이동의 √2 간격 불균일 제거)."""
    pts = np.vstack([points, points[:1]]) if closed else points
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 2 * step:
        return None
    if closed:
        # 닫힌 곡선은 전체 길이를 정수 개로 나눠 이음매에서도 간격이 같게 (짧은 마지막 구간 = 가짜 곡률 스파이크)
        t = np.linspace(0, s[-1], max(3, int(round(s[-1] / step))), endpoint=False)
    else:
        t = np.arange(0, s[-1] + 1e-9, step)
    return np.column_stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])])


# ---------------------------------------------------------------------------
# 곡률과 에너지
# ---------------------------------------------------------------------------
def curvature_profile(points, closed, scale, window):
    """무차원 좌표(÷scale)에서 κ(s), dκ/ds(s). 1px 균일 표본 가정. (κ, dκ/ds, 유효 표본 마스크)"""
    ds = 1.0 / scale
    x, y = points[:, 0] / scale, points[:, 1] / scale
    mode = "wrap" if closed else "interp"
    d = lambda v, k: savgol_filter(v, window, 3, deriv=k, delta=ds, mode=mode)
    x1, y1, x2, y2 = d(x, 1), d(y, 1), d(x, 2), d(y, 2)
    speed = np.maximum(x1 * x1 + y1 * y1, 1e-12)
    kappa = (x1 * y2 - y1 * x2) / speed ** 1.5
    dkds = np.gradient(kappa, ds) if not closed else (np.roll(kappa, -1) - np.roll(kappa, 1)) / (2 * ds)
    valid = np.ones(len(kappa), bool)
    if not closed:                      # 열린 곡선 양 끝은 창이 넘쳐 미분이 부정확
        valid[: window // 2] = False
        valid[len(valid) - window // 2:] = False
    return kappa, dkds, valid


def _mask_corners(kappa, valid, half):
    corner = np.abs(kappa) > KAPPA_CORNER
    if corner.any():
        grown = np.convolve(corner.astype(float), np.ones(2 * half + 1), mode="same") > 0
        valid = valid & ~grown
    return valid, corner


def calculate_phi_5_curvature(image_path_or_array, beta=BETA5, curve="contour",
                              smooth_ratio=SMOOTH_RATIO, binarization="sauvola") -> dict:
    """단일 크롭 이미지의 φ₅ (곡률 변화 에너지).

    curve  "contour"(기본, 외곽선) | "skeleton"(골격선 = 펜 궤적)

    반환 dict
      phi_5              정규화 점수 [0, 1) (유효 획이 없으면 None)
      raw_energy         길이 가중 평균 E_κ (무차원, 글자 높이 기준)
      num_valid_strokes  에너지 계산에 쓴 획 수
      char_height        정규화에 쓴 글자 높이 (px)
      window             Savitzky-Golay 창 (표본 수 = px)
      strokes            획별 상세: points(원본 px), energy_local((dκ/ds)², 제외 구간은 nan),
                         corner 마스크, length(px), energy, valid
      reason             phi_5 가 None 인 이유
    """
    img = load_image(image_path_or_array)
    mask = _remove_specks(binarize(img, binarization))
    mask, _ = _merge_fragments(mask)
    mask = _cut_thin_lines(mask)
    result = {"phi_5": None, "raw_energy": None, "num_valid_strokes": 0, "char_height": None,
              "window": None, "strokes": [], "curve": curve, "reason": ""}

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        result["reason"] = "획 픽셀 없음"
        return result
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = heights[areas >= 0.3 * np.median(areas)] if len(areas) > 2 else heights
    char_h = float(np.median(big))
    window = max(MIN_WINDOW, int(round(smooth_ratio * char_h)) | 1)
    result.update(char_height=char_h, window=window)

    if curve == "contour":
        paths = trace_contours(mask)
        trim = 0
    else:
        dist = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
        skel = _prune_spurs(skeletonize(mask), dist)
        paths = trace_skeleton(skel)
        half_w = float(np.median(dist[skel])) if skel.any() else 1.0
        trim = int(round(2 * half_w))   # 교차점/끝점 근처 골격 휨 제거 (획 폭만큼)

    min_len = MIN_LENGTH_RATIO * char_h
    total_w, total_e = 0.0, 0.0
    for points, closed in paths:
        pts = resample(points, closed)
        if pts is None:
            continue
        if not closed and trim:
            pts = pts[trim: len(pts) - trim] if len(pts) > 2 * trim else pts[:0]
        if len(pts) < max(window + 2, min_len):
            result["strokes"].append({"points": pts, "energy_local": None, "corner": None,
                                      "length": float(len(pts)), "energy": None, "valid": False})
            continue
        kappa, dkds, valid = curvature_profile(pts, closed, char_h, window)
        valid, corner = _mask_corners(kappa, valid, window // 2)
        local = np.clip(dkds, -DKDS_CLIP, DKDS_CLIP) ** 2
        local = np.where(valid, local, np.nan)
        length = float(valid.sum())                    # 1px 표본이므로 표본 수 = 길이(px)
        ok = length >= min_len
        energy = float(np.nanmean(local)) if ok else None
        result["strokes"].append({"points": pts, "energy_local": local, "corner": corner,
                                  "length": length, "energy": energy, "valid": ok})
        if ok:
            total_w += length
            total_e += length * energy

    valid_strokes = [s for s in result["strokes"] if s["valid"]]
    result["num_valid_strokes"] = len(valid_strokes)
    if not valid_strokes:
        result["reason"] = f"길이 {min_len:.0f}px 이상인 유효 획 없음"
        return result
    e = total_e / total_w
    result.update(raw_energy=e, phi_5=e / (e + beta))
    return result


# ---------------------------------------------------------------------------
# 디버깅 시각화
# ---------------------------------------------------------------------------
def draw_curvature_debug(image_path_or_array, result, scale=None, colormap=cv2.COLORMAP_JET, beta=BETA5):
    """궤적을 국소 (dκ/ds)² 로 칠한다 (로그 스케일, β 중심 ±2 자릿수). 회색: 짧아서 제외된 획,
    흰 점: 모서리로 제외된 구간. 오른쪽 아래에 색 막대."""
    img = load_image(image_path_or_array)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    canvas = cv2.cvtColor((gray * 0.55 + 100).astype(np.uint8), cv2.COLOR_GRAY2BGR)   # 바탕을 옅게
    if scale is None:
        scale = max(1.0, 600 / max(canvas.shape[:2]))
    if scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    lo, hi = math.log10(beta) - 2, math.log10(beta) + 2
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], colormap)[:, 0]
    t = max(2, int(round(1.5 * scale)))
    for s in result["strokes"]:
        pts = (s["points"] * scale).astype(np.int32)
        if not s["valid"] or s["energy_local"] is None:
            if len(pts) > 1:
                cv2.polylines(canvas, [pts], False, (150, 150, 150), 1, cv2.LINE_AA)
            continue
        val = np.log10(np.maximum(s["energy_local"], 1e-12))
        idx = np.clip((val - lo) / (hi - lo) * 255, 0, 255)
        for i in range(len(pts) - 1):
            if np.isnan(s["energy_local"][i]):
                color = (255, 255, 255) if s["corner"] is not None and s["corner"][i] else (150, 150, 150)
            else:
                color = tuple(int(c) for c in lut[int(idx[i])])
            cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]), color, t, cv2.LINE_AA)
    # 색 막대
    h, w = canvas.shape[:2]
    bar = cv2.resize(lut[::-1][:, None, :], (14, min(120, h - 40)), interpolation=cv2.INTER_NEAREST)
    y0, x0 = h - bar.shape[0] - 10, w - 70
    canvas[y0:y0 + bar.shape[0], x0:x0 + 14] = bar
    cv2.putText(canvas, f"1e{hi:.0f}", (x0 + 18, y0 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"1e{lo:.0f}", (x0 + 18, y0 + bar.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
                cv2.LINE_AA)
    phi = result["phi_5"]
    text = (f"phi5={phi:.3f}  E={result['raw_energy']:.3g}  strokes={result['num_valid_strokes']}  "
            f"H={result['char_height']:.0f}px win={result['window']}"
            if phi is not None else f"phi5=None: {result['reason']}")
    cv2.rectangle(canvas, (0, 0), (12 + 8 * len(text), 24), (0, 0, 0), -1)
    cv2.putText(canvas, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# 합성 시험 데이터
# ---------------------------------------------------------------------------
def add_tremor(img, amplitude=1.5, wavelength=10.0, seed=0, background=200):
    """손떨림 흉내: 파장 wavelength(px), 진폭 amplitude(px) 의 매끈한 무작위 변위장으로 영상을 휜다."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    field = []
    for _ in range(2):
        f = cv2.GaussianBlur(rng.normal(size=(h, w)).astype(np.float32), (0, 0), wavelength / 4)
        field.append(f / (f.std() + 1e-9) * amplitude)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + field[0], gy + field[1], cv2.INTER_LINEAR, borderValue=background)


def _glyph_paths(ch, x0, y0, h):
    """글자 모양을 정확한 원호/직선의 조밀한 좌표열로 (OpenCV 글꼴은 짧은 직선을 이은 다각형이라 곡선에
    가짜 곡률 스파이크가 있어 '기계식 = 매끈한 곡률' 시험에 맞지 않음)."""
    w, r = 0.6 * h, 0.3 * h
    t = lambda a0, a1, n=400: np.linspace(np.radians(a0), np.radians(a1), n)
    arc = lambda cx, cy, rx, ry, a0, a1: np.column_stack([cx + rx * np.cos(t(a0, a1)), cy + ry * np.sin(t(a0, a1))])
    line = lambda p, q: np.linspace(p, q, max(2, int(np.hypot(q[0] - p[0], q[1] - p[1]) * 2)))
    cx, top, mid, bot = x0 + w / 2, y0, y0 + h / 2, y0 + h
    if ch == "O":
        return [arc(cx, mid, w / 2, h / 2, 0, 360)]
    if ch == "C":
        return [arc(cx, mid, w / 2, h / 2, 40, 320)]
    if ch == "S":
        return [np.vstack([arc(cx, top + h / 4, w / 2, h / 4, -20, -270)[::-1][::-1],
                           arc(cx, bot - h / 4, w / 2, h / 4, -90, 160)])]
    if ch == "8":
        return [arc(cx, top + h / 4, w / 2.4, h / 4, 0, 360), arc(cx, bot - h / 4, w / 2, h / 4, 0, 360)]
    if ch == "3":
        return [arc(x0 + w / 2.2, top + h / 4, w / 2, h / 4, -160, 90), arc(x0 + w / 2.2, bot - h / 4, w / 2, h / 4, -90, 160)]
    if ch == "U":
        return [np.vstack([line((x0, top), (x0, bot - r)), arc(cx, bot - r, w / 2, r, 180, 0),
                           line((x0 + w, bot - r), (x0 + w, top))])]
    if ch == "L":
        return [line((x0, top), (x0, bot)), line((x0, bot), (x0 + w, bot))]
    if ch == "6":
        return [arc(cx, bot - h / 3.2, w / 2, h / 3.2, 0, 360), arc(cx + w * 0.55, mid, w * 1.05, h / 1.9, 180, 250)]
    raise ValueError(ch)


def render_glyph_line(text="OS83UC6L", h=46, thickness=6, tremor_amp=0.0, tremor_wavelength=10.0,
                      seed=0, size=(110, 520)):
    """정밀 원호/직선 글자 줄. tremor_amp > 0 이면 궤적 법선 방향으로 파장 tremor_wavelength(px)의
    띠 제한 떨림을 더한다 (손떨림 = 펜 진행 방향에 수직인 미세 진동)."""
    rng = np.random.default_rng(seed)
    img = np.full(size, 200, np.uint8)
    x = 15
    for ch in text:
        for p in _glyph_paths(ch, x, (size[0] - h) / 2, h):
            p = resample(p, False, step=0.5)
            if tremor_amp > 0:
                tang = np.gradient(p, axis=0)
                normal = np.column_stack([-tang[:, 1], tang[:, 0]]) / (np.hypot(*tang.T)[:, None] + 1e-9)
                noise = cv2.GaussianBlur(rng.normal(size=(len(p), 1)).astype(np.float32), (1, 0),
                                         sigmaX=1, sigmaY=tremor_wavelength / 4 / 0.5)[:, 0]
                p = p + normal * (noise / (noise.std() + 1e-9) * tremor_amp)[:, None]
            cv2.polylines(img, [np.round(p * 16).astype(np.int32)], False, 40, thickness, cv2.LINE_AA, shift=4)
        x += int(0.6 * h + 14)
    return img


def _save(path, image):
    ok, buf = cv2.imencode(".png", image)
    buf.tofile(path)


if __name__ == "__main__":
    import argparse

    from orientation_feature import add_plate_noise, render_printed

    parser = argparse.ArgumentParser(description="φ₅ 곡률 변화 에너지 계산 + 디버그 시각화")
    parser.add_argument("image", nargs="?", help="크롭 이미지 (없으면 합성 예시)")
    parser.add_argument("-o", "--output", default="phi5_debug.png")
    parser.add_argument("--curve", default="contour", choices=["skeleton", "contour"])
    args = parser.parse_args()

    if args.image:
        samples = [(args.image, args.image)]
    else:
        printed = render_printed("SB3 R80 G6")
        samples = [("printed", add_plate_noise(printed)),
                   ("printed 10deg", add_plate_noise(render_printed("SB3 R80 G6", angle=10))),
                   ("tremor 1px", add_plate_noise(add_tremor(printed, 1.0, 10, seed=1))),
                   ("tremor 2px", add_plate_noise(add_tremor(printed, 2.0, 10, seed=2)))]
    panels = []
    for name, src in samples:
        res = calculate_phi_5_curvature(src, curve=args.curve)
        if res["phi_5"] is None:
            print(f"{name:14s} phi_5=None ({res['reason']})")
        else:
            print(f"{name:14s} phi_5={res['phi_5']:.3f}  E={res['raw_energy']:.4g}  strokes={res['num_valid_strokes']}"
                  f"  H={res['char_height']:.0f}px  window={res['window']}")
        panels.append(draw_curvature_debug(src, res, scale=1.0))
    width = max(p.shape[1] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, 4, 0, width - p.shape[1], cv2.BORDER_CONSTANT) for p in panels]
    _save(args.output, np.vstack(panels))
    print(f"시각화 저장: {args.output}")
