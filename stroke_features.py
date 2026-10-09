"""
수기(Handwritten) vs 기계식 마킹(Printed) 판별 특징 - φ₁ 획 두께 변화 (Stroke Thickness Variation)

정의
  B        : 이진화된 획 영역
  D(x, y)  : B 의 각 점에서 가장 가까운 배경(B^c)까지의 유클리드 거리 (거리 변환)
  S        : 획의 중심 골격선 픽셀 {p_1 … p_N}
  W_k      : 2·D(p_k)                       (골격점의 국소 선폭)
  μ_W, σ_W : W 의 평균, 표준편차 (모집단 표준편차)
  φ₁       : tanh(α₁ · σ_W / μ_W) ∈ [0, 1)

기계식 마킹(잉크젯 도트, 스텐실, 타각)은 노즐/공구 폭이 일정해 φ₁ 이 작고,
손글씨는 필압·속도·필기구 기울기에 따라 굵기가 변해 φ₁ 이 크다.

정의를 그대로 계산하면 값이 흔들리는 원인과 대책 (모두 끌 수 있음)
  1. 골격 끝점: 획 끝으로 갈수록 D 가 작아져 W 를 과소 추정  -> 끝점 주변 골격 제외
  2. 교차점(T, X 자): 두 획이 만나는 곳은 D 가 커져 W 를 과대 추정 -> 교차점 주변 골격 제외
  3. 가지(spur): 획 테두리의 요철에서 뻗은 짧은 골격 가지 -> 이진 마스크를 다듬은 뒤 골격화
  4. 획 안의 작은 구멍(도막 기포, 녹 반점): D 가 갑자기 작아짐 -> 작은 구멍 메우기
  5. 가는 획의 화소 양자화: 폭 3~4px 이면 W 가 정수 단위로만 변해 CV 가 부풀려짐
     -> 획이 가늘면 확대한 뒤 이진화
  6. 남은 이상치 -> 양쪽 trim 비율만큼 잘라낸 뒤 μ, σ (trim=0 이면 정의 그대로)
"""
from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.morphology import remove_small_holes, remove_small_objects, skeletonize

ALPHA1 = 4.0   # tanh 기울기: φ₁=0.5 가 CV≈0.137 (합성 인쇄체 CV 0.09~0.12 와 손글씨 0.14~0.46 사이). 현장 데이터로 재보정 필요


@dataclass
class StrokeThickness:
    phi1: float                 # 정규화 지표 tanh(α·CV), 판정 불가면 nan
    cv: float                   # σ_W / μ_W
    mu_w: float                 # 평균 선폭 (입력 이미지 px 기준)
    sigma_w: float              # 선폭 표준편차 (입력 이미지 px 기준)
    n_samples: int              # 통계에 쓴 골격점 수
    valid: bool                 # 표본이 충분해 믿을 만한가
    reason: str = ""            # valid=False 인 이유
    debug: dict = field(default_factory=dict, repr=False)  # 이진 마스크, 골격, 사용 골격 등


# ---------------------------------------------------------------------------
# 이진화
# ---------------------------------------------------------------------------
def binarize_strokes(crop, polarity="auto", bg_kernel_ratio=0.5):
    """크롭 이미지 -> 획 마스크 (bool).

    조명 불균일/강판 얼룩을 줄이려고 큰 커널의 배경(모폴로지 닫힘·열림의 중간값)을 빼고 Otsu.
    polarity: "dark"(어두운 획), "light"(밝은 획), "auto"(배경 밝기에서 먼 쪽을 획으로).
    이미 이진(0/255 또는 bool) 영상이면 그대로 쓴다.
    """
    img = np.asarray(crop)
    if img.dtype == bool:
        return img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.astype(np.uint8)
    values = np.unique(gray)
    if len(values) == 1:
        return np.zeros_like(gray, bool)
    if len(values) == 2:                       # 이미 2단계 영상: 극성 규칙만 적용
        lo_v, hi_v = values
        if polarity == "dark":
            return gray == lo_v
        if polarity == "light":
            return gray == hi_v
        return gray == (lo_v if (gray == lo_v).sum() <= (gray == hi_v).sum() else hi_v)  # 소수가 획

    k = max(15, int(min(gray.shape) * bg_kernel_ratio) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg_for_dark = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se)   # 어두운 획을 지운 배경
    bg_for_light = cv2.morphologyEx(gray, cv2.MORPH_OPEN, se)   # 밝은 획을 지운 배경
    dark = cv2.subtract(bg_for_dark, gray)                       # 어두운 획이 밝게
    light = cv2.subtract(gray, bg_for_light)                     # 밝은 획이 밝게

    def otsu(x):
        x = cv2.GaussianBlur(x, (3, 3), 0)
        t, m = cv2.threshold(x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return m > 0, t

    if polarity == "dark":
        return otsu(dark)[0]
    if polarity == "light":
        return otsu(light)[0]
    # 배경이 다수: 중앙값(배경 밝기)에서 더 멀리 떨어진 쪽 극단이 획이다.
    # (반대쪽 top-hat 응답은 획 '주변 배경'에서 커지므로 Otsu 임계값 비교로는 고를 수 없음)
    lo, med, hi = np.percentile(gray, [2, 50, 98])
    return otsu(dark)[0] if med - lo >= hi - med else otsu(light)[0]


# ---------------------------------------------------------------------------
# φ₁
# ---------------------------------------------------------------------------
def _clean_mask(mask, min_object_area, hole_area):
    mask = remove_small_objects(mask, max_size=max(1, min_object_area - 1))
    mask = remove_small_holes(mask, max_size=max(1, hole_area - 1))
    # 테두리 요철 다듬기 -> 골격 가지(spur) 감소
    m = mask.astype(np.uint8)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, se)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, se)
    return m.astype(bool)


def _skeleton_special_points(skel):
    """골격 8-이웃 수: 1 = 끝점, 3 이상 = 교차점."""
    nb = cv2.filter2D(skel.astype(np.uint8), -1, np.ones((3, 3), np.float32),
                      borderType=cv2.BORDER_CONSTANT) - skel.astype(np.uint8)
    ends = skel & (nb == 1)
    junctions = skel & (nb >= 3)
    return ends, junctions


def _prune_spurs(skel, dist, factor=2.0, rounds=2):
    """골격 가지치기: 끝점에서 시작해 교차점에서 끝나는 짧은 가지(길이 < factor × 중앙 반폭)를 지운다.

    모서리나 테두리 요철에서 뻗은 가지는 D 가 작아 W 분포에 가짜 '가는 획' 을 만든다.
    끝점 주변 제외 반경은 그 점의 D 로 정해서, D 가 작은 가지 자체는 걸러 내지 못한다.
    """
    half_width = np.median(dist[skel]) if skel.any() else 0
    for _ in range(rounds):
        ends, junctions = _skeleton_special_points(skel)
        if not junctions.any() or not ends.any():
            break
        near_junction = cv2.dilate(junctions.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        n, lab = cv2.connectedComponents((skel & ~near_junction).astype(np.uint8), connectivity=8)
        length = np.bincount(lab[skel & ~near_junction], minlength=n)
        terminal = np.bincount(lab[ends], minlength=n) > 0
        spur = terminal & (length < factor * half_width)
        spur[0] = False
        if not spur.any():
            break
        skel = skel & ~spur[lab]
        # 가지가 붙어 있던 교차점 픽셀만 남는 경우 정리를 위해 다시 골격화
        skel = skeletonize(skel)
    return skel


def stroke_thickness_variation(crop, alpha=ALPHA1, polarity="auto", trim=0.05,
                               exclude_ends=True, exclude_junctions=True, exclusion_factor=1.5,
                               min_width_px=6.0, max_upscale=4, min_samples=30):
    """단일 크롭 이미지의 φ₁ = tanh(α · σ_W/μ_W).

    crop               BGR/그레이 영상 또는 이진 마스크 (글자 한 개 또는 한 줄)
    alpha              α₁
    polarity           "auto" | "dark" | "light" (binarize_strokes 참고)
    trim               W 분포 양쪽에서 잘라낼 비율 (0 이면 정의 그대로의 평균/표준편차)
    exclude_ends       골격 끝점 주변 제외 (반경 = exclusion_factor × 그 점의 D)
    exclude_junctions  골격 교차점 주변 제외
    min_width_px       평균 선폭이 이보다 가늘면 확대 후 다시 계산 (화소 양자화 완화)
    max_upscale        최대 확대 배율
    min_samples        통계에 쓸 최소 골격점 수 (미만이면 valid=False)

    선폭(μ_W, σ_W)은 확대 여부와 관계없이 입력 이미지 px 기준으로 돌려준다. CV, φ₁ 은 배율과 무관.
    """
    mask = binarize_strokes(crop, polarity)
    if mask.sum() == 0:
        return StrokeThickness(np.nan, np.nan, np.nan, np.nan, 0, False, "획 픽셀 없음")

    scale = 1
    result = None
    while True:
        result = _phi_from_mask(mask, scale, alpha, trim, exclude_ends, exclude_junctions,
                                exclusion_factor, min_samples)
        too_thin = np.isfinite(result.mu_w) and result.mu_w * scale < min_width_px
        if not too_thin or scale >= max_upscale:
            break
        # 가는 획: 원본을 확대해 다시 이진화 (이진 마스크를 확대하면 계단이 그대로 커질 뿐이라 원본 사용)
        scale = min(max_upscale, int(np.ceil(min_width_px / max(result.mu_w * scale, 1e-6))) * scale)
        src = np.asarray(crop)
        if src.dtype == bool or len(np.unique(src)) <= 2:
            # 이진/2단계 영상: 획 마스크를 먼저 구한 뒤 부드럽게 확대 (계단 완화)
            base = binarize_strokes(src, polarity).astype(np.uint8) * 255
            up = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            up = cv2.GaussianBlur(up, (0, 0), scale / 2)
            mask = up > 127
        else:
            up = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            mask = binarize_strokes(up, polarity)
    result.debug["scale"] = scale
    return result


def _phi_from_mask(mask, scale, alpha, trim, exclude_ends, exclude_junctions,
                   exclusion_factor, min_samples):
    area = mask.sum()
    mask = _clean_mask(mask, min_object_area=max(4, int(area * 0.01)), hole_area=max(4, 4 * scale * scale))

    # D: 가장자리 0 패딩 (획이 크롭 경계에 닿아도 배경까지 거리를 제대로 재게)
    padded = np.pad(mask, 1).astype(np.uint8)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]

    skel = _prune_spurs(skeletonize(mask), dist)
    keep = skel.copy()

    # 잉크가 번져 거의 막힌 속공간(예: 두꺼운 '0', '8', 'B' 의 작은 구멍) 옆 골격점은 D 가
    # 획 바깥이 아니라 작은 구멍까지의 거리로 잡혀 선폭이 과소 추정된다.
    # 획 폭²(×2) 보다 작은 구멍을 메운 거리 변환과 비교해 크게 달라지는 점은 뺀다.
    if skel.any():
        typical_w = 2 * np.median(dist[skel])
        filled = remove_small_holes(mask, max_size=max(1, int(2 * typical_w ** 2)))
        if (filled != mask).any():
            dist_filled = cv2.distanceTransform(np.pad(filled, 1).astype(np.uint8), cv2.DIST_L2,
                                                cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
            near_small_hole = dist_filled > 1.3 * dist + 0.5
            if (keep & ~near_small_hole).sum() >= min_samples:
                keep &= ~near_small_hole
    if exclude_ends or exclude_junctions:
        ends, junctions = _skeleton_special_points(skel)
        special = (ends if exclude_ends else 0) | (junctions if exclude_junctions else 0)
        special = np.asarray(special, bool)
        if special.any():
            # 각 골격점에서 가장 가까운 끝점/교차점까지 거리 < factor·D(p) 이면 제외
            to_special = cv2.distanceTransform((~special).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            near = to_special < exclusion_factor * np.maximum(dist, 1.0)
            keep &= ~near
            if keep.sum() < min_samples:      # 짧은 획(점, 짧은 '-')은 제외 없이
                keep = skel.copy()

    widths = 2.0 * dist[keep]
    debug = {"mask": mask, "skeleton": skel, "used": keep}
    if widths.size < min_samples:
        return StrokeThickness(np.nan, np.nan, np.nan, np.nan, int(widths.size), False,
                               f"골격점 {widths.size}개 < {min_samples}", debug)
    if trim > 0:
        lo, hi = np.quantile(widths, [trim, 1 - trim])
        widths = widths[(widths >= lo) & (widths <= hi)]
    mu = float(widths.mean())
    sigma = float(widths.std())          # 모집단 표준편차 (정의의 1/N)
    cv_ = sigma / mu if mu > 0 else np.nan
    return StrokeThickness(float(np.tanh(alpha * cv_)), cv_, mu / scale, sigma / scale,
                           int(widths.size), True, "", debug)


# ---------------------------------------------------------------------------
# 합성 시험 데이터
# ---------------------------------------------------------------------------
def render_printed(text, size=(120, 420), thickness=8, scale=2.2, font=cv2.FONT_HERSHEY_SIMPLEX):
    """기계식 마킹 흉내: 일정한 굵기의 글꼴 (밝은 바탕, 어두운 획)."""
    img = np.full(size, 200, np.uint8)
    cv2.putText(img, text, (10, size[0] - 30), font, scale, 40, thickness, cv2.LINE_AA)
    return img


def render_handwritten(text, size=(120, 420), seed=0, base=8, swing=0.6):
    """수기 흉내: 획을 따라 굵기가 필압처럼 변하는 글자 (짧은 원판을 촘촘히 찍어 그림)."""
    rng = np.random.default_rng(seed)
    canvas = np.zeros((size[0] * 2, size[1] * 2), np.uint8)
    cv2.putText(canvas, text, (20, size[0] * 2 - 60), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 4.4, 255, 2, cv2.LINE_AA)
    path = np.column_stack(np.nonzero(canvas > 127))      # 글자 중심선 근사 (y, x)
    # 필압: 획 하나 길이(약 10~20px) 정도로 매끈하게 변하는 무작위 장 (방향과 무관)
    pressure = cv2.GaussianBlur(rng.normal(size=size).astype(np.float32), (0, 0), 6)
    pressure /= pressure.std() + 1e-6
    img = np.full(size, 200, np.uint8)
    for y, x in path[::2]:
        yy, xx = int(y / 2), int(x / 2)
        r = base / 2 * (1 + swing * np.clip(pressure[yy, xx], -1.5, 1.5) / 1.5)
        cv2.circle(img, (xx, yy), max(1, int(round(r))), 40, -1, cv2.LINE_AA)
    return img


if __name__ == "__main__":
    for name, img in [("printed  B12-SP3", render_printed("B12-SP3")),
                      ("printed  4500", render_printed("4500")),
                      ("handwrit B12-SP3", render_handwritten("B12-SP3", seed=1)),
                      ("handwrit 4500", render_handwritten("4500", seed=2))]:
        r = stroke_thickness_variation(img)
        print(f"{name:18s} φ1={r.phi1:.3f}  CV={r.cv:.3f}  μW={r.mu_w:.1f}px  σW={r.sigma_w:.2f}  N={r.n_samples}")
