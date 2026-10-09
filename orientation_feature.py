"""
수기(Handwritten) vs 기계식 마킹(Printed) 판별 특징 - φ₃ 기울기 변화 (Orientation Variation)

정의
  글자 단위 연결 요소(CC) k = 1..M 의 2차 중심 모멘트 μ₂₀, μ₀₂, μ₁₁ 로 주축 각도
      θ_k = ½ · arctan2(2μ₁₁, μ₂₀ − μ₀₂)            θ_k ∈ [−π/2, π/2]
  각도의 퍼짐 σ_θ 를 [0, 1] 로 정규화
      φ₃ = tanh(α₃ · σ_θ)        또는   φ₃ = min(1, σ_θ / θ_max)

구현상 결정 (이유)
  1. σ_θ 는 '축 방향 원형 표준편차' 로 구한다.
     주축은 방향이 아니라 축이라 +89° 와 −89° 는 2° 차이다. 일반 표준편차는 이를 178° 로 본다.
     각도를 2배 해 단위원에 올린 평균 합성벡터 길이 R 로  σ_θ = ½ · sqrt(−2 ln R).
  2. 편심률이 낮은(원·정사각형에 가까운) 글자는 주축이 불안정 -> 기준 미만 제외,
     나머지는 이방성 (λ₁−λ₂)/(λ₁+λ₂) 로 가중.
  3. 인쇄체도 글자 모양마다 주축이 다르다 ('1' 세로, '-' 가로, '7' 비스듬).
     문장부호(높이가 중앙값의 절반 미만)는 각도 통계에서 뺀다. 그래도 인쇄체 σ_θ 는 0 이 아니다.
  4. 잉크젯 도트/스텐실처럼 글자가 여러 조각으로 끊긴 경우: 조각이 많고 작으면 이웃 간격만큼
     닫힘 연산으로 글자 단위로 붙인 뒤 계산한다 (조각 하나하나의 각도는 의미가 없음).
"""
import math

import cv2
import numpy as np
from scipy.spatial import cKDTree

# 정규화
ALPHA3 = 0.07            # tanh 기울기 (1/deg): φ₃=0.5 ≈ σθ 7.8°. 합성 데이터(robust) 인쇄체 중앙 5.5~6°, 수기 8.5~11°. 현장 데이터로 재보정 필요
THETA_MAX_DEG = 30.0     # 선형 정규화 상한 (method="linear")

# 노이즈 / 글자 판정
MIN_AREA_RATIO = 0.15    # 글자 후보 중앙 면적 대비 이보다 작으면 노이즈 (녹 반점, 블라스트 요철)
MAX_LENGTH_RATIO = 2.5   # 주축 길이가 중앙 글자 높이의 이 배수보다 길면 긁힘/선 노이즈
MAX_ELONGATION = 12.0    # 주축/부축 비가 이보다 크면 긁힘/선 노이즈 ('1', 'I' 는 보통 3~8)
PUNCT_HEIGHT_RATIO = 0.5 # 높이가 중앙값의 절반 미만이면 문장부호 -> 각도 통계 제외
MIN_ECCENTRICITY = 0.5   # 편심률 e = sqrt(1 − λ₂/λ₁) 이 이보다 작으면 주축 불안정 -> 제외
FRAGMENT_MIN_COUNT = 8   # 조각이 이만큼 이상이고
FRAGMENT_SIZE_RATIO = 0.2  # 조각 크기 중앙값이 크롭 높이의 이 비율보다 작고
FRAGMENT_FILL_MIN = 0.6    # 상자 채움 중앙값이 이 이상(도트처럼 꽉 찬 모양, 원 ≈ 0.78)이고
FRAGMENT_AREA_CV = 0.35    # 조각 면적이 고르고 (변동계수 < 이 값)
FRAGMENT_SOLIDITY_MIN = 0.9  # 볼록도 중앙값이 이 이상(도트 ≈ 1)이면 '끊긴 글자' 로 보고 병합


# ---------------------------------------------------------------------------
# 입력 / 이진화
# ---------------------------------------------------------------------------
def load_image(image_path_or_array):
    """경로(한글 경로 포함) 또는 배열 -> BGR/그레이 uint8."""
    if isinstance(image_path_or_array, (str, bytes)) or hasattr(image_path_or_array, "__fspath__"):
        data = np.fromfile(image_path_or_array, np.uint8)   # cv2.imread 는 윈도우 한글 경로를 못 읽음
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {image_path_or_array}")
        return img
    img = np.asarray(image_path_or_array)
    if img.dtype == bool:
        return img.astype(np.uint8) * 255
    if img.dtype != np.uint8:
        img = cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def _to_dark_strokes(gray):
    """획이 어둡게 오도록 정렬. 배경이 다수이므로 중앙값에서 더 먼 쪽 극단이 획."""
    lo, med, hi = np.percentile(gray, [2, 50, 98])
    return gray if med - lo >= hi - med else 255 - gray


def sauvola_threshold(gray, window, k=0.2, r=128.0):
    """Sauvola: T = m · (1 + k·(s/R − 1)). 국소 평균 m, 표준편차 s 는 박스 필터(적분 영상)로 O(1)."""
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, -1, (window, window), borderType=cv2.BORDER_REFLECT)
    sq = cv2.boxFilter(g * g, -1, (window, window), borderType=cv2.BORDER_REFLECT)
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    return mean * (1 + k * (std / r - 1))


def binarize(img, method="sauvola"):
    """획 마스크 (bool). method: "sauvola"(조명 불균일·반사광에 강함) | "otsu" | "both"(두 결과의 교집합)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    values = np.unique(gray)
    if len(values) == 1:
        return np.zeros(gray.shape, bool)
    if len(values) == 2:                      # 이미 이진: 소수 쪽이 획
        lo_mask = gray == values[0]
        return lo_mask if lo_mask.sum() <= (~lo_mask).sum() else ~lo_mask
    dark = cv2.GaussianBlur(_to_dark_strokes(gray), (3, 3), 0)
    _, otsu = cv2.threshold(dark, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    otsu = otsu > 0
    if method == "otsu":
        return otsu
    window = max(15, (min(gray.shape) // 3) | 1)
    sauvola = dark < sauvola_threshold(dark, window)
    return sauvola & otsu if method == "both" else sauvola


# ---------------------------------------------------------------------------
# 글자 단위 성분
# ---------------------------------------------------------------------------
def _component_stats(mask):
    """연결 성분별 면적, 무게중심, 2차 중심 모멘트 (벡터화, bincount)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    ys, xs = np.nonzero(labels)
    lab = labels[ys, xs]
    xs, ys = xs.astype(np.float64), ys.astype(np.float64)
    area = np.bincount(lab, minlength=n).astype(np.float64)
    safe = np.maximum(area, 1)
    cx = np.bincount(lab, xs, n) / safe
    cy = np.bincount(lab, ys, n) / safe
    dx, dy = xs - cx[lab], ys - cy[lab]
    mu20 = np.bincount(lab, dx * dx, n)          # 정규화하지 않은 중심 모멘트 (cv2.moments 의 mu20 과 같음)
    mu02 = np.bincount(lab, dy * dy, n)
    mu11 = np.bincount(lab, dx * dy, n)
    return labels, stats, area, cx, cy, mu20, mu02, mu11


def _remove_specks(mask):
    """잉크 대부분(면적 상위 90%)을 이루는 성분의 중앙 면적보다 훨씬 작은 점(녹, 블라스트 요철)을 지운다.

    도트 마킹은 도트가 잉크 대부분이라 기준이 도트 크기가 되어 도트는 남는다.
    조각 병합 전에 해야 한다: 점 잡음이 많으면 '잘게 끊긴 글자' 로 오인해 글자 전체를 붙여 버린다.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1]
    cum = np.cumsum(areas[order])
    main = order[: np.searchsorted(cum, 0.9 * cum[-1]) + 1]
    keep = np.zeros(n, bool)
    keep[1:] = areas >= MIN_AREA_RATIO * np.median(areas[main])
    return keep[labels]


def _cut_thin_lines(mask):
    """획보다 훨씬 가는 선(긁힘, 마킹선 잔여)을 열림 연산으로 끊는다.

    가는 긁힘이 여러 글자를 가로지르면 글자들이 한 성분으로 이어져 통째로 '선 노이즈' 로 버려진다.
    열림 커널은 획 두께의 절반 정도로, 획이 가늘면(≤4px) 글자가 지워지지 않게 하지 않는다.
    """
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    inner = dist[dist > 0]
    if inner.size == 0:
        return mask
    stroke = 2 * np.percentile(inner, 90)      # 획 중심부 거리 ≈ 반폭
    k = int(stroke // 2)
    if k < 2:
        return mask
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k | 1, k | 1))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, se) > 0


def _merge_fragments(mask):
    """도트/스텐실처럼 글자가 잘게 끊겼으면 이웃 조각 간격만큼 닫아 글자 단위로 붙인다."""
    _, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    w, h, area = (stats[1:, i].astype(float) for i in (cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA))
    sizes = np.maximum(w, h)
    if len(sizes) < FRAGMENT_MIN_COUNT or np.median(sizes) >= FRAGMENT_SIZE_RATIO * mask.shape[0]:
        return mask, 0
    # 조각은 도트처럼 '작고 둥글고 꽉 찬' 모양이어야 한다. 크기 기준만 쓰면 여러 줄이 찍힌 사진에서
    # 진짜 글자(길쭉하거나 속이 빈 모양, 상자 채움 0.2~0.4)도 '작은 조각' 이 되어 글자끼리 덩어리로 붙는다.
    fill = area / np.maximum(w * h, 1)
    aspect = w / np.maximum(h, 1)
    if (np.median(fill) < FRAGMENT_FILL_MIN or not (0.6 <= np.median(aspect) <= 1.7)
            or area.std() / max(area.mean(), 1) > FRAGMENT_AREA_CV):
        return mask, 0
    # 볼록도(면적 / 볼록 껍질 면적): 도트는 ≈1, 굵은 글자도 구멍·오목한 곳이 있어 0.85 아래
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity = [cv2.contourArea(c) / max(cv2.contourArea(cv2.convexHull(c)), 1e-6) for c in contours
                if cv2.contourArea(c) > 0]
    if not solidity or np.median(solidity) < FRAGMENT_SOLIDITY_MIN:
        return mask, 0
    tree = cKDTree(centroids[1:])
    nn, _ = tree.query(centroids[1:], k=2)                     # 자기 자신 다음으로 가까운 조각
    k = int(round(np.median(nn[:, 1]))) | 1           # 도트 간격만큼만 (더 크면 이웃 글자까지 붙음)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, se) > 0, k


def axial_circular_mad(theta, weights=None):
    """축 방향 각도의 강건한 퍼짐 (라디안): 원형 중앙값 축으로부터 각도 차(축 주기 π 로 감음)의
    중앙 절대 편차 × 1.4826 (정규분포면 표준편차와 같아지는 배율).

    인쇄체도 'S', '2', '7' 처럼 모양 때문에 주축이 크게 다른 글자가 몇 개 섞여 표준편차를 키운다.
    중앙값 기반이라 이런 소수 글자에는 둔감하고, 글자 대부분이 제각각 기운 수기에는 반응한다.
    """
    theta = np.asarray(theta, np.float64)
    w = np.ones_like(theta) if weights is None else np.asarray(weights, np.float64)
    _, _, r = axial_circular_std(theta, w)
    # 중심도 강건하게: 평균은 이상치 쪽으로 끌려가므로, 가중 절대 편차 합이 최소인 표본 각도(원형 중앙값)
    dev_all = np.abs((theta[:, None] - theta[None, :] + np.pi / 2) % np.pi - np.pi / 2)   # (M, M), M 작음
    center = theta[np.argmin(dev_all @ w)]
    dev = np.abs((theta - center + np.pi / 2) % np.pi - np.pi / 2)
    return center, 1.4826 * float(np.median(dev)), r


def axial_circular_std(theta, weights=None):
    """축 방향 데이터의 원형 표준편차 (라디안). 각도를 2배 해 평균 합성벡터 길이 R 로 sqrt(−2 ln R)/2.

    (평균 축 각도, σ_θ, R) 반환.
    """
    theta = np.asarray(theta, np.float64)
    w = np.ones_like(theta) if weights is None else np.asarray(weights, np.float64)
    z = np.sum(w * np.exp(2j * theta)) / np.sum(w)
    r = min(abs(z), 1.0)
    mean = np.angle(z) / 2
    sigma = math.sqrt(-2 * math.log(r)) / 2 if r > 1e-12 else math.pi / 2
    return mean, sigma, r


# ---------------------------------------------------------------------------
# φ₃
# ---------------------------------------------------------------------------
def calculate_phi_3_orientation(image_path_or_array, alpha=ALPHA3, method="tanh",
                                theta_max_deg=THETA_MAX_DEG, binarization="sauvola",
                                min_eccentricity=MIN_ECCENTRICITY, spread="robust",
                                fallback=None) -> dict:
    """단일 크롭 이미지의 φ₃ (기울기 변화).

    spread  "circular": 가중 축 방향 원형 표준편차 (정의 그대로)
            "robust"  : 축 방향 원형 MAD × 1.4826 (기본, 글자 모양 이상치에 강건 - axial_circular_mad 참고)

    반환 dict
      phi_3            정규화 점수 [0, 1] (유효 글자 2개 미만이면 fallback, 기본 None)
      thetas           유효 글자들의 주축 각도 (도, 이미지 좌표: x축 기준, 아래쪽이 +y)
      num_valid_chars  각도 통계에 쓴 글자 수 M
      sigma_theta_deg  가중 축 방향 원형 표준편차 (도)
      mean_theta_deg   가중 평균 주축 각도 (도)
      weights          글자별 가중치 (이방성)
      components       성분별 상세 (시각화/디버깅용: 중심, 각도, 편심률, 사용 여부, 제외 이유)
      reason           phi_3 가 None 인 이유
    """
    img = load_image(image_path_or_array)
    mask = _remove_specks(binarize(img, binarization))
    mask, merge_k = _merge_fragments(mask)
    mask = _cut_thin_lines(mask)
    labels, stats, area, cx, cy, mu20, mu02, mu11 = _component_stats(mask)

    # 공분산 고유값 (λ₁ ≥ λ₂), 주축 길이 ≈ 4·sqrt(λ₁) (균일 분포 선분 L: 분산 L²/12 -> L = sqrt(12λ))
    safe = np.maximum(area, 1)
    a, b, c = mu20 / safe, mu02 / safe, mu11 / safe
    half = np.sqrt(((a - b) / 2) ** 2 + c ** 2)
    lam1, lam2 = (a + b) / 2 + half, np.maximum((a + b) / 2 - half, 0)
    length = np.sqrt(12 * lam1)
    eccentricity = np.sqrt(np.clip(1 - lam2 / np.maximum(lam1, 1e-12), 0, 1))
    anisotropy = np.where(lam1 + lam2 > 0, (lam1 - lam2) / np.maximum(lam1 + lam2, 1e-12), 0)
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    height = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float64)

    comps = []
    ids = np.arange(1, len(area))
    if len(ids):
        # 노이즈 필터: 면적(상대), 지나치게 길거나 가는 성분
        big = ids[area[ids] >= np.percentile(area[ids], 50) * 0.5] if len(ids) > 2 else ids
        median_area = np.median(area[big])
        median_len = np.median(length[big])
        for i in ids:
            reason = ""
            elong = math.sqrt(lam1[i] / max(lam2[i], 1e-12))
            bx, by, bw, bh = stats[i, :4]
            if area[i] < MIN_AREA_RATIO * median_area:
                reason = "noise:small"
            elif length[i] > MAX_LENGTH_RATIO * median_len or elong > MAX_ELONGATION:
                reason = "noise:line"
            elif bx == 0 or by == 0 or bx + bw >= mask.shape[1] or by + bh >= mask.shape[0]:
                reason = "border"          # 크롭 경계에 잘린 글자: 모양이 잘려 주축이 틀어짐
            comps.append({"id": int(i), "centroid": (float(cx[i]), float(cy[i])),
                          "theta_deg": float(np.degrees(theta[i])), "eccentricity": float(eccentricity[i]),
                          "weight": float(anisotropy[i]), "length": float(length[i]),
                          "height": float(height[i]), "box": tuple(int(v) for v in stats[i, :4]),
                          "used": False, "reason": reason})
        chars = [cp for cp in comps if not cp["reason"]]
        if chars:
            median_h = np.median([cp["height"] for cp in chars])
            for cp in chars:
                if cp["height"] < PUNCT_HEIGHT_RATIO * median_h:
                    cp["reason"] = "punctuation"
                elif cp["eccentricity"] < min_eccentricity:
                    cp["reason"] = "low_eccentricity"
                else:
                    cp["used"] = True

    used = [cp for cp in comps if cp["used"]]
    result = {"phi_3": fallback, "thetas": [cp["theta_deg"] for cp in used],
              "num_valid_chars": len(used), "sigma_theta_deg": None, "mean_theta_deg": None,
              "weights": [cp["weight"] for cp in used], "components": comps,
              "merge_kernel": merge_k, "mask": mask, "reason": ""}
    if len(used) < 2:
        result["reason"] = f"유효 글자 {len(used)}개 (2개 이상 필요)"
        return result

    stat = axial_circular_mad if spread == "robust" else axial_circular_std
    mean, sigma, r = stat(np.radians(result["thetas"]), result["weights"])
    sigma_deg = math.degrees(sigma)
    phi = math.tanh(alpha * sigma_deg) if method == "tanh" else min(1.0, sigma_deg / theta_max_deg)
    result.update(phi_3=float(phi), sigma_theta_deg=sigma_deg, mean_theta_deg=math.degrees(mean),
                  resultant_length=r)
    return result


# ---------------------------------------------------------------------------
# 디버깅 시각화
# ---------------------------------------------------------------------------
COLORS = {"used": (0, 200, 0), "low_eccentricity": (0, 200, 255), "punctuation": (255, 160, 0),
          "border": (200, 0, 200),
          "noise:small": (160, 160, 160), "noise:line": (0, 0, 255)}


def draw_orientation_debug(image_path_or_array, result, scale=None):
    """원본 위에 글자별 무게중심(점)과 주축(양방향 화살표)을 그린다.

    초록: 통계에 사용 / 주황: 편심률 낮음(제외) / 파랑: 문장부호 / 보라: 경계에 잘림
    회색: 작은 노이즈 / 빨강: 선 노이즈
    """
    img = load_image(image_path_or_array)
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    if scale is None:
        scale = max(1.0, 600 / max(canvas.shape[:2]))   # 작은 크롭은 키워서 보기 좋게
    if scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for cp in result["components"]:
        color = COLORS["used"] if cp["used"] else COLORS.get(cp["reason"], (128, 128, 128))
        x, y = cp["centroid"]
        x, y = x * scale, y * scale
        if cp["reason"].startswith("noise"):
            bx, by, bw, bh = (int(v * scale) for v in cp["box"])
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), color, 1)
            continue
        t = math.radians(cp["theta_deg"])
        half = 0.5 * cp["length"] * scale
        p0 = (int(x - half * math.cos(t)), int(y - half * math.sin(t)))
        p1 = (int(x + half * math.cos(t)), int(y + half * math.sin(t)))
        thick = max(1, int(round(scale)))
        cv2.arrowedLine(canvas, p0, p1, color, thick, cv2.LINE_AA, tipLength=0.15)
        cv2.arrowedLine(canvas, p1, p0, color, thick, cv2.LINE_AA, tipLength=0.15)
        cv2.circle(canvas, (int(x), int(y)), thick + 2, color, -1, cv2.LINE_AA)
        if cp["used"]:
            cv2.putText(canvas, f"{cp['theta_deg']:.0f}", (int(x) + 4, int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4 * thick, color, thick, cv2.LINE_AA)
    phi = result["phi_3"]
    text = (f"phi3={phi:.3f}  sigma={result['sigma_theta_deg']:.1f}deg  M={result['num_valid_chars']}"
            if phi is not None else f"phi3=None ({result['num_valid_chars']} valid)")
    cv2.rectangle(canvas, (0, 0), (12 + 9 * len(text), 24), (0, 0, 0), -1)
    cv2.putText(canvas, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# 합성 시험 데이터
# ---------------------------------------------------------------------------
def render_printed(text, angle=0.0, size=(140, 520), thickness=6, scale=2.2, dots=False):
    """기계식 마킹: 같은 글꼴, 글자 모두 같은 기울기 (줄 전체를 angle 만큼 회전). dots=True 면 잉크젯 도트."""
    img = np.full(size, 200, np.uint8)
    cv2.putText(img, text, (20, size[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, scale, 40, thickness, cv2.LINE_AA)
    if dots:
        ink = img < 120
        img = np.full(size, 200, np.uint8)
        pitch = 7
        for y in range(0, size[0], pitch):
            for x in range(0, size[1], pitch):
                if ink[y, x]:
                    cv2.circle(img, (x, y), 3, 40, -1, cv2.LINE_AA)
    M = cv2.getRotationMatrix2D((size[1] / 2, size[0] / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (size[1], size[0]), borderValue=200)


def render_handwritten(text, seed=0, tilt_std=10.0, size=(140, 520), thickness=6, baseline_jitter=8):
    """수기: 글자마다 기울기·크기·높이가 제각각 (글자를 따로 그려 각자 회전)."""
    rng = np.random.default_rng(seed)
    img = np.full(size, 200, np.uint8)
    x = 20
    for ch in text:
        if ch == " ":
            x += 30
            continue
        s = 2.2 * rng.uniform(0.85, 1.15)
        (w, h), _ = cv2.getTextSize(ch, cv2.FONT_HERSHEY_SIMPLEX, s, thickness)
        tile = np.full((h + 40, w + 40), 200, np.uint8)
        cv2.putText(tile, ch, (20, h + 20), cv2.FONT_HERSHEY_SIMPLEX, s, 40, thickness, cv2.LINE_AA)
        M = cv2.getRotationMatrix2D((tile.shape[1] / 2, tile.shape[0] / 2), rng.normal(0, tilt_std), 1.0)
        tile = cv2.warpAffine(tile, M, (tile.shape[1], tile.shape[0]), borderValue=200)
        y0 = int(size[0] - 40 - h - 20 + rng.integers(-baseline_jitter, baseline_jitter + 1))
        y0 = int(np.clip(y0, 0, size[0] - tile.shape[0]))
        if x + tile.shape[1] > size[1]:
            break
        region = img[y0:y0 + tile.shape[0], x:x + tile.shape[1]]
        np.minimum(region, tile, out=region)
        x += w + int(rng.integers(2, 10))
    return img


def add_plate_noise(img, seed=0):
    """녹 반점, 숏블라스트 요철, 반사광 얼룩, 긁힘."""
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32)
    out += rng.normal(0, 8, out.shape)                                   # 블라스트 요철
    h, w = out.shape
    yy, xx = np.mgrid[0:h, 0:w]
    out += 50 * np.exp(-((xx - w * 0.7) ** 2 + (yy - h * 0.3) ** 2) / (2 * (w * 0.12) ** 2))  # 반사광
    for _ in range(60):                                                  # 녹 반점
        cv2.circle(out, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(1, 3)), 70, -1)
    cv2.line(out, (10, int(h * 0.15)), (w - 10, int(h * 0.2)), 90, 1)    # 긁힘
    return np.clip(out, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="φ₃ 기울기 변화 계산 + 디버그 시각화")
    parser.add_argument("image", nargs="?", help="크롭 이미지 (없으면 합성 예시)")
    parser.add_argument("-o", "--output", default="phi3_debug.png")
    args = parser.parse_args()

    if args.image:
        samples = [(args.image, args.image)]
    else:
        samples = [("printed", add_plate_noise(render_printed("B12-SP3 4500"))),
                   ("printed 15deg", add_plate_noise(render_printed("B12-SP3 4500", angle=15))),
                   ("printed dots", add_plate_noise(render_printed("B12-SP3 4500", dots=True))),
                   ("handwritten", add_plate_noise(render_handwritten("B12-SP3 4500", seed=3)))]
    panels = []
    for name, src in samples:
        res = calculate_phi_3_orientation(src)
        sigma = res["sigma_theta_deg"]
        print(f"{name:15s} phi_3={res['phi_3'] if res['phi_3'] is None else round(res['phi_3'], 3)}  "
              f"sigma={sigma if sigma is None else round(sigma, 1)}deg  M={res['num_valid_chars']}  "
              f"thetas={[round(t) for t in res['thetas']]}")
        panels.append(draw_orientation_debug(src, res, scale=1.0))
    width = max(p.shape[1] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, 4, 0, width - p.shape[1], cv2.BORDER_CONSTANT) for p in panels]
    ok, buf = cv2.imencode(".png", np.vstack(panels))
    buf.tofile(args.output)
    print(f"시각화 저장: {args.output}")
