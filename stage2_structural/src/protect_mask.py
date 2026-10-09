# -*- coding: utf-8 -*-
"""
protect_mask.py
2차 공통 모듈 ①: 문자 보호 마스크  (스크래치·녹·얼룩 검출기가 모두 같이 사용)

출력은 3층이다.
  strong  확실한 보호   → 어떤 검출기도 절대 지우지 않음
  suspect 의심 영역     → 지우지 않고 '불확실'로 표시 (3단 판정에서 사용)
  나머지               → 각 검출기가 노이즈 여부를 판정할 수 있는 영역

근거 구분
  [논문 3]  Huang et al. 2018: 문자 보호 마스크를 두고 가는 선 노이즈를 제거한다는 구조 (초록 수준 확인)
  [논문 4]  Chen et al. 2016: 불확실하면 지우지 말고 표시한다는 원칙 (원문 확인)
  [논문 9]  Sauvola & Pietikäinen 2000: 국소 평균·표준편차 기반 적응 이진화
            T = m · (1 + k · (s / R − 1))  — 원문 미열람, scikit-image 문서의 공식을 따름
  [우리 선택] 아래 PROTECT_CFG의 모든 값, 선(線) 분리 방법, 덩어리 분류 규칙, 흐린 획 검출 방법

처리 순서
  1. 밝기(L) 채널에서 마킹 방향(polarity)에 맞게 '마킹이 밝게 보이는' 영상 F를 만든다
  2. Sauvola로 획 후보 C를 찾는다                                   [논문 9]
  3. C에서 긴 직선(허프 변환)을 분리하고, 폭으로 가는 선/굵은 선을 나눈다   [우리 선택]
       가는 선 → 보호하지 않음(스크래치 후보) / 굵은 선 → 의심 영역
  4. 나머지 덩어리를 '글자다움'으로 분류한다                          [우리 선택]
       여러 덩어리가 모여 있고 길쭉하지 않음 → 확실한 보호 / 애매함 → 의심 영역
  5. 흐린 획 후보: 배경 대비 밝기가 잡음 수준의 3배 이상인 곳 → 의심 영역   [우리 선택]
     (Sauvola가 놓치는 흐린 글씨 대비. 단, 길고 곧은 단독 선은 제외)
  6. 확실한 보호 주변 여백, 선과 글자가 만나는 곳, 글자에 딸린 가는 선(화살표 등) → 의심 영역 [우리 선택]
  7. 정형 문자 박스가 주어지면 박스 전체를 확실한 보호에 추가
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

PROTECT_CFG = {
    # 마킹 방향: "bright" = 배경보다 밝은 마킹(흰·노랑 페인트, 분필), "dark" = 어두운 마킹(검은 마커)
    "polarity": "bright",
    # 아래 픽셀 단위 값은 긴 변 1600px 기준이며 이미지 크기에 비례해 자동 조정 (1차의 크기 보정과 같은 방식)
    "ref_long_side": 1600,
    # Sauvola [논문 9의 형태, 값은 우리 설정]
    "sauvola_window": 51, "sauvola_k": 0.2, "sauvola_R": 128.0,
    "min_component_area": 30,
    # 선 분리 (허프 변환) [우리 설정]
    "line_min_length": 120, "line_max_gap": 6, "hough_threshold": 60,
    "thin_line_ratio": 0.6,       # 선 폭 ≤ 글자 획 폭 × 이 값 → '가는 선'
    "thin_line_abs_px": 3.0,      # 또는 선 폭 ≤ 이 값 → '가는 선'
    "attach_radius": 100,         # 글자 무리에서 이 거리 안을 '글자 주변'으로 봄
    "attach_frac": 0.95,          # 가는 선 길이의 이 비율 이상이 글자 주변에 있으면 '글자에 딸린 선'(화살표 등) → 의심 영역
    # 글자다움 [우리 설정]
    "group_radius": 25,           # 이 거리 안의 덩어리들을 한 무리로 봄
    "text_min_group": 2,          # 무리 안 덩어리 수가 이 이상이면 글자답다고 봄
    "elong_line": 6.0,            # 긴 변/짧은 변이 이 이상이면 '선 같은 덩어리'
    # 흐린 획 [우리 설정]
    "weak_blur_sigma": 1.5, "weak_bg_median": 61, "weak_k_sigma": 3.0, "weak_dilate": 5,
    # 여백 [우리 설정]
    "strong_margin": 3, "box_pad": 5,
}


@dataclass
class ProtectResult:
    strong: np.ndarray            # 확실한 보호 (bool)
    suspect: np.ndarray           # 의심 영역 (bool), strong과 겹치지 않음
    thin_lines: np.ndarray        # 보호하지 않은 가는 직선 (스크래치 후보, 참고용)
    thick_lines: np.ndarray       # 굵은 직선 (의심 영역에 포함됨, 참고용)
    weak: np.ndarray              # 흐린 획 후보 (의심 영역에 포함됨, 참고용)
    info: dict = field(default_factory=dict)

    @property
    def protected(self):
        """지우면 안 되는 전체 영역 = 확실한 보호 ∪ 의심 영역"""
        return self.strong | self.suspect


# ------------------------------------------------------------------
def scaled(cfg, shape):
    s = max(shape[:2]) / cfg["ref_long_side"]
    c = dict(cfg)
    for k in ("sauvola_window", "weak_bg_median"):
        v = max(3, int(round(cfg[k] * s)))
        c[k] = v if v % 2 == 1 else v + 1
    for k in ("line_min_length", "line_max_gap", "group_radius", "attach_radius", "weak_dilate", "strong_margin", "box_pad",
              "min_component_area"):
        p = 2 if k == "min_component_area" else 1
        c[k] = max(1, int(round(cfg[k] * (s ** p))))
    c["hough_threshold"] = max(10, int(round(cfg["hough_threshold"] * s)))
    c["_scale"] = s
    return c


def foreground_image(img_bgr, polarity):
    """마킹이 '밝게' 보이도록 만든 밝기 영상 F (float32)."""
    L = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    if polarity == "bright":
        return L
    if polarity == "dark":
        return 255.0 - L
    raise ValueError("polarity는 'bright' 또는 'dark'")


def sauvola_foreground(F, window, k, R):
    """Sauvola [논문 9]: 어두운 글자 기준 공식이므로 D = 255 − F 에 적용한다.
    T = m · (1 + k · (s/R − 1)), 글자 = D < T"""
    D = 255.0 - F
    m = cv2.boxFilter(D, cv2.CV_64F, (window, window), borderType=cv2.BORDER_REFLECT)
    m2 = cv2.boxFilter(D * D, cv2.CV_64F, (window, window), borderType=cv2.BORDER_REFLECT)
    s = np.sqrt(np.maximum(m2 - m * m, 0))
    T = m * (1 + k * (s / R - 1))
    return D < T


def remove_small(mask, min_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def stroke_width_estimate(mask):
    """대표 획 폭 = 거리 변환 90번째 백분위 × 2 (생성기·평가와 같은 측정 방법)."""
    if not mask.any():
        return 0.0
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return float(2 * np.percentile(dt[mask], 90))


def split_lines(C, cfg):
    """C에서 긴 직선 선분을 찾아 선 마스크로 분리. 각 선의 폭도 추정. [우리 선택]"""
    segs = cv2.HoughLinesP(C.astype(np.uint8) * 255, 1, np.pi / 180, cfg["hough_threshold"],
                           minLineLength=cfg["line_min_length"], maxLineGap=cfg["line_max_gap"])
    dt = cv2.distanceTransform(C.astype(np.uint8), cv2.DIST_L2, 5)
    lines = []
    if segs is None:
        return lines
    for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):   # OpenCV 4(N,1,4)·5(N,4) 모두 대응
        n = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
        xs = np.linspace(x1, x2, n).round().astype(int)
        ys = np.linspace(y1, y2, n).round().astype(int)
        vals = dt[ys, xs]
        vals = vals[vals > 0]
        width = float(2 * np.median(vals)) if len(vals) else 1.0
        lines.append(((int(x1), int(y1), int(x2), int(y2)), max(width, 1.0)))
    return lines


def draw_lines(shape, lines, extra=2):
    m = np.zeros(shape, np.uint8)
    for (x1, y1, x2, y2), w in lines:
        cv2.line(m, (x1, y1), (x2, y2), 255, int(round(w)) + extra)
    return m > 0


def classify_components(mask, cfg):
    """덩어리별 글자다움 판정. 반환: (글자다운 마스크, 애매한 마스크, 선 같은 단독 마스크)"""
    text_like = np.zeros(mask.shape, bool)
    ambiguous = np.zeros(mask.shape, bool)
    line_like = np.zeros(mask.shape, bool)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return text_like, ambiguous, line_like
    r = cfg["group_radius"]
    grown = cv2.dilate(mask.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    _, glab = cv2.connectedComponents(grown, connectivity=8)
    # 각 덩어리가 속한 무리 → 무리 안 덩어리 수
    comp_group = np.zeros(n, int)
    ys, xs = np.nonzero(lab)
    comp_group[lab[ys, xs]] = glab[ys, xs]
    group_count = np.bincount(comp_group[1:], minlength=glab.max() + 1)
    cls = np.zeros(n, np.uint8)        # 1=text, 2=ambiguous, 3=line
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < cfg["min_component_area"]:
            continue
        x, y, bw, bh = st[i, :4]
        yy, xx = np.nonzero(lab[y:y + bh, x:x + bw] == i)
        pts = np.column_stack([xx + x, yy + y]).astype(np.float32)
        (_, _), (a, b), _ = cv2.minAreaRect(pts)
        elong = max(a, b) / max(min(a, b), 1.0)
        grouped = group_count[comp_group[i]] >= cfg["text_min_group"]
        if elong >= cfg["elong_line"] and not grouped:
            cls[i] = 3
        elif grouped and elong < cfg["elong_line"]:
            cls[i] = 1
        else:
            cls[i] = 2
    text_like, ambiguous, line_like = cls[lab] == 1, cls[lab] == 2, cls[lab] == 3
    return text_like, ambiguous, line_like


# ------------------------------------------------------------------
def build_protect_mask(img_bgr, protect_boxes=None, cfg=None):
    cfg = scaled({**PROTECT_CFG, **(cfg or {})}, img_bgr.shape)
    h, w = img_bgr.shape[:2]
    F = foreground_image(img_bgr, cfg["polarity"])

    # 2. Sauvola 획 후보 [논문 9]
    C = sauvola_foreground(F, cfg["sauvola_window"], cfg["sauvola_k"], cfg["sauvola_R"])
    C = remove_small(C, cfg["min_component_area"])

    # 3. 긴 직선 분리 [우리 선택]
    lines = split_lines(C, cfg)
    line_area = draw_lines((h, w), lines) & C
    rest = C & ~line_area
    text_w = stroke_width_estimate(remove_small(rest, cfg["min_component_area"])) or 1.0
    thin_thr = max(cfg["thin_line_abs_px"], cfg["thin_line_ratio"] * text_w)
    thin = [l for l in lines if l[1] <= thin_thr]
    thick = [l for l in lines if l[1] > thin_thr]
    thin_lines = draw_lines((h, w), thin) & C
    thick_lines = draw_lines((h, w), thick) & C

    # 4. 덩어리 분류 [우리 선택]
    text_like, ambiguous, line_like = classify_components(rest, cfg)
    strong = text_like.copy()

    # 7. 정형 문자 박스 [우리 설계: 파이프라인 3단계 결과 활용]
    if protect_boxes:
        p = cfg["box_pad"]
        for x1, y1, x2, y2 in protect_boxes:
            strong[max(0, y1 - p):min(h, y2 + p), max(0, x1 - p):min(w, x2 + p)] = True

    # 5. 흐린 획 후보 [우리 선택]
    L8 = np.clip(F, 0, 255).astype(np.uint8)
    smooth = cv2.GaussianBlur(F, (0, 0), cfg["weak_blur_sigma"])
    contrast = smooth - cv2.medianBlur(L8, cfg["weak_bg_median"]).astype(np.float32)
    noise = 1.4826 * float(np.median(np.abs(contrast - np.median(contrast)))) + 1e-6
    weak = contrast > cfg["weak_k_sigma"] * noise
    weak = cv2.morphologyEx(weak.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    weak &= ~cv2.dilate((thin_lines | line_like).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    w_text, w_amb, _ = classify_components(remove_small(weak, cfg["min_component_area"]), cfg)
    weak = w_text | w_amb
    k = cfg["weak_dilate"]
    weak = cv2.dilate(weak.astype(np.uint8), np.ones((k, k), np.uint8)) > 0

    # 4-1. 글자에 딸린 가는 선(화살표·밑줄 등)은 보호하지 않으면 안 됨 → 의심 영역 [우리 선택]
    #      스크래치는 글자를 가로질러 멀리 뻗지만, 기호의 선은 대부분 글자 주변에 머문다는 가정
    a = cfg["attach_radius"]
    text_zone = cv2.dilate((text_like | strong).astype(np.uint8), np.ones((2 * a + 1, 2 * a + 1), np.uint8)) > 0
    attached, free = [], []
    for (x1, y1, x2, y2), wd in thin:
        n = int(max(abs(x2 - x1), abs(y2 - y1))) + 1
        xs = np.linspace(x1, x2, n).round().astype(int)
        ys = np.linspace(y1, y2, n).round().astype(int)
        (attached if text_zone[ys, xs].mean() >= cfg["attach_frac"] else free).append(((x1, y1, x2, y2), wd))
    attached_lines = draw_lines((h, w), attached) & C
    thin_lines = draw_lines((h, w), free) & C & ~attached_lines

    # 6. 여백과 교차점 [우리 선택]
    m = cfg["strong_margin"]
    ring = cv2.dilate(strong.astype(np.uint8), np.ones((2 * m + 1, 2 * m + 1), np.uint8)) > 0
    crossings = thin_lines & cv2.dilate(text_like.astype(np.uint8), np.ones((2 * m + 1, 2 * m + 1), np.uint8)).astype(bool)

    suspect = (ambiguous | thick_lines | attached_lines | weak | ring | crossings) & ~strong

    info = {
        "polarity": cfg["polarity"], "scale": round(cfg["_scale"], 3),
        "text_stroke_width_px": round(text_w, 2), "thin_line_threshold_px": round(thin_thr, 2),
        "n_lines": len(lines), "n_thin_lines": len(free), "n_attached_lines": len(attached),
        "n_thick_lines": len(thick),
        "noise_sigma": round(noise, 3),
        "area_ratio": {"strong": round(float(strong.mean()), 5), "suspect": round(float(suspect.mean()), 5)},
    }
    return ProtectResult(strong=strong, suspect=suspect, thin_lines=thin_lines, thick_lines=thick_lines,
                         weak=weak, info=info)
