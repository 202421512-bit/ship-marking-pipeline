"""
수기 판별 목적함수 (2단계 계층형)

  0단계  (선택) 색상 전처리 (color_preprocess, color="auto" 등): 녹을 가리고 배경 대비 색 거리로 마킹을 부각한
         회색 영상을 만든다. 기본은 끔 (회색 영상 그대로) - 아래 color 인자 설명 참고.
  1단계  기계 패턴 사전 판별 (mechanical_pattern_detector): 도트 매트릭스·스텐실이 검출되면 φ 계산 없이
         인쇄체로 조기 확정.
         - 도트·스텐실은 획 기반 지표(φ₁·φ₅·φ₆·φ₇·φ₈)를 거꾸로 움직이므로 2단계에 들어가면 안 된다.
  2단계  8개 특징 φ₁~φ₈ 를 가중 합해 수기 점수 S_hand ∈ [0, 1] 과 신뢰도 C, 판정을 낸다.

  S_hand = Σ_{i∈유효} w_i φ_i / Σ_{i∈유효} w_i
    - 가중치 w 는 benchmark_and_optimize.py 가 데이터로 구해 objective_weights.json 에 저장한다
      (없으면 균등 가중치).
    - 동적 재분배: 글자가 1개뿐이라 φ₂(간격)·φ₄(기준선)·φ₆(크기)·φ₃(기울기) 가 None 이면 그 항을 빼고
      남은 지표의 가중치 합이 1 이 되게 비례 재할당한다.
  유착 가산점: 글자끼리 닿은 덩어리(touching_detector, has_touching_chars)는 폰트 메트릭스로 자간을 제어하는
  기계식에서는 나올 수 없으므로 수기의 강한 증거다. 1단계에서 기계 패턴이 확정되지 않았을 때만
  S ← S + b·(1 − S) 로 점수를 끌어올린다 (b: 접점 1개 TOUCH_BOOST, 2개 이상 TOUCH_BOOST_MULTI, 가는 연결선뿐이면 TOUCH_BOOST_BRIDGED).
  φ₆ 는 유착 덩어리를 분할 추정하고, φ₈ 은 겹침이 만든 X·T 접점을 그대로 쓴다.
  목적: 받은 영상의 표기가 수기인지 판별한다. 판정은 점수(와 1단계 패턴)로만 내리고 신뢰도에 묶지 않는다.
  반사광은 복원·보정하지 않는다. 대신 안전장치로, 글자 영역의 포화 픽셀(V > 245) 비율이 15% 를 넘으면
  획이 뭉개져 점수가 왜곡되므로 계산하지 않고 flag "REJECTED_SPECULAR_NOISE" 와 Uncertain 을 바로 돌려준다.
  큰 사진(휴대폰 원본 등)은 긴 변 WORK_LONG_SIDE 로 줄여서 분석한다. 영상 품질에 대한 신뢰도 평가는 이후 별도 단계에서
  하며, 그 단계에서 쓸 수 있도록 참고 신뢰도 C 를 함께 돌려준다:
  참고 신뢰도 C = C_contrast × C_mask × C_coverage ∈ [0, 1]
    - C_contrast: 배경-획 대비 잡음비 CNR = |배경 중앙값 − 획 중앙값| / 배경 표준편차 가 낮을수록 낮음
    - C_coverage: 계산된 지표의 가중치 비율 (지표가 많이 빠질수록 낮음)
  판정: S 가 두 임계값 사이(완충 구간)면 "Uncertain (Need Review)",
        S ≥ 수기 임계값이면 "Confirmed Handwritten", S ≤ 인쇄체 임계값이면 "Confirmed Printed".
"""
import json
import math
import os

import cv2
import numpy as np

from baseline_feature import calculate_phi_4_baseline
from color_preprocess import preprocess_color, text_region_penalty
from layout import analyze_layout
from connectivity_feature import calculate_phi_8_connectivity
from contour_roughness_feature import calculate_phi_7_roughness
from curvature_feature import calculate_phi_5_curvature
from mechanical_pattern_detector import detect_mechanical_pattern
from orientation_feature import _remove_specks, binarize, calculate_phi_3_orientation, load_image
from size_variance_feature import calculate_phi_6_size_variance
from spacing_feature import calculate_phi_2_spacing
from stroke_features import stroke_thickness_variation
from touching_detector import detect_touching_chars

FEATURES = ["phi_1", "phi_2", "phi_3", "phi_4", "phi_5", "phi_6", "phi_7", "phi_8"]
FEATURE_NAMES = {
    "phi_1": "stroke width", "phi_2": "spacing", "phi_3": "orientation", "phi_4": "baseline",
    "phi_5": "curvature", "phi_6": "size", "phi_7": "roughness", "phi_8": "connectivity",
}
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "objective_weights.json")

# 신뢰도 / 판정 (benchmark_and_optimize.py 가 임계값을 데이터로 정해 json 에 덮어씀)
CNR_LOW, CNR_HIGH = 2.0, 6.0
MASK_FULL = 0.3            # 글자 주변 녹 비율이 이 이상이면 C_mask = 0
DEFAULT_THRESHOLDS = {"printed": 0.45, "handwritten": 0.70}
SPECULAR_V = 245           # 포화(반사광) 픽셀: HSV V(회색이면 밝기) > 이 값
SPECULAR_REJECT = 0.15     # 글자 영역에서 포화 픽셀이 이 비율을 넘으면 점수 계산 없이 기각
TOUCH_BOOST = 0.5          # 유착 접점 1개: 남은 거리(1 − S)의 이 비율만큼 S 를 올림
TOUCH_BOOST_MULTI = 0.7    # 유착 접점 2개 이상 (글자 3개 이상이 이어짐 / 덩어리 여럿)
TOUCH_BOOST_BRIDGED = 0.3  # 접점이 모두 가는 연결선(bridged)뿐이면 약하게: 인쇄체의 하이픈이 양옆 글자에 닿는 경우를 감안
TOUCH_PRIOR = 0.5          # 지표를 하나도 못 구했을 때 가산점의 출발점
WORK_LONG_SIDE = 1280      # 분석 해상도 (긴 변). 큰 사진은 줄여서 계산 (지표들은 크기 불변으로 설계됨)


def load_calibration(path=WEIGHTS_PATH):
    """저장된 가중치·임계값. 없으면 균등 가중치와 기본 임계값."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data["weights"], data.get("thresholds", DEFAULT_THRESHOLDS)
    return {k: 1 / len(FEATURES) for k in FEATURES}, dict(DEFAULT_THRESHOLDS)


# ---------------------------------------------------------------------------
# 특징 추출
# ---------------------------------------------------------------------------
def _num(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v)


LINE_FEATURES = {"phi_2": calculate_phi_2_spacing, "phi_4": calculate_phi_4_baseline,
                 "phi_6": calculate_phi_6_size_variance}
LINE_MIN_CHARS = 3         # 줄 지표(간격·기준선·크기)에 필요한 최소 글자 수. 2글자면 기준선 오차가 항상 0,
                           # 간격이 하나뿐이라 '아주 규칙적 = 인쇄체' 로 거짓 판단됨 -> 계산 불가(None) 처리


def extract_features(image_path_or_array, use_layout=True):
    """φ₁~φ₈ (계산 불가면 None) 과 각 모듈의 원 결과.

    use_layout: 경계 성분(종이·판 가장자리)을 지우고 줄을 나눠, 줄 지표(φ₂·φ₄·φ₆)는 줄마다 계산해
                유효 글자 수로 가중 평균한다. 모양 지표(φ₁·φ₃·φ₅·φ₇·φ₈)는 정리된 전체 영상에서 계산.
    """
    img = load_image(image_path_or_array)
    if use_layout:
        lay = analyze_layout(img)
        clean = lay["gray"]
    else:
        lay, clean = None, img
    r1 = stroke_thickness_variation(clean)
    raw = {
        "phi_1": r1,
        "phi_3": calculate_phi_3_orientation(clean),
        "phi_5": calculate_phi_5_curvature(clean),
        "phi_7": calculate_phi_7_roughness(clean),
        "phi_8": calculate_phi_8_connectivity(clean),
    }
    phis = {"phi_1": _num(r1.phi1) if r1.valid else None}
    for k in ("phi_3", "phi_5", "phi_7", "phi_8"):
        phis[k] = _num(raw[k][k])

    # 줄 지표
    line_images = [ln["image"] for ln in lay["lines"]] if lay and lay["lines"] else [clean]
    for k, fn in LINE_FEATURES.items():
        vals, ws, per_line = [], [], []
        for li in line_images:
            r = fn(li)
            n = r.get("num_valid_chars", 0)
            v = _num(r[k])
            per_line.append(r)
            if v is not None and n >= LINE_MIN_CHARS:
                vals.append(v)
                ws.append(n)
        phis[k] = float(np.average(vals, weights=ws)) if vals else None
        raw[k] = {"per_line": per_line, k: phis[k], "lines_used": len(vals),
                  "reason": "" if vals else f"글자 {LINE_MIN_CHARS}개 이상인 줄 없음"}
    raw["layout"] = lay
    raw["touching"] = detect_touching_chars(_remove_specks(binarize(clean)))
    phis = {k: phis[k] for k in FEATURES}
    return phis, raw


def image_confidence(img, pre=None):
    """참고 신뢰도 (판정에는 쓰지 않음, 이후 신뢰도 평가 단계용). (C_contrast, C_mask, 상세) 반환.
    img 는 0단계를 거친 회색 영상, pre 는 preprocess_color 결과."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    ink = _remove_specks(binarize(img))
    detail = {"cnr": 0.0, "rust_near_text": 0.0}
    if ink.sum() < 20 or (~ink).sum() < 20:
        return 0.0, 1.0, detail
    background = ~cv2.dilate(ink.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)  # 획 테두리 제외
    bg = gray[background] if background.any() else gray[~ink]
    cnr = abs(float(np.median(bg)) - float(np.median(gray[ink]))) / (float(bg.std()) + 1e-6)
    c_contrast = float(np.clip((cnr - CNR_LOW) / (CNR_HIGH - CNR_LOW), 0, 1))
    rust = text_region_penalty(pre, ink) if pre is not None and pre["is_color"] else 0.0
    c_mask = float(np.clip(1 - rust / MASK_FULL, 0, 1))
    detail.update(cnr=cnr, rust_near_text=rust)
    return c_contrast, c_mask, detail


def text_region(work, pad_ratio=0.05):
    """글자 영역 상자 (x0, y0, x1, y1): 잉크 성분 중 영상 경계에 닿지 않는 것들의 외접 상자 + 여유."""
    ink = _remove_specks(binarize(work))
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    h, w = ink.shape
    keep = [i for i in range(1, n) if st[i, 0] > 0 and st[i, 1] > 0 and st[i, 0] + st[i, 2] < w and st[i, 1] + st[i, 3] < h]
    if not keep:
        keep = list(range(1, n))
    if not keep:
        return 0, 0, w, h
    x0 = min(st[i, 0] for i in keep)
    y0 = min(st[i, 1] for i in keep)
    x1 = max(st[i, 0] + st[i, 2] for i in keep)
    y1 = max(st[i, 1] + st[i, 3] for i in keep)
    px, py = int(pad_ratio * (x1 - x0)) + 2, int(pad_ratio * (y1 - y0)) + 2
    return max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)


def specular_ratio(img, box):
    """글자 영역 안에서 포화(반사광) 픽셀 비율. 컬러면 HSV V, 회색이면 밝기."""
    x0, y0, x1, y1 = box
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2] if img.ndim == 3 else img
    region = v[y0:y1, x0:x1]
    return float((region > SPECULAR_V).mean()) if region.size else 0.0


def to_work_resolution(img):
    scale = WORK_LONG_SIDE / max(img.shape[:2])
    if scale >= 1:
        return img, 1.0
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale


def combine(phis, weights):
    """동적 재분배 가중 합. (S, 실제 쓴 가중치, 유효 가중치 비율)"""
    avail = {k: w for k, w in weights.items() if phis.get(k) is not None and w > 0}
    total = sum(weights.values())
    if not avail:
        return None, {}, 0.0
    s = sum(avail.values())
    used = {k: w / s for k, w in avail.items()}
    return float(sum(used[k] * phis[k] for k in used)), used, float(s / total) if total > 0 else 0.0


def apply_touching_boost(score, touching):
    """유착이 검출됐으면 수기 점수에 가산점. 반환: (새 점수, 적용한 b). 유착 없으면 (score, 0.0)."""
    if not touching or not touching["has_touching_chars"]:
        return score, 0.0
    comps = touching.get("components", [])
    if comps and all(c["kind"] == "bridged" for c in comps):
        b = TOUCH_BOOST_BRIDGED
    else:
        b = TOUCH_BOOST_MULTI if touching["est_joints"] >= 2 else TOUCH_BOOST
    base = TOUCH_PRIOR if score is None else score
    return float(base + b * (1.0 - base)), b


def evaluate_handwritten_score(image_path_or_array, weights=None, thresholds=None, color="off",
                               early_exit=True) -> dict:
    """2단계 계층형 판정.

    weights     {"phi_1": w1, ...}. None 이면 objective_weights.json (없으면 균등)
    thresholds  {"printed": t_p, "handwritten": t_h}. None 이면 json 의 값
    color       0단계: "off"(기본, 회색 영상 그대로) | "auto" | "dark" | "white" | "yellow" (색상 전처리)
                반사광을 다루지 않는 조건의 합성 실험에서 회색 이진화가 색상 전처리보다 같거나 나았다
                (잉크 마스크 IoU: 검정 0.96 vs 0.94, 노랑 0.98 vs 0.92, 흰색 0.93 vs 0.94). 실제 녹 색이
                합성과 다를 수 있으니 현장 사진에서 다시 비교할 것.
    early_exit  1단계 기계 패턴 조기 확정 사용 여부

    반환 dict: stage(0 = 반사광 기각, 1 = 조기 확정, 2 = φ 점수), flags, specular_ratio, text_box,
              has_touching_chars / touching(유착 검출 결과) / score_before_touch / touch_boost(2단계만),
              score(S_hand: 유착 가산점 반영 후, 조기 확정이면 0, 기각이면 None), verdict(점수로만),
              confidence(참고 신뢰도, 판정에 쓰지 않음),
              mechanical(1단계 결과), features(φ, 1단계 확정이면 모두 None), weights_used, coverage,
              confidence_parts, preprocess(0단계 요약), raw
    """
    img, work_scale = to_work_resolution(load_image(image_path_or_array))
    cal_w, cal_t = load_calibration()
    weights = dict(weights or cal_w)
    thresholds = dict(thresholds or cal_t)

    # 0단계: 색상 전처리
    pre = preprocess_color(img, marking=color) if color != "off" else preprocess_color(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img)
    work = pre["enhanced"]
    c_contrast, c_mask, detail = image_confidence(work, pre)
    summary = {"is_color": pre["is_color"], "marking": pre["marking"]}
    box = text_region(work)
    spec = specular_ratio(img, box)
    flags = []
    if detail.get("cnr", 0.0) < CNR_LOW:
        flags.append("LOW_CONTRAST")                    # 경고만 (판정은 계속)
    base = {"thresholds": thresholds, "preprocess": summary, "work_image": work, "work_scale": work_scale,
            "text_box": box, "specular_ratio": spec, "flags": flags}

    # 안전장치: 글자 영역이 반사광으로 포화되면 점수를 계산하지 않고 기각
    if spec > SPECULAR_REJECT:
        flags.insert(0, "REJECTED_SPECULAR_NOISE")
        return {**base, "stage": 0, "score": None, "verdict": "Uncertain (Need Review)",
                "confidence": 0.0, "mechanical": None, "has_touching_chars": False, "touching": None, "features": {k: None for k in FEATURES},
                "weights_used": {}, "coverage": 0.0,
                "confidence_parts": {"contrast": c_contrast, "mask": c_mask, "coverage": 0.0, **detail}, "raw": {}}

    # 1단계: 기계 패턴 -> 인쇄체 조기 확정
    mech = detect_mechanical_pattern(work)
    if early_exit and mech["is_mechanical"]:
        return {**base, "stage": 1, "score": 0.0, "verdict": "Confirmed Printed",
                "confidence": float(c_contrast * c_mask), "mechanical": mech, "has_touching_chars": False, "touching": None,
                "features": {k: None for k in FEATURES}, "weights_used": {}, "coverage": 0.0,
                "confidence_parts": {"contrast": c_contrast, "mask": c_mask, "coverage": 1.0, **detail}, "raw": {}}

    # 2단계: φ 점수 -> 판정 (신뢰도와 무관하게 점수 임계값으로)
    phis, raw = extract_features(work)
    score, used, coverage = combine(phis, weights)
    touching = raw["touching"]
    score_raw = score
    score, boost = apply_touching_boost(score, touching)
    if boost:
        flags.append("TOUCHING_CHARS")
    if score is None:
        verdict = "Uncertain (Need Review)"
    elif score >= thresholds["handwritten"]:
        verdict = "Confirmed Handwritten"
    elif score <= thresholds["printed"]:
        verdict = "Confirmed Printed"
    else:
        verdict = "Uncertain (Need Review)"          # 완충 구간: 점수만으로는 확정하지 않음
    return {**base, "stage": 2, "score": score, "score_before_touch": score_raw, "touch_boost": boost,
            "has_touching_chars": touching["has_touching_chars"], "touching": touching, "verdict": verdict,
            "confidence": float(c_contrast * c_mask * coverage), "mechanical": mech, "features": phis,
            "weights_used": used, "coverage": coverage,
            "confidence_parts": {"contrast": c_contrast, "mask": c_mask, "coverage": coverage, **detail}, "raw": raw}


# ---------------------------------------------------------------------------
# 시각화: 원본 + 레이더 차트 + 지표별 막대
# ---------------------------------------------------------------------------
VERDICT_COLORS = {"Confirmed Handwritten": (40, 40, 220), "Confirmed Printed": (40, 160, 40),
                  "Uncertain (Need Review)": (0, 150, 230)}


def _radar(phis, used, size=400):
    img = np.full((size, size, 3), 255, np.uint8)
    c = (size // 2, size // 2 + 10)
    r = int(size * 0.33)
    n = len(FEATURES)
    ang = [-math.pi / 2 + 2 * math.pi * i / n for i in range(n)]
    for level in (0.25, 0.5, 0.75, 1.0):
        ring = [(int(c[0] + level * r * math.cos(a)), int(c[1] + level * r * math.sin(a))) for a in ang]
        cv2.polylines(img, [np.array(ring)], True, (215, 215, 215), 1, cv2.LINE_AA)
    pts = []
    for i, (k, a) in enumerate(zip(FEATURES, ang)):
        end = (int(c[0] + r * math.cos(a)), int(c[1] + r * math.sin(a)))
        cv2.line(img, c, end, (200, 200, 200), 1, cv2.LINE_AA)
        v = phis.get(k)
        label = f"{k.replace('phi_', 'p')} {FEATURE_NAMES[k]}"
        val = "n/a" if v is None else f"{v:.2f}"
        lx = int(c[0] + (r + 14) * math.cos(a)) - (len(label) * 3 if math.cos(a) < -0.1 else 0 if math.cos(a) > 0.1 else len(label) * 3 // 2)
        ly = int(c[1] + (r + 14) * math.sin(a)) + (10 if math.sin(a) > 0.1 else -2)
        lx = int(np.clip(lx, 2, size - 7 * len(label) - 2))      # 캔버스 밖으로 잘리지 않게
        color = (60, 60, 60) if v is not None else (170, 170, 170)
        cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        cv2.putText(img, val, (lx, ly + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        vv = 0.0 if v is None else v
        pts.append((int(c[0] + vv * r * math.cos(a)), int(c[1] + vv * r * math.sin(a))))
    overlay = img.copy()
    cv2.fillPoly(overlay, [np.array(pts)], (230, 160, 90))
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    cv2.polylines(img, [np.array(pts)], True, (200, 110, 40), 2, cv2.LINE_AA)
    for (px, py), k in zip(pts, FEATURES):
        cv2.circle(img, (px, py), 3 if phis.get(k) is not None else 2,
                   (200, 110, 40) if phis.get(k) is not None else (170, 170, 170), -1, cv2.LINE_AA)
    return img


def _bars(phis, used, width=400, height=250):
    img = np.full((height, width, 3), 255, np.uint8)
    n = len(FEATURES)
    bw = (width - 60) // n
    base = height - 40
    cv2.line(img, (40, base), (width - 10, base), (120, 120, 120), 1)
    for t in (0.5, 1.0):
        y = int(base - t * (height - 80))
        cv2.line(img, (40, y), (width - 10, y), (225, 225, 225), 1)
        cv2.putText(img, f"{t:.1f}", (8, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1, cv2.LINE_AA)
    for i, k in enumerate(FEATURES):
        x = 45 + i * bw
        v = phis.get(k)
        if v is None:
            cv2.putText(img, "n/a", (x + 2, base - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1, cv2.LINE_AA)
        else:
            top = int(base - v * (height - 80))
            cv2.rectangle(img, (x + 4, top), (x + bw - 6, base), (200, 110, 40), -1)
        w = used.get(k, 0.0)
        cv2.putText(img, f"p{k[-1]}", (x + 6, base + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(img, f"w{w:.2f}", (x + 1, base + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (60, 60, 60) if w > 0 else (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(img, "phi (bar) / weight used (w)", (40, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
    return img


def draw_objective_debug(image_path_or_array, result):
    """왼쪽: 원본 + 판정 띠, 오른쪽: 레이더 차트(위) + 지표별 φ 막대와 실제 쓴 가중치(아래)."""
    img = load_image(image_path_or_array)
    src = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    panel_h = 400 + 250
    scale = min(560 / src.shape[1], (panel_h - 80) / src.shape[0])
    src = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    left = np.full((panel_h, max(560, src.shape[1]), 3), 255, np.uint8)
    left[80:80 + src.shape[0], :src.shape[1]] = src
    color = VERDICT_COLORS[result["verdict"]]
    cv2.rectangle(left, (0, 0), (left.shape[1], 76), color, -1)
    s = "n/a" if result["score"] is None else f"{result['score']:.3f}"
    cv2.putText(left, result["verdict"], (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cp = result["confidence_parts"]
    stage = {0: f"REJECTED_SPECULAR_NOISE (saturated {result['specular_ratio']:.0%} of text area)",
             1: f"stage 1: {result['mechanical']['kind'] if result['mechanical'] else ''} pattern -> early exit",
             2: "stage 2: phi score"}[result["stage"]]
    cv2.putText(left, f"{stage}   S_hand={s}  C={result['confidence']:.2f}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(left, f"ref. confidence (not used for verdict): contrast {cp['contrast']:.2f} x rust mask {cp['mask']:.2f}"
                      f" x coverage {cp['coverage']:.2f}", (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                cv2.LINE_AA)
    right = np.vstack([_radar(result["features"], result["weights_used"]),
                       _bars(result["features"], result["weights_used"])])
    return np.hstack([left, right])


def _save(path, image):
    ok, buf = cv2.imencode(".png", image)
    buf.tofile(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="수기 판별 목적함수: 점수·신뢰도·판정 + 레이더 시각화")
    parser.add_argument("image", nargs="?", help="크롭 이미지 (없으면 합성 예시)")
    parser.add_argument("-o", "--output", default="objective_debug.png")
    args = parser.parse_args()
    if args.image:
        samples = [(args.image, args.image)]
    else:
        import synthetic_dataset as sd
        rng = np.random.default_rng(7)
        samples = [("printed", sd.make_sample(0, rng, "sans")[0]), ("handwritten", sd.make_sample(1, rng)[0])]
    panels = []
    for name, src in samples:
        res = evaluate_handwritten_score(src)
        feats = "  ".join(f"{k[-1]}:{'-' if v is None else f'{v:.2f}'}" for k, v in res["features"].items())
        s = "None" if res["score"] is None else f"{res['score']:.3f}"
        print(f"{name:12s} S={s}  C={res['confidence']:.2f}  {res['verdict']}  | {feats}")
        panels.append(draw_objective_debug(src, res))
    width = max(p.shape[1] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, 6, 0, width - p.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
              for p in panels]
    _save(args.output, np.vstack(panels))
    print(f"시각화 저장: {args.output}")
