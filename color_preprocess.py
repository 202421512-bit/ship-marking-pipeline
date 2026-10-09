"""
색상 기반 강건 전처리: 컬러 입력에서 녹을 가려내고, 배경 대비 '색 거리' 로 마킹을 부각한 회색 영상을 만든다.

그레이스케일 이진화가 무너지는 경우
  - 녹(적갈색)은 회색으로 바꾸면 어두워서 검은 잉크와 구분되지 않는다
  - 노란 도료는 밝기만 보면 강판과 대비가 약할 수 있다

처리
  1. HSV 로 녹 마스크 (적갈색, 채도 있음, 너무 밝지 않음)
  2. LAB 로 국소 배경(큰 커널 중앙값) 을 추정하고, 각 픽셀의 배경 대비 ΔE = |Lab − Lab_bg| 를 계산
     marking="auto" 는 ΔE 전체 (어두운 잉크·흰 도료·노란 도료 모두), "dark"/"white"/"yellow" 는 해당 방향만
  3. 녹 픽셀은 0 으로 두고, ΔE 를 [0, 255] 로 정규화해 '흰 바탕에 어두운 획' 영상으로 뒤집는다
     -> 이후 φ 모듈과 기계 패턴 검출기에 그대로 넣는다
  4. 글자 근처(획 주변) 의 녹 비율은 참고 신뢰도 값으로 돌려준다

반사광은 다루지 않는다: 이 단계의 목적은 받은 영상에서 수기 여부를 판별하는 것이고, 반사광으로 날아간
부분은 판별 대상에서 신경 쓰지 않는다 (영상 품질 평가는 이후 별도 신뢰도 단계에서).
그레이스케일 입력은 그대로 통과 (녹 판단 불가).
녹과 같은 적갈색 계열 도료는 녹으로 가려질 수 있다 (marking="dark"/"white"/"yellow" 로 지정하면 해당 색만 봄).
"""
import cv2
import numpy as np

RUST_H = (0, 18)            # OpenCV H(0~180): 적갈색
RUST_S_MIN = 70
RUST_V = (30, 210)
YELLOW_H = (18, 40)
BG_KERNEL_RATIO = 0.5       # 배경 추정 중앙값 커널 = 이 비율 × 짧은 변


def is_color(img):
    if img.ndim != 3 or img.shape[2] != 3:
        return False
    b, g, r = cv2.split(img.astype(np.int16))
    return bool(max(np.abs(b - g).mean(), np.abs(g - r).mean()) > 2.0)


def _odd(n):
    n = int(n)
    return n if n % 2 else n + 1


def preprocess_color(img, marking="auto"):
    """반환 dict: enhanced(uint8, 흰 바탕·어두운 획), rust_mask, is_color, marking, delta_e(정규화 전)."""
    if not is_color(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return {"enhanced": gray, "rust_mask": np.zeros(gray.shape, bool),
                "is_color": False, "marking": "gray"}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    rust = ((h >= RUST_H[0]) & (h <= RUST_H[1]) | (h >= 172)) & (s >= RUST_S_MIN) & (v >= RUST_V[0]) & (v <= RUST_V[1])
    rust = cv2.morphologyEx(rust.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)
    k = _odd(max(15, BG_KERNEL_RATIO * min(img.shape[:2])))
    k = min(k, 255)
    # 배경 추정: 녹은 배경 계산에서도 영향을 줄이려 나머지 중앙값으로 메운 뒤 중앙값 필터
    fill = rust
    bg = []
    for ch in (L, A, B):
        c = ch.copy()
        if fill.any():
            c[fill] = np.median(ch[~fill]) if (~fill).any() else np.median(ch)
        bg.append(cv2.medianBlur(np.clip(c, 0, 255).astype(np.uint8), k).astype(np.float32))
    dL, dA, dB = L - bg[0], A - bg[1], B - bg[2]
    if marking == "dark":
        delta = np.maximum(-dL, 0)
    elif marking == "white":
        delta = np.maximum(dL, 0) * (np.hypot(dA, dB) < 20)
    elif marking == "yellow":
        delta = np.maximum(dB, 0) * ((h >= YELLOW_H[0]) & (h <= YELLOW_H[1]))
    else:
        delta = np.sqrt(dL * dL + dA * dA + dB * dB)
    delta[fill] = 0
    top = float(np.percentile(delta, 99.5)) or 1.0
    enhanced = 255 - np.clip(delta / top * 200, 0, 255).astype(np.uint8)    # 바탕 ≈ 255, 획 ≈ 55
    return {"enhanced": enhanced, "rust_mask": rust, "is_color": True,
            "marking": marking, "delta_e": delta}


def text_region_penalty(pre, ink_mask, margin=None):
    """글자 주변(획을 넓힌 영역) 에서 녹이 차지하는 비율."""
    if not ink_mask.any():
        return 0.0
    if margin is None:
        margin = max(5, int(0.02 * max(ink_mask.shape)))
    region = cv2.dilate(ink_mask.astype(np.uint8), np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)) > 0
    area = region.sum()
    return float((pre["rust_mask"] & region).sum() / area)
