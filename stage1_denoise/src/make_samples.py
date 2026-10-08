# -*- coding: utf-8 -*-
"""
make_samples.py
철판 마킹 '가상(합성) 테스트 이미지' 생성기

[중요] 이 파일은 논문 내용이 아닙니다.
실제 현장 이미지가 아직 없어서, 1차 노이즈 제거 코드를 시험하기 위해
우리가 임의로 만든 테스트 도구입니다. (구현을 위한 우리의 가정/선택)

만들어지는 것
  data/samples/        : 노이즈가 섞인 가상 철판 이미지 (입력용)
  data/samples_gt/     : 각 이미지에서 '진짜 글자 획'이 있는 위치 (흰색=획)
                         -> 나중에 노이즈 제거 후 획이 얼마나 남았는지 측정할 때 사용

실행 방법 (VS Code 터미널, (venv) 상태에서)
  python src/make_samples.py
"""

import os
import cv2
import numpy as np

# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
H, W = 1200, 1600                     # 이미지 크기 (세로, 가로)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "samples")
GT_DIR = os.path.join(ROOT, "data", "samples_gt")


# ------------------------------------------------------------
# 1. 철판 바탕 + 브러시 결(가로 줄무늬) 질감
# ------------------------------------------------------------
def make_plate(rng):
    base = np.full((H, W, 3), (118, 122, 125), np.float32)   # 회색 철판 (B,G,R)
    # 미세 입자 노이즈
    grain = rng.normal(0, 6, (H, W, 1)).astype(np.float32)
    # 가로 방향으로 길게 늘린 노이즈 = 브러시 결
    streak = rng.normal(0, 18, (H, W)).astype(np.float32)
    streak = cv2.blur(streak, (61, 1))[..., None]
    return base + grain + streak


# ------------------------------------------------------------
# 2. 손글씨 느낌의 마킹 (페인트 마커 / 석필 느낌)
#    stroke_mask: 실제 획 위치 (정답 마스크)
# ------------------------------------------------------------
MARKINGS = [
    # (문자열, 중심 x, 중심 y, 크기, 기울기(도), 색(B,G,R))
    ("B12-P3",  520,  330, 3.2, -8,  (235, 235, 235)),   # 흰색 마커
    ("FR45 S",  1100, 560, 2.6,  12, (60, 215, 240)),    # 노란색 마커
    ("No.7 L",  480,  840, 2.8,  4,  (235, 235, 235)),
    ("CUT>",    1180, 930, 2.4, -15, (60, 215, 240)),
]


def draw_markings(rng, faded=False):
    paint = np.zeros((H, W, 3), np.float32)   # 마킹 색
    mask = np.zeros((H, W), np.uint8)         # 정답 획 위치

    for text, cx, cy, scale, angle, color in MARKINGS:
        layer = np.zeros((H, W), np.uint8)
        thick = int(rng.integers(7, 11))
        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX          # 필기체 비슷한 폰트
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        cv2.putText(layer, text, (cx - tw // 2, cy + th // 2), font, scale, 255, thick, cv2.LINE_AA)
        # 동그라미 기호 하나 추가 (현장 표기 관행 흉내)
        cv2.ellipse(layer, (cx + tw // 2 + 50, cy), (32, 32), 0, 0, 330, 255, thick - 2, cv2.LINE_AA)
        # 글자 기울이기
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        layer = cv2.warpAffine(layer, M, (W, H))
        mask = np.maximum(mask, layer)
        for c in range(3):
            paint[..., c] = np.where(layer > 0, color[c], paint[..., c])

    # 획 강도(얼마나 진하게 칠해졌는지): 붓 자국처럼 들쭉날쭉하게
    alpha = (mask.astype(np.float32) / 255.0)
    patchy = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 6)
    patchy = (patchy - patchy.min()) / (patchy.max() - patchy.min() + 1e-6)
    strength = 0.55 + 0.45 * patchy
    if faded:   # 일부 지워진 글씨
        strength = np.clip(patchy * 1.6 - 0.35, 0, 1) * 0.6
    alpha = alpha * strength

    # 정답 마스크는 '원래 칠해진 위치' 기준 (지워진 부분도 획으로 간주)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return paint, alpha[..., None], mask


# ------------------------------------------------------------
# 3. 녹 / 산화 얼룩
# ------------------------------------------------------------
def rust_layer(rng, amount):
    n = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 25)
    n = (n - n.min()) / (n.max() - n.min() + 1e-6)
    fine = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 2)
    thr = 1.0 - amount
    a = np.clip((n - thr) / 0.12, 0, 1) * (0.6 + 0.4 * fine)
    rust_color = np.dstack([
        35 + 25 * fine,      # B
        70 + 35 * fine,      # G
        140 + 40 * fine,     # R  (갈색/주황)
    ]).astype(np.float32)
    return rust_color, a[..., None]


# ------------------------------------------------------------
# 4. 스크래치 (얇은 선, 밝거나 어둡게)
# ------------------------------------------------------------
def add_scratches(img, rng, count):
    layer = np.zeros((H, W), np.float32)
    for _ in range(count):
        x1, y1 = int(rng.integers(0, W)), int(rng.integers(0, H))
        length = rng.integers(80, 500)
        ang = rng.uniform(0, np.pi)
        x2, y2 = int(x1 + length * np.cos(ang)), int(y1 + length * np.sin(ang))
        val = rng.choice([-1.0, 1.0]) * rng.uniform(25, 60)
        cv2.line(layer, (x1, y1), (x2, y2), float(val), int(rng.integers(1, 3)), cv2.LINE_AA)
    return img + layer[..., None]


# ------------------------------------------------------------
# 5. 조명: 불균일 밝기, 그림자, 반사(하이라이트)
# ------------------------------------------------------------
def lighting(img, rng, gradient=0.35, shadow=0.0, glare=0.0):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # 한쪽에서 들어오는 빛 (밝기 기울기)
    g = 1.0 + gradient * ((xx / W) - 0.5) * 2 * rng.choice([-1, 1])
    out = img * g[..., None]

    if shadow > 0:   # 경계가 부드러운 그림자 띠
        m = np.zeros((H, W), np.float32)
        pts = np.array([[0, int(H * 0.55)], [W, int(H * 0.35)], [W, H], [0, H]], np.int32)
        cv2.fillPoly(m, [pts], 1.0)
        m = cv2.GaussianBlur(m, (0, 0), 40)
        out = out * (1.0 - shadow * m)[..., None]

    if glare > 0:    # 금속 반사로 하얗게 날아간 부분
        cx, cy = rng.integers(W // 4, 3 * W // 4), rng.integers(H // 4, 3 * H // 4)
        d = ((xx - cx) / 260) ** 2 + ((yy - cy) / 160) ** 2
        out = out + glare * 255 * np.exp(-d)[..., None]
    return out


# ------------------------------------------------------------
# 6. 카메라 흐림 + 센서 노이즈
# ------------------------------------------------------------
def camera(img, rng, blur=0.0, noise=4.0):
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    img = img + rng.normal(0, noise, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


# ------------------------------------------------------------
# 시나리오별 이미지 만들기
#   각 이미지가 '어떤 노이즈를 주로 시험하는지' 이름에 표시
# ------------------------------------------------------------
SCENARIOS = [
    # 이름,                    녹양, 스크래치, 그림자, 반사, 흐림, 지워진글씨
    ("s01_mixed",              0.25, 25, 0.0, 0.0, 0.8, False),
    ("s02_strong_glare",       0.15, 15, 0.0, 0.9, 0.8, False),
    ("s03_deep_shadow",        0.15, 15, 0.7, 0.0, 0.8, False),
    ("s04_heavy_rust",         0.55, 15, 0.0, 0.0, 0.8, False),
    ("s05_dense_scratches",    0.15, 120, 0.0, 0.0, 0.8, False),
    ("s06_faded_blurry",       0.25, 25, 0.3, 0.0, 2.0, True),
]


def make_one(seed, rust_amt, n_scratch, shadow, glare, blur, faded):
    rng = np.random.default_rng(seed)
    img = make_plate(rng)

    # 녹은 마킹보다 먼저(철판 표면에) 생기고, 일부는 마킹 위에도 덮이도록 두 번 나눠 적용
    rc, ra = rust_layer(rng, rust_amt)
    img = img * (1 - ra) + rc * ra

    paint, pa, mask = draw_markings(rng, faded=faded)
    img = img * (1 - pa) + paint * pa

    rc2, ra2 = rust_layer(rng, rust_amt * 0.5)          # 마킹 위를 덮는 녹
    img = img * (1 - ra2 * 0.7) + rc2 * (ra2 * 0.7)

    img = add_scratches(img, rng, n_scratch)
    img = lighting(img, rng, gradient=0.35, shadow=shadow, glare=glare)
    img = camera(img, rng, blur=blur)
    return img, mask


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(GT_DIR, exist_ok=True)
    for i, (name, rust_amt, n_scr, shadow, glare, blur, faded) in enumerate(SCENARIOS):
        img, mask = make_one(100 + i, rust_amt, n_scr, shadow, glare, blur, faded)
        cv2.imwrite(os.path.join(OUT_DIR, name + ".png"), img)
        cv2.imwrite(os.path.join(GT_DIR, name + "_mask.png"), mask)
        print("저장 완료:", name)
    print("\n샘플 이미지 위치:", OUT_DIR)


if __name__ == "__main__":
    main()
