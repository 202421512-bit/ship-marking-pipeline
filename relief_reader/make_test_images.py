"""테스트용 가짜 사진 만들기: 녹 얼룩에 가려진 각인 HK357을 4방향 조명 + 평소 조명으로 렌더링.

  python relief_reader/make_test_images.py 저장폴더
"""
import sys, cv2, numpy as np
out = sys.argv[1]; H, W = 500, 1000
rng = np.random.default_rng(1)
# 각인된 글자: 얕은 홈 (높이 맵)
mask = np.zeros((H, W), np.uint8)
cv2.putText(mask, "HK357", (90, 330), cv2.FONT_HERSHEY_SIMPLEX, 6, 255, 22)
height = -cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), 3) * 2.0
height += cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 2) * 0.15  # 표면 거칠기
# 색(알베도): 녹 얼룩 + 페인트 자국이 글자와 무관하게 덮여 있음 → 일반 사진으로는 글자가 안 보임
alb = 0.55 + cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 25) * 1.6
alb += cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 4) * 0.25
alb = np.clip(alb, 0.15, 1.0)
gy, gx = np.gradient(height)
n = np.dstack([-gx, -gy, np.ones_like(gx)]); n /= np.linalg.norm(n, axis=2, keepdims=True)
elev = np.deg2rad(25)
L0 = np.array([0, 0, 1.0]); flat_shade = np.clip((n * L0).sum(2), 0, None)
cv2.imwrite(f"{out}/normal.jpg", np.clip((alb * flat_shade * 0.95 + rng.normal(0, 0.02, (H, W))) * 255, 0, 255).astype(np.uint8))
dirs = {"top": (0, -1), "bottom": (0, 1), "left": (-1, 0), "right": (1, 0)}
for i, (name, (dx, dy)) in enumerate(dirs.items()):
    L = np.array([dx * np.cos(elev), dy * np.cos(elev), np.sin(elev)])
    shade = np.clip((n * L).sum(2), 0, None)
    yy, xx = np.mgrid[0:H, 0:W]
    falloff = 1 - 0.35 * ((xx / W - 0.5) * dx + (yy / H - 0.5) * dy + 0.5)   # 손전등 쪽이 더 밝음
    img = alb * shade * falloff * 1.9 + rng.normal(0, 0.02, (H, W))
    shift = np.float32([[1, 0, rng.integers(-3, 4)], [0, 1, rng.integers(-3, 4)]])  # 손떨림
    img = cv2.warpAffine(img.astype(np.float32), shift, (W, H), borderMode=cv2.BORDER_REFLECT)
    cv2.imwrite(f"{out}/{name}.jpg", np.clip(img * 255, 0, 255).astype(np.uint8))
