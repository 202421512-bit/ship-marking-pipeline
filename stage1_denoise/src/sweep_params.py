# -*- coding: utf-8 -*-
"""
sweep_params.py
1차 노이즈 제거의 '변수(파라미터) 영향' 실험  — [우리 추가, 논문에 없음]

방법: stage1_denoise.py 의 CONFIG(기본값)에서 '한 번에 변수 하나만' 바꿔가며
      같은 이미지를 처리하고, 결과를 숫자와 그림으로 비교합니다.

실행 (VS Code 터미널, (venv) 상태에서)
  python src/sweep_params.py

출력: results/sweep/
  - sweep_metrics.csv          : 모든 조합의 수치
  - sweep_<변수이름>.png        : 값에 따라 글자 주변이 어떻게 변하는지 확대 비교

수치 설명 (모두 L 채널 기준)
  contrast : 획 평균 밝기와 주변 배경 평균 밝기의 차이  → 클수록 글자가 진하게 구별됨
  bg_noise : 주변 배경의 밝기 표준편차                → 작을수록 배경이 깨끗함
  CNR      : contrast / bg_noise                     → 둘을 합친 종합 지표
  ※ CNR이 올랐을 때 '배경이 깨끗해져서'인지 '획이 진해져서'인지 구분하려고
    contrast와 bg_noise를 따로 기록합니다.
"""

import os
import copy
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stage1_denoise import CONFIG, run_pipeline, imread_any, ROOT

SAMPLE_DIR = os.path.join(ROOT, "data", "samples")
GT_DIR = os.path.join(ROOT, "data", "samples_gt")
OUT_DIR = os.path.join(ROOT, "results", "sweep")

# 바꿔볼 변수와 값 (기본값은 stage1_denoise.py의 CONFIG)
SWEEPS = {
    "clahe1_clip":     [1.0, 2.0, 4.0, 8.0],
    "clahe1_tile":     [2, 4, 8, 16],
    "bil_sigma_color": [10, 40, 75, 150],
    "bil_d":           [5, 9, 15],
    "unsharp_amount":  [0.0, 0.5, 1.0, 2.0],
    "unsharp_sigma":   [1.0, 3.0, 6.0],
}

# 그림에서 확대해서 볼 부분 (이미지 이름: 확대 중심 x, y) — 합성 샘플 기준
ZOOM = {
    "s02_strong_glare":    (1150, 900),   # 반사 근처 "CUT>"
    "s03_deep_shadow":     (480, 840),    # 그림자 속 "No.7 L"
    "s05_dense_scratches": (500, 330),    # 스크래치 위 "B12-P3"
    "s06_faded_blurry":    (1100, 560),   # 흐리고 지워진 "FR45 S"
}
ZW, ZH = 300, 160    # 확대 영역 반폭, 반높이


def measure(L, mask):
    stroke = mask > 0
    ring = cv2.dilate(mask, np.ones((25, 25), np.uint8)) > 0
    bg = ring & ~(cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0)
    Lf = L.astype(np.float32)
    contrast = abs(Lf[stroke].mean() - Lf[bg].mean())
    noise = Lf[bg].std()
    return contrast, noise, contrast / (noise + 1e-6)


def crop(img, cx, cy):
    h, w = img.shape[:2]
    x0, y0 = max(cx - ZW, 0), max(cy - ZH, 0)
    return img[y0:min(cy + ZH, h), x0:min(cx + ZW, w)]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(SAMPLE_DIR) if f.endswith(".png"))
    images = {n: imread_any(os.path.join(SAMPLE_DIR, n + ".png")) for n in names}
    masks = {n: imread_any(os.path.join(GT_DIR, n + "_mask.png"), cv2.IMREAD_GRAYSCALE) for n in names}

    rows = []
    # 원본(처리 전) 수치도 기준으로 기록
    for n in names:
        L0 = cv2.cvtColor(images[n], cv2.COLOR_BGR2LAB)[..., 0]
        c, s, r = measure(L0, masks[n])
        rows.append(["(original)", "-", n, f"{c:.1f}", f"{s:.1f}", f"{r:.2f}"])

    total = sum(len(v) for v in SWEEPS.values())
    k = 0
    for param, values in SWEEPS.items():
        fig, axes = plt.subplots(len(values), len(ZOOM), figsize=(4.2 * len(ZOOM), 2.4 * len(values)))
        for i, v in enumerate(values):
            k += 1
            cfg = copy.deepcopy(CONFIG)
            cfg[param] = v
            print(f"[{k}/{total}] {param} = {v}")
            for n in names:
                out, _, Ls = run_pipeline(images[n], cfg)
                c, s, r = measure(Ls["L3"], masks[n])
                rows.append([param, v, n, f"{c:.1f}", f"{s:.1f}", f"{r:.2f}"])
                if n in ZOOM:
                    j = list(ZOOM).index(n)
                    ax = axes[i, j]
                    ax.imshow(cv2.cvtColor(crop(out, *ZOOM[n]), cv2.COLOR_BGR2RGB))
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.set_title(f"{n.split('_', 1)[1]} | CNR {r:.2f}", fontsize=9)
                    if j == 0:
                        ax.set_ylabel(f"{param}\n= {v}", fontsize=10)
        fig.suptitle(f"Sweep: {param}  (other params = default)", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"sweep_{param}.png"), dpi=90)
        plt.close(fig)

    with open(os.path.join(OUT_DIR, "sweep_metrics.csv"), "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(["param", "value", "image", "contrast", "bg_noise", "CNR"])
        w.writerows(rows)
    print("\n결과 저장 위치:", OUT_DIR)


if __name__ == "__main__":
    main()
