# -*- coding: utf-8 -*-
"""
ablation.py
단계별 끄기 실험 (Ablation)  — [우리 추가, 논문에 없음]

논문은 전처리를 '켠 경우'만 보고했고, 각 단계가 실제로 도움이 되는지
비교(ablation)하지 않았습니다. 그래서 1단계 / 2단계-① / 2단계-② 를
하나씩 끄거나 하나만 켜서 같은 이미지를 처리해 비교합니다.

실행 (VS Code 터미널, (venv) 상태에서)
  python src/ablation.py

출력: results/ablation/
  - ablation_metrics.csv : 조합별 contrast / bg_noise / CNR
  - ablation_<이미지>.png : 조합별 결과를 글자 주변 확대로 비교
"""

import os
import copy
import csv
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stage1_denoise import CONFIG, run_pipeline, imread_any, ROOT
from sweep_params import measure, crop, ZOOM

SAMPLE_DIR = os.path.join(ROOT, "data", "samples")
GT_DIR = os.path.join(ROOT, "data", "samples_gt")
OUT_DIR = os.path.join(ROOT, "results", "ablation")

# 조합 이름: (1단계, 2단계-①, 2단계-②) — 기본값은 CONFIG의 방법, "none"은 끔
D1, D2A, D2B = CONFIG["step1_method"], CONFIG["step2a_method"], CONFIG["step2b_method"]
VARIANTS = {
    "A. original (all off)":      ("none", "none", "none"),
    "B. step1 only":              (D1,     "none", "none"),
    "C. step1 + 2a (no 2b)":      (D1,     D2A,    "none"),
    "D. full (1 + 2a + 2b)":      (D1,     D2A,    D2B),
    "E. no step1 (2a + 2b)":      ("none", D2A,    D2B),
    "F. no 2a (1 + 2b)":          (D1,     "none", D2B),
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(SAMPLE_DIR) if f.endswith(".png"))
    rows = []
    for n in names:
        img = imread_any(os.path.join(SAMPLE_DIR, n + ".png"))
        mask = imread_any(os.path.join(GT_DIR, n + "_mask.png"), cv2.IMREAD_GRAYSCALE)
        outs = {}
        for vname, (m1, m2a, m2b) in VARIANTS.items():
            cfg = copy.deepcopy(CONFIG)
            cfg["step1_method"], cfg["step2a_method"], cfg["step2b_method"] = m1, m2a, m2b
            out, _, Ls = run_pipeline(img, cfg)
            c, s, r = measure(Ls["L3"], mask)
            rows.append([n, vname, f"{c:.1f}", f"{s:.1f}", f"{r:.2f}"])
            outs[vname] = (out, r)
        print(n, " | ".join(f"{k.split('.')[0]}={v[1]:.2f}" for k, v in outs.items()))

        if n in ZOOM:
            fig, axes = plt.subplots(2, 3, figsize=(13, 5.2))
            for ax, (vname, (out, r)) in zip(axes.ravel(), outs.items()):
                ax.imshow(cv2.cvtColor(crop(out, *ZOOM[n]), cv2.COLOR_BGR2RGB))
                ax.set_title(f"{vname} | CNR {r:.2f}", fontsize=10)
                ax.axis("off")
            fig.suptitle(f"Ablation: {n}", fontsize=13)
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, f"ablation_{n}.png"), dpi=90)
            plt.close(fig)

    with open(os.path.join(OUT_DIR, "ablation_metrics.csv"), "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(["image", "variant", "contrast", "bg_noise", "CNR"])
        w.writerows(rows)
    print("\n결과 저장 위치:", OUT_DIR)


if __name__ == "__main__":
    main()
