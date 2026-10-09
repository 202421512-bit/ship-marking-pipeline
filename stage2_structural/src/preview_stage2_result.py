# -*- coding: utf-8 -*-
"""
preview_stage2_result.py
2차 처리 결과 미리보기  [우리 도구]
  왼쪽: 입력 / 가운데: 처리 결과 / 오른쪽: 판정 (초록=제거, 노랑=불확실로 남김, 파랑=확실한 보호)
실행:  python stage2_structural/src/preview_stage2_result.py [--split dev] [--k 2]
"""
import argparse
import os

import cv2
import numpy as np

from make_stage2_samples import CONDITIONS, DATA_DIR, ROOT
from pipeline import run_stage2
from protect_mask import build_protect_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--k", type=int, default=2)
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, "results", "preview")
    os.makedirs(out_dir, exist_ok=True)
    for cond in CONDITIONS:
        rows = []
        for i in range(args.k):
            d = os.path.join(DATA_DIR, args.split, cond, f"{i:03d}")
            if not os.path.isdir(d):
                break
            img = cv2.imread(os.path.join(d, "image.png"))
            res = run_stage2(img)
            pr = build_protect_mask(img)
            vis = (img * 0.3).astype(np.uint8)
            vis[pr.strong] = (255, 120, 0)
            vis[res.uncertain_mask] = (0, 220, 255)
            vis[res.remove_mask] = (0, 220, 0)
            label = f"{cond} #{i:03d} needs_check={res.needs_check}"
            row = cv2.resize(np.hstack([img, res.image, vis]), (1800, 450))
            cv2.putText(row, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            rows.append(row)
        if rows:
            p = os.path.join(out_dir, f"result_{args.split}_{cond}.png")
            cv2.imwrite(p, np.vstack(rows))
            print("저장:", p)


if __name__ == "__main__":
    main()
