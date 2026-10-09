# -*- coding: utf-8 -*-
"""
preview_stage2_samples.py
생성된 가상 데이터를 조건별로 눈으로 확인하는 미리보기 그림을 만든다. [우리 도구]

색 표시 (오른쪽 그림)
  초록 = 보호 대상 중 온전한 픽셀 (gt_protect_intact)
  빨강 = 스크래치 (gt_scratch, 보호 대상과 겹친 부분 제외)
  노랑 = 이미 손상된 보호 픽셀 (gt_preexisting_damage)
  파랑 = 긴 직선 마킹 (gt_marking_line, 온전한 부분)
  보라 = 반사 포화 영역 (gt_glare)

실행:  python stage2_structural/src/preview_stage2_samples.py [--split dev] [--k 3]
출력:  stage2_structural/results/preview/preview_<split>_<조건>.png
"""
import argparse
import json
import os

import cv2
import numpy as np

from make_stage2_samples import CONDITIONS, DATA_DIR, ROOT


def overlay(d):
    def m(name):
        return cv2.imread(os.path.join(d, name + ".png"), cv2.IMREAD_GRAYSCALE) > 0
    img = cv2.imread(os.path.join(d, "image.png"))
    vis = (img * 0.35).astype(np.uint8)
    vis[m("gt_scratch")] = (0, 0, 255)
    vis[m("gt_protect_intact")] = (0, 200, 0)
    vis[m("gt_marking_line") & m("gt_protect_intact")] = (255, 120, 0)
    vis[m("gt_preexisting_damage")] = (0, 255, 255)
    g = m("gt_glare")
    vis[g] = (0.5 * vis[g] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
    return img, vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--k", type=int, default=3, help="조건별로 보여줄 장수")
    args = ap.parse_args()
    out_dir = os.path.join(ROOT, "results", "preview")
    os.makedirs(out_dir, exist_ok=True)
    for cond in CONDITIONS:
        rows = []
        for i in range(args.k):
            d = os.path.join(DATA_DIR, args.split, cond, f"{i:03d}")
            if not os.path.isdir(d):
                break
            img, vis = overlay(d)
            meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
            row = np.hstack([img, vis])
            row = cv2.resize(row, (1600, 600))
            label = f"{cond} #{i:03d} seed={meta['seed']} text={meta['gt_text_strings']}"
            cv2.putText(row, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            rows.append(row)
        if rows:
            p = os.path.join(out_dir, f"preview_{args.split}_{cond}.png")
            cv2.imwrite(p, np.vstack(rows))
            print("저장:", p)


if __name__ == "__main__":
    main()
