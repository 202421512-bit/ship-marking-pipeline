# -*- coding: utf-8 -*-
"""
eval_protect.py
3단계 평가: 문자 보호 마스크만 따로 평가한다.  [우리 평가 도구]

지표 (조건별, 이미지 평균. 분모가 0인 이미지는 그 지표에서 제외)
  보호 재현율 (온전한 보호 픽셀 중 보호된 비율, 반사 영역 제외) — 1에 가까울수록 좋음
    clear_* : 선명한 문자·기호     faint_* : 흐린 문자·기호     line_* : 긴 직선 마킹
    *_strong : 확실한 보호만       *_any : 확실한 보호 ∪ 의심 영역 (= 지우지 않는 영역)
  과보호율 (지워야 할 스크래치 중 보호 영역에 들어간 비율) — 낮을수록 좋음
    overprotect_strong / overprotect_any
    → overprotect_any 가 크면 4단계에서 지울 수 있는 스크래치가 그만큼 줄어든다 (제거 재현율의 상한)

실행:  python stage2_structural/src/eval_protect.py [--split dev] [--polarity bright]
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from make_stage2_samples import CONDITIONS, DATA_DIR, ROOT
from protect_mask import build_protect_mask

GT = ["gt_text", "gt_symbol", "gt_text_faint", "gt_marking_line", "gt_protect", "gt_scratch",
      "gt_protect_intact", "gt_glare"]


def ratio(a, b):
    nb = int(b.sum())
    return float((a & b).sum()) / nb if nb else None


def eval_one(pr, g):
    base = g["gt_protect_intact"] & ~g["gt_glare"]
    clear = base & (g["gt_text"] | g["gt_symbol"]) & ~g["gt_text_faint"]
    faint = base & g["gt_text_faint"]
    line = base & g["gt_marking_line"]
    should_remove = g["gt_scratch"] & ~g["gt_protect"]
    anyp = pr.protected
    return {
        "clear_strong": ratio(pr.strong, clear), "clear_any": ratio(anyp, clear),
        "faint_strong": ratio(pr.strong, faint), "faint_any": ratio(anyp, faint),
        "line_strong": ratio(pr.strong, line), "line_any": ratio(anyp, line),
        "overprotect_strong": ratio(pr.strong, should_remove), "overprotect_any": ratio(anyp, should_remove),
        "strong_area": float(pr.strong.mean()), "suspect_area": float(pr.suspect.mean()),
    }


def run(split="dev", data_root=DATA_DIR, cfg=None, use_boxes=False, verbose=True):
    manifest = json.load(open(os.path.join(data_root, split, "manifest.json"), encoding="utf-8"))
    rows = []
    for item in manifest:
        d = os.path.join(data_root, item["path"])
        img = cv2.imread(os.path.join(d, "image.png"))
        g = {k: cv2.imread(os.path.join(d, k + ".png"), cv2.IMREAD_GRAYSCALE) > 0 for k in GT}
        meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        boxes = [tuple(m["box_xyxy"]) for m in meta["markings"]] if use_boxes else None
        t0 = time.perf_counter()
        pr = build_protect_mask(img, boxes, cfg)
        r = eval_one(pr, g)
        r.update(condition=meta["condition"], path=item["path"], time_s=time.perf_counter() - t0)
        rows.append(r)
        if verbose:
            f = lambda v: "-" if v is None else f"{v:.3f}"
            print(f"{item['path']:24s} clear={f(r['clear_any'])} faint={f(r['faint_any'])} "
                  f"line={f(r['line_any'])} overprotect={f(r['overprotect_any'])} t={r['time_s']:.2f}s")
    return rows


def save_preview(split="dev", k=2, data_root=DATA_DIR, cfg=None, out_dir=None):
    """조건별 k장: 왼쪽 입력, 오른쪽 보호 마스크
    초록=확실한 보호, 노랑=의심 영역(지우지 않음), 빨강=보호하지 않은 가는 직선(스크래치 후보)"""
    out_dir = out_dir or os.path.join(ROOT, "results", "eval_protect")
    os.makedirs(out_dir, exist_ok=True)
    for cond in CONDITIONS:
        rows = []
        for i in range(k):
            d = os.path.join(data_root, split, cond, f"{i:03d}")
            if not os.path.isdir(d):
                break
            img = cv2.imread(os.path.join(d, "image.png"))
            pr = build_protect_mask(img, None, cfg)
            vis = (img * 0.3).astype(np.uint8)
            vis[pr.thin_lines] = (0, 0, 255)
            vis[pr.suspect] = (0, 220, 255)
            vis[pr.strong] = (0, 200, 0)
            rows.append(cv2.resize(np.hstack([img, vis]), (1600, 600)))
        if rows:
            p = os.path.join(out_dir, f"preview_protect_{split}_{cond}.png")
            cv2.imwrite(p, np.vstack(rows))
            print("미리보기 저장:", p)


def summarize(rows):
    keys = [k for k in rows[0] if k not in ("condition", "path")]
    out = []
    for cond in list(CONDITIONS) + ["ALL"]:
        sub = [r for r in rows if cond == "ALL" or r["condition"] == cond]
        s = {"condition": cond, "n": len(sub)}
        for k in keys:
            v = [r[k] for r in sub if r[k] is not None]
            s[k] = round(float(np.mean(v)), 4) if v else None
            if k in ("faint_any", "line_any"):
                s[k + "_n_img"] = len(v)
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--polarity", default="bright", choices=["bright", "dark"])
    ap.add_argument("--boxes", action="store_true", help="정형 문자 박스를 입력으로 제공")
    ap.add_argument("--preview", type=int, default=2, help="조건별 미리보기 장수 (0이면 생략)")
    args = ap.parse_args()
    rows = run(args.split, cfg={"polarity": args.polarity}, use_boxes=args.boxes)
    summ = summarize(rows)
    out_dir = os.path.join(ROOT, "results", "eval_protect")
    os.makedirs(out_dir, exist_ok=True)
    name = f"{args.split}_{args.polarity}{'_boxes' if args.boxes else ''}"
    with open(os.path.join(out_dir, name + ".json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summ, "rows": rows}, f, ensure_ascii=False, indent=1)
    cols = ["clear_strong", "clear_any", "faint_strong", "faint_any", "line_any", "overprotect_strong",
            "overprotect_any", "time_s"]
    print("\n" + f"{'condition':>14s} " + " ".join(f"{c:>18s}" for c in cols))
    for s in summ:
        print(f"{s['condition']:>14s} " + " ".join(f"{('-' if s[c] is None else s[c]):>18}" for c in cols))
    print("\n저장:", os.path.join(out_dir, name + ".json"))
    if args.preview:
        save_preview(args.split, args.preview, cfg={"polarity": args.polarity})


if __name__ == "__main__":
    main()
