# -*- coding: utf-8 -*-
"""
sweep_stage2.py
4단계 실험 파라미터 비교  [우리 평가 도구]
  - 안전 거리 safety_px: 2 / 4 / 8  → 제거 재현율, 획 손상, 기존 손상 지움
  - needs_check 반경: 10 / 20 / 30  → 위험 이미지 검출률, 불필요 확인 요청률
실행:  python stage2_structural/src/sweep_stage2.py [--split dev]
"""
import argparse
import json
import os

import numpy as np

from decision import needs_check_flag
from evaluate_stage2 import evaluate_one, load_sample
from make_stage2_samples import CONDITIONS, DATA_DIR, ROOT
from pipeline import run_stage2
from protect_mask import build_protect_mask

SAFETY = (2, 4, 8)
RADII = (10, 20, 30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()
    manifest = json.load(open(os.path.join(DATA_DIR, args.split, "manifest.json"), encoding="utf-8"))
    rows = []
    for item in manifest:
        img, gt, meta = load_sample(os.path.join(DATA_DIR, item["path"]))
        strong = build_protect_mask(img).strong
        s = max(img.shape[:2]) / 1600
        for sp in SAFETY:
            res = run_stage2(img, cfg={"decision": {"safety_px": sp}})
            r = evaluate_one(img, gt, meta, res)
            for rad in RADII:
                r[f"nc_{rad}"] = needs_check_flag(res.uncertain_mask, strong, int(round(rad * s)))
            r.update(safety=sp, condition=meta["condition"])
            rows.append(r)
        print(item["path"], "done")

    print("\n[안전 거리 비교]  (제거 재현율·검출 F1은 이미지 평균, 손상은 합계)")
    print(f"{'safety':>7} {'조건':>14} {'검출F1px':>9} {'제거R_px':>9} {'제거R_선분':>10} {'획손상px':>8} {'보호변경px':>9} {'기존손상지움px':>12}")
    for sp in SAFETY:
        for cond in list(CONDITIONS) + ["ALL"]:
            sub = [r for r in rows if r["safety"] == sp and (cond == "ALL" or r["condition"] == cond)]
            seg = [r["seg_rem_R80"] for r in sub if r["seg_rem_R80"] is not None]
            print(f"{sp:>7} {cond:>14} {np.mean([r['det_F1_tol'] for r in sub]):9.3f} "
                  f"{np.mean([r['rem_R_tol'] for r in sub]):9.3f} {np.mean(seg) if seg else float('nan'):10.3f} "
                  f"{sum(r['algo_damage_measured_px'] for r in sub):8d} "
                  f"{sum(r['alg_protected_changed_px'] or 0 for r in sub):9d} {sum(r['preexisting_removed_px'] for r in sub):12d}")
    print("\n[needs_check 반경 비교] (안전 거리 4px 결과 기준)")
    sub = [r for r in rows if r["safety"] == 4]
    risk = [r for r in sub if r["risk_image"]]
    safe = [r for r in sub if not r["risk_image"]]
    for rad in RADII:
        print(f"반경 {rad:>2}px: 위험 이미지 검출률 {np.mean([r[f'nc_{rad}'] for r in risk]):.2f} (n={len(risk)}), "
              f"불필요 확인 요청률 {np.mean([r[f'nc_{rad}'] for r in safe]) if safe else float('nan'):.2f} (n={len(safe)})")
    out = os.path.join(ROOT, "results", "eval", f"sweep_{args.split}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    print("\n저장:", out)


if __name__ == "__main__":
    main()
