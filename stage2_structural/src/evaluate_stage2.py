# -*- coding: utf-8 -*-
"""
evaluate_stage2.py
2차 노이즈 제거 평가 코드  [우리 평가 도구, 논문 아님]

무엇을 재나 (모든 기준값은 [우리 설정], 파일 위쪽 EVAL_CFG에 모아 둠)

  A. 스크래치 '검출' 성능   detect_mask  vs  gt_scratch
  B. 스크래치 '제거' 성능   remove_mask  vs  제거해야 할 픽셀(= gt_scratch − gt_protect)
     → 이미 손상된 획 픽셀(gt_preexisting_damage)은 '제거 대상'에서 뺀다.
       지우면 그 자리의 획 정보까지 사라지므로, 남겨 두는 것이 보수적 설계의 정답.
     A와 B를 따로 재므로 "찾았지만 일부러 안 지움"을 구분할 수 있다.
     정밀도/재현율은 가는 선의 1~2px 어긋남을 감안해 허용 거리(tol_px) 기준으로도 함께 낸다.

  C. 보호 대상(문자·기호·직선 마킹) 보존
     - 알고리즘이 추가로 만든 손상: 입력에서 온전했던 보호 픽셀(gt_protect_intact) 중
         (선언) remove_mask에 들어간 픽셀
         (실측) 처리 전후 밝기가 change_thresh 이상 바뀐 픽셀  ← 선언하지 않은 부작용까지 잡음
       보존율 = 1 − 손상 픽셀 / 온전한 보호 픽셀
     - 이미 있던 손상(gt_preexisting_damage)은 따로 집계하고, 그중 알고리즘이 지운 픽셀도 기록
     - 반사로 가려진 보호 픽셀은 분모에서 빼고 따로 센다 (원래 정보가 없음)

  D. 반사 영역 표시 성능   glare_mask vs gt_glare (반사 있는 이미지), 반사 없는 이미지의 오검출 면적
  E. needs_check 비율, 불확실 영역 면적
  F. 처리 시간 (장당 초)
  G. (선택) OCR CER: ocr_fn을 넘기면 처리 전/후 이미지의 마킹 박스를 잘라 인식하고 CER 비교
       ocr_fn(crops: list[np.ndarray]) -> list[str]
       CER = 편집 거리(예측, 정답) / 정답 글자 수

실행 예 (저장소 최상위 폴더에서)
  python stage2_structural/src/evaluate_stage2.py --method identity
  python stage2_structural/src/evaluate_stage2.py --method oracle_conservative --split dev
결과: stage2_structural/results/eval/<split>_<method>/ per_image.csv, summary.csv, summary.json
"""
import argparse
import csv
import json
import os
import time

import cv2
import numpy as np

from make_stage2_samples import CONDITIONS, DATA_DIR, ROOT
from stage2_io import Stage2Result, empty_result

EVAL_CFG = {
    "tol_px": 2,            # 허용 거리 정밀도/재현율의 허용 픽셀 [우리 설정]
    "change_thresh": 20,    # 회색조 밝기가 이만큼 이상 바뀌면 '변경됨' [우리 설정]
    "ocr_box_pad": 10,      # CER 평가 때 마킹 박스 여백 px [우리 설정]
}

GT_NAMES = ["gt_protect", "gt_scratch", "gt_preexisting_damage", "gt_protect_intact", "gt_glare"]


# ------------------------------------------------------------------
# 기본 지표
# ------------------------------------------------------------------
def prf(pred, gt, tol=0, forbid=None):
    """픽셀 정밀도/재현율/F1. tol>0이면 tol px 안의 어긋남은 맞은 것으로 본다.
    forbid: 허용 거리 안이어도 '맞은 것'으로 봐주면 안 되는 픽셀 (예: 제거 평가에서 보호 대상)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if tol > 0:
        k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
        gt_d = cv2.dilate(gt.astype(np.uint8), k) > 0
        if forbid is not None:
            gt_d &= ~forbid.astype(bool)
        pred_d = cv2.dilate(pred.astype(np.uint8), k) > 0
        tp_p, tp_r = (pred & gt_d).sum(), (gt & pred_d).sum()
    else:
        tp_p = tp_r = (pred & gt).sum()
    n_pred, n_gt = pred.sum(), gt.sum()
    p = tp_p / n_pred if n_pred else (1.0 if n_gt == 0 else 0.0)
    r = tp_r / n_gt if n_gt else 1.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return float(p), float(r), float(f)


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def crop_boxes(img, boxes, pad):
    h, w = img.shape[:2]
    out = []
    for x1, y1, x2, y2 in boxes:
        out.append(img[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)].copy())
    return out


def cer_for(img, meta, ocr_fn, pad):
    boxes = [m["box_xyxy"] for m in meta["markings"]]
    preds = ocr_fn(crop_boxes(img, boxes, pad))
    gts = meta["gt_text_strings"]
    edits = sum(edit_distance(p or "", g) for p, g in zip(preds, gts))
    chars = sum(len(g) for g in gts)
    exact = sum(int((p or "") == g) for p, g in zip(preds, gts))
    return edits, chars, exact, len(gts)


# ------------------------------------------------------------------
# 한 장 평가
# ------------------------------------------------------------------
def load_sample(d):
    img = cv2.imread(os.path.join(d, "image.png"))
    gt = {k: cv2.imread(os.path.join(d, k + ".png"), cv2.IMREAD_GRAYSCALE) > 0 for k in GT_NAMES}
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    return img, gt, meta


def evaluate_one(img, gt, meta, res: Stage2Result, cfg=EVAL_CFG, ocr_fn=None):
    tol = cfg["tol_px"]
    should_remove = gt["gt_scratch"] & ~gt["gt_protect"]
    glare = gt["gt_glare"]
    intact = gt["gt_protect_intact"] & ~glare            # 반사로 가려진 픽셀은 분모에서 제외
    pre = gt["gt_preexisting_damage"]

    g_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.int16)
    g_out = cv2.cvtColor(res.image, cv2.COLOR_BGR2GRAY).astype(np.int16)
    changed = np.abs(g_out - g_in) >= cfg["change_thresh"]

    r = {"seed": meta["seed"], "condition": meta["condition"], "has_glare": bool(glare.any())}
    # A. 검출
    r["det_P"], r["det_R"], r["det_F1"] = prf(res.detect_mask, gt["gt_scratch"])
    r["det_P_tol"], r["det_R_tol"], r["det_F1_tol"] = prf(res.detect_mask, gt["gt_scratch"], tol)
    # B. 제거
    r["rem_P"], r["rem_R"], r["rem_F1"] = prf(res.remove_mask, should_remove)
    # 보호 대상 위를 지운 것은 허용 거리 안이어도 오답으로 처리
    r["rem_P_tol"], r["rem_R_tol"], r["rem_F1_tol"] = prf(res.remove_mask, should_remove, tol,
                                                         forbid=gt["gt_protect"])
    # C. 보호 대상 보존
    n_intact = int(intact.sum())
    dmg_decl = int((res.remove_mask & intact).sum())
    dmg_meas = int((changed & intact).sum())
    r.update({
        "protect_intact_px": n_intact,
        "algo_damage_declared_px": dmg_decl,
        "algo_damage_measured_px": dmg_meas,
        "preserve_rate_declared": 1 - dmg_decl / n_intact if n_intact else 1.0,
        "preserve_rate_measured": 1 - dmg_meas / n_intact if n_intact else 1.0,
        "preexisting_damage_px": int(pre.sum()),
        "preexisting_removed_px": int((res.remove_mask & pre).sum()),
        "protect_in_glare_px": int((gt["gt_protect_intact"] & glare).sum()),
        # 선언하지 않았는데 바뀐 픽셀 (전체 이미지) — 부작용 점검용
        "undeclared_change_px": int((changed & ~res.remove_mask).sum()),
    })
    # D. 반사
    if glare.any():
        r["glare_P"], r["glare_R"], r["glare_F1"] = prf(res.glare_mask, glare)
        r["glare_false_px_noglare_img"] = None
    else:
        r["glare_P"] = r["glare_R"] = r["glare_F1"] = None
        r["glare_false_px_noglare_img"] = int(res.glare_mask.sum())
    # E. 확인 필요 / 불확실
    r["needs_check"] = bool(res.needs_check)
    r["uncertain_px"] = int(res.uncertain_mask.sum())
    r["uncertain_on_scratch_px"] = int((res.uncertain_mask & gt["gt_scratch"]).sum())
    # G. CER (선택)
    if ocr_fn is not None:
        pad = cfg["ocr_box_pad"]
        for tag, im in (("before", img), ("after", res.image)):
            e, c, x, n = cer_for(im, meta, ocr_fn, pad)
            r[f"cer_{tag}"] = e / c if c else 0.0
            r[f"exact_{tag}"] = x / n if n else 0.0
            r[f"_cer_{tag}_edits"], r[f"_cer_{tag}_chars"] = e, c
    return r


# ------------------------------------------------------------------
# 평가 기준선 (평가 코드 점검용. 정답을 쓰므로 '알고리즘'이 아님)
# ------------------------------------------------------------------
def method_identity(img, protect_boxes=None, gt=None):
    """아무것도 하지 않음 → 검출 0, 손상 0이어야 정상."""
    return empty_result(img)


def _inpaint(img, mask):
    return cv2.inpaint(img, mask.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA) if mask.any() else img.copy()


def method_oracle_conservative(img, protect_boxes=None, gt=None):
    """[정답 사용] 스크래치를 완벽히 찾고, 보호 대상과 겹친 부분은 남김 → 이상적인 보수적 결과."""
    det = gt["gt_scratch"]
    rem = det & ~gt["gt_protect"]
    res = empty_result(img)
    res.image, res.detect_mask, res.remove_mask = _inpaint(img, rem), det, rem
    res.uncertain_mask = det & gt["gt_protect"]
    res.glare_mask = gt["gt_glare"]
    return res


def method_oracle_remove_all(img, protect_boxes=None, gt=None):
    """[정답 사용] 스크래치를 완벽히 찾아 겹친 부분까지 전부 지움 → 공격적 제거의 비교 기준."""
    det = gt["gt_scratch"]
    res = empty_result(img)
    res.image, res.detect_mask, res.remove_mask = _inpaint(img, det), det, det
    res.glare_mask = gt["gt_glare"]
    return res


METHODS = {
    "identity": method_identity,
    "oracle_conservative": method_oracle_conservative,
    "oracle_remove_all": method_oracle_remove_all,
}


# ------------------------------------------------------------------
# 데이터셋 전체 평가
# ------------------------------------------------------------------
def summarize(rows):
    """조건별 요약: 비율 지표는 이미지 평균(macro), 픽셀 수는 합계."""
    out = []
    for cond in list(CONDITIONS) + ["ALL"]:
        sub = [r for r in rows if cond == "ALL" or r["condition"] == cond]
        if not sub:
            continue
        s = {"condition": cond, "n_images": len(sub)}
        for k in sub[0]:
            if k in ("seed", "condition", "has_glare", "path") or k.startswith("_"):
                continue
            vals = [r[k] for r in sub if r.get(k) is not None]
            if not vals:
                s[k] = None
            elif k.endswith("_px"):
                s[k] = int(sum(vals))
            else:
                s[k] = round(float(np.mean(vals)), 4)
        if "_cer_before_edits" in sub[0]:
            for tag in ("before", "after"):
                s[f"cer_{tag}_micro"] = round(sum(r[f"_cer_{tag}_edits"] for r in sub) /
                                              max(1, sum(r[f"_cer_{tag}_chars"] for r in sub)), 4)
        out.append(s)
    return out


def run(method, split="dev", data_root=DATA_DIR, out_dir=None, ocr_fn=None, use_protect_boxes=False,
        verbose=True):
    """method: METHODS의 이름 또는 함수 f(img, protect_boxes=None, gt=None) -> Stage2Result"""
    fn = METHODS[method] if isinstance(method, str) else method
    name = method if isinstance(method, str) else getattr(method, "__name__", "custom")
    mpath = os.path.join(data_root, split, "manifest.json")
    if not os.path.exists(mpath):
        raise SystemExit(f"평가 데이터가 없습니다: {mpath}\n"
                         f"먼저 실행하세요: python stage2_structural/src/make_stage2_samples.py --split {split}")
    manifest = json.load(open(mpath, encoding="utf-8"))
    rows = []
    for item in manifest:
        d = os.path.join(data_root, item["path"])
        img, gt, meta = load_sample(d)
        boxes = [tuple(m["box_xyxy"]) for m in meta["markings"]] if use_protect_boxes else None
        t0 = time.perf_counter()
        res = fn(img, protect_boxes=boxes, gt=gt).validate(img)
        dt = time.perf_counter() - t0
        r = evaluate_one(img, gt, meta, res, ocr_fn=ocr_fn)
        r["time_s"] = round(dt, 4)
        r["path"] = item["path"]
        rows.append(r)
        if verbose:
            print(f"{item['path']:28s} detF1={r['det_F1_tol']:.3f} remF1={r['rem_F1_tol']:.3f} "
                  f"preserve={r['preserve_rate_measured']:.4f} t={dt:.2f}s")
    summary = summarize(rows)
    if out_dir is None:
        out_dir = os.path.join(ROOT, "results", "eval", f"{split}_{name}")
    os.makedirs(out_dir, exist_ok=True)
    keys = list(rows[0].keys())
    with open(os.path.join(out_dir, "per_image.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    skeys = list(dict.fromkeys(k for s in summary for k in s))
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=skeys)
        w.writeheader()
        w.writerows(summary)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"method": name, "split": split, "eval_cfg": EVAL_CFG, "summary": summary},
                  f, ensure_ascii=False, indent=1)
    return rows, summary, out_dir


def print_summary(summary):
    cols = [("condition", "조건"), ("det_F1_tol", "검출F1"), ("rem_F1_tol", "제거F1"),
            ("preserve_rate_measured", "획보존율"), ("algo_damage_measured_px", "알고리즘손상px"),
            ("preexisting_removed_px", "기존손상지움px"), ("glare_F1", "반사F1"),
            ("needs_check", "확인필요비율"), ("time_s", "초/장")]
    print("  ".join(f"{h:>10s}" for _, h in cols))
    for s in summary:
        print("  ".join(f"{str(s.get(k) if s.get(k) is not None else '-'):>10s}" for k, _ in cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=list(METHODS), default="identity")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--protect-boxes", action="store_true", help="정형 문자 박스를 입력으로 제공")
    args = ap.parse_args()
    _, summary, out_dir = run(args.method, args.split, use_protect_boxes=args.protect_boxes)
    print()
    print_summary(summary)
    print("\n결과 저장 위치:", out_dir)


if __name__ == "__main__":
    main()
