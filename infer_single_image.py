"""
실제 촬영 사진 한 장 판별 (합성 데이터 평가와 별개의 독립 실행 스크립트)

  python infer_single_image.py "C:\\Users\\hjlee\\OneDrive\\바탕 화면\\공모전\\PAC2026\\sample.jpg"
  python infer_single_image.py 사진.HEIC -o result_sample.png --color auto

콘솔: 0단계 반사광·대비 점검 -> 1단계 기계 패턴(도트/스텐실) -> 2단계 φ₁~φ₈ 와 결합 점수 S -> 최종 판정
그림: 원본(글자 영역·포화 픽셀 표시) | 이진화 마스크(줄 구분) | 획 골격·외곽선 | 1단계 패턴 검출 | 점수 막대

입력은 마킹 주변을 대략 잘라 낸 사진을 권장한다 (넓은 배경이 많으면 배경 물체가 글자로 잡힐 수 있음).
JPG/PNG/BMP 외에 HEIC 는 pillow-heif 가 설치돼 있으면 읽는다. 한글 경로 가능.
"""
import argparse
import os
import sys

import cv2
import numpy as np

import objective_function as of
from mechanical_pattern_detector import draw_mechanical_debug
from orientation_feature import _remove_specks, binarize

PANEL_W = 640


def read_image(path):
    data = np.fromfile(path, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        import pillow_heif
        from PIL import Image, ImageOps
    except ImportError:
        sys.exit(f"읽을 수 없는 이미지입니다: {path} (HEIC 면 'pip install pillow-heif')")
    pillow_heif.register_heif_opener()
    rgb = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# 콘솔
# ---------------------------------------------------------------------------
def print_report(path, img, r):
    line = "-" * 72
    print(line)
    print(f"입력: {path}")
    print(f"크기: {img.shape[1]}x{img.shape[0]} -> 분석 해상도 배율 {r['work_scale']:.3f}")
    print(line)
    cp = r["confidence_parts"]
    print("[0단계] 영상 점검")
    print(f"  반사광 포화(V>{of.SPECULAR_V}) 비율 (글자 영역): {r['specular_ratio']:.1%} "
          f"(기각 기준 {of.SPECULAR_REJECT:.0%}) -> {'기각' if 'REJECTED_SPECULAR_NOISE' in r['flags'] else '정상'}")
    print(f"  배경-획 대비 CNR: {cp.get('cnr', 0.0):.2f} (낮음 기준 {of.CNR_LOW}) -> "
          f"{'대비 낮음' if 'LOW_CONTRAST' in r['flags'] else '정상'}")
    print(f"  색상 전처리: {'켜짐 (' + r['preprocess']['marking'] + ')' if r['preprocess']['is_color'] and r['preprocess']['marking'] != 'gray' else '끔 (회색 영상)'}")
    if r["stage"] == 0:
        print(line)
        print(f"최종 판정: {r['verdict']}   flags={r['flags']}")
        print("  글자 영역이 반사광으로 포화되어 점수를 계산하지 않았습니다.")
        print(line)
        return
    m = r["mechanical"]
    d, s = m["dot"], m["stencil"]
    print("[1단계] 기계식 마킹 패턴")
    print(f"  도트 매트릭스: {'검출' if m['is_dot_matrix'] else '아님'}  (점수 {d['score']:.2f}, 도트 {d['num_dots']}개"
          + (f", 반지름 {d['radius']:.1f}px, 피치 {d['pitch']:.1f}px" if d.get("radius") else "")
          + (f"; {d['reason']}" if d["reason"] else "") + ")")
    print(f"  스텐실       : {'검출' if m['is_stencil'] else '아님'}  (점수 {s['score']:.2f}, 다리 {s['num_bridges']}개"
          + (f"; {s['reason']}" if s["reason"] else "") + ")")
    if r["stage"] == 1:
        print("  -> 기계 패턴이 검출되어 2단계 없이 인쇄체로 조기 확정")
    else:
        print("[2단계] 특징 지표 (1 에 가까울수록 수기 쪽)")
        lay = r["raw"].get("layout")
        if lay:
            print(f"  배치: 줄 {len(lay['lines'])}개, 글자 높이 ≈ {lay['char_height'] or 0:.0f}px, "
                  f"가장자리 성분 {lay['removed_border']}개 제거")
        for k in of.FEATURES:
            v = r["features"][k]
            w = r["weights_used"].get(k, 0.0)
            note = ""
            if v is None:
                rr = r["raw"].get(k)
                note = (rr.get("reason") if isinstance(rr, dict) else getattr(rr, "reason", "")) or "계산 불가"
            print(f"  {k} {of.FEATURE_NAMES[k]:13s} {'   -  ' if v is None else f'{v:6.3f}'}   가중치 {w:.3f}"
                  + (f"   ({note})" if note else ""))
        print(f"  결합 점수 S = {r['score']:.3f}   (인쇄체 확정 ≤ {r['thresholds']['printed']:.2f}, "
              f"수기 확정 ≥ {r['thresholds']['handwritten']:.2f}, 사이는 판단 보류)")
    print(line)
    print(f"최종 판정: {r['verdict']}" + (f"   flags={r['flags']}" if r["flags"] else ""))
    print(f"참고 신뢰도 C = {r['confidence']:.2f}  (대비 {cp['contrast']:.2f} x 녹 {cp['mask']:.2f} x 지표 가용 "
          f"{cp['coverage']:.2f}; 판정에는 쓰지 않음)")
    print(line)


# ---------------------------------------------------------------------------
# 종합 그림
# ---------------------------------------------------------------------------
def _fit(img, width=PANEL_W, height=None):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    scale = width / img.shape[1]
    if height is not None:
        scale = min(scale, height / img.shape[0])
    out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.full((height or out.shape[0], width, 3), 255, np.uint8)
    canvas[:out.shape[0], :out.shape[1]] = out
    return canvas, scale


def _titled(panel, title):
    bar = np.full((28, panel.shape[1], 3), 40, np.uint8)
    cv2.putText(bar, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])


def panel_original(img_work, r):
    vis = img_work.copy() if img_work.ndim == 3 else cv2.cvtColor(img_work, cv2.COLOR_GRAY2BGR)
    v = cv2.cvtColor(vis, cv2.COLOR_BGR2HSV)[..., 2]
    vis[v > of.SPECULAR_V] = (0.5 * vis[v > of.SPECULAR_V] + [0, 0, 127]).astype(np.uint8)   # 포화 픽셀 붉게
    x0, y0, x1, y1 = r["text_box"]
    cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 200, 255), 3)
    return vis


def panel_mask(work, r):
    ink = _remove_specks(binarize(work))
    vis = cv2.cvtColor((ink * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    lay = r["raw"].get("layout") if r.get("raw") else None
    if lay:
        palette = [(255, 160, 60), (60, 200, 60), (60, 160, 255), (200, 80, 200), (60, 220, 220), (220, 220, 60)]
        for i, ln in enumerate(lay["lines"]):
            x0, y0, x1, y1 = ln["box"]
            cv2.rectangle(vis, (x0, y0), (x1, y1), palette[i % len(palette)], 3)
            cv2.putText(vis, f"line {i + 1}", (x0 + 4, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        palette[i % len(palette)], 2, cv2.LINE_AA)
    return vis


def panel_strokes(work, r):
    base = cv2.cvtColor((work * 0.5 + 127).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    raw = r.get("raw") or {}
    r7 = raw.get("phi_7")
    if isinstance(r7, dict):
        for c in r7.get("contours", []):
            if c.get("valid"):
                cv2.polylines(base, [np.asarray(c["points"]).astype(np.int32)], True, (0, 180, 0), 2, cv2.LINE_AA)
    r8 = raw.get("phi_8")
    if isinstance(r8, dict) and r8.get("skeleton") is not None:
        sk = cv2.dilate(r8["skeleton"].astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
        base[sk] = (255, 90, 0)
        for x, y in r8.get("junctions", []):
            cv2.circle(base, (int(x), int(y)), 7, (0, 0, 255), 2, cv2.LINE_AA)
        for x, y in r8.get("endpoints", []):
            cv2.circle(base, (int(x), int(y)), 6, (0, 200, 0), 2, cv2.LINE_AA)
    return base


def panel_scores(r, width):
    h = 300
    img = np.full((h, width, 3), 255, np.uint8)
    names = of.FEATURES + ["S"]
    vals = [r["features"].get(k) for k in of.FEATURES] + [r["score"]]
    left, top, bottom = 60, 30, h - 60
    bw = (width - left - 20) // len(names)
    th = r["thresholds"]
    ys = lambda v: int(bottom - v * (bottom - top))
    cv2.rectangle(img, (left, ys(th["handwritten"])), (width - 20, ys(th["printed"])), (210, 235, 250), -1)
    for t, label in ((th["printed"], "printed <="), (th["handwritten"], "handwritten >=")):
        cv2.line(img, (left, ys(t)), (width - 20, ys(t)), (0, 140, 230), 1)
        cv2.putText(img, f"{label} {t:.2f}", (width - 190, ys(t) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 110, 200),
                    1, cv2.LINE_AA)
    for t in (0.0, 0.5, 1.0):
        cv2.putText(img, f"{t:.1f}", (20, ys(t) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.line(img, (left, bottom), (width - 20, bottom), (90, 90, 90), 1)
    for i, (k, v) in enumerate(zip(names, vals)):
        x = left + i * bw
        color = (40, 40, 200) if k == "S" else (200, 120, 40)
        if v is None:
            cv2.putText(img, "n/a", (x + 8, bottom - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (x + 6, ys(v)), (x + bw - 6, bottom), color, -1)
            cv2.putText(img, f"{v:.2f}", (x + 6, ys(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
        label = "S" if k == "S" else f"p{k[-1]}"
        cv2.putText(img, label, (x + bw // 2 - 8, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1,
                    cv2.LINE_AA)
        if k != "S":
            w = r["weights_used"].get(k, 0.0)
            cv2.putText(img, f"w{w:.2f}", (x + 4, bottom + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (90, 90, 90) if w > 0 else (180, 180, 180), 1, cv2.LINE_AA)
    return img


def compose(img, r):
    work_color, _ = of.to_work_resolution(img)
    work = r["work_image"]
    rows_h = int(PANEL_W * work.shape[0] / work.shape[1])
    a, _ = _fit(panel_original(work_color, r), height=rows_h)
    b, _ = _fit(panel_mask(work, r), height=rows_h)
    c, _ = _fit(panel_strokes(work, r) if r["stage"] == 2 else cv2.cvtColor(work, cv2.COLOR_GRAY2BGR), height=rows_h)
    d, _ = _fit(draw_mechanical_debug(work, r["mechanical"]) if r["mechanical"] else
                cv2.cvtColor(work, cv2.COLOR_GRAY2BGR), height=rows_h)
    grid = np.vstack([np.hstack([_titled(a, "original (text box, saturated px in red)"),
                                 _titled(b, "binary mask + text lines")]),
                      np.hstack([_titled(c, "strokes: skeleton / contours / junctions"),
                                 _titled(d, "stage 1: dot / stencil detection")])])
    color = of.VERDICT_COLORS[r["verdict"]]
    head = np.full((70, grid.shape[1], 3), color, np.uint8)
    s = "n/a" if r["score"] is None else f"{r['score']:.3f}"
    cv2.putText(head, r["verdict"], (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    sub = f"stage {r['stage']}   S_hand = {s}   ref. C = {r['confidence']:.2f}" + (f"   flags: {', '.join(r['flags'])}"
                                                                                  if r["flags"] else "")
    cv2.putText(head, sub, (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([head, grid, panel_scores(r, grid.shape[1])])


def main():
    parser = argparse.ArgumentParser(description="실제 사진 한 장 수기/인쇄체 판별")
    parser.add_argument("image", help="입력 이미지 (JPG, PNG, HEIC ...)")
    parser.add_argument("-o", "--output", default="result_sample.png", help="종합 결과 이미지 경로")
    parser.add_argument("--color", default="off", choices=["off", "auto", "dark", "white", "yellow"],
                        help="0단계 색상 전처리 (기본 off)")
    args = parser.parse_args()
    if not os.path.exists(args.image):
        sys.exit(f"파일이 없습니다: {args.image}")
    img = read_image(args.image)
    r = of.evaluate_handwritten_score(img, color=args.color)
    print_report(args.image, img, r)
    ok, buf = cv2.imencode(".png", compose(img, r))
    buf.tofile(args.output)
    print(f"종합 결과 이미지: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
