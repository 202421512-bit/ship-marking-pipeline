"""Manual ground-truth stroke labelling (OpenCV brush). Nothing is pre-filled from predictions.

Usage: .venv\\Scripts\\python.exe scripts\\gt_label_tool.py <image name in data/field_test/images, e.g. 2.png>
  left drag  : paint STROKE (gt = 255)          right drag : erase
  i          : toggle IGNORE brush (ignore = 255, for undecidable pixels)
  + / -      : brush size        z : zoom 1x/2x/3x/4x      s : save        q / Esc : quit (asks nothing, unsaved work lost)
Saves data/field_test/masks/<stem>_gt.png and <stem>_ignore.png (original size, 0/255).
Mark the role of each labelled image in data/field_test/masks/split.csv (dev = used to tune, eval = untouched).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "data" / "field_test" / "images"
OUT = ROOT / "data" / "field_test" / "masks"


def main() -> int:
    name = sys.argv[1]
    alias = {f"example{i}": f"예시{i}.png" for i in (1, 2, 3)}       # ASCII aliases for batch files
    name = alias.get(Path(name).stem, name)
    if not name.lower().endswith(".png"):
        name += ".png"
    img = cv2.imdecode(np.fromfile(str(IMG / name), np.uint8), cv2.IMREAD_COLOR)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    gp, ip = OUT / f"{stem}_gt.png", OUT / f"{stem}_ignore.png"
    rd = lambda p: cv2.imdecode(np.fromfile(str(p), np.uint8), 0) if p.exists() else np.zeros(img.shape[:2], np.uint8)  # noqa: E731
    gt, ign = rd(gp), rd(ip)
    st = {"r": 2, "z": 3, "ignore": False, "draw": 0}

    def paint(x, y, val):
        target = ign if st["ignore"] else gt
        cv2.circle(target, (x // st["z"], y // st["z"]), st["r"], val, -1)

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            st["draw"] = 255
        elif ev == cv2.EVENT_RBUTTONDOWN:
            st["draw"] = -1
        elif ev in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            st["draw"] = 0
        if st["draw"]:
            paint(x, y, 255 if st["draw"] > 0 else 0)

    win = f"GT {name}"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        v = img.copy()
        v[gt > 0] = (0.4 * v[gt > 0] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
        v[ign > 0] = (0.4 * v[ign > 0] + 0.6 * np.array([255, 0, 255])).astype(np.uint8)
        v = cv2.resize(v, None, fx=st["z"], fy=st["z"], interpolation=cv2.INTER_NEAREST)
        cv2.putText(v, f"{'IGNORE' if st['ignore'] else 'STROKE'} r={st['r']} zoom={st['z']}  s=save q=quit", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imshow(win, v)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("i"):
            st["ignore"] = not st["ignore"]
        if k in (ord("+"), ord("=")):
            st["r"] += 1
        if k == ord("-"):
            st["r"] = max(0, st["r"] - 1)
        if k == ord("z"):
            st["z"] = st["z"] % 4 + 1
        if k == ord("s"):
            cv2.imencode(".png", gt)[1].tofile(str(gp))
            cv2.imencode(".png", ign)[1].tofile(str(ip))
            print("saved", gp, ip)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
