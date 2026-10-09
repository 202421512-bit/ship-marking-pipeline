"""EXPERIMENTAL changes on top of the frozen stable recovery pipeline (stable code is not modified).

E1 local structure suppression: a stable-final component is removed only if it is NOT protected and
   (touches the image border AND its long side >= e1_len_frac of the image side AND >= e1_edge_overlap of its pixels lie on
   large-object edges). Protected = components that do not touch the border (handwriting zone proxy; NOT ground truth).
E2 thick-stroke recovery: geodesic reconstruction from the stable final strokes (seeds) into a thick-stroke mask =
   same-polarity contrast to a wide local background (window e2_bg_window) above a robust threshold, restricted to a
   dilation radius e2_max_grow around the seeds - recovers the body of thick marker strokes whose cores were found.
   Growth is forbidden on stable's removed structure (dilated 2 px) and on large-object edges (added after the first
   comparison showed E2 regrowing the example1 seam and field 1 floor edges).
Both outputs and the protection / structure masks are returned separately.
"""
from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from .recovery import _thr

EXP_CFG = {"e1_len_frac": 0.25, "e1_edge_overlap": 0.30, "border_px": 4, "e2_bg_window": 61, "e2_k": 3.0, "e2_max_grow": 12}


def experimental(stable: Dict[str, object], final: np.ndarray, cfg: dict = EXP_CFG) -> Dict[str, np.ndarray]:
    ch = stable["channels"]
    H, W = final.shape
    sedge = stable["structure_edges"]
    n, lab, st, _ = cv2.connectedComponentsWithStats(final.astype(np.uint8), connectivity=8)
    protect = np.zeros_like(final)
    struct = np.zeros_like(final)
    b = cfg["border_px"]
    for i in range(1, n):
        sel = lab == i
        x0, y0, w, h = st[i, 0], st[i, 1], st[i, 2], st[i, 3]
        border = x0 <= b or y0 <= b or x0 + w >= W - b or y0 + h >= H - b
        if not border:
            protect |= sel
            continue
        if max(w, h) >= cfg["e1_len_frac"] * max(H, W) and float(sedge[sel].mean()) >= cfg["e1_edge_overlap"]:
            struct |= sel
    e1 = final & ~struct
    # E2 thick strokes
    gray = ch["gray"].astype(np.float32)
    bg = cv2.medianBlur(ch["gray"], cfg["e2_bg_window"]).astype(np.float32)
    thick = np.zeros_like(final)
    for pol, sign in (("bright", 1.0), ("dark", -1.0)):
        c = sign * (gray - bg)
        thick |= (c > max(4.0, _thr(c, cfg["e2_k"])))
    r = cfg["e2_max_grow"]
    zone = cv2.dilate(e1.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(bool)
    rem = stable["removed"]
    stable_struct = rem["structure_segment"] | rem["structure_large"] | rem["structure_edge"]
    no_grow = cv2.dilate(stable_struct.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) | sedge
    allowed = (thick & zone & ~struct & ~no_grow) | e1
    nl, labl = cv2.connectedComponents(allowed.astype(np.uint8), connectivity=8)
    hit = np.zeros(nl, bool)
    hit[np.unique(labl[e1])] = True
    hit[0] = False
    e2 = hit[labl]
    return {"final": e2, "e1_only": e1, "protect": protect, "structure_removed": struct, "thick_added": e2 & ~e1}
