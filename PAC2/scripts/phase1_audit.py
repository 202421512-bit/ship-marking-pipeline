"""Phase 1: audit of Source/formal (20) and Source/handwritten (100). Read-only on Source/.

Writes reports/phase1/: audit.csv, duplicates.csv, contact sheets, summary.txt
Run: .venv\\Scripts\\python.exe scripts\\phase1_audit.py
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Source"
OUT = ROOT / "reports" / "phase1"
CLASSES = {"formal": 0, "handwritten": 1}


def read(path: Path):
    buf = np.fromfile(str(path), np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def dhash(gray: np.ndarray, n: int = 16) -> int:
    s = cv2.resize(gray, (n + 1, n), interpolation=cv2.INTER_AREA)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def stroke_stats(gray: np.ndarray) -> dict:
    """Polarity-agnostic quick look: bright (top-hat) and dark (black-hat) thin-structure evidence."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    out = {}
    for name, op in (("bright", cv2.MORPH_TOPHAT), ("dark", cv2.MORPH_BLACKHAT)):
        r = cv2.morphologyEx(gray, op, k)
        thr, m = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        out[f"{name}_tophat_p99"] = float(np.percentile(r, 99))
        out[f"{name}_mask_frac"] = float((m > 0).mean())
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, hashes = [], []
    for cls, label in CLASSES.items():
        for p in sorted((SRC / cls).iterdir()):
            r = {"image_id": f"{cls}/{p.name}", "class": cls, "label": label, "file": str(p.relative_to(ROOT)),
                 "ext": p.suffix.lower(), "bytes": p.stat().st_size}
            img = read(p) if p.is_file() else None
            if img is None:
                r.update(status="UNREADABLE")
                rows.append(r)
                continue
            ch = 1 if img.ndim == 2 else img.shape[2]
            bgr = img if ch == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if ch == 1 else cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            sat = float(hsv[..., 1].mean())
            r.update(status="OK", width=img.shape[1], height=img.shape[0], channels=ch, dtype=str(img.dtype),
                     alpha_present=ch == 4, sha256=hashlib.sha256(p.read_bytes()).hexdigest()[:16],
                     gray_mean=round(float(gray.mean()), 1), gray_std=round(float(gray.std()), 1),
                     saturation_mean=round(sat, 1), near_grayscale=sat < 12,
                     laplacian_var=round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1))
            r.update({k: round(v, 4) for k, v in stroke_stats(gray).items()})
            rows.append(r)
            hashes.append((r["image_id"], dhash(gray), r["sha256"], gray))
    # duplicates: exact (sha) and near (dHash hamming <= 12 of 256 bits)
    dup = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            a, b = hashes[i], hashes[j]
            ham = bin(a[1] ^ b[1]).count("1")
            if a[2] == b[2] or ham <= 12:
                dup.append({"a": a[0], "b": b[0], "exact_same_file": a[2] == b[2], "dhash_hamming_256": ham})
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / "audit.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "duplicates.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["a", "b", "exact_same_file", "dhash_hamming_256"])
        w.writeheader()
        w.writerows(dup)
    # contact sheets (display only, aspect preserved, 160x160 tiles)
    for cls in CLASSES:
        items = [r for r in rows if r["class"] == cls and r["status"] == "OK"]
        tiles = []
        for r in items:
            img = read(ROOT / r["file"])
            img = img if img.ndim == 3 and img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR if img.ndim == 2 else cv2.COLOR_BGRA2BGR)
            s = 160 / max(img.shape[:2])
            t = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
            tile = np.full((180, 160, 3), 255, np.uint8)
            tile[:t.shape[0], :t.shape[1]] = t
            cv2.putText(tile, Path(r["file"]).stem[:22], (2, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 200), 1)
            tiles.append(tile)
        per = 10
        for s0 in range(0, len(tiles), 50):
            chunk = tiles[s0:s0 + 50]
            while len(chunk) % per:
                chunk.append(np.full((180, 160, 3), 255, np.uint8))
            sheet = np.vstack([np.hstack(chunk[k:k + per]) for k in range(0, len(chunk), per)])
            cv2.imencode(".png", sheet)[1].tofile(str(OUT / f"contact_{cls}_{s0 // 50 + 1}.png"))
    ok = [r for r in rows if r["status"] == "OK"]
    counts = ", ".join(f"{c}={sum(r['class'] == c for r in rows)}" for c in CLASSES)
    lines = [f"files: {len(rows)} ({counts})",
             f"unreadable: {sum(r['status'] != 'OK' for r in rows)}",
             f"extensions: {sorted({r['ext'] for r in rows})}",
             f"channels: {sorted({r.get('channels') for r in ok})}, alpha present: {sum(bool(r.get('alpha_present')) for r in ok)}"]
    for c in CLASSES:
        rs = [r for r in ok if r["class"] == c]
        W = np.array([r["width"] for r in rs]); Hh = np.array([r["height"] for r in rs])
        lines.append(f"{c}: width {W.min()}-{W.max()} (median {int(np.median(W))}), height {Hh.min()}-{Hh.max()} (median {int(np.median(Hh))}), "
                     f"gray mean {np.mean([r['gray_mean'] for r in rs]):.1f}, gray std {np.mean([r['gray_std'] for r in rs]):.1f}, "
                     f"saturation {np.mean([r['saturation_mean'] for r in rs]):.1f}, near-grayscale {sum(r['near_grayscale'] for r in rs)}, "
                     f"sharpness(lap var) median {np.median([r['laplacian_var'] for r in rs]):.0f}")
    lines.append(f"exact duplicate pairs: {sum(d['exact_same_file'] for d in dup)}; near-duplicate pairs (dHash<=12/256): {len(dup)}")
    cross = [d for d in dup if d["a"].split("/")[0] != d["b"].split("/")[0]]
    lines.append(f"near-duplicates across classes: {len(cross)}")
    (OUT / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
