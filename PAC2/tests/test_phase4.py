"""Phase 4 tests: results sync, checkpoints, grouping, missing features, masks, outputs, figures, test-set guard."""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pac2 import load_config  # noqa: E402
from pac2.extraction import (CKPT_DIR, MODELS, TestSetAccessError, extract_image, group_components, guard_not_test,  # noqa: E402
                             load_checkpoint, predict_proba)
from pac2.features import FEATURES  # noqa: E402

CFG = load_config()
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
G3 = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9"]
G4 = ["G10", "G11", "G12", "G13", "G14", "G15"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_results_tree_exists():
    for t in ["phase3/figures", "phase3/tables", "phase4/figures", "phase4/extraction_examples", "phase4/tables", "final_presentation"]:
        assert (ROOT / "results" / t).is_dir(), t
    assert (ROOT / "results" / "manifest.csv").exists()


def test_g1_g9_copied_with_identical_hash():
    figs = ROOT / "results" / "phase3" / "figures"
    for g in G3:
        for ext in ("png", "svg"):
            dst = next(figs.glob(f"{g}_*.{ext}"))
            assert sha(dst) == sha(ROOT / "reports" / "phase3" / "figures" / dst.name)


def test_sync_is_idempotent():
    out = subprocess.run([PY, str(ROOT / "scripts" / "sync_presentation_results.py")], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0 and "COPIED" not in out.stdout and "CONFLICT" not in out.stdout


def test_checkpoints_load_and_are_dev_only():
    for m in MODELS:
        ck = load_checkpoint(m)
        assert ck["features"] == FEATURES and ck["trained_on"]["test_images"] == 0
        assert {"recall_priority", "precision_priority"} <= set(ck["thresholds"])


def test_missing_features_are_imputed_not_rejected():
    ck = load_checkpoint("S3_inner_cv_logistic")
    x = np.full((1, 7), np.nan)
    p = predict_proba(ck, x)
    assert np.isfinite(p).all()


def test_grouping_links_close_components_and_isolates_far_ones():
    m = np.zeros((100, 300), np.uint8)
    m[40:60, 20:26] = 255
    m[40:60, 32:38] = 255           # close to the first
    m[40:60, 250:256] = 255         # far away
    g = group_components(m)
    assert sorted(x["n_cc"] for x in g) == [1, 2]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("p4")
    cks = {m: load_checkpoint(m) for m in MODELS}
    sel = json.loads((CKPT_DIR / "selection.json").read_text(encoding="utf-8"))
    img = ROOT / "data" / "field_test" / "images" / "1.png"
    res = extract_image(img, cks, CFG, sel["dev_ranges"], sel["primary_model"], {"recall_priority": 0.3, "precision_priority": 0.7}, out)
    return img, out, res


def test_masks_same_size_binary_and_strokes_only(run):
    img, out, res = run
    h, w = cv2.imdecode(np.fromfile(str(img), np.uint8), 1).shape[:2]
    for n in ("candidate_mask", "accepted_mask", "rejected_mask", "final_mask"):
        m = cv2.imdecode(np.fromfile(str(out / f"{n}.png"), np.uint8), cv2.IMREAD_GRAYSCALE)
        assert m.shape == (h, w) and set(np.unique(m)) <= {0, 255}
    acc = cv2.imdecode(np.fromfile(str(out / "accepted_mask.png"), np.uint8), 0) > 0
    cand = cv2.imdecode(np.fromfile(str(out / "candidate_mask.png"), np.uint8), 0) > 0
    assert not (acc & ~cand).any()                      # never fills bounding boxes beyond candidate strokes


def test_overlay_and_csv_saved(run):
    _, out, res = run
    assert (out / "overlay.png").exists() and (out / "handwritten_only.png").exists()
    with open(out / "candidate_scores.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(res["groups"]) and all(f"p_{m}" in rows[0] for m in MODELS)


def test_g10_g15_figures_exist():
    figs = ROOT / "results" / "phase4" / "figures"
    for g in G4:
        assert list(figs.glob(f"{g}_*.png")) and list(figs.glob(f"{g}_*.svg")), g


def test_visualization_does_not_train():
    src = (ROOT / "scripts" / "phase4_visualize.py").read_text(encoding="utf-8") + \
          (ROOT / "scripts" / "generate_presentation_figures.py").read_text(encoding="utf-8")
    for bad in (".fit(", "TorchLogistic", "RandomForestClassifier", "fit_hard_example"):
        assert bad not in src


def test_test_set_access_is_blocked():
    with open(ROOT / "data" / "splits" / "split_manifest.csv", encoding="utf-8-sig") as f:
        t = next(r for r in csv.DictReader(f) if r["set"] == "test")
    with pytest.raises(TestSetAccessError):
        guard_not_test(ROOT / t["path"])
