"""Freeze the current recovery pipeline as versions/stable_baseline (code copies, config, checkpoint hashes, output masks).
Refuses to overwrite an existing snapshot. Run once: .venv\\Scripts\\python.exe scripts\\freeze_stable.py
"""
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "versions" / "stable_baseline"
CODE = ["src/pac2/recovery.py", "src/pac2/extraction.py", "src/pac2/candidates.py", "src/pac2/features.py", "src/pac2/normalize.py",
        "src/pac2/prep.py", "scripts/phase4_recovery.py", "configs/default.yaml"]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731


def main() -> int:
    if SNAP.exists():
        print("stable_baseline already frozen - not overwritten:", SNAP)
        return 0
    (SNAP / "code").mkdir(parents=True)
    files = {}
    for rel in CODE:
        dst = SNAP / "code" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
        files[rel] = sha(ROOT / rel)
    ck = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in sorted((ROOT / "models" / "phase4").rglob("*")) if p.is_file()}
    shutil.copytree(ROOT / "reports" / "phase4_recovery" / "inference", SNAP / "outputs",
                    ignore=shutil.ignore_patterns("*.csv", "*.json"), dirs_exist_ok=False)
    for d in (ROOT / "reports" / "phase4_recovery" / "inference").iterdir():
        for f in d.glob("*.csv"):
            shutil.copy2(f, SNAP / "outputs" / d.name / f.name)
        shutil.copy2(d / "run_info.json", SNAP / "outputs" / d.name / "run_info.json")
    sys.path.insert(0, str(ROOT / "src"))
    from pac2.recovery import RECOVERY_CFG
    sel = json.loads((ROOT / "models" / "phase4" / "selection.json").read_text(encoding="utf-8"))
    meta = {"version": "stable_baseline", "frozen_at": dt.datetime.now().isoformat(timespec="seconds"),
            "pipeline": "phase4_recovery (dual polarity + low-contrast + structure suppression + RF auxiliary reject)",
            "recovery_cfg": {k: list(v) if isinstance(v, tuple) else v for k, v in RECOVERY_CFG.items()},
            "primary_model": sel["primary_model"], "rf_threshold_rule": "min(recall_priority, precision_priority) of the checkpoint",
            "code_sha256": files, "checkpoint_sha256": ck,
            "output_masks_sha256": {str(p.relative_to(SNAP)).replace("\\", "/"): sha(p) for p in sorted((SNAP / "outputs").rglob("final_mask.png"))},
            "known_weaknesses": ["V8 thick marker mostly missed", "machinery / plate edges partly kept (field 1, field 2)",
                                 "W79 '79' and '1' partly removed as structure", "parameters tuned while viewing the 7 field images (demo, not generalization)"],
            "reproduce": ".venv\\Scripts\\python.exe scripts\\extract.py --version stable"}
    (SNAP / "VERSION.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("frozen:", SNAP, "| code files", len(files), "| checkpoints", len(ck), "| masks", len(meta["output_masks_sha256"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
