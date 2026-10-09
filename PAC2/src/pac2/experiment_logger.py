"""Experiment logger: every row written to a CSV carries experiment id, seed, model, fold and config."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Dict, List


class ExperimentLogger:
    def __init__(self, out_dir: Path, seed: int, config: dict):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        cfg_hash = hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:8]
        self.experiment_id = f"phase3_{dt.datetime.now():%Y%m%d_%H%M%S}_{cfg_hash}"
        self.seed = seed
        self.tables: Dict[str, List[dict]] = {}
        (self.out_dir / "experiment_meta.json").write_text(json.dumps(
            {"experiment_id": self.experiment_id, "seed": seed, "config": config}, indent=1, default=str), encoding="utf-8")

    def log(self, table: str, model: str, fold, settings: dict, rows):
        rows = rows if isinstance(rows, list) else [rows]
        meta = {"experiment_id": self.experiment_id, "seed": self.seed, "model": model, "fold": fold,
                "settings": json.dumps(settings, sort_keys=True, default=str)}
        self.tables.setdefault(table, []).extend({**meta, **r} for r in rows)

    def save(self):
        for name, rows in self.tables.items():
            cols = list(dict.fromkeys(k for r in rows for k in r))
            with open(self.out_dir / f"{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)
