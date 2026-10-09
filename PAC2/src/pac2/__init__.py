"""PAC2 - handwritten marking stroke extraction with 7 shape features."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict:
    with open(path or ROOT / "configs" / "default.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
