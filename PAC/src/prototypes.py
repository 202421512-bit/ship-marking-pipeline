"""Prototype statistical reference model.

This is NOT training. For each class, reference glyphs are rendered from system
fonts with Pillow, augmented (rotation, scaling, translation, dilation,
erosion, blur, threshold variation), and the topology + geometry features of
every sample are summarized by per-class mean and standard deviation.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import PROTOTYPE_DIR, AppConfig
from .geometry import GeometryResult, analyze_geometry, geometry_feature_dict
from .image_io import fallback_font, find_font, render_glyph
from .preprocessing import mask_from_glyph
from .topology import TopologyResult, analyze_topology, topology_feature_dict


@dataclass
class FeatureBundle:
    """Topology + geometry analysis of one mask and its flat feature vector."""

    topology: TopologyResult
    geometry: GeometryResult
    features: Dict[str, float]


def extract_features(mask: np.ndarray, cfg: AppConfig, with_persistence: bool = False) -> FeatureBundle:
    """Run topology and geometry analysis and merge their feature dicts."""
    topo = analyze_topology(mask, cfg.topology, with_persistence=with_persistence)
    geo = analyze_geometry(mask, topo)
    feats = topology_feature_dict(topo, cfg.preprocessing.normalized_size)
    feats.update(geometry_feature_dict(geo))
    return FeatureBundle(topology=topo, geometry=geo, features=feats)


@dataclass
class PrototypeModel:
    """Per-class feature statistics of the reference prototypes."""

    classes: List[str]
    feature_names: List[str]
    mean: Dict[str, Dict[str, float]]
    std: Dict[str, Dict[str, float]]
    n_samples: Dict[str, int]
    fonts_used: List[str]
    metadata: Dict[str, object] = field(default_factory=dict)
    templates: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def to_json(self) -> Dict[str, object]:
        """JSON-serializable form (templates are stored separately as npz)."""
        data = asdict(self)
        data.pop("templates", None)
        return data


def _augment(gray: np.ndarray, rng: np.random.Generator, cfg: AppConfig) -> Tuple[np.ndarray, int]:
    """Apply one random augmentation; returns (gray, threshold)."""
    p = cfg.prototypes
    h, w = gray.shape
    angle = float(rng.uniform(-p.rotation_deg, p.rotation_deg))
    scale = float(rng.uniform(*p.scale_range))
    tx, ty = rng.integers(-p.translation_px, p.translation_px + 1, size=2)
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    matrix[:, 2] += (float(tx), float(ty))
    out = cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    roll = rng.random()
    kernel = np.ones((3, 3), np.uint8)
    if roll < p.morph_probability / 2:
        out = cv2.erode(out, kernel)       # ink is dark: erode = thicker strokes (dilation of ink)
    elif roll < p.morph_probability:
        out = cv2.dilate(out, kernel)      # thinner strokes (erosion of ink)
    sigma = float(rng.uniform(*p.blur_sigma_range))
    if sigma > 0.3:
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    thr = int(rng.integers(p.threshold_range[0], p.threshold_range[1] + 1))
    return out, thr


def _available_fonts(cfg: AppConfig) -> List[Path]:
    """Resolve configured fonts; fall back to DejaVu (matplotlib) if none exist."""
    fonts = [f for f in (find_font(name) for name in cfg.prototypes.fonts) if f is not None]
    if not fonts:
        fb = fallback_font()
        if fb is not None:
            fonts = [fb]
    return fonts


def build_prototype_model(cfg: AppConfig, verbose: bool = True) -> PrototypeModel:
    """Render, augment and summarize prototypes for every configured class."""
    p = cfg.prototypes
    rng = np.random.default_rng(p.seed)
    fonts = _available_fonts(cfg)
    font_list: List[Optional[Path]] = list(fonts) if fonts else [None]
    samples: Dict[str, List[Dict[str, float]]] = {c: [] for c in cfg.classes}
    templates: Dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    for label in cfg.classes:
        acc = np.zeros((cfg.preprocessing.normalized_size,) * 2, dtype=np.float32)
        for font in font_list:
            base = render_glyph(label, font, p.render_canvas, p.render_font_size)
            variants = [(base, 128)] + [_augment(base, rng, cfg) for _ in range(p.augmentations_per_font)]
            for gray, thr in variants:
                try:
                    mask = mask_from_glyph(gray, thr, cfg.preprocessing)
                    samples[label].append(extract_features(mask, cfg).features)
                    acc += mask
                except Exception:
                    continue  # a degenerate augmentation is simply skipped
        templates[label] = acc / max(1, len(samples[label]))
    names = list(next(iter(samples.values()))[0].keys())
    mean, std = {}, {}
    for label, rows in samples.items():
        arr = np.array([[r[n] for n in names] for r in rows], dtype=np.float64)
        mean[label] = {n: float(v) for n, v in zip(names, arr.mean(axis=0))}
        std[label] = {n: float(v) for n, v in zip(names, arr.std(axis=0))}
    model = PrototypeModel(
        classes=list(cfg.classes),
        feature_names=names,
        mean=mean,
        std=std,
        n_samples={k: len(v) for k, v in samples.items()},
        fonts_used=[f.name if f else "PIL default" for f in font_list],
        metadata={
            "description": "prototype statistical reference model (not trained OCR)",
            "cache_version": p.cache_version,
            "seed": p.seed,
            "augmentations_per_font": p.augmentations_per_font,
            "build_seconds": round(time.perf_counter() - t0, 3),
        },
        templates=templates,
    )
    if verbose:
        total = sum(model.n_samples.values())
        print(f"[PROTOTYPES] built {total} samples, {len(model.classes)} classes, "
              f"{len(model.fonts_used)} fonts in {model.metadata['build_seconds']} s")
    return model


def _cache_signature(cfg: AppConfig) -> Dict[str, object]:
    """Values that invalidate the cache when changed."""
    return {"classes": list(cfg.classes), "cache_version": cfg.prototypes.cache_version,
            "fonts": list(cfg.prototypes.fonts), "seed": cfg.prototypes.seed,
            "augmentations_per_font": cfg.prototypes.augmentations_per_font,
            "canvas": cfg.preprocessing.normalized_size}


def _class_key(label: str) -> str:
    """npz-safe key for any character label ('/', '.', '°' ...)."""
    return "u" + "_".join(f"{ord(c):04x}" for c in label)


def _summarize_rows(rows: List[Dict[str, object]], names: List[str]) -> Tuple[Dict, Dict, Dict]:
    """Per-class mean / std / count over feature rows that carry a 'class' field."""
    by_class: Dict[str, List[List[float]]] = {}
    for r in rows:
        by_class.setdefault(str(r["class"]), []).append([float(r[n]) for n in names])
    mean, std, count = {}, {}, {}
    for label, values in by_class.items():
        arr = np.array(values, dtype=np.float64)
        mean[label] = {n: float(v) for n, v in zip(names, arr.mean(axis=0))}
        std[label] = {n: float(v) for n, v in zip(names, arr.std(axis=0))}
        count[label] = len(values)
    return mean, std, count


def build_data_fitted_model(cfg: AppConfig, verbose: bool = True) -> Dict[str, object]:
    """DATA-FITTED REFERENCE MODEL from manually transcribed, count-validated real images.

    Re-runs the dataset audit (so newly typed transcriptions are validated), extracts
    topology + geometry features of every character segment of trainable images,
    and stores per-class statistics: Model A (class) and Model B (class + style).
    Classes with fewer than ``min_samples_per_class`` samples are reported but not
    used. If nothing is trainable, any stale data-fitted model is removed and the
    pipeline keeps the synthetic font-prototype model.
    """
    from .config import (DATA_FITTED_MODEL_PATH, DATA_FITTED_TEMPLATES_PATH, RESULTS_DIR,
                         TRAINING_FEATURES_PATH)
    from .dataset import run_dataset_audit, write_csv_rows
    from .preprocessing import normalize_size

    audit = run_dataset_audit(cfg, verbose=verbose)
    rows: List[Dict[str, object]] = []
    accum: Dict[str, np.ndarray] = {}
    for m in audit.manifest:
        if m["valid_for_training"] is not True:
            continue
        seg = audit.segmentations.get(str(m["image_path"]))
        text = str(m["normalized_transcription"])
        if seg is None or seg.count != len(text):
            continue
        for idx, (char_mask, label) in enumerate(zip(seg.char_masks, text)):
            try:
                gray = np.where(char_mask, 0, 255).astype(np.float32)
                mask_n, _, _, _ = normalize_size(char_mask, char_mask.astype(np.float32), gray, cfg.preprocessing)
                bundle = extract_features(mask_n, cfg)
            except Exception as exc:
                print(f"[TRAIN] skip {m['image_path']} char {idx}: {exc}")
                continue
            rows.append({"image_path": m["image_path"], "style": m["style"], "group_id": m["group_id"],
                         "variant": m["variant"], "char_index": idx, "class": label,
                         **{k: round(v, 6) for k, v in bundle.features.items()}})
            accum[label] = accum.get(label, np.zeros_like(mask_n, dtype=np.float32)) + mask_n

    names = list(rows[0].keys())[6:] if rows else []
    columns = ["image_path", "style", "group_id", "variant", "char_index", "class"] + names
    write_csv_rows(TRAINING_FEATURES_PATH, columns, rows)
    write_csv_rows(RESULTS_DIR / "training_features.csv", columns, rows)

    min_n = cfg.dataset.min_samples_per_class
    mean_a, std_a, count_a = _summarize_rows(rows, names) if rows else ({}, {}, {})
    used = sorted(c for c, n in count_a.items() if n >= min_n)
    by_style: Dict[str, Dict[str, object]] = {}
    for style in sorted({str(r["style"]) for r in rows}):
        ms, ss, cs = _summarize_rows([r for r in rows if r["style"] == style], names)
        by_style[style] = {c: {"n": cs[c], "mean": ms[c], "std": ss[c]} for c in sorted(cs)}
    stats = {
        "model": "DATA-FITTED REFERENCE MODEL",
        "status": "BUILT" if len(used) >= 2 else "NOT_BUILT",
        "reason": None if len(used) >= 2 else
        f"need >= 2 classes with >= {min_n} validated samples (have {len(used)})",
        "training_images": sum(1 for m in audit.manifest if m["valid_for_training"] is True),
        "training_characters": len(rows),
        "min_samples_per_class": min_n,
        "classes_used": used,
        "classes_excluded_insufficient": {c: n for c, n in sorted(count_a.items()) if n < min_n},
        "feature_names": names,
        "model_A_class": {c: {"n": count_a[c], "mean": mean_a[c], "std": std_a[c]} for c in used},
        "model_B_class_style": by_style,
        "note": "statistics of real, manually transcribed samples; small n -> std is unreliable (sigma floors apply)",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "model_statistics.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")
    if stats["status"] == "BUILT":
        model = PrototypeModel(
            classes=used, feature_names=names,
            mean={c: mean_a[c] for c in used}, std={c: std_a[c] for c in used},
            n_samples={c: count_a[c] for c in used}, fonts_used=[],
            metadata={"source": "DATA_FITTED", "description": "data-fitted reference model (real Symbols/ data)",
                      "training_characters": len(rows), "min_samples_per_class": min_n},
        )
        DATA_FITTED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        DATA_FITTED_MODEL_PATH.write_text(json.dumps(model.to_json(), indent=2, ensure_ascii=False),
                                          encoding="utf-8")
        np.savez_compressed(DATA_FITTED_TEMPLATES_PATH,
                            **{_class_key(c): accum[c] / count_a[c] for c in used})
    else:
        for stale in (DATA_FITTED_MODEL_PATH, DATA_FITTED_TEMPLATES_PATH):
            if stale.exists():
                stale.unlink()
    if verbose:
        print(f"[TRAIN] training_features.csv rows = {len(rows)} | classes used = {len(used)} "
              f"| model status = {stats['status']}" + (f" ({stats['reason']})" if stats["reason"] else ""))
        if rows:
            print(f"[TRAIN] Total samples = {len(rows)}  "
                  + "  ".join(f"{k} = {v}" for k, v in sorted(
                      {s: sum(1 for r in rows if r['style'] == s) for s in by_style}.items()))
                  + f"  Classes = {len(count_a)}")
    return stats


def load_reference_model(cfg: AppConfig, rebuild: bool = False) -> PrototypeModel:
    """Reference model used by the single-image pipeline.

    'auto': the data-fitted model when it exists (>= 2 classes), otherwise the
    synthetic font-prototype model. 'data' fails over to synthetic with a warning.
    """
    from .config import DATA_FITTED_MODEL_PATH, DATA_FITTED_TEMPLATES_PATH

    if cfg.reference_model in ("auto", "data") and DATA_FITTED_MODEL_PATH.exists():
        try:
            data = json.loads(DATA_FITTED_MODEL_PATH.read_text(encoding="utf-8"))
            templates: Dict[str, np.ndarray] = {}
            if DATA_FITTED_TEMPLATES_PATH.exists():
                with np.load(DATA_FITTED_TEMPLATES_PATH) as npz:
                    templates = {c: npz[_class_key(c)] for c in data["classes"] if _class_key(c) in npz.files}
            if len(data["classes"]) >= 2:
                return PrototypeModel(classes=data["classes"], feature_names=data["feature_names"],
                                      mean=data["mean"], std=data["std"], n_samples=data["n_samples"],
                                      fonts_used=[], metadata=data["metadata"], templates=templates)
        except Exception as exc:
            print(f"[MODEL] data-fitted model unreadable ({exc}); using synthetic prototypes.")
    elif cfg.reference_model in ("data", "auto"):
        print("[MODEL] WARNING: no data-fitted model (run --train-reference); falling back to SYNTHETIC font "
              "prototypes (16 classes 0-9, A-F) - not a real-data model.")
    model = load_or_build_prototypes(cfg, rebuild=rebuild)
    model.metadata.setdefault("source", "SYNTHETIC_FONT_PROTOTYPE")
    return model


def load_or_build_prototypes(cfg: AppConfig, rebuild: bool = False,
                             directory: Path = PROTOTYPE_DIR) -> PrototypeModel:
    """Load the cached prototype model, rebuilding when config changed."""
    directory.mkdir(parents=True, exist_ok=True)
    stats_path = directory / cfg.prototypes.cache_file
    tmpl_path = directory / cfg.prototypes.template_file
    signature = _cache_signature(cfg)
    if not rebuild and stats_path.exists() and tmpl_path.exists():
        try:
            data = json.loads(stats_path.read_text(encoding="utf-8"))
            if data.get("signature") == signature:
                with np.load(tmpl_path) as npz:
                    templates = {k: npz[k] for k in npz.files}
                return PrototypeModel(
                    classes=data["classes"], feature_names=data["feature_names"],
                    mean=data["mean"], std=data["std"], n_samples=data["n_samples"],
                    fonts_used=data["fonts_used"], metadata=data["metadata"],
                    templates=templates,
                )
        except Exception as exc:
            print(f"[PROTOTYPES] cache unreadable ({exc}); rebuilding.")
    model = build_prototype_model(cfg)
    try:
        payload = model.to_json()
        payload["signature"] = signature
        stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        np.savez_compressed(tmpl_path, **model.templates)
    except OSError as exc:
        print(f"[PROTOTYPES] could not write cache: {exc}")
    return model
