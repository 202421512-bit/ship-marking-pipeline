"""End-to-end analysis pipeline with timing and traceability.

input -> preprocessing -> topology -> geometry -> restoration -> (topology,
geometry of restored) -> bayesian -> vlm -> fusion -> decision -> outputs
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional

import numpy as np

from .bayesian import BayesianResult, BayesianShapeClassifier
from .config import PROJECT_ROOT, RESULTS_DIR, AppConfig, PROTOTYPE_LABEL
from .decision import DecisionResult, decide
from .fusion import FusionResult, fuse
from .image_io import LoadedImage, load_image, save_image
from .preprocessing import PreprocessingResult, preprocess
from .prototypes import FeatureBundle, PrototypeModel, extract_features, load_reference_model
from .restoration import RestorationResult, restore, write_candidates_csv
from .vlm import VLMResult, verify

ProgressFn = Callable[[str], None]


@dataclass
class PipelineResult:
    """Every intermediate object of one analysis run."""

    run_id: str
    loaded: LoadedImage
    preprocessing: PreprocessingResult
    observed: FeatureBundle
    restoration: RestorationResult
    restored: FeatureBundle
    bayesian: BayesianResult            # after restoration (final Bayesian evidence)
    bayesian_before: BayesianResult     # on observed features, before restoration
    vlm: VLMResult
    fusion: FusionResult
    decision: DecisionResult
    model: PrototypeModel
    timing_ms: Dict[str, float]
    output_dir: Path
    files: Dict[str, str] = field(default_factory=dict)
    result_json: Dict[str, object] = field(default_factory=dict)
    extra_input_info: Dict[str, object] = field(default_factory=dict)
    config: Optional[AppConfig] = None


@contextmanager
def _timed(timing: Dict[str, float], name: str) -> Iterator[None]:
    """Record the wall time of a block in milliseconds."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timing[name] = round((time.perf_counter() - t0) * 1000.0, 2)


def _rel(path: Path) -> str:
    """Project-relative path string when possible (portable traceability)."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_pipeline(input_path: Path, cfg: AppConfig, use_vlm: bool = True,
                 output_dir: Path = RESULTS_DIR, progress: Optional[ProgressFn] = None,
                 rebuild_prototypes: bool = False, extra_input_info: Optional[Dict[str, object]] = None,
                 make_figures: bool = True) -> PipelineResult:
    """Run the full analysis on one cropped marking image.

    Args:
        input_path: image file (never modified).
        cfg: application configuration.
        use_vlm: call the VLM when an API key is available.
        output_dir: where figures, result.json and CSV are written.
        progress: optional callback receiving stage names (GUI).
        rebuild_prototypes: force rebuilding the prototype reference model.
        extra_input_info: metadata merged into result.json["input"] (e.g. demo info).
        make_figures: render PNG figures.
    """
    say = progress or (lambda _msg: None)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = output_dir / "artifacts"
    artifacts.mkdir(exist_ok=True)
    timing: Dict[str, float] = {}
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]

    say("Loading prototype reference model")
    with _timed(timing, "Prototype model (load/build)"):
        model = load_reference_model(cfg, rebuild=rebuild_prototypes)
        clf = BayesianShapeClassifier(model, cfg.bayesian)
        classes = list(model.classes)  # labels that exist in the reference model actually used

    say("Loading input")
    loaded = load_image(Path(input_path))

    say("Preprocessing")
    with _timed(timing, "Preprocessing"):
        pre = preprocess(loaded.image, cfg.preprocessing)

    say("Topology analysis")
    with _timed(timing, "Topology"):
        from .topology import analyze_topology

        observed_topo = analyze_topology(pre.mask, cfg.topology, with_persistence=True)

    say("Geometry analysis")
    with _timed(timing, "Geometry"):
        observed = extract_features(pre.mask, cfg)
        observed.topology = observed_topo

    say("Bayesian inference (before restoration)")
    with _timed(timing, "Bayesian (before restoration)"):
        bayes_before = clf.predict(observed.features)

    say("Topology-constrained restoration")
    with _timed(timing, "Restoration"):
        rest = restore(pre.mask, pre.soft, clf, cfg, observed, preliminary=bayes_before)
        restored = extract_features(rest.best.mask, cfg, with_persistence=True)

    say("Bayesian inference (after restoration)")
    with _timed(timing, "Bayesian"):
        # Fresh, independent evaluation on the restored mask's re-extracted features.
        bayes = clf.predict(restored.features)

    say("VLM verification")
    with _timed(timing, "VLM"):
        vlm = verify(pre.stages["1_grayscale"], rest.best.mask, classes, cfg.vlm, enabled=use_vlm)

    say("Evidence fusion")
    with _timed(timing, "Fusion"):
        fusion = fuse(bayes, vlm, classes, cfg.fusion)

    with _timed(timing, "Decision"):
        decision = decide(bayes, vlm, fusion, restored.features, clf, cfg.decision,
                          bayes_before=bayes_before, observed_features=observed.features)

    result = PipelineResult(
        run_id=run_id, loaded=loaded, preprocessing=pre, observed=observed, restoration=rest,
        restored=restored, bayesian=bayes, bayesian_before=bayes_before, vlm=vlm, fusion=fusion, decision=decision, model=model,
        timing_ms=timing, output_dir=output_dir, extra_input_info=dict(extra_input_info or {}),
        config=cfg,
    )

    say("Saving artifacts")
    files = {
        "observed_mask": save_image(artifacts / "observed_mask.png", ~pre.mask),
        "restored_mask": save_image(artifacts / "restored_mask.png", ~rest.best.mask),
        "skeleton": save_image(artifacts / "restored_skeleton.png", restored.topology.skeleton),
        "restoration_candidates_csv": write_candidates_csv(rest, output_dir / "restoration_candidates.csv"),
    }
    result.files = {k: _rel(v) for k, v in files.items()}

    if make_figures:
        say("Rendering figures")
        with _timed(timing, "Visualization"):
            from .visualization import render_all

            figs = render_all(result)
        result.files.update({k: _rel(v) for k, v in figs.items()})

    timing["Total (analysis stages)"] = round(sum(
        timing.get(k, 0.0) for k in ("Preprocessing", "Topology", "Geometry", "Bayesian (before restoration)",
                                     "Restoration", "Bayesian", "VLM", "Fusion", "Decision")), 2)
    result.result_json = build_result_json(result, cfg)
    json_path = output_dir / "result.json"
    json_path.write_text(json.dumps(result.result_json, indent=2, ensure_ascii=False,
                                    default=json_default), encoding="utf-8")
    result.files["result_json"] = _rel(json_path)
    say("Done")
    return result


def build_result_json(r: PipelineResult, cfg: AppConfig) -> Dict[str, object]:
    """Structured, traceable output for downstream semantic / welding modules."""
    pre_info = dict(r.preprocessing.info)
    return {
        "schema_version": "pac-m3-marking-0.1",
        "label": PROTOTYPE_LABEL,
        "reference_model_source": r.model.metadata.get("source", "SYNTHETIC_FONT_PROTOTYPE"),
        "evaluation_type": ("SYNTHETIC DEMO" if r.extra_input_info.get("demo") else "SINGLE IMAGE")
                           + " (not a real-dataset evaluation)",
        "run_id": r.run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "raw_image": _rel(r.loaded.path),
        "recognized_mark": r.fusion.final_candidate,
        "recognition_confidence": round(r.fusion.fusion_score, 6),
        "recognition_confidence_type": (
            "Bayesian posterior (prototype reference model)" if r.vlm.status != "OK"
            else "heuristic fusion score (Bayesian posterior x VLM reported confidence)"),
        "recognition_confidence_calibrated": False,
        "decision": r.decision.final_decision,
        "decision_level": r.decision.level,
        "semantic_status": "NOT_IMPLEMENTED",
        "welding_status": "NOT_IMPLEMENTED",
        "input": {
            "path": _rel(r.loaded.path), "sha256": r.loaded.sha256,
            "width": r.loaded.width, "height": r.loaded.height, "channels": r.loaded.channels,
            "original_modified": False, **r.extra_input_info,
        },
        "preprocessing": {"stage_id": "S1", "input_ref": "input", **pre_info,
                          "output_artifact": r.files.get("observed_mask")},
        "topology": {
            "stage_id": "S2", "input_ref": "S1",
            "observed": r.observed.topology.summary(),
            "restored": {**r.restored.topology.summary(), "input_ref": "S4"},
        },
        "geometry": {
            "stage_id": "S3", "input_ref": "S1",
            "observed": r.observed.geometry.summary(),
            "restored": {**r.restored.geometry.summary(), "input_ref": "S4"},
        },
        "restoration": {"stage_id": "S4", "input_ref": ["S1", "S2", "S3"],
                        "weights": {"lambda_data": cfg.restoration.lambda_data,
                                    "lambda_topology": cfg.restoration.lambda_topology,
                                    "lambda_geometry": cfg.restoration.lambda_geometry,
                                    "lambda_change": cfg.restoration.lambda_change},
                        **r.restoration.summary(),
                        "output_artifact": r.files.get("restored_mask"),
                        "csv": r.files.get("restoration_candidates_csv")},
        "bayesian_before_restoration": {
            "stage_id": "S3b", "input_ref": "S2+S3 (observed features)",
            "role": "preliminary evidence; Top-K membership selects restoration reference prototypes",
            "features": {k: round(v, 6) for k, v in r.observed.features.items()},
            **r.bayesian_before.summary()},
        "bayesian_after_restoration": {
            "stage_id": "S5", "input_ref": "S4 (restored features, re-extracted)",
            "role": "final Bayesian evidence used for fusion",
            **r.bayesian.summary()},
        "restoration_effect": {
            "top1_before": r.bayesian_before.top1, "top1_after": r.bayesian.top1,
            "top1_changed_by_restoration": r.bayesian_before.top1 != r.bayesian.top1,
            "posterior_of_final_top1_before": round(r.bayesian_before.posteriors.get(r.bayesian.top1, 0.0), 6),
            "posterior_of_final_top1_after": round(r.bayesian.top1_posterior, 6),
            "entropy_before": round(r.bayesian_before.entropy, 6),
            "entropy_after": round(r.bayesian.entropy, 6),
            "note": ("after-restoration posterior is conditional on a restoration chosen with prototype "
                     "references; it is not independent confirmation of the restoration"),
        },
        "bayesian": {"stage_id": "S5", "input_ref": "S4 (restored features)",
                     "alias_of": "bayesian_after_restoration",
                     "prototype_model": {"source": r.model.metadata.get("source"), "classes": r.model.classes,
                                         "fonts": r.model.fonts_used, "n_samples": r.model.n_samples,
                                         **r.model.metadata},
                     "features": {k: round(v, 6) for k, v in r.restored.features.items()},
                     **r.bayesian.summary()},
        "vlm": {"stage_id": "S6", "input_ref": ["input", "S4"], **r.vlm.summary()},
        "fusion": {"stage_id": "S7", "input_ref": ["S5", "S6"], **r.fusion.summary()},
        "decision_detail": {"stage_id": "S8", "input_ref": ["S2", "S3", "S5", "S6", "S7"],
                            **r.decision.summary()},
        "traceability_chain": [
            "input", "S1 preprocessing", "S2 topology", "S3 geometry",
            "S3b bayesian_before_restoration", "S4 restoration",
            "S5 bayesian", "S6 vlm", "S7 fusion", "S8 decision",
        ],
        "downstream_interfaces": {
            "semantic_interpretation": {
                "status": "NOT_IMPLEMENTED",
                "expected_input": ["recognized_mark", "review_candidates", "decision", "raw_image"],
                "note": "will map marks/abbreviations to work instructions (e.g. weld type, side, length)",
            },
            "welding_condition_engine": {
                "status": "NOT_IMPLEMENTED",
                "note": "rule engine will consume semantic output; no welding condition is determined here",
            },
            "robot_controller": {"status": "NOT_IMPLEMENTED"},
        },
        "timing_ms": r.timing_ms,
        "files": r.files,
        "disclaimer": ("Research prototype. Posteriors are relative to a prototype statistical reference "
                       "model; VLM confidence is model-reported; thresholds are demo thresholds."),
    }


def json_default(obj: object) -> object:
    """json.dumps fallback for numpy scalars / arrays and paths."""
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")
