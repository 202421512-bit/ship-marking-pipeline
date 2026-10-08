"""Optional VLM visual verification (independent evidence, not an OCR engine).

The VLM sees only the original crop and the restored crop. It is NOT shown the
Bayesian result so that the two evidence sources stay independent. Its
confidence is a model-reported value, not a calibrated probability.
If no API key is configured, or any error occurs, the status is reported and
the rest of the pipeline continues.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from .config import ENV_FILE, VLMConfig

SYSTEM_PROMPT = (
    "You are a visual verification assistant for a research prototype that reads "
    "single painted or handwritten markings on ship steel blocks. You are not an OCR engine "
    "and must not guess beyond what is visible."
)

USER_PROMPT = (
    "Image 1 is the ORIGINAL crop of one marking (may contain rust, dirt, glare, low light, "
    "rotation, broken strokes). Image 2 is an algorithmically RESTORED binary version (ink = black); "
    "the restoration may itself introduce errors.\n"
    "Tasks:\n"
    "1. Describe the shape that is actually visible (strokes, loops/holes, gaps, damage).\n"
    "2. Give the most likely character. Prefer one of these labels: {labels}. "
    "If none fits, use \"UNKNOWN\".\n"
    "3. If several readings are plausible, list alternatives.\n"
    "4. Do not invent information that is not in the images. If ambiguous, say so and lower confidence.\n"
    "Return ONLY JSON with this schema:\n"
    '{{"top_candidate": "<label>", "confidence": <0..1>, '
    '"alternatives": [{{"label": "<label>", "confidence": <0..1>}}], '
    '"visual_reason": "<short explanation>", "uncertainty": "<low|medium|high>"}}'
)


@dataclass
class VLMResult:
    """Outcome of the VLM verification."""

    status: str                         # OK | UNAVAILABLE | DISABLED | ERROR
    model: str = ""
    top_candidate: Optional[str] = None
    confidence: Optional[float] = None  # model-reported, NOT calibrated
    alternatives: List[Dict[str, object]] = field(default_factory=list)
    visual_reason: str = ""
    uncertainty: str = ""
    message: str = ""
    raw_response: str = ""

    def summary(self) -> Dict[str, object]:
        """JSON-serializable summary."""
        return {
            "status": self.status,
            "role": "independent visual verification (not OCR)",
            "confidence_type": "Model-Reported Confidence (not calibrated)",
            "model": self.model,
            "top_candidate": self.top_candidate,
            "reported_confidence": self.confidence,
            "alternatives": self.alternatives,
            "visual_reason": self.visual_reason,
            "uncertainty": self.uncertainty,
            "message": self.message,
        }


def _encode_png(image: np.ndarray, side: int) -> str:
    """Resize (nearest for masks) and encode as base64 PNG data URL."""
    if image.dtype == bool:
        image = np.where(image, 0, 255).astype(np.uint8)
    h, w = image.shape[:2]
    scale = side / float(max(h, w))
    interp = cv2.INTER_NEAREST if len(np.unique(image)) <= 2 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=interp)
    ok, buf = cv2.imencode(".png", resized)
    if not ok:
        raise ValueError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _clamp(value: object) -> float:
    try:
        return float(min(1.0, max(0.0, float(value))))
    except (TypeError, ValueError):
        return 0.0


def parse_response(text: str) -> Dict[str, object]:
    """Parse and sanitize the JSON returned by the VLM."""
    start, end = text.find("{"), text.rfind("}")
    data = json.loads(text[start:end + 1] if start >= 0 else text)
    alts = []
    for alt in data.get("alternatives", []) or []:
        if isinstance(alt, dict) and alt.get("label") is not None:
            alts.append({"label": str(alt["label"]).strip(), "confidence": _clamp(alt.get("confidence"))})
    return {
        "top_candidate": str(data.get("top_candidate", "UNKNOWN")).strip(),
        "confidence": _clamp(data.get("confidence")),
        "alternatives": alts,
        "visual_reason": str(data.get("visual_reason", "")),
        "uncertainty": str(data.get("uncertainty", "")),
    }


def verify(original: np.ndarray, restored: np.ndarray, labels: Sequence[str],
           cfg: VLMConfig, enabled: bool = True) -> VLMResult:
    """Ask the VLM for an independent reading of the marking.

    Never raises: failures are returned as status UNAVAILABLE / ERROR.
    """
    if not enabled:
        return VLMResult(status="DISABLED", message="VLM disabled by --no-vlm / GUI option.")
    try:
        from dotenv import load_dotenv

        load_dotenv(ENV_FILE, override=False)
    except Exception:
        pass
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    model = os.environ.get("OPENAI_VLM_MODEL", "").strip() or cfg.default_model
    if not api_key:
        return VLMResult(status="UNAVAILABLE", model=model,
                         message="OPENAI_API_KEY not set (.env). Pipeline continues without VLM.")
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=cfg.timeout_sec)
        content = [
            {"type": "text", "text": USER_PROMPT.format(labels=", ".join(labels))},
            {"type": "image_url", "image_url": {"url": _encode_png(original, cfg.image_side)}},
            {"type": "image_url", "image_url": {"url": _encode_png(restored, cfg.image_side)}},
        ]
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": content}],
            response_format={"type": "json_object"},
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
        text = response.choices[0].message.content or ""
        parsed = parse_response(text)
        return VLMResult(status="OK", model=model, raw_response=text, **parsed)
    except Exception as exc:
        return VLMResult(status="ERROR", model=model, message=f"{type(exc).__name__}: {exc}"[:400])
