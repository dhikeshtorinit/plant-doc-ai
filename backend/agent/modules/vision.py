"""Vision Analysis Module — sends plant image to a multimodal LLM and extracts symptoms."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

from openai import OpenAI

from backend.agent.models import VisionResult
from backend.config.settings import settings

logger = logging.getLogger(__name__)

VISION_SYSTEM_PROMPT = """You are an expert botanist analyzing a plant photo.
Examine the image carefully and return a JSON object with exactly these fields:

{
  "is_plant": true or false,
  "plant_type_guess": "best guess of the plant species or common name",
  "symptoms": ["list", "of", "visible", "symptoms"],
  "confidence": 0.0 to 1.0,
  "raw_description": "detailed plain-text description of what you observe"
}

IMPORTANT:
- Set "is_plant" to false if the image does not contain a real, living plant
  (e.g. toys, drawings, non-plant objects, artificial plants).
- If "is_plant" is false, set confidence to 0.0 and symptoms to an empty list.
- If "is_plant" is false, write the raw_description as a short, friendly,
  lightly humorous note acknowledging what the image actually is and gently
  nudging the user to upload a real plant photo. Keep it warm.

If the image IS a real plant, focus on:
- Leaf color, texture, spots, wilting, curling
- Stem condition
- Soil visibility and moisture
- Pests or fungal growths
- Overall plant vigor

Return ONLY valid JSON, no markdown fences."""


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _detect_mime(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/jpeg")


def analyze_image(image_path: str, user_description: str = "") -> VisionResult:
    """Analyze a plant image using a multimodal model and return structured symptoms."""
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    client = OpenAI(api_key=settings.openai_api_key)
    b64 = _encode_image(image_path)
    mime = _detect_mime(image_path)

    user_content: list[dict] = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
        },
        {
            "type": "text",
            "text": (
                "Analyze this plant image and identify any health issues."
                + (f"\n\nUser note: {user_description}" if user_description else "")
            ),
        },
    ]

    logger.info("Sending image to vision model: %s", settings.openai_model)

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=settings.vision_max_tokens,
        temperature=0.2,
    )

    raw = response.choices[0].message.content or "{}"
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Vision model returned non-JSON: %s", raw[:200])
        data = {
            "plant_type_guess": "Unknown",
            "symptoms": [],
            "confidence": 0.0,
            "raw_description": raw,
        }

    return VisionResult(**data)
