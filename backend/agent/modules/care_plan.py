"""Care Plan Generator — enriches diagnosis with detailed treatment and timeline.

In most cases the diagnosis module already produces a treatment plan. This module
exists as a dedicated refinement step that can be called when a richer, standalone
care plan is needed (e.g. the user explicitly requests more detail).
"""

from __future__ import annotations

import json
import logging

from openai import OpenAI

from backend.agent.models import DiagnosisResult
from backend.config.settings import settings

logger = logging.getLogger(__name__)

CARE_PLAN_PROMPT = """You are a plant care advisor.
Given the diagnosis and what the user reported, produce a detailed JSON care plan.

Tailor the treatment plan to the user's situation. For example:
- If they said they use tap water → emphasize switching to filtered/distilled
- If they said no drainage holes → include repotting with drainage as a priority step
- If they mist the plant → tell them to stop misting and water at soil level only
- If they water very frequently → give a specific reduced schedule

{
  "treatment_plan": [
    "Step 1: ...",
    "Step 2: ..."
  ],
  "recovery_timeline": "Week 1: ... Week 2: ... (or Week 1-2: ... Week 3-4: ...)",
  "warning_signs": ["sign that things are getting worse"]
}

Format recovery_timeline as one string with multiple segments, each starting with
\"Week \" plus a number or range (e.g. Week 1:, Week 2:, Week 1-2:), then the
expected progress for that period. Separate segments with a space before the next Week.

Be specific: include quantities, frequencies, product names if applicable.
Return ONLY valid JSON, no markdown fences."""


def refine_care_plan(
    diagnosis: DiagnosisResult,
    user_answers: list[str] | None = None,
    questions_asked: list[str] | None = None,
) -> DiagnosisResult:
    """Call the LLM to expand the treatment plan, optionally personalizing from user answers."""
    if len(diagnosis.treatment_plan) >= 3 and len(diagnosis.warning_signs) >= 2 and not (user_answers and questions_asked):
        logger.info("Care plan already detailed and no user context — skipping refinement.")
        return diagnosis

    client = OpenAI(api_key=settings.openai_api_key)

    user_context = ""
    if questions_asked and user_answers:
        qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in zip(questions_asked, user_answers, strict=False))
        user_context = f"\n\nWhat the user reported (tailor the plan to this):\n{qa}\n"

    user_prompt = f"""Diagnosis: {diagnosis.diagnosis}
Plant: {diagnosis.plant_type_guess}
Symptoms: {', '.join(diagnosis.symptoms_detected)}
Existing treatment steps: {json.dumps(diagnosis.treatment_plan)}{user_context}

Provide a detailed, personalized care plan."""

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": CARE_PLAN_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1024,
        temperature=0.3,
    )

    raw = response.choices[0].message.content or "{}"
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        data = json.loads(raw)
        diagnosis.treatment_plan = data.get("treatment_plan", diagnosis.treatment_plan)
        diagnosis.recovery_timeline = data.get("recovery_timeline", diagnosis.recovery_timeline)
        diagnosis.warning_signs = data.get("warning_signs", diagnosis.warning_signs)
    except json.JSONDecodeError:
        logger.warning("Care plan refinement returned non-JSON — keeping original.")

    return diagnosis
