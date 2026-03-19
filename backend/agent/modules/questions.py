"""Question Generation Module — produces targeted diagnostic questions to reduce uncertainty."""

from __future__ import annotations

import json
import logging

from openai import OpenAI

from backend.agent.models import DiagnosticQuestions, RetrievedDocument
from backend.config.settings import settings

logger = logging.getLogger(__name__)

QUESTION_SYSTEM_PROMPT = """You are an expert plant diagnostician.
Given the detected symptoms and relevant plant knowledge, generate 2-3 targeted
follow-up questions to ask the plant owner.

CRITICAL: Derive your questions from the retrieved knowledge. Do not use a fixed
checklist. Each question must be traceable to a specific cause or distinguishing
factor mentioned in the knowledge.

If the knowledge suggests multiple possible causes (e.g. overwatering vs bacterial
infection), ask questions that help distinguish between them — target the factors
that differentiate one cause from another.

Return a JSON object with exactly these fields:

{
  "questions": ["question 1", "question 2", "question 3"],
  "reasoning": "brief explanation of why you chose these questions, citing the knowledge"
}

Return ONLY valid JSON, no markdown fences."""


def generate_questions(
    symptoms: list[str],
    plant_type: str,
    retrieved_docs: list[RetrievedDocument],
) -> DiagnosticQuestions:
    """Generate diagnostic questions based on symptoms and retrieved knowledge."""
    client = OpenAI(api_key=settings.openai_api_key)

    knowledge_context = "\n---\n".join(doc.content for doc in retrieved_docs[:5])

    user_prompt = f"""Plant type (best guess): {plant_type}
Detected symptoms: {', '.join(symptoms) if symptoms else 'None clearly detected'}

Relevant plant knowledge:
{knowledge_context or 'No relevant documents found.'}

Generate 2-3 diagnostic questions to ask the plant owner."""

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": QUESTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=512,
        temperature=0.3,
    )

    raw = response.choices[0].message.content or "{}"
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Question generation returned non-JSON: %s", raw[:200])
        data = {
            "questions": [
                "How often do you water this plant?",
                "How much sunlight does it receive daily?",
                "Have you noticed any pests or unusual spots?",
            ],
            "reasoning": "Fallback questions due to parsing error.",
        }

    return DiagnosticQuestions(**data)
