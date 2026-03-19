"""Diagnosis Reasoning Module — structured reasoning to determine plant health issues."""

from __future__ import annotations

import json
import logging

from openai import OpenAI

from backend.agent.models import DiagnosisResult, RetrievedDocument, WebSearchResult
from backend.config.settings import settings

logger = logging.getLogger(__name__)

DIAGNOSIS_SYSTEM_PROMPT = """You are an expert plant pathologist performing structured diagnosis.

Follow this reasoning process:
1. List all observed symptoms.
2. Cross-reference with the provided plant knowledge and web search results (if available).
3. Generate possible causes (hypotheses).
4. Use the user's answers to rule in or rule out each hypothesis. Their answers are critical — they directly distinguish between causes.
5. Rank hypotheses by likelihood based on both symptoms AND user answers.
6. Select the most likely diagnosis. If user answers contradict a hypothesis, do not select it.
7. Generate a treatment plan tailored to what the user reported (e.g. if they use tap water, mention switching to filtered; if no drainage, emphasize repotting).

The user was asked specific questions for a reason. Their answers must materially affect your diagnosis and treatment plan. Do not ignore them.

Use both retrieved plant knowledge and web search results when available.
Prefer reliable and consistent information — if sources conflict, favor the
explanation that best matches the observed symptoms, the user's answers to your
diagnostic questions, and any other user-reported conditions.

Return a JSON object with exactly these fields:

{
  "plant_type_guess": "best guess of plant species",
  "symptoms_detected": ["symptom1", "symptom2"],
  "possible_causes": ["cause1", "cause2", "cause3"],
  "diagnosis": "most likely diagnosis with brief explanation",
  "confidence": 0.0 to 1.0,
  "treatment_plan": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ..."
  ],
  "recovery_timeline": "Week 1: ... Week 2: ... Week 3: ...",
  "warning_signs": ["sign1", "sign2"]
}

Be specific and actionable in treatment steps.
Return ONLY valid JSON, no markdown fences."""


def _build_knowledge_section(
    retrieved_docs: list[RetrievedDocument],
    web_results: list[WebSearchResult],
) -> str:
    """Combine RAG documents and web results into a single knowledge section."""
    sections: list[str] = []

    if retrieved_docs:
        rag_text = "\n---\n".join(doc.content for doc in retrieved_docs[:5])
        sections.append(f"Plant Knowledge Base:\n{rag_text}")

    if web_results:
        web_entries = []
        for wr in web_results[:5]:
            entry = f"- {wr.title}: {wr.summary}"
            if wr.source and wr.source not in ("fallback_simulated", "tavily_answer"):
                entry += f" (source: {wr.source})"
            web_entries.append(entry)
        sections.append("Web Search Results:\n" + "\n".join(web_entries))

    return "\n\n".join(sections) if sections else "No relevant documents found."


def diagnose(
    plant_type: str,
    symptoms: list[str],
    retrieved_docs: list[RetrievedDocument],
    user_answers: list[str],
    questions_asked: list[str],
    web_results: list[WebSearchResult] | None = None,
) -> DiagnosisResult:
    """Perform structured diagnosis using symptoms, knowledge, web results, and user answers."""
    client = OpenAI(api_key=settings.openai_api_key)

    knowledge_section = _build_knowledge_section(retrieved_docs, web_results or [])

    qa_section = ""
    if questions_asked and user_answers:
        qa_pairs = [
            f"Q: {q}\nA: {a}"
            for q, a in zip(questions_asked, user_answers, strict=False)
        ]
        qa_section = """
USER-PROVIDED CONTEXT (use these answers to distinguish between causes and tailor the treatment plan):
""" + "\n\n".join(qa_pairs) + "\n"

    user_prompt = f"""Plant type (best guess): {plant_type}
Detected symptoms: {', '.join(symptoms) if symptoms else 'None clearly detected'}
{qa_section}
{knowledge_section}

Perform a structured diagnosis. Use the user's answers to rule in/out hypotheses and to tailor the treatment plan.
Provide specific, actionable treatment steps and a realistic recovery timeline."""

    logger.info("Running diagnosis for plant=%s, symptoms=%s, web_results=%d",
                plant_type, symptoms, len(web_results or []))

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": DIAGNOSIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=settings.diagnosis_max_tokens,
        temperature=0.2,
    )

    raw = response.choices[0].message.content or "{}"
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Diagnosis returned non-JSON: %s", raw[:300])
        data = {
            "plant_type_guess": plant_type,
            "symptoms_detected": symptoms,
            "possible_causes": [],
            "diagnosis": "Unable to parse diagnosis. Please try again.",
            "confidence": 0.0,
            "treatment_plan": [],
            "recovery_timeline": "",
            "warning_signs": [],
        }

    return DiagnosisResult(**data)
