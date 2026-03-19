"""Web Search Tool — retrieves external plant health information as a fallback/enrichment for RAG.

Uses the Tavily search API when available, falls back to a simulated search otherwise.
This tool is triggered when:
  - RAG retrieval returns empty or low-relevance results
  - Initial diagnosis confidence is below threshold
"""

from __future__ import annotations

import logging
from typing import Any

from backend.agent.models import WebSearchResult
from backend.config.settings import settings

logger = logging.getLogger(__name__)

MIN_RAG_RELEVANCE = 0.3
LOW_CONFIDENCE_THRESHOLD = 0.6


def build_search_query(plant_type: str, symptoms: list[str]) -> str:
    """Construct a targeted search query from plant type and symptoms."""
    parts = []
    if plant_type:
        parts.append(plant_type)
    parts.extend(symptoms[:4])
    parts.append("cause treatment")
    return " ".join(parts)


def _search_tavily(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Call Tavily search API and return structured results."""
    from tavily import TavilyClient

    client = TavilyClient(api_key=settings.tavily_api_key)
    response = client.search(
        query=query,
        search_depth="basic",
        max_results=max_results,
        include_answer=True,
        topic="general",
    )

    results: list[dict[str, str]] = []

    if response.get("answer"):
        results.append({
            "title": "Tavily AI Summary",
            "summary": response["answer"],
            "source": "tavily_answer",
        })

    for item in response.get("results", []):
        results.append({
            "title": item.get("title", ""),
            "summary": item.get("content", "")[:500],
            "source": item.get("url", ""),
        })

    return results


def web_search(plant_type: str, symptoms: list[str]) -> list[WebSearchResult]:
    """Run a web search for plant health information via Tavily.

    Returns an empty list if no API key is configured or if the search fails.
    """
    query = build_search_query(plant_type, symptoms)
    logger.info("Web search triggered — query: '%s'", query)

    if not settings.tavily_api_key:
        logger.warning("No TAVILY_API_KEY set — skipping web search")
        return []

    try:
        raw_results = _search_tavily(query)
        logger.info("Tavily returned %d results", len(raw_results))
        return [WebSearchResult(**r) for r in raw_results]
    except Exception:
        logger.exception("Tavily search failed — continuing without web results")
        return []


def should_trigger_web_search(
    retrieved_docs: list[dict],
    diagnosis_confidence: float | None = None,
) -> bool:
    """Decide whether web search should be triggered.

    Triggers when:
      - RAG returned no documents
      - All RAG documents have low relevance scores
      - Diagnosis confidence is below threshold (re-diagnosis flow)
    """
    if not retrieved_docs:
        logger.info("Web search trigger: no RAG documents retrieved")
        return True

    max_relevance = max(
        (doc.get("relevance_score", 0.0) for doc in retrieved_docs),
        default=0.0,
    )
    if max_relevance < MIN_RAG_RELEVANCE:
        logger.info(
            "Web search trigger: max RAG relevance %.3f < threshold %.3f",
            max_relevance, MIN_RAG_RELEVANCE,
        )
        return True

    if diagnosis_confidence is not None and diagnosis_confidence < LOW_CONFIDENCE_THRESHOLD:
        logger.info(
            "Web search trigger: diagnosis confidence %.2f < threshold %.2f",
            diagnosis_confidence, LOW_CONFIDENCE_THRESHOLD,
        )
        return True

    return False
