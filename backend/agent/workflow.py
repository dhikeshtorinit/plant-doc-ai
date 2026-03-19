"""Agent Workflow — LangGraph-based orchestration of the PlantDoc AI pipeline.

The graph has two phases:

Phase 1 (automatic): image → vision → RAG → [web search if needed] → question generation
  Returns diagnostic questions to the user.

Phase 2 (after user answers): diagnosis → [web search if low confidence → re-diagnosis] → care plan → logging
  Returns the final structured diagnosis.

Web search is triggered conditionally:
  - In Phase 1: when RAG retrieval returns empty or low-relevance results
  - In Phase 2: when initial diagnosis confidence is below threshold
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from backend.agent.logging_utils import append_trace, save_session_log
from backend.agent.models import AgentState
from backend.agent.modules.care_plan import refine_care_plan
from backend.agent.modules.diagnosis import diagnose
from backend.agent.modules.questions import generate_questions
from backend.agent.modules.rag import retrieve
from backend.agent.modules.vision import analyze_image
from backend.agent.tools.web_search import (
    LOW_CONFIDENCE_THRESHOLD,
    MIN_RAG_RELEVANCE,
    should_trigger_web_search,
    web_search,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangGraph state is a TypedDict so the graph can track field-level updates.
# ---------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    session_id: str
    image_path: str
    user_description: str
    vision_result: dict | None
    retrieved_docs: list[dict]
    web_results: list[dict]
    diagnostic_questions: dict | None
    user_answers: list[str]
    diagnosis: dict | None
    reasoning_trace: list[dict]
    error: str | None
    current_step: str


# ---------------------------------------------------------------------------
# Phase 1 nodes
# ---------------------------------------------------------------------------


def vision_node(state: GraphState) -> GraphState:
    """Analyze the uploaded plant image."""
    try:
        result = analyze_image(state["image_path"], state.get("user_description", ""))
        trace = append_trace(
            state.get("reasoning_trace", []),
            "vision_analysis",
            {
                "is_plant": result.is_plant,
                "plant_type_guess": result.plant_type_guess,
                "symptoms": result.symptoms,
            },
        )
        return {
            "vision_result": result.model_dump(),
            "reasoning_trace": trace,
            "current_step": "vision_complete",
        }
    except Exception as e:
        logger.exception("Vision analysis failed")
        return {"error": f"Vision analysis failed: {e}", "current_step": "error"}


def _after_vision(state: GraphState) -> str:
    """Gate after vision: stop early if the image is not a real plant."""
    vision = state.get("vision_result") or {}
    if not vision.get("is_plant", True):
        return "not_a_plant"
    return "retrieval"


def retrieval_node(state: GraphState) -> GraphState:
    """Retrieve relevant plant knowledge based on vision output."""
    try:
        vision = state.get("vision_result") or {}
        symptoms = vision.get("symptoms", [])
        plant_type = vision.get("plant_type_guess", "")
        query = f"{plant_type} {' '.join(symptoms)}"

        docs = retrieve(query)
        relevance_scores = [d.relevance_score for d in docs]
        trace = append_trace(
            state.get("reasoning_trace", []),
            "knowledge_retrieval",
            {
                "query": query,
                "num_docs": len(docs),
                "relevance_scores": relevance_scores,
                "max_relevance": max(relevance_scores, default=0.0),
                "min_relevance": min(relevance_scores, default=0.0) if relevance_scores else 0.0,
                "top_doc_preview": docs[0].content[:100] if docs else "",
            },
        )
        return {
            "retrieved_docs": [d.model_dump() for d in docs],
            "reasoning_trace": trace,
            "current_step": "retrieval_complete",
        }
    except Exception as e:
        logger.exception("Knowledge retrieval failed")
        return {
            "retrieved_docs": [],
            "reasoning_trace": append_trace(
                state.get("reasoning_trace", []),
                "knowledge_retrieval",
                {"error": str(e)},
            ),
            "current_step": "retrieval_complete",
        }


def _web_search_trigger_reason(state: GraphState) -> tuple[str, dict]:
    """Infer why web search was triggered and return (reason, detail dict)."""
    docs = state.get("retrieved_docs", [])
    relevance_scores = [d.get("relevance_score", 0.0) for d in docs]
    max_relevance = max(relevance_scores, default=0.0)
    diag = state.get("diagnosis") or {}
    confidence = diag.get("confidence")

    if not docs:
        return "no_rag_documents", {"num_retrieved_docs": 0}
    if max_relevance < MIN_RAG_RELEVANCE:
        return "low_rag_relevance", {
            "relevance_scores": relevance_scores,
            "max_relevance": max_relevance,
            "threshold": MIN_RAG_RELEVANCE,
        }
    if confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD:
        return "low_diagnosis_confidence", {
            "diagnosis_confidence": confidence,
            "threshold": LOW_CONFIDENCE_THRESHOLD,
        }
    return "unknown", {"relevance_scores": relevance_scores, "max_relevance": max_relevance}


def web_search_node(state: GraphState) -> GraphState:
    """Fetch external plant health information via web search."""
    vision = state.get("vision_result") or {}
    plant_type = vision.get("plant_type_guess", "")
    symptoms = vision.get("symptoms", [])

    trigger_reason, trigger_detail = _web_search_trigger_reason(state)
    search_query = f"{plant_type} {' '.join(symptoms)}".strip()

    try:
        results = web_search(plant_type, symptoms)
        trace = append_trace(
            state.get("reasoning_trace", []),
            "web_search",
            {
                "triggered": True,
                "trigger_reason": trigger_reason,
                **trigger_detail,
                "query": search_query,
                "num_results": len(results),
            },
        )
        return {
            "web_results": [r.model_dump() for r in results],
            "reasoning_trace": trace,
            "current_step": "web_search_complete",
        }
    except Exception as e:
        logger.exception("Web search failed — continuing without web results")
        trace = append_trace(
            state.get("reasoning_trace", []),
            "web_search",
            {
                "triggered": True,
                "trigger_reason": trigger_reason,
                **trigger_detail,
                "query": search_query,
                "error": str(e),
            },
        )
        return {
            "web_results": [],
            "reasoning_trace": trace,
            "current_step": "web_search_complete",
        }


def _should_web_search_phase1(state: GraphState) -> str:
    """Conditional edge: route to web_search or straight to questions."""
    docs = state.get("retrieved_docs", [])
    if should_trigger_web_search(docs):
        return "web_search"
    return "questions"


def question_node(state: GraphState) -> GraphState:
    """Generate diagnostic questions for the user."""
    try:
        vision = state.get("vision_result") or {}
        from backend.agent.models import RetrievedDocument

        docs = [RetrievedDocument(**d) for d in state.get("retrieved_docs", [])]
        questions = generate_questions(
            symptoms=vision.get("symptoms", []),
            plant_type=vision.get("plant_type_guess", ""),
            retrieved_docs=docs,
        )
        trace = append_trace(
            state.get("reasoning_trace", []),
            "question_generation",
            {"questions": questions.questions},
        )
        return {
            "diagnostic_questions": questions.model_dump(),
            "reasoning_trace": trace,
            "current_step": "questions_ready",
        }
    except Exception as e:
        logger.exception("Question generation failed")
        return {
            "diagnostic_questions": {
                "questions": [
                    "How often do you water this plant?",
                    "How much direct sunlight does it get?",
                    "Have you changed anything recently (soil, pot, location)?",
                ],
                "reasoning": "Fallback questions due to error.",
            },
            "current_step": "questions_ready",
        }


# ---------------------------------------------------------------------------
# Phase 2 nodes
# ---------------------------------------------------------------------------


def diagnosis_node(state: GraphState) -> GraphState:
    """Run structured diagnosis using all collected evidence."""
    try:
        vision = state.get("vision_result") or {}
        from backend.agent.models import RetrievedDocument, WebSearchResult

        docs = [RetrievedDocument(**d) for d in state.get("retrieved_docs", [])]
        web = [WebSearchResult(**w) for w in state.get("web_results", [])]
        questions = (state.get("diagnostic_questions") or {}).get("questions", [])

        result = diagnose(
            plant_type=vision.get("plant_type_guess", ""),
            symptoms=vision.get("symptoms", []),
            retrieved_docs=docs,
            user_answers=state.get("user_answers", []),
            questions_asked=questions,
            web_results=web,
        )
        trace = append_trace(
            state.get("reasoning_trace", []),
            "diagnosis",
            {"diagnosis": result.diagnosis, "confidence": result.confidence},
        )
        return {
            "diagnosis": result.model_dump(),
            "reasoning_trace": trace,
            "current_step": "diagnosis_complete",
        }
    except Exception as e:
        logger.exception("Diagnosis failed")
        return {"error": f"Diagnosis failed: {e}", "current_step": "error"}


def _should_web_search_phase2(state: GraphState) -> str:
    """Conditional edge after diagnosis: if confidence is low and no web results yet, search and re-diagnose."""
    diag = state.get("diagnosis") or {}
    confidence = diag.get("confidence", 1.0)
    existing_web = state.get("web_results", [])

    if not existing_web and should_trigger_web_search(
        state.get("retrieved_docs", []),
        diagnosis_confidence=confidence,
    ):
        return "web_search_phase2"
    return "care_plan"


def web_search_phase2_node(state: GraphState) -> GraphState:
    """Web search triggered in Phase 2 due to low diagnosis confidence."""
    logger.info("Low confidence diagnosis (%.2f) — triggering web search for enrichment",
                (state.get("diagnosis") or {}).get("confidence", 0.0))
    return web_search_node(state)


def rediagnosis_node(state: GraphState) -> GraphState:
    """Re-run diagnosis with web search results included."""
    logger.info("Re-running diagnosis with web search results")
    return diagnosis_node(state)


def care_plan_node(state: GraphState) -> GraphState:
    """Refine the care plan, personalizing from user answers when available."""
    try:
        from backend.agent.models import DiagnosisResult

        diag = DiagnosisResult(**(state.get("diagnosis") or {}))
        questions = (state.get("diagnostic_questions") or {}).get("questions", [])
        refined = refine_care_plan(
            diag,
            user_answers=state.get("user_answers"),
            questions_asked=questions,
        )
        trace = append_trace(
            state.get("reasoning_trace", []),
            "care_plan_refinement",
            {"treatment_steps": len(refined.treatment_plan)},
        )
        return {
            "diagnosis": refined.model_dump(),
            "reasoning_trace": trace,
            "current_step": "care_plan_complete",
        }
    except Exception as e:
        logger.exception("Care plan refinement failed")
        return {"current_step": "care_plan_complete"}


def logging_node(state: GraphState) -> GraphState:
    """Persist the reasoning trace and final output."""
    try:
        save_session_log(
            session_id=state.get("session_id", "unknown"),
            reasoning_trace=state.get("reasoning_trace", []),
            final_output=state.get("diagnosis") or {},
        )
    except Exception:
        logger.exception("Failed to save session log")
    return {"current_step": "complete"}


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def build_phase1_graph() -> StateGraph:
    """Phase 1: vision → [gate: is it a plant?] → retrieval → [web search if needed] → questions.

    After vision, a conditional edge checks whether the image contains a real
    plant. If not, the graph ends immediately. After retrieval, another
    conditional edge checks whether RAG results are sufficient.
    """
    graph = StateGraph(GraphState)

    graph.add_node("vision", vision_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("questions", question_node)

    graph.set_entry_point("vision")
    graph.add_conditional_edges("vision", _after_vision, {
        "retrieval": "retrieval",
        "not_a_plant": END,
    })
    graph.add_conditional_edges("retrieval", _should_web_search_phase1, {
        "web_search": "web_search",
        "questions": "questions",
    })
    graph.add_edge("web_search", "questions")
    graph.add_edge("questions", END)

    return graph


def build_phase2_graph() -> StateGraph:
    """Phase 2: diagnosis → [web search + re-diagnosis if low confidence] → care plan → logging.

    After initial diagnosis, if confidence is below threshold and no web results
    exist yet, the graph routes through web search and re-diagnosis.
    """
    graph = StateGraph(GraphState)

    graph.add_node("diagnosis", diagnosis_node)
    graph.add_node("web_search_phase2", web_search_phase2_node)
    graph.add_node("rediagnosis", rediagnosis_node)
    graph.add_node("care_plan", care_plan_node)
    graph.add_node("logging", logging_node)

    graph.set_entry_point("diagnosis")
    graph.add_conditional_edges("diagnosis", _should_web_search_phase2, {
        "web_search_phase2": "web_search_phase2",
        "care_plan": "care_plan",
    })
    graph.add_edge("web_search_phase2", "rediagnosis")
    graph.add_edge("rediagnosis", "care_plan")
    graph.add_edge("care_plan", "logging")
    graph.add_edge("logging", END)

    return graph


# Compile once at module level for reuse
phase1_app = build_phase1_graph().compile()
phase2_app = build_phase2_graph().compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_phase1(image_path: str, user_description: str = "") -> dict:
    """Run Phase 1: analyze image, retrieve knowledge, generate questions.

    Returns the full graph state dict (including diagnostic_questions).
    """
    session_id = str(uuid.uuid4())[:8]
    initial_state: GraphState = {
        "session_id": session_id,
        "image_path": image_path,
        "user_description": user_description,
        "vision_result": None,
        "retrieved_docs": [],
        "web_results": [],
        "diagnostic_questions": None,
        "user_answers": [],
        "diagnosis": None,
        "reasoning_trace": [],
        "error": None,
        "current_step": "init",
    }

    result = phase1_app.invoke(initial_state)
    logger.info("Phase 1 complete for session %s", session_id)
    return dict(result)


def run_phase2(phase1_state: dict, user_answers: list[str]) -> dict:
    """Run Phase 2: diagnose and generate care plan.

    Takes the state from Phase 1 and the user's answers.
    Returns the full graph state dict (including diagnosis).
    """
    phase1_state["user_answers"] = user_answers
    result = phase2_app.invoke(phase1_state)
    logger.info("Phase 2 complete for session %s", phase1_state.get("session_id"))
    return dict(result)


def run_phase1_streaming(image_path: str, user_description: str = ""):
    """Run Phase 1 step-by-step, yielding progress events for streaming UI.

    Yields dicts with "type": "progress" | "complete" | "error".
    Progress events include "step" and "message".
    """
    session_id = str(uuid.uuid4())[:8]
    state: GraphState = {
        "session_id": session_id,
        "image_path": image_path,
        "user_description": user_description,
        "vision_result": None,
        "retrieved_docs": [],
        "web_results": [],
        "diagnostic_questions": None,
        "user_answers": [],
        "diagnosis": None,
        "reasoning_trace": [],
        "error": None,
        "current_step": "init",
    }

    try:
        # Step 1: Vision
        yield {"type": "progress", "step": "vision", "message": "Analyzing image..."}
        update = vision_node(state)
        state = {**state, **update}

        vision = state.get("vision_result") or {}
        plant = vision.get("plant_type_guess", "")
        symptoms = vision.get("symptoms", [])
        msg = f"{plant} — {', '.join(symptoms)}" if (plant or symptoms) else "Image analyzed"
        yield {"type": "progress", "step": "vision", "message": msg}

        if not vision.get("is_plant", True):
            yield {
                "type": "complete",
                "is_plant": False,
                "description": vision.get("raw_description", "Not a real plant."),
                "reasoning_trace": list(state.get("reasoning_trace", [])),
            }
            return

        # Step 2: Retrieval
        yield {"type": "progress", "step": "retrieval", "message": "Searching knowledge base..."}
        update = retrieval_node(state)
        state = {**state, **update}
        num_docs = len(state.get("retrieved_docs", []))
        yield {"type": "progress", "step": "retrieval", "message": f"Found {num_docs} relevant documents"}

        # Step 3: Web search (conditional)
        if should_trigger_web_search(state.get("retrieved_docs", [])):
            yield {"type": "progress", "step": "web_search", "message": "Searching the web..."}
            update = web_search_node(state)
            state = {**state, **update}
            num_web = len(state.get("web_results", []))
            yield {"type": "progress", "step": "web_search", "message": f"Found {num_web} web results"}

        # Step 4: Questions
        yield {"type": "progress", "step": "questions", "message": "Generating diagnostic questions..."}
        update = question_node(state)
        state = {**state, **update}

        questions = (state.get("diagnostic_questions") or {}).get("questions", [])
        yield {"type": "progress", "step": "questions", "message": f"Generated {len(questions)} questions"}

        # Final result (include full state for session storage)
        yield {
            "type": "complete",
            "is_plant": True,
            "session_id": session_id,
            "plant_type_guess": vision.get("plant_type_guess", ""),
            "symptoms_detected": vision.get("symptoms", []),
            "confidence": vision.get("confidence", 0.0),
            "description": vision.get("raw_description", ""),
            "diagnostic_questions": questions,
            "reasoning_trace": list(state.get("reasoning_trace", [])),
            "state": dict(state),
        }
    except Exception as e:
        logger.exception("Phase 1 streaming failed")
        yield {"type": "error", "message": str(e)}


def run_phase2_streaming(phase1_state: dict, user_answers: list[str]):
    """Run Phase 2 step-by-step, yielding progress events for streaming UI.

    Yields dicts with "type": "progress" | "complete" | "error".
    """
    state: GraphState = {**phase1_state, "user_answers": user_answers}

    try:
        # Step 1: Diagnosis
        yield {"type": "progress", "step": "diagnosis", "message": "Analyzing symptoms and your answers..."}
        update = diagnosis_node(state)
        state = {**state, **update}

        diag = state.get("diagnosis") or {}
        diagnosis_text = diag.get("diagnosis", "")[:80]
        if diagnosis_text:
            yield {"type": "progress", "step": "diagnosis", "message": f"{diagnosis_text}..."}
        else:
            yield {"type": "progress", "step": "diagnosis", "message": "Initial diagnosis complete"}

        # Step 2: Web search (conditional)
        if _should_web_search_phase2(state) == "web_search_phase2":
            yield {"type": "progress", "step": "web_search", "message": "Searching the web for more information..."}
            update = web_search_phase2_node(state)
            state = {**state, **update}
            num_web = len(state.get("web_results", []))
            yield {"type": "progress", "step": "web_search", "message": f"Found {num_web} web results"}

            yield {"type": "progress", "step": "rediagnosis", "message": "Re-analyzing with additional context..."}
            update = rediagnosis_node(state)
            state = {**state, **update}
            yield {"type": "progress", "step": "rediagnosis", "message": "Diagnosis updated"}

        # Step 3: Care plan
        yield {"type": "progress", "step": "care_plan", "message": "Generating treatment plan..."}
        update = care_plan_node(state)
        state = {**state, **update}
        diag = state.get("diagnosis") or {}
        yield {"type": "progress", "step": "care_plan", "message": f"{len(diag.get('treatment_plan', []))} treatment steps"}

        # Step 4: Logging (silent)
        logging_node(state)

        if state.get("error"):
            yield {"type": "error", "message": state["error"]}
            return

        yield {
            "type": "complete",
            "diagnosis": (state.get("diagnosis") or {}).copy(),
            "reasoning_trace": list(state.get("reasoning_trace", [])),
        }
    except Exception as e:
        logger.exception("Phase 2 streaming failed")
        yield {"type": "error", "message": str(e)}
