"""Pydantic models for the PlantDoc AI agent pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VisionResult(BaseModel):
    is_plant: bool = True
    plant_type_guess: str = ""
    symptoms: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    raw_description: str = ""


class RetrievedDocument(BaseModel):
    content: str
    metadata: dict = Field(default_factory=dict)
    relevance_score: float = 0.0


class DiagnosticQuestions(BaseModel):
    questions: list[str] = Field(default_factory=list)
    reasoning: str = ""


class WebSearchResult(BaseModel):
    title: str = ""
    summary: str = ""
    source: str = ""


class DiagnosisResult(BaseModel):
    plant_type_guess: str = ""
    symptoms_detected: list[str] = Field(default_factory=list)
    possible_causes: list[str] = Field(default_factory=list)
    diagnosis: str = ""
    confidence: float = 0.0
    treatment_plan: list[str] = Field(default_factory=list)
    recovery_timeline: str = ""
    warning_signs: list[str] = Field(default_factory=list)


class AgentState(BaseModel):
    """Full state carried through the agent workflow."""

    session_id: str = ""
    image_path: str = ""
    user_description: str = ""

    vision_result: VisionResult | None = None
    retrieved_docs: list[RetrievedDocument] = Field(default_factory=list)
    web_results: list[WebSearchResult] = Field(default_factory=list)
    diagnostic_questions: DiagnosticQuestions | None = None
    user_answers: list[str] = Field(default_factory=list)
    diagnosis: DiagnosisResult | None = None

    reasoning_trace: list[dict] = Field(default_factory=list)
    error: str | None = None
    current_step: str = "init"
