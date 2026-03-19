"""FastAPI server — exposes the PlantDoc AI agent as a REST API."""

from __future__ import annotations

import json
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.agent.logging_utils import configure_logging
from backend.agent.modules.rag import load_knowledge_base
from backend.agent.workflow import run_phase1, run_phase1_streaming, run_phase2, run_phase2_streaming
from backend.config.settings import settings

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PlantDoc AI",
    description="AI-powered plant health diagnosis agent",
    version="0.1.0",
)

_cors_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials="*" not in _cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (swap for Redis/DB in production)
_sessions: dict[str, dict] = {}

UPLOAD_DIR = Path(tempfile.gettempdir()) / "plantdoc_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
MAX_UPLOAD_BYTES = int(settings.max_upload_mb * 1024 * 1024)


def _save_upload(file: UploadFile, save_path: Path) -> None:
    """Save uploaded file with size limit."""
    size = 0
    with open(save_path, "wb") as f:
        for chunk in iter(lambda: file.file.read(65536), b""):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                save_path.unlink(missing_ok=True)
                raise HTTPException(413, f"Image too large. Max size: {settings.max_upload_mb} MB")
            f.write(chunk)


def _cleanup_upload(save_path: Path) -> None:
    """Remove uploaded file after processing."""
    try:
        if save_path.exists():
            save_path.unlink()
    except OSError:
        logger.warning("Could not remove upload %s", save_path)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup() -> None:
    if not (settings.openai_api_key or "").strip():
        logger.warning("OPENAI_API_KEY is not set — vision and diagnosis will fail at runtime.")
    logger.info("Loading plant knowledge base...")
    try:
        count = load_knowledge_base()
        logger.info("Knowledge base ready (%d chunks).", count)
    except Exception:
        logger.exception("Failed to load knowledge base — RAG may not work.")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class Phase2Request(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=64)
    answers: list[str] = Field(..., min_length=1, max_length=20)


class HealthResponse(BaseModel):
    status: str = "ok"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@app.post("/analyze")
async def analyze_plant(
    image: UploadFile = File(...),
    description: str = Form(""),
) -> dict[str, Any]:
    """Phase 1: upload image → vision analysis → RAG → diagnostic questions."""

    suffix = Path(image.filename or "upload.jpg").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported image format: {suffix}")

    file_id = uuid.uuid4().hex[:12]
    save_path = UPLOAD_DIR / f"{file_id}{suffix}"

    try:
        _save_upload(image, save_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Failed to save uploaded image: {exc}")

    try:
        state = run_phase1(str(save_path), description)
    except Exception as exc:
        _cleanup_upload(save_path)
        logger.exception("Phase 1 failed")
        raise HTTPException(500, f"Analysis failed: {exc}")

    if state.get("error"):
        _cleanup_upload(save_path)
        raise HTTPException(500, state["error"])

    vision = state.get("vision_result") or {}

    if not vision.get("is_plant", True):
        _cleanup_upload(save_path)
        return {
            "is_plant": False,
            "description": vision.get("raw_description", "The uploaded image does not appear to contain a real plant."),
        }

    session_id = state.get("session_id", file_id)
    _sessions[session_id] = state

    questions = (state.get("diagnostic_questions") or {}).get("questions", [])
    _cleanup_upload(save_path)

    return {
        "is_plant": True,
        "session_id": session_id,
        "plant_type_guess": vision.get("plant_type_guess", ""),
        "symptoms_detected": vision.get("symptoms", []),
        "confidence": vision.get("confidence", 0.0),
        "description": vision.get("raw_description", ""),
        "diagnostic_questions": questions,
    }


@app.post("/analyze/stream")
async def analyze_plant_stream(
    image: UploadFile = File(...),
    description: str = Form(""),
):
    """Phase 1 with streaming progress: yields NDJSON events (progress + complete)."""

    suffix = Path(image.filename or "upload.jpg").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported image format: {suffix}")

    file_id = uuid.uuid4().hex[:12]
    save_path = UPLOAD_DIR / f"{file_id}{suffix}"

    try:
        _save_upload(image, save_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Failed to save uploaded image: {exc}")

    def _serialize(event: dict) -> str:
        return json.dumps(event, default=str) + "\n"

    def generate() -> Iterator[str]:
        try:
            for event in run_phase1_streaming(str(save_path), description):
                if event.get("type") == "error":
                    yield _serialize({"type": "error", "message": event.get("message", "Unknown error")})
                    return
                if event.get("type") == "complete" and event.get("is_plant"):
                    state = event.pop("state", None)
                    if state:
                        _sessions[event["session_id"]] = state
                yield _serialize(event)
        except Exception as exc:
            logger.exception("Streaming analyze failed")
            yield _serialize({"type": "error", "message": str(exc)})
        finally:
            _cleanup_upload(save_path)

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/diagnose")
async def diagnose_plant(request: Phase2Request) -> dict[str, Any]:
    """Phase 2: receive user answers → diagnosis → care plan."""

    state = _sessions.get(request.session_id)
    if state is None:
        raise HTTPException(404, "Session not found. Please re-upload the image.")

    if not request.answers:
        raise HTTPException(400, "Please provide answers to the diagnostic questions.")

    try:
        result = run_phase2(state, request.answers)
    except Exception as exc:
        logger.exception("Phase 2 failed")
        raise HTTPException(500, f"Diagnosis failed: {exc}")

    if result.get("error"):
        raise HTTPException(500, result["error"])

    # Clean up session after diagnosis
    _sessions.pop(request.session_id, None)

    return result.get("diagnosis") or {}


@app.post("/diagnose/stream")
async def diagnose_plant_stream(request: Phase2Request):
    """Phase 2 with streaming progress: yields NDJSON events (progress + complete)."""

    state = _sessions.get(request.session_id)
    if state is None:
        raise HTTPException(404, "Session not found. Please re-upload the image.")

    if not request.answers:
        raise HTTPException(400, "Please provide answers to the diagnostic questions.")

    session_id = request.session_id

    def _serialize(event: dict) -> str:
        return json.dumps(event, default=str) + "\n"

    def generate():
        try:
            for event in run_phase2_streaming(state, request.answers):
                if event.get("type") == "error":
                    yield _serialize({"type": "error", "message": event.get("message", "Unknown error")})
                    return
                if event.get("type") == "complete":
                    _sessions.pop(session_id, None)
                yield _serialize(event)
        except Exception as exc:
            logger.exception("Streaming diagnose failed")
            yield _serialize({"type": "error", "message": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    """Retrieve the current state of a session (for debugging)."""
    state = _sessions.get(session_id)
    if state is None:
        raise HTTPException(404, "Session not found.")
    return {
        "session_id": session_id,
        "current_step": state.get("current_step"),
        "has_vision_result": state.get("vision_result") is not None,
        "num_retrieved_docs": len(state.get("retrieved_docs", [])),
        "questions": (state.get("diagnostic_questions") or {}).get("questions", []),
    }
