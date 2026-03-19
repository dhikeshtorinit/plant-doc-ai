"""Logging and observability utilities — JSON-based reasoning trace logger."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from backend.config.settings import settings

logger = logging.getLogger(__name__)


def _ensure_log_dir() -> Path:
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def save_session_log(session_id: str, reasoning_trace: list[dict], final_output: dict) -> str:
    """Persist a full session's reasoning trace and final output to a JSON file."""
    log_dir = _ensure_log_dir()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"{session_id}_{timestamp}.json"
    filepath = log_dir / filename

    log_entry = {
        "session_id": session_id,
        "timestamp": timestamp,
        "reasoning_trace": reasoning_trace,
        "final_output": final_output,
    }

    filepath.write_text(json.dumps(log_entry, indent=2, default=str), encoding="utf-8")
    logger.info("Session log saved: %s", filepath)
    return str(filepath)


def append_trace(trace: list[dict], step: str, data: dict) -> list[dict]:
    """Append a reasoning step to the trace list (immutable-style)."""
    entry = {
        "step": step,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }
    return [*trace, entry]


def configure_logging() -> None:
    """Set up structured logging for the application."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
