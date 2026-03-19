"""Streamlit frontend for PlantDoc AI."""

from __future__ import annotations

import json
import os
import re
import requests
import streamlit as st


def _split_recovery_timeline(text: str) -> list[str]:
    """Split timeline into week chunks (Week 1..., Week 1-2..., etc.) or by newlines."""
    if not text or not text.strip():
        return []
    t = text.strip()
    parts = re.split(r"\s+(?=Week\s*\d)", t, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        return parts
    if "\n" in t:
        return [ln.strip() for ln in t.split("\n") if ln.strip()]
    return [t]

# Backend URL: set PLANTDOC_API_BASE in env for deployment (e.g. https://api.yourdomain.com)
API_BASE = os.environ.get("PLANTDOC_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="PlantDoc AI", page_icon="🌿", layout="centered")

# Green buttons app-wide (overrides default primary / theme red or blue accents).
_BUTTON_GREEN = "#2E7D32"
_BUTTON_GREEN_HOVER = "#1B5E20"
_BUTTON_GREEN_BORDER = "#1B5E20"
st.markdown(
    f"""
    <style>
        /*
         * Green ONLY for explicit primary buttons.
         * Avoid :not(secondary) — many Streamlit builds omit baseButton-secondary,
         * so secondary "Start Over" was still forced green.
         */
        div[data-testid="stButton"] > button[data-testid="baseButton-primary"],
        div[data-testid="stButton"] > button[kind="primary"],
        div[data-testid="stFormSubmitButton"] > button {{
            background-color: {_BUTTON_GREEN} !important;
            color: #ffffff !important;
            border: 1px solid {_BUTTON_GREEN_BORDER} !important;
        }}
        div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:hover,
        div[data-testid="stButton"] > button[kind="primary"]:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {{
            background-color: {_BUTTON_GREEN_HOVER} !important;
            color: #ffffff !important;
            border-color: #145214 !important;
        }}
        div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:focus,
        div[data-testid="stButton"] > button[kind="primary"]:focus,
        div[data-testid="stFormSubmitButton"] > button:focus {{
            box-shadow: 0 0 0 0.2rem rgba(46, 125, 50, 0.45) !important;
        }}
        div[data-testid="stDownloadButton"] > button {{
            background-color: {_BUTTON_GREEN} !important;
            color: #ffffff !important;
            border: 1px solid {_BUTTON_GREEN_BORDER} !important;
        }}
        div[data-testid="stDownloadButton"] > button:hover {{
            background-color: {_BUTTON_GREEN_HOVER} !important;
            color: #ffffff !important;
        }}
        /* Start Over (key questions_start_over): neutral + one line */
        div[class*="st-key-questions_start_over"] button {{
            background-color: #ffffff !important;
            color: #1f2937 !important;
            border: 1px solid #9ca3af !important;
            white-space: nowrap !important;
        }}
        div[class*="st-key-questions_start_over"] button:hover {{
            background-color: #f9fafb !important;
            border-color: #6b7280 !important;
        }}
        div[class*="st-key-questions_start_over"] button:focus {{
            box-shadow: 0 0 0 0.2rem rgba(107, 114, 128, 0.35) !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Demo-level password protection (Streamlit secrets)
# ---------------------------------------------------------------------------

def check_password() -> bool:
    """Require a password (stored in `st.secrets["APP_PASSWORD"]`) for demo access."""
    # Track authentication across reruns.
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # Already authenticated: do not render login UI.
    if st.session_state.authenticated:
        return True

    # Not authenticated yet: render login UI.
    st.title("PlantDoc AI Login")

    if "APP_PASSWORD" not in st.secrets:
        st.error("APP_PASSWORD is not configured in Streamlit secrets.")
        return False

    expected_password = st.secrets["APP_PASSWORD"]

    # st.form: pressing Enter in the password field submits the form (same as Continue).
    with st.form("plantdoc_login", clear_on_submit=False):
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Continue", type="primary", use_container_width=True)

    if submitted:
        entered = (password or "").strip()
        if not entered:
            st.warning("Enter a password, then press Continue or Enter.")
        elif entered == str(expected_password).strip():
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")

    return False


# Gate: ensure the rest of the app does not render until authenticated.
if not check_password():
    st.stop()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "phase" not in st.session_state:
    st.session_state.phase = "upload"
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "diagnosis" not in st.session_state:
    st.session_state.diagnosis = None
if "reasoning_trace" not in st.session_state:
    st.session_state.reasoning_trace = None
if "trace_sidebar_open" not in st.session_state:
    st.session_state.trace_sidebar_open = False


def reset() -> None:
    st.session_state.phase = "upload"
    st.session_state.session_id = None
    st.session_state.analysis = None
    st.session_state.diagnosis = None
    st.session_state.reasoning_trace = None
    st.session_state.trace_sidebar_open = False


# ---------------------------------------------------------------------------
# Header + trace sidebar
# ---------------------------------------------------------------------------

def _render_trace_sidebar() -> None:
    trace = st.session_state.reasoning_trace
    if not trace:
        return
    if not st.session_state.trace_sidebar_open:
        return
    with st.sidebar:
        st.markdown("### 📋 Reasoning trace")
        st.caption("Full pipeline steps (vision → RAG → diagnosis → care plan).")
        if st.button("Close panel", key="close_trace_sidebar"):
            st.session_state.trace_sidebar_open = False
            st.rerun()
        st.divider()
        trace_json = json.dumps(trace, indent=2, default=str)
        st.download_button(
            "Download trace (JSON)",
            data=trace_json,
            file_name="plantdoc_trace.json",
            mime="application/json",
            key="dl_trace",
        )
        with st.container(height=600):
            for i, entry in enumerate(trace):
                step = entry.get("step", "?")
                ts = entry.get("timestamp", "")
                st.markdown(f"**{i + 1}. `{step}`**  \n_{ts}_")
                st.json(entry.get("data", {}))
                st.divider()


_render_trace_sidebar()

_h1, _h2 = st.columns([4, 1])
with _h1:
    st.title("🌿 PlantDoc AI")
    st.caption("Upload a photo of your plant and get an AI-powered health diagnosis.")
with _h2:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if st.session_state.reasoning_trace:
        if st.button("📋 Full trace", key="open_trace_top", help="Open sidebar with full reasoning trace"):
            st.session_state.trace_sidebar_open = True
            st.rerun()


# ---------------------------------------------------------------------------
# Phase 1 — Upload and analyze
# ---------------------------------------------------------------------------

if st.session_state.phase == "upload":
    uploaded = st.file_uploader(
        "Upload a plant photo",
        type=["jpg", "jpeg", "png", "webp"],
        key="uploader",
    )
    description = st.text_input(
        "Describe what you're seeing (optional)",
        placeholder="e.g. The leaves are turning yellow and drooping",
    )

    if uploaded and st.button("Analyze Plant", type="primary"):
        st.session_state.reasoning_trace = None
        progress_placeholder = st.empty()
        step_order = ["vision", "retrieval", "web_search", "questions"]
        steps: dict[str, str] = {}

        def render_progress():
            if not steps:
                progress_placeholder.markdown("**Analyzing your plant...**")
                return
            lines = []
            for step in step_order:
                if step not in steps:
                    continue
                lines.append(f"✓ **{step.replace('_', ' ').title()}:** {steps[step]}")
            progress_placeholder.markdown("\n\n".join(lines))

        try:
            files = {"image": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            data = {"description": description}
            resp = requests.post(
                f"{API_BASE}/analyze/stream",
                files=files,
                data=data,
                timeout=120,
                stream=True,
            )
            resp.raise_for_status()

            result = None
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if event.get("type") == "progress":
                    step = event.get("step", "unknown")
                    msg = event.get("message", "")
                    steps[step] = msg
                    render_progress()

                elif event.get("type") == "complete":
                    result = event
                    break

                elif event.get("type") == "error":
                    progress_placeholder.empty()
                    st.error(f"Analysis failed: {event.get('message', 'Unknown error')}")
                    st.stop()

            progress_placeholder.empty()

            if result is None:
                st.error("No response received from the server.")
                st.stop()

            rt = result.get("reasoning_trace")
            if rt is not None:
                st.session_state.reasoning_trace = rt

            if not result.get("is_plant", True):
                description_text = result.get(
                    "description",
                    "Please upload a photo of a real plant to get a diagnosis.",
                )
                st.markdown(
                    f"""<div style="background: #FFF8E7; border-left: 4px solid #FFB020;
                    border-radius: 8px; padding: 20px 24px; margin: 12px 0;">
                    <span style="font-size: 28px; vertical-align: middle;">🌱</span>
                    <strong style="font-size: 16px; margin-left: 8px;">Not a real plant</strong>
                    <p style="margin: 10px 0 0; color: #444; font-size: 15px; line-height: 1.5;">
                    {description_text}</p></div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.session_state.session_id = result["session_id"]
                st.session_state.analysis = {
                    "is_plant": True,
                    "session_id": result["session_id"],
                    "plant_type_guess": result.get("plant_type_guess", ""),
                    "symptoms_detected": result.get("symptoms_detected", []),
                    "confidence": result.get("confidence", 0.0),
                    "description": result.get("description", ""),
                    "diagnostic_questions": result.get("diagnostic_questions", []),
                }
                st.session_state.phase = "questions"
                st.rerun()
        except requests.RequestException as e:
            progress_placeholder.empty()
            st.error(f"Failed to connect to the backend: {e}")
        except Exception as e:
            progress_placeholder.empty()
            st.error(f"Analysis failed: {e}")


# ---------------------------------------------------------------------------
# Phase 2 — Show analysis & ask questions
# ---------------------------------------------------------------------------

elif st.session_state.phase == "questions":
    analysis = st.session_state.analysis

    st.subheader("Initial Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Plant Type", analysis.get("plant_type_guess", "Unknown"))
    with col2:
        confidence = analysis.get("confidence", 0)
        st.metric("Confidence", f"{confidence:.0%}")

    symptoms = analysis.get("symptoms_detected", [])
    if symptoms:
        st.write("**Symptoms detected:**")
        for s in symptoms:
            st.write(f"- {s}")

    if analysis.get("description"):
        with st.expander("Detailed description"):
            st.write(analysis["description"])

    st.divider()
    st.subheader("Diagnostic Questions")
    st.write("Please answer these questions to help refine the diagnosis:")

    questions = analysis.get("diagnostic_questions", [])
    answers = []
    for i, q in enumerate(questions):
        answer = st.text_input(q, key=f"q_{i}", placeholder="Your answer...")
        answers.append(answer)

    # Buttons in columns; progress UI below uses full page width (not trapped in narrow column).
    col_submit, _col_spacer, col_reset = st.columns([2, 4, 2])
    with col_submit:
        get_diagnosis = st.button("Get Diagnosis", type="primary")
    with col_reset:
        # NBSP keeps "Start" + "Over" from breaking across lines if width is tight
        if st.button("Start\u00a0Over", type="secondary", key="questions_start_over"):
            reset()
            st.rerun()

    progress_placeholder = st.empty()

    if get_diagnosis:
        if not any(answers):
            st.warning("Please answer at least one question.")
        else:
            step_order = ["diagnosis", "web_search", "rediagnosis", "care_plan"]
            steps: dict[str, str] = {}

            def render_progress() -> None:
                if not steps:
                    progress_placeholder.markdown("**Generating diagnosis...**")
                    return
                lines = []
                for step in step_order:
                    if step not in steps:
                        continue
                    lines.append(f"✓ **{step.replace('_', ' ').title()}:** {steps[step]}")
                progress_placeholder.markdown("\n\n".join(lines))

            try:
                resp = requests.post(
                    f"{API_BASE}/diagnose/stream",
                    json={
                        "session_id": st.session_state.session_id,
                        "answers": answers,
                    },
                    timeout=120,
                    stream=True,
                )
                resp.raise_for_status()

                result = None
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if event.get("type") == "progress":
                        step = event.get("step", "unknown")
                        msg = event.get("message", "")
                        steps[step] = msg
                        render_progress()

                    elif event.get("type") == "complete":
                        result = event.get("diagnosis")
                        if event.get("reasoning_trace") is not None:
                            st.session_state.reasoning_trace = event["reasoning_trace"]
                        break

                    elif event.get("type") == "error":
                        progress_placeholder.empty()
                        st.error(f"Diagnosis failed: {event.get('message', 'Unknown error')}")
                        st.stop()

                progress_placeholder.empty()

                if result is not None:
                    st.session_state.diagnosis = result
                    st.session_state.phase = "result"
                    st.rerun()
                else:
                    st.error("No response received from the server.")
            except requests.RequestException as e:
                progress_placeholder.empty()
                st.error(f"Failed to connect to the backend: {e}")
            except Exception as e:
                progress_placeholder.empty()
                st.error(f"Diagnosis failed: {e}")


# ---------------------------------------------------------------------------
# Phase 3 — Show diagnosis
# ---------------------------------------------------------------------------

elif st.session_state.phase == "result":
    diag = st.session_state.diagnosis

    st.subheader("Diagnosis")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Plant", diag.get("plant_type_guess", "Unknown"))
    with col2:
        confidence = diag.get("confidence", 0)
        color = "🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.4 else "🔴"
        st.metric("Confidence", f"{color} {confidence:.0%}")

    st.info(diag.get("diagnosis", "No diagnosis available."))

    st.divider()
    st.subheader("Treatment Plan")
    for step in diag.get("treatment_plan", []):
        st.markdown(step)

    if diag.get("recovery_timeline"):
        st.divider()
        st.subheader("Recovery Timeline")
        chunks = _split_recovery_timeline(diag["recovery_timeline"])
        st.markdown("\n\n".join(chunks))

    if diag.get("warning_signs"):
        st.divider()
        st.subheader("⚠️ Warning Signs")
        for w in diag["warning_signs"]:
            st.write(f"- {w}")

    st.divider()
    if st.button("Diagnose Another Plant", type="primary"):
        reset()
        st.rerun()
