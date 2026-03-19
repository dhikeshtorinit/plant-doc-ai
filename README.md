# PlantDoc AI

AI-powered plant health diagnosis: upload a photo, answer a few questions, and get a diagnosis with treatment and recovery guidance. Built with FastAPI, Streamlit, LangGraph, and OpenAI.

## Features

- **Vision analysis** — Detects plant type and symptoms from a photo (or gently rejects non-plant images).
- **RAG + optional web search** — Uses a local plant-knowledge base (ChromaDB); triggers Tavily web search when relevance is low or diagnosis confidence is low.
- **Two-phase flow** — Phase 1: analyze image → diagnostic questions. Phase 2: your answers → diagnosis, treatment plan, recovery timeline, warning signs.
- **Reasoning trace** — Full pipeline trace (vision → RAG → questions → diagnosis → care plan) viewable and downloadable in the UI.

## Quick start

1. **Clone and install**

   ```bash
   cd PlantDocAI
   python -m venv .venv
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Configure**

   ```bash
   cp .env.example .env
   # Edit .env and set OPENAI_API_KEY (required). Optionally set TAVILY_API_KEY.
   ```

3. **Run**

   - Start the API (from project root):

     ```bash
     uvicorn backend.api.server:app --reload --host 0.0.0.0 --port 8000
     ```

   - In another terminal, start the Streamlit app:

     ```bash
     streamlit run frontend/streamlit_app.py
     ```

   - Open the URL shown by Streamlit (e.g. http://localhost:8501), upload a plant photo, and follow the flow.

## Deployment (demo / production)

- **Backend**
  - Set `CORS_ORIGINS` to your frontend origin(s), comma-separated (e.g. `https://app.yourdomain.com`). Use `*` only for local/dev.
  - Optionally set `MAX_UPLOAD_MB` (default 10).
  - Run with a production ASGI server, e.g. `uvicorn backend.api.server:app --host 0.0.0.0 --port 8000` (no `--reload`).
  - Sessions are in-memory; for multi-instance or long-lived deployments, replace with Redis or a database.

- **Frontend**
  - Set `PLANTDOC_API_BASE` to your backend URL (e.g. `https://api.yourdomain.com`) so the app talks to the correct API.
  - Run Streamlit behind a reverse proxy (e.g. Nginx) with HTTPS if needed.

- **Secrets**
  - Never commit `.env`. Ensure `OPENAI_API_KEY` (and `TAVILY_API_KEY` if used) are set in the environment or a secure secret store in production.

## Project layout

- `backend/api/server.py` — FastAPI app: `/health`, `/analyze`, `/analyze/stream`, `/diagnose`, `/diagnose/stream`, `/sessions/{id}`.
- `backend/agent/workflow.py` — LangGraph Phase 1 (vision → RAG → web search? → questions) and Phase 2 (diagnosis → web search? → care plan → logging).
- `backend/agent/modules/` — Vision, RAG, questions, diagnosis, care plan.
- `backend/agent/tools/web_search.py` — Tavily integration and trigger logic.
- `frontend/streamlit_app.py` — Streamlit UI (upload → questions → diagnosis + trace panel).
- `data/plant_knowledge/` — Source documents for the knowledge base (loaded into ChromaDB at startup).

## License

Use as you like; no warranty.
