"""FastAPI entrypoint.

    uvicorn app.main:app --host 127.0.0.1 --port 8000

Serves the browser client at "/" and the voice websocket at "/ws".
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from google import genai

from .config import ROOT, settings
from .live_session import LiveVoiceSession
from .state import AppState
from .tools import ToolService
from .tripo_client import TripoClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.validate()  # raises a readable error if keys are missing
    tripo = TripoClient(settings.tripo_api_key, settings.tripo_base_url)
    state = AppState.load(settings.state_file, settings.monthly_credit_budget)
    app.state.service = ToolService(tripo, state, settings)
    app.state.genai = genai.Client(api_key=settings.gemini_api_key)
    app.state.app_state = state
    log.info("Ready. Live model=%s tool_mode=%s tripo=%s", settings.gemini_live_model, settings.tool_mode, settings.tripo_base_url)
    yield
    await tripo.aclose()


app = FastAPI(title="Gemini Live + Tripo 3D voice assistant", lifespan=lifespan)


@app.get("/api/status")
async def status() -> dict:
    """Handy for debugging: what is the 'last model' and how many credits are tracked as spent."""
    st: AppState = app.state.app_state
    return {
        "last_model": st.last_model.__dict__ if st.last_model else None,
        "credits_spent_this_month": st.spent,
        "credits_remaining_estimate": st.remaining,
        "job": app.state.service.job_status(),
    }


@app.websocket("/ws")
async def voice_ws(ws: WebSocket) -> None:
    await ws.accept()
    session = LiveVoiceSession(ws, ws.app.state.service, settings, ws.app.state.genai)
    await session.run()


# Mounted last so it doesn't shadow /api and /ws.
app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
