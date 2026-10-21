"""Central configuration.

Everything tunable lives in environment variables (or a local .env file) so you can
change models, timeouts and credit behaviour without touching code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Keys
    gemini_api_key: str
    tripo_api_key: str

    # Gemini Live
    gemini_live_model: str
    gemini_voice: str
    tool_mode: str  # background | async | blocking

    # Tripo
    tripo_base_url: str
    tripo_generation_model: str
    tripo_segmentation_model: str
    tripo_segmentation_granularity: str
    texture: bool
    pbr: bool
    poll_interval_s: float
    generation_timeout_s: float
    segmentation_timeout_s: float

    # Credits
    monthly_credit_budget: float
    cost_generation: float
    cost_segmentation: float

    # Persistence
    state_file: Path

    def validate(self) -> None:
        """Fail fast with a readable message instead of a cryptic 401 later."""
        missing = [
            name
            for name, value in (
                ("GEMINI_API_KEY", self.gemini_api_key),
                ("TRIPO_API_KEY", self.tripo_api_key),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"Missing environment variable(s): {', '.join(missing)}. "
                "Copy .env.example to .env and fill them in."
            )
        if self.tool_mode not in {"background", "async", "blocking"}:
            raise RuntimeError("TOOL_MODE must be one of: background, async, blocking")


def load_settings() -> Settings:
    texture = _bool("TRIPO_TEXTURE", True)
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        tripo_api_key=os.getenv("TRIPO_API_KEY", "").strip(),
        gemini_live_model=os.getenv("GEMINI_LIVE_MODEL", "gemini-3.8-live").strip(),
        gemini_voice=os.getenv("GEMINI_VOICE", "Puck").strip(),
        tool_mode=os.getenv("TOOL_MODE", "background").strip().lower(),
        tripo_base_url=os.getenv("TRIPO_BASE_URL", "https://openapi.tripo3d.ai/v3").strip(),
        tripo_generation_model=os.getenv("TRIPO_GENERATION_MODEL", "v3.1-20260211").strip(),
        tripo_segmentation_model=os.getenv("TRIPO_SEGMENTATION_MODEL", "v2.0-20260430").strip(),
        tripo_segmentation_granularity=os.getenv("TRIPO_SEGMENTATION_GRANULARITY", "balanced").strip(),
        texture=texture,
        # Tripo forces texture=true whenever pbr=true, so only allow PBR with texture.
        pbr=_bool("TRIPO_PBR", True) and texture,
        poll_interval_s=_float("TRIPO_POLL_INTERVAL", 3.0),
        generation_timeout_s=_float("TRIPO_GENERATION_TIMEOUT", 300.0),
        segmentation_timeout_s=_float("TRIPO_SEGMENTATION_TIMEOUT", 420.0),
        monthly_credit_budget=_float("MONTHLY_CREDIT_BUDGET", 200.0),
        cost_generation=_float("TRIPO_COST_GENERATION", 20.0 if texture else 10.0),
        cost_segmentation=_float("TRIPO_COST_SEGMENTATION", 40.0),
        state_file=Path(os.getenv("STATE_FILE", str(ROOT / "state.json"))),
    )


settings = load_settings()
