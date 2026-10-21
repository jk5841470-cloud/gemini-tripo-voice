"""Small JSON-file backed state.

Why persist it?
* The "last generated model" task_id must survive a browser refresh / server restart, otherwise
  "separate the parts" would fail with "no model yet" after any reconnect.
* Segmentation results are cached per source task_id, so asking twice costs 0 extra credits.
* A local monthly spend counter gives a cheap guard against burning the free 200 credits.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("state")

MAX_CACHED_SEGMENTATIONS = 5


def _month_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


@dataclass
class ModelRecord:
    task_id: str
    prompt: str
    model_url: str
    preview_url: Optional[str]
    created_at: str


class AppState:
    def __init__(self, path: Path, monthly_budget: float):
        self.path = path
        self.monthly_budget = monthly_budget
        self.last_model: Optional[ModelRecord] = None
        self._segmentations: dict[str, dict] = {}
        self._spend_month = _month_key()
        self._spent = 0.0

    # ------------------------------------------------------------------ persistence
    @classmethod
    def load(cls, path: Path, monthly_budget: float) -> "AppState":
        state = cls(path, monthly_budget)
        try:
            raw = json.loads(path.read_text())
            if raw.get("last_model"):
                state.last_model = ModelRecord(**raw["last_model"])
            state._segmentations = raw.get("segmentations", {})
            budget = raw.get("budget", {})
            state._spend_month = budget.get("month", state._spend_month)
            state._spent = float(budget.get("spent", 0.0))
        except FileNotFoundError:
            pass
        except Exception:
            log.exception("Could not read %s; starting with empty state", path)
        state._roll_month()
        return state

    def _save(self) -> None:
        payload = {
            "last_model": asdict(self.last_model) if self.last_model else None,
            "segmentations": self._segmentations,
            "budget": {"month": self._spend_month, "spent": self._spent},
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.path)  # atomic: never leaves a half-written file

    # ------------------------------------------------------------------ model tracking
    def set_last_model(self, record: ModelRecord) -> None:
        self.last_model = record
        self._save()

    def get_segmentation(self, source_task_id: str) -> Optional[dict]:
        return self._segmentations.get(source_task_id)

    def save_segmentation(self, source_task_id: str, result: dict) -> None:
        self._segmentations[source_task_id] = result
        while len(self._segmentations) > MAX_CACHED_SEGMENTATIONS:
            self._segmentations.pop(next(iter(self._segmentations)))
        self._save()

    # ------------------------------------------------------------------ credits
    def _roll_month(self) -> None:
        # Calendar-month approximation. Tripo's real reset follows your billing cycle.
        if self._spend_month != _month_key():
            self._spend_month, self._spent = _month_key(), 0.0

    @property
    def spent(self) -> float:
        self._roll_month()
        return self._spent

    @property
    def remaining(self) -> Optional[float]:
        """None when the guard is disabled (budget <= 0)."""
        if self.monthly_budget <= 0:
            return None
        return max(self.monthly_budget - self.spent, 0.0)

    def can_afford(self, cost: float) -> bool:
        return self.monthly_budget <= 0 or self.spent + cost <= self.monthly_budget

    def add_spend(self, amount: float) -> None:
        self._roll_month()
        self._spent += amount
        self._save()
