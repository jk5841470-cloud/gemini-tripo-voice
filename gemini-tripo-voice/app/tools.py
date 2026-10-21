"""The two tools Gemini can call (plus a status helper).

This module knows about Tripo and credits but NOTHING about Gemini or audio, so you can unit test
or reuse it from a CLI / another agent framework.

Contract: every public coroutine returns a plain dict and NEVER raises. Errors come back as
    {"status": "error", "error_type": "...", "message": "<speakable sentence>"}
so Gemini always has something to say to the user.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional

from .config import Settings
from .glb import extract_parts
from .state import AppState, ModelRecord
from .tripo_client import TripoClient, TripoError

log = logging.getLogger("tools")

EventCallback = Callable[[dict], Awaitable[None]]


def _error(error_type: str, message: str, **extra) -> dict:
    return {"status": "error", "error_type": error_type, "message": message, **extra}


class ToolService:
    def __init__(self, tripo: TripoClient, state: AppState, settings: Settings):
        self.tripo = tripo
        self.state = state
        self.cfg = settings
        # Only one Tripo job at a time: protects credits from accidental double-requests
        # (e.g. the user repeating themselves while the first job is still running).
        self._busy: Optional[str] = None
        self._job: dict = {"kind": None, "status": "idle", "progress": 0, "started_at": None}

    # ------------------------------------------------------------------ job bookkeeping
    def job_status(self) -> dict:
        """Snapshot Gemini can read out when the user asks 'is it done yet?'."""
        job = dict(self._job)
        if job["started_at"]:
            job["elapsed_seconds"] = int(time.monotonic() - job["started_at"])
        job.pop("started_at", None)
        job["last_model_prompt"] = self.state.last_model.prompt if self.state.last_model else None
        job["credits_remaining_estimate"] = self.state.remaining
        return job

    async def _set_job(self, on_event: Optional[EventCallback], kind: str, status: str, progress: int, message: str) -> None:
        self._job.update(kind=kind, status=status, progress=progress)
        if status in ("queued", "running") and not self._job.get("started_at"):
            self._job["started_at"] = time.monotonic()
        if on_event:
            await on_event({"type": "job", "kind": kind, "status": status, "progress": progress, "message": message})

    def _progress_cb(self, on_event: Optional[EventCallback], kind: str):
        async def cb(status: str, progress: int) -> None:
            await self._set_job(on_event, kind, status, progress, f"{kind} {status} ({progress}%)")
        return cb

    # ------------------------------------------------------------------ tool 1: generate
    async def generate_3d_model(self, prompt: str, on_event: Optional[EventCallback] = None) -> dict:
        prompt = (prompt or "").strip()
        if not prompt:
            return _error("invalid_prompt", "I need a description of what to build, for example 'a wooden treasure chest'.")
        if self._busy:
            return _error("busy", f"A {self._busy} job is still running. Let's wait for it to finish first.")
        cost = self.cfg.cost_generation
        if not self.state.can_afford(cost):
            return _error(
                "budget",
                f"That would cost about {cost:.0f} credits but only about {self.state.remaining:.0f} are left in this month's budget.",
            )

        self._busy = "generation"
        self._job["started_at"] = None
        try:
            await self._set_job(on_event, "generation", "queued", 0, "Submitting to Tripo")
            task_id = await self.tripo.create_text_to_model(
                prompt,
                model=self.cfg.tripo_generation_model,
                texture=self.cfg.texture,
                pbr=self.cfg.pbr,
            )
            log.info("generation task %s started for prompt %r", task_id, prompt)
            data = await self.tripo.wait_for_task(
                task_id,
                timeout=self.cfg.generation_timeout_s,
                poll_interval=self.cfg.poll_interval_s,
                on_progress=self._progress_cb(on_event, "generation"),
            )
            output = data.get("output") or {}
            # "model_url" per the v3 docs; older field names kept as a fallback.
            model_url = output.get("model_url") or output.get("pbr_model") or output.get("model")
            if not model_url:
                raise TripoError("Tripo finished but did not return a model file.", task_id=task_id)

            spent = float(data.get("credits_consumed") or cost)
            self.state.add_spend(spent)
            self.state.set_last_model(
                ModelRecord(
                    task_id=task_id,
                    prompt=prompt,
                    model_url=model_url,
                    preview_url=output.get("rendered_image_url"),
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
            )
            await self._set_job(on_event, "generation", "success", 100, "Model ready")
            if on_event:
                await on_event({"type": "model", "kind": "generated", "url": model_url, "preview": output.get("rendered_image_url")})
            return {
                "status": "success",
                "task_id": task_id,
                "prompt_used": prompt,
                "model_url": model_url,
                "preview_url": output.get("rendered_image_url"),
                "credits_used": spent,
                "credits_remaining_estimate": self.state.remaining,
                "assistant_hint": (
                    "Tell the user the model is ready and is shown on their screen. Do NOT read the URL aloud. "
                    "Offer to separate it into parts and explain them (that costs about "
                    f"{self.cfg.cost_segmentation:.0f} more credits)."
                ),
            }
        except TripoError as exc:
            log.warning("generation failed: %s", exc.message)
            await self._set_job(on_event, "generation", "failed", self._job["progress"], exc.message)
            return _error(exc.error_type, exc.message, task_id=exc.task_id)
        except Exception:
            log.exception("unexpected error during generation")
            await self._set_job(on_event, "generation", "failed", self._job["progress"], "Unexpected error")
            return _error("unexpected", "Something unexpected went wrong while generating the model.")
        finally:
            self._busy = None

    # ------------------------------------------------------------------ tool 2: segment + explain
    async def separate_parts_and_explain(self, on_event: Optional[EventCallback] = None) -> dict:
        last = self.state.last_model
        if last is None:
            return _error("no_model", "There's no model to separate yet. Ask me to generate one first, for example 'make a 3D model of a robot'.")

        cached = self.state.get_segmentation(last.task_id)
        if cached:
            if on_event and cached.get("segmented_model_url"):
                await on_event({"type": "model", "kind": "segmented", "url": cached["segmented_model_url"], "preview": cached.get("preview_url")})
            return {**cached, "from_cache": True, "credits_used": 0, "assistant_hint": cached.get("assistant_hint", "")}

        if self._busy:
            return _error("busy", f"A {self._busy} job is still running. Let's wait for it to finish first.")
        cost = self.cfg.cost_segmentation
        if not self.state.can_afford(cost):
            return _error(
                "budget",
                f"Segmentation costs about {cost:.0f} credits but only about {self.state.remaining:.0f} are left in this month's budget.",
            )

        self._busy = "segmentation"
        self._job["started_at"] = None
        try:
            await self._set_job(on_event, "segmentation", "queued", 0, "Submitting to Tripo")
            seg_task_id = await self.tripo.create_mesh_segmentation(
                last.task_id,
                model=self.cfg.tripo_segmentation_model,
                granularity=self.cfg.tripo_segmentation_granularity,
            )
            log.info("segmentation task %s started for model task %s", seg_task_id, last.task_id)
            data = await self.tripo.wait_for_task(
                seg_task_id,
                timeout=self.cfg.segmentation_timeout_s,
                poll_interval=self.cfg.poll_interval_s,
                on_progress=self._progress_cb(on_event, "segmentation"),
            )
            output = data.get("output") or {}
            seg_url = output.get("model_url") or output.get("pbr_model") or output.get("model")
            if not seg_url:
                raise TripoError("Tripo finished segmenting but did not return a model file.", task_id=seg_task_id)

            spent = float(data.get("credits_consumed") or cost)
            self.state.add_spend(spent)

            # Part info lives inside the GLB. If reading it fails we still succeed (the user has
            # the segmented model) and tell Gemini to explain in general terms.
            parts_info: dict = {"labels_are_semantic": False, "total_instances": 0, "parts": [], "omitted_small_parts": 0}
            read_error = None
            try:
                parts_info = await extract_parts(seg_url)
            except Exception as exc:
                log.exception("could not read parts from segmented GLB")
                read_error = str(exc)

            result = {
                "status": "success",
                "segmentation_task_id": seg_task_id,
                "original_prompt": last.prompt,
                "segmented_model_url": seg_url,
                "preview_url": output.get("rendered_image_url"),
                "part_count": parts_info["total_instances"],
                "labels_are_semantic": parts_info["labels_are_semantic"],
                "parts": parts_info["parts"],
                "omitted_small_parts": parts_info["omitted_small_parts"],
                "part_info_error": read_error,
                "credits_used": spent,
                "credits_remaining_estimate": self.state.remaining,
                "assistant_hint": self._explain_hint(parts_info, read_error),
            }
            self.state.save_segmentation(last.task_id, result)
            await self._set_job(on_event, "segmentation", "success", 100, "Parts ready")
            if on_event:
                await on_event({"type": "model", "kind": "segmented", "url": seg_url, "preview": output.get("rendered_image_url"), "parts": parts_info["parts"]})
            return result
        except TripoError as exc:
            log.warning("segmentation failed: %s", exc.message)
            await self._set_job(on_event, "segmentation", "failed", self._job["progress"], exc.message)
            return _error(exc.error_type, exc.message, task_id=exc.task_id)
        except Exception:
            log.exception("unexpected error during segmentation")
            await self._set_job(on_event, "segmentation", "failed", self._job["progress"], "Unexpected error")
            return _error("unexpected", "Something unexpected went wrong while separating the parts.")
        finally:
            self._busy = None

    @staticmethod
    def _explain_hint(parts_info: dict, read_error: Optional[str]) -> str:
        if read_error or not parts_info["parts"]:
            return (
                "The model was segmented but I could not read the part list. Tell the user it is split and shown on "
                "screen, then explain in general terms what parts an object like 'original_prompt' typically has, "
                "clearly saying you are describing it generally rather than reading exact labels."
            )
        if parts_info["labels_are_semantic"]:
            return (
                "Explain each part by voice in a natural, conversational way: one or two sentences each covering what it "
                "is, what it does, and where it sits (use 'location' and 'relative_size'). Group repeated parts using "
                "'count'. Start with a one-sentence overview of how many parts there are. Do not read raw field names."
            )
        return (
            "The parts have no reliable names. Using 'original_prompt' plus each part's location and relative size, say "
            "what each part most likely is and does, and be upfront that you are inferring the names. Keep it short and natural."
        )
