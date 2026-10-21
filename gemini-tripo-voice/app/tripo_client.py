"""Thin async client for the Tripo v3 REST API.

Endpoints used (all return {"code": 0, "data": {...}}):
    POST /generation/text-to-model   -> {"task_id": ...}
    POST /mesh/segment               -> {"task_id": ...}
    GET  /tasks/{task_id}            -> {"status", "progress", "output": {"model_url", ...}, ...}

Design notes
------------
* Every failure is converted into a TripoError subclass whose `message` is written to be
  spoken to the user, so the voice layer never has to interpret raw HTTP errors.
* GET requests (polling) are retried on network blips. POST requests that CREATE tasks are
  never blindly retried after a timeout, because the task may already exist and a retry
  could double-charge credits.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Optional

import httpx

log = logging.getLogger("tripo")

# Terminal task states that are not "success". "banned"/"expired" existed in v2; harmless here.
TERMINAL_FAILURE = {"failed", "cancelled", "banned", "expired"}
# Best-effort: v2 used code 2010 for "insufficient credit". v3 may differ, so we also
# match on HTTP 402 and on the error text (see _looks_like_credit_problem).
INSUFFICIENT_CREDIT_CODES = {2010}

ProgressCallback = Callable[[str, int], Awaitable[None]]


# --------------------------------------------------------------------------- errors
class TripoError(Exception):
    """Base class. `message` is safe/useful to read aloud to the user."""

    error_type = "tripo_error"

    def __init__(self, message: str, *, task_id: Optional[str] = None, code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.task_id = task_id
        self.code = code


class TripoAuthError(TripoError):
    error_type = "auth"


class TripoInsufficientCreditsError(TripoError):
    error_type = "insufficient_credits"


class TripoRateLimitError(TripoError):
    error_type = "rate_limited"


class TripoTaskFailedError(TripoError):
    error_type = "task_failed"


class TripoTimeoutError(TripoError):
    error_type = "timeout"


class TripoNetworkError(TripoError):
    error_type = "network"


def _looks_like_credit_problem(text: str) -> bool:
    t = text.lower()
    return "credit" in t and any(w in t for w in ("insufficient", "enough", "exhaust", "balance", "run out", "need more"))


# --------------------------------------------------------------------------- client
class TripoClient:
    def __init__(self, api_key: str, base_url: str, request_timeout: float = 30.0):
        # Trailing slash + relative paths keeps the "/v3" prefix when httpx merges URLs.
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=httpx.Timeout(request_timeout),
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------ low level
    async def _request(self, method: str, path: str, *, json: Optional[dict] = None, retries: int = 0) -> Any:
        attempt = 0
        while True:
            try:
                resp = await self._http.request(method, path, json=json)
                break
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt < retries:
                    attempt += 1
                    delay = min(2 ** attempt, 8)
                    log.warning("Tripo %s %s failed (%s); retry %d/%d in %ss", method, path, exc.__class__.__name__, attempt, retries, delay)
                    await asyncio.sleep(delay)
                    continue
                if method == "POST":
                    raise TripoNetworkError(
                        "I couldn't reach Tripo before the request timed out. The job may or may not have started, "
                        "so please check your Tripo dashboard before trying again."
                    ) from exc
                raise TripoNetworkError("I lost the connection to Tripo while checking on the job.") from exc
        return self._parse(resp)

    @staticmethod
    def _parse(resp: httpx.Response) -> Any:
        try:
            body = resp.json()
        except ValueError:
            body = {}
        code = body.get("code") if isinstance(body, dict) else None
        message = ""
        if isinstance(body, dict):
            message = str(body.get("message") or body.get("error") or body.get("suggestion") or "")
        if not message:
            message = resp.text[:200]

        if resp.is_success and code in (None, 0):
            if isinstance(body, dict) and "data" in body:
                return body["data"]
            return body

        # Order matters: some APIs answer 403 for "no credits", so check credits before auth.
        if resp.status_code == 402 or code in INSUFFICIENT_CREDIT_CODES or _looks_like_credit_problem(message):
            raise TripoInsufficientCreditsError(
                "Tripo says this account is out of credits, so I can't start that job.", code=code
            )
        if resp.status_code in (401, 403):
            raise TripoAuthError("Tripo rejected the API key. Please check TRIPO_API_KEY.", code=code)
        if resp.status_code == 429:
            raise TripoRateLimitError("Tripo is rate limiting me right now. Please try again in a minute.", code=code)
        raise TripoError(f"Tripo returned an error ({resp.status_code}, code {code}): {message}", code=code)

    @staticmethod
    def _task_id(data: Any) -> str:
        task_id = data.get("task_id") if isinstance(data, dict) else None
        if not task_id:
            raise TripoError(f"Tripo did not return a task id (got: {data!r})")
        return str(task_id)

    # ------------------------------------------------------------------ task creation
    async def create_text_to_model(self, prompt: str, *, model: str, texture: bool = True, pbr: bool = True) -> str:
        payload = {
            "prompt": prompt[:1024],  # documented limit
            "model": model,
            "texture": texture,
            "pbr": pbr and texture,  # pbr=true forces texture=true on Tripo's side anyway
        }
        data = await self._request("POST", "generation/text-to-model", json=payload)
        return self._task_id(data)

    async def create_mesh_segmentation(self, input_task_id: str, *, model: str, granularity: str = "balanced") -> str:
        payload: dict[str, Any] = {"input": input_task_id, "model": model}
        if model.startswith("v2"):
            # granularity is a v2-only parameter; sending it to v1 could be rejected.
            payload["segmentation_granularity"] = granularity
        data = await self._request("POST", "mesh/segment", json=payload)
        return self._task_id(data)

    # ------------------------------------------------------------------ polling
    async def get_task(self, task_id: str) -> dict:
        return await self._request("GET", f"tasks/{task_id}", retries=3)

    async def wait_for_task(
        self,
        task_id: str,
        *,
        timeout: float,
        poll_interval: float = 3.0,
        on_progress: Optional[ProgressCallback] = None,
    ) -> dict:
        """Poll until the task succeeds. Raises TripoTaskFailedError / TripoTimeoutError otherwise.

        The poll interval backs off gently (x1.25 up to 8s) to stay far below rate limits.
        On timeout the task keeps running on Tripo's side; its id is attached to the error.
        """
        started = time.monotonic()
        interval = poll_interval
        last_seen: tuple[Optional[str], int] = (None, -1)

        while True:
            data = await self.get_task(task_id)
            status = str(data.get("status", "")).lower()
            try:
                progress = int(data.get("progress") or 0)
            except (TypeError, ValueError):
                progress = 0

            if on_progress and (status, progress) != last_seen:
                last_seen = (status, progress)
                try:
                    await on_progress(status, progress)
                except Exception:  # a UI hiccup must never kill the poll loop
                    log.exception("progress callback failed")

            if status == "success":
                return data

            if status in TERMINAL_FAILURE:
                detail = str(data.get("error_message") or f"task ended with status '{status}'")
                if _looks_like_credit_problem(detail):
                    raise TripoInsufficientCreditsError(
                        "Tripo says this account is out of credits.", task_id=task_id, code=data.get("error_code")
                    )
                raise TripoTaskFailedError(
                    f"Tripo couldn't finish the job: {detail}.", task_id=task_id, code=data.get("error_code")
                )

            if time.monotonic() - started > timeout:
                raise TripoTimeoutError(
                    f"The job is taking longer than {int(timeout)} seconds. It may still finish on Tripo's side "
                    f"(task {task_id}).",
                    task_id=task_id,
                )

            await asyncio.sleep(interval)
            interval = min(interval * 1.25, 8.0)
