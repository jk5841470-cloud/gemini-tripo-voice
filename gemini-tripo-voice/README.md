# Voice 3D Studio — Gemini Live + Tripo AI

Talk to a Gemini Live voice agent. Say *"Get me a 3D model of a car heart"* and Tripo builds it; say
*"Separate the parts and explain them"* and Tripo segments the last model while Gemini explains every part out loud.

## Project structure

```
gemini-tripo-voice/
├── .env.example            # copy to .env and add your two API keys
├── requirements.txt
├── README.md
├── app/
│   ├── config.py           # all settings from env vars
│   ├── tripo_client.py     # async Tripo v3 client: create task, poll, typed errors
│   ├── state.py            # last model task_id, segmentation cache, monthly credit guard (state.json)
│   ├── glb.py              # reads part names/positions out of the segmented GLB (streams only its header)
│   ├── tools.py            # generate_3d_model / separate_parts_and_explain / job status (no Gemini code here)
│   ├── live_session.py     # Gemini Live bridge: audio, tool declarations, async job delivery
│   └── main.py             # FastAPI: /ws voice socket, /api/status, serves the web UI
├── static/index.html       # browser client: mic, speaker, transcript, progress bar, 3D viewer
└── tests/test_offline.py   # no-network smoke tests
```

## 1. Get the keys

* **Gemini**: https://aistudio.google.com/apikey
* **Tripo**: create an API key in the Tripo developer platform (https://developers.tripo3d.com/).

## 2. Install

Python 3.10+ (3.11+ recommended).

```bash
cd gemini-tripo-voice
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -U google-genai                             # Live API needs a recent SDK
cp .env.example .env                                    # then edit .env: GEMINI_API_KEY, TRIPO_API_KEY
```

## 3. Run

```bash
python -m tests.test_offline         # optional: verifies parsing/credit logic without any API calls
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000, click **Start talking**, allow the microphone. **Use headphones** (otherwise the mic hears
Gemini and it will interrupt itself). No mic? Type commands in the box instead.

## 4. Example conversation

| You | What happens |
|---|---|
| "Get me a 3D model of a car heart." | Gemini calls `generate_3d_model(prompt="a detailed, realistic ... car engine heart ...")` and immediately says "I'm generating that now, it'll take about a minute or two." You can keep chatting. The page shows a progress bar. |
| *(~1–2 min later)* | The job finishes, the server pushes a `[SYSTEM NOTICE]` into the live session, Gemini says "Your model is ready — it's on your screen. Want me to split it into parts?" The model appears in the viewer. |
| "Separate the parts and explain them to me." | Gemini calls `separate_parts_and_explain()`, says it will take 1–3 minutes. The server segments the last task id on Tripo. |
| *(done)* | Gemini: "It split into 7 parts. The big one at the bottom is the engine block, which houses the cylinders… the four pistons…" |
| "Is it done yet?" (mid-job) | Gemini calls `check_job_status()` and reads the live progress. |
| "Separate the parts" with no model | Tool returns `no_model`; Gemini: "I don't have a model yet — want me to make one first?" |

## 5. Decisions worth knowing

* **Why "background" tool mode by default.** Live tool calls are synchronous: Gemini stays silent until you answer
  the call, and a 90-second silent wait is terrible for a voice app. So the tool answers instantly ("started, ~1–2 min"),
  the job runs in an asyncio task, and on completion we send a text `[SYSTEM NOTICE]` into the session so Gemini speaks
  the result. This works on every Live model. Google's docs list native `NON_BLOCKING` async tools for some models but
  not for Gemini 3.1 Flash Live, and I couldn't confirm it for `gemini-3.8-live`, so `TOOL_MODE=async` is there to try
  and `blocking` is there as the fallback.
* **Segmentation returns a GLB, not a parts list.** Tripo's task response gives one model URL. Part names are read from
  the node names inside that GLB (`app/glb.py`). v2 segmentation is documented as "semantic labeling", so I expect
  meaningful names, but I couldn't verify it against a real output. If the names are generic, the server flags
  `labels_are_semantic=false` and Gemini is instructed to say it is inferring each part's identity from the original prompt
  plus its position and size. **Please test one real model and check the names; that's the first thing to adjust if needed.**
* **Credit saving.** One job at a time (no accidental double-spend); segmentation results are cached per model, so asking
  twice costs nothing; a local monthly guard refuses jobs that would exceed `MONTHLY_CREDIT_BUDGET`; real spend is read from
  Tripo's `credits_consumed`. Set `TRIPO_TEXTURE=false` to drop generation from 20 to ~10 credits (untextured).
* **Errors handled**: out of credits (HTTP 402, code 2010, or error text — detection is best-effort for v3, see below),
  bad API key, rate limit (429), failed/cancelled task, no previous model, network timeouts (polling retries 3× with
  backoff; task-creation POSTs are never auto-retried so you can't be double-charged), overall job timeout (task id is
  reported; the job may still finish on Tripo's side).

## 6. Credits, latency, limitations

* **Credits (free plan = 200/month).** Textured+PBR generation ≈ 20 (Tripo's pricing table for v3.1: 10 untextured,
  20 standard texture, 30 detailed). Segmentation ≈ 40 (its docs example shows 40). One full "generate + separate" ≈ 60, so
  about **3 complete runs per month** (180 credits), plus one extra generation. Untextured: ~50 per run → 4 runs.
* **Latency.** Generation: roughly 40 s untextured, ~2 min textured (Tripo's model page). Segmentation: roughly 1–3 min.
  Gemini's voice responses themselves are real-time.
* **Not verified by me:** Gemini Live free-tier limits (check Google's pricing/rate-limit pages), the exact error code Tripo v3
  uses for "insufficient credits" (I match HTTP 402, v2's code 2010, and wording like "not enough credits"), and the exact
  node-naming of segmented GLBs. If one of these misbehaves, the fix is isolated to `_parse` in `tripo_client.py`, `build_live_config`
  in `live_session.py`, or `summarize_parts` in `glb.py`.
* **Model name.** Default is `gemini-3.8-live` (the name used in Google's current Live API docs). If your key can't access it, set
  `GEMINI_LIVE_MODEL=gemini-3.1-flash-live-preview` (synchronous tools only; keep `TOOL_MODE=background`).
* **Session length.** Live websockets are time-limited; sliding-window compression is enabled, and if Gemini sends a go-away
  notice the UI tells you to reconnect. Your last model survives reconnects and restarts (`state.json`).
* **Orientation words** (left/front/top) in part descriptions assume glTF axes (+Y up) and are approximate.
* **Security.** The server holds both API keys and is meant to run locally. Don't expose it publicly without adding auth.
* Segmentation input is the generation `task_id`. Tripo also accepts a model URL; if a very old task id is ever rejected,
  switch `create_mesh_segmentation` to pass `last.model_url` instead.
