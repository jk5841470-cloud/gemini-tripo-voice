"""Offline smoke test (no API keys, no network). Run:  python -m tests.test_offline

Checks the parts of the system that don't need Gemini or Tripo:
  1. GLB scene-graph parsing -> grouped, located parts
  2. Credit guard + persisted "last model" state
  3. generate -> segment tool flow against a fake Tripo client (incl. error paths)
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from app import tools as tools_mod
from app.config import load_settings
from app.glb import summarize_parts
from app.state import AppState
from app.tripo_client import TripoInsufficientCreditsError


def _node(name, mesh, tx=0, ty=0, tz=0):
    return {"name": name, "mesh": mesh, "translation": [tx, ty, tz]}


def make_gltf(names_positions):
    """Each part is a 1x1x1 cube mesh placed at the given translation."""
    accessors, meshes, nodes = [], [], []
    for i, (name, pos) in enumerate(names_positions):
        accessors.append({"min": [-0.5, -0.5, -0.5], "max": [0.5, 0.5, 0.5]})
        meshes.append({"primitives": [{"attributes": {"POSITION": i}}]})
        nodes.append(_node(name, i, *pos))
    return {"scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes, "accessors": accessors}


def test_glb_semantic():
    gltf = make_gltf([("wheel.001", (-2, 0, 2)), ("wheel.002", (2, 0, 2)), ("wheel.003", (-2, 0, -2)),
                      ("wheel.004", (2, 0, -2)), ("body", (0, 1.5, 0))])
    out = summarize_parts(gltf)
    names = {p["name"]: p for p in out["parts"]}
    assert out["labels_are_semantic"] is True
    assert names["wheel"]["count"] == 4, names
    assert "top" in names["body"]["location"], names["body"]
    print("glb semantic OK:", [(p["name"], p["count"], p["location"]) for p in out["parts"]])


def test_glb_generic():
    gltf = make_gltf([("tripo_node_ab12cd34", (0, 0, 0)), ("Mesh_2", (3, 0, 0)), ("node_3", (0, 3, 0))])
    out = summarize_parts(gltf)
    assert out["labels_are_semantic"] is False
    assert [p["name"] for p in out["parts"]][0].startswith("Part ")
    print("glb generic OK:", [p["name"] for p in out["parts"]])


def test_state_and_budget():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "state.json"
        st = AppState.load(path, monthly_budget=60)
        assert st.can_afford(60) and not st.can_afford(61)
        st.add_spend(20)
        assert st.remaining == 40
        st2 = AppState.load(path, monthly_budget=60)  # survives restart
        assert st2.spent == 20
        print("state/budget OK")


class FakeTripo:
    def __init__(self, fail_credits=False):
        self.fail_credits = fail_credits
        self.seg_calls = 0

    async def create_text_to_model(self, prompt, **kw):
        if self.fail_credits:
            raise TripoInsufficientCreditsError("Tripo says this account is out of credits, so I can't start that job.")
        return "task_gen_1"

    async def create_mesh_segmentation(self, input_task_id, **kw):
        assert input_task_id == "task_gen_1"
        self.seg_calls += 1
        return "task_seg_1"

    async def wait_for_task(self, task_id, *, timeout, poll_interval, on_progress=None):
        if on_progress:
            await on_progress("running", 50)
        url = "https://cdn.example/model.glb" if task_id == "task_gen_1" else "https://cdn.example/seg.glb"
        return {"status": "success", "output": {"model_url": url}, "credits_consumed": 20 if task_id == "task_gen_1" else 40}


async def test_tool_flow():
    cfg = load_settings()
    with tempfile.TemporaryDirectory() as d:
        st = AppState.load(Path(d) / "s.json", monthly_budget=200)
        events = []

        async def on_event(e):
            events.append(e)

        # segment before generate -> friendly error
        svc = tools_mod.ToolService(FakeTripo(), st, cfg)
        r = await svc.separate_parts_and_explain(on_event)
        assert r["error_type"] == "no_model", r

        # generate
        r = await svc.generate_3d_model("a wooden treasure chest", on_event)
        assert r["status"] == "success" and st.last_model.task_id == "task_gen_1", r

        # segment (GLB fetch is patched to avoid the network)
        async def fake_extract(url):
            return summarize_parts(make_gltf([("lid", (0, 1, 0)), ("base", (0, -1, 0)), ("lock", (0, 0, 1))]))

        tools_mod.extract_parts = fake_extract
        r = await svc.separate_parts_and_explain(on_event)
        assert r["status"] == "success" and r["part_count"] == 3, r
        assert st.spent == 60

        # second request is served from cache: no extra Tripo call, no extra credits
        r2 = await svc.separate_parts_and_explain(on_event)
        assert r2["from_cache"] is True and svc.tripo.seg_calls == 1 and st.spent == 60
        print("tool flow OK (spent", st.spent, "credits; cache hit works)")

        # insufficient credits comes back as a speakable error, not an exception
        svc2 = tools_mod.ToolService(FakeTripo(fail_credits=True), AppState.load(Path(d) / "s2.json", 200), cfg)
        r = await svc2.generate_3d_model("a robot")
        assert r["status"] == "error" and r["error_type"] == "insufficient_credits", r

        # local budget guard stops the job before calling Tripo
        st3 = AppState.load(Path(d) / "s3.json", monthly_budget=30)
        st3.add_spend(20)
        r = await tools_mod.ToolService(FakeTripo(), st3, cfg).generate_3d_model("a robot")
        assert r["error_type"] == "budget", r
        print("error paths OK")


if __name__ == "__main__":
    test_glb_semantic()
    test_glb_generic()
    test_state_and_budget()
    asyncio.run(test_tool_flow())
    print("\nAll offline checks passed.")
