"""Streamlit dashboard: four panels showing the three-memory agent benchmark."""
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.synapse_client import SynapseClient

load_dotenv(_ROOT / ".env")

_RESULTS_DIR = _ROOT / "results"
_CURVE_PNG = _RESULTS_DIR / "learning_curve.png"
_LIVE_LOG = _RESULTS_DIR / "benchmark_live.log"
_SYNAPSE_URL = os.getenv("SYNAPSE_URL", "http://localhost:8080")
_SYNAPSE_API_KEY = os.getenv("SYNAPSE_API_KEY", "")
_LLM_MODEL = os.getenv("LLM_MODEL", "unknown")


def _latest_benchmark() -> Path | None:
    files = sorted(_RESULTS_DIR.glob("benchmark_*.json"))
    return files[-1] if files else None


@st.cache_data
def _load_benchmark() -> dict | None:
    src = _latest_benchmark()
    if src is None:
        return None
    return json.loads(src.read_text(encoding="utf-8"))


def _run_synapse(coro_factory) -> dict | None:
    async def _go():
        client = SynapseClient(_SYNAPSE_URL, _SYNAPSE_API_KEY)
        try:
            return await coro_factory(client)
        finally:
            await client.aclose()

    try:
        return asyncio.run(_go())
    except Exception:
        return None


def _derive_activity_events(data: dict) -> list[dict]:
    events: list[dict] = []
    if _LIVE_LOG.exists():
        for line in _LIVE_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
            if "schema into semantic memory" in line:
                events.append({"kind": "remember", "dataset": "northwind_schema", "detail": "semantic schema loaded"})
                break

    for entry in data.get("per_epoch", []):
        epoch = entry["epoch"]
        stored = entry.get("hashes_stored_this_epoch", 0)
        events.append({
            "kind": "remember",
            "dataset": f"episode_epoch_{epoch}_hash_*",
            "detail": f"epoch {epoch}: {stored} episode datasets stored",
        })
        for forget_dataset in entry.get("forget_events", []):
            events.append({"kind": "forget", "dataset": forget_dataset, "detail": f"epoch {epoch} forget"})

    forget_line = _epoch5_forget_from_log()
    if forget_line:
        events.append(forget_line)

    return events


def _epoch5_forget_from_log() -> dict | None:
    if not _LIVE_LOG.exists():
        return None
    for line in _LIVE_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        if "forget:" in line and "datasets removed" in line:
            return {"kind": "forget", "dataset": "episode_epoch_1..4_hash_*", "detail": line.strip()}
    return None


def _render_learning_curve() -> None:
    st.subheader("Learning curve — Three-Memory-Type Agent vs Vanilla Baseline")
    if _CURVE_PNG.exists():
        st.image(str(_CURVE_PNG), use_container_width=True)
    else:
        st.warning("results/learning_curve.png not found. Run benchmarks/plot.py first.")
    if st.button("Regenerate from latest benchmark JSON"):
        result = subprocess.run(
            ["uv", "run", "python", "benchmarks/plot.py"],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            st.cache_data.clear()
            st.success("Regenerated learning_curve.png")
            st.rerun()
        else:
            st.error(result.stderr or "plot.py failed")


def _render_synapse_status() -> None:
    st.subheader("Synapse memory")
    stats = _run_synapse(lambda c: c.get_stats("0"))
    if stats is None:
        st.error("Synapse offline — could not reach " + _SYNAPSE_URL)
        return

    st.metric("Fill", f"{stats.get('fillPercent', 0.0):.2f}%")
    st.metric("Write head", stats.get("writeHead", 0))
    st.metric("Used slots", f"{stats.get('usedSlots', 0)} / {stats.get('capacity', 0)}")

    st.markdown("**Recall probe**")
    probe_hash = st.number_input("stateHash", min_value=0, value=99999, step=1)
    if st.button("Check best-next"):
        hit = _run_synapse(lambda c: c.get_best_next("0", int(probe_hash)))
        if hit:
            st.markdown(f":green[● HIT] — thought {hit.get('thoughtId', '?')}")
        else:
            st.markdown(":red[● MISS] — no reinforced path for this state")


def _render_activity_log(data: dict) -> None:
    st.subheader("Cognee activity")
    events = _derive_activity_events(data)
    if not events:
        st.info("No memory events recorded.")
        return
    for event in events[-10:]:
        icon = "" if event["kind"] == "remember" else ""
        st.markdown(f"{icon} **{event['kind']}** · `{event['dataset']}`  \n{event['detail']}")


def _render_result_box(data: dict) -> None:
    st.subheader("Result summary")
    per_epoch = data.get("per_epoch", [])
    peak = max((e["memory_train_accuracy"] for e in per_epoch), default=0.0) * 100
    vanilla = data.get("vanilla_train_accuracy", 0.0) * 100
    cost = data.get("cost", {}).get("estimated_gbp", 0.0)

    st.metric("Memory agent peak", f"{peak:.0f}%")
    st.metric("Vanilla baseline", f"{vanilla:.0f}%")
    st.metric("Gap", f"+{peak - vanilla:.0f} points")
    st.metric("Epochs run", len(per_epoch))
    st.metric("Total cost", f"£{cost:.3f}")
    st.caption(f"Model: {_LLM_MODEL}")


def main() -> None:
    st.set_page_config(page_title="Three-Memory-Type SQL Agent", layout="wide")
    st.title("Three-Memory-Type SQL Agent")
    st.caption("Semantic + Episodic (Cognee) · Procedural (Synapse) · Northwind text-to-SQL")

    data = _load_benchmark()
    if data is None:
        st.error("No benchmark_*.json found in results/. Run the benchmark first.")
        return

    _render_learning_curve()
    st.divider()

    col_synapse, col_activity, col_result = st.columns(3)
    with col_synapse:
        _render_synapse_status()
    with col_activity:
        _render_activity_log(data)
    with col_result:
        _render_result_box(data)


main()
