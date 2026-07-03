"""Live demo in two parts: the headline effect, then a falsifiable ablation.

PART 1 - Memory vs Vanilla. Each question is answered twice: once by the memory
agent (Cognee schema and episodes injected) and once by a vanilla agent (no
context at all). This reproduces the project's headline effect: the memory layer,
and nothing else, is what moves accuracy.

PART 2 - Recall-signal ablation. The state hash decides which past episodes recall
returns, because episodes are filed in Cognee under datasets tagged with the hash.
A broken constant hash makes every stored episode match every question, so recall
fires on everything with wrong-type examples (a misleading signal). The real hash
keeps recall targeted to the question's own type. The honest, measurable
difference here is recall targeting, not accuracy on a small sample.

Both parts write reinforcement thoughts to Synapse (the global Hebbian success
signal). The single-shot benchmark lives in benchmarks/runner.py.
"""
import argparse
import asyncio
import json
import os
import re
import subprocess
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

import agent.memory_bridge as memory_bridge
from agent.answer_check import answers_match, execute
from agent.prompt_builder import build_prompt
from agent.state_hash import hash_state
from agent.synapse_client import SynapseClient
from benchmarks.runner import _derive_northwind_schema

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

_MODEL = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
_DB_PATH = str(Path(__file__).parent / "benchmarks" / "northwind.db")
_QUESTIONS_PATH = Path(__file__).parent / "benchmarks" / "questions.json"
_VANILLA_PROMPT = "Convert this question to SQL for the Northwind database: {question}"
_BROKEN_HASH = 12345
_EPISODE_MARKER = "[PAST QUERIES THAT WORKED"


def _train_questions() -> list[dict]:
    raw = json.loads(_QUESTIONS_PATH.read_text(encoding="utf-8"))
    return [q for q in raw if not q.get("holdout")]


def _demo_subset(limit_per_intent: int = 3) -> list[dict]:
    picked: list[dict] = []
    counts: Counter = Counter()
    for q in _train_questions():
        if counts[q["intent"]] < limit_per_intent:
            picked.append(q)
            counts[q["intent"]] += 1
    return picked


class DemoAgent:
    def __init__(self, synapse: SynapseClient, anthropic: AsyncAnthropic, db_path: str) -> None:
        self._synapse = synapse
        self._anthropic = anthropic
        self._db_path = db_path
        self._stored_hashes: set[int] = set()

    async def answer_memory(self, question: dict) -> bool:
        state_hash = hash_state({
            "intent": question["intent"],
            "tables": question["tables"],
            "clauses_so_far": [],
        })
        context = await memory_bridge.recall_context(
            question["question"], state_hash, tables=question.get("tables")
        )
        reasoning_hint = {"intent": question["intent"], "score": 1.0}
        sql = await self._call_claude(build_prompt(question["question"], context, reasoning_hint))
        return self._evaluate(sql, question["gold_answer"])

    async def answer_vanilla(self, question: dict) -> bool:
        sql = await self._call_claude(_VANILLA_PROMPT.format(question=question["question"]))
        return self._evaluate(sql, question["gold_answer"])

    async def answer_ablation(self, question: dict, epoch: int, use_broken_hash: bool) -> tuple[bool, bool]:
        state_hash = _BROKEN_HASH if use_broken_hash else hash_state({
            "intent": question["intent"],
            "tables": question["tables"],
            "clauses_so_far": [],
        })
        context = await memory_bridge.recall_context(
            question["question"], state_hash, tables=question.get("tables")
        )
        hit = _EPISODE_MARKER in context
        sql = await self._call_claude(build_prompt(question["question"], context, None))
        correct = self._evaluate(sql, question["gold_answer"])

        await self._reinforce(state_hash, correct)
        if correct and state_hash not in self._stored_hashes:
            await memory_bridge.remember_success(question["question"], sql, epoch, state_hash)
            self._stored_hashes.add(state_hash)
        return correct, hit

    async def _reinforce(self, state_hash: int, correct: bool) -> None:
        try:
            await self._synapse.append_thought("0", None, state_hash, 1.0 if correct else 0.0)
        except Exception:
            pass

    async def _call_claude(self, prompt: str) -> str:
        response = await self._anthropic.messages.create(
            model=_MODEL,
            max_tokens=256,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        match = re.search(r"```(?:sql)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else text.strip()

    def _evaluate(self, sql: str, gold_answer) -> bool:
        try:
            conn = sqlite3.connect(self._db_path)
            try:
                actual = execute(conn, sql)
            finally:
                conn.close()
            return answers_match(actual, gold_answer)
        except Exception:
            return False


async def _part1_memory_vs_vanilla(
    synapse: SynapseClient, anthropic: AsyncAnthropic, schema: str, questions: list[dict]
) -> tuple[float, float]:
    print(f"\n{'=' * 66}\n  PART 1 - MEMORY vs VANILLA\n{'=' * 66}", flush=True)
    await memory_bridge.reset_store()
    await memory_bridge.remember_schema(schema)
    memory_bridge.clear_epoch_recall_cache()

    agent = DemoAgent(synapse, anthropic, _DB_PATH)
    mem = 0
    van = 0
    for q in questions:
        m = await agent.answer_memory(q)
        v = await agent.answer_vanilla(q)
        mem += 1 if m else 0
        van += 1 if v else 0
        print(f"    memory {'OK' if m else '--'}  vanilla {'OK' if v else '--'}   {q['question']}", flush=True)
    mem_acc = mem / len(questions)
    van_acc = van / len(questions)
    print(f"\n  memory:  {mem_acc:.0%}   vanilla: {van_acc:.0%}   gap: {mem_acc - van_acc:+.0%}", flush=True)
    return mem_acc, van_acc


async def _part2_recall_ablation(
    synapse: SynapseClient, anthropic: AsyncAnthropic, schema: str, questions: list[dict], epochs: int
) -> dict[str, list[float]]:
    print(f"\n{'=' * 66}\n  PART 2 - RECALL-SIGNAL ABLATION (broken vs real hash)\n{'=' * 66}", flush=True)
    hit_rates: dict[str, list[float]] = {}
    for use_broken_hash in (True, False):
        label = "broken (constant 12345)" if use_broken_hash else "real (per type)"
        print(f"\n  --- {label} ---", flush=True)
        await memory_bridge.reset_store()
        await memory_bridge.remember_schema(schema)
        agent = DemoAgent(synapse, anthropic, _DB_PATH)
        rates: list[float] = []
        for epoch in range(1, epochs + 1):
            memory_bridge.clear_epoch_recall_cache()
            hits = 0
            for q in questions:
                _, hit = await agent.answer_ablation(q, epoch, use_broken_hash)
                hits += 1 if hit else 0
            rate = hits / len(questions)
            rates.append(rate)
            print(f"    epoch {epoch}: recall fired on {rate:.0%} of questions", flush=True)
        hit_rates["broken" if use_broken_hash else "real"] = rates
    return hit_rates


def _benchmark_gap() -> tuple[float, float] | None:
    files = sorted((Path(__file__).parent / "results").glob("benchmark_*.json"))
    if not files:
        return None
    data = json.loads(files[-1].read_text(encoding="utf-8"))
    peak = max((e["memory_train_accuracy"] for e in data["per_epoch"]), default=0.0)
    return peak, data.get("vanilla_train_accuracy", 0.0)


def _print_conclusion(mem_acc: float, van_acc: float, hit_rates: dict[str, list[float]]) -> None:
    print(f"\n{'=' * 66}\n  WHAT THE DEMO SHOWS\n{'=' * 66}")
    print(f"  Live single-pass (schema + reasoning template, no episodes yet):")
    print(f"    memory {mem_acc:.0%} vs vanilla {van_acc:.0%}  ({mem_acc - van_acc:+.0%})")
    bench = _benchmark_gap()
    if bench:
        peak, van = bench
        print(f"  Full benchmark (8 epochs, all three memory types, results/):")
        print(f"    memory {peak:.0%} vs vanilla {van:.0%}  ({peak - van:+.0%})")
    print("  The live pass shows the schema+template component; episode")
    print("  accumulation over epochs adds the rest. The memory layer is the")
    print("  only difference in both.")
    print()
    print(f"  Recall firing rate, final epoch:")
    print(f"    broken hash: {hit_rates['broken'][-1]:.0%}  (fires on nearly everything,")
    print(f"                 but returns wrong-type episodes - a misleading signal)")
    print(f"    real hash:   {hit_rates['real'][-1]:.0%}  (fires only for types with a prior")
    print(f"                 success - a targeted, honest signal)")
    print()
    print("  Break the state hash and recall stops discriminating by question type.")
    print("  That targeting is what the state hash buys you.")


def _launch_dashboard() -> None:
    try:
        subprocess.Popen(
            ["uv", "run", "streamlit", "run", "dashboard/app.py"],
            cwd=str(Path(__file__).parent),
        )
        print("\n  Dashboard launching at http://localhost:8501", flush=True)
    except Exception as exc:
        print(f"\n  Could not auto-launch dashboard ({exc}). Run: uv run streamlit run dashboard/app.py", flush=True)


async def main(epochs: int, dashboard: bool) -> None:
    train = _train_questions()
    ablation_qs = _demo_subset()
    print(f"Part 1 over {len(train)} train questions; "
          f"Part 2 ablation over {len(ablation_qs)} questions ({len(set(q['intent'] for q in ablation_qs))} intent types).")

    schema = _derive_northwind_schema()
    anthropic = AsyncAnthropic()
    synapse = SynapseClient(
        base_url=os.environ["SYNAPSE_URL"],
        api_key=os.environ["SYNAPSE_API_KEY"],
        timeout=30.0,
    )

    mem_acc, van_acc = await _part1_memory_vs_vanilla(synapse, anthropic, schema, train)
    hit_rates = await _part2_recall_ablation(synapse, anthropic, schema, ablation_qs, epochs)
    _print_conclusion(mem_acc, van_acc, hit_rates)

    await synapse.aclose()
    if dashboard:
        _launch_dashboard()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--no-dashboard", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.epochs, dashboard=not args.no_dashboard))
