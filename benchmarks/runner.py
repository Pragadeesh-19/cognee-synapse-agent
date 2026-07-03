"""Epoch runner: memory agent vs vanilla baseline over 50 train + 10 holdout questions."""
import argparse
import asyncio
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

import agent.memory_bridge as memory_bridge
from agent.sql_agent import SQLAgent
from agent.state_hash import hash_state
from agent.synapse_client import SynapseClient

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

_DB_PATH = str(Path(__file__).parent / "northwind.db")
_QUESTIONS_PATH = Path(__file__).parent / "questions.json"
_RESULTS_DIR = Path(__file__).parent.parent / "results"

# single-shot by design: multi-step exceeds API budget for 500 questions
# haiku-4-5: ~$0.0006/call at ~600 tok in / 50 tok out vs sonnet at $0.005/call
_GBP_PER_MEMORY_CALL = 0.0005
_GBP_PER_COGNIFY_CALL = 0.002
_GBP_PER_IMPROVE_CALL = 0.003
_BUDGET_CEILING_GBP = 4.50


def _derive_northwind_schema() -> str:
    conn = sqlite3.connect(_DB_PATH)
    try:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        lines: list[str] = []
        for table in tables:
            cols = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
            col_defs = ", ".join(
                f"{c[1]} (PK)" if c[5] else c[1] for c in cols
            )
            lines.append(f"Table {table}: {col_defs}")
            fks = conn.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()
            for fk in fks:
                lines.append(f"  FK: {table}.{fk[3]} -> {fk[2]}.{fk[4]}")
        lines.append("")
        lines.append("SQLite dialect: use strftime('%Y', column) not YEAR(column)")
        return "\n".join(lines)
    finally:
        conn.close()


def _load_questions() -> tuple[list[dict], list[dict]]:
    raw = json.loads(_QUESTIONS_PATH.read_text(encoding="utf-8"))
    train = [q for q in raw if not q.get("holdout")]
    holdout = [q for q in raw if q.get("holdout")]
    return train, holdout


class CostTracker:
    def __init__(self) -> None:
        self.sql_calls = 0
        self.cognify_calls = 0
        self.improve_calls = 0

    def add_sql(self, n: int = 1) -> None:
        self.sql_calls += n

    def add_cognify(self, n: int = 1) -> None:
        self.cognify_calls += n

    def add_improve(self, n: int = 1) -> None:
        self.improve_calls += n

    @property
    def estimated_gbp(self) -> float:
        return (
            self.sql_calls * _GBP_PER_MEMORY_CALL
            + self.cognify_calls * _GBP_PER_COGNIFY_CALL
            + self.improve_calls * _GBP_PER_IMPROVE_CALL
        )

    def check_ceiling(self) -> None:
        est = self.estimated_gbp
        if est > _BUDGET_CEILING_GBP:
            raise RuntimeError(
                f"BUDGET CEILING: estimated {est:.3f} GBP exceeds {_BUDGET_CEILING_GBP} GBP limit. "
                f"sql={self.sql_calls}, cognify={self.cognify_calls}, improve={self.improve_calls}"
            )

    def summary(self) -> str:
        return (
            f"sql={self.sql_calls} cognify={self.cognify_calls} "
            f"improve={self.improve_calls} est={self.estimated_gbp:.3f} GBP"
        )


async def _run_vanilla_pass(
    agent: SQLAgent, questions: list[dict], cost: CostTracker
) -> list[bool]:
    results = []
    for i, q in enumerate(questions, 1):
        result = await agent.answer_question(q, epoch=0, use_memory=False)
        results.append(result["correct"])
        cost.add_sql()
        print(f"  vanilla Q{i:02d}/50: {'OK' if result['correct'] else '--'}  running={sum(results)}/{i}", flush=True)
    return results


async def _run_train_epoch(
    agent: SQLAgent,
    questions: list[dict],
    epoch: int,
    failed_patterns: dict[int, int],
    cost: CostTracker,
) -> tuple[list[bool], set[int]]:
    memory_bridge.clear_epoch_recall_cache()
    epoch_seen: set[int] = set()
    corrects = []
    for i, q in enumerate(questions, 1):
        result = await agent.answer_question(
            q, epoch=epoch, use_memory=True, reinforce=True, stored_hashes=epoch_seen
        )
        cost.add_sql()
        cost.check_ceiling()

        corrects.append(result["correct"])
        if result["remembered"]:
            # one cognify call per unique (hash, epoch) via remember_success
            cost.add_cognify()
        if not result["correct"]:
            h = hash_state({"intent": q["intent"], "tables": q["tables"], "clauses_so_far": []})
            failed_patterns[h] = failed_patterns.get(h, 0) + 1

        print(
            f"  epoch {epoch} Q{i:02d}/50: {'OK' if result['correct'] else '--'}"
            f"  acc={sum(corrects)/i:.0%}  {cost.summary()}",
            flush=True,
        )

    return corrects, epoch_seen


async def _run_holdout_epoch(
    agent: SQLAgent, questions: list[dict], epoch: int, cost: CostTracker
) -> list[bool]:
    corrects = []
    for q in questions:
        result = await agent.answer_question(
            q, epoch=epoch, use_memory=True, reinforce=False
        )
        cost.add_sql()
        corrects.append(result["correct"])
    return corrects


async def run(epochs: int = 10) -> None:
    train_qs, holdout_qs = _load_questions()
    print(f"Loaded {len(train_qs)} train, {len(holdout_qs)} holdout questions", flush=True)

    anthropic = AsyncAnthropic()
    # 30s timeout: bootstrap endpoint can be slow after idle periods
    synapse = SynapseClient(
        base_url=os.environ["SYNAPSE_URL"],
        api_key=os.environ["SYNAPSE_API_KEY"],
        timeout=30.0,
    )
    agent = SQLAgent(synapse, memory_bridge, anthropic, _DB_PATH)

    cost = CostTracker()
    failed_patterns: dict[int, int] = {}
    per_epoch: list[dict] = []
    forget_events: list[dict] = []

    print("Connecting to Synapse...", flush=True)
    try:
        stats = await synapse.get_stats("0")
        await synapse.get_best_next("0", 0)
        print(f"Synapse ready: writeHead={stats.get('writeHead', 0)}", flush=True)
    except Exception as exc:
        print(f"Synapse unreachable at startup ({exc}); proceeding without procedural memory", flush=True)

    print("\nResetting Cognee stores...", flush=True)
    await memory_bridge.reset_store()
    print("Loading Northwind schema into semantic memory...", flush=True)
    await memory_bridge.remember_schema(_derive_northwind_schema())
    print("Schema loaded.", flush=True)

    print("\n=== VANILLA PASS (once, temperature=0) ===", flush=True)
    vanilla_results = await _run_vanilla_pass(agent, train_qs, cost)
    vanilla_acc = sum(vanilla_results) / len(vanilla_results)
    print(f"Vanilla accuracy: {vanilla_acc:.1%}  |  {cost.summary()}", flush=True)

    print("\n=== MEMORY AGENT EPOCHS ===", flush=True)
    for epoch in range(1, epochs + 1):
        t0 = time.monotonic()
        print(f"\n--- Epoch {epoch}/{epochs} ---", flush=True)

        train_corrects, hashes_stored = await _run_train_epoch(
            agent, train_qs, epoch, failed_patterns, cost
        )
        train_acc = sum(train_corrects) / len(train_corrects)

        # epoch-5 checkpoint: forget episodic buckets where Synapse signals persistent failure
        epoch_forget_events: list[str] = []
        if epoch == 5:
            failing = [h for h, c in failed_patterns.items() if c > 3]
            if failing:
                forgotten = await memory_bridge.forget_failures(failing, list(range(1, 6)))
                epoch_forget_events = forgotten
                for ds in forgotten:
                    forget_events.append({
                        "epoch": epoch,
                        "dataset": ds,
                        "reason": "failure_count>3",
                        "ts": datetime.now(timezone.utc).isoformat(),
                    })
                print(f"  Epoch 5 forget: {len(forgotten)} datasets removed", flush=True)

        await memory_bridge.consolidate()
        cost.add_improve()
        cost.check_ceiling()

        holdout_corrects = await _run_holdout_epoch(agent, holdout_qs, epoch, cost)
        holdout_acc = sum(holdout_corrects) / len(holdout_corrects)

        elapsed = time.monotonic() - t0
        print(
            f"\nEPOCH {epoch} RESULT: memory_train={train_acc:.1%}  vanilla={vanilla_acc:.1%}"
            f"  holdout={holdout_acc:.1%}  [{elapsed:.0f}s]  {cost.summary()}",
            flush=True,
        )

        per_epoch.append({
            "epoch": epoch,
            "memory_train_accuracy": train_acc,
            "memory_train_correct": sum(train_corrects),
            "memory_train_total": len(train_corrects),
            "memory_holdout_accuracy": holdout_acc,
            "memory_holdout_correct": sum(holdout_corrects),
            "memory_holdout_total": len(holdout_corrects),
            "hashes_stored_this_epoch": len(hashes_stored),
            "forget_events": epoch_forget_events,
        })

    _RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = _RESULTS_DIR / f"benchmark_{timestamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "config": {
                    "epochs": epochs,
                    "train_questions": len(train_qs),
                    "holdout_questions": len(holdout_qs),
                    "budget_ceiling_gbp": _BUDGET_CEILING_GBP,
                },
                "vanilla_train_accuracy": vanilla_acc,
                "vanilla_train_correct": sum(vanilla_results),
                "per_epoch": per_epoch,
                "forget_events": forget_events,
                "cost": {
                    "sql_calls": cost.sql_calls,
                    "cognify_calls": cost.cognify_calls,
                    "improve_calls": cost.improve_calls,
                    "estimated_gbp": cost.estimated_gbp,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n=== RESULTS SAVED: {out_path} ===", flush=True)
    print(f"Vanilla baseline:         {vanilla_acc:.1%}", flush=True)
    print(f"Memory epoch 1 train:     {per_epoch[0]['memory_train_accuracy']:.1%}", flush=True)
    print(f"Memory epoch {epochs} train:   {per_epoch[-1]['memory_train_accuracy']:.1%}", flush=True)
    print(f"Memory epoch {epochs} holdout: {per_epoch[-1]['memory_holdout_accuracy']:.1%}", flush=True)
    print(f"Total cost:               {cost.summary()}", flush=True)

    gain = per_epoch[-1]["memory_train_accuracy"] - per_epoch[0]["memory_train_accuracy"]
    holdout_gain = (
        per_epoch[-1]["memory_holdout_accuracy"] - per_epoch[0]["memory_holdout_accuracy"]
    )
    print("\n=== EXIT GATE ===", flush=True)
    print(f"  Train gain (epoch 1->{epochs}): {gain:+.1%}", flush=True)
    print(f"  Holdout gain (epoch 1->{epochs}): {holdout_gain:+.1%}", flush=True)
    print(f"  Under budget: {'YES' if cost.estimated_gbp < _BUDGET_CEILING_GBP else 'NO'}", flush=True)

    await synapse.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(run(args.epochs))
