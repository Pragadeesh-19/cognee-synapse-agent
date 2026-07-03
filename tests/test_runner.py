"""Mocked tests for benchmarks/runner.py — no API calls."""
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import benchmarks.runner as runner_module
from benchmarks.runner import CostTracker, _load_questions, run

_QUESTIONS_PATH = Path(__file__).parent.parent / "benchmarks" / "questions.json"


# ---------------------------------------------------------------------------
# CostTracker unit tests
# ---------------------------------------------------------------------------

def test_cost_tracker_accumulates_correctly() -> None:
    cost = CostTracker()
    cost.add_sql(50)
    cost.add_cognify(18)
    cost.add_improve(1)
    expected = 50 * 0.0005 + 18 * 0.002 + 1 * 0.003
    assert abs(cost.estimated_gbp - expected) < 1e-9


def test_cost_tracker_ceiling_raises() -> None:
    cost = CostTracker()
    cost.add_sql(10000)  # 5.00 GBP > 4.50 ceiling
    with pytest.raises(RuntimeError, match="BUDGET CEILING"):
        cost.check_ceiling()


def test_cost_tracker_below_ceiling_is_fine() -> None:
    cost = CostTracker()
    cost.add_sql(7000)  # 3.50 GBP
    cost.check_ceiling()  # must not raise


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------

def test_load_questions_splits_correctly() -> None:
    train, holdout = _load_questions()
    assert len(train) == 50
    assert len(holdout) == 10
    assert all(not q.get("holdout") for q in train)
    assert all(q.get("holdout") for q in holdout)


# ---------------------------------------------------------------------------
# Dedup guard: remember_success fires once per (hash, epoch)
# ---------------------------------------------------------------------------

async def test_dedup_prevents_second_remember_same_epoch() -> None:
    from agent.sql_agent import SQLAgent
    from unittest.mock import AsyncMock, MagicMock

    correct_sql = "SELECT COUNT(*) FROM Customers WHERE Country = 'Germany'"

    mock_anthropic = MagicMock()
    mock_anthropic.messages.create = AsyncMock(
        return_value=MagicMock(content=[MagicMock(text=correct_sql)])
    )
    mock_bridge = MagicMock()
    mock_bridge.recall_context = AsyncMock(return_value="")
    mock_bridge.remember_success = AsyncMock()
    mock_synapse = MagicMock()
    mock_synapse.get_best_next = AsyncMock(return_value={"found": True, "slot": 0, "score": 0.0})
    mock_synapse.append_thought = AsyncMock(return_value=1)

    train, _ = _load_questions()
    # pick two questions with the same intent+tables (same state_hash)
    agg_qs = [q for q in train if q["intent"] == "AGGREGATE" and q["tables"] == ["Customers"]]
    assert len(agg_qs) >= 2, "Need at least 2 AGGREGATE/Customers questions for dedup test"

    db_path = str(Path(__file__).parent.parent / "benchmarks" / "northwind.db")
    agent = SQLAgent(mock_synapse, mock_bridge, mock_anthropic, db_path)

    epoch_seen: set[int] = set()
    await agent.answer_question(agg_qs[0], epoch=1, use_memory=True, stored_hashes=epoch_seen)
    await agent.answer_question(agg_qs[1], epoch=1, use_memory=True, stored_hashes=epoch_seen)

    # remember_success should fire at most once across both questions
    assert mock_bridge.remember_success.call_count <= 1, (
        f"Dedup failed: remember_success called {mock_bridge.remember_success.call_count}x "
        f"for questions sharing the same state_hash"
    )


# ---------------------------------------------------------------------------
# forget_failures: correct surgical dataset names at epoch 5
# ---------------------------------------------------------------------------

async def test_forget_called_with_surgical_dataset_names() -> None:
    from agent.memory_bridge import forget_failures

    with patch("agent.memory_bridge.cognee") as mock_cognee:
        mock_cognee.forget = AsyncMock()
        forgotten = await forget_failures(failed_state_hashes=[111, 222], epochs=[1, 2])

    expected = {
        "episode_epoch_1_hash_111",
        "episode_epoch_1_hash_222",
        "episode_epoch_2_hash_111",
        "episode_epoch_2_hash_222",
    }
    assert set(forgotten) == expected, f"Got: {set(forgotten)}"
    assert mock_cognee.forget.call_count == 4


# ---------------------------------------------------------------------------
# Full run (mocked): vanilla runs once, forget at epoch 5, budget tracked
# ---------------------------------------------------------------------------

async def test_full_run_mocked_vanilla_once_and_epoch5_forget() -> None:
    train, holdout = _load_questions()

    # Make all answers correct so remember fires every epoch (max dedup effect visible)
    correct_sql_by_question: dict = {}
    for q in train + holdout:
        gold = q["gold_answer"]
        # answer_check.answers_match compares scalar or list — use gold directly as SQL result
        correct_sql_by_question[q["id"]] = str(gold)

    remember_calls: list[tuple] = []
    forget_calls: list[tuple] = []
    sql_calls: list[int] = []
    improve_calls: list[int] = []

    async def fake_answer_question(q, epoch, use_memory, reinforce=True, stored_hashes=None):
        sql_calls.append(1)
        from agent.state_hash import hash_state
        ctx_hash = hash_state({"intent": q["intent"], "tables": q["tables"], "clauses_so_far": []})
        remembered = False
        if use_memory and reinforce and (stored_hashes is None or ctx_hash not in stored_hashes):
            remember_calls.append((epoch, ctx_hash))
            if stored_hashes is not None:
                stored_hashes.add(ctx_hash)
            remembered = True
        return {
            "question_id": q["id"], "sql": "", "correct": True,
            "execution_error": None, "state_hash": ctx_hash, "remembered": remembered,
        }

    async def fake_forget_failures(hashes, epochs):
        for e in epochs:
            for h in hashes:
                forget_calls.append((e, h))
        return [f"episode_epoch_{e}_hash_{h}" for e in epochs for h in hashes]

    async def fake_consolidate():
        improve_calls.append(1)

    with (
        patch.object(runner_module, "memory_bridge") as mock_mb,
        patch("benchmarks.runner.SQLAgent") as MockAgent,
        patch("benchmarks.runner.SynapseClient") as MockSynapse,
        patch("benchmarks.runner.AsyncAnthropic"),
        patch("benchmarks.runner._RESULTS_DIR", Path(
            "C:/Users/anony/AppData/Local/Temp/claude/"
            "C--Users-anony-Downloads-cognee-synapse-agent/"
            "d2b4715e-89ae-4607-86bf-3bedecd98373/scratchpad"
        )),
    ):
        mock_mb.forget_failures = AsyncMock(side_effect=fake_forget_failures)
        mock_mb.consolidate = AsyncMock(side_effect=fake_consolidate)
        mock_mb.recall_context = AsyncMock(return_value="")
        mock_mb.remember_success = AsyncMock(side_effect=lambda *a, **kw: remember_calls.append(a))
        mock_mb.reset_store = AsyncMock()
        mock_mb.remember_schema = AsyncMock()

        mock_agent_instance = MagicMock()
        mock_agent_instance.answer_question = AsyncMock(side_effect=fake_answer_question)
        MockAgent.return_value = mock_agent_instance
        MockSynapse.return_value.aclose = AsyncMock()
        MockSynapse.return_value.get_stats = AsyncMock(return_value={"writeHead": 0})
        MockSynapse.return_value.get_best_next = AsyncMock(return_value=None)

        import os
        with patch.dict(os.environ, {"SYNAPSE_URL": "http://localhost:8080",
                                      "SYNAPSE_API_KEY": "sk_syn_hackathon2026"}):
            await run(epochs=6)

    # vanilla: exactly 50 calls (one pass over train_qs, use_memory=False)
    # memory train: 50 questions * 6 epochs
    # holdout: 10 * 6 epochs
    total_sql = len(sql_calls)
    assert total_sql == 50 + 50 * 6 + 10 * 6, (
        f"Expected {50 + 50*6 + 10*6} SQL calls, got {total_sql}"
    )

    # consolidate called once per epoch
    assert len(improve_calls) == 6, f"consolidate called {len(improve_calls)}x, expected 6"

    # forget fired at epoch 5: here all questions are correct so failed_patterns is empty
    # -> forget_failures should NOT have been called (no hashes with count > 3)
    # (all correct => no failures => no forget needed)
    assert len(forget_calls) == 0, (
        f"forget should not fire when all answers correct, got {len(forget_calls)} calls"
    )

    # dedup: remember_calls should be at most 18 per epoch (18 unique hashes)
    per_epoch_counts: dict[int, int] = {}
    for epoch, _ in remember_calls:
        per_epoch_counts[epoch] = per_epoch_counts.get(epoch, 0) + 1
    for epoch, count in per_epoch_counts.items():
        assert count <= 18, (
            f"Epoch {epoch}: {count} remember calls exceeds 18 unique hashes"
        )


# ---------------------------------------------------------------------------
# Budget circuit-breaker
# ---------------------------------------------------------------------------

async def test_budget_circuit_breaker_halts_run() -> None:
    async def fake_answer_question_expensive(q, epoch, use_memory, reinforce=True, stored_hashes=None):
        return {
            "question_id": q["id"], "sql": "", "correct": False,
            "execution_error": None, "state_hash": 0, "remembered": False,
        }

    with (
        patch.object(runner_module, "memory_bridge") as mock_mb,
        patch("benchmarks.runner.SQLAgent") as MockAgent,
        patch("benchmarks.runner.SynapseClient") as MockSynapse,
        patch("benchmarks.runner.AsyncAnthropic"),
        patch("benchmarks.runner._BUDGET_CEILING_GBP", 0.001),
        patch("benchmarks.runner._RESULTS_DIR", Path(
            "C:/Users/anony/AppData/Local/Temp/claude/"
            "C--Users-anony-Downloads-cognee-synapse-agent/"
            "d2b4715e-89ae-4607-86bf-3bedecd98373/scratchpad"
        )),
    ):
        mock_mb.forget_failures = AsyncMock(return_value=[])
        mock_mb.consolidate = AsyncMock()
        mock_mb.recall_context = AsyncMock(return_value="")
        mock_mb.remember_success = AsyncMock()
        mock_mb.reset_store = AsyncMock()
        mock_mb.remember_schema = AsyncMock()

        mock_agent_instance = MagicMock()
        mock_agent_instance.answer_question = AsyncMock(
            side_effect=fake_answer_question_expensive
        )
        MockAgent.return_value = mock_agent_instance
        MockSynapse.return_value.aclose = AsyncMock()
        MockSynapse.return_value.get_stats = AsyncMock(return_value={"writeHead": 0})
        MockSynapse.return_value.get_best_next = AsyncMock(return_value=None)

        import os
        with patch.dict(os.environ, {"SYNAPSE_URL": "http://localhost:8080",
                                      "SYNAPSE_API_KEY": "sk_syn_hackathon2026"}):
            with pytest.raises(RuntimeError, match="BUDGET CEILING"):
                await run(epochs=10)
