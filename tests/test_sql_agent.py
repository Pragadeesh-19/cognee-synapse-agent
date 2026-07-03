"""Tests for agent/sql_agent.py — mocked tests run first, live tests in Step 6."""
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import cognee
import pytest

import agent.memory_bridge as memory_bridge_module
from agent.sql_agent import SQLAgent
from agent.state_hash import hash_state

DB_PATH = str(Path(__file__).parent.parent / "benchmarks" / "northwind.db")

with open(Path(__file__).parent.parent / "benchmarks" / "questions.json", encoding="utf-8") as _f:
    _ALL_QUESTIONS = {q["id"]: q for q in json.load(_f)}

_Q1 = _ALL_QUESTIONS[1]   # "How many customers are from Germany?" gold=11  AGGREGATE
_Q3 = _ALL_QUESTIONS[3]   # "How many customers are there in total?" gold=93 AGGREGATE
_Q49 = _ALL_QUESTIONS[49] # "List all shippers." gold=[...3 items]          SELECT
_Q11 = _ALL_QUESTIONS[11] # "How many products are there in total?" gold=77  AGGREGATE

_CORRECT_SQL_Q1 = "SELECT COUNT(*) FROM Customers WHERE Country = 'Germany'"


def _mock_anthropic(sql_text: str) -> MagicMock:
    mock = MagicMock()
    mock.messages.create = AsyncMock(
        return_value=MagicMock(content=[MagicMock(text=sql_text)])
    )
    return mock


def _mock_bridge(context: str = "") -> MagicMock:
    bridge = MagicMock()
    bridge.recall_context = AsyncMock(return_value=context)
    bridge.remember_success = AsyncMock()
    return bridge


def _mock_synapse(score: float = 0.0) -> MagicMock:
    synapse = MagicMock()
    synapse.get_best_next = AsyncMock(return_value={"found": True, "slot": 0, "score": score})
    synapse.append_thought = AsyncMock(return_value=42)
    return synapse


# ---------------------------------------------------------------------------
# Mocked tests (no API cost)
# ---------------------------------------------------------------------------

async def test_vanilla_agent_gets_no_memory_context() -> None:
    agent = SQLAgent(_mock_synapse(), _mock_bridge(), _mock_anthropic(_CORRECT_SQL_Q1), DB_PATH)
    await agent.answer_question(_Q1, epoch=1, use_memory=False)

    call_kwargs = agent._anthropic.messages.create.call_args.kwargs
    prompt = call_kwargs["messages"][0]["content"]

    assert "[SCHEMA]" not in prompt
    assert "PAST QUERIES" not in prompt
    assert "REASONING APPROACH" not in prompt
    agent._memory.recall_context.assert_not_called()
    agent._synapse.get_best_next.assert_not_called()
    agent._synapse.append_thought.assert_not_called()


async def test_memory_agent_includes_context() -> None:
    rich_context = (
        "[SCHEMA]\nTable Customers: CustomerID (PK), CompanyName, Country\n\n"
        "[PAST QUERIES THAT WORKED FOR SIMILAR QUESTIONS]\n"
        "Question: How many customers?\nSQL: SELECT COUNT(*) FROM Customers"
    )
    # score > 0 so reasoning hint should be injected
    synapse = _mock_synapse(score=1.5)
    agent = SQLAgent(synapse, _mock_bridge(rich_context), _mock_anthropic(_CORRECT_SQL_Q1), DB_PATH)
    result = await agent.answer_question(_Q1, epoch=1, use_memory=True)

    call_kwargs = agent._anthropic.messages.create.call_args.kwargs
    prompt = call_kwargs["messages"][0]["content"]

    assert "[SCHEMA]" in prompt
    assert "PAST QUERIES" in prompt
    assert "[REASONING APPROACH FOR THIS QUESTION TYPE]" in prompt
    assert result["correct"] is True
    assert "state_hash" in result
    assert isinstance(result["state_hash"], int)
    assert result["remembered"] is True
    agent._memory.recall_context.assert_called_once()
    agent._memory.remember_success.assert_called_once()


async def test_reinforce_false_skips_synapse_and_remember() -> None:
    agent = SQLAgent(_mock_synapse(), _mock_bridge(), _mock_anthropic(_CORRECT_SQL_Q1), DB_PATH)
    result = await agent.answer_question(_Q1, epoch=1, use_memory=True, reinforce=False)

    assert result["correct"] is True
    assert result["remembered"] is False
    agent._synapse.append_thought.assert_not_called()
    agent._memory.remember_success.assert_not_called()


async def test_stored_hashes_dedup_prevents_second_remember() -> None:
    seen: set[int] = set()
    agent = SQLAgent(_mock_synapse(), _mock_bridge(), _mock_anthropic(_CORRECT_SQL_Q1), DB_PATH)

    result1 = await agent.answer_question(_Q1, epoch=1, use_memory=True, stored_hashes=seen)
    assert result1["remembered"] is True
    assert result1["state_hash"] in seen

    result2 = await agent.answer_question(_Q1, epoch=1, use_memory=True, stored_hashes=seen)
    assert result2["remembered"] is False
    assert agent._memory.remember_success.call_count == 1


async def test_malformed_sql_treated_as_incorrect_not_crash() -> None:
    agent = SQLAgent(
        _mock_synapse(), _mock_bridge(),
        _mock_anthropic("This is absolutely not valid SQL!!!"),
        DB_PATH,
    )
    result = await agent.answer_question(_Q1, epoch=1, use_memory=False)

    assert result["correct"] is False
    assert result["execution_error"] is not None
    assert result["question_id"] == _Q1["id"]


# ---------------------------------------------------------------------------
# Live tests (real Synapse + Cognee + Claude) — Step 6
# ---------------------------------------------------------------------------

async def test_answer_question_executes_sql_against_db(live_synapse) -> None:
    """Live end-to-end: 3 questions, real Claude, verify no crashes and writeHead advances."""
    await cognee.prune.prune_data()

    from anthropic import AsyncAnthropic
    from dotenv import load_dotenv
    load_dotenv()

    client = AsyncAnthropic()
    agent = SQLAgent(live_synapse, memory_bridge_module, client, DB_PATH)

    stats_before = await live_synapse.get_stats("0")
    write_head_before = stats_before["writeHead"]

    for q in [_Q3, _Q11, _Q49]:
        result = await agent.answer_question(q, epoch=1, use_memory=True)
        assert "question_id" in result
        assert "sql" in result
        assert isinstance(result["correct"], bool)
        assert "execution_error" in result

    stats_after = await live_synapse.get_stats("0")
    assert stats_after["writeHead"] > write_head_before, (
        f"Synapse writeHead did not advance: before={write_head_before} after={stats_after['writeHead']}"
    )

    await cognee.prune.prune_data()


async def test_synapse_thought_appended_after_question(live_synapse) -> None:
    """TRIPWIRE: verify Synapse stores per-hash scores, not a global fallback."""
    # Mock Claude to return the provably correct SQL for Q1 (gold=11, verified against DB)
    agent = SQLAgent(
        live_synapse,
        _mock_bridge(),
        _mock_anthropic(_CORRECT_SQL_Q1),
        DB_PATH,
    )

    stats_before = await live_synapse.get_stats("0")
    write_head_before = stats_before["writeHead"]

    result = await agent.answer_question(_Q1, epoch=1, use_memory=True)

    stats_after = await live_synapse.get_stats("0")
    assert stats_after["writeHead"] > write_head_before, (
        f"writeHead did not advance: before={write_head_before} after={stats_after['writeHead']}"
    )
    assert result["correct"] is True, f"Q1 should be correct; got sql={result['sql']!r}"

    q1_hash = hash_state({"intent": _Q1["intent"], "tables": _Q1["tables"], "clauses_so_far": []})
    unused_hash = hash_state({"intent": "SELECT", "tables": ["Territories"], "clauses_so_far": []})

    hint_q1 = await live_synapse.get_best_next("0", q1_hash)
    hint_unused = await live_synapse.get_best_next("0", unused_hash)

    q1_score = hint_q1.get("score", 0.0) if hint_q1 else 0.0
    unused_score = hint_unused.get("score", 0.0) if hint_unused else 0.0

    print(f"\nTRIPWIRE:")
    print(f"  Q1 hash={q1_hash}    -> score={q1_score}")
    print(f"  Unused hash={unused_hash} -> score={unused_score}")

    assert q1_score > unused_score, (
        f"TRIPWIRE FIRED: stateHash appears to be ignored by Synapse.\n"
        f"  After appending score=1.0 for hash {q1_hash}, got score={q1_score}.\n"
        f"  Unused hash {unused_hash} also returned score={unused_score}.\n"
        f"  Expected q1_score > unused_score. Do NOT proceed to Phase E.\n"
        f"  hint_q1={hint_q1}, hint_unused={hint_unused}"
    )
