"""Single-shot SQL agent: question -> SQL -> execute -> score -> reinforce memory."""
import re
import sqlite3
from typing import Any

import anthropic

from agent.answer_check import answers_match, execute
from agent.prompt_builder import build_prompt
from agent.state_hash import hash_state
from agent.synapse_client import SynapseClient

# single-shot by design: multi-step exceeds API budget for 500 questions
_MODEL = "claude-haiku-4-5-20251001"
_VANILLA_PROMPT = "Convert this question to SQL for the Northwind database: {question}"


def _extract_sql(text: str) -> str:
    """Strip markdown code fences from Claude's response, returning raw SQL."""
    match = re.search(r"```(?:sql)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


class SQLAgent:
    def __init__(
        self,
        synapse_client: SynapseClient,
        memory_bridge: Any,
        anthropic_client: anthropic.AsyncAnthropic,
        db_path: str,
    ) -> None:
        self._synapse = synapse_client
        self._memory = memory_bridge
        self._anthropic = anthropic_client
        self._db_path = db_path

    async def answer_question(
        self,
        question: dict,
        epoch: int,
        use_memory: bool,
        reinforce: bool = True,
        stored_hashes: set[int] | None = None,
    ) -> dict:
        """Answer one benchmark question; return result dict including state_hash and remembered."""
        if use_memory:
            return await self._memory_answer(question, epoch, reinforce, stored_hashes)
        return await self._vanilla_answer(question)

    async def _memory_answer(
        self,
        question: dict,
        epoch: int,
        reinforce: bool,
        stored_hashes: set[int] | None,
    ) -> dict:
        ctx_hash = hash_state({
            "intent": question["intent"],
            "tables": question["tables"],
            "clauses_so_far": [],
        })
        cognee_ctx = await self._memory.recall_context(
            question["question"], ctx_hash, tables=question.get("tables")
        )
        try:
            hint = await self._synapse.get_best_next("0", ctx_hash)
        except Exception:
            hint = None
        if hint is not None:
            hint["intent"] = question["intent"]
        prompt = build_prompt(question["question"], cognee_ctx, hint)

        sql = await self._call_claude(prompt, temperature=0.0)
        correct, error = self._evaluate(sql, question["gold_answer"])

        if reinforce:
            try:
                await self._synapse.append_thought("0", None, ctx_hash, 1.0 if correct else 0.0)
            except Exception:
                pass

        remembered = False
        if reinforce and correct and (stored_hashes is None or ctx_hash not in stored_hashes):
            await self._memory.remember_success(question["question"], sql, epoch, ctx_hash)
            if stored_hashes is not None:
                stored_hashes.add(ctx_hash)
            remembered = True

        return {
            "question_id": question["id"],
            "sql": sql,
            "correct": correct,
            "execution_error": error,
            "state_hash": ctx_hash,
            "remembered": remembered,
        }

    async def _vanilla_answer(self, question: dict) -> dict:
        prompt = _VANILLA_PROMPT.format(question=question["question"])
        sql = await self._call_claude(prompt, temperature=0.0)
        correct, error = self._evaluate(sql, question["gold_answer"])
        return {
            "question_id": question["id"],
            "sql": sql,
            "correct": correct,
            "execution_error": error,
            "state_hash": None,
            "remembered": False,
        }

    async def _call_claude(self, prompt: str, temperature: float = 0.0) -> str:
        response = await self._anthropic.messages.create(
            model=_MODEL,
            max_tokens=256,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return _extract_sql(response.content[0].text)

    def _evaluate(self, sql: str, gold_answer: Any) -> tuple[bool, str | None]:
        try:
            conn = sqlite3.connect(self._db_path)
            try:
                actual = execute(conn, sql)
            finally:
                conn.close()
            return answers_match(actual, gold_answer), None
        except Exception as exc:
            return False, str(exc)
