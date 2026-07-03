"""Cognee operations: schema memory, episodic memory, and selective forgetting."""
import logging
import re
import time

import cognee

logger = logging.getLogger(__name__)

# dataset_name required: forget() cannot target individual records, only datasets
_SCHEMA_DATASET = "northwind_schema"

# None = never attempted; str (even "") = attempted — guards cache hit check
_schema_cache: str | None = None
_epoch_recall_cache: dict[int, str] = {}

# Raw schema text preserved exactly as stored — Cognee's graph normalises column names
# to snake_case, so we inject directly from this rather than from recall results.
_raw_schema_by_table: dict[str, str] = {}
_dialect_note: str = ""


def clear_epoch_recall_cache() -> None:
    """Reset per-epoch recall cache so new episodic memories are fetched next epoch."""
    global _epoch_recall_cache
    _epoch_recall_cache = {}


async def reset_store() -> None:
    """Purge all Cognee stores and in-memory caches before a fresh benchmark run."""
    global _schema_cache, _epoch_recall_cache, _raw_schema_by_table, _dialect_note
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(graph=True, vector=True, metadata=True)
    _schema_cache = None
    _epoch_recall_cache = {}
    _raw_schema_by_table = {}
    _dialect_note = ""


def _parse_schema_by_table(schema_text: str) -> tuple[dict[str, str], str]:
    by_table: dict[str, str] = {}
    current_table: str | None = None
    current_lines: list[str] = []
    dialect = ""
    for line in schema_text.splitlines():
        if line.startswith("Table "):
            if current_table:
                by_table[current_table] = "\n".join(current_lines)
            current_table = line.split(":")[0].removeprefix("Table ").strip()
            current_lines = [line]
        elif line.startswith("SQLite dialect"):
            dialect = line
        elif current_table:
            current_lines.append(line)
    if current_table:
        by_table[current_table] = "\n".join(current_lines)
    return by_table, dialect


async def remember_schema(schema_text: str) -> None:
    """Store the Northwind schema as permanent semantic memory."""
    global _raw_schema_by_table, _dialect_note
    _raw_schema_by_table, _dialect_note = _parse_schema_by_table(schema_text)
    await cognee.remember(schema_text, dataset_name=_SCHEMA_DATASET)


async def remember_success(question: str, sql: str, epoch: int, state_hash: int) -> None:
    """Store a correct question/SQL pair in the (epoch, hash) episodic bucket."""
    text = f"Question: {question}\nSQL: {sql}"
    # self_improvement=False: cognee.improve() is called once per epoch via consolidate()
    await cognee.remember(
        text,
        dataset_name=f"episode_epoch_{epoch}_hash_{state_hash}",
        self_improvement=False,
    )


async def consolidate() -> None:
    """Run Cognee graph self-optimisation once after all remembers for an epoch."""
    await cognee.improve()


def _extract_node_content(text: str) -> str:
    match = re.search(r"__node_content_start__(.*?)__node_content_end__", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _is_episode(content: str, dataset_name: str | None, state_hash: int | None) -> bool:
    if dataset_name is not None:
        return state_hash is not None and dataset_name.endswith(f"_hash_{state_hash}")
    # Cognee strips dataset_name from graph nodes; infer from content
    return "Question:" in content and "SQL:" in content


async def recall_context(
    question: str,
    state_hash: int | None = None,
    tables: list[str] | None = None,
) -> str:
    """Return Cognee memory context for the question, cached by state_hash within each epoch."""
    global _schema_cache, _epoch_recall_cache

    if state_hash is not None and _schema_cache is not None and state_hash in _epoch_recall_cache:
        return _epoch_recall_cache[state_hash]

    episode_fragments: list[str] = []
    results = await cognee.recall(question, only_context=True)
    if results:
        for result in results:
            raw_text = getattr(result, "text", "") or ""
            dataset_name = getattr(result, "dataset_name", None)
            content = _extract_node_content(raw_text)
            if content and _is_episode(content, dataset_name, state_hash):
                episode_fragments.append(content)

    if _schema_cache is None:
        _schema_cache = ""

    sections: list[str] = []

    if _raw_schema_by_table:
        if tables:
            schema_lines = [
                _raw_schema_by_table[t] for t in tables if t in _raw_schema_by_table
            ]
        else:
            schema_lines = list(_raw_schema_by_table.values())
        if _dialect_note:
            schema_lines.append(_dialect_note)
        if schema_lines:
            sections.append("[SCHEMA]\n" + "\n".join(schema_lines))

    if episode_fragments:
        sections.append(
            "[PAST QUERIES THAT WORKED FOR SIMILAR QUESTIONS]\n"
            + "\n\n".join(episode_fragments[:2])
        )

    body = "\n\n".join(sections)[:1400]
    context = f"[MEMORY CONTEXT]\n{body}" if body else ""

    if state_hash is not None:
        _epoch_recall_cache[state_hash] = context

    return context


async def forget_failures(failed_state_hashes: list[int], epochs: list[int]) -> list[str]:
    """Forget the (epoch, hash) episodic buckets where Synapse signals persistent failure."""
    forgotten: list[str] = []
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for e in epochs:
        for h in failed_state_hashes:
            dataset = f"episode_epoch_{e}_hash_{h}"
            try:
                await cognee.forget(dataset=dataset)
                forgotten.append(dataset)
                logger.info(
                    "forget ts=%s dataset=%s reason=failure_count>3",
                    timestamp,
                    dataset,
                )
            except Exception:
                logger.exception("forget failed for dataset=%s — continuing", dataset)

    return forgotten
