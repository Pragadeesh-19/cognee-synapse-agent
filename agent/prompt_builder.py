"""Construct the single-shot LLM prompt from question, Cognee context, and Synapse hint."""

_REASONING_TEMPLATES: dict[str, str] = {
    "AGGREGATE": (
        "For aggregate questions (COUNT, SUM, AVG):\n"
        "  Step 1: Identify which table(s) hold the data to count or sum\n"
        "  Step 2: If filtering by a property in another table, JOIN first then aggregate\n"
        "  Step 3: Use GROUP BY only when aggregating per category; omit for total counts"
    ),
    "JOIN": (
        "For join questions (linking two or more tables):\n"
        "  Step 1: Identify the foreign key relationship between the tables\n"
        "  Step 2: Write the JOIN condition explicitly (table1.col = table2.col)\n"
        "  Step 3: Apply WHERE filters after the JOIN"
    ),
    "FILTER": (
        "For filter questions (WHERE clause with a specific value):\n"
        "  Step 1: Identify the target table and the filter column\n"
        "  Step 2: Match the exact string value (case-sensitive in SQLite)\n"
        "  Step 3: Use ORDER BY for deterministic output"
    ),
    "SELECT": (
        "For simple retrieval questions (list all / show all):\n"
        "  Step 1: SELECT the display column (e.g., CompanyName, ProductName)\n"
        "  Step 2: Use ORDER BY for deterministic output\n"
        "  Step 3: No WHERE clause unless the question specifies a condition"
    ),
}


def build_prompt(question: str, cognee_context: str, synapse_hint: dict | None) -> str:
    """Build the user-message prompt with injected schema, episodes, and optional reasoning hint."""
    parts: list[str] = []

    if cognee_context:
        parts.append(cognee_context)

    # score > 0: Synapse returns found=True with score=0.0 for all unseen hashes; score is the real reinforcement signal
    if synapse_hint is not None and synapse_hint.get("score", 0) > 0:
        intent = synapse_hint.get("intent", "")
        template = _REASONING_TEMPLATES.get(intent, "")
        if template:
            parts.append(f"[REASONING APPROACH FOR THIS QUESTION TYPE]\n{template}")

    parts.append(f"Now answer:\n{question}\nReturn only the SQL query, nothing else.")

    return "\n\n".join(parts)
