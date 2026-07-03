"""Verify all 60 gold_sql queries execute and match stored gold_answer values."""
import json
import sqlite3
import sys

from agent.answer_check import answers_match, execute

DB = "benchmarks/northwind.db"
QF = "benchmarks/questions.json"


def main():
    with open(QF, encoding="utf-8") as f:
        questions = json.load(f)

    conn = sqlite3.connect(DB)
    failures = []

    for q in questions:
        try:
            actual = execute(conn, q["gold_sql"])
        except Exception as e:
            failures.append(f"Q{q['id']:02d} SQL ERROR: {e}\n  SQL: {q['gold_sql']}")
            continue

        if not answers_match(actual, q["gold_answer"]):
            failures.append(
                f"Q{q['id']:02d} MISMATCH\n"
                f"  question: {q['question']}\n"
                f"  expected: {str(q['gold_answer'])[:80]}\n"
                f"  actual:   {str(actual)[:80]}"
            )

    conn.close()

    if failures:
        for msg in failures:
            print(msg)
        print(f"\nFAIL: {len(failures)} of {len(questions)} questions failed verification")
        sys.exit(1)
    else:
        print(f"PASS: all {len(questions)} questions verified against {DB}")


if __name__ == "__main__":
    main()
