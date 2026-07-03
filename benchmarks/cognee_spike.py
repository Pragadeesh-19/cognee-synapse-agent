"""Phase C spike: verify cognee.forget(dataset=...) selectively removes only named datasets."""
import asyncio

import cognee


async def main() -> None:
    print("=== Phase C Cognee Spike ===\n")

    print("[0] Pruning all data to start clean...")
    await cognee.prune.prune_data()
    print("    Done.\n")

    print("[1] remember schema fact -> dataset_name='northwind_schema'")
    await cognee.remember(
        "Table Customers: CustomerID (PK), CompanyName, Country. "
        "Table Orders: OrderID (PK), CustomerID (FK).",
        dataset_name="northwind_schema",
    )
    print("    Done.\n")

    print("[2] remember episode -> dataset_name='episode_epoch_1'")
    await cognee.remember(
        "Question: How many customers from Germany? "
        "SQL: SELECT COUNT(*) FROM Customers WHERE Country = 'Germany' "
        "Result: 11",
        dataset_name="episode_epoch_1",
    )
    print("    Done.\n")

    print("[3] remember episode -> dataset_name='episode_epoch_2'")
    await cognee.remember(
        "Question: List all products ordered by category. "
        "SQL: SELECT ProductName, CategoryID FROM Products ORDER BY CategoryID "
        "Result: 77 rows",
        dataset_name="episode_epoch_2",
    )
    print("    Done.\n")

    print("[4] recall BEFORE forget (query: 'customers Germany')...")
    results_before = await cognee.recall("customers Germany", only_context=True)
    _print_results("BEFORE forget", results_before)

    print("[5] cognee.forget(dataset='episode_epoch_1')...")
    await cognee.forget(dataset="episode_epoch_1")
    print("    Done.\n")

    print("[6] recall AFTER forget (query: 'customers Germany')...")
    results_after = await cognee.recall("customers Germany", only_context=True)
    _print_results("AFTER forget", results_after)

    print("[7] recall epoch_2 content still present (query: 'products category')...")
    results_ep2 = await cognee.recall("products category", only_context=True)
    _print_results("epoch_2 check", results_ep2)

    print("[8] recall schema still present (query: 'CustomerID primary key')...")
    results_schema = await cognee.recall("CustomerID primary key", only_context=True)
    _print_results("schema check", results_schema)

    print("[9] Final prune to reset state...")
    await cognee.prune.prune_data()
    print("    Done.\n")

    _print_verdict(results_before, results_after, results_ep2, results_schema)


def _print_results(label: str, results: list) -> None:
    print(f"    [{label}] {len(results)} result(s)")
    for i, r in enumerate(results[:5]):
        text = getattr(r, "text", str(r))[:100]
        dataset = getattr(r, "dataset_name", "?")
        kind = getattr(r, "kind", "?")
        print(f"      [{i}] dataset={dataset!r} kind={kind} text={text!r}")
    print()


def _print_verdict(before: list, after: list, ep2: list, schema: list) -> None:
    print("=== VERDICT ===")
    epoch1_text = "SELECT COUNT(*) FROM Customers WHERE Country"

    before_has_ep1 = any(epoch1_text in getattr(r, "text", "") for r in before)
    after_has_ep1 = any(epoch1_text in getattr(r, "text", "") for r in after)
    ep2_present = len(ep2) > 0
    schema_present = len(schema) > 0

    print(f"  epoch_1 content present BEFORE forget:  {before_has_ep1}")
    print(f"  epoch_1 content present AFTER forget:   {after_has_ep1}")
    print(f"  epoch_2 still present after forget:     {ep2_present}")
    print(f"  schema still present after forget:      {schema_present}")
    print()

    if before_has_ep1 and not after_has_ep1 and ep2_present and schema_present:
        print("  PASS -- forget() is selective: removed epoch_1, kept epoch_2 + schema")
        print("  Proceed to Step 2: build agent/memory_bridge.py")
    elif not before_has_ep1:
        print("  INCONCLUSIVE -- epoch_1 content never appeared in recall (check remember())")
        print("  before results had no epoch_1 text -- either recall is not working,")
        print("  or the text was chunked/processed differently than expected.")
        print("  Investigate before proceeding.")
    elif after_has_ep1:
        print("  FAIL -- epoch_1 content still present after forget() -- NOT selective")
        print("  STOP: do not proceed to Phase D until this is resolved")
    else:
        print("  PARTIAL -- epoch_1 gone but epoch_2 or schema also missing")
        print("  Investigate: forget() may be wiping too broadly")


if __name__ == "__main__":
    asyncio.run(main())
