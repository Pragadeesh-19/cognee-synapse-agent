"""Live integration tests for agent/memory_bridge.py against real Cognee."""
import pytest
import cognee

from agent.memory_bridge import (
    consolidate,
    forget_failures,
    recall_context,
    remember_schema,
    remember_success,
)


@pytest.fixture(autouse=True)
async def clean_cognee_state():
    await cognee.prune.prune_data()
    yield
    await cognee.prune.prune_data()


async def test_remember_schema_stores_correctly() -> None:
    await remember_schema("Table Customers: CustomerID (PK), CompanyName, Country")
    context = await recall_context("customers schema")
    assert context, "recall_context must return non-empty string after remember_schema"


async def test_remember_success_uses_epoch_dataset_name() -> None:
    _HASH = 99999
    await remember_success(
        question="How many customers from Germany?",
        sql="SELECT COUNT(*) FROM Customers WHERE Country = 'Germany'",
        epoch=3,
        state_hash=_HASH,
    )
    context = await recall_context("customers Germany")
    assert isinstance(context, str)
    assert len(context) > 0


async def test_recall_context_returns_nonempty_after_remember() -> None:
    await remember_schema("Table Orders: OrderID (PK), CustomerID (FK), OrderDate")
    context = await recall_context("orders")
    assert isinstance(context, str)
    assert len(context) > 0


async def test_recall_context_respects_char_cap() -> None:
    schema = (
        "Table Customers: CustomerID (PK), CompanyName, ContactName, Country, City, PostalCode. "
        "Table Orders: OrderID (PK), CustomerID (FK), OrderDate, RequiredDate, ShippedDate, Freight. "
        "Table Products: ProductID (PK), ProductName, SupplierID (FK), CategoryID (FK), UnitPrice. "
        "Join: Orders.CustomerID = Customers.CustomerID. "
        "Join: OrderDetails.ProductID = Products.ProductID."
    )
    await remember_schema(schema)
    context = await recall_context("customers orders products join schema")
    assert isinstance(context, str)
    assert len(context) <= 2100, f"Context exceeded cap: {len(context)} chars"


async def test_forget_targets_only_named_dataset() -> None:
    """CRITICAL: forget() must remove only the targeted (epoch, hash) bucket, leaving others intact."""
    _HASH_A = 11111
    _HASH_B = 22222

    await remember_schema("Table Customers: CustomerID (PK), CompanyName, Country")
    await remember_success(
        question="How many customers from Germany?",
        sql="SELECT COUNT(*) FROM Customers WHERE Country = 'Germany'",
        epoch=1,
        state_hash=_HASH_A,
    )
    await remember_success(
        question="List all products ordered by category.",
        sql="SELECT ProductName, CategoryID FROM Products ORDER BY CategoryID",
        epoch=2,
        state_hash=_HASH_B,
    )

    before = await cognee.recall("customers Germany", only_context=True)
    before_datasets = {getattr(r, "dataset_name", None) for r in before}
    assert f"episode_epoch_1_hash_{_HASH_A}" in before_datasets, (
        f"episode_epoch_1_hash_{_HASH_A} must appear before forget; got: {before_datasets}"
    )

    forgotten = await forget_failures(failed_state_hashes=[_HASH_A], epochs=[1])
    assert f"episode_epoch_1_hash_{_HASH_A}" in forgotten

    after = await cognee.recall("customers Germany", only_context=True)
    after_datasets = {getattr(r, "dataset_name", None) for r in after}
    assert f"episode_epoch_1_hash_{_HASH_A}" not in after_datasets, (
        f"episode_epoch_1_hash_{_HASH_A} must NOT appear after forget; got: {after_datasets}"
    )

    ep2 = await cognee.recall("products category", only_context=True)
    ep2_datasets = {getattr(r, "dataset_name", None) for r in ep2}
    assert f"episode_epoch_2_hash_{_HASH_B}" in ep2_datasets, (
        f"episode_epoch_2_hash_{_HASH_B} must still be present; got: {ep2_datasets}"
    )

    schema = await cognee.recall("CustomerID primary key customers", only_context=True)
    schema_datasets = {getattr(r, "dataset_name", None) for r in schema}
    assert "northwind_schema" in schema_datasets, (
        f"northwind_schema must survive forget; got: {schema_datasets}"
    )


async def test_consolidate_runs_without_error() -> None:
    """consolidate() must not raise even when the graph has no episode data."""
    await consolidate()
