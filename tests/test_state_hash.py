import os
import subprocess
import sys

from agent.state_hash import classify_intent, extract_tables, hash_state


def test_hash_is_deterministic():
    ctx = {"intent": "AGGREGATE", "tables": ["Orders", "Customers"], "clauses_so_far": []}
    assert hash_state(ctx) == hash_state(ctx)


def test_hash_stable_across_subprocess():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = (
        "import sys; sys.path.insert(0, '.'); "
        "from agent.state_hash import hash_state; "
        "print(hash_state({'intent': 'AGGREGATE', 'tables': ['Orders', 'Customers'], 'clauses_so_far': []}))"
    )
    r1 = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=True, cwd=project_dir,
    )
    r2 = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, check=True, cwd=project_dir,
    )
    assert r1.stdout.strip() == r2.stdout.strip()
    in_process = hash_state({"intent": "AGGREGATE", "tables": ["Orders", "Customers"], "clauses_so_far": []})
    assert r1.stdout.strip() == str(in_process)


def test_hash_is_positive_31_bit():
    result = hash_state({"intent": "JOIN", "tables": ["Products", "Suppliers"], "clauses_so_far": ["WHERE"]})
    assert 0 <= result <= 0x7FFFFFFF


def test_hash_order_independent():
    ctx_ab = {"intent": "AGGREGATE", "tables": ["Orders", "Customers"], "clauses_so_far": []}
    ctx_ba = {"intent": "AGGREGATE", "tables": ["Customers", "Orders"], "clauses_so_far": []}
    assert hash_state(ctx_ab) == hash_state(ctx_ba)


def test_classify_aggregate_how_many():
    assert classify_intent("How many customers are from Germany?") == "AGGREGATE"


def test_classify_aggregate_howmany_german_customers():
    # CLAUDE.md edge case: aggregate even though it spans two tables
    assert classify_intent("How many orders were placed by German customers?") == "AGGREGATE"


def test_classify_filter_list_orders_from_germany():
    # CLAUDE.md edge case: FILTER not JOIN because "in Germany" is a value constraint
    assert classify_intent("List all orders from customers in Germany.") == "FILTER"


def test_classify_join_orders_and_customers():
    assert classify_intent("List all orders along with their customer names.") == "JOIN"


def test_classify_select_default():
    assert classify_intent("List all products.") == "SELECT"


def test_extract_order_details():
    tables = extract_tables("Show line items for each product.")
    assert "Order Details" in tables


def test_extract_tables_customers_only():
    tables = extract_tables("How many customers are from Germany?")
    assert "Customers" in tables
    assert "Orders" not in tables


def test_extract_tables_multi():
    tables = extract_tables("List orders placed by customers in the USA.")
    assert "Orders" in tables
    assert "Customers" in tables


def test_similar_questions_share_hash():
    ctx1 = {"intent": "AGGREGATE", "tables": ["Customers"], "clauses_so_far": []}
    ctx2 = {"intent": "AGGREGATE", "tables": ["Customers"], "clauses_so_far": []}
    assert hash_state(ctx1) == hash_state(ctx2)


def test_different_intents_different_hash():
    q_agg = {"intent": "AGGREGATE", "tables": ["Orders"], "clauses_so_far": []}
    q_flt = {"intent": "FILTER", "tables": ["Orders"], "clauses_so_far": []}
    assert hash_state(q_agg) != hash_state(q_flt)


def test_order_details_not_confused_with_orders():
    tables_od = extract_tables("Show me the line items for order 10248.")
    tables_o = extract_tables("Show me the order with ID 10248.")
    assert "Order Details" in tables_od
    assert "Order Details" not in tables_o
