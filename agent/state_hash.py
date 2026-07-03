import hashlib
import re

# 31-bit mask: Synapse stateHash is a signed Java int, must be positive
_HASH_MASK = 0x7FFFFFFF

_AGGREGATE_KEYWORDS = [
    "how many", "total", "average", "avg", "count", "sum",
    "max", "min", "most", "fewest", "highest", "lowest",
    "least", "greatest", "largest", "smallest",
]

_FILTER_TRIGGER_WORDS = frozenset([
    "from", "in", "where", "which", "that", "with", "by", "between",
])

_ARTICLES = frozenset(["the", "a", "an"])

_TABLE_LOWER_NAMES = frozenset([
    "customers", "customer", "orders", "order",
    "products", "product", "suppliers", "supplier",
    "categories", "category", "employees", "employee",
    "shippers", "shipper", "regions", "region",
    "territories", "territory",
    "order details", "order detail", "line items", "line item",
])

# Multi-word aliases listed before single-word to prevent premature matching
_TABLE_ALIASES: list[tuple[list[str], str]] = [
    (["order detail", "order details", "line item", "line items"], "Order Details"),
    (["customer", "customers", "client", "clients", "buyer", "buyers"], "Customers"),
    (["order", "orders", "purchase", "purchases"], "Orders"),
    (["product", "products", "item", "items", "goods"], "Products"),
    (["supplier", "suppliers", "vendor", "vendors"], "Suppliers"),
    (["category", "categories"], "Categories"),
    (["employee", "employees", "staff", "salesperson", "salespeople"], "Employees"),
    (["shipper", "shippers", "carrier", "carriers"], "Shippers"),
    (["region", "regions"], "Regions"),
    (["territory", "territories"], "Territories"),
]


def classify_intent(question: str) -> str:
    """Classify question intent as AGGREGATE, FILTER, JOIN, or SELECT."""
    q_lower = question.lower()

    for kw in _AGGREGATE_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", q_lower):
            return "AGGREGATE"

    tokens = re.findall(r"\b[A-Za-z0-9]+\b", question)
    if _has_filter_value(tokens):
        return "FILTER"

    if len(_referenced_entities(question)) >= 2:
        return "JOIN"

    return "SELECT"


def extract_tables(question: str) -> list[str]:
    """Extract Northwind table names referenced in the question, in order of first mention."""
    q_lower = question.lower()
    positions: list[tuple[int, str]] = []

    for aliases, table_name in _TABLE_ALIASES:
        for alias in aliases:
            for match in re.finditer(r"\b" + re.escape(alias) + r"\b", q_lower):
                positions.append((match.start(), table_name))

    seen: set[str] = set()
    result: list[str] = []
    for _, table_name in sorted(positions):
        if table_name not in seen:
            seen.add(table_name)
            result.append(table_name)
    return result


def hash_state(context: dict) -> int:
    """Map question context to a stable 31-bit Synapse bucket identifier."""
    intent = context.get("intent", "")
    tables = sorted(context.get("tables", []))
    clauses = list(context.get("clauses_so_far", []))
    canonical = f"{intent}|{','.join(tables)}|{','.join(clauses)}"
    digest = hashlib.sha256(canonical.encode()).digest()
    return int.from_bytes(digest[:4], "big") & _HASH_MASK


def _has_filter_value(tokens: list[str]) -> bool:
    for i, token in enumerate(tokens):
        if token.lower() not in _FILTER_TRIGGER_WORDS:
            continue
        j = i + 1
        while j < len(tokens) and tokens[j].lower() in _ARTICLES:
            j += 1
        if j >= len(tokens):
            continue
        next_tok = tokens[j]
        is_value_noun = next_tok[0].isupper() and next_tok.lower() not in _TABLE_LOWER_NAMES
        is_number = bool(re.match(r"^\d{2,4}$", next_tok))
        if is_value_noun or is_number:
            return True
    return False


def _referenced_entities(question: str) -> set[str]:
    q_lower = question.lower()
    entities: set[str] = set()
    for aliases, table_name in _TABLE_ALIASES:
        for alias in aliases:
            if re.search(r"\b" + re.escape(alias) + r"\b", q_lower):
                entities.add(table_name)
                break
    return entities
