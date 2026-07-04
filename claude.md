# Cognee + Synapse Hackathon Project
## Three-Memory-Type SQL Agent
Version 2.0 FINAL | Python 3.12 | Hackathon: June 29 to July 6 2026

This file is the complete specification. Claude Code reads this before writing
a single line of code. Every architectural decision is documented here with
its justification. Nothing is left to inference.

---

## Outcome vs plan (read this first)

This file is the original build spec, preserved as written. The shipped result
diverged from it in three honest ways. All three are documented in full in
README.md under "Honest caveats"; where the plan below and the result disagree,
the README is authoritative.

  1. Model: the run used `claude-haiku-4-5` for cost (~$0.0006/call against
     ~$0.005 for Sonnet), not the `claude-sonnet-4-6` this spec mandates.
  2. Accuracy: learning plateaued at 58% from epoch 3, not the 70-75% target.
     The hardest JOIN questions fail on epoch 1 at temperature 0, so they never
     seed an episode to recall later, which caps the ceiling. The robust claim is
     the memory-vs-vanilla gap: 58% vs 26% (+32 points), not the epoch climb.
  3. Synapse: `best-next` ignores stateHash in the deployed build (unseen hashes
     return the same salience as written ones). Per-type differentiation therefore
     runs through Cognee's hash-keyed episodic recall; Synapse provides the global
     Hebbian reinforce-and-decay signal that drives forget(). demo.py PART 2 is the
     falsifiable proof that the state hash is the mechanism.

Everything below is the plan as specified before those findings.

---

## What This Is

The first AI agent runtime with all three human memory types working simultaneously:

  Semantic memory   -> Cognee knowledge graph   (what things mean)
  Episodic memory   -> Cognee knowledge graph   (what happened before)
  Procedural memory -> Synapse-DB               (HOW to do things)

The proof: a text-to-SQL agent that climbs from ~45% to ~75% accuracy over 10 epochs.
Same Claude model. No fine-tuning. Only the memory layer changes.
A vanilla baseline stays flat at ~47%. That gap is the product.

---

## Hackathon Rules

Everything built from scratch after June 29. Cognee APIs and integrations permitted.

Synapse-DB is infrastructure. Same category as PostgreSQL or Redis in any other project.
The submission is everything in this repo built during the hackathon week.

README must state this clearly:
Synapse-DB was built before the hackathon and is used as the procedural memory engine.
All agent code, benchmark, state hash, memory bridge, dashboard, and demo were built
from scratch during the hackathon week.

---

## Infrastructure Already Running

Synapse-DB:
  URL:     http://localhost:8080
  API Key: sk_syn_hackathon2026 (header: X-Api-Key)
  Agent:   agentId=0 (pre-seeded)
  Status:  Running in Docker, verified

Cognee:
  Version:   1.2.2
  LLM:       claude-sonnet-4-6
  Embedding: fastembed BAAI/bge-small-en-v1.5 (local, no API cost)
  Status:    remember() and recall() verified working

Northwind SQLite:
  Path:    benchmarks/northwind.db
  Source:  github.com/jpwhite3/northwind-SQLite3
  Must exist before Phase D begins.

All keys are in .env. Always load with python-dotenv.

---

## Commands

```bash
uv run pytest
uv run python benchmarks/hash_spike.py
uv run python benchmarks/runner.py --epochs 10
uv run streamlit run dashboard/app.py
uv run python demo.py
```

---

## Critical Decision: Single-Shot for Benchmark

Multi-step reasoning (3-5 LLM calls per question) costs:
  500 questions x 4 steps = 2000 calls = ~12 GBP. Over budget.

Single-shot (1 LLM call per question) costs:
  500 training + 500 vanilla = 1000 calls = ~5 GBP. Within budget.

DECISION: The benchmark uses single-shot. One LLM call per question.

Synapse still receives one thought node per question. State hash groups question
types. Reinforcement tracks which SQL approach works per type. Procedural memory
is real and meaningful. The agent learns which approach to use for AGGREGATE vs
JOIN vs FILTER question types, accumulated across hundreds of attempts.

Multi-step is used ONLY in demo.py to make the reasoning chain visible.
In the demo, show each reasoning step being written to Synapse one at a time
so judges can see the procedural memory being built.

This distinction must be explicit in the README.

---

## Three Memory Types: Exact Mapping

SEMANTIC (Cognee, loaded once before benchmark):
  What:    Northwind schema, table names, columns, FK relationships
  When:    Phase D setup, before any questions are answered
  How:     cognee.remember(schema_text, dataset_name="northwind_schema")
  Recall:  cognee.recall(question) surfaces relevant schema fragments
  Forget:  NEVER. Schema does not change. "northwind_schema" is never pruned.

EPISODIC (Cognee, stored after each correct answer):
  What:    "Question: {q} SQL: {sql} Result: {result}"
  When:    After each correct execution during training epochs
  How:     cognee.remember(text, dataset_name=f"episode_epoch_{epoch}")
  Recall:  cognee.recall(question) surfaces similar past successful queries
  Forget:  Epoch 5 checkpoint, driven by Synapse failure signal

PROCEDURAL (Synapse-DB, stored per question attempt):
  What:    Which SQL approach worked for which question type
  When:    After every question, correct or wrong
  How:     synapse.append_thought(agent_id="0", state_hash, success_score)
  Recall:  synapse.get_best_next("0", state_hash)
  Improve: success_score=1.0 correct, 0.0 wrong. Hebbian reinforcement automatic.
  Decay:   Time-decay in Synapse: stale patterns lose salience automatically.

---

## The forget() Integration: The Differentiator

All four Cognee verbs used. Forget driven by real Synapse signal.
No other team at this hackathon has this signal. We do.

CRITICAL RULE: every cognee.remember() call MUST pass dataset_name.
If no dataset_name is passed, forget() cannot target it later.
This rule has no exceptions.

Dataset naming convention:
  "northwind_schema"        permanent, never forgotten
  "episode_epoch_{n}"       successful query pairs from epoch n (n is int)

At epoch 5 checkpoint:
  1. Review failed_patterns dict (tracked in runner.py across the run)
     Structure: {state_hash: int (failure count)}
  2. For any state_hash where failed_count > 3:
     Call cognee.forget(f"episode_epoch_{n}") for epochs 1-4
     where those state hashes dominated the failure pattern
  3. Log every forget() call: timestamp, dataset_name, reason, failure_count

Why this works:
  Cognee may have remembered wrong SQL from early epochs before the agent
  learned the pattern was wrong. forget() removes those misleading episodic
  memories so they stop polluting future recall.
  The agent self-cleans without human intervention.

Test forget() with named datasets in Phase C BEFORE anything else depends on it.
This is the highest-risk integration. Find breakage in Phase C, not Phase E.

---

## The Vanilla Baseline: Zero Help

Same Claude model. Same execution harness. Zero memory context injected.

Vanilla prompt:
  "Convert this question to SQL for the Northwind database: {question}"

Do NOT give vanilla: schema hints, table names, column names, few-shot examples,
step-by-step instructions, or any contextual help whatsoever.

This pushes baseline to ~45-50% on an unfamiliar schema.
The memory agent receives full Cognee recall plus Synapse best-next context.

If the gap at epoch 10 is less than 15 percentage points, debug recall injection
before submitting any results. A small gap means the memory layer is not contributing.

---

## State Hash

Implementation in agent/state_hash.py:

```python
def hash_state(context: dict) -> int:
    intent  = context.get("intent")
    tables  = frozenset(context.get("tables", []))
    clauses = tuple(context.get("clauses_so_far", []))
    return hash((intent, tables, clauses)) & 0x7FFFFFFF
```

Intent classification (keyword extraction, not string matching):
  AGGREGATE -> how many, total, average, count, sum, max, min
  JOIN      -> question references multiple distinct entity types
  FILTER    -> where, which, that, in, from with a specific value constraint
  SELECT    -> simple retrieval, no aggregation, no explicit conditions

Edge cases:
  "list all orders from customers in Germany" -> FILTER not JOIN
  Intent is determined by what the question asks, not what tables it touches.
  "how many orders from German customers" -> AGGREGATE, tables={Orders, Customers}

Northwind tables for extraction:
  Customers, Orders, OrderDetails, Products, Suppliers,
  Categories, Employees, Shippers, Region, Territories

Plan B if spike fails:
  Lower threshold to 50% within-family before stopping entirely.
  Add MIXED intent for genuinely ambiguous questions.
  Do not proceed with below 40% within-family under any circumstances.

Phase B exit gate:
  within-family collision rate > 60%
  cross-family collision rate  < 10%
  DO NOT proceed to Phase C without passing both conditions.

---

## Recall Injection Format

Always inject into the USER message. Never as system instructions.
Evidence gets reasoned about. Instructions get followed blindly.

```
[SCHEMA]
Table Customers: CustomerID (PK), CompanyName, ContactName, Country, City
Table Orders: OrderID (PK), CustomerID (FK -> Customers), OrderDate, Freight
Join: Orders.CustomerID = Customers.CustomerID

[PAST QUERIES THAT WORKED FOR SIMILAR QUESTIONS]
Q: How many orders were placed by French customers?
SQL: SELECT COUNT(*) FROM Orders o JOIN Customers c
     ON o.CustomerID = c.CustomerID WHERE c.Country = 'France'

[REASONING APPROACH FOR THIS QUESTION TYPE]
For aggregate questions over joined tables:
  Step 1: Identify the join key (CustomerID)
  Step 2: Apply WHERE filter before aggregating
  Step 3: Use COUNT(*) after the join

Now answer:
{current_question}
Return only the SQL query, nothing else.
```

Constraints:
  Maximum 2 Cognee episodic examples per prompt
  Maximum 1 Synapse procedural hint per prompt
  Only inject Synapse hint if that thought has been reinforced at least once
  Fresh thoughts (never reinforced) carry no signal. Do not inject them.

---

## questions.json Format

60 questions total. 15 per intent category.
Written manually. Every gold_sql verified against northwind.db before saving.

```json
[
  {
    "id": 1,
    "question": "How many customers are from Germany?",
    "intent": "AGGREGATE",
    "tables": ["Customers"],
    "gold_sql": "SELECT COUNT(*) FROM Customers WHERE Country = 'Germany'",
    "gold_answer": 11
  }
]
```

Make the benchmark hard enough that vanilla fails but memory helps:
  5 questions requiring multi-table JOINs
  5 questions with date arithmetic (strftime, YEAR equivalent)
  5 questions with nested aggregation or subqueries
  These are the questions where the vanilla agent fails and procedural memory guides.

Split:
  50 training questions (reinforced each epoch)
  10 hold-out questions (never reinforced, test generalisation)

---

## Accuracy Targets

These are targets. Publish actual results. Never fabricate.

Training set (50 questions):
  Epoch 1:  ~45% (cold start, no memory)
  Epoch 5:  ~60-65%
  Epoch 10: ~70-75%

Hold-out set (10 questions, never reinforced):
  Epoch 10: ~57-60% (improvement proves generalisation, not memorisation)

Vanilla baseline (same model, no context, all epochs):
  ~45-50%, flat

The gap at epoch 10 is the claim. A 25-30 point gap is compelling.
A gap below 15 points requires investigation before submission.

Budget:
  500 training calls + 500 vanilla calls = 1000 total = ~5 GBP
  You have exactly enough for one clean run plus one debug run.
  Do not add epochs without checking cost first.
  Log actual cost in results/ and report it in the README.

---

## Project Structure

cognee-synapse-agent/
+-- agent/
|   +-- __init__.py
|   +-- synapse_client.py     # HTTP client for Synapse REST API only
|   +-- memory_bridge.py      # All Cognee operations only
|   +-- state_hash.py         # Context dict -> 31-bit int
|   +-- sql_agent.py          # Question -> SQL via Claude, single-shot
|   +-- prompt_builder.py     # Memory context -> LLM prompt string
+-- benchmarks/
|   +-- northwind.db          # SQLite Northwind (download before Phase D)
|   +-- questions.json        # 60 questions, written and verified manually
|   +-- hash_spike.py         # Phase B: collision matrix
|   +-- runner.py             # Epoch runner, logs to results/
|   +-- plot.py               # Two-line learning curve PNG
+-- tests/
|   +-- test_synapse_client.py    # Unit: mocked httpx responses
|   +-- test_state_hash.py        # Unit: deterministic, 31-bit, clustering
|   +-- test_memory_bridge.py     # Unit: Cognee round-trip, forget() targeting
|   +-- test_sql_agent.py         # Integration: 3 questions end-to-end
|   +-- conftest.py               # Fixtures: live_synapse, live_cognee
+-- dashboard/
|   +-- app.py                # Streamlit: 4 panels
+-- results/
|   +-- .gitkeep              # gitignored except .gitkeep
+-- .env
+-- .gitignore
+-- CLAUDE.md
+-- README.md
+-- demo.py
+-- pyproject.toml

---

## The Failure Mode Demo (Required in Phase G)

demo.py runs TWO benchmark runs in sequence.

Run 1: Broken hash (use constant 12345 for all questions)
  Every question maps to the same Synapse bucket
  Recall returns the same path regardless of question type
  Dashboard shows: Synapse HIT on every question (false, misleading signal)
  Learning curve: flat or worse than vanilla

Run 2: Real hash (correct implementation)
  State correctly differentiates AGGREGATE vs JOIN vs FILTER vs SELECT
  Dashboard shows: MISS on first encounter, HIT on subsequent same-type questions
  Learning curve: climbs 25-30 points above vanilla

This is the falsifiability proof. Without it a judge can claim coincidence.
With it, you have experimental evidence that procedural memory is the mechanism.

---

## Dashboard (dashboard/app.py)

Must show four panels:

1. Learning curve:
   X=epoch, Y=accuracy%
   Two lines: Memory Agent (blue) and Vanilla (grey)
   Updates after each epoch during a live run

2. Synapse memory status:
   Fill %: how full is the agent shard
   Write head: how many thoughts stored
   HIT (green dot) or MISS (red dot) for the last question answered
   This makes procedural memory visible in real time

3. Cognee activity log:
   Last 5 remember() calls with dataset_name shown
   Any forget() calls: timestamp, dataset_name, failure count that triggered it

4. Cost tracker:
   LLM calls made this run
   Estimated GBP spent so far

The HIT/MISS indicator is not optional. It is the centrepiece of the live demo.
Judges watching in real time must be able to see the moment Synapse starts returning
cached paths rather than misses. That moment is when the agent has learned.

---

## Phase Order and Exit Gates

/plan-eng-review before every phase. PLAN ONLY before any code is written.

PHASE A: Synapse HTTP Client (Day 1 morning)
  Files: agent/synapse_client.py, tests/test_synapse_client.py, tests/conftest.py
  Methods:
    async append_thought(agent_id: str, parent_slot: int | None,
                         state_hash: int, success_score: float) -> int
    async get_best_next(agent_id: str, state_hash: int) -> dict | None
    async get_stats(agent_id: str) -> dict
  Exit gate:
    get_stats("0") returns valid JSON from live Synapse container
    append_thought("0", None, 99999, 0.0) returns a thoughtId integer
    get_best_next("0", 99999) returns the thought just appended
    All unit tests pass with mocked httpx responses

PHASE B: State Hash + Spike (Day 1 afternoon)
  Files: agent/state_hash.py, tests/test_state_hash.py
         benchmarks/questions.json (written manually, SQL verified against northwind.db)
         benchmarks/hash_spike.py (prints collision matrix for all 60 questions)
  Exit gate:
    hash_state() deterministic: same input always returns same output
    hash_state() always returns positive int in range [0, 0x7FFFFFFF]
    hash_spike.py output: within-family > 60%, cross-family < 10%
  DO NOT proceed to Phase C without both conditions passing.
  Fix intent classification and table extraction first.

PHASE C: Cognee Memory Bridge (Day 2 morning)
  Files: agent/memory_bridge.py, tests/test_memory_bridge.py
  Methods:
    async remember_schema(schema_text: str) -> None
    async remember_success(question: str, sql: str, epoch: int) -> None
    async recall_context(question: str) -> str
    async forget_failures(failed_state_hashes: list[int], epochs: list[int]) -> None
  Exit gate:
    remember_schema() stores to dataset "northwind_schema" without error
    remember_success() stores to dataset "episode_epoch_{n}" without error
    recall_context() returns non-empty string containing schema facts
    forget() with dataset "episode_epoch_1" removes ONLY that dataset
    CRITICAL: if forget() cannot target named datasets, stop and flag immediately
              Do not proceed until this is resolved. The forget() integration
              depends entirely on named dataset targeting working correctly.

PHASE D: SQL Agent + Prompt Builder (Day 2 afternoon)
  Files: agent/sql_agent.py, agent/prompt_builder.py, tests/test_sql_agent.py
  Prerequisites: northwind.db exists, questions.json has 10+ verified questions
  Agent flow per question (single-shot):
    hash_state(context) -> state_hash
    memory_bridge.recall_context(question) -> schema and episodic context
    synapse_client.get_best_next("0", state_hash) -> procedural hint or None
    prompt_builder.build_prompt(question, context, hint) -> prompt string
    anthropic_client.messages.create(prompt) -> sql string
    sqlite3.execute(sql, northwind.db) -> result
    compare result to gold_answer -> correct: bool
    synapse_client.append_thought("0", None, state_hash, 1.0 if correct else 0.0)
    if correct: memory_bridge.remember_success(question, sql, epoch)
  Exit gate:
    3 questions complete end-to-end without exceptions
    SQL executes against northwind.db without error (wrong answer is fine in Phase D)
    Synapse writeHead advances after each question
    Vanilla mode produces lower accuracy than memory mode on the same 3 questions

PHASE E: Benchmark Runner (Day 3)
  Files: benchmarks/runner.py, benchmarks/plot.py
  Run config:
    50 training questions, 10 hold-out questions
    10 epochs on training set = 500 LLM calls for memory agent
    Same 10 epochs for vanilla agent = 500 more LLM calls
    Total: ~1000 calls, ~5 GBP
    Epoch 5 checkpoint: call forget_failures() with state hashes where failed_count > 3
    Log per-epoch accuracy to results/benchmark_{timestamp}.json
    Log every forget() call: timestamp, dataset_name, reason
  Exit gate:
    Memory agent epoch 10 training accuracy > epoch 1 by 20+ percentage points
    Vanilla baseline flat across all epochs (less than 5 point change)
    Hold-out improvement > 10 percentage points (proves generalisation)
    plot.py generates results/learning_curve.png with both lines clearly labelled
  If flat, debug in this order:
    1. Check recall hit rate in runner logs (if below 20%, state hash too granular)
    2. Check prompt injection: add logging to prompt_builder to verify context appears
    3. Check reinforcement: verify thoughtId returned and success_score correct
    4. Verify Cognee remember() storing correctly (check Cognee logs)
  DO NOT fake results. A real flat curve with honest diagnosis is better than fabricated numbers.

PHASE F: Dashboard (Day 4 morning)
  Files: dashboard/app.py
  Exit gate:
    uv run streamlit run dashboard/app.py opens in browser without errors
    All four panels visible and populated from a previous benchmark run
    HIT/MISS indicator updates correctly in a live single-question test

PHASE G: Demo, README, Blog (Day 4 to 5)
  Files: demo.py, README.md, blog_post.md
  demo.py:
    Run 1: broken hash, show flat curve, explain why
    Run 2: real hash, show climbing curve
    Opens dashboard at end
    Total runtime under 15 minutes
  README.md:
    5-step quickstart (clone, install, start Docker, run demo)
    Real benchmark numbers from Phase E run (not estimated)
    ASCII architecture diagram showing all three memory types
    "Why not just a vector database?" section (see answer below)
    Actual cost per full benchmark run in GBP
    Links to GitHub, Cognee, Synapse-DB repo
  blog_post.md:
    Title: "Why AI agents need three types of memory (and how we built all of them)"
    Sections: The problem, The brain analogy, Cognee for two types,
              Synapse for the third, The forget() insight, The benchmark numbers,
              What comes next
    Submit to dev.to on Day 6, Show HN on Day 7

---

## Synapse REST API

ONLY call from agent/synapse_client.py. Never from anywhere else.

POST /api/v1/agents/0/thoughts
  Header: X-Api-Key: sk_syn_hackathon2026
  Body:   { parentSlot: int|null, stateHash: int, successScore: float }
  Return: { thoughtId: int }

GET /api/v1/agents/0/thoughts/best-next?stateHash={int}
  Header: X-Api-Key: sk_syn_hackathon2026
  Return: thought dict, or 404 if no history for this state hash

GET /api/v1/agents/0/memory/stats
  Header: X-Api-Key: sk_syn_hackathon2026
  Return: { fillPercent: float, writeHead: int, sessionCount: int }

---

## Cognee API

ONLY call from agent/memory_bridge.py. Never from anywhere else.
Use v1.0 API only.

cognee.remember(text: str, dataset_name: str)   # MUST pass dataset_name
cognee.recall(query: str) -> list               # semantic + episodic combined
cognee.forget(dataset_name: str)                # prune named dataset
cognee.improve()                                # self-optimise graph (after each epoch)

Do NOT use: cognee.add(), cognee.cognify(), cognee.search() (deprecated v0 API)

---

## Code Philosophy

Write code that reads like prose.
If a line needs a comment to be understood, rename or extract a function instead.

ACCEPTABLE comments:
  # 31-bit mask: Synapse stateHash is a signed Java int, must be positive
  # dataset_name required: forget() cannot target individual records, only datasets
  # attempt_count > 3: distinguishes untried paths from confidently-wrong ones
  # single-shot by design: multi-step exceeds API budget for 500 questions

UNACCEPTABLE comments:
  # Call the Synapse API
  # Check if result is None
  # Return the hash value
  # Loop through questions
  # Initialize the client
  # Step 1: ...
  # Step 2: ...

Rules:
  Docstrings on public methods only. One sentence. What it does, not how.
  No docstrings on private methods (underscore prefix).
  Type annotations on all function signatures without exception.
  No inline comments on code that reads clearly from its naming.
  No header comments like Step 1, Initialize, Helper function, Return result.
  If you are about to write a comment explaining what a line does,
  rename the variable or extract a function instead.

---

## pyproject.toml Dependencies

[project.dependencies]
httpx = ">=0.27"
anthropic = ">=0.30"
cognee = ">=1.2.2"
fastembed = ">=0.3"
streamlit = ">=1.35"
matplotlib = ">=3.9"
python-dotenv = ">=1.0"

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]

---

## What NOT To Do

DO NOT write comments that restate what the variable name or function name says.
DO NOT write docstrings on private methods.
DO NOT write header comments of any kind.

DO NOT modify Synapse-DB Docker container or its source code.
DO NOT call Synapse REST API from anywhere except agent/synapse_client.py.
DO NOT call Cognee from anywhere except agent/memory_bridge.py.
DO NOT use Cognee v0 API: add(), cognify(), search().
DO NOT use plain pip. Always: uv pip install.
DO NOT use OpenAI. Only claude-sonnet-4-6 via anthropic SDK.
DO NOT call cognee.remember() without dataset_name.
DO NOT proceed to Phase C without Phase B exit gate passing.
DO NOT proceed to Phase E without Phase C forget() verified with named datasets.
DO NOT run the benchmark without the vanilla baseline running in parallel.
DO NOT fake benchmark results. Actual numbers only.
DO NOT add features not in this CLAUDE.md without flagging first.

---

## gstack Workflow

/plan-eng-review  -> Opus | before each phase | PLAN ONLY, no code, no files
implement         -> Sonnet
/review           -> Opus | after each phase | explicitly flag unnecessary comments in diff
/ship             -> manual: uv run pytest -> git add -> git commit -> git push
test command:        uv run pytest
no npm, no node, no TypeScript build steps

After /review: if Opus flags comments that restate the code, remove them before /ship.
Code review is the comment cleanup gate.

---

## Environment Variables

SYNAPSE_URL=http://localhost:8080
SYNAPSE_API_KEY=sk_syn_hackathon2026
ANTHROPIC_API_KEY=sk-ant-...
LLM_API_KEY=sk-ant-...
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-6
EMBEDDING_PROVIDER=fastembed
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

---

## First Claude Code Session

Open Claude Code in cognee-synapse-agent/ and paste exactly this:

  Read CLAUDE.md fully before responding.

  Phase A: Build agent/synapse_client.py

  Three async methods using httpx:
    append_thought(agent_id: str, parent_slot: int | None, state_hash: int,
                   success_score: float) -> int
    get_best_next(agent_id: str, state_hash: int) -> dict | None
    get_stats(agent_id: str) -> dict

  Load SYNAPSE_URL and SYNAPSE_API_KEY from .env via python-dotenv.
  All calls to localhost:8080 with X-Api-Key header.
  get_best_next returns None on HTTP 404.

  Also write:
    tests/test_synapse_client.py: unit tests with mocked httpx responses
    tests/conftest.py: live_synapse fixture that skips if Synapse not reachable

  Synapse is running at http://localhost:8080. Agent ID is always 0.

  /plan-eng-review first. PLAN ONLY. No code, no files created yet.

---

## One-Sentence Answer for Judges

"Why not just use a vector database?"

Know this without hesitation:
"A vector database finds the most semantically similar past question and returns
that SQL. Synapse tracks which SQL approach works for which question TYPE across
hundreds of attempts, with Hebbian reinforcement so successful patterns strengthen
and failed ones decay. It learns how to think about a class of problem, not just
what answer worked once for a similar question."

---

## Why This Wins

Potential Impact:     Memory that improves automatically with zero ML engineering
Creativity:           First combination of all three human memory types in one agent
Technical Excellence: Synapse infrastructure plus clean Python layer plus real benchmark
Best Use of Cognee:   All four verbs used, forget() driven by Synapse Hebbian signal
User Experience:      Dashboard shows memory working in real time via HIT/MISS indicator
Presentation:         Two-line chart, failure mode demo, cost transparency, blog post

The submission in one sentence:
"We gave AI agents the three memory types the brain uses, drove all four Cognee
verbs with a principled architecture, and proved it works with a 25-30 percentage
point accuracy improvement over a memoryless baseline using the same model."