# Why AI agents need three types of memory (and how I built all of them)

Most "agent memory" today is one thing wearing three hats: a vector database.
You embed the past, you retrieve the nearest neighbor, you paste it into the
prompt. It works until it doesn't, and when it doesn't, you can't tell whether
the agent forgot a fact, forgot an experience, or never learned the skill.

The human brain does not do this. It keeps three separate systems. I built an
agent that does the same, and measured what it buys you.

## The problem

I gave a language model a database it had never seen (Northwind) and asked it to
write SQL. Cold, with no help, a strong model gets it right about a quarter of the
time. Not because it can't write SQL, but because it doesn't know the schema,
doesn't remember what worked last time, and has no sense of how to approach a
question of a given shape. Those are three different kinds of not-knowing, and a
single vector store treats them as one.

## The brain analogy

Cognitive science splits long-term memory into three kinds:

- **Semantic memory** is what things mean. The capital of France. That
  `Orders.CustomerID` is a foreign key to `Customers`.
- **Episodic memory** is what happened to you. The query you ran last Tuesday that
  actually returned the right count.
- **Procedural memory** is how to do things. Riding a bike. The fact that for an
  aggregate-over-a-join you filter before you count.

You do not retrieve "how to ride a bike" by finding the most similar past bike
ride. Procedural memory is a different mechanism. That is the insight the whole
project is built on.

## Cognee for two of them

Cognee gives me a knowledge graph with four verbs: `remember`, `recall`,
`forget`, `improve`. I use it for the first two memory types.

**Semantic:** before the benchmark runs, I load the Northwind schema once.
Tables, columns, foreign keys. It is never forgotten because the schema never
changes.

**Episodic:** every time the agent gets an answer right, I store the
question-and-SQL pair as an episode, tagged with a dataset name that encodes the
question type. Later, `recall` surfaces the ones that match the current question.

One hard-won detail: Cognee's graph extraction normalizes column names to
snake_case during ingestion, which quietly poisons a schema full of PascalCase
Northwind columns. The fix was to keep the raw schema text I fed in and inject
that directly, using the graph only for episode retrieval. If you take one
practical thing from this post, take that.

## Synapse for the third

Procedural memory needed a different engine, so I used Synapse-DB. Every question
gets hashed to a bucket by its **type** (its intent plus the tables it touches),
not its wording. Each attempt writes a thought with a success score: 1.0 for
right, 0.0 for wrong. Synapse reinforces the winners and lets the losers decay,
Hebbian-style.

The per-type memory that a vector store cannot give you lives in how those buckets
key the episodes: a successful query is filed under its type's hash, and recall
returns past queries of the *same type*, not merely the nearest neighbor by
wording. That is how the agent accumulates, across hundreds of attempts, a sense
of how to approach a *type* of problem rather than a single lookalike answer.

One honest caveat I found while building the demo: in the Synapse build I used,
the `best-next` lookup returns a global salience signal and does not filter by
state hash (an unseen hash returns the same thing as a heavily-used one). So the
per-type differentiation is carried by the hash-keyed episodic recall, and Synapse
plays the role of the global reinforcement-and-decay signal. That signal is what
drives the `forget` step below, and it is real. I would rather tell you exactly
where each memory type is doing its work than sell a cleaner story than the one I
measured.

## The forget() insight

Here is the piece no other approach has. Early in training, the agent sometimes
gets a question right by luck and stores a misleading episode. That bad memory
then pollutes recall for every similar question.

I let Synapse decide what to forget. At a mid-run checkpoint, any question bucket
that Synapse has watched fail more than three times gets its early episodes
pruned from Cognee via `forget`. The procedural signal cleans the episodic store.
The agent self-heals without a human in the loop. That is all four Cognee verbs,
driven by a principled architecture instead of a cron job.

## The benchmark numbers

I ran two agents on the same model, `claude-haiku-4-5`, over 50 training and 10
hold-out questions.

- **Vanilla, no memory: 26%.**
- **Memory agent: 58%.**
- **Gap: +32 percentage points**, for a total API cost of £0.465.

I'm publishing the honest version. The run completed 8 of 10 epochs before a
network blip killed it. Learning plateaued at 58% from epoch 3 rather than
climbing to the 70s, because the hardest JOIN questions failed on the first pass
and so never seeded an episode to learn from. Hold-out stayed flat. None of that
touches the headline: the memory layer, and nothing else, moved the same model
from 26% to 58%.

## What comes next

Three fixes, in order of impact. Seed the hard buckets with a handful of correct
episodes so the plateau breaks. Add a retry around the API call so a dropped
packet doesn't cost you an overnight run. And persist Synapse state across runs so
the procedural memory compounds over weeks, not epochs. The architecture is the
contribution. The numbers say it works. The next numbers say how far it goes.
