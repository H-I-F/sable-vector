[English](hybrid-retrieval.md) | [简体中文](hybrid-retrieval.zh-CN.md)

# Recipe: tenant-scoped hybrid retrieval

> A question that comes up quickly once a collection has more than one owner:
> "search the whole thing, but only within this tenant, and only recent records".
> This recipe builds that query on the library API, shows what it costs, and
> lists the three ways it can quietly give you the wrong answer. All snippets
> come from code that ships in this repository; the demo is self-asserting.

## The short version

Declare the fields you will filter on **at open time**, write them into the
payload, then resolve the predicate before scoring:

```python
import numpy as np
from sable_vector.engine import VectorEngine
from sable_vector.query.prefilter import PrefilterExecutor
from sable_vector.index.scalar import ScalarIndex
from sable_vector.index.flat import FlatIndex

engine = VectorEngine.open(
    "collection.sv",
    metric="cosine",
    indexed_fields=("tenant", "lang", "year"),
)

engine.upsert("doc-1", np.random.randn(1536).astype("float32"),
              payload={"tenant": "acme", "lang": "en", "year": 2024})
engine.upsert("doc-2", np.random.randn(1536).astype("float32"),
              payload={"tenant": "acme", "lang": "en", "year": 2025})
engine.upsert("doc-3", np.random.randn(1536).astype("float32"),
              payload={"tenant": "globex", "lang": "en", "year": 2025})

flat = FlatIndex.materialise(engine.storage, metric="cosine", dim=engine.dim)
scalar = ScalarIndex(engine.storage, ("tenant", "lang", "year"))
executor = PrefilterExecutor(engine.storage, flat, scalar)

outcome = executor.search(query, k=5,
                          predicate={"tenant": "acme", "year": {"$gte": 2025}})
print(outcome.candidates, outcome.scored, len(outcome.matches))
```

`outcome.scored` is the number to watch. With three rows it is not dramatic; with
50,000 rows and a selective tenant it is the difference between scoring 50,000
vectors and scoring 400.

## Why `indexed_fields` is a write-time decision

The inverted index is not schema. `ScalarIndex.reindex()` writes `idx:` entries
only for the fields listed in `VectorEngine.open(indexed_fields=...)`; a field you
did not declare leaves no `idx:` key at all. `matching_ids()` therefore raises
`ValueError` for an undeclared field rather than returning a partial answer:

```
ValueError: field 'owner' is not indexed; indexed fields are ['tenant', 'lang', 'year']
```

Adding a field later requires rebuilding the existing corpus — the entries do
not exist on disk for the rows already written. Decide the filterable attributes
before the first ingest.

## The three ways this goes wrong

| Symptom | Cause | What to do |
| --- | --- | --- |
| Filtered search misses rows that plainly match | They were written before the field was declared, so no `idx:` entry exists | Rebuild the corpus, or use post-filtering |
| `ValueError` on a field that exists in the payload | The field was not declared in `indexed_fields` | Declare it and rebuild |
| Fewer than `k` rows returned even though matches exist | You are on the post-filter path and the global top-N was full of rows that fail the predicate | Check `outcome.complete`; widen `examine_budget`, or switch to pre-filtering |

The first two are specific to pre-filtering. [ADR-003](../decisions/adr-003-dual-filter-strategies.md)
documents them as the price of the cheaper cost curve. Post-filtering has no blind
spot — it reads the payload itself — but can return fewer than `k` and always
scores the whole corpus.

## Reading the cost, not just the result

| Field | `PrefilterOutcome` | `PostfilterOutcome` |
| --- | --- | --- |
| `candidates` / `scored` | ids admitted / rows scored — should be much smaller than corpus size | — / always corpus size |
| `examined` | — | payload reads; roughly `k` for a broad predicate, deep for a selective one |
| `complete` | — | `False` means the ranking (or your budget) ran out before `k` survivors |

A predicate that is broad — say `{"year": {"$gte": 2000}}` — is a bad fit for
pre-filtering: the candidate set is nearly the whole corpus, so you paid for an
index lookup and gained nothing. It is a good fit for post-filtering, which stops
after about `k` payload reads. `tests/test_postfilter.py` asserts exactly that
(`examined == 5` for `k=5`).

Operators are the usual set — `$eq`, `$ne`, `$in`, `$nin`, `$gt`, `$gte`, `$lt`,
`$lte` — and a bare scalar means equality. Numeric values are rendered as
fixed-width text by `render_value()`, so `$gte` works as a range rather than a
lexicographic surprise.

## The offline demo, already in the repository

[`examples/langchain_rag_demo.py`](../../examples/langchain_rag_demo.py) runs the
whole thing with no model download and no network call. Its embedder is a signed
hashing trick — not semantic, but deterministic — and it prints `scored` for three
searches: unfiltered, filtered to one tenant with a year range, and filtered on a
field that was never indexed. The third one exercises the fallback: it still
returns the right documents, costs a full ranking, and reports the mode instead
of raising.

```bash
python examples/langchain_rag_demo.py
```

It asserts its own claim at the end — `narrow >= baseline` is a failure — so this
is a runnable check that the filter reduced the work, not just the result set:

```python
store.similarity_search(QUESTION, k=3)
baseline = store.last_search.scored
store.similarity_search(QUESTION, k=3, filter={"tenant": "acme"})
narrow = store.last_search.scored
if narrow >= baseline:
    return 1
```

The adapter takes the same predicate shape through `filter=`:

```python
store.similarity_search_with_score(
    QUESTION, k=3, filter={"tenant": "acme", "year": {"$gte": 2025}}
)
```

## Operational notes

- Keep `indexed_fields` in configuration, not scattered across call sites.
  A mismatch between what was declared at ingest and what a query passes is
  silent on the write side and loud on the read side.
- Do not treat a short result list as "no matches". Check `complete` first.
- If a predicate changes shape often (many ad-hoc attributes), post-filtering is
  the more honest default; you cannot declare an index for every attribute.
- Payload reads are per surviving candidate on post-filtering. Under a latency
  SLA, set `examine_budget` and handle `complete=False` in the caller.
- Both executors must return the same rows. If yours disagree, that is a bug —
  compare against `FlatIndex.search()` without a predicate.

## Where to read next

| Concern | Source |
| --- | --- |
| The full mechanism and the deterministic gather | [hybrid query strategies](../skills/hybrid-query-strategies.md) |
| The decision behind two executors | [ADR-003](../decisions/adr-003-dual-filter-strategies.md) |
| Prefix keys and rendered values | [`storage.py`](../../src/sable_vector/storage.py) |
| Offline `VectorStore` adapter | [`adapter/langchain.py`](../../src/sable_vector/adapter/langchain.py) |
| Runnable before/after comparison | [`examples/langchain_rag_demo.py`](../../examples/langchain_rag_demo.py) |
