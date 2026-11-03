[English](hybrid-query-strategies.md) | [简体中文](hybrid-query-strategies.zh-CN.md)

# Hybrid query strategies: two executors, two cost curves

> Both executors answer the same `search(query, k, predicate)` signature and
> promise the same `Match` list. They are not interchangeable. One puts the
> predicate before the geometry, the other after it, and that single choice
> decides what the query costs and how it is allowed to fail. [ADR-003](../decisions/adr-003-dual-filter-strategies.md)
> keeps both on purpose.

## 1. The shared contract

```python
PrefilterExecutor(storage, flat, scalar).search(query, k, predicate=...)
PostfilterExecutor(storage, flat).search(query, k, predicate=..., examine_budget=...)
```

Both attach payloads to their matches and return best-first rankings. The
difference is visible in their outcome objects before you look at a single row:

| | `PrefilterOutcome` | `PostfilterOutcome` |
| --- | --- | --- |
| `candidates` | ids that survived the predicate, or `None` when unrestricted | — |
| `scored` | rows handed to the kernel | always the whole corpus |
| `examined` | — | payload reads performed |
| `complete` | — | whether `k` survivors were found |

`scored` is the number that says whether pre-filtering paid off. `examined` is the
number that says how far a post-filtered walk had to go.

## 2. Pre-filtering: resolve the predicate into a candidate set

```python
def search(self, query, k, *, predicate=None) -> PrefilterOutcome:
    allowed = self._scalar.matching_ids(predicate)
    candidates = None if allowed is None else len(allowed)
    matches = self._flat.search(query, k, candidate_ids=allowed)
    scored = candidates if candidates is not None else self._flat.size
    return PrefilterOutcome(matches=tuple(self._with_payloads(matches)),
                            candidates=candidates, scored=scored)
```

`ScalarIndex.matching_ids()` does the work:

```python
if not predicate:
    return None                       # "no restriction", distinct from an empty set
for field in predicate:
    if field not in self._fields:
        raise ValueError(f"field {field!r} is not indexed; indexed fields are ...")
ordered = sorted(predicate.items(),
                 key=lambda item: len(_probe_values(item[1]) or []))
result = None
for field, condition in ordered:
    ids = self.field_ids(field, condition)
    result = ids if result is None else (result & ids)
    if not result:
        return set()
return result
```

Three things are worth reading slowly:

- **`None` vs empty set.** An empty predicate means "no restriction at all", so
  `candidate_ids=None` and the executor degenerates to a plain flat search. A
  predicate that matched nothing is `set()` — an empty candidate block, zero rows
  scored. Collapsing these two would silently turn "no filter" into "return
  nothing".
- **Most selective field first.** `_probe_values` estimates how many index
  lookups a condition needs; sorting by it makes the intersection start small.
  An early empty result short-circuits the rest.
- **Undeclared fields raise.** Indexed fields are metadata on the write path,
  fixed by `VectorEngine.open(indexed_fields=...)`. A predicate naming a field
  that was never declared cannot be answered by the inverted index, so it fails
  loudly instead of quietly returning an incomplete answer.

### Predicates are rendered text

`render_value()` makes values sortable, so an equality probe is one prefix scan
and a range is one contiguous scan:

| Input | Rendered |
| --- | --- |
| `True` / `False` | `true` / `false` |
| numbers | `f"{value:020.6f}"` — lexical order matches numeric order |
| strings | UTF-8 as-is |

`evaluate_condition()` understands `FILTER_OPERATORS`: `$eq`, `$ne`, `$in`,
`$nin`, `$gt`, `$gte`, `$lt`, `$lte`. A plain scalar is equality. An expression
with an unknown operator raises `ValueError` **before any data is read**, so a
malformed query is rejected even against an empty field. In the demo this is
`{"tenant": "acme", "year": {"$gte": 2025}}`.

### Why the candidate block is sorted before gathering

A Python set of ids iterates in hash order, randomised per process. Gathering
rows in that order produces a different matrix block and therefore different
floating-point rounding in the gemv; two candidates with exactly equal scores
would swap places depending on the hash seed. `FlatIndex.search()` sorts row
positions, then does one gather:

```python
rows = sorted(self._positions[label] for label in candidate_ids
              if label in self._positions)
grid = self._matrix[rows]
```

It is a gather, not a mask — the candidate block goes to the same kernel as the
unfiltered path. Determinism costs one sort over the candidate list.

## 3. Post-filtering: rank everything, then read payloads

```python
def search(self, query, k, *, predicate=None, examine_budget=None):
    if not predicate:
        matches = flat.search(query, k)
        return PostfilterOutcome(..., examined=len(matches), scored=flat.size,
                                 complete=len(matches) >= k)
    scores = score_matrix(flat.matrix, probe, flat.metric)
    order = np.argsort(-scores, kind="stable")   # full ordering, not a selection
    limit = len(order) if examine_budget is None else min(len(order), examine_budget)
    accepted = []
    for position in order[:limit]:
        payload = self._storage.get_payload(labels[int(position)]) or {}
        examined += 1
        if not self._satisfies(payload, predicate):
            continue
        accepted.append(Match(...))
        if len(accepted) >= k:
            break
    return PostfilterOutcome(..., examined=examined, scored=flat.size,
                             complete=len(accepted) >= k)
```

A **full** ordering is mandatory here: the walk consumes the ranking from the
top, and a partial sort cannot tell it which row is next. `_satisfies()` evaluates
the same `render_value()` / `evaluate_condition()` rules against the payload, so
the two executors agree on what a predicate means. A field the payload does not
carry cannot satisfy its condition — consistent with the inverted index, where
entries exist only for values that were actually present.

The cost profile is the mirror of pre-filtering. For a permissive predicate, the
survivors sit at the top of the ranking and the walk stops after roughly `k`
payload reads; the inverted index would have been pure overhead.
`tests/test_postfilter.py` asserts `examined == 5` for `k=5` on a broad
predicate. For a selective one, the walk goes deep — `examine_budget` caps it,
and `complete=False` tells the caller the result is a prefix rather than a full
answer. Setting a budget is choosing latency over completeness; the default
(`None`) walks the whole ranking, which is the honest one for correctness.

## 4. The failure modes are different, and both are documented

| | `PrefilterExecutor` | `PostfilterExecutor` |
| --- | --- | --- |
| Blind spot | rows written before a field was declared indexed are invisible | none — it reads the payload itself |
| Can it return fewer than `k`? | only if the candidate set is genuinely smaller | yes, when near-neighbours fail the predicate |
| Worst case | an unindexed field raises | a selective predicate walks most of the corpus |
| Requires | `indexed_fields`, a rebuilt corpus if fields change | nothing beyond the payloads |

[ADR-003](../decisions/adr-003-dual-filter-strategies.md) spells out the
trade: for a selective predicate, narrowing 50k rows to 400 removes 99% of the
arithmetic; for a permissive one, the survivors are already at the top, so
post-filtering stops early. Since both promise the same rows, the test suite
checks their equivalence against the exhaustive flat index — a disagreement is a
bug, not a tuning difference.

## 5. Deterministic race tests for the query path

`tests/test_concurrent.py` is the reference for how to test concurrent reads here,
and its opening paragraph is a useful rule in general: a test that merely runs
threads until something breaks proves nothing about which states are legal, and
a test with a `sleep` where the race should be does worse, because the sleep
hides the interleaving it was meant to expose.

- `threading.Barrier` releases all participants at once, so the interleaving is
  a property of the test.
- Readers run a fixed number of passes; an `Event`, not a timer, gates the
  post-write state.
- A guard records any exception raised inside a thread and breaks the barrier —
  Python otherwise reports it as a warning and lets the test pass for the wrong
  reason.
- The invariant asserted is directional: a snapshot reader never sees a payload
  whose vector is missing, under concurrent upserts **and** deletes; a snapshot is
  frozen; and pins are returned on release. The suite deliberately does not
  mutate the same id from two threads — atomicity here is per record within one
  writer.

[The cross-layer case study](../case-studies/cross-layer-bugfix.md) is what
happens when an upstream invariant is implemented on only one read path: two
failing tests here traced back to a `sable` memtable that disagreed with its own
range scans about what a pinned sequence resolves to.

## 6. Checklist for choosing an executor

1. Is the predicate selective, and is every field it names declared indexed?
   → pre-filter, and check `outcome.scored` actually dropped.
2. Broad predicate, small corpus, or a field added after the corpus was built?
   → post-filter, and inspect `outcome.examined` and `outcome.complete`.
3. Under a latency SLA, set `examine_budget` and handle `complete=False`
   explicitly; do not treat a short list as "no matches".
4. Never add an indexed field without rebuilding the existing corpus.
5. Keep the candidate gather sorted — a set-ordered gather breaks reproducibility.
6. When in doubt, run both against the flat index and compare; that is what the
   test suite does, and it is cheap.

## Where to read next

| Concern | Source |
| --- | --- |
| Predicate → candidate ids | [`index/scalar.py`](../../src/sable_vector/index/scalar.py) |
| Candidate gather and exhaustive kernel | [`index/flat.py`](../../src/sable_vector/index/flat.py) |
| The two executors | [`query/prefilter.py`](../../src/sable_vector/query/prefilter.py), [`query/postfilter.py`](../../src/sable_vector/query/postfilter.py) |
| The decision | [ADR-003](../decisions/adr-003-dual-filter-strategies.md) |
| Equivalence and cost assertions | [`tests/test_prefilter.py`](../../tests/test_prefilter.py), [`tests/test_postfilter.py`](../../tests/test_postfilter.py) |
| A runnable before/after comparison | [hybrid retrieval recipe](../recipes/hybrid-retrieval.md) |
