[English](adr-003-dual-filter-strategies.md) | [简体中文](adr-003-dual-filter-strategies.zh-CN.md)

# ADR-003: Ship both filter strategies

**Status:** Accepted — implemented in `src/sable_vector/query/`.

## Context

A metadata predicate can be very selective (`tenant = acme`) or match almost
everything. One execution order cannot be cheapest for both: scoring a candidate
block costs a matrix–vector product proportional to the block size, while reading
payloads costs one storage read per candidate. Retreating to "retrieve a large buffer
and filter in Python" quietly degrades recall and makes latency depend on how
selective the predicate happens to be.

## Decision

Expose **two executors**, returning the same `Match` list and deliberately *not*
interchangeable:

- `PrefilterExecutor` resolves the predicate first through
  `ScalarIndex.matching_ids()`, then hands the surviving ids to
  `FlatIndex.search(query, k, candidate_ids=...)`. It raises `ValueError` when the
  predicate names a field that was not declared in `VectorEngine.open(indexed_fields=...)`.
- `PostfilterExecutor` computes the global top-N, walks it in ranking order, reads
  each payload, and stops once it has `k` survivors. It needs no index at all —
  the predicate is evaluated against the payload itself.

Both live behind the same `search(query, k, predicate)` signature so the caller
chooses a failure mode, not an API.

## Consequences

For a selective predicate, narrowing 50k rows to 400 removes 99% of the arithmetic.
For a permissive one, the survivors sit at the top of the ranking, so post-filtering
stops after roughly `k` payload reads and the inverted index would have been pure
overhead — `PostfilterOutcome` reports how far down the ranking it had to go.

The failure modes are different and are documented rather than hidden: pre-filtering
is blind to rows written before a field was declared indexed, and post-filtering can
return fewer than `k` because the global top-N may be full of near-neighbours that
fail the predicate. Since both implementations promise the same rows, the test suite
checks their equivalence against the exhaustive flat index; a disagreement is a bug,
not a tuning difference.
