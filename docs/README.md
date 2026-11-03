[English](README.md) | [简体中文](README.zh-CN.md)

# sable-vector — Engineering notes

> The [project README](../README.md) is for people using the library. This directory
> is for people who want to know **why** a record is laid out the way it is, what a
> query strategy actually costs, and which constraints were chosen on purpose.
> Every claim below points at a real symbol in `src/sable_vector/`.

## What lives here

| Section | For | Format |
| --- | --- | --- |
| [Release notes](#release-notes) | upgrading between versions | one announcement per tag, bilingual |
| [Case studies](#case-studies) | post-mortems, design review | narrative with the reproduction |
| [Decision records](#architecture-decision-records) | anyone changing the key space or write contract | ADR-001 … ADR-005, bilingual |
| [Benchmarks](#benchmarks) | picking an operating point | runnable harness + measured report |
| [Source map](#source-map) | finding the code behind a concept | module table |

## Release notes

- [v0.1.0](RELEASE_NOTES_v0.1.0.md) / [简体中文](RELEASE_NOTES_v0.1.0.zh-CN.md) — first tagged
  release: durable hybrid retrieval, pre/post filtering, IVF, offline adapter.
- [v0.1.1](RELEASE_NOTES_v0.1.1.md) / [简体中文](RELEASE_NOTES_v0.1.1.zh-CN.md) — the MCP stdio
  server, the `mcp<2` pin, and the quickstart corrections.

## Case studies

- [Cross-layer bug hunt](case-studies/cross-layer-bugfix.md) /
  [简体中文](case-studies/cross-layer-bugfix.zh-CN.md) — two deterministic test failures in this
  project traced to a `sable` memtable that disagreed with its own range scans about what a pinned
  sequence resolves to. Includes the pure-upstream reproduction and why it could not be fixed here.

## Architecture Decision Records

Five decisions behind the on-disk and in-memory contracts, each in
**Status / Context / Decision / Consequences** form. Read them before changing a key
prefix, a commit boundary or a dependency pin.

- [ADR-001: One key space, separated by prefix](decisions/adr-001-single-keyspace.md) /
  [简体中文](decisions/adr-001-single-keyspace.zh-CN.md) — vectors, payloads, inverted entries and
  IVF state share one `sable` tree, so there is one WAL and one recovery story.
- [ADR-002: The payload is the commit record](decisions/adr-002-payload-commit-record.md) /
  [简体中文](decisions/adr-002-payload-commit-record.zh-CN.md) — stage every other key with
  `sync=False`, flush once on the payload.
- [ADR-003: Ship both filter strategies](decisions/adr-003-dual-filter-strategies.md) /
  [简体中文](decisions/adr-003-dual-filter-strategies.zh-CN.md) — pre- and post-filtering return the
  same rows with different cost curves and different failure modes.
- [ADR-004: float32 zero-copy images, higher-is-closer scores](decisions/adr-004-float32-codec-and-score-orientation.md) /
  [简体中文](decisions/adr-004-float32-codec-and-score-orientation.zh-CN.md) — no framing bytes, no
  Python row loop, one score orientation for all three metrics.
- [ADR-005: Pin the dependencies that define the API](decisions/adr-005-pinned-dependencies.md) /
  [简体中文](decisions/adr-005-pinned-dependencies.zh-CN.md) — `mcp<2` and the `sable@v0.1.1`
  direct reference, and why Dependabot is told to leave them alone.

## Benchmarks

[`benchmarks/bench_vector.py`](../benchmarks/bench_vector.py) is self-asserting: the resident
`FlatIndex` is exhaustive and defines recall = 1.0, so every approximate number is measured against
it rather than assumed. Run it with `python -m benchmarks.bench_vector --dim 1536 --sizes 10000`.

- [Measured report](reports/benchmark-2026-11.md) /
  [简体中文](reports/benchmark-2026-11.zh-CN.md) — latency, throughput and the recall/`nprobe` curve
  on two corpus sizes, with the host and the parts the run does not cover stated plainly.

## Source map

| Module | Responsibility | Key symbols |
| --- | --- | --- |
| [`types.py`](../src/sable_vector/types.py) | ids, metrics, records, outcomes | `VectorId`, `DistanceMetric`, `VectorRecord`, `Match` |
| [`codec.py`](../src/sable_vector/codec.py) | float32 ↔ bytes, payload JSON | `encode_vector`, `decode_vector`, `decode_matrix` |
| [`distance.py`](../src/sable_vector/distance.py) | NumPy kernels, score orientation, top-k | `score_matrix`, `top_k`, `normalise_matrix` |
| [`storage.py`](../src/sable_vector/storage.py) | prefixed namespace adapter | `VectorStorage`, `prefix_upper_bound`, `escape_segment` |
| [`engine.py`](../src/sable_vector/engine.py) | durable upsert / delete / reclaim | `VectorEngine`, `RepairReport` |
| [`recovery.py`](../src/sable_vector/recovery.py) | crash rebuild, index health | `recover`, `rebuild_ivf`, `IndexHealth` |
| [`index/flat.py`](../src/sable_vector/index/flat.py) | exhaustive matrix search | `FlatIndex`, `search_streaming` |
| [`index/scalar.py`](../src/sable_vector/index/scalar.py) | inverted prefix index | `ScalarIndex`, `matching_ids` |
| [`index/ivf_cluster.py`](../src/sable_vector/index/ivf_cluster.py) | centroid training | `kmeans`, `assign`, `residuals` |
| [`index/ivf_storage.py`](../src/sable_vector/index/ivf_storage.py) | posting lists, `nprobe` pruning | `IvfIndex`, `IvfOutcome` |
| [`query/prefilter.py`](../src/sable_vector/query/prefilter.py) | predicate first | `PrefilterExecutor`, `PrefilterOutcome` |
| [`query/postfilter.py`](../src/sable_vector/query/postfilter.py) | geometry first | `PostfilterExecutor`, `PostfilterOutcome` |
| [`adapter/langchain.py`](../src/sable_vector/adapter/langchain.py) | offline `VectorStore` | `SableVectorStore` |
| [`adapter/mcp_server.py`](../src/sable_vector/adapter/mcp_server.py) | stdio MCP tools | `list_tools`, `call_tool` |
| [`cli.py`](../src/sable_vector/cli.py) | `serve-mcp` and CLI entry | `app` |

## Where to start

1. [`README.md`](../README.md) — the one-page mental model and the three Mermaid diagrams.
2. [ADR-001](decisions/adr-001-single-keyspace.md) then
   [ADR-002](decisions/adr-002-payload-commit-record.md) — the two decisions every durability claim
   rests on.
3. [ADR-003](decisions/adr-003-dual-filter-strategies.md) — why there are two executors instead of
   one clever one.
4. [Case study](case-studies/cross-layer-bugfix.md) — what happens when an upstream invariant is
   only implemented on one read path.
5. [Benchmark report](reports/benchmark-2026-11.md) — before choosing an `nprobe`.

---

[← Back to project README](../README.md)
