# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from 1.0.0
onward. The 0.1.x line is the initial, pre-stable API surface.

## [Unreleased]

## [0.1.1] — 2026-10-31

Follow-up to the first tagged release, adding the Model Context Protocol surface
and recording the cross-layer bug hunt.

### Added

**Model Context Protocol**

- stdio MCP server (`adapter/mcp_server.py`) exposing three tools over the
  existing public surface — `sable_vector_search` (embedding query with optional
  payload equality filters routed through `PrefilterExecutor`),
  `sable_vector_upsert` (one durable record: vector plus JSON attributes), and
  `sable_vector_describe` (record count, vector width, metric). (#23)
- `sable-vector serve-mcp` CLI command (`cli.py`), registered as a console
  script. MCP is asynchronous and Typer is synchronous, so the transport loop is
  driven by a single `asyncio.run` rather than mixed into the command. (#23)
- `mcp` optional extra and `typer` core dependency. The extra is pinned to `<2`
  because this server uses the `Server.list_tools` / `Server.call_tool`
  decorator API that 2.x replaced with constructor callbacks. (#23)
- The container image, the CI matrix and the contributor guide install the `mcp`
  extra, so the adapter is covered rather than assumed. (#23)

**Documentation**

- `docs/case-studies/cross-layer-bugfix.md` — the cross-layer snapshot-read bug
  hunt, from symptom through pure-`sable` reproduction to the upstream fix.
  Both READMEs now flag it as *"Built on sable — with critical upstream bugfixes
  contributed"*. (#22)

### Fixed

- `FlatIndex.materialise` takes the **storage** (not the engine) and requires
  `dim`, so the quickstart snippet published in both READMEs and the v0.1.0
  notes raised `TypeError` as written. All three occurrences corrected. (#23)
- Unquoted Mermaid node labels containing parentheses broke the architecture
  diagrams on GitHub; both READMEs now quote them.

### Changed

- Version `0.1.0` → `0.1.1`.

## [0.1.0] — 2026-10-31

First tagged release of the embedded hybrid vector engine.

### Added

**Core engine**

- Domain types and NumPy distance kernels — L2 and cosine over float32 vectors
  (`types.py`, `distance.py`). (#01)
- Zero-copy float32 ↔ contiguous-bytes codec for durable vector storage
  (`codec.py`). (#02)
- Physical namespace adapter mapping the vector key space onto the `sable` LSM
  storage engine (`storage.py`). (#03)
- WAL-level atomic multi-key upsert so a payload and its vector land together
  (`engine.py`). (#04)

**Index & query**

- Matrix-vectorised Flat brute-force search over the resident or disk-streamed
  index (`index/flat.py`). (#05)
- Scalar inverted prefix index for metadata filtering (`index/scalar.py`). (#06)
- Pre-filtering and post-filtering executors (`query/prefilter.py`,
  `query/postfilter.py`). (#07, #08)
- IVF clustering and posting-list storage for approximate search
  (`index/ivf_cluster.py`, `index/ivf_storage.py`). (#09, #10)

**Durability & integration**

- Crash/recovery rebuild of in-memory structures from persisted state
  (`recovery.py`). (#11)
- Concurrent-read snapshot isolation; verifies the engine under parallel readers
  (`tests/test_concurrent.py`). (#12)
- Offline LangChain adapter and a runnable RAG demo
  (`adapter/langchain.py`, `examples/langchain_rag_demo.py`). (#13)
- SIGKILL chaos injection test exercising recovery under abrupt termination
  (`tests/test_chaos.py`). (#14)

**Packaging, docs & CI**

- Self-validating container image and bilingual (English / 简体中文) portal
  README (`Dockerfile`, `.dockerignore`, `README.md`, `README.zh-CN.md`). (#15)
- 1536-dimension ANN benchmark matrix
  (`benchmarks/bench_vector.py`). (#16)
- GitHub Actions CI: ruff + pytest across Python 3.10 / 3.11
  (`.github/workflows/ci.yml`). (#17)
- Multi-stage Docker build and a CI `docker` job that builds and self-tests the
  image. (#18)

### Dependencies

- Built on `sable` storage engine, pinned to `v0.1.1` (snapshot-read
  consistency fix that also resolved the concurrent-read regression tests).

[Unreleased]: https://github.com/MKA930/sable-vector/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/MKA930/sable-vector/releases/tag/v0.1.1
[0.1.0]: https://github.com/MKA930/sable-vector/releases/tag/v0.1.0
