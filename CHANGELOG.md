# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from 1.0.0
onward. The 0.1.x line is the initial, pre-stable API surface.

## [Unreleased]

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

[Unreleased]: https://github.com/MKA930/sable-vector/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/MKA930/sable-vector/releases/tag/v0.1.0
