# sable-vector v0.1.2

A documentation release. 3 November 2026.

The query engine, the key space, the codec and every index are exactly as they
were in 0.1.1. No public API moved, no on-disk format changed, and a collection
written by 0.1.1 opens unchanged. The `sable` dependency is still pinned to
`v0.1.1`: that is the tag carrying the snapshot-read fix this project's own
concurrency tests uncovered, and 0.1.2 changes no behaviour that would justify
moving it.

The version number exists to mark one thing — the engineering notes under `docs/`
are now complete, bilingual, and reachable from one place.

## What is in it

- **Two indexes.** `docs/README.md` and `docs/README.zh-CN.md` list every page,
  with a source map and a reading order that goes from ADR-001 through the case
  study to the benchmark report.
- **Five ADRs in both languages** — the single key space, the payload commit
  record, the two filter strategies, the float32 codec and score orientation, and
  the pinned dependencies.
- **A FAQ, three implementation guides and two recipes**, all bilingual:
  `vector-search-qa`, the `skills/` set on the zero-copy codec, durable upsert and
  hybrid query strategies, and the `recipes/` set on tenant-scoped hybrid
  retrieval and choosing `nprobe`.
- **A measured benchmark report** (`docs/reports/benchmark-2026-11.md`), the
  v0.1.0 and v0.1.1 announcements, and the cross-layer case study — each with a
  Chinese companion. Chinese editions of `CHANGELOG`, `CONTRIBUTING`, `SECURITY`
  and `PROGRESS` round out the root.
- **README wiring.** A Documentation table and a governance list pairing English
  and 简体中文 for every entry.

## What is not in it

No benchmark was re-run. Recall, latency and the `nprobe` figures quoted in the
recipes come straight from the November 2026 report, including the p95 outlier
that was left in rather than smoothed away. Every code snippet in the guides is
taken from `src/sable_vector/`, not written for the sake of the documentation.

The `[Unreleased]` section of the CHANGELOG still holds the deterministic PR
review report and the move to the H.I.F organization. Those are tooling and
governance items with their own timeline; this tag is about documentation.

## Upgrading

```sh
pip install --upgrade "sable-vector[sable,mcp] @ git+https://github.com/H-I-F/sable-vector.git@v0.1.2"
```

- Repository: <https://github.com/H-I-F/sable-vector>
- Release: <https://github.com/H-I-F/sable-vector/releases/tag/v0.1.2>
- Full history: [CHANGELOG.md](../CHANGELOG.md) · [简体中文](../CHANGELOG.zh-CN.md)
- Engineering notes: [index](README.md) · [简体中文](README.zh-CN.md)
