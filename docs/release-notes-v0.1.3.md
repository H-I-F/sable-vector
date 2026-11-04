# sable-vector v0.1.3

A small release on the MCP surface. 4 November 2026.

The three tools exposed by `serve-mcp` now say what they do before a host asks
anyone to approve a call. Each one declares `readOnlyHint`, `destructiveHint`,
`idempotentHint` and `openWorldHint` explicitly. Search and describe are
read-only and idempotent; upsert replaces an existing id, which makes it a
destructive but logically idempotent write. All three work on a local collection
directory, so `openWorldHint` is false throughout.

Leaving a hint out hands the SDK's default to the client, and three of those
defaults land on the unsafe side. A contract test now pins the wire names and
all four boolean values, so a renamed tool or a dropped hint fails the suite.

## What is in it

- **Four explicit behaviour hints on each of the three MCP tools.** A host can
  tell a read from a write without guessing. `ToolAnnotations` exists across the
  whole `mcp>=1.9,<2` range, so the dependency pin did not move.
- **A contract test** asserting the advertised tool set and all four hints on
  every tool, keyed by wire name rather than an internal constant.
- **The CHANGELOG backlog cleared.** The deterministic PR review report and the
  move to the H.I.F organization had been sitting under `[Unreleased]` since
  v0.1.2; they now sit in the version history where they belong.

## What is not in it

No engine change. The key space, index format, public API, dependency pins and
query behaviour are exactly as they were in 0.1.2 — a collection written by
0.1.2 opens unchanged under 0.1.3. The hints are advisory metadata on the
advertised tool list; they change what a host can know up front, not what a tool
does. The `sable` dependency stays pinned to `v0.1.1`.

No benchmark was re-run, and no guide was rewritten: nothing they describe
changed.

## Upgrading

```sh
pip install --upgrade "sable-vector[sable,mcp] @ git+https://github.com/H-I-F/sable-vector.git@v0.1.3"
```

- Repository: <https://github.com/H-I-F/sable-vector>
- Release: <https://github.com/H-I-F/sable-vector/releases/tag/v0.1.3>
- Full history: [CHANGELOG.md](../CHANGELOG.md) · [简体中文](../CHANGELOG.zh-CN.md)
- Engineering notes: [index](README.md) · [简体中文](README.zh-CN.md)
