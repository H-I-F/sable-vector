# sable-vector v0.1.4

A maintenance release. 11 November 2026.

Nothing in the library changes. The engine, key space, index format, public API,
query behaviour and every dependency pin are exactly as they were in 0.1.3 — a
collection written by 0.1.3 opens unchanged under 0.1.4. The `mcp<2` pin and
the `sable @ git+...@v0.1.1` direct reference are where they were.

## What is in it

This version records a change to how the repository looks after itself.

- **Dependency updates merge on their own.** A `workflow_run` workflow arms
  `auto-merge` on a Dependabot pull request once CI is green. Branch protection
  still decides when the merge lands: required checks, a linear history and
  resolved conversations all have to pass first, so a failing build never
  reaches `main`. It runs on the default branch, needs no extra token or
  third-party action, and only touches pull requests opened by Dependabot.
- **Dependabot scans monthly, in groups.** The pip, GitHub Actions and Docker
  ecosystems moved off the weekly schedule, and each now groups into a single
  pull request. The `mcp>=2` ignore is unchanged — Dependabot still leaves that
  family alone, because a 2.x release breaks the `@server.list_tools` API.
- **Automated security fixes are on.** Slowing routine updates down does not
  slow down a reported vulnerability, which still gets its own fix pull request.

## What gets a release from now on

Changes that do not touch the engine — dependency bumps, CI and workflow edits,
bot configuration, documentation — now land through the auto-merge path without
a manual review, and no longer get a release of their own. A tag is reserved for
a change to the core logic, or another change large enough that someone
installing the package should read about it.

This is the version that puts that rule on the record. It is the last release
made for a maintenance-only change; after it the version number moves only when
the engine does.

## What is not in it

No engine change, no new runtime dependency, no format change and no benchmark
re-run. Installing from this tag gives the same library as 0.1.3 with a
different version string. No guide was rewritten: nothing they describe changed.

## Upgrading

```sh
pip install --upgrade "sable-vector[sable,mcp] @ git+https://github.com/H-I-F/sable-vector.git@v0.1.4"
```

- Repository: <https://github.com/H-I-F/sable-vector>
- Release: <https://github.com/H-I-F/sable-vector/releases/tag/v0.1.4>
- Full history: [CHANGELOG.md](../CHANGELOG.md) · [简体中文](../CHANGELOG.zh-CN.md)
- Engineering notes: [index](README.md) · [简体中文](README.zh-CN.md)
