# Contributing to sable-vector

sable-vector is an embedded hybrid (dense + scalar-filter) vector engine for
local AI agent runtimes, built on top of the [`sable`](https://github.com/H-I-F/sable)
LSM storage engine. This document describes how to set up a development
environment and the conventions the project follows so a patch lands cleanly.

## Development setup

The project targets **Python 3.10+**. `sable` is not published to PyPI, so it
is installed from its repository tag.

```bash
git clone https://github.com/H-I-F/sable-vector.git
cd sable-vector
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[sable,mcp,test,lint]"
```

The `[sable]` extra pulls the storage engine from its released git tag, `[test]`
brings in `pytest`, and `[lint]` brings in `ruff`. After this, `sable_vector`
imports from the working tree.

## Local verification

Run these before pushing — CI runs the same commands:

```bash
ruff check .
pytest -q
python -m benchmarks.bench_vector --sizes 1000 --queries 50 --no-ivf
```

The container image is self-validating: `docker build -t sable-vector .` then
`docker run --rm sable-vector pytest -q`. The image build is exercised by the CI
`docker` job, so you do not need a local build to contribute.

## Branch and commit conventions

Branches follow a `<type>/<short-slug>` pattern that matches the merge commit
prefix:

| Type       | Example                       |
|------------|-------------------------------|
| `feat`     | `feat/pq-compression`         |
| `fix`      | `fix/seek-overflow`           |
| `docs`     | `docs/portal-readme`          |
| `bench`    | `bench/ann-benchmarks`        |
| `ci`       | `ci/actions`                  |
| `build`    | `build/docker`                |
| `chore`    | `chore/release`               |

Commits use [Conventional Commits](https://www.conventionalcommits.org/). Keep
the subject under 72 characters and explain the *why* in the body for anything
non-obvious. Every commit from this project carries:

```
Co-authored-by: Leisurely Winston <winston_cn@yeah.net>
```

## Pull requests

- Open the PR against `main`.
- Describe the change and, when it adds or changes behaviour, how to verify it.
- Reference the row it closes in `PROGRESS.md` (the project tracks delivered
  work there rather than in issue trackers).
- CI must be green: ruff, pytest on Python 3.10/3.11, and the docker build.

## Project layout

| Path               | Responsibility                                           |
|--------------------|----------------------------------------------------------|
| `src/sable_vector/`| Engine, index (flat / IVF), query, distance, adapter     |
| `benchmarks/`      | ANN benchmark suite (`python -m benchmarks.bench_vector`) |
| `examples/`        | Runnable integration demos (e.g. LangChain RAG)          |
| `tests/`           | pytest suite                                             |

## Code style

- `ruff` is the linter and formatter gate; run `ruff check .` locally.
- Target Python is 3.10 — avoid syntax that is 3.11-only unless guarded.
- No `TODO`/`FIXME` placeholders in merged code; if something is incomplete,
  discuss it in the PR instead.

## Maintenance and releases

Most changes here never touch the engine: dependency bumps, CI and workflow edits,
bot configuration, documentation, dev-tool pins. Those are `routine`. A Dependabot
PR classified `routine` merges on its own once CI is green — you do not need to
review or click merge, and it does not get a release of its own.

The split is computed, not negotiated at review time.
`.github/scripts/change_policy.py` is the single definition: it reads the changed
files and returns one of three tiers — `core` (anything under `src/`, or a runtime
dependency; the only tier that earns a release), `review` (a base image or action
reference jump, which stays with a human), and `routine`. The auto-merge workflow,
the PR report and the release tooling all read the same verdict, so a rule change
happens in exactly one place. An empty or malformed file list is an error, never a
`routine` result.

`policy-check.yml` runs both self-tests plus a release-consistency check on every
push and pull request, and is a required status check. The full convention,
including the steps to cut a release, is in
[RELEASING.md](./RELEASING.md) · [简体中文](./RELEASING.zh-CN.md).
