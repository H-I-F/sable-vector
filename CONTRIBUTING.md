# Contributing to sable-vector

sable-vector is an embedded hybrid (dense + scalar-filter) vector engine for
local AI agent runtimes, built on top of the [`sable`](https://github.com/MKA930/sable)
LSM storage engine. This document describes how to set up a development
environment and the conventions the project follows so a patch lands cleanly.

## Development setup

The project targets **Python 3.10+**. `sable` is not published to PyPI, so it
is installed from its repository tag.

```bash
git clone https://github.com/MKA930/sable-vector.git
cd sable-vector
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[sable,test,lint]"
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
| `benchmarks/`      | ANN benchmark suite (`python -m benchmarks.bench_vector`)| 
| `examples/`        | Runnable integration demos (e.g. LangChain RAG)          |
| `tests/`           | pytest suite                                             |

## Code style

- `ruff` is the linter and formatter gate; run `ruff check .` locally.
- Target Python is 3.10 — avoid syntax that is 3.11-only unless guarded.
- No `TODO`/`FIXME` placeholders in merged code; if something is incomplete,
  discuss it in the PR instead.
