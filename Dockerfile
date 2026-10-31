# sable-vector — multi-stage container image
#
# Stage 1 (builder) compiles the dependency set and installs the engine into a
# virtualenv. `sable` is fetched from its repository (it is not published to
# PyPI), so `git` lives only in this stage.
#
# Stage 2 (runtime) copies the populated virtualenv onto a clean slim base, so
# the shipped image carries no compiler, no git and no build caches. The test
# runner is part of the image so the build is self-validating: the default
# command runs the whole suite.

# --- builder ---------------------------------------------------------------
FROM python:3.15-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# The virtualenv is the only artefact this stage hands to the next one.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /build

# Metadata first: this layer only rebuilds when the dependency set changes, not
# when the source does.
COPY pyproject.toml ./

# Source next: needed so `pip install .` can build the package.
COPY src ./src

RUN pip install --upgrade pip \
    && pip install "sable @ git+https://github.com/H-I-F/sable.git@v0.1.1" \
    && pip install ".[mcp]" \
    && pip install "pytest>=7.4"

# --- runtime ---------------------------------------------------------------
FROM python:3.15-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:${PATH}"

# Bring across the builder's populated virtualenv verbatim.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# The test corpus and examples travel with the image so `pytest` has something
# to run; the engine itself lives in the virtualenv's site-packages.
COPY tests ./tests
COPY examples ./examples

# Self-validating entrypoint: the image proves itself by running the suite.
CMD ["sh", "-c", "pytest -q tests/"]
