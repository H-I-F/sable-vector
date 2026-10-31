# sable-vector — container image
#
# Single-stage build: the engine and its dependency `sable` are installed from
# source, so the image carries everything needed to run the test suite and the
# embedded library. `git` is required because `sable` is fetched from its
# repository (it is not published to PyPI).

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# `git` is needed to install the `sable` storage engine from its repository.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the package metadata and sources first; this layer only re-builds when the
# project itself changes, not when the dependency set does.
COPY pyproject.toml ./
COPY src ./src
COPY tests ./tests
COPY examples ./examples

RUN pip install --upgrade pip \
    && pip install "sable @ git+https://github.com/MKA930/sable.git@v0.1.1" \
    && pip install numpy pydantic rich typer \
    && pip install -e .

# The test runner is part of the image so the build is self-validating.
RUN pip install "pytest>=7.4"

# The default command validates the image by running the full test suite.
CMD ["sh", "-c", "pytest -q tests/"]
