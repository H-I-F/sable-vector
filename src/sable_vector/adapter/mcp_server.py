"""Model Context Protocol server for sable-vector.

Speaks MCP over stdio so an agent host can write embeddings into, and run hybrid
searches against, a local collection without importing this library. Nothing here
implements retrieval: every tool calls the same public surface the Python API
offers, and the point of the module is translating JSON arguments into those
calls and back.

Three tools are advertised: one write, one read, one introspection. Arguments are
validated against the schemas declared in :func:`list_tools`, so the tool bodies
can assume well-formed input.

Requires the ``mcp`` extra: ``pip install "sable-vector[mcp]"``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from sable_vector.engine import VectorEngine
from sable_vector.index.flat import FlatIndex
from sable_vector.index.scalar import ScalarIndex
from sable_vector.query.prefilter import PrefilterExecutor
from sable_vector.types import DistanceMetric, Match

SEARCH = "sable_vector_search"
UPSERT = "sable_vector_upsert"
DESCRIBE = "sable_vector_describe"

_COLLECTION = {
    "type": "string",
    "description": "Directory holding the collection. Created if it does not exist.",
}

_METRIC = {
    "type": "string",
    "enum": [member.value for member in DistanceMetric],
    "description": "Distance geometry used for this call. Defaults to cosine.",
}

_DEFAULT_TOP_K = 5


def _self_version() -> str:
    """Installed version, so the handshake can report something honest."""
    try:
        return version("sable-vector")
    except PackageNotFoundError:  # running from a checkout that was never installed
        return "0.0.0"


def _vector_schema(description: str) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "number"},
        "minItems": 1,
        "description": description,
    }


server = Server("sable-vector", version=_self_version())


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Advertise the three capabilities this server exposes."""
    return [
        types.Tool(
            name=SEARCH,
            description=(
                "Search a collection by embedding, optionally narrowed by payload "
                "equality filters (hybrid retrieval). The caller supplies the query "
                "vector; this server never computes embeddings."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": _COLLECTION,
                    "query": _vector_schema("Query embedding."),
                    "k": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Neighbours to return.",
                    },
                    "metric": _METRIC,
                    "filters": {
                        "type": "object",
                        "description": (
                            "Payload equality filters. Only records written with these "
                            "fields declared as indexed are visible to the filter."
                        ),
                    },
                },
                "required": ["collection", "query"],
            },
        ),
        types.Tool(
            name=UPSERT,
            description=(
                "Insert or replace one record: its embedding plus arbitrary JSON "
                "attributes, written as one durable unit."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": _COLLECTION,
                    "id": {"type": "string", "description": "Record identifier."},
                    "values": _vector_schema("Embedding to store."),
                    "payload": {"type": "object", "description": "Arbitrary JSON attributes."},
                    "indexed_fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Payload fields to invert so they become usable as search "
                            "filters. Must be declared at write time."
                        ),
                    },
                },
                "required": ["collection", "id", "values"],
            },
        ),
        types.Tool(
            name=DESCRIBE,
            description="Report how many records a collection holds, its width and its metric.",
            inputSchema={
                "type": "object",
                "properties": {"collection": _COLLECTION},
                "required": ["collection"],
            },
        ),
    ]


def _metric(arguments: dict[str, Any]) -> DistanceMetric:
    raw = arguments.get("metric")
    return DistanceMetric(str(raw).lower()) if raw else DistanceMetric.COSINE


def _indexed_fields(arguments: dict[str, Any], key: str) -> tuple[str, ...]:
    declared = arguments.get(key) or ()
    return tuple(str(field) for field in declared)


def _open(arguments: dict[str, Any], indexed_fields: tuple[str, ...] = ()) -> VectorEngine:
    """Open (or recover) the collection the caller named.

    Opening runs reclimation, so a caller never observes a record whose payload
    and vector disagree, whichever process died earlier.
    """
    return VectorEngine.open(
        Path(str(arguments["collection"])),
        metric=_metric(arguments),
        indexed_fields=indexed_fields,
    )


def _materialise(engine: VectorEngine, fallback_dim: int) -> FlatIndex:
    """Stack the whole corpus into one matrix.

    Safe on an empty collection: ``dim`` is only unknown while nothing has been
    written yet, so the query supplies the width and the index reports zero rows.
    """
    return FlatIndex.materialise(
        engine.storage,
        metric=engine.metric,
        dim=int(engine.dim or fallback_dim),
    )


def _record_count(engine: VectorEngine) -> int:
    """Count committed records. A record *is* its payload, so payloads are truth."""
    return sum(1 for _ in engine.storage.iter_payloads())


def _as_json(match: Match) -> dict[str, Any]:
    return {
        "id": str(match.id),
        "score": float(match.score),
        "distance": float(match.distance),
        "payload": dict(match.payload),
    }


async def _search(arguments: dict[str, Any]) -> dict[str, Any]:
    query = np.asarray(arguments["query"], dtype="float32")
    top_k = int(arguments.get("k") or _DEFAULT_TOP_K)
    filters = dict(arguments.get("filters") or {})

    engine = _open(arguments, tuple(sorted(filters)))
    try:
        flat = _materialise(engine, int(query.shape[0]))

        if filters:
            executor = PrefilterExecutor(
                engine.storage,
                flat,
                ScalarIndex(engine.storage, tuple(sorted(filters))),
            )
            outcome = executor.search(query, k=top_k, predicate=filters)
            return {
                "matches": [_as_json(match) for match in outcome.matches],
                "count": len(outcome.matches),
                "candidates": outcome.candidates,
                "scored": outcome.scored,
            }

        # The Flat index scores and nothing else, so it hands back empty payloads;
        # attaching them costs one read per returned neighbour, the same trade the
        # query executors make.
        scored = flat.search(query, k=top_k)
        matches = [
            Match(match.id, match.score, match.distance, engine.storage.get_payload(match.id) or {})
            for match in scored
        ]
        return {
            "matches": [_as_json(match) for match in matches],
            "count": len(matches),
            "candidates": None,
            "scored": flat.size,
        }
    finally:
        engine.close()


async def _upsert(arguments: dict[str, Any]) -> dict[str, Any]:
    values = np.asarray(arguments["values"], dtype="float32")
    vector_id = str(arguments["id"])

    engine = _open(arguments, _indexed_fields(arguments, "indexed_fields"))
    try:
        engine.upsert(vector_id, values, dict(arguments.get("payload") or {}))
        return {
            "id": vector_id,
            "dim": int(values.shape[0]),
            "indexed_fields": list(_indexed_fields(arguments, "indexed_fields")),
            "total": _record_count(engine),
        }
    finally:
        engine.close()


async def _describe(arguments: dict[str, Any]) -> dict[str, Any]:
    engine = _open(arguments)
    try:
        return {
            "collection": str(engine.home),
            "total": _record_count(engine),
            "dim": engine.dim,
            "metric": engine.metric.value,
        }
    finally:
        engine.close()


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one tool call to its implementation.

    Returning a mapping lets the SDK populate ``structuredContent`` and derive the
    text block itself, so there is one representation rather than two to keep in sync.
    """
    if name == SEARCH:
        return await _search(arguments)
    if name == UPSERT:
        return await _upsert(arguments)
    if name == DESCRIBE:
        return await _describe(arguments)
    raise ValueError(f"unknown tool: {name}")


async def run_server() -> None:
    """Serve over stdio until the client closes the stream.

    Blocking by design: this is the transport loop, and it returns only when the
    peer disconnects. Call it from a process whose job is to be this server.
    """
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
