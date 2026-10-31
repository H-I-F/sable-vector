"""The MCP adapter has to stay bound to the tools it advertises.

These tests cover the contract the transport layer cannot check on its own: the
tool list and every declared schema line up, and each advertised tool actually
runs against a real collection. They skip cleanly when the ``mcp`` extra is not
installed, because the library must keep working without it.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mcp", reason="the mcp extra is not installed")

from sable_vector.adapter.mcp_server import (  # noqa: E402
    DESCRIBE,
    SEARCH,
    UPSERT,
    call_tool,
    list_tools,
)

EXPECTED = {SEARCH, UPSERT, DESCRIBE}

_DIM = 8


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


@pytest.fixture()
def collection(tmp_path: Any) -> str:
    return str(tmp_path / "collection.sv")


def test_every_advertised_tool_has_a_valid_object_schema() -> None:
    tools = _run(list_tools())

    assert {tool.name for tool in tools} == EXPECTED
    for tool in tools:
        assert tool.description, f"{tool.name} ships without a description"
        schema = tool.inputSchema
        assert schema["type"] == "object", f"{tool.name} schema must be an object"
        assert "collection" in schema["properties"], f"{tool.name} cannot locate a store"
        assert "collection" in schema["required"], f"{tool.name} must require the collection"


def test_write_then_read_roundtrip(collection: str) -> None:
    first, second = np.random.randn(_DIM).astype("float32"), np.random.randn(_DIM).astype("float32")

    written = _run(
        call_tool(
            UPSERT,
            {
                "collection": collection,
                "id": "doc-1",
                "values": first.tolist(),
                "payload": {"tenant": "acme"},
                "indexed_fields": ["tenant"],
            },
        )
    )
    assert written["id"] == "doc-1"
    assert written["dim"] == _DIM
    assert written["total"] == 1

    found = _run(
        call_tool(SEARCH, {"collection": collection, "query": first.tolist(), "k": 1})
    )
    assert found["count"] == 1
    assert found["matches"][0]["id"] == "doc-1"
    assert found["matches"][0]["payload"] == {"tenant": "acme"}

    described = _run(call_tool(DESCRIBE, {"collection": collection}))
    assert described["total"] == 1
    assert described["dim"] == _DIM
    assert described["metric"] == "cosine"
    assert second is not None


def test_a_restricted_search_returns_only_matching_records(collection: str) -> None:
    rng = np.random.default_rng(0)
    acme = rng.standard_normal(_DIM).astype("float32")
    other = rng.standard_normal(_DIM).astype("float32")

    common = {"collection": collection, "indexed_fields": ["tenant"]}
    _run(call_tool(UPSERT, {**common, "id": "in", "values": acme.tolist(),
                            "payload": {"tenant": "acme"}}))
    _run(call_tool(UPSERT, {**common, "id": "out", "values": other.tolist(),
                            "payload": {"tenant": "globex"}}))

    filtered = _run(
        call_tool(
            SEARCH,
            {"collection": collection, "query": acme.tolist(), "k": 5,
             "filters": {"tenant": "acme"}},
        )
    )
    assert [match["id"] for match in filtered["matches"]] == ["in"]

    unrestricted = _run(
        call_tool(SEARCH, {"collection": collection, "query": acme.tolist(), "k": 5})
    )
    assert len(unrestricted["matches"]) == 2


def test_searching_an_empty_collection_is_not_an_error(collection: str) -> None:
    found = _run(
        call_tool(SEARCH, {"collection": collection, "query": [0.1] * _DIM, "k": 3})
    )

    assert found == {"matches": [], "count": 0, "candidates": None, "scored": 0}


def test_an_unknown_tool_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown tool"):
        _run(call_tool("sable_vector_delete_everything", {"collection": "whatever"}))
