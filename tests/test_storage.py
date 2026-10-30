"""Key-space contracts for the sable namespace adapter.

The adapter exists so that a predicate is a *range* rather than a per-record
test, and that property is only worth having if the key space actually holds:
one badly escaped separator turns ``idx:a:b:c`` into two different predicates,
and one overlapping prefix turns a namespace scan into a leak. These tests pin
the layout, the escaping, the disjointness and the round trip through a real
restart, because every one of those fails silently otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sable.types import EngineOptions

from sable_vector.storage import (
    CENTROID_PREFIX,
    INDEX_PREFIX,
    META_PREFIX,
    PAYLOAD_PREFIX,
    POSTING_PREFIX,
    VECTOR_PREFIX,
    VectorStorage,
    index_key,
    index_prefix,
    payload_key,
    prefix_upper_bound,
    vector_key,
)


@pytest.fixture()
def storage(tmp_path: Path) -> VectorStorage:
    handle = VectorStorage.open(tmp_path / "db", EngineOptions(block_size_bytes=256))
    yield handle
    handle.close()


def test_key_layout_is_literal() -> None:
    assert vector_key("doc-1") == b"v:doc-1"
    assert payload_key("doc-1") == b"p:doc-1"
    assert index_key("tenant", "acme", "doc-1") == b"idx:tenant:acme:doc-1"


def test_namespaces_are_mutually_disjoint() -> None:
    prefixes = [VECTOR_PREFIX, PAYLOAD_PREFIX, INDEX_PREFIX, CENTROID_PREFIX, POSTING_PREFIX, META_PREFIX]
    for left in prefixes:
        for right in prefixes:
            if left is right:
                continue
            assert not left.startswith(right), f"{left!r} overlaps {right!r}"
            assert not right.startswith(left), f"{right!r} overlaps {left!r}"


def test_prefix_upper_bound_increments_the_last_byte() -> None:
    assert prefix_upper_bound(b"v:") == b"v;"
    assert prefix_upper_bound(b"idx:a") == b"idx:b"


def test_prefix_upper_bound_carries_across_trailing_0xff() -> None:
    assert prefix_upper_bound(b"a\xff\xff") == b"b"


def test_prefix_upper_bound_is_none_when_there_is_no_upper_bound() -> None:
    assert prefix_upper_bound(b"\xff\xff") is None


# -- escaping -------------------------------------------------------------


def test_a_separator_inside_an_id_cannot_forge_a_segment(storage: VectorStorage) -> None:
    storage.put_vector("a:b", b"1234")
    assert list(storage.scan_prefix(VECTOR_PREFIX)) == [(vector_key("a:b"), b"1234")]
    assert storage.get_vector_blob("a:b") == b"1234"
    assert list(storage.iter_vector_ids()) == ["a:b"]


def test_a_backslash_inside_an_id_round_trips(storage: VectorStorage) -> None:
    storage.put_vector("a\\b", b"xyzw")
    assert storage.get_vector_blob("a\\b") == b"xyzw"
    assert list(storage.iter_vector_ids()) == ["a\\b"]


def test_a_separator_inside_an_indexed_value_round_trips(storage: VectorStorage) -> None:
    storage.put_index_key("path", "/a/b", "doc:1")
    assert list(storage.scan_index("path", "/a/b")) == ["doc:1"]
    assert list(storage.scan_index_field("path")) == [("/a/b", "doc:1")]


def test_similar_ids_do_not_collide(storage: VectorStorage) -> None:
    storage.put_vector("a:b", b"1111")
    storage.put_vector("a", b"2222")
    assert storage.get_vector_blob("a") == b"2222"
    assert storage.get_vector_blob("a:b") == b"1111"
    assert sorted(storage.iter_vector_ids()) == ["a", "a:b"]


# -- namespace isolation --------------------------------------------------


def test_each_namespace_scan_returns_only_its_own_keys(storage: VectorStorage) -> None:
    storage.put_vector("doc-1", b"\x00" * 8)
    storage.put_payload("doc-1", {"tenant": "acme"})
    storage.put_index_key("tenant", "acme", "doc-1")
    storage.put_raw(CENTROID_PREFIX + b"3", b"\x00" * 4)
    storage.put_raw(POSTING_PREFIX + b"3:doc-1", b"\x00" * 4)
    storage.put_raw(META_PREFIX + b"dim", b"8")

    assert [key for key, _ in storage.scan_prefix(VECTOR_PREFIX)] == [vector_key("doc-1")]
    assert [key for key, _ in storage.scan_prefix(PAYLOAD_PREFIX)] == [payload_key("doc-1")]
    assert [key for key, _ in storage.scan_prefix(INDEX_PREFIX)] == [
        index_key("tenant", "acme", "doc-1")
    ]
    assert [key for key, _ in storage.scan_prefix(CENTROID_PREFIX)] == [CENTROID_PREFIX + b"3"]
    assert [key for key, _ in storage.scan_prefix(META_PREFIX)] == [META_PREFIX + b"dim"]


def test_vector_and_payload_ids_are_independent(storage: VectorStorage) -> None:
    storage.put_vector("shared", b"aaaa")
    assert storage.get_payload("shared") is None
    storage.put_payload("shared", {"k": 1})
    assert storage.get_vector_blob("shared") == b"aaaa"
    storage.delete_vector("shared")
    assert storage.get_vector_blob("shared") is None
    assert storage.get_payload("shared") == {"k": 1}


def test_scan_prefix_limit_caps_the_emitted_pairs(storage: VectorStorage) -> None:
    for index in range(5):
        storage.put_vector(f"doc-{index}", b"\x01\x02\x03\x04")
    assert len(list(storage.scan_prefix(VECTOR_PREFIX, limit=2))) == 2


# -- payloads -------------------------------------------------------------


def test_payload_round_trip_preserves_structure(storage: VectorStorage) -> None:
    payload = {"tenant": "acme", "score": 0.5, "tags": ["x"], "nested": {"a": 1}}
    storage.put_payload("doc-1", payload)
    assert storage.get_payload("doc-1") == payload


def test_iter_payloads_returns_every_committed_payload(storage: VectorStorage) -> None:
    storage.put_payload("doc-2", {"tenant": "globex"})
    storage.put_payload("doc-1", {"tenant": "acme"})
    assert dict(storage.iter_payloads()) == {
        "doc-2": {"tenant": "globex"},
        "doc-1": {"tenant": "acme"},
    }


def test_deleting_a_payload_removes_only_that_payload(storage: VectorStorage) -> None:
    storage.put_payload("keep", {"a": 1})
    storage.put_payload("drop", {"a": 2})
    storage.delete_payload("drop")
    assert list(storage.iter_payloads()) == [("keep", {"a": 1})]


# -- inverted index -------------------------------------------------------


def test_index_entries_are_empty_values(storage: VectorStorage) -> None:
    storage.put_index_key("tenant", "acme", "doc-1")
    assert storage.get_raw(index_key("tenant", "acme", "doc-1")) == b""


def test_writing_an_index_entry_twice_is_idempotent(storage: VectorStorage) -> None:
    for _ in range(3):
        storage.put_index_key("tenant", "acme", "doc-1")
    assert list(storage.scan_index("tenant", "acme")) == ["doc-1"]


def test_scan_index_returns_only_the_matching_value(storage: VectorStorage) -> None:
    storage.put_index_key("tenant", "acme", "doc-1")
    storage.put_index_key("tenant", "acme", "doc-2")
    storage.put_index_key("tenant", "globex", "doc-3")
    assert sorted(storage.scan_index("tenant", "acme")) == ["doc-1", "doc-2"]
    assert list(storage.scan_index("tenant", "globex")) == ["doc-3"]


def test_deleting_an_index_entry_leaves_its_siblings(storage: VectorStorage) -> None:
    storage.put_index_key("tenant", "acme", "doc-1")
    storage.put_index_key("tenant", "acme", "doc-2")
    storage.delete_index_key("tenant", "acme", "doc-1")
    assert list(storage.scan_index("tenant", "acme")) == ["doc-2"]


def test_numeric_values_render_in_numeric_order() -> None:
    """Lexical order over the rendered key must match numeric order.

    Fixed width is what makes ``2 < 10`` hold as text, and the sign is what
    makes ``-1`` sort before both: ``-`` is 0x2D, below every digit.
    """
    values = (-1, 2, 10)
    rendered = [index_key("n", value, "id") for value in values]
    assert rendered == sorted(rendered)


def test_booleans_render_as_stable_literals(storage: VectorStorage) -> None:
    storage.put_index_key("flag", True, "doc-1")
    storage.put_index_key("flag", False, "doc-2")
    assert list(storage.scan_index("flag", True)) == ["doc-1"]
    assert list(storage.scan_index("flag", False)) == ["doc-2"]


def test_index_prefix_covers_the_field_and_narrows_by_value() -> None:
    assert index_prefix("tenant") == b"idx:tenant:"
    assert index_prefix("tenant", "acme") == b"idx:tenant:acme:"


def test_scan_index_field_walks_values_in_order(storage: VectorStorage) -> None:
    for value in (30, 5, 100):
        storage.put_index_key("size", value, f"doc-{value}")
    assert [pair[1] for pair in storage.scan_index_field("size")] == ["doc-5", "doc-30", "doc-100"]


# -- durability -----------------------------------------------------------


def test_data_survives_a_close_and_reopen(tmp_path: Path) -> None:
    home = tmp_path / "db"
    first = VectorStorage.open(home)
    first.put_vector("doc-1", b"\x01\x02\x03\x04")
    first.put_payload("doc-1", {"tenant": "acme"})
    first.put_index_key("tenant", "acme", "doc-1")
    first.close()

    second = VectorStorage.open(home)
    try:
        assert second.get_vector_blob("doc-1") == b"\x01\x02\x03\x04"
        assert second.get_payload("doc-1") == {"tenant": "acme"}
        assert list(second.scan_index("tenant", "acme")) == ["doc-1"]
    finally:
        second.close()


def test_data_survives_a_flush_without_a_restart(storage: VectorStorage) -> None:
    storage.put_vector("doc-1", b"\x01\x02\x03\x04")
    storage.flush()
    assert storage.get_vector_blob("doc-1") == b"\x01\x02\x03\x04"


def test_compaction_is_reported_and_preserves_the_key_space(storage: VectorStorage) -> None:
    for index in range(40):
        storage.put_vector(f"doc-{index:03d}", b"\x01\x02\x03\x04")
        storage.flush()
    progressed = storage.compact()
    assert isinstance(progressed, bool)
    assert len(list(storage.scan_prefix(VECTOR_PREFIX))) == 40


def test_stats_are_exposed_from_the_storage_engine(storage: VectorStorage) -> None:
    storage.put_vector("doc-1", b"\x01\x02\x03\x04")
    assert storage.stats.puts >= 1
