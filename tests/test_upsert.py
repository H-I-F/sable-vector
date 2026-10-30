"""Atomicity of the write path.

A vector record is three physically separate keys, and a key-value store cannot
commit three keys together — so the atomicity here is constructed from an
ordering rule and a single fsync boundary, and both can silently stop holding.
These tests pin them:

* the payload lands *last* and is the only key that forces the log to stable
  storage, so a record costs one `fsync` no matter how many keys it spans;
* a torn write can only ever leave a vector without a payload, never a payload
  without a vector, and the reaper proves it by removing the vector;
* deletion runs the same rule backwards, so a torn delete is equally detectable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sable_vector.codec import encode_vector
from sable_vector.engine import VectorEngine
from sable_vector.storage import VECTOR_PREFIX, VectorStorage, index_key, vector_key
from sable_vector.types import DistanceMetric, VectorRecord

DIM = 8
FIELDS = ("tenant", "lang")


def vector(seed: int, dim: int = DIM) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=dim).astype("<f4")


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    return tmp_path / "collection"


def open_engine(home: Path, **kwargs: object) -> VectorEngine:
    return VectorEngine.open(home, indexed_fields=FIELDS, **kwargs)  # type: ignore[arg-type]


def record_calls(storage: VectorStorage) -> tuple[list[tuple[bytes, bool]], list[bytes]]:
    """Spy on the raw key writes so the durability boundary can be inspected.

    Returns:
        ``(writes, deletes)`` — ``writes`` pairs each key with the ``sync`` flag
        it was written under, which is the observable form of the fsync budget.
    """
    writes: list[tuple[bytes, bool]] = []
    deletes: list[bytes] = []
    real_put = storage.put_raw
    real_delete = storage.delete_raw

    def spy_put(key: bytes, value: bytes, *, sync: bool = True) -> None:
        writes.append((key, sync))
        real_put(key, value, sync=sync)

    def spy_delete(key: bytes, *, sync: bool = True) -> None:
        deletes.append(key)
        real_delete(key, sync=sync)

    storage.put_raw = spy_put  # type: ignore[method-assign]
    storage.delete_raw = spy_delete  # type: ignore[method-assign]
    return writes, deletes


# -- one fsync per record --------------------------------------------------


def test_a_record_costs_exactly_one_synced_write(home: Path) -> None:
    engine = open_engine(home)
    writes, _ = record_calls(engine.storage)
    engine.upsert("doc-1", vector(1), {"tenant": "acme", "lang": "en"})
    engine.close()

    synced = [key for key, sync in writes if sync]
    assert synced == [b"p:doc-1"], "only the payload may force the log to disk"


def test_the_payload_is_the_last_key_written(home: Path) -> None:
    engine = open_engine(home)
    writes, _ = record_calls(engine.storage)
    engine.upsert("doc-1", vector(1), {"tenant": "acme", "lang": "en"})
    engine.close()

    order = [key[:2] for key, _ in writes]
    assert order[-1] == b"p:", "the commit record must be written last"
    assert order.index(b"v:") < order.index(b"p:"), "the image precedes its payload"


def test_a_multi_record_batch_shares_one_synced_write(home: Path) -> None:
    engine = open_engine(home)
    writes, _ = record_calls(engine.storage)
    engine.upsert_many(
        [VectorRecord(f"doc-{i}", vector(i), {"tenant": "acme"}) for i in range(6)]
    )
    engine.close()

    assert len([key for key, sync in writes if sync]) == 1
    assert writes[-1][0] == b"p:doc-5"


def test_deletion_drops_the_payload_before_the_vector(home: Path) -> None:
    engine = open_engine(home)
    engine.upsert("doc-1", vector(1), {"tenant": "acme"})
    _, deletes = record_calls(engine.storage)
    engine.delete("doc-1")
    engine.close()

    assert [key[:2] for key in deletes] == [b"p:", b"id", b"v:"]


# -- torn writes -----------------------------------------------------------


def test_a_torn_write_leaves_no_orphan_payload(home: Path) -> None:
    """Crash after the vector, before the payload: the vector must go, not stay."""
    storage = VectorStorage.open(home)
    storage.put_vector("doc-1", encode_vector(vector(1)), sync=True)
    engine = VectorEngine(storage, indexed_fields=FIELDS)
    report = engine.reclaim()

    assert report.orphan_vectors == 1
    assert report.orphan_payloads == 0
    assert storage.get_vector_blob("doc-1") is None
    assert storage.get_payload("doc-1") is None
    engine.close()


def test_a_torn_write_is_invisible_after_a_restart(home: Path) -> None:
    first = VectorStorage.open(home)
    first.put_vector("doc-1", encode_vector(vector(1)), sync=True)
    first.close()

    engine = open_engine(home)
    try:
        assert list(engine.storage.scan_prefix(VECTOR_PREFIX)) == []
        assert engine.storage.get_payload("doc-1") is None
        assert engine.delete("doc-1") is False, "a torn record was never a record"
    finally:
        engine.close()


def test_a_torn_delete_leaves_no_orphan_payload(home: Path) -> None:
    """Crash after the payload is dropped, before the vector is: reclaim it."""
    engine = open_engine(home)
    engine.upsert("doc-1", vector(1), {"tenant": "acme"})
    engine.storage.delete_payload("doc-1", sync=True)
    engine.close()

    reopened = open_engine(home)
    try:
        assert reopened.storage.get_vector_blob("doc-1") is None
        assert reopened.storage.get_payload("doc-1") is None
        assert list(reopened.storage.scan_prefix(VECTOR_PREFIX)) == []
    finally:
        reopened.close()


def test_reclaim_reports_nothing_on_a_healthy_store(home: Path) -> None:
    engine = open_engine(home)
    engine.upsert("doc-1", vector(1), {"tenant": "acme"})
    engine.close()

    reopened = open_engine(home)
    try:
        assert reopened.reclaim().total == 0
    finally:
        reopened.close()


# -- inverted entries ------------------------------------------------------


def test_upsert_writes_every_indexed_field(home: Path) -> None:
    engine = open_engine(home)
    engine.upsert("doc-1", vector(1), {"tenant": "acme", "lang": "en"})
    keys = [key for key, _ in engine.storage.scan_prefix(b"idx:")]
    assert sorted(keys) == sorted(
        [index_key("tenant", "acme", "doc-1"), index_key("lang", "en", "doc-1")]
    )
    engine.close()


def test_an_update_replaces_only_the_changed_entry(home: Path) -> None:
    engine = open_engine(home)
    engine.upsert("doc-1", vector(1), {"tenant": "acme", "lang": "en"})
    _, deletes = record_calls(engine.storage)
    engine.upsert("doc-1", vector(2), {"tenant": "globex", "lang": "en"})

    assert deletes == [index_key("tenant", "acme", "doc-1")]
    assert list(engine.storage.scan_index("tenant", "acme")) == []
    assert list(engine.storage.scan_index("tenant", "globex")) == ["doc-1"]
    assert list(engine.storage.scan_index("lang", "en")) == ["doc-1"]
    engine.close()


def test_rewriting_an_unchanged_field_is_free(home: Path) -> None:
    engine = open_engine(home)
    engine.upsert("doc-1", vector(1), {"tenant": "acme"})
    writes, deletes = record_calls(engine.storage)
    engine.upsert("doc-1", vector(2), {"tenant": "acme"})
    engine.close()

    assert deletes == []
    assert [key for key, _ in writes] == [vector_key("doc-1"), b"p:doc-1"]


def test_a_payload_missing_an_indexed_field_creates_no_entry(home: Path) -> None:
    engine = open_engine(home)
    engine.upsert("doc-1", vector(1), {"tenant": "acme"})
    assert list(engine.storage.scan_index_field("lang")) == []
    engine.close()


def test_reclaim_rebuilds_an_index_entry_that_was_never_written(home: Path) -> None:
    engine = open_engine(home)
    engine.upsert("doc-1", vector(1), {"tenant": "acme"})
    engine.storage.delete_index_key("tenant", "acme", "doc-1", sync=True)
    engine.close()

    reopened = open_engine(home)
    try:
        assert list(reopened.storage.scan_index("tenant", "acme")) == ["doc-1"]
    finally:
        reopened.close()


def test_reclaim_drops_an_index_entry_for_a_dead_id(home: Path) -> None:
    engine = open_engine(home)
    engine.upsert("doc-1", vector(1), {"tenant": "acme"})
    engine.storage.delete_payload("doc-1", sync=True)
    engine.storage.delete_vector("doc-1", sync=True)
    engine.close()

    reopened = open_engine(home)
    try:
        assert list(reopened.storage.scan_index_field("tenant")) == []
    finally:
        reopened.close()


def test_delete_removes_every_key_of_a_record(home: Path) -> None:
    engine = open_engine(home)
    engine.upsert("doc-1", vector(1), {"tenant": "acme", "lang": "en"})
    assert engine.delete("doc-1") is True
    assert engine.delete("doc-1") is False
    assert list(engine.storage.scan_prefix(b"idx:")) == []
    assert engine.storage.get_vector_blob("doc-1") is None
    assert engine.storage.get_payload("doc-1") is None
    engine.close()


def test_delete_of_a_torn_record_tidies_up(home: Path) -> None:
    storage = VectorStorage.open(home)
    storage.put_vector("doc-1", encode_vector(vector(1)), sync=True)
    storage.put_index_key("tenant", "acme", "doc-1", sync=True)
    engine = VectorEngine(storage, indexed_fields=FIELDS)
    assert engine.delete("doc-1") is True
    assert storage.get_vector_blob("doc-1") is None
    assert list(storage.scan_index_field("tenant")) == []
    engine.close()


# -- geometry --------------------------------------------------------------


def test_the_first_write_fixes_the_collection_width(home: Path) -> None:
    engine = open_engine(home)
    assert engine.dim is None
    engine.upsert("doc-1", vector(1))
    assert engine.dim == DIM
    engine.close()


def test_a_mismatched_width_is_rejected(home: Path) -> None:
    engine = open_engine(home)
    engine.upsert("doc-1", vector(1))
    with pytest.raises(ValueError, match="does not match the collection width"):
        engine.upsert("doc-2", vector(2, dim=DIM + 1))
    engine.close()


def test_the_width_survives_a_restart(home: Path) -> None:
    engine = open_engine(home)
    engine.upsert("doc-1", vector(1), {"tenant": "acme"})
    engine.close()

    reopened = open_engine(home)
    try:
        assert reopened.dim == DIM
    finally:
        reopened.close()


def test_an_explicit_width_is_enforced_from_the_first_write(home: Path) -> None:
    engine = open_engine(home, dim=DIM)
    with pytest.raises(ValueError, match="does not match the collection width"):
        engine.upsert("doc-1", vector(1, dim=DIM * 2))
    engine.close()


def test_the_metric_is_part_of_the_collection_identity(home: Path) -> None:
    engine = open_engine(home, metric=DistanceMetric.L2)
    assert engine.metric is DistanceMetric.L2
    assert engine.indexed_fields == FIELDS
    engine.close()


def test_an_empty_batch_touches_nothing(home: Path) -> None:
    engine = open_engine(home)
    writes, _ = record_calls(engine.storage)
    assert engine.upsert_many([]) == 0
    assert writes == []
    engine.close()


def test_a_context_manager_closes_the_handle(home: Path) -> None:
    with open_engine(home) as engine:
        engine.upsert("doc-1", vector(1), {"tenant": "acme"})
    reopened = open_engine(home)
    try:
        assert reopened.storage.get_payload("doc-1") == {"tenant": "acme"}
    finally:
        reopened.close()
