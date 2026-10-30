"""Idempotence and query contracts for the scalar inverted index.

The index is derived data: every entry it holds is a function of some payload,
which means it can be wrong in two directions — an entry a payload should own
but does not, and an entry nothing owns any more. Both are reachable by an
ordinary crash, so both are treated as state to be measured and repaired rather
than as impossible conditions.

The idempotence tests matter for the same reason maintenance runs on every
write: an entry rewritten on an unchanged field would be deleted and re-added
in that order, erasing itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sable_vector.engine import VectorEngine
from sable_vector.index.scalar import IndexAudit, ScalarIndex, evaluate_condition
from sable_vector.storage import VectorStorage, render_value

FIELDS = ("tenant", "tier", "lang")


@pytest.fixture()
def storage(tmp_path: Path) -> VectorStorage:
    handle = VectorStorage.open(tmp_path / "db")
    yield handle
    handle.close()


@pytest.fixture()
def index(storage: VectorStorage) -> ScalarIndex:
    return ScalarIndex(storage, FIELDS)


def payload(tenant: str = "acme", tier: int = 3, lang: str | None = "en") -> dict:
    document: dict = {"tenant": tenant, "tier": tier}
    if lang is not None:
        document["lang"] = lang
    return document


# -- maintenance -----------------------------------------------------------


def test_owned_covers_only_present_fields(index: ScalarIndex) -> None:
    assert index.owned("doc-1", payload(lang=None)) == [("tenant", "acme"), ("tier", 3)]
    assert index.owned("doc-1", {}) == []


def test_index_writes_one_entry_per_field(index: ScalarIndex, storage: VectorStorage) -> None:
    assert index.index("doc-1", payload()) == 3
    assert len(list(storage.scan_index_field("tenant"))) == 1


def test_indexing_the_same_payload_twice_is_idempotent(
    index: ScalarIndex, storage: VectorStorage
) -> None:
    assert index.index("doc-1", payload()) == 3
    assert index.index("doc-1", payload()) == 3
    assert list(storage.scan_index("tenant", "acme")) == ["doc-1"]


def test_two_records_share_a_value(index: ScalarIndex, storage: VectorStorage) -> None:
    index.index("doc-1", payload())
    index.index("doc-2", payload())
    assert sorted(storage.scan_index("tenant", "acme")) == ["doc-1", "doc-2"]


def test_deindex_removes_exactly_that_records_entries(
    index: ScalarIndex, storage: VectorStorage
) -> None:
    index.index("doc-1", payload())
    index.index("doc-2", payload(tier=7))
    assert index.deindex("doc-1", payload()) == 3
    assert list(storage.scan_index("tenant", "acme")) == ["doc-2"]
    assert list(storage.scan_index("tier", 7)) == ["doc-2"]


def test_deindex_without_a_payload_walks_the_field(
    index: ScalarIndex, storage: VectorStorage
) -> None:
    index.index("doc-1", payload())
    index.index("doc-2", payload())
    assert index.deindex("doc-1") == 3
    assert list(storage.scan_index_field("tenant")) == [("acme", "doc-2")]


def test_reindex_touches_only_changed_fields(index: ScalarIndex, storage: VectorStorage) -> None:
    index.index("doc-1", payload())
    assert index.reindex("doc-1", payload(), payload(tenant="globex")) == 2
    assert list(storage.scan_index("tenant", "acme")) == []
    assert list(storage.scan_index("tenant", "globex")) == ["doc-1"]
    assert list(storage.scan_index("lang", "en")) == ["doc-1"]


def test_reindex_with_an_unchanged_payload_is_free(
    index: ScalarIndex, storage: VectorStorage
) -> None:
    index.index("doc-1", payload())
    assert index.reindex("doc-1", payload(), payload()) == 0
    assert list(storage.scan_index("tenant", "acme")) == ["doc-1"]


def test_reindex_handles_a_field_appearing_and_disappearing(
    index: ScalarIndex, storage: VectorStorage
) -> None:
    index.index("doc-1", payload())
    assert index.reindex("doc-1", payload(), payload(lang=None)) == 1
    assert list(storage.scan_index_field("lang")) == []


def test_reindex_from_no_previous_payload(index: ScalarIndex, storage: VectorStorage) -> None:
    assert index.reindex("doc-1", None, payload()) == 3
    assert list(storage.scan_index("tenant", "acme")) == ["doc-1"]


# -- querying --------------------------------------------------------------


def test_an_empty_predicate_imposes_no_restriction(index: ScalarIndex) -> None:
    assert index.matching_ids(None) is None
    assert index.matching_ids({}) is None


def test_equality_is_a_single_value(index: ScalarIndex, storage: VectorStorage) -> None:
    index.index("doc-1", payload(tenant="acme"))
    index.index("doc-2", payload(tenant="globex"))
    assert index.matching_ids({"tenant": "acme"}) == {"doc-1"}


def test_a_conjunction_intersects(index: ScalarIndex, storage: VectorStorage) -> None:
    index.index("doc-1", payload(tenant="acme", tier=3))
    index.index("doc-2", payload(tenant="acme", tier=7))
    index.index("doc-3", payload(tenant="globex", tier=3))
    assert index.matching_ids({"tenant": "acme", "tier": 3}) == {"doc-1"}


def test_an_empty_match_is_distinct_from_no_restriction(
    index: ScalarIndex, storage: VectorStorage
) -> None:
    index.index("doc-1", payload())
    assert index.matching_ids({"tenant": "nobody"}) == set()
    assert index.matching_ids({}) is None


def test_operators(index: ScalarIndex, storage: VectorStorage) -> None:
    for tier in (1, 5, 9):
        index.index(f"doc-{tier}", payload(tier=tier))
    assert index.matching_ids({"tier": {"$eq": 5}}) == {"doc-5"}
    assert index.matching_ids({"tier": {"$in": [1, 9]}}) == {"doc-1", "doc-9"}
    assert index.matching_ids({"tier": {"$ne": 5}}) == {"doc-1", "doc-9"}
    assert index.matching_ids({"tier": {"$nin": [1, 5]}}) == {"doc-9"}


def test_ranges_follow_numeric_order_not_lexical(index: ScalarIndex, storage: VectorStorage) -> None:
    for tier in (-3, 2, 10, 40):
        index.index(f"doc-{tier}", payload(tier=tier))
    assert index.matching_ids({"tier": {"$gt": 2}}) == {"doc-10", "doc-40"}
    assert index.matching_ids({"tier": {"$gte": 2}}) == {"doc-2", "doc-10", "doc-40"}
    assert index.matching_ids({"tier": {"$lt": 2}}) == {"doc--3"}
    assert index.matching_ids({"tier": {"$lte": 2}}) == {"doc--3", "doc-2"}


def test_a_range_and_an_equality_combine(index: ScalarIndex, storage: VectorStorage) -> None:
    index.index("doc-1", payload(tenant="acme", tier=2))
    index.index("doc-2", payload(tenant="acme", tier=90))
    index.index("doc-3", payload(tenant="globex", tier=2))
    assert index.matching_ids({"tenant": "acme", "tier": {"$lte": 10}}) == {"doc-1"}


def test_booleans_and_strings_are_comparable(index: ScalarIndex, storage: VectorStorage) -> None:
    index.index("doc-1", payload(tenant="beta"))
    index.index("doc-2", payload(tenant="alpha"))
    assert index.matching_ids({"tenant": {"$lt": "b"}}) == {"doc-2"}


def test_an_unindexed_field_is_rejected(index: ScalarIndex) -> None:
    with pytest.raises(ValueError, match="is not indexed"):
        index.matching_ids({"missing": 1})


def test_an_unknown_operator_is_rejected(index: ScalarIndex) -> None:
    with pytest.raises(ValueError, match="unsupported filter operators"):
        index.matching_ids({"tenant": {"$regex": "a"}})


def test_values_of_lists_indexed_values(index: ScalarIndex, storage: VectorStorage) -> None:
    index.index("doc-1", payload(tenant="acme"))
    index.index("doc-2", payload(tenant="globex"))
    assert index.values_of("tenant") == ["acme", "globex"]


def test_field_ids_without_a_predicate_is_unrestricted(index: ScalarIndex) -> None:
    index.index("doc-1", payload())
    assert index.field_ids("tenant", "acme") == {"doc-1"}


# -- condition evaluation --------------------------------------------------


def test_evaluate_condition_covers_every_operator() -> None:
    rendered = render_value(5).decode("utf-8")
    assert evaluate_condition(5, rendered) is True
    assert evaluate_condition({"$eq": 5}, rendered) is True
    assert evaluate_condition({"$ne": 5}, rendered) is False
    assert evaluate_condition({"$in": [4, 5]}, rendered) is True
    assert evaluate_condition({"$nin": [4, 5]}, rendered) is False
    assert evaluate_condition({"$gt": 4}, rendered) is True
    assert evaluate_condition({"$gte": 5}, rendered) is True
    assert evaluate_condition({"$lt": 6}, rendered) is True
    assert evaluate_condition({"$lte": 5}, rendered) is True
    assert evaluate_condition({"$gt": 5, "$lt": 6}, rendered) is False


def test_evaluate_condition_rejects_unknown_operators() -> None:
    with pytest.raises(ValueError, match="unsupported filter operators"):
        evaluate_condition({"$like": "x"}, "x")


# -- consistency -----------------------------------------------------------


def test_audit_is_consistent_after_normal_maintenance(
    index: ScalarIndex, storage: VectorStorage
) -> None:
    storage.put_payload("doc-1", payload())
    storage.put_payload("doc-2", payload(tenant="globex"))
    index.index("doc-1", payload())
    index.index("doc-2", payload(tenant="globex"))
    assert index.audit().consistent is True


def test_audit_reports_a_missing_entry(index: ScalarIndex, storage: VectorStorage) -> None:
    storage.put_payload("doc-1", payload())
    index.index("doc-1", payload())
    storage.delete_index_key("tenant", "acme", "doc-1")
    report = index.audit()
    assert report.missing == (("tenant", "acme", "doc-1"),)
    assert report.orphaned == ()
    assert report.total == 1


def test_audit_reports_an_orphaned_entry(index: ScalarIndex, storage: VectorStorage) -> None:
    storage.put_index_key("tenant", "ghost", "doc-99")
    report = index.audit()
    assert report.orphaned == (("tenant", "ghost", "doc-99"),)
    assert report.consistent is False


def test_repair_fixes_both_directions(index: ScalarIndex, storage: VectorStorage) -> None:
    storage.put_payload("doc-1", payload())
    index.index("doc-1", payload())
    storage.delete_index_key("tenant", "acme", "doc-1")
    storage.put_index_key("tenant", "ghost", "doc-99")

    before = index.repair()
    assert (len(before.missing), len(before.orphaned)) == (1, 1)
    assert index.audit().consistent is True
    assert list(storage.scan_index("tenant", "acme")) == ["doc-1"]
    assert list(storage.scan_index("tenant", "ghost")) == []


def test_repair_is_idempotent(index: ScalarIndex, storage: VectorStorage) -> None:
    storage.put_payload("doc-1", payload())
    index.index("doc-1", payload())
    index.repair()
    assert index.repair().total == 0


def test_audit_stays_consistent_across_a_crud_sequence(tmp_path: Path) -> None:
    """Create, update and delete must each leave the index exactly consistent."""
    engine = VectorEngine.open(tmp_path / "db", indexed_fields=FIELDS)
    index = ScalarIndex(engine.storage, FIELDS)
    try:
        engine.upsert("doc-1", [0.1] * 4, payload())
        assert index.audit().consistent
        engine.upsert("doc-1", [0.2] * 4, payload(tenant="globex"))
        assert index.audit().consistent
        engine.upsert("doc-1", [0.3] * 4, payload(tenant="globex", tier=9))
        assert index.audit().consistent
        assert index.matching_ids({"tenant": "globex", "tier": 9}) == {"doc-1"}
        engine.delete("doc-1")
        assert index.audit().consistent
        assert index.matching_ids({"tenant": "globex"}) == set()
    finally:
        engine.close()


def test_audit_survives_a_restart(tmp_path: Path) -> None:
    engine = VectorEngine.open(tmp_path / "db", indexed_fields=FIELDS)
    engine.upsert("doc-1", [0.1] * 4, payload())
    engine.upsert("doc-2", [0.2] * 4, payload(tenant="globex"))
    engine.close()

    reopened = VectorEngine.open(tmp_path / "db", indexed_fields=FIELDS)
    index = ScalarIndex(reopened.storage, FIELDS)
    try:
        assert index.audit().consistent
        assert index.matching_ids({"tenant": "acme"}) == {"doc-1"}
    finally:
        reopened.close()


def test_an_empty_audit_is_consistent(index: ScalarIndex) -> None:
    assert IndexAudit().consistent is True
    assert IndexAudit().total == 0
