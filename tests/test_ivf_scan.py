"""Partition-pruning contracts for the on-disk IVF index.

An IVF index earns its keep by *not reading* most of the corpus. That is the
whole property, and it is also the easiest one to lose silently: a bug that
probes every partition, or reads postings through a path that bypasses the
prefix, still returns correct neighbours — it just costs as much as Flat while
looking like an optimisation. Correctness tests cannot see that regression.

So the acceptance here is I/O, not recall. A spy on the namespace adapter
records every prefix handed to the storage engine, and the assertions are about
which partitions were *not* touched:

* an ``nprobe=1`` search over 4 partitions opens exactly one ``b:`` prefix;
* the number of postings decoded is a fraction of the corpus, not all of it;
* raising ``nprobe`` to the partition count is the only thing that makes the
  search exhaustive — and then it reproduces the float64 brute force exactly.

Recall is checked the same way everywhere else in this project: computed
independently in float64 from the raw vectors, never from the index's own
kernels, so a shared bug in both cannot agree with itself and pass.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from sable_vector.codec import encode_vector
from sable_vector.index.ivf_cluster import kmeans
from sable_vector.index.ivf_storage import (
    IVF_META_KEY,
    IvfIndex,
    centroid_key,
    posting_prefix,
)
from sable_vector.storage import (
    CENTROID_PREFIX,
    POSTING_PREFIX,
    VECTOR_PREFIX,
    VectorStorage,
    decode_id,
)
from sable_vector.types import DistanceMetric, VectorId

DIM = 16
BLOBS = 4
PER_BLOB = 40
ROWS = BLOBS * PER_BLOB
PARTITIONS = 4


def corpus(rows: int = PER_BLOB, dim: int = DIM, *, seed: int = 5) -> np.ndarray:
    """Four well-separated Gaussian blobs, the ground truth a partition should find."""
    generator = np.random.default_rng(seed)
    centres = generator.normal(scale=12.0, size=(BLOBS, dim))
    parts = [
        centre + generator.normal(scale=0.4, size=(rows, dim)) for centre in centres
    ]
    return np.ascontiguousarray(np.concatenate(parts), dtype="<f4")


def labels(count: int = ROWS) -> list[VectorId]:
    return [f"doc-{index:04d}" for index in range(count)]


@pytest.fixture()
def store(tmp_path: Path) -> Iterator[VectorStorage]:
    handle = VectorStorage.open(tmp_path / "db")
    for name, row in zip(labels(), corpus()):
        handle.put_vector(name, encode_vector(row))
    yield handle
    handle.close()


@pytest.fixture()
def index(store: VectorStorage) -> IvfIndex:
    return IvfIndex.train(
        store, metric=DistanceMetric.L2, dim=DIM, partitions=PARTITIONS, seed=11
    )


def brute_force(
    matrix: np.ndarray, query: np.ndarray, metric: DistanceMetric, k: int
) -> list[VectorId]:
    """Independent float64 ranking, used as the definition of correct recall."""
    grid = matrix.astype(np.float64)
    probe = query.astype(np.float64)
    if metric is DistanceMetric.L2:
        scores = -np.linalg.norm(grid - probe, axis=1)
    elif metric is DistanceMetric.COSINE:
        grid = grid / np.linalg.norm(grid, axis=1)[:, None]
        probe = probe / np.linalg.norm(probe)
        scores = grid @ probe
    else:
        scores = grid @ probe
    order = np.argsort(-scores, kind="stable")[:k]
    return [labels()[int(row)] for row in order]


def spy_scans(handle: VectorStorage) -> list[bytes]:
    """Wrap ``scan_prefix`` and return the live list of prefixes it is called with."""
    seen: list[bytes] = []
    original = handle.scan_prefix

    def recorder(prefix: bytes, *, limit: int | None = None):
        seen.append(bytes(prefix))
        return original(prefix, limit=limit)

    handle.scan_prefix = recorder  # type: ignore[method-assign]
    return seen


def spy_writes(handle: VectorStorage) -> list[tuple[bytes, bool]]:
    """Wrap ``put_raw`` and return the live list of ``(key, sync)`` it is called with."""
    seen: list[tuple[bytes, bool]] = []
    original = handle.put_raw

    def recorder(key: bytes, value: bytes, *, sync: bool = True) -> None:
        seen.append((bytes(key), sync))
        original(key, value, sync=sync)

    handle.put_raw = recorder  # type: ignore[method-assign]
    return seen


# -- key space -------------------------------------------------------------


def test_centroid_key_is_zero_padded_so_lexical_order_is_cluster_order() -> None:
    keys = [centroid_key(cluster) for cluster in (0, 1, 9, 10, 999)]
    assert keys == sorted(keys)
    assert all(key.startswith(CENTROID_PREFIX) for key in keys)
    assert len(centroid_key(0)) == len(centroid_key(999))


def test_posting_prefix_terminates_the_cluster_segment() -> None:
    prefix = posting_prefix(3)
    assert prefix.startswith(POSTING_PREFIX)
    assert prefix.endswith(b":")
    assert not posting_prefix(3)[:-1].endswith(b":")


def test_partition_prefixes_are_distinct() -> None:
    ordered = [posting_prefix(cluster) for cluster in range(8)]
    assert len(set(ordered)) == len(ordered)
    assert ordered == sorted(ordered)


# -- training --------------------------------------------------------------


def test_train_writes_centroids_postings_and_metadata(store: VectorStorage) -> None:
    trained = IvfIndex.train(
        store, metric=DistanceMetric.L2, dim=DIM, partitions=PARTITIONS, seed=11
    )
    assert trained.partitions == PARTITIONS
    assert trained.centroids.shape == (PARTITIONS, DIM)
    stored = list(store.scan_prefix(CENTROID_PREFIX))
    assert len(stored) == PARTITIONS
    assert sum(1 for _ in store.scan_prefix(POSTING_PREFIX)) == ROWS
    meta = trained.metadata()
    assert meta is not None
    assert meta["k"] == PARTITIONS
    assert meta["dim"] == DIM
    assert meta["vectors"] == ROWS


def test_train_rejects_an_empty_corpus(tmp_path: Path) -> None:
    handle = VectorStorage.open(tmp_path / "db")
    try:
        with pytest.raises(ValueError, match="empty corpus"):
            IvfIndex.train(handle, metric=DistanceMetric.L2, dim=DIM, partitions=2)
    finally:
        handle.close()


def test_train_rejects_a_corpus_smaller_than_the_partition_count(
    tmp_path: Path,
) -> None:
    handle = VectorStorage.open(tmp_path / "db")
    try:
        for name, row in zip(labels(3), corpus(1)[:3]):
            handle.put_vector(name, encode_vector(row))
        with pytest.raises(ValueError, match="cannot fill"):
            IvfIndex.train(handle, metric=DistanceMetric.L2, dim=DIM, partitions=8)
    finally:
        handle.close()


def test_train_commits_with_one_fsync(store: VectorStorage) -> None:
    writes = spy_writes(store)
    IvfIndex.train(
        store, metric=DistanceMetric.L2, dim=DIM, partitions=PARTITIONS, seed=11
    )
    synced = [key for key, sync in writes if sync]
    # Centroids and postings are derived data and are written unsynced; the
    # metadata key is the commit record and is the only thing fsynced. A crash
    # mid-rebuild therefore leaves postings that nothing references.
    assert synced == [IVF_META_KEY]
    assert len(writes) == PARTITIONS + ROWS + 1


def test_load_reopens_the_trained_centroids(index: IvfIndex, store: VectorStorage) -> None:
    reopened = IvfIndex.load(store, metric=DistanceMetric.L2, dim=DIM)
    assert reopened.partitions == PARTITIONS
    np.testing.assert_allclose(
        reopened.centroids, index.centroids, rtol=0, atol=0
    )


def test_load_rejects_a_store_that_was_never_trained(tmp_path: Path) -> None:
    handle = VectorStorage.open(tmp_path / "db")
    try:
        with pytest.raises(ValueError, match="train the index first"):
            IvfIndex.load(handle, metric=DistanceMetric.L2, dim=DIM)
    finally:
        handle.close()


def test_metadata_is_absent_before_training(tmp_path: Path) -> None:
    handle = VectorStorage.open(tmp_path / "db")
    try:
        empty = IvfIndex(
            handle,
            metric=DistanceMetric.L2,
            dim=DIM,
            centroids=np.zeros((1, DIM), dtype="<f4"),
        )
        assert empty.metadata() is None
    finally:
        handle.close()


# -- partition contents ----------------------------------------------------


def test_postings_partition_the_corpus_exactly(index: IvfIndex) -> None:
    seen: list[VectorId] = []
    for cluster in range(index.partitions):
        seen.extend(index.postings_in(cluster))
    assert len(seen) == ROWS
    assert set(seen) == set(labels())
    assert len(set(seen)) == ROWS


def test_every_partition_is_non_empty_over_a_separable_corpus(index: IvfIndex) -> None:
    for cluster in range(index.partitions):
        assert index.postings_in(cluster), f"partition {cluster} captured nothing"


def test_postings_in_returns_key_order(index: IvfIndex) -> None:
    for cluster in range(index.partitions):
        members = index.postings_in(cluster)
        assert members == sorted(members)


def test_cluster_of_matches_a_brute_force_assignment(index: IvfIndex) -> None:
    grid = corpus()
    expected = kmeans(grid, PARTITIONS, seed=11).assignments
    for row, name in enumerate(labels()):
        assert index.cluster_of(grid[row]) == int(expected[row])


def test_residual_round_trip_reconstructs_the_vector(
    index: IvfIndex, store: VectorStorage
) -> None:
    grid = corpus()
    position = {name: row for row, name in enumerate(labels())}
    rebuilt_rows = 0
    for cluster in range(index.partitions):
        prefix = posting_prefix(cluster)
        for key, blob in store.scan_prefix(prefix):
            name = decode_id(key[len(prefix) :].decode("utf-8"))
            residual = np.frombuffer(blob, dtype="<f4")
            np.testing.assert_allclose(
                residual + index.centroids[cluster],
                grid[position[name]],
                rtol=1e-5,
                atol=1e-4,
            )
            rebuilt_rows += 1
    assert rebuilt_rows == ROWS


def test_residuals_have_a_smaller_dynamic_range_than_the_vectors(
    index: IvfIndex, store: VectorStorage
) -> None:
    raw = corpus()
    stored: list[np.ndarray] = []
    for cluster in range(index.partitions):
        for _, blob in store.scan_prefix(posting_prefix(cluster)):
            stored.append(np.frombuffer(blob, dtype="<f4"))
    block = np.stack(stored)
    assert np.abs(block).max() < np.abs(raw).max()


# -- recall ----------------------------------------------------------------


@pytest.mark.parametrize("metric", list(DistanceMetric))
def test_exhaustive_probe_reproduces_the_brute_force_ranking(
    store: VectorStorage, metric: DistanceMetric
) -> None:
    trained = IvfIndex.train(
        store, metric=metric, dim=DIM, partitions=PARTITIONS, seed=11
    )
    grid = corpus()
    probe = np.random.default_rng(99).normal(size=DIM).astype("<f4")
    outcome = trained.search(probe, 10, nprobe=PARTITIONS)
    assert [match.id for match in outcome.matches] == brute_force(grid, probe, metric, 10)


def test_exhaustive_probe_reports_every_partition(index: IvfIndex) -> None:
    probe = np.random.default_rng(7).normal(size=DIM).astype("<f4")
    outcome = index.search(probe, 5, nprobe=PARTITIONS)
    assert outcome.buckets_scanned == PARTITIONS
    assert outcome.postings_read == ROWS


def test_recall_at_one_probe_is_a_subset_of_the_exhaustive_answer(
    index: IvfIndex,
) -> None:
    grid = corpus()
    probe = grid[0] + np.full(DIM, 0.01, dtype="<f4")
    narrow = {match.id for match in index.search(probe, 10, nprobe=1).matches}
    full = {match.id for match in index.search(probe, 10, nprobe=PARTITIONS).matches}
    assert narrow
    # Reading fewer partitions can only lose neighbours, never invent them.
    assert narrow <= full


def test_widening_the_probe_never_lowers_recall(index: IvfIndex) -> None:
    grid = corpus()
    truth = set(brute_force(grid, grid[17], DistanceMetric.L2, 10))
    hits = [
        len(
            truth
            & {match.id for match in index.search(grid[17], 10, nprobe=n).matches}
        )
        for n in (1, 2, 3, 4)
    ]
    assert hits == sorted(hits)
    assert hits[-1] == len(truth)


def test_query_near_a_centroid_finds_its_own_partition_first(index: IvfIndex) -> None:
    probe = index.centroids[2]
    outcome = index.search(probe, 5, nprobe=1)
    members = set(index.postings_in(2))
    assert {match.id for match in outcome.matches} <= members


# -- the acceptance: pruning is real I/O reduction -------------------------


def test_single_probe_opens_exactly_one_partition(index: IvfIndex, store: VectorStorage) -> None:
    seen = spy_scans(store)
    probe = index.centroids[1] + np.full(DIM, 0.05, dtype="<f4")
    outcome = index.search(probe, 5, nprobe=1)
    partition_reads = [prefix for prefix in seen if prefix.startswith(POSTING_PREFIX)]
    assert len(partition_reads) == 1
    assert outcome.buckets_scanned == 1
    assert outcome.partitions == PARTITIONS


def test_unprobed_partitions_are_never_touched(index: IvfIndex, store: VectorStorage) -> None:
    seen = spy_scans(store)
    index.search(index.centroids[3], 5, nprobe=1)
    opened = {prefix for prefix in seen if prefix.startswith(POSTING_PREFIX)}
    assert opened == {posting_prefix(3)}
    for cluster in (0, 1, 2):
        assert posting_prefix(cluster) not in opened


def test_pruning_reads_a_fraction_of_the_corpus(index: IvfIndex) -> None:
    outcome = index.search(index.centroids[0], 10, nprobe=1)
    assert 0 < outcome.postings_read < ROWS
    assert outcome.postings_read <= ROWS // 2


def test_nprobe_is_capped_at_the_partition_count(index: IvfIndex) -> None:
    outcome = index.search(index.centroids[0], 5, nprobe=99)
    assert outcome.buckets_scanned == PARTITIONS


def test_nprobe_must_be_positive(index: IvfIndex) -> None:
    with pytest.raises(ValueError, match="nprobe must be positive"):
        index.search(index.centroids[0], 5, nprobe=0)


def test_non_positive_k_returns_nothing_and_reads_nothing(index: IvfIndex) -> None:
    outcome = index.search(index.centroids[0], 0, nprobe=2)
    assert outcome.matches == ()
    assert outcome.buckets_scanned == 0
    assert outcome.postings_read == 0


# -- filtered search -------------------------------------------------------


def test_candidate_ids_restrict_the_result(index: IvfIndex) -> None:
    allow = labels()[10:20]
    outcome = index.search(index.centroids[0], 5, nprobe=PARTITIONS, candidate_ids=allow)
    assert outcome.matches
    assert {match.id for match in outcome.matches} <= set(allow)


def test_candidate_ids_do_not_widen_the_partitions_read(
    index: IvfIndex, store: VectorStorage
) -> None:
    seen = spy_scans(store)
    index.search(
        index.centroids[2], 5, nprobe=1, candidate_ids=labels()[:5]
    )
    opened = [prefix for prefix in seen if prefix.startswith(POSTING_PREFIX)]
    assert opened == [posting_prefix(2)]


def test_an_empty_allow_list_matches_nothing(index: IvfIndex) -> None:
    outcome = index.search(index.centroids[0], 5, nprobe=PARTITIONS, candidate_ids=[])
    assert outcome.matches == ()


# -- mutation --------------------------------------------------------------


def test_add_assigns_the_nearest_partition_and_is_searchable(
    index: IvfIndex, store: VectorStorage
) -> None:
    fresh = index.centroids[1] + np.full(DIM, 0.01, dtype="<f4")
    cluster = index.add("fresh", fresh)
    assert cluster == 1
    assert "fresh" in index.postings_in(1)
    hits = index.search(fresh, 3, nprobe=PARTITIONS)
    assert "fresh" in {match.id for match in hits.matches}


def test_remove_drops_the_posting(index: IvfIndex) -> None:
    victim = index.postings_in(2)[0]
    assert index.remove(victim) is True
    assert victim not in index.postings_in(2)
    assert victim not in {match.id for match in index.search(index.centroids[2], 5, nprobe=PARTITIONS).matches}


def test_remove_reports_false_for_an_unknown_id(index: IvfIndex) -> None:
    assert index.remove("never-seen") is False


def test_ids_holding_separators_survive_the_key_space(index: IvfIndex) -> None:
    awkward = "tenant:acme::doc\\1"
    index.add(awkward, index.centroids[0] + np.full(DIM, 0.02, dtype="<f4"))
    assert awkward in index.postings_in(0)
    assert index.remove(awkward) is True


def test_an_awkward_id_is_never_mistaken_for_another_partition(index: IvfIndex) -> None:
    index.add("0:x", index.centroids[0])
    index.add("1:x", index.centroids[1])
    assert "0:x" in index.postings_in(0)
    assert "1:x" in index.postings_in(1)


# -- durability ------------------------------------------------------------


def test_trained_postings_survive_flush_reopen_and_compaction(tmp_path: Path) -> None:
    home = tmp_path / "db"
    handle = VectorStorage.open(home)
    for name, row in zip(labels(), corpus()):
        handle.put_vector(name, encode_vector(row))
    IvfIndex.train(handle, metric=DistanceMetric.L2, dim=DIM, partitions=PARTITIONS, seed=11)
    handle.flush()
    assert handle.compact() in (True, False)
    handle.close()

    reopened = VectorStorage.open(home)
    try:
        loaded = IvfIndex.load(reopened, metric=DistanceMetric.L2, dim=DIM)
        assert sum(1 for _ in reopened.scan_prefix(POSTING_PREFIX)) == ROWS
        probe = loaded.centroids[3]
        outcome = loaded.search(probe, 5, nprobe=1)
        assert {match.id for match in outcome.matches} <= set(loaded.postings_in(3))
    finally:
        reopened.close()


def test_training_twice_replaces_the_postings(index: IvfIndex, store: VectorStorage) -> None:
    first = [index.postings_in(cluster) for cluster in range(PARTITIONS)]
    IvfIndex.train(
        store, metric=DistanceMetric.L2, dim=DIM, partitions=PARTITIONS, seed=11
    )
    second = [index.postings_in(cluster) for cluster in range(PARTITIONS)]
    assert second == first
    assert sum(len(part) for part in second) == ROWS


def test_a_high_dimension_corpus_trains_and_prunes(tmp_path: Path) -> None:
    dim = 1536
    handle = VectorStorage.open(tmp_path / "db")
    try:
        generator = np.random.default_rng(4)
        grid = generator.normal(size=(64, dim)).astype("<f4")
        for row, row_vector in enumerate(grid):
            handle.put_vector(f"v{row:03d}", encode_vector(row_vector))
        trained = IvfIndex.train(
            handle, metric=DistanceMetric.COSINE, dim=dim, partitions=4, seed=2
        )
        outcome = trained.search(grid[5], 5, nprobe=1)
        assert outcome.partitions == 4
        assert outcome.buckets_scanned == 1
        assert 0 < outcome.postings_read <= 64
    finally:
        handle.close()


def test_vector_namespace_is_never_scanned_during_search(
    index: IvfIndex, store: VectorStorage
) -> None:
    seen = spy_scans(store)
    index.search(index.centroids[0], 5, nprobe=1)
    assert VECTOR_PREFIX not in seen
    assert CENTROID_PREFIX not in seen
