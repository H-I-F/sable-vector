"""Crash-recovery contracts: every derived key is recomputable, so prove it.

The engine's crash story rests on one claim — that everything in the tree except
the payloads is a *function* of the payloads, and can therefore be rebuilt
rather than trusted. These tests attack that claim from both sides.

From the outside, by killing a process: a writer is SIGKILLed part-way through a
batch, and a trainer is SIGKILLed part-way through writing postings. Both leave
whatever the WAL happened to hold; the recovery pass has to accept that state
and converge, not assume a shape.

From the inside, by surgery on the key space: a posting is deleted, an orphan
posting is injected, an id is written into two partitions, the metadata key is
removed, a centroid is dropped. Each is a corruption that real code paths cannot
produce, which is exactly why it is worth testing — a bug in the write path
produces precisely this kind of state, and the recovery pass is the last thing
standing between it and a silently wrong answer.

The properties asserted are the ones that make recovery safe to run on every
open rather than on suspicion: it is **idempotent** (recovering a healthy store
reports zero), it **converges** (after it, every committed id has exactly one
posting and every posting names a committed id), and it **measures before it
decides** (small drift is spliced, large drift is retrained, and :func:`inspect`
can see the difference without writing).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from sable_vector.codec import encode_vector
from sable_vector.engine import VectorEngine
from sable_vector.index.ivf_storage import IVF_META_KEY, IvfIndex, posting_prefix
from sable_vector.recovery import (
    DEFAULT_REBUILD_RATIO,
    IndexHealth,
    clear_namespace,
    inspect_ivf,
    rebuild_ivf,
    recover,
)
from sable_vector.storage import (
    CENTROID_PREFIX,
    POSTING_PREFIX,
    VECTOR_PREFIX,
    VectorStorage,
)
from sable_vector.types import DistanceMetric, VectorId, VectorRecord

DIM = 12
ROWS = 24
PARTITIONS = 4
FIELDS = ("tenant",)


def vectors(rows: int = ROWS, dim: int = DIM, *, seed: int = 8) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(rows, dim)).astype("<f4")


def names(count: int = ROWS) -> list[VectorId]:
    return [f"doc-{index:04d}" for index in range(count)]


def payloads(count: int = ROWS) -> list[dict[str, object]]:
    return [
        {"tenant": "acme" if index % 2 == 0 else "globex", "rank": index}
        for index in range(count)
    ]


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    return tmp_path / "db"


@pytest.fixture()
def engine(home: Path) -> Iterator[VectorEngine]:
    handle = VectorEngine.open(
        home, metric=DistanceMetric.L2, dim=DIM, indexed_fields=FIELDS
    )
    grid = vectors()
    handle.upsert_many(
        VectorRecord(name, grid[row], payloads()[row])
        for row, name in enumerate(names())
    )
    yield handle
    handle.close()


@pytest.fixture()
def trained(engine: VectorEngine) -> IvfIndex:
    return IvfIndex.train(
        engine.storage,
        metric=engine.metric,
        dim=DIM,
        partitions=PARTITIONS,
        seed=11,
    )


def health_of(handle: VectorEngine) -> IndexHealth:
    return inspect_ivf(handle.storage, metric=handle.metric, dim=DIM)


def posting_ids(handle: VectorEngine) -> list[VectorId]:
    seen: list[VectorId] = []
    for cluster in range(PARTITIONS):
        prefix = posting_prefix(cluster)
        for key, _ in handle.storage.scan_prefix(prefix):
            seen.append(key[len(prefix) :].decode("utf-8").replace("\\:", ":"))
    return seen


# -- namespace clearing ----------------------------------------------------


def test_clear_namespace_removes_every_key_under_a_prefix(engine: VectorEngine) -> None:
    IvfIndex.train(
        engine.storage, metric=engine.metric, dim=DIM, partitions=PARTITIONS, seed=11
    )
    before = sum(1 for _ in engine.storage.scan_prefix(POSTING_PREFIX))
    removed = clear_namespace(engine.storage, POSTING_PREFIX)
    assert removed == before == ROWS
    assert sum(1 for _ in engine.storage.scan_prefix(POSTING_PREFIX)) == 0


def test_clear_namespace_leaves_other_namespaces_alone(engine: VectorEngine) -> None:
    IvfIndex.train(
        engine.storage, metric=engine.metric, dim=DIM, partitions=PARTITIONS, seed=11
    )
    clear_namespace(engine.storage, POSTING_PREFIX)
    assert sum(1 for _ in engine.storage.scan_prefix(VECTOR_PREFIX)) == ROWS
    assert sum(1 for _ in engine.storage.scan_prefix(CENTROID_PREFIX)) == PARTITIONS


def test_clear_namespace_on_an_empty_prefix_reports_zero(engine: VectorEngine) -> None:
    assert clear_namespace(engine.storage, POSTING_PREFIX) == 0


# -- inspection ------------------------------------------------------------


def test_inspect_reports_an_untrained_store_as_a_state_not_an_error(
    engine: VectorEngine,
) -> None:
    health = health_of(engine)
    assert health.trained is False
    assert health.vectors == ROWS
    assert health.drift == 0
    assert health.metadata is None


def test_inspect_reports_a_healthy_trained_index(trained: IvfIndex, engine: VectorEngine) -> None:
    health = health_of(engine)
    assert health.trained is True
    assert health.partitions == PARTITIONS
    assert health.postings == ROWS
    assert health.missing == ()
    assert health.stale == ()
    assert health.duplicated == ()
    assert health.healthy is True


def test_inspect_detects_a_deleted_posting(trained: IvfIndex, engine: VectorEngine) -> None:
    victim = posting_ids(engine)[3]
    for cluster in range(PARTITIONS):
        engine.storage.delete_raw(posting_prefix(cluster) + victim.encode("utf-8"))
    health = health_of(engine)
    assert health.missing == (victim,)
    assert health.stale == ()


def test_inspect_detects_an_orphan_posting(trained: IvfIndex, engine: VectorEngine) -> None:
    engine.storage.put_raw(
        posting_prefix(0) + b"ghost", np.zeros(DIM, dtype="<f4").tobytes()
    )
    health = health_of(engine)
    assert health.stale == ("ghost",)
    assert health.missing == ()


def test_inspect_detects_an_id_in_two_partitions(
    trained: IvfIndex, engine: VectorEngine
) -> None:
    victim = posting_ids(engine)[2]
    engine.storage.put_raw(
        posting_prefix((0 + 1) % PARTITIONS) + victim.encode("utf-8"),
        np.zeros(DIM, dtype="<f4").tobytes(),
    )
    health = health_of(engine)
    assert health.duplicated == (victim,)


def test_inspect_reports_missing_and_stale_sorted(trained: IvfIndex, engine: VectorEngine) -> None:
    held = posting_ids(engine)
    for victim in (held[9], held[1], held[5]):
        for cluster in range(PARTITIONS):
            engine.storage.delete_raw(posting_prefix(cluster) + victim.encode("utf-8"))
    health = health_of(engine)
    assert list(health.missing) == sorted(health.missing)
    assert set(health.missing) == {held[9], held[1], held[5]}


# -- the retrain decision --------------------------------------------------


def test_absent_metadata_forces_a_retrain(trained: IvfIndex, engine: VectorEngine) -> None:
    engine.storage.delete_raw(IVF_META_KEY)
    assert health_of(engine).needs_retrain() is True


def test_a_dimension_mismatch_forces_a_retrain(
    trained: IvfIndex, engine: VectorEngine
) -> None:
    meta = json.loads(engine.storage.get_raw(IVF_META_KEY).decode("utf-8"))
    meta["dim"] = DIM + 1
    engine.storage.put_raw(IVF_META_KEY, json.dumps(meta).encode("utf-8"))
    assert health_of(engine).needs_retrain() is True


def test_a_partition_count_mismatch_forces_a_retrain(
    trained: IvfIndex, engine: VectorEngine
) -> None:
    meta = json.loads(engine.storage.get_raw(IVF_META_KEY).decode("utf-8"))
    meta["k"] = PARTITIONS + 3
    engine.storage.put_raw(IVF_META_KEY, json.dumps(meta).encode("utf-8"))
    assert health_of(engine).needs_retrain() is True


def test_a_duplicate_always_forces_a_retrain(trained: IvfIndex, engine: VectorEngine) -> None:
    victim = posting_ids(engine)[2]
    engine.storage.put_raw(
        posting_prefix(1) + victim.encode("utf-8"), b"\x00" * (DIM * 4)
    )
    assert health_of(engine).needs_retrain() is True


def test_drift_below_the_ratio_is_spliced(trained: IvfIndex, engine: VectorEngine) -> None:
    held = posting_ids(engine)
    for cluster in range(PARTITIONS):
        engine.storage.delete_raw(posting_prefix(cluster) + held[0].encode("utf-8"))
    health = health_of(engine)
    assert health.drift == 1
    assert health.needs_retrain(DEFAULT_REBUILD_RATIO) is False


def test_drift_above_the_ratio_is_retrained(trained: IvfIndex, engine: VectorEngine) -> None:
    held = posting_ids(engine)
    doomed = held[: ROWS // 2]
    for victim in doomed:
        for cluster in range(PARTITIONS):
            engine.storage.delete_raw(posting_prefix(cluster) + victim.encode("utf-8"))
    health = health_of(engine)
    assert health.drift > DEFAULT_REBUILD_RATIO * health.vectors
    assert health.needs_retrain(DEFAULT_REBUILD_RATIO) is True


def test_the_ratio_boundary_is_exclusive(trained: IvfIndex, engine: VectorEngine) -> None:
    held = posting_ids(engine)
    quarter = ROWS // 4
    for victim in held[:quarter]:
        for cluster in range(PARTITIONS):
            engine.storage.delete_raw(posting_prefix(cluster) + victim.encode("utf-8"))
    exactly = health_of(engine)
    assert exactly.drift == quarter == DEFAULT_REBUILD_RATIO * exactly.vectors
    assert exactly.needs_retrain(DEFAULT_REBUILD_RATIO) is False
    assert exactly.needs_retrain(DEFAULT_REBUILD_RATIO - 0.01) is True


# -- recovery: idempotence and torn records --------------------------------


def test_recovering_a_healthy_store_changes_nothing(
    trained: IvfIndex, engine: VectorEngine
) -> None:
    report = recover(engine, partitions=PARTITIONS, seed=11)
    assert report.total == 0
    assert report.repaired is False
    assert health_of(engine).healthy is True


def test_recovery_is_idempotent(trained: IvfIndex, engine: VectorEngine) -> None:
    engine.storage.delete_raw(IVF_META_KEY)
    first = recover(engine, partitions=PARTITIONS, seed=11)
    second = recover(engine, partitions=PARTITIONS, seed=11)
    assert first.retrained is True
    assert second.total == 0
    assert second.retrained is False


def test_recover_reclaims_a_vector_whose_payload_never_landed(home: Path) -> None:
    storage = VectorStorage.open(home)
    handle = VectorEngine(storage, metric=DistanceMetric.L2, dim=DIM, indexed_fields=FIELDS)
    grid = vectors()
    for row, name in enumerate(names(3)):
        handle.upsert(name, grid[row], payloads()[row])
    # A torn write: the image reached the log, the commit record did not.
    storage.put_vector("torn", encode_vector(grid[3]), sync=True)
    report = recover(handle, partitions=2, seed=1, train=False)
    assert report.torn_vectors == 1
    assert "torn" not in set(storage.iter_vector_ids())
    handle.close()


def test_recover_reclaims_a_payload_with_no_vector(home: Path) -> None:
    storage = VectorStorage.open(home)
    handle = VectorEngine(storage, metric=DistanceMetric.L2, dim=DIM, indexed_fields=FIELDS)
    grid = vectors()
    for row, name in enumerate(names(3)):
        handle.upsert(name, grid[row], payloads()[row])
    storage.put_payload("orphan", {"tenant": "acme"}, sync=True)
    report = recover(handle, partitions=2, seed=1, train=False)
    assert report.torn_payloads == 1
    assert "orphan" not in {name for name, _ in storage.iter_payloads()}
    handle.close()


def test_recover_rebuilds_the_inverted_entries_of_a_torn_record(home: Path) -> None:
    storage = VectorStorage.open(home)
    handle = VectorEngine(storage, metric=DistanceMetric.L2, dim=DIM, indexed_fields=FIELDS)
    grid = vectors()
    handle.upsert("doc-0000", grid[0], {"tenant": "acme"})
    storage.delete_raw(b"idx:tenant:acme:doc-0000", sync=True)
    report = recover(handle, partitions=2, seed=1, train=False)
    assert report.rebuilt_index_keys == 1
    assert list(storage.scan_index("tenant", "acme")) == ["doc-0000"]
    handle.close()


def test_train_false_leaves_a_drifted_index_alone(trained: IvfIndex, engine: VectorEngine) -> None:
    engine.storage.delete_raw(IVF_META_KEY)
    report = recover(engine, partitions=PARTITIONS, seed=11, train=False)
    assert report.retrained is False
    assert health_of(engine).metadata is None


def test_recover_without_a_dimension_skips_the_ivf_pass(home: Path) -> None:
    handle = VectorEngine(VectorStorage.open(home), metric=DistanceMetric.L2, indexed_fields=FIELDS)
    assert handle.dim is None
    report = recover(handle, partitions=PARTITIONS)
    assert report.total == 0
    assert report.retrained is False
    handle.close()


# -- recovery: splicing small drift ----------------------------------------


def test_small_drift_is_spliced_without_retraining(
    trained: IvfIndex, engine: VectorEngine
) -> None:
    held = posting_ids(engine)
    for cluster in range(PARTITIONS):
        engine.storage.delete_raw(posting_prefix(cluster) + held[4].encode("utf-8"))
    report = recover(engine, partitions=PARTITIONS, seed=11)
    assert report.retrained is False
    assert report.missing_postings == 1
    assert health_of(engine).healthy is True


def test_splicing_puts_the_vector_back_in_its_nearest_partition(
    trained: IvfIndex, engine: VectorEngine
) -> None:
    held = posting_ids(engine)
    victim = held[4]
    for cluster in range(PARTITIONS):
        engine.storage.delete_raw(posting_prefix(cluster) + victim.encode("utf-8"))
    recover(engine, partitions=PARTITIONS, seed=11)
    index = IvfIndex.load(engine.storage, metric=engine.metric, dim=DIM)
    row = names().index(victim)
    where = next(
        cluster
        for cluster in range(PARTITIONS)
        if victim in index.postings_in(cluster)
    )
    assert where == index.cluster_of(vectors()[row])


def test_a_stale_posting_is_dropped_not_adopted(trained: IvfIndex, engine: VectorEngine) -> None:
    engine.storage.put_raw(posting_prefix(1) + b"ghost", b"\x00" * (DIM * 4), sync=True)
    report = recover(engine, partitions=PARTITIONS, seed=11)
    assert report.stale_postings == 1
    assert b"ghost" not in [key for key, _ in engine.storage.scan_prefix(POSTING_PREFIX)]
    assert health_of(engine).healthy is True


def test_recovering_after_a_delete_drops_the_dead_posting(
    trained: IvfIndex, engine: VectorEngine
) -> None:
    engine.delete("doc-0003")
    health = health_of(engine)
    assert health.stale == ("doc-0003",)
    report = recover(engine, partitions=PARTITIONS, seed=11)
    assert report.stale_postings == 1
    assert health_of(engine).healthy is True


def test_recovering_after_an_upsert_adds_the_new_posting(
    trained: IvfIndex, engine: VectorEngine
) -> None:
    engine.upsert("fresh", vectors()[5], {"tenant": "acme"})
    report = recover(engine, partitions=PARTITIONS, seed=11)
    assert report.missing_postings == 1
    index = IvfIndex.load(engine.storage, metric=engine.metric, dim=DIM)
    assert "fresh" in [name for cluster in range(PARTITIONS) for name in index.postings_in(cluster)]


# -- recovery: rebuilding large drift --------------------------------------


def test_removed_metadata_is_rebuilt_by_retraining(
    trained: IvfIndex, engine: VectorEngine
) -> None:
    engine.storage.delete_raw(IVF_META_KEY)
    report = recover(engine, partitions=PARTITIONS, seed=11)
    assert report.retrained is True
    assert health_of(engine).healthy is True


def test_dropped_centroids_are_rebuilt_by_retraining(
    trained: IvfIndex, engine: VectorEngine
) -> None:
    clear_namespace(engine.storage, CENTROID_PREFIX)
    assert health_of(engine).trained is False
    report = recover(engine, partitions=PARTITIONS, seed=11)
    assert report.retrained is True
    assert health_of(engine).healthy is True


def test_large_drift_is_rebuilt_rather_than_spliced(
    trained: IvfIndex, engine: VectorEngine
) -> None:
    held = posting_ids(engine)
    for victim in held[: ROWS // 2]:
        for cluster in range(PARTITIONS):
            engine.storage.delete_raw(posting_prefix(cluster) + victim.encode("utf-8"))
    report = recover(engine, partitions=PARTITIONS, seed=11)
    assert report.retrained is True
    assert report.missing_postings == len(held) // 2
    assert health_of(engine).healthy is True


def test_rebuild_ivf_replaces_every_partition(trained: IvfIndex, engine: VectorEngine) -> None:
    before = [IvfIndex.load(engine.storage, metric=engine.metric, dim=DIM).postings_in(c) for c in range(PARTITIONS)]
    fresh = rebuild_ivf(
        engine.storage, metric=engine.metric, dim=DIM, partitions=PARTITIONS, seed=11
    )
    after = [fresh.postings_in(cluster) for cluster in range(PARTITIONS)]
    assert sorted(x for part in after for x in part) == sorted(x for part in before for x in part)
    assert fresh.metadata()["k"] == PARTITIONS


def test_an_emptied_corpus_is_left_without_an_index(trained: IvfIndex, engine: VectorEngine) -> None:
    for name in names():
        engine.delete(name)
    report = recover(engine, partitions=PARTITIONS, seed=11)
    assert report.retrained is True
    assert health_of(engine).trained is False
    assert sum(1 for _ in engine.storage.scan_prefix(POSTING_PREFIX)) == 0


def test_a_recovered_index_still_prunes(trained: IvfIndex, engine: VectorEngine) -> None:
    engine.storage.delete_raw(IVF_META_KEY)
    recover(engine, partitions=PARTITIONS, seed=11)
    index = IvfIndex.load(engine.storage, metric=engine.metric, dim=DIM)
    outcome = index.search(index.centroids[2], 5, nprobe=1)
    assert outcome.buckets_scanned == 1
    assert set(index.postings_in(2)) >= {match.id for match in outcome.matches}


def test_the_report_totals_what_it_counts(trained: IvfIndex, engine: VectorEngine) -> None:
    engine.storage.put_raw(posting_prefix(0) + b"ghost", b"\x00" * (DIM * 4), sync=True)
    held = posting_ids(engine)
    for cluster in range(PARTITIONS):
        engine.storage.delete_raw(posting_prefix(cluster) + held[1].encode("utf-8"))
    report = recover(engine, partitions=PARTITIONS, seed=11)
    assert report.total == report.stale_postings + report.missing_postings == 2
    assert report.repaired is True


# -- real crashes ----------------------------------------------------------

_CRASH_WRITER = textwrap.dedent(
    """
    import os, signal, sys
    from pathlib import Path
    import numpy as np
    from sable_vector.engine import VectorEngine
    from sable_vector.types import DistanceMetric

    home = Path(sys.argv[1])
    total, kill_at, marker, dim = int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5])
    handle = VectorEngine.open(
        home, metric=DistanceMetric.L2, dim=dim, indexed_fields=["tenant"]
    )
    grid = np.random.default_rng(8).normal(size=(total, dim)).astype("<f4")
    with marker.open("w", buffering=1) as log:
        for row in range(total):
            handle.upsert(f"doc-{row:04d}", grid[row], {"tenant": "acme" if row % 2 == 0 else "globex"})
            log.write(f"doc-{row:04d}\\n")
            log.flush()
            os.fsync(log.fileno())
            if row == kill_at:
                os.kill(os.getpid(), signal.SIGKILL)
    handle.close()
    """
)

_CRASH_TRAINER = textwrap.dedent(
    """
    import os, signal, sys
    from pathlib import Path
    import numpy as np
    from sable_vector.engine import VectorEngine
    from sable_vector.index.ivf_storage import IvfIndex
    from sable_vector.storage import VectorStorage
    from sable_vector.types import DistanceMetric

    home, limit, dim = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    grid = np.random.default_rng(3).normal(size=(48, dim)).astype("<f4")
    seed = VectorStorage.open(home)
    for row in range(48):
        VectorEngine(seed, metric=DistanceMetric.L2, dim=dim).upsert(f"v{row:03d}", grid[row], {})
    seed.flush()
    seed.close()

    storage = VectorStorage.open(home)
    written = 0
    original = storage.put_raw

    def tripwire(key, value, *, sync=True):
        global written
        written += 1
        original(key, value, sync=sync)
        if written > limit:
            os.kill(os.getpid(), signal.SIGKILL)

    storage.put_raw = tripwire
    IvfIndex.train(storage, metric=DistanceMetric.L2, dim=dim, partitions=4, seed=5)
    """
)

_SRC = str(Path(__file__).resolve().parents[1] / "src")


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, PYTHONPATH=_SRC)
    return subprocess.run(
        [sys.executable, "-c", script, *args],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )


def test_a_writer_killed_mid_batch_recovers_to_exactly_what_was_acked(
    tmp_path: Path,
) -> None:
    home = tmp_path / "db"
    marker = tmp_path / "acked.txt"
    total, kill_at = 40, 21
    result = _run(_CRASH_WRITER, str(home), str(total), str(kill_at), str(marker), str(DIM))
    assert result.returncode == -signal.SIGKILL, result.stderr

    acked = set(marker.read_text(encoding="utf-8").split())
    assert len(acked) == kill_at + 1

    storage = VectorStorage.open(home)
    handle = VectorEngine(storage, metric=DistanceMetric.L2, dim=DIM, indexed_fields=FIELDS)
    report = recover(handle, partitions=PARTITIONS, seed=11)

    present = {name for name, _ in storage.iter_payloads()}
    # Everything acknowledged before the kill was fsynced, so all of it survived.
    assert acked <= present
    # The process died between two writes; at most one record beyond the last
    # acknowledgement can have reached stable storage.
    assert len(present - acked) <= 1
    # No torn record survives: a vector exists if and only if its payload does.
    assert set(storage.iter_vector_ids()) == present
    assert report.torn_vectors == 0
    assert report.torn_payloads == 0
    handle.close()


def test_a_writer_killed_mid_batch_keeps_the_inverted_consistent(tmp_path: Path) -> None:
    home = tmp_path / "db"
    marker = tmp_path / "acked.txt"
    result = _run(_CRASH_WRITER, str(home), "30", "17", str(marker), str(DIM))
    assert result.returncode == -signal.SIGKILL, result.stderr

    handle = VectorEngine.open(
        home, metric=DistanceMetric.L2, dim=DIM, indexed_fields=FIELDS
    )
    try:
        recover(handle, partitions=PARTITIONS, seed=11)
        indexed = {name for _, name in handle.storage.scan_index_field("tenant")}
        assert indexed == {name for name, _ in handle.storage.iter_payloads()}
    finally:
        handle.close()


def test_a_trainer_killed_mid_rebuild_converges(tmp_path: Path) -> None:
    home = tmp_path / "db"
    result = _run(_CRASH_TRAINER, str(home), "6", str(DIM))
    assert result.returncode == -signal.SIGKILL, result.stderr

    storage = VectorStorage.open(home)
    handle = VectorEngine(storage, metric=DistanceMetric.L2, dim=DIM)
    before = inspect_ivf(storage, metric=DistanceMetric.L2, dim=DIM)
    # Whatever the WAL happened to hold, the half-written index cannot claim to
    # be complete: either nothing of it survived at all, or it survived without
    # the metadata that would make it usable. Both demand a rebuild.
    assert before.needs_retrain() is True

    report = recover(handle, partitions=PARTITIONS, seed=5)
    after = inspect_ivf(storage, metric=DistanceMetric.L2, dim=DIM)
    assert report.retrained is True
    assert after.healthy is True
    assert after.vectors == 48
    assert after.postings == 48
    handle.close()


def test_a_killed_writer_leaves_the_store_openable_repeatedly(tmp_path: Path) -> None:
    home = tmp_path / "db"
    marker = tmp_path / "acked.txt"
    result = _run(_CRASH_WRITER, str(home), "20", "9", str(marker), str(DIM))
    assert result.returncode == -signal.SIGKILL, result.stderr

    counts: list[int] = []
    for _ in range(3):
        handle = VectorEngine.open(
            home, metric=DistanceMetric.L2, dim=DIM, indexed_fields=FIELDS
        )
        recover(handle, partitions=PARTITIONS, seed=11)
        counts.append(sum(1 for _ in handle.storage.iter_payloads()))
        handle.close()
    assert len(set(counts)) == 1


def test_recovery_survives_a_corrupted_metadata_blob(
    trained: IvfIndex, engine: VectorEngine
) -> None:
    engine.storage.put_raw(IVF_META_KEY, b"not-json-at-all", sync=True)
    report = recover(engine, partitions=PARTITIONS, seed=11)
    assert report.retrained is True
    assert health_of(engine).healthy is True


def test_recovery_handles_a_posting_of_the_wrong_width(
    trained: IvfIndex, engine: VectorEngine
) -> None:
    engine.storage.put_raw(posting_prefix(0) + b"short", b"\x00" * 8, sync=True)
    report = recover(engine, partitions=PARTITIONS, seed=11)
    assert report.retrained is True
    assert health_of(engine).healthy is True


def test_recovery_handles_a_centroid_of_the_wrong_width(
    trained: IvfIndex, engine: VectorEngine
) -> None:
    engine.storage.put_raw(b"c:000000", b"\x00" * 8, sync=True)
    report = recover(engine, partitions=PARTITIONS, seed=11)
    assert report.retrained is True
    assert health_of(engine).healthy is True


def test_recovering_a_fresh_store_is_a_no_op(home: Path) -> None:
    handle = VectorEngine.open(home, metric=DistanceMetric.L2, dim=DIM)
    health = health_of(handle)
    assert health.trained is False
    assert health.vectors == 0
    assert health.drift == 0
    # Nothing was ever written, so there is nothing to recompute — a fresh store
    # is not a damaged one, and reporting work here would make the report
    # useless as a signal.
    report = recover(handle, partitions=PARTITIONS)
    assert report.retrained is False
    assert report.total == 0
    handle.close()
