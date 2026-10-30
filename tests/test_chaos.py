"""Chaos injection: five ways to die, and what must still be true afterwards.

A crash test that kills a process once proves the recovery path runs. It does
not prove the recovery path *converges*, because one kill at one moment leaves
one state, and the interesting states are the ones you did not happen to hit.
So this module kills the process in five structurally different places, and then
kills it five times in a row on the same database, because a store that recovers
from the first crash and not from the fourth is not a store.

The five places are chosen to land on different parts of the write path:

1. **mid-ingest** — between two acknowledged records, so the log holds a whole
   number of records and nothing partial;
2. **mid-batch** — inside one `upsert_many`, where a batch shares a single fsync
   boundary and therefore dies with frames buffered and no commit record;
3. **mid-training** — inside an IVF rebuild, which writes unsynced derived keys
   and commits with one metadata key;
4. **mid-delete** — between payload-first and vector-last, the ordering that
   makes a torn delete detectable rather than silent;
5. **mid-recovery** — inside the recovery pass itself, so a repair interrupted
   by another crash has to be idempotent enough to finish on the next open.

Every kill is a real `SIGKILL`: no signal handler runs, no `finally` block runs,
no buffer is flushed by the language runtime. What survives is what the operating
system already had.

What is asserted is deliberately narrow and deliberately strong: the store
reopens, nothing acknowledged before the kill is lost, every surviving vector has
a payload and every payload has a vector, the inverted index agrees with the
payloads, and the derived index converges — including after five crashes in a
row. Nothing here asserts a particular number of records, because how many
survive is up to the storage engine; that all of them are whole is not.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from sable_vector.codec import decode_vector
from sable_vector.engine import VectorEngine
from sable_vector.index.ivf_storage import IvfIndex
from sable_vector.recovery import inspect_ivf, recover
from sable_vector.storage import CENTROID_PREFIX, POSTING_PREFIX, VECTOR_PREFIX
from sable_vector.types import DistanceMetric

DIM = 16
PARTITIONS = 4
SEED_ROWS = 120
JOIN_TIMEOUT = 300.0

MODES = ("ingest", "batch", "train", "delete", "recover")


_CHAOS = textwrap.dedent(
    """
    import json, os, signal, sys
    from pathlib import Path
    import numpy as np
    from sable_vector.engine import VectorEngine
    from sable_vector.index.ivf_storage import IvfIndex
    from sable_vector.recovery import recover
    from sable_vector.storage import VectorStorage
    from sable_vector.types import DistanceMetric, VectorRecord

    home = Path(sys.argv[1])
    mode = sys.argv[2]
    limit = int(sys.argv[3])
    marker = Path(sys.argv[4])
    dim = int(sys.argv[5])
    partitions = int(sys.argv[6])

    grid = np.random.default_rng(7).normal(size=(600, dim)).astype("<f4")

    def name(row):
        return f"doc-{row:05d}"

    def ack(record):
        with marker.open("a", buffering=1) as log:
            log.write(json.dumps(record) + "\\n")
            log.flush()
            os.fsync(log.fileno())

    def open_engine():
        return VectorEngine.open(
            home, metric=DistanceMetric.L2, dim=dim, indexed_fields=["tenant"]
        )

    def die():
        # No handler, no finally, no runtime flush: whatever the kernel already
        # has is what the next open gets to work with.
        os.kill(os.getpid(), signal.SIGKILL)

    def tripwire(storage, budget):
        written = 0
        original = storage.put_raw

        def counted(key, value, *, sync=True):
            nonlocal written
            written += 1
            original(key, value, sync=sync)
            if written > budget:
                die()

        storage.put_raw = counted

    if mode == "ingest":
        engine = open_engine()
        for row in range(limit + 500):
            engine.upsert(name(row), grid[row], {"tenant": "acme" if row % 2 == 0 else "globex"})
            ack({"op": "put", "id": name(row)})
            if row == limit:
                die()
        engine.close()

    elif mode == "batch":
        engine = open_engine()
        storage = engine.storage
        tripwire(storage, limit)
        engine.upsert_many(
            VectorRecord(name(row), grid[row], {"tenant": "acme"}) for row in range(400)
        )
        engine.close()

    elif mode == "train":
        seed = open_engine()
        for row in range(SEED_ROWS_HERE):
            seed.upsert(name(row), grid[row], {"tenant": "acme"})
        seed.storage.flush()
        seed.close()
        storage = VectorStorage.open(home)
        engine = VectorEngine(storage, metric=DistanceMetric.L2, dim=dim, indexed_fields=["tenant"])
        tripwire(storage, limit)
        IvfIndex.train(storage, metric=DistanceMetric.L2, dim=dim, partitions=partitions, seed=5)

    elif mode == "delete":
        seed = open_engine()
        for row in range(SEED_ROWS_HERE):
            seed.upsert(name(row), grid[row], {"tenant": "acme"})
        seed.storage.flush()
        seed.close()
        engine = open_engine()
        for row in range(limit + 500):
            engine.delete(name(row))
            ack({"op": "delete", "id": name(row)})
            if row == limit:
                die()
        engine.close()

    elif mode == "recover":
        seed = open_engine()
        for row in range(SEED_ROWS_HERE):
            seed.upsert(name(row), grid[row], {"tenant": "acme"})
        seed.storage.flush()
        seed.close()
        storage = VectorStorage.open(home)
        engine = VectorEngine(storage, metric=DistanceMetric.L2, dim=dim, indexed_fields=["tenant"])
        tripwire(storage, limit)
        recover(engine, partitions=partitions, seed=5)

    else:
        raise SystemExit(f"unknown mode {mode}")
    """
)

_SRC = str(Path(__file__).resolve().parents[1] / "src")


def crash(
    home: Path, mode: str, limit: int, marker: Path
) -> subprocess.CompletedProcess[str]:
    """Run one killable child and require it to die by SIGKILL."""
    script = _CHAOS.replace("SEED_ROWS_HERE", str(SEED_ROWS))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(home),
            mode,
            str(limit),
            str(marker),
            str(DIM),
            str(PARTITIONS),
        ],
        capture_output=True,
        text=True,
        timeout=JOIN_TIMEOUT,
        env=dict(os.environ, PYTHONPATH=_SRC),
    )
    assert result.returncode == -signal.SIGKILL, (
        f"{mode} exited {result.returncode}, expected SIGKILL; stderr:\\n{result.stderr}"
    )
    return result


def acked(marker: Path) -> list[dict[str, str]]:
    if not marker.exists():
        return []
    return [
        json.loads(line)
        for line in marker.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def assert_whole(engine: VectorEngine) -> None:
    """A record is whole or absent: never a half of one."""
    storage = engine.storage
    vectors = set(storage.iter_vector_ids())
    payloads = {vector_id for vector_id, _ in storage.iter_payloads()}
    assert vectors == payloads, f"torn records: {vectors ^ payloads}"
    indexed = {vector_id for _, vector_id in storage.scan_index_field("tenant")}
    assert indexed == payloads, f"index drift: {indexed ^ payloads}"


def assert_converges(engine: VectorEngine) -> None:
    """Recovery must leave the derived index agreeing with the corpus."""
    recover(engine, partitions=PARTITIONS, seed=5)
    health = inspect_ivf(engine.storage, metric=engine.metric, dim=DIM)
    assert health.healthy, health
    # And a second pass must find nothing left to do, which is what makes it
    # safe to run on every open rather than on suspicion.
    assert recover(engine, partitions=PARTITIONS, seed=5).total == 0


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    return tmp_path / "chaos"


# -- one kill in each structurally different place --------------------------


def test_kill_during_ingest_keeps_every_acknowledged_record(
    home: Path, tmp_path: Path
) -> None:
    marker = tmp_path / "ingest.jsonl"
    crash(home, "ingest", 40, marker)
    entries = acked(marker)
    assert len(entries) == 41

    engine = VectorEngine.open(
        home, metric=DistanceMetric.L2, dim=DIM, indexed_fields=("tenant",)
    )
    try:
        present = {vector_id for vector_id, _ in engine.storage.iter_payloads()}
        assert {item["id"] for item in entries} <= present
        assert len(present) - len(entries) <= 1
        assert_whole(engine)
        assert_converges(engine)
    finally:
        engine.close()


def test_kill_inside_a_batch_is_all_or_nothing(home: Path, tmp_path: Path) -> None:
    # A batch shares one fsync boundary: only its final payload is synced, and
    # that one fsync carries every frame buffered before it. Killing the process
    # part-way through the batch therefore cannot leave a half of it — the
    # durable states are "none of it" and "all of it", and nothing between.
    crash(home, "batch", 850, tmp_path / "batch.jsonl")

    engine = VectorEngine.open(
        home, metric=DistanceMetric.L2, dim=DIM, indexed_fields=("tenant",)
    )
    try:
        assert_whole(engine)
        assert_converges(engine)
        surviving = sum(1 for _ in engine.storage.iter_payloads())
        assert surviving in (0, 400), surviving
        for vector_id in (vector_id for vector_id, _ in engine.storage.iter_payloads()):
            assert engine.storage.get_vector_blob(vector_id) is not None
    finally:
        engine.close()


def test_kill_inside_a_training_run_converges(home: Path, tmp_path: Path) -> None:
    crash(home, "train", 7, tmp_path / "train.jsonl")

    engine = VectorEngine.open(
        home, metric=DistanceMetric.L2, dim=DIM, indexed_fields=("tenant",)
    )
    try:
        assert_whole(engine)
        assert_converges(engine)
        health = inspect_ivf(engine.storage, metric=engine.metric, dim=DIM)
        assert health.vectors == SEED_ROWS
        assert health.postings == SEED_ROWS
    finally:
        engine.close()


def test_kill_during_a_delete_sweep_converges(home: Path, tmp_path: Path) -> None:
    marker = tmp_path / "delete.jsonl"
    crash(home, "delete", 30, marker)
    entries = acked(marker)
    assert len(entries) == 31

    engine = VectorEngine.open(
        home, metric=DistanceMetric.L2, dim=DIM, indexed_fields=("tenant",)
    )
    try:
        surviving = {vector_id for vector_id, _ in engine.storage.iter_payloads()}
        # Deletion drops the payload first, so an acknowledged delete can never
        # come back: the payload is the commit record in both directions.
        assert not surviving & {item["id"] for item in entries}
        assert_whole(engine)
        assert_converges(engine)
        health = inspect_ivf(engine.storage, metric=engine.metric, dim=DIM)
        assert health.vectors == SEED_ROWS - len(entries)
    finally:
        engine.close()


def test_kill_inside_the_recovery_pass_itself(home: Path, tmp_path: Path) -> None:
    crash(home, "recover", 11, tmp_path / "recover.jsonl")

    engine = VectorEngine.open(
        home, metric=DistanceMetric.L2, dim=DIM, indexed_fields=("tenant",)
    )
    try:
        assert_whole(engine)
        # A repair interrupted by another crash has to be idempotent enough to
        # finish on the next open; running it twice must be a no-op after that.
        assert_converges(engine)
        second = recover(engine, partitions=PARTITIONS, seed=5)
        assert second.total == 0
        assert second.retrained is False
    finally:
        engine.close()


# -- five crashes in a row --------------------------------------------------


def test_five_crashes_in_a_row_still_converge(home: Path, tmp_path: Path) -> None:
    marker = tmp_path / "repeat.jsonl"
    seen: list[int] = []
    for round_index in range(5):
        crash(home, "ingest", 20 + round_index * 7, marker)
        engine = VectorEngine.open(
            home, metric=DistanceMetric.L2, dim=DIM, indexed_fields=("tenant",)
        )
        try:
            assert_whole(engine)
            assert_converges(engine)
            seen.append(sum(1 for _ in engine.storage.iter_payloads()))
        finally:
            engine.close()

    # Records only accumulate: a crash may lose the record in flight, never one
    # that was already acknowledged in an earlier round.
    assert seen == sorted(seen)
    assert seen[-1] >= seen[0]


def test_the_surviving_index_answers_queries_after_five_crashes(
    home: Path, tmp_path: Path
) -> None:
    marker = tmp_path / "queries.jsonl"
    for round_index in range(5):
        crash(home, "ingest", 15 + round_index * 5, marker)

    engine = VectorEngine.open(
        home, metric=DistanceMetric.L2, dim=DIM, indexed_fields=("tenant",)
    )
    try:
        assert_converges(engine)
        index = IvfIndex.load(engine.storage, metric=engine.metric, dim=DIM)
        first = next(iter(engine.storage.scan_prefix(VECTOR_PREFIX)))
        probe = decode_vector(first[1], dim=DIM)
        outcome = index.search(probe, 5, nprobe=PARTITIONS)
        assert outcome.matches
        # Every id the pruned index returns is a real, whole record.
        for match in outcome.matches:
            assert engine.storage.get_payload(match.id) is not None
            assert engine.storage.get_vector_blob(match.id) is not None
    finally:
        engine.close()


def test_a_crashed_store_never_leaks_a_centroid_without_postings(
    home: Path, tmp_path: Path
) -> None:
    for round_index in range(5):
        crash(home, "train", 3 + round_index, tmp_path / f"train{round_index}.jsonl")
        engine = VectorEngine.open(
            home, metric=DistanceMetric.L2, dim=DIM, indexed_fields=("tenant",)
        )
        try:
            assert_converges(engine)
            centroids = sum(1 for _ in engine.storage.scan_prefix(CENTROID_PREFIX))
            postings = sum(1 for _ in engine.storage.scan_prefix(POSTING_PREFIX))
            # Either the rebuild never committed and both are empty, or it did
            # and every row of the corpus has exactly one posting.
            assert (centroids, postings) in {(0, 0), (PARTITIONS, SEED_ROWS)}
        finally:
            engine.close()


def test_repeated_opens_after_a_crash_are_stable(home: Path, tmp_path: Path) -> None:
    crash(home, "ingest", 25, tmp_path / "stable.jsonl")
    counts: list[int] = []
    for _ in range(3):
        engine = VectorEngine.open(
            home, metric=DistanceMetric.L2, dim=DIM, indexed_fields=("tenant",)
        )
        try:
            recover(engine, partitions=PARTITIONS, seed=5)
            counts.append(sum(1 for _ in engine.storage.iter_payloads()))
        finally:
            engine.close()
    assert len(set(counts)) == 1


def test_a_crash_during_ingest_never_changes_the_collection_width(
    home: Path, tmp_path: Path
) -> None:
    crash(home, "ingest", 12, tmp_path / "width.jsonl")
    engine = VectorEngine.open(
        home, metric=DistanceMetric.L2, dim=DIM, indexed_fields=("tenant",)
    )
    try:
        assert engine.dim == DIM
        for _, blob in engine.storage.scan_prefix(
            VECTOR_PREFIX, limit=20
        ):
            assert len(blob) == DIM * 4
    finally:
        engine.close()


def test_the_kill_is_real_not_a_clean_exit(home: Path, tmp_path: Path) -> None:
    """A child that exits on its own would make every test above vacuous."""
    marker = tmp_path / "real.jsonl"
    outcome = crash(home, "ingest", 3, marker)
    # SIGKILL gives the child no chance to run a handler, a ``finally`` or an
    # atexit hook, so a clean exit or a caught exception would both be visible
    # here — as a different return code, or as a traceback on stderr.
    assert outcome.returncode == -signal.SIGKILL
    assert outcome.stderr == ""
    assert len(acked(marker)) == 4


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_converges(home: Path, mode: str, tmp_path: Path) -> None:
    """The backbone: each of the five ways to die leaves a whole, converged store."""
    crash(home, mode, 9, tmp_path / f"{mode}.jsonl")
    engine = VectorEngine.open(
        home, metric=DistanceMetric.L2, dim=DIM, indexed_fields=("tenant",)
    )
    try:
        assert_whole(engine)
        assert_converges(engine)
    finally:
        engine.close()
