"""Crash recovery and derived-index rebuild.

A crash leaves the tree in one of a small number of states, and this module's
job is to make that set *enumerable* rather than open-ended. Every derived key
in the store — the inverted entries, the IVF centroids, the posting lists — is a
pure function of the committed payloads. So the correct state after any crash is
computable, and the only question is whether to reach it by patching the drift
or by recomputing the structure.

That question is answered by measuring the drift, not by guessing:

* **Small drift is repaired in place.** A handful of missing or stale postings is
  cheaper to splice than to rebuild, and splicing cannot change the clustering
  a running query already depends on.
* **Large drift is rebuilt.** Once a quarter of the postings disagree with the
  corpus, the centroids themselves are stale — patching entries onto a partition
  map that no longer describes the data buys correctness of bookkeeping at the
  cost of recall, which is the wrong trade.

Two properties make the whole thing safe to run at open time rather than on
demand. It is **idempotent**: recovering a healthy store is a no-op that reports
zero, so running it on every open costs one pass and removes the question of
whether it was run. And it is **read-only until it decides**:
:func:`inspect_ivf` measures the drift without writing anything, so a caller can
see what is wrong before committing to the fix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from sable_vector.codec import decode_vector
from sable_vector.index.ivf_storage import (
    IVF_META_KEY,
    IvfIndex,
    posting_prefix,
)
from sable_vector.storage import (
    CENTROID_PREFIX,
    POSTING_PREFIX,
    VectorStorage,
    decode_id,
)
from sable_vector.types import DistanceMetric, VectorId

__all__ = [
    "DEFAULT_REBUILD_RATIO",
    "IndexHealth",
    "RecoveryReport",
    "clear_namespace",
    "inspect_ivf",
    "rebuild_ivf",
    "recover",
]

DEFAULT_REBUILD_RATIO = 0.25
"""Fraction of the corpus that may drift before the partitions are recomputed."""


@dataclass(frozen=True, slots=True)
class IndexHealth:
    """What is wrong with the IVF index, measured without writing anything.

    Attributes:
        trained: Centroids exist, so a partition map is present at all.
        dim: Vector width the caller is measuring against.
        partitions: Centroid count found on disk.
        vectors: Committed records in the corpus.
        postings: Posting entries on disk, including wrong ones.
        missing: Committed ids with no posting anywhere.
        stale: Posting ids that are not committed — survivors of a delete that
            was interrupted, or of a crash during a rebuild.
        duplicated: Ids appearing in more than one partition. One of them is
            correct and the rest are garbage, but which one cannot be known
            without recomputing, so this is always a rebuild trigger.
        metadata: Training metadata as stored, or ``None``.
        malformed: A stored image is not ``dim`` floats wide, or the metadata is
            not decodable. Nothing about such a structure can be trusted, so it
            is a rebuild trigger rather than an exception at load time.
    """

    trained: bool
    dim: int
    partitions: int = 0
    vectors: int = 0
    postings: int = 0
    missing: tuple[VectorId, ...] = ()
    stale: tuple[VectorId, ...] = ()
    duplicated: tuple[VectorId, ...] = ()
    metadata: dict[str, Any] | None = None
    malformed: bool = False

    @property
    def drift(self) -> int:
        """Postings that disagree with the corpus, in either direction."""
        return len(self.missing) + len(self.stale) + len(self.duplicated)

    @property
    def healthy(self) -> bool:
        """Whether the index agrees with the corpus and nothing is damaged.

        An empty corpus is healthy when it has *no* index: there is nothing to
        cluster, so a partition map would be debris rather than structure. A
        non-empty one is healthy only when it has one that matches.
        """
        if self.malformed or self.drift:
            return False
        if self.vectors == 0:
            return not self.trained
        return self.trained and self.metadata is not None

    def needs_retrain(self, rebuild_ratio: float = DEFAULT_REBUILD_RATIO) -> bool:
        """Whether the drift is big enough to recompute rather than splice."""
        if self.malformed:
            return True
        if not self.trained:
            # No partitions at all. With a corpus present that is an index that
            # was never built — or one whose rebuild was interrupted before its
            # first centroid survived — so it is trained from scratch. With an
            # empty corpus there is nothing to cluster and nothing to do.
            return self.vectors > 0
        if self.metadata is None:
            return True
        # An empty corpus has no partition map worth keeping, so anything still
        # on disk is debris and must go rather than be spliced onto nothing.
        if self.vectors == 0:
            return True
        if self.metadata.get("dim") != self.dim:
            return True
        if self.metadata.get("k") != self.partitions:
            return True
        if self.duplicated:
            return True
        return self.drift > rebuild_ratio * self.vectors


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    """What a recovery pass changed.

    Attributes:
        torn_vectors: Vector images whose payload never landed.
        torn_payloads: Payloads with no vector — impossible under the documented
            write order, and reported rather than assumed away.
        stale_index_keys: Inverted entries pointing at uncommitted ids.
        rebuilt_index_keys: Inverted entries a committed record was missing.
        missing_postings: Posting entries added for committed vectors.
        stale_postings: Posting entries dropped for uncommitted ids.
        retrained: ``True`` when the partitions were recomputed wholesale
            instead of spliced.
    """

    torn_vectors: int = 0
    torn_payloads: int = 0
    stale_index_keys: int = 0
    rebuilt_index_keys: int = 0
    missing_postings: int = 0
    stale_postings: int = 0
    retrained: bool = False

    @property
    def total(self) -> int:
        return (
            self.torn_vectors
            + self.torn_payloads
            + self.stale_index_keys
            + self.rebuilt_index_keys
            + self.missing_postings
            + self.stale_postings
        )

    @property
    def repaired(self) -> bool:
        return self.total > 0 or self.retrained


def clear_namespace(storage: VectorStorage, prefix: bytes) -> int:
    """Delete every key under ``prefix``.

    Keys are collected before any deletion because a scan over a range that is
    being mutated is not defined: a cursor that has already passed a key cannot
    observe it being removed, and one that has not may or may not see it.

    Returns:
        How many keys were deleted.
    """
    keys = [key for key, _ in storage.scan_prefix(prefix)]
    for key in keys:
        storage.delete_raw(key, sync=False)
    if keys:
        storage.flush()
    return len(keys)


def inspect_ivf(
    storage: VectorStorage, *, metric: DistanceMetric, dim: int
) -> IndexHealth:
    """Measure the IVF index against the committed corpus, without writing.

    Args:
        storage: Namespace adapter to inspect.
        metric: Geometry the collection was created with.
        dim: Vector width.

    Returns:
        An :class:`IndexHealth`. An untrained store reports ``trained=False``
        and zero drift rather than raising — "nothing to reconcile" is a state,
        not an error.
    """
    committed = {vector_id for vector_id, _ in storage.iter_payloads()}
    centroids = [
        blob for _, blob in storage.scan_prefix(CENTROID_PREFIX)
    ]
    if not centroids:
        return IndexHealth(trained=False, dim=dim, vectors=len(committed))

    # Every image is measured before it is believed. A blob of the wrong width
    # would otherwise surface as an exception further down the query path,
    # where "the store is damaged" is the least actionable possible diagnosis.
    row_bytes = dim * 4
    malformed = any(len(blob) != row_bytes for blob in centroids)

    seen: dict[VectorId, int] = {}
    postings = 0
    for cluster in range(len(centroids)):
        prefix = posting_prefix(cluster)
        for key, blob in storage.scan_prefix(prefix):
            postings += 1
            if len(blob) != row_bytes:
                malformed = True
            vector_id = decode_id(key[len(prefix) :].decode("utf-8"))
            seen[vector_id] = seen.get(vector_id, 0) + 1

    held = set(seen)
    metadata: dict[str, Any] | None = None
    blob = storage.get_raw(IVF_META_KEY)
    if blob is not None:
        try:
            metadata = dict(json.loads(blob.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError):
            malformed = True

    return IndexHealth(
        trained=True,
        dim=dim,
        partitions=len(centroids),
        vectors=len(committed),
        postings=postings,
        missing=tuple(sorted(committed - held)),
        stale=tuple(sorted(held - committed)),
        duplicated=tuple(sorted(name for name, count in seen.items() if count > 1)),
        metadata=metadata,
        malformed=malformed,
    )


def rebuild_ivf(
    storage: VectorStorage,
    *,
    metric: DistanceMetric,
    dim: int,
    partitions: int = 8,
    seed: int = 0,
) -> IvfIndex:
    """Discard every centroid and posting, then retrain from the corpus.

    Args:
        storage: Namespace adapter holding ``v:``, ``c:`` and ``b:``.
        metric: Geometry to bind.
        dim: Vector width.
        partitions: Centroid count.
        seed: Clustering seed.

    Returns:
        The freshly trained index.

    Raises:
        ValueError: The corpus is empty; an IVF index has nothing to cluster.
    """
    clear_namespace(storage, POSTING_PREFIX)
    clear_namespace(storage, CENTROID_PREFIX)
    storage.delete_raw(IVF_META_KEY, sync=False)
    return IvfIndex.train(
        storage, metric=metric, dim=dim, partitions=partitions, seed=seed
    )


def recover(
    engine: Any,
    *,
    partitions: int = 8,
    seed: int = 0,
    rebuild_ratio: float = DEFAULT_REBUILD_RATIO,
    train: bool = True,
) -> RecoveryReport:
    """Bring a reopened collection back to a consistent state.

    Args:
        engine: A :class:`~sable_vector.engine.VectorEngine` handle.
        partitions: Centroid count to use when a retrain is needed.
        seed: Clustering seed for that retrain.
        rebuild_ratio: Drift fraction above which the partitions are recomputed
            rather than spliced.
        train: When ``False``, an untrained or badly drifted index is left alone
            and only the torn records are reclaimed.

    Returns:
        A :class:`RecoveryReport`. A healthy store yields an all-zero report,
        which is what makes this safe to call on every open.
    """
    repair = engine.reclaim()
    report = RecoveryReport(
        torn_vectors=repair.orphan_vectors,
        torn_payloads=repair.orphan_payloads,
        stale_index_keys=repair.stale_index_keys,
        rebuilt_index_keys=repair.rebuilt_index_keys,
    )

    dim = engine.dim
    if dim is None:
        return report

    storage = engine.storage
    health = inspect_ivf(storage, metric=engine.metric, dim=dim)
    if not train:
        return report

    if health.needs_retrain(rebuild_ratio):
        # An empty corpus cannot be clustered; leaving the index absent is the
        # consistent state, and the next write that fills it will train one.
        if health.vectors == 0:
            clear_namespace(storage, POSTING_PREFIX)
            clear_namespace(storage, CENTROID_PREFIX)
            storage.delete_raw(IVF_META_KEY, sync=False)
            return _bump(report, retrained=True)
        rebuild_ivf(
            storage,
            metric=engine.metric,
            dim=dim,
            partitions=min(partitions, health.vectors),
            seed=seed,
        )
        return _bump(
            report,
            missing_postings=len(health.missing),
            stale_postings=len(health.stale),
            retrained=True,
        )

    if not health.trained:
        return report

    index = IvfIndex.load(storage, metric=engine.metric, dim=dim)
    dropped = 0
    for vector_id in health.stale:
        if index.remove(vector_id):
            dropped += 1
    added = 0
    for vector_id in health.missing:
        blob = storage.get_vector_blob(vector_id)
        if blob is None:
            continue
        index.add(vector_id, decode_vector(blob, dim=dim))
        added += 1
    return _bump(report, missing_postings=added, stale_postings=dropped)


def _bump(report: RecoveryReport, **delta: Any) -> RecoveryReport:
    """Return ``report`` with ``delta`` applied, keeping the value frozen."""
    return RecoveryReport(
        torn_vectors=report.torn_vectors + int(delta.get("torn_vectors", 0)),
        torn_payloads=report.torn_payloads + int(delta.get("torn_payloads", 0)),
        stale_index_keys=report.stale_index_keys + int(delta.get("stale_index_keys", 0)),
        rebuilt_index_keys=report.rebuilt_index_keys
        + int(delta.get("rebuilt_index_keys", 0)),
        missing_postings=report.missing_postings + int(delta.get("missing_postings", 0)),
        stale_postings=report.stale_postings + int(delta.get("stale_postings", 0)),
        retrained=report.retrained or bool(delta.get("retrained", False)),
    )
