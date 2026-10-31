"""The write path: one acknowledgement per record, one fsync per record.

Every layer below this one is a key-value store, and a key-value store cannot
commit two keys together. So the atomicity of a vector record — its float32
image, its payload and its inverted-index entries are three physically separate
keys — has to be *constructed* rather than inherited. It is built from two rules
that together make a partial write both detectable and repairable.

**Rule one: the payload is the commit record.** A record is committed exactly
when ``p:<id>`` exists. Every write therefore stages the cheap, large, derived
keys first and writes the payload last:

    v:<id>   float32 image     (sync=False)
    idx:...  inverted entries  (sync=False)
    p:<id>   payload           (sync=True)  <- the record becomes visible here

A crash between the first and last key leaves a vector with no payload, never a
payload with no vector. That direction is the whole point: an orphan payload
would be a record the engine claims to have while being unable to score it,
whereas an orphan vector is merely garbage that the reaper can prove is garbage.

**Rule two: one fsync boundary per record.** ``sync=False`` leaves the frame in
the write-ahead log's buffer; the payload's ``sync=True`` flushes and fsyncs the
log, which carries every frame buffered before it. The record therefore reaches
stable storage in a single `fsync(2)` regardless of how many keys it spans.

Deletion runs the same reasoning backwards — drop the payload first, the vector
last — so a torn delete leaves the same detectable orphan rather than a
half-erased record.

What this does *not* promise: a multi-record batch shares one fsync boundary,
so individual records inside it can become visible before the batch as a whole
is acknowledged. They are never partially visible, which is the property that
matters; a batch is a performance construct, not a transaction boundary.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sable.types import EngineOptions

from sable_vector.codec import encode_vector
from sable_vector.distance import as_query
from sable_vector.index.scalar import ScalarIndex
from sable_vector.storage import META_PREFIX, VectorStorage
from sable_vector.types import DistanceMetric, VectorId, VectorRecord

__all__ = ["RepairReport", "VectorEngine"]

_DIM_KEY = META_PREFIX + b"dim"


@dataclass(frozen=True, slots=True)
class RepairReport:
    """What a reclamation pass removed or restored.

    Attributes:
        orphan_vectors: Vectors whose payload never landed — torn writes.
        orphan_payloads: Payloads whose vector is gone — impossible under the
            documented write order, and reported rather than assumed away.
        stale_index_keys: Inverted entries pointing at ids no longer committed.
        rebuilt_index_keys: Entries a committed record was missing.
    """

    orphan_vectors: int = 0
    orphan_payloads: int = 0
    stale_index_keys: int = 0
    rebuilt_index_keys: int = 0

    @property
    def total(self) -> int:
        return (
            self.orphan_vectors
            + self.orphan_payloads
            + self.stale_index_keys
            + self.rebuilt_index_keys
        )


class VectorEngine:
    """Write half of the vector engine: durable upsert, delete and reclamation.

    Args:
        storage: Namespace adapter already opened on a database directory.
        metric: Geometry the collection was created with.
        dim: Fixed width of every vector, or ``None`` to adopt the width of the
            first write and persist it.
        indexed_fields: Payload attributes to maintain inverted entries for.
    """

    def __init__(
        self,
        storage: VectorStorage,
        *,
        metric: DistanceMetric = DistanceMetric.COSINE,
        dim: int | None = None,
        indexed_fields: Sequence[str] = (),
    ) -> None:
        self._storage = storage
        self._metric = metric
        self._indexed_fields = tuple(indexed_fields)
        self._scalar = ScalarIndex(storage, self._indexed_fields)
        self._dim = self._restore_dim() if dim is None else dim

    @classmethod
    def open(
        cls,
        home: Path,
        *,
        metric: DistanceMetric = DistanceMetric.COSINE,
        dim: int | None = None,
        indexed_fields: Sequence[str] = (),
        options: EngineOptions | None = None,
    ) -> VectorEngine:
        """Open or recover a collection, reclaiming anything a crash left behind.

        Reclamation runs here rather than on demand so that a caller never
        observes a torn record: by the time the handle exists, the tree holds
        only records whose payload and vector agree.
        """
        engine = cls(
            VectorStorage.open(Path(home), options),
            metric=metric,
            dim=dim,
            indexed_fields=indexed_fields,
        )
        engine.reclaim()
        return engine

    # -- introspection -----------------------------------------------------

    @property
    def home(self) -> Path:
        return self._storage.home

    @property
    def storage(self) -> VectorStorage:
        return self._storage

    @property
    def metric(self) -> DistanceMetric:
        return self._metric

    @property
    def dim(self) -> int | None:
        """Fixed vector width, ``None`` until the first write establishes it."""
        return self._dim

    @property
    def indexed_fields(self) -> tuple[str, ...]:
        return self._indexed_fields

    @property
    def scalar(self) -> ScalarIndex:
        """The inverted index over :attr:`indexed_fields`."""
        return self._scalar

    @property
    def stats(self) -> Any:
        return self._storage.stats

    def close(self) -> None:
        self._storage.close()

    def __enter__(self) -> VectorEngine:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- writes ------------------------------------------------------------

    def upsert(
        self,
        vector_id: VectorId,
        vector: np.ndarray,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        """Write one record atomically.

        Args:
            vector_id: Identifier to create or replace.
            vector: Embedding; must match the collection width once set.
            payload: Attributes to store and, for indexed fields, invert.

        Raises:
            ValueError: The vector width disagrees with the collection.
        """
        self._write_batch([(vector_id, vector, dict(payload or {}))])

    def upsert_many(self, records: Iterable[VectorRecord]) -> int:
        """Write many records across one fsync boundary.

        Args:
            records: Records to write, in order.

        Returns:
            How many were written.
        """
        batch = [(record.id, record.vector, record.payload) for record in records]
        self._write_batch(batch)
        return len(batch)

    def delete(self, vector_id: VectorId) -> bool:
        """Remove one record, payload first so a torn delete stays detectable.

        Returns:
            ``True`` when something was removed.
        """
        payload = self._storage.get_payload(vector_id)
        if payload is None and self._storage.get_vector_blob(vector_id) is None:
            return False
        if payload is not None:
            self._storage.delete_payload(vector_id, sync=False)
            self._drop_index_keys(vector_id, payload, sync=False)
        else:
            self._drop_unknown_index_keys(vector_id, sync=False)
        self._storage.delete_vector(vector_id, sync=True)
        return True

    # -- reclamation -------------------------------------------------------

    def reclaim(self) -> RepairReport:
        """Bring the tree back to "a record is its payload".

        A record is committed when its payload exists, so anything else in the
        tree is derived and can be rebuilt or discarded: vectors without a
        payload are torn writes, and inverted entries without a committed id are
        references to them.

        Returns:
            A :class:`RepairReport` counting what the pass changed.
        """
        vector_ids = set(self._storage.iter_vector_ids())
        payloads = dict(self._storage.iter_payloads())
        payload_ids = set(payloads)

        report = RepairReport(
            orphan_vectors=len(vector_ids - payload_ids),
            orphan_payloads=len(payload_ids - vector_ids),
        )

        for orphan in sorted(vector_ids - payload_ids):
            self._storage.delete_vector(orphan)
            report = _bump(report, stale_index_keys=self._drop_unknown_index_keys(orphan))
        for orphan in sorted(payload_ids - vector_ids):
            self._storage.delete_payload(orphan)
            report = _bump(report, stale_index_keys=self._drop_unknown_index_keys(orphan))

        # The inverted entries are a pure function of the committed payloads, so
        # once the torn records are gone the index can be diffed against them in
        # both directions and made to agree.
        audit = self._scalar.repair()
        return _bump(
            report,
            stale_index_keys=len(audit.orphaned),
            rebuilt_index_keys=len(audit.missing),
        )

    # -- internals ---------------------------------------------------------

    def _write_batch(
        self, items: Sequence[tuple[VectorId, np.ndarray, dict[str, Any]]]
    ) -> None:
        """Stage every key of every record, then commit the payloads.

        Only the final payload carries ``sync=True``; every frame before it is
        buffered, so the whole batch reaches stable storage on that one fsync.
        """
        if not items:
            return

        prepared: list[tuple[VectorId, np.ndarray, dict[str, Any]]] = []
        for vector_id, vector, payload in items:
            prepared.append((vector_id, self._admit(vector), payload))

        # 1. large, cheap, derived: the float32 images.
        for vector_id, vector, _ in prepared:
            self._storage.put_vector(vector_id, encode_vector(vector), sync=False)

        # 2. inverted entries, computed against the payload each one replaces.
        for vector_id, _, payload in prepared:
            previous = self._storage.get_payload(vector_id)
            self._apply_index_delta(vector_id, previous or {}, payload, sync=False)

        # 3. the commit records. Only the last one forces the log to stable
        #    storage, which is what makes the batch a single durable unit.
        last = len(prepared) - 1
        for index, (vector_id, _, payload) in enumerate(prepared):
            self._storage.put_payload(vector_id, payload, sync=index == last)

    def _admit(self, vector: np.ndarray) -> np.ndarray:
        """Validate one vector's width and fix the collection's width if unset.

        Raises:
            ValueError: The width disagrees with the established one.
        """
        query = as_query(vector)
        width = int(query.shape[0])
        if self._dim is None:
            self._dim = width
            # Part of the batch, not a separate durable step: if the batch never
            # commits, the width was never established either.
            self._storage.put_raw(_DIM_KEY, str(width).encode("utf-8"), sync=False)
            return query
        if self._dim != width:
            raise ValueError(
                f"vector of width {width} does not match the collection width {self._dim}"
            )
        return query

    def _apply_index_delta(
        self,
        vector_id: VectorId,
        previous: Mapping[str, Any],
        payload: Mapping[str, Any],
        *,
        sync: bool,
    ) -> None:
        """Move the inverted entries from one payload to the next."""
        self._scalar.reindex(vector_id, previous, payload, sync=sync)

    def _drop_index_keys(
        self, vector_id: VectorId, payload: Mapping[str, Any], *, sync: bool
    ) -> int:
        """Drop every inverted entry a payload owns. Returns how many."""
        return self._scalar.deindex(vector_id, payload, sync=sync)

    def _drop_unknown_index_keys(self, vector_id: VectorId, *, sync: bool = True) -> int:
        """Drop the entries of a record whose payload is already gone.

        The value segment cannot be derived without the payload, which is exactly
        the torn-write case, so the field is walked instead.
        """
        return self._scalar.deindex(vector_id, sync=sync)

    def _restore_dim(self) -> int | None:
        stored = self._storage.get_raw(_DIM_KEY)
        return None if stored is None else int(stored.decode("utf-8"))


def _bump(report: RepairReport, **delta: int) -> RepairReport:
    """Return ``report`` with ``delta`` added, keeping the value frozen."""
    return RepairReport(
        orphan_vectors=report.orphan_vectors + delta.get("orphan_vectors", 0),
        orphan_payloads=report.orphan_payloads + delta.get("orphan_payloads", 0),
        stale_index_keys=report.stale_index_keys + delta.get("stale_index_keys", 0),
        rebuilt_index_keys=report.rebuilt_index_keys + delta.get("rebuilt_index_keys", 0),
    )
