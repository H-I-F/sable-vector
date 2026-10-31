"""Exhaustive search: score every vector, so recall is exactly 100%.

Flat is the reference every approximate index is measured against, and for the
sizes this engine targets — a local agent's corpus, tens of thousands of
1536-dimension embeddings — it is also the right answer. A 50k x 1536 matrix is
300 MB of float32 and one gemv per query is a millisecond; an index that reads
less than that buys back microseconds at the cost of recall, which is a bad
trade until the corpus stops fitting in RAM.

Two execution modes share one scoring path, because the difference between them
is a memory budget and nothing else:

* **Resident** — :meth:`search` scores a materialised ``(n, d)`` matrix. One
  pass, no disk I/O, memory proportional to the corpus.
* **Streaming** — :meth:`search_streaming` walks the vector namespace a block at
  a time and keeps only one block plus the running scores resident. Same result
  to the last bit, memory proportional to *n* rather than to *n x d*.

Concurrency contract
--------------------
The matrix is never mutated in place. :meth:`add` and :meth:`remove` build a new
array and rebind it, so a reader that captured the previous array keeps a
consistent view for as long as it holds the reference — which is what lets a
scan run against a snapshot without taking a lock.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

import numpy as np

from sable_vector.codec import decode_matrix
from sable_vector.distance import VECTOR_DTYPE, as_query, score_matrix, top_k
from sable_vector.storage import VECTOR_PREFIX, VectorStorage, decode_id
from sable_vector.types import DistanceMetric, Match, VectorId

__all__ = ["DEFAULT_BLOCK_ROWS", "FlatIndex"]

DEFAULT_BLOCK_ROWS = 512


class FlatIndex:
    """A materialised matrix plus the ids that address its rows.

    Args:
        metric: Geometry used by every search on this index.
        dim: Row width; every vector must agree with it.
        ids: Row labels, parallel to ``matrix``.
        matrix: ``(n, d)`` float32 block, or an empty ``(0, d)`` block.
    """

    def __init__(
        self,
        *,
        metric: DistanceMetric,
        dim: int,
        ids: Sequence[VectorId] | None = None,
        matrix: np.ndarray | None = None,
    ) -> None:
        self._metric = metric
        self._dim = dim
        self._ids: list[VectorId] = list(ids or ())
        self._matrix: np.ndarray = (
            matrix
            if matrix is not None
            else np.empty((0, dim), dtype=VECTOR_DTYPE)
        )
        if self._matrix.shape[0] != len(self._ids):
            raise ValueError("ids and matrix must have the same number of rows")
        self._positions: dict[VectorId, int] = {
            vector_id: row for row, vector_id in enumerate(self._ids)
        }

    # -- introspection -----------------------------------------------------

    @property
    def metric(self) -> DistanceMetric:
        return self._metric

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def ids(self) -> list[VectorId]:
        return list(self._ids)

    @property
    def matrix(self) -> np.ndarray:
        """The resident block. Replaced, never mutated in place."""
        return self._matrix

    @property
    def size(self) -> int:
        return len(self._ids)

    def __len__(self) -> int:
        return len(self._ids)

    # -- construction ------------------------------------------------------

    @classmethod
    def empty(cls, *, metric: DistanceMetric, dim: int) -> FlatIndex:
        """An index with no rows."""
        return cls(metric=metric, dim=dim)

    @classmethod
    def materialise(
        cls, storage: VectorStorage, *, metric: DistanceMetric, dim: int
    ) -> FlatIndex:
        """Read every vector image out of the store and stack it into one matrix.

        The images are concatenated once and reinterpreted as a block, so the
        matrix aliases a single buffer rather than one allocation per row — the
        difference between one 300 MB object and 50k of them.

        Args:
            storage: Namespace adapter to read ``v:`` from.
            metric: Geometry to bind to the index.
            dim: Row width; a blob that is not a whole number of rows raises.

        Returns:
            A resident index covering every stored vector.
        """
        ids: list[VectorId] = []
        images: list[bytes] = []
        for key, blob in storage.scan_prefix(VECTOR_PREFIX):
            ids.append(decode_id(key[len(VECTOR_PREFIX) :].decode("utf-8")))
            images.append(blob)
        if not images:
            return cls.empty(metric=metric, dim=dim)
        joined = b"".join(images)
        return cls(metric=metric, dim=dim, ids=ids, matrix=decode_matrix(joined, dim=dim))

    # -- mutation ----------------------------------------------------------

    def add(self, vector_id: VectorId, vector: np.ndarray) -> None:
        """Insert or replace one row, replacing the matrix rather than editing it.

        Args:
            vector_id: Row label.
            vector: Embedding; must match :attr:`dim`.

        Raises:
            ValueError: The width disagrees with :attr:`dim`.
        """
        probe = as_query(vector, dim=self._dim)
        row = self._positions.get(vector_id)
        if row is not None:
            # Copy first: the previous array may still be held by a reader that
            # captured it before this write.
            matrix = self._matrix.copy()
            matrix[row] = probe
            self._matrix = matrix
            return
        self._ids.append(vector_id)
        self._positions[vector_id] = len(self._ids) - 1
        self._matrix = np.vstack((self._matrix, probe.reshape(1, self._dim)))

    def remove(self, vector_id: VectorId) -> bool:
        """Drop one row, renumbering the rest.

        Returns:
            ``True`` when a row was removed.
        """
        row = self._positions.pop(vector_id, None)
        if row is None:
            return False
        del self._ids[row]
        self._positions = {label: index for index, label in enumerate(self._ids)}
        self._matrix = np.delete(self._matrix, row, axis=0)
        return True

    # -- search ------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        candidate_ids: Iterable[VectorId] | None = None,
    ) -> list[Match]:
        """Score the resident matrix exhaustively.

        Args:
            query: ``(d,)`` embedding.
            k: How many neighbours to return.
            candidate_ids: When given, only these ids are scored — which is what
                turns a filtered query into less arithmetic rather than into a
                filter applied afterwards.

        Returns:
            Up to ``k`` matches, best first. Payloads are left empty; filling
            them is a storage read and belongs to the query layer.
        """
        probe = as_query(query, dim=self._dim)
        if self.size == 0 or k <= 0:
            return []

        if candidate_ids is None:
            grid = self._matrix
            labels = self._ids
        else:
            # Sorted, not set-ordered. A candidate set iterates in Python's
            # string-hash order, which is randomised per process, so gathering in
            # that order would make the gathered block — and therefore the
            # gemv's own rounding — differ between runs. Two candidates with
            # exactly equal scores would then swap places depending on the hash
            # seed, which is a ranking that cannot be reproduced from a bug
            # report. Sorting by row restores matrix order and pins the result.
            rows = sorted(
                self._positions[label]
                for label in candidate_ids
                if label in self._positions
            )
            if not rows:
                return []
            # A gather, not a mask: the candidate block is built once and then
            # handed to the same gemv the unfiltered path uses.
            grid = self._matrix[rows]
            labels = [self._ids[row] for row in rows]

        scores = score_matrix(grid, probe, self._metric)
        order, values = top_k(scores, min(k, scores.size))
        return [
            Match(labels[int(index)], float(value), -float(value), {})
            for index, value in zip(order, values, strict=False)
        ]

    def search_streaming(
        self,
        storage: VectorStorage,
        query: np.ndarray,
        k: int,
        *,
        block_rows: int = DEFAULT_BLOCK_ROWS,
        candidate_ids: Iterable[VectorId] | None = None,
    ) -> list[Match]:
        """Score straight off the SSTables, one block resident at a time.

        Identical results to :meth:`search`, different memory profile: instead of
        ``n x d`` floats only ``block_rows x d`` plus ``n`` scores are held,
        which is the mode to use when the corpus is larger than the budget.

        Args:
            storage: Namespace adapter to stream ``v:`` from.
            query: ``(d,)`` embedding.
            k: How many neighbours to return.
            block_rows: Rows decoded per block.
            candidate_ids: Optional allow-list, applied before decoding so a
                filtered query also skips the I/O.

        Returns:
            Up to ``k`` matches, best first.
        """
        probe = as_query(query, dim=self._dim)
        if k <= 0:
            return []
        allowed = None if candidate_ids is None else set(candidate_ids)

        labels: list[VectorId] = []
        chunks: list[np.ndarray] = []
        pending_ids: list[VectorId] = []
        pending_images: list[bytes] = []

        def drain() -> None:
            if not pending_images:
                return
            block = decode_matrix(b"".join(pending_images), dim=self._dim)
            chunks.append(score_matrix(block, probe, self._metric))
            pending_images.clear()

        for key, blob in storage.scan_prefix(VECTOR_PREFIX):
            vector_id = decode_id(key[len(VECTOR_PREFIX) :].decode("utf-8"))
            if allowed is not None and vector_id not in allowed:
                continue
            pending_ids.append(vector_id)
            pending_images.append(blob)
            if len(pending_images) >= block_rows:
                labels.extend(pending_ids)
                pending_ids.clear()
                drain()
        labels.extend(pending_ids)
        drain()

        if not chunks:
            return []
        scores = np.concatenate(chunks)
        order, values = top_k(scores, min(k, scores.size))
        return [
            Match(labels[int(index)], float(value), -float(value), {})
            for index, value in zip(order, values, strict=False)
        ]

    def iter_rows(self) -> Iterator[tuple[VectorId, np.ndarray]]:
        """Walk ``(id, row)`` pairs in matrix order."""
        for row, vector_id in enumerate(self._ids):
            yield vector_id, self._matrix[row]
