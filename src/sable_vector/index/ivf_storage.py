"""IVF posting lists persisted in the LSM tree, with partition pruning.

The inverted file index is the point where this engine stops reading the whole
corpus. Vectors are assigned to the nearest of *k* centroids, and a query reads
only the partitions whose centroids are closest to it — so the cost of a search
is governed by ``nprobe`` rather than by the size of the collection.

Two decisions make that worth the extra bookkeeping.

**Postings live in the same tree as everything else.** Centroids sit under
``c:<cluster>`` and members under ``b:<cluster>:<id>``, sharing one WAL and one
manifest with the vectors and payloads. A partition is therefore a *prefix*:
reading it is one ordered scan, which is exactly the operation the storage
engine is built for. A partition that is not probed is never opened, which is
the property the acceptance test measures.

**What is stored is the residual, not the vector.** ``x - c`` costs the same
bytes as ``x`` and has a much smaller dynamic range, which is what makes the
quantisation a later revision can add cheap. The vector is reconstructed as
``c + residual``; the round trip is exact to within float32 representation, and
the consequence is bounded — a ranking can flip only between candidates whose
scores already agree to about seven significant digits.

The trade is explicit and measurable: ``nprobe`` below the partition count buys
latency with recall. :class:`IvfOutcome` reports how many partitions were read
and how many postings were decoded, because an index that silently reads every
partition is not an IVF index.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from sable_vector.codec import decode_matrix, decode_vector, encode_vector
from sable_vector.distance import VECTOR_DTYPE, as_query, score_matrix, top_k
from sable_vector.index.flat import FlatIndex
from sable_vector.index.ivf_cluster import assign, kmeans, residuals, squared_distances
from sable_vector.storage import (
    CENTROID_PREFIX,
    META_PREFIX,
    POSTING_PREFIX,
    VectorStorage,
    decode_id,
    escape_segment,
)
from sable_vector.types import DistanceMetric, Match, VectorId

__all__ = ["IvfIndex", "IvfOutcome", "IVF_META_KEY", "centroid_key", "posting_prefix"]

IVF_META_KEY = META_PREFIX + b"ivf"
_CLUSTER_WIDTH = 6


@dataclass(frozen=True, slots=True)
class IvfOutcome:
    """Result of one partition-pruned search.

    Attributes:
        matches: Up to ``k`` nearest records found in the probed partitions.
        buckets_scanned: Partitions actually read. Normally ``nprobe``, which is
            the number that says whether pruning happened.
        postings_read: Posting entries read out of the scanned partitions. This
            is the I/O cost of the query, and the figure that distinguishes a
            pruned search from a full one.
        partitions: Total partitions in the index, for the caller to compare
            against ``buckets_scanned`` and see the fraction of the corpus read.
    """

    matches: tuple[Match, ...]
    buckets_scanned: int
    postings_read: int
    partitions: int


def centroid_key(cluster: int) -> bytes:
    """``c:`` key for one centroid.

    The cluster number is zero-padded to a fixed width so that lexical key
    order matches numeric cluster order, which is what lets a range read over
    ``c:`` come back in cluster order without a sort.
    """
    return CENTROID_PREFIX + f"{cluster:0{_CLUSTER_WIDTH}d}".encode("utf-8")


def posting_prefix(cluster: int) -> bytes:
    """``b:`` prefix covering every member of one partition."""
    return POSTING_PREFIX + f"{cluster:0{_CLUSTER_WIDTH}d}:".encode("utf-8")


class IvfIndex:
    """Centroids plus disk-backed posting lists, pruned by ``nprobe``.

    Args:
        storage: Namespace adapter holding the ``c:`` and ``b:`` prefixes.
        metric: Geometry used for scoring.
        dim: Vector width.
        centroids: ``(k, d)`` centroid block.
    """

    def __init__(
        self,
        storage: VectorStorage,
        *,
        metric: DistanceMetric,
        dim: int,
        centroids: np.ndarray,
    ) -> None:
        self._storage = storage
        self._metric = metric
        self._dim = dim
        self._centroids = np.ascontiguousarray(centroids, dtype=VECTOR_DTYPE)

    # -- introspection -----------------------------------------------------

    @property
    def metric(self) -> DistanceMetric:
        return self._metric

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def centroids(self) -> np.ndarray:
        return self._centroids

    @property
    def partitions(self) -> int:
        return int(self._centroids.shape[0])

    def cluster_of(self, vector: np.ndarray) -> int:
        """Nearest partition for one vector."""
        probe = as_query(vector, dim=self._dim)
        labels, _ = assign(self._centroids, probe.reshape(1, self._dim))
        return int(labels[0])

    # -- construction ------------------------------------------------------

    @classmethod
    def train(
        cls,
        storage: VectorStorage,
        *,
        metric: DistanceMetric,
        dim: int,
        partitions: int = 8,
        seed: int = 0,
        max_iterations: int = 16,
    ) -> IvfIndex:
        """Cluster the stored corpus and write its posting lists.

        Args:
            storage: Source of the vectors, and the destination of the postings.
            metric: Geometry to bind.
            dim: Vector width.
            partitions: Number of centroids.
            seed: Clustering seed.
            max_iterations: Lloyd budget.

        Returns:
            A trained index whose postings are already durable.

        Raises:
            ValueError: The corpus is empty, or smaller than ``partitions``.
        """
        flat = FlatIndex.materialise(storage, metric=metric, dim=dim)
        if flat.size == 0:
            raise ValueError("cannot train an IVF index over an empty corpus")
        if flat.size < partitions:
            raise ValueError(
                f"{flat.size} vectors cannot fill {partitions} partitions"
            )

        clustering = kmeans(
            flat.matrix, partitions, seed=seed, max_iterations=max_iterations
        )
        index = cls(storage, metric=metric, dim=dim, centroids=clustering.centroids)

        # Centroids and postings are derived data: they are written unsynced and
        # the metadata key closes the batch, so one fsync publishes the whole
        # rebuild. A crash mid-rebuild leaves postings that no metadata
        # references, which the next train overwrites — never a half-built index
        # that claims to be complete.
        spread = residuals(flat.matrix, clustering.centroids, clustering.assignments)
        for cluster in range(index.partitions):
            index._write_centroid(cluster, sync=False)
        for row, vector_id in enumerate(flat.ids):
            index._write_posting(
                int(clustering.assignments[row]), vector_id, spread[row], sync=False
            )
        storage.put_raw(
            IVF_META_KEY,
            json.dumps(
                {
                    "k": index.partitions,
                    "dim": dim,
                    "seed": seed,
                    "metric": metric.value,
                    "vectors": flat.size,
                }
            ).encode("utf-8"),
        )
        return index

    @classmethod
    def load(
        cls, storage: VectorStorage, *, metric: DistanceMetric, dim: int
    ) -> IvfIndex:
        """Reopen a trained index from its persisted centroids.

        Raises:
            ValueError: No centroids are stored — the index was never trained.
        """
        rows: list[np.ndarray] = []
        for _, blob in storage.scan_prefix(CENTROID_PREFIX):
            rows.append(decode_vector(blob, dim=dim))
        if not rows:
            raise ValueError("no centroids are stored; train the index first")
        return cls(storage, metric=metric, dim=dim, centroids=np.stack(rows))

    def metadata(self) -> dict[str, object] | None:
        """Training metadata as persisted, or ``None`` when absent."""
        blob = self._storage.get_raw(IVF_META_KEY)
        return None if blob is None else json.loads(blob.decode("utf-8"))

    # -- mutation ----------------------------------------------------------

    def add(self, vector_id: VectorId, vector: np.ndarray) -> int:
        """Assign one vector and write its posting.

        Returns:
            The partition it landed in.
        """
        probe = as_query(vector, dim=self._dim)
        cluster = self.cluster_of(probe)
        self._write_posting(cluster, vector_id, probe - self._centroids[cluster])
        return cluster

    def remove(self, vector_id: VectorId) -> bool:
        """Drop a vector's posting from whichever partition holds it.

        Returns:
            ``True`` when a posting was removed.
        """
        escaped = escape_segment(vector_id.encode("utf-8"))
        for cluster in range(self.partitions):
            key = posting_prefix(cluster) + escaped
            if self._storage.get_raw(key) is not None:
                self._storage.delete_raw(key)
                return True
        return False

    # -- search ------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        k: int,
        *,
        nprobe: int = 1,
        candidate_ids: list[VectorId] | None = None,
    ) -> IvfOutcome:
        """Nearest ``k`` records across the ``nprobe`` closest partitions.

        Args:
            query: ``(d,)`` embedding.
            k: How many neighbours to return.
            nprobe: Partitions to read. ``1`` is the fastest and least accurate;
                ``partitions`` degenerates to a full scan with exact recall.
            candidate_ids: Optional allow-list, applied while decoding so a
                filtered query also skips the arithmetic.

        Returns:
            An :class:`IvfOutcome`.

        Raises:
            ValueError: ``nprobe`` is not positive.
        """
        if nprobe <= 0:
            raise ValueError(f"nprobe must be positive, got {nprobe}")
        if k <= 0:
            return IvfOutcome((), 0, 0, self.partitions)

        probe = as_query(query, dim=self._dim)
        # One gemm ranks the partitions; everything after this reads only the
        # winners, which is the entire point of the index.
        distances = squared_distances(probe.reshape(1, self._dim), self._centroids)[0]
        order = np.argsort(distances, kind="stable")[: min(nprobe, self.partitions)]

        allowed = None if candidate_ids is None else set(candidate_ids)
        labels: list[VectorId] = []
        chunks: list[np.ndarray] = []
        postings = 0

        for cluster in order:
            images: list[bytes] = []
            members: list[VectorId] = []
            prefix = posting_prefix(int(cluster))
            for key, blob in self._storage.scan_prefix(prefix):
                postings += 1
                vector_id = decode_id(key[len(prefix) :].decode("utf-8"))
                if allowed is not None and vector_id not in allowed:
                    continue
                members.append(vector_id)
                images.append(blob)
            if not images:
                continue
            # Reconstruct the partition as one block: every member shares this
            # cluster's centroid, so the add is a broadcast rather than a loop.
            block = decode_matrix(b"".join(images), dim=self._dim)
            block = np.asarray(
                block + self._centroids[int(cluster)], dtype=VECTOR_DTYPE
            )
            chunks.append(score_matrix(block, probe, self._metric))
            labels.extend(members)

        if not chunks:
            return IvfOutcome((), len(order), postings, self.partitions)

        scores = np.concatenate(chunks)
        ranking, values = top_k(scores, min(k, scores.size))
        return IvfOutcome(
            matches=tuple(
                Match(labels[int(index)], float(value), -float(value), {})
                for index, value in zip(ranking, values)
            ),
            buckets_scanned=len(order),
            postings_read=postings,
            partitions=self.partitions,
        )

    def postings_in(self, cluster: int) -> list[VectorId]:
        """Ids held by one partition, in key order."""
        prefix = posting_prefix(cluster)
        return [
            decode_id(key[len(prefix) :].decode("utf-8"))
            for key, _ in self._storage.scan_prefix(prefix)
        ]

    # -- internals ---------------------------------------------------------

    def _write_centroid(self, cluster: int, *, sync: bool = True) -> None:
        self._storage.put_raw(
            centroid_key(cluster),
            encode_vector(self._centroids[cluster]),
            sync=sync,
        )

    def _write_posting(
        self,
        cluster: int,
        vector_id: VectorId,
        residual: np.ndarray,
        *,
        sync: bool = True,
    ) -> None:
        self._storage.put_raw(
            posting_prefix(cluster) + escape_segment(vector_id.encode("utf-8")),
            encode_vector(residual),
            sync=sync,
        )
