"""Centroid clustering for the inverted file index.

An IVF index partitions the corpus by proximity to *k* centroids, so a query
only has to read the partitions nearest to it. Everything that makes that work
is decided here, before a single posting list is written:

* **Initialisation is seeded and deterministic.** Plain random seeding leaves a
  cluster empty or two centroids in the same basin often enough to matter, and
  the failure is invisible — recall quietly drops. k-means++ spreads the initial
  centroids by squared distance, and the generator is seeded, so a corpus always
  clusters the same way and a recall regression is reproducible from the same
  data instead of hiding behind nondeterminism.
* **Assignment is one gemm.** Distances to every centroid come from the same
  Gram expansion the distance kernels use: ``||x||² - 2 x·c + ||c||²``. That is
  an ``(n, d) x (d, k)`` product, which is the whole cost of a Lloyd iteration
  beyond the per-cluster means.
* **An empty cluster is reseeded, never left NaN.** A centroid that captured
  nothing would poison the next assignment with NaN distances. It is moved onto
  the point currently worst served, which is the choice that reduces inertia
  most on the following step and keeps the iteration monotone.
* **Inertia is recorded, not just returned.** Lloyd's alternation can only
  decrease it, so a recorded history is the difference between "it stopped" and
  "it converged": a run whose history ever increases is a bug in the update,
  not a slow corpus.

Residuals are the other output worth having: storing ``x - c`` instead of ``x``
costs the same bytes and, because the residual has a much smaller dynamic range,
makes the quantisation the IVF path will eventually add far less lossy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sable_vector.distance import VECTOR_DTYPE, as_matrix

__all__ = [
    "Clustering",
    "assign",
    "initialise_centroids",
    "kmeans",
    "residuals",
    "squared_distances",
]

DEFAULT_MAX_ITERATIONS = 32
DEFAULT_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class Clustering:
    """State of one converged (or exhausted) k-means run.

    Attributes:
        centroids: ``(k, d)`` centroid block.
        assignments: ``(n,)`` cluster index per row.
        counts: ``(k,)`` members per cluster.
        inertia: Sum of squared distances from each row to its centroid.
        iterations: Lloyd iterations actually performed.
        converged: Whether the centroid shift fell below the tolerance rather
            than the iteration budget running out.
        inertia_history: Inertia after each iteration, starting with the first
            assignment. Non-increasing by construction; an increase anywhere is
            a defect in the update step.
    """

    centroids: np.ndarray
    assignments: np.ndarray
    counts: np.ndarray
    inertia: float
    iterations: int
    converged: bool
    inertia_history: tuple[float, ...]


def squared_distances(matrix: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance from every row to every centroid.

    Args:
        matrix: ``(n, d)`` rows.
        centroids: ``(k, d)`` centroids.

    Returns:
        ``(n, k)`` float64 distances, clipped at zero.

    Raises:
        ValueError: The two blocks have different widths.
    """
    grid = as_matrix(matrix)
    cores = as_matrix(centroids)
    if grid.shape[1] != cores.shape[1]:
        raise ValueError(
            f"matrix width {grid.shape[1]} does not match centroid width {cores.shape[1]}"
        )
    row_norms = np.einsum("ij,ij->i", grid, grid, dtype=np.float64)[:, None]
    core_norms = np.einsum("ij,ij->i", cores, cores, dtype=np.float64)[None, :]
    cross = np.asarray(grid @ cores.T, dtype=np.float64)
    distances = row_norms - 2.0 * cross + core_norms
    # Non-negative in exact arithmetic; the expansion can undershoot by the gemm's
    # rounding noise, and a negative would sort as "closer than anything".
    np.maximum(distances, 0.0, out=distances)
    return distances


def assign(centroids: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign every row to its nearest centroid.

    Args:
        centroids: ``(k, d)`` centroids.
        matrix: ``(n, d)`` rows.

    Returns:
        ``(labels, distances)`` — the winning cluster index and its squared
        distance for each row.
    """
    grid = squared_distances(matrix, centroids)
    labels = np.argmin(grid, axis=1)
    return labels, grid[np.arange(grid.shape[0]), labels]


def initialise_centroids(
    matrix: np.ndarray, k: int, *, seed: int = 0
) -> np.ndarray:
    """Seeded k-means++ initialisation.

    Args:
        matrix: ``(n, d)`` rows.
        k: Number of centroids; must not exceed ``n``.
        seed: Generator seed; the same corpus and seed always give the same
            starting point.

    Returns:
        ``(k, d)`` centroids drawn from actual rows.

    Raises:
        ValueError: The matrix is empty, or ``k`` exceeds the row count.
    """
    grid = as_matrix(matrix)
    rows = grid.shape[0]
    if rows == 0:
        raise ValueError("cannot cluster an empty matrix")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > rows:
        raise ValueError(f"k={k} exceeds the {rows} rows available")

    generator = np.random.default_rng(seed)
    centroids = np.empty((k, grid.shape[1]), dtype=VECTOR_DTYPE)
    centroids[0] = grid[generator.integers(rows)]
    closest = squared_distances(grid, centroids[:1])[:, 0]

    for index in range(1, k):
        total = float(closest.sum())
        if total <= 0.0:
            # Every row coincides with a centroid already chosen; the remaining
            # choices cannot be weighted, so they are arbitrary but still seeded.
            pick = int(generator.integers(rows))
        else:
            weights = closest / total
            pick = int(generator.choice(rows, p=weights))
        centroids[index] = grid[pick]
        np.minimum(closest, squared_distances(grid, centroids[index : index + 1])[:, 0], out=closest)
    return centroids


def kmeans(
    matrix: np.ndarray,
    k: int,
    *,
    seed: int = 0,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Clustering:
    """Lloyd's algorithm over a float32 block.

    Args:
        matrix: ``(n, d)`` rows.
        k: Number of clusters.
        seed: Initialisation seed.
        max_iterations: Iteration budget.
        tolerance: Stop once no centroid moves further than this, measured as
            squared Euclidean distance.

    Returns:
        A :class:`Clustering`.

    Raises:
        ValueError: The matrix is empty, ``k`` is out of range, or the budget is
            not positive.
    """
    grid = as_matrix(matrix)
    if grid.shape[0] == 0:
        raise ValueError("cannot cluster an empty matrix")
    if max_iterations <= 0:
        raise ValueError(f"max_iterations must be positive, got {max_iterations}")

    centroids = initialise_centroids(grid, k, seed=seed)
    history: list[float] = []
    labels = np.zeros(grid.shape[0], dtype=np.int64)
    converged = False

    for _iteration in range(1, max_iterations + 1):
        labels, distances = assign(centroids, grid)
        history.append(float(distances.sum()))

        updated = _recompute_centroids(grid, labels, k, centroids, distances)
        shift = float(np.einsum("ij,ij->i", updated - centroids, updated - centroids, dtype=np.float64).max())
        centroids = updated
        if shift <= tolerance:
            converged = True
            break

    # The loop's last action was a centroid update, so the assignments it left
    # behind describe the previous centroids. One final assignment makes the
    # returned labels, inertia and centroids all describe the same state.
    labels, distances = assign(centroids, grid)
    history.append(float(distances.sum()))

    counts = np.bincount(labels, minlength=k)
    return Clustering(
        centroids=centroids,
        assignments=labels,
        counts=counts,
        inertia=float(distances.sum()),
        iterations=len(history) - 1,
        converged=converged,
        inertia_history=tuple(history),
    )


def residuals(matrix: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """``x - c`` for every row: same bytes, far smaller dynamic range.

    Args:
        matrix: ``(n, d)`` rows.
        centroids: ``(k, d)`` centroids.
        labels: ``(n,)`` cluster index per row.

    Returns:
        ``(n, d)`` residual block.

    Raises:
        ValueError: A label is outside the centroid block.
    """
    grid = as_matrix(matrix)
    cores = as_matrix(centroids)
    if labels.size != grid.shape[0]:
        raise ValueError("labels must have one entry per row")
    if labels.size and (int(labels.max()) >= cores.shape[0] or int(labels.min()) < 0):
        raise ValueError("a label addresses a centroid that does not exist")
    return np.asarray(grid - cores[labels], dtype=VECTOR_DTYPE)


def _recompute_centroids(
    grid: np.ndarray,
    labels: np.ndarray,
    k: int,
    previous: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    """Recompute cluster means, reseeding any cluster that captured nothing.

    Args:
        grid: ``(n, d)`` rows.
        labels: Current assignment.
        k: Number of clusters.
        previous: Centroids being replaced.
        distances: Squared distance from each row to its assigned centroid,
            used to pick the worst-served row when a cluster is empty.

    Returns:
        ``(k, d)`` centroids with no NaN rows.
    """
    updated = np.empty_like(previous)
    occupied = np.zeros(k, dtype=bool)
    for cluster in range(k):
        members = grid[labels == cluster]
        if members.shape[0]:
            updated[cluster] = members.mean(axis=0)
            occupied[cluster] = True
    if occupied.all():
        return updated

    # An empty centroid would produce NaN distances and collapse the next
    # assignment. Moving it onto the worst-served row is the choice that lowers
    # inertia most on the following step.
    available = np.flatnonzero(~occupied)
    worst_served = np.argsort(-distances, kind="stable")[: available.size]
    for slot, row in zip(available, worst_served, strict=False):
        updated[slot] = grid[int(row)]
    return updated
