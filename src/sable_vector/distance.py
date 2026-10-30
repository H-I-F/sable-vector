"""Similarity kernels. Matrix-vector products only; never a Python loop.

Everything a caller perceives as "search" collapses here into one of three BLAS
calls, so the shape of the data matters more than the shape of the code:

* inputs are **contiguous little-endian float32** ``(n, d)`` matrices. BLAS
  reaches its vectorised kernels only on contiguous operands, and a
  non-contiguous view silently costs a copy per query — the single largest
  avoidable latency term in a brute-force scan.
* scores are **float64**. Reading them from a float32 accumulator is pointless
  when the ranking decision is a subtraction of two nearby numbers.

Numerical contract
------------------
:func:`l2_distances` uses the Gram expansion ``||m - q||² = ||m||² - 2 m·q +
||q||²`` so that a 50k x 1536 scan never materialises an ``(n, d)`` difference
temporary. The expansion is exact in real arithmetic and carries a *bounded
absolute* error in float32: a dot product over ``d`` terms accumulates roughly
``eps * sqrt(d)`` relative error, so once the three terms cancel near zero
distance what is left is ``sqrt(2 * eps * sqrt(d)) * ||q||`` — about 1e-2 on a
64-dimension unit-scale vector. That is far below any ranking decision the
engine makes, and it is why ``precise=True`` exists for callers who need to
*print* a small distance rather than order by one.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sable_vector.types import DistanceMetric

__all__ = [
    "SCORE_DTYPE",
    "VECTOR_DTYPE",
    "as_matrix",
    "as_query",
    "cosine_similarities",
    "dot_scores",
    "l2_distances",
    "normalise",
    "normalise_matrix",
    "score_matrix",
    "top_k",
]

VECTOR_DTYPE = np.dtype("<f4")
SCORE_DTYPE = np.dtype("<f8")


def as_matrix(vectors: np.ndarray | Sequence[np.ndarray], *, dim: int | None = None) -> np.ndarray:
    """Coerce ``vectors`` into a contiguous ``(n, d)`` float32 matrix.

    Args:
        vectors: A 2-D array, or a sequence of 1-D arrays of equal length.
        dim: Expected column count; mismatches are rejected rather than
            silently padded, because a padded vector scores as a different point.

    Returns:
        C-contiguous ``(n, d)`` array in :data:`VECTOR_DTYPE`.

    Raises:
        ValueError: An element is not one-dimensional, lengths disagree, or the
            width does not match ``dim``.
    """
    if isinstance(vectors, np.ndarray) and vectors.ndim == 2:
        matrix = np.ascontiguousarray(vectors, dtype=VECTOR_DTYPE)
    else:
        rows = [as_query(row) for row in vectors]
        if not rows:
            width = dim if dim is not None else 0
            return np.empty((0, width), dtype=VECTOR_DTYPE)
        widths = {row.shape[0] for row in rows}
        if len(widths) != 1:
            raise ValueError(f"vectors have inconsistent dimensions: {sorted(widths)}")
        matrix = np.ascontiguousarray(np.stack(rows), dtype=VECTOR_DTYPE)

    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {matrix.shape}")
    if dim is not None and matrix.shape[1] != dim:
        raise ValueError(f"expected dimension {dim}, got {matrix.shape[1]}")
    return matrix


def as_query(vector: np.ndarray | Sequence[float], *, dim: int | None = None) -> np.ndarray:
    """Coerce one embedding into a contiguous 1-D float32 vector."""
    query = np.ascontiguousarray(vector, dtype=VECTOR_DTYPE)
    if query.ndim != 1:
        raise ValueError(f"query must be one-dimensional, got shape {query.shape}")
    if dim is not None and query.shape[0] != dim:
        raise ValueError(f"expected dimension {dim}, got {query.shape[0]}")
    return query


def normalise(vector: np.ndarray) -> np.ndarray:
    """Project one vector onto the unit sphere, leaving the zero vector alone."""
    query = as_query(vector)
    norm = float(np.sqrt(np.einsum("i,i->", query, query, dtype=np.float64)))
    if norm == 0.0:
        return np.zeros_like(query)
    return np.asarray(query / norm, dtype=VECTOR_DTYPE)


def normalise_matrix(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation. Zero rows stay zero instead of producing NaN.

    Row norms accumulate in float64: a 1536-dimension float32 inner product of
    large-magnitude components can overflow the accumulator long before it
    overflows the result.
    """
    grid = as_matrix(matrix)
    norms = np.sqrt(np.einsum("ij,ij->i", grid, grid, dtype=np.float64))
    scaling = np.where(norms > 0.0, norms, 1.0).astype(VECTOR_DTYPE)[:, None]
    return np.asarray(grid / scaling, dtype=VECTOR_DTYPE)


def l2_distances(matrix: np.ndarray, query: np.ndarray, *, precise: bool = False) -> np.ndarray:
    """Euclidean distance from ``query`` to every row of ``matrix``.

    Args:
        matrix: ``(n, d)`` vectors.
        query: ``(d,)`` vector.
        precise: Use the subtractive form. Costs an ``(n, d)`` temporary and is
            therefore reserved for validation and for callers that need the
            exact small-distance value.

    Returns:
        ``(n,)`` float64 distances, lower is closer.
    """
    grid = as_matrix(matrix)
    probe = as_query(query, dim=grid.shape[1])

    if precise:
        difference = grid.astype(np.float64) - probe.astype(np.float64)
        return np.sqrt(np.einsum("ij,ij->i", difference, difference))

    gram = np.einsum("ij,ij->i", grid, grid, dtype=np.float64)
    cross = np.asarray(grid @ probe, dtype=np.float64)
    query_gram = float(np.einsum("i,i->", probe, probe, dtype=np.float64))
    squared = gram - 2.0 * cross + query_gram
    # ||m||² - 2 m·q + ||q||² is non-negative in exact arithmetic; in float32 it
    # can undershoot zero by the dot product's rounding noise, and sqrt of a
    # negative would be NaN rather than "very close".
    np.maximum(squared, 0.0, out=squared)
    return np.sqrt(squared)


def cosine_similarities(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine similarity in ``[-1, 1]``, higher is closer."""
    grid = as_matrix(matrix)
    probe = as_query(query, dim=grid.shape[1])
    return np.asarray(
        normalise_matrix(grid) @ normalise(probe),
        dtype=SCORE_DTYPE,
    )


def dot_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Inner product, unbounded, higher is closer."""
    grid = as_matrix(matrix)
    probe = as_query(query, dim=grid.shape[1])
    return np.asarray(grid @ probe, dtype=SCORE_DTYPE)


def distance_matrix(matrix: np.ndarray, query: np.ndarray, metric: DistanceMetric) -> np.ndarray:
    """Metric-native distances: lower is closer under every metric.

    For cosine and inner product the distance is the negated similarity, which
    is not a metric in the mathematical sense but is monotone in it — enough for
    ranking, which is the only thing that consumes it.
    """
    if metric is DistanceMetric.L2:
        return l2_distances(matrix, query)
    if metric is DistanceMetric.COSINE:
        return -cosine_similarities(matrix, query)
    return -dot_scores(matrix, query)


def score_matrix(matrix: np.ndarray, query: np.ndarray, metric: DistanceMetric) -> np.ndarray:
    """Similarity scores oriented so that higher always means closer."""
    if metric is DistanceMetric.L2:
        return -l2_distances(matrix, query)
    if metric is DistanceMetric.COSINE:
        return cosine_similarities(matrix, query)
    return dot_scores(matrix, query)


def top_k(
    scores: np.ndarray,
    k: int,
    *,
    larger_is_better: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the ``k`` best entries by partial sort.

    ``argpartition`` is O(n) rather than the O(n log n) of a full sort, which
    matters once n is the whole collection and k is ten. Only the surviving k
    entries are then fully ordered, and that ordering is stable so equal scores
    keep insertion order instead of shuffling between otherwise identical runs.

    Args:
        scores: ``(n,)`` scores.
        k: How many to keep; ``k >= n`` degenerates to a full stable sort.
        larger_is_better: Comparison direction under the active metric.

    Returns:
        ``(indices, values)`` best-first. Empty when ``k <= 0`` or ``n == 0``.
    """
    values = np.asarray(scores, dtype=SCORE_DTYPE)
    if k <= 0 or values.size == 0:
        empty_indices = np.empty(0, dtype=np.intp)
        return empty_indices, np.empty(0, dtype=SCORE_DTYPE)

    directed = -values if larger_is_better else values
    if k >= values.size:
        order = np.argsort(directed, kind="stable")
        return order, values[order]

    # kth is inclusive, so k-1 yields exactly k entries on the low side.
    candidates = np.argpartition(directed, k - 1)[:k]
    ranked = candidates[np.argsort(directed[candidates], kind="stable")]
    return ranked, values[ranked]
