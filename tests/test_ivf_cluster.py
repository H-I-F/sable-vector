"""Convergence and assignment contracts for the clustering state machine.

Clustering is the one place in this engine where a wrong answer is not a crash
but a quiet recall loss: two centroids in the same basin, or one that captured
nothing, leaves partitions the query path will faithfully skip. So these tests
assert the properties that make a partition trustworthy rather than the numbers
that happen to come out.

* every row is assigned to its actual nearest centroid, checked against an
  independent float64 brute force;
* inertia never increases across iterations — the defining property of Lloyd's
  alternation, and the difference between "it stopped" and "it converged";
* no centroid is ever NaN, including when a cluster captures nothing;
* the same corpus and seed give the same clustering, so a recall regression is
  reproducible instead of hiding behind nondeterminism.
"""

from __future__ import annotations

import numpy as np
import pytest

from sable_vector.index.ivf_cluster import (
    Clustering,
    assign,
    initialise_centroids,
    kmeans,
    residuals,
    squared_distances,
)

DIM = 8
K = 4


def blobs(per_blob: int = 40, dim: int = DIM, *, seed: int = 3, spread: float = 0.05) -> np.ndarray:
    """Well-separated Gaussian blobs: the ground truth a partition should find."""
    generator = np.random.default_rng(seed)
    centres = generator.normal(scale=12.0, size=(4, dim))
    parts = [
        centre + generator.normal(scale=spread, size=(per_blob, dim)) for centre in centres
    ]
    return np.ascontiguousarray(np.concatenate(parts), dtype="<f4")


def brute_force(matrix: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Independent float64 nearest-centroid assignment."""
    grid = matrix.astype(np.float64)
    cores = centroids.astype(np.float64)
    distances = np.linalg.norm(grid[:, None, :] - cores[None, :, :], axis=2)
    return np.argmin(distances, axis=1)


# -- distance block --------------------------------------------------------


def test_squared_distances_match_a_direct_computation() -> None:
    grid = blobs(20)
    cores = initialise_centroids(grid, 3, seed=1)
    expected = np.linalg.norm(
        grid.astype(np.float64)[:, None, :] - cores.astype(np.float64)[None, :, :], axis=2
    ) ** 2
    # An absolute bound, not a relative one: the gemm's rounding error scales
    # with the magnitude of the operands (the blob centres sit at scale 12),
    # while the quantity being computed here is a *within-blob* distance of
    # order 1e-2. Cancellation makes the ratio meaningless and the difference
    # is what actually matters.
    np.testing.assert_allclose(squared_distances(grid, cores), expected, rtol=0.0, atol=1e-3)


def test_squared_distances_are_never_negative() -> None:
    grid = blobs(30)
    cores = initialise_centroids(grid, 4, seed=2)
    assert (squared_distances(grid, cores) >= 0.0).all()


def test_a_width_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not match centroid width"):
        squared_distances(blobs(10, dim=8), np.zeros((3, 5), dtype="<f4"))


# -- assignment ------------------------------------------------------------


def test_assign_picks_the_nearest_centroid() -> None:
    grid = blobs(30)
    cores = initialise_centroids(grid, K, seed=5)
    labels, distances = assign(cores, grid)

    np.testing.assert_array_equal(labels, brute_force(grid, cores))
    assert distances.shape == (grid.shape[0],)
    assert (distances >= 0.0).all()


def test_assignment_is_a_fixed_point_after_convergence() -> None:
    """No row may prefer another centroid once the iteration has settled."""
    grid = blobs(50)
    result = kmeans(grid, K, seed=7)
    np.testing.assert_array_equal(result.assignments, brute_force(grid, result.centroids))


def test_counts_sum_to_the_corpus() -> None:
    result = kmeans(blobs(45), K, seed=9)
    assert int(result.counts.sum()) == 180
    assert result.assignments.shape == (180,)


# -- convergence -----------------------------------------------------------


def test_inertia_never_increases() -> None:
    grid = blobs(60)
    result = kmeans(grid, K, seed=11)
    history = np.asarray(result.inertia_history)
    assert np.all(np.diff(history) <= 1e-6), f"inertia rose: {history.tolist()}"


def test_a_separated_corpus_converges() -> None:
    result = kmeans(blobs(50), K, seed=13)
    assert result.converged is True
    assert result.iterations >= 1
    assert result.centroids.shape == (K, DIM)
    assert result.inertia == pytest.approx(result.inertia_history[-1])


def test_the_iteration_budget_is_respected() -> None:
    result = kmeans(blobs(40), K, seed=17, max_iterations=2)
    assert result.iterations <= 2


def test_a_max_iteration_of_zero_is_rejected() -> None:
    with pytest.raises(ValueError, match="max_iterations must be positive"):
        kmeans(blobs(10), 2, max_iterations=0)


def test_clustering_recovers_the_blobs() -> None:
    """Up to relabelling, the partition must match the generating groups."""
    grid = blobs(40)
    result = kmeans(grid, K, seed=19)
    truth = np.repeat(np.arange(4), 40)
    agreement = sum(
        np.bincount(result.assignments[truth == group]).max() for group in range(4)
    )
    assert agreement == grid.shape[0]


def test_a_single_cluster_centroid_is_the_mean() -> None:
    grid = blobs(20)
    result = kmeans(grid, 1, seed=23)
    np.testing.assert_allclose(result.centroids[0], grid.mean(axis=0), rtol=1e-5)


def test_centroids_are_finite_even_when_a_cluster_captures_nothing() -> None:
    """An empty cluster must be reseeded, never left as a NaN row."""
    grid = np.zeros((6, 4), dtype="<f4")
    grid[:, 0] = np.arange(6, dtype="<f4")
    result = kmeans(grid, 6, seed=29)
    assert np.isfinite(result.centroids).all()
    assert np.isfinite(result.inertia)


def test_identical_rows_do_not_produce_nan_centroids() -> None:
    grid = np.ones((10, 4), dtype="<f4")
    result = kmeans(grid, 3, seed=31)
    assert np.isfinite(result.centroids).all()


# -- determinism -----------------------------------------------------------


def test_the_same_seed_reproduces_the_clustering() -> None:
    grid = blobs(50)
    first = kmeans(grid, K, seed=41)
    second = kmeans(grid, K, seed=41)
    np.testing.assert_array_equal(first.centroids, second.centroids)
    np.testing.assert_array_equal(first.assignments, second.assignments)


def test_a_different_seed_can_move_the_starting_point() -> None:
    grid = blobs(60)
    assert not np.array_equal(
        initialise_centroids(grid, K, seed=1), initialise_centroids(grid, K, seed=2)
    )


def test_initialisation_draws_from_actual_rows() -> None:
    grid = blobs(30)
    cores = initialise_centroids(grid, K, seed=43)
    for row in cores:
        assert any(np.array_equal(row, candidate) for candidate in grid)


# -- residuals -------------------------------------------------------------


def test_residuals_are_the_difference_from_the_centroid() -> None:
    grid = blobs(30)
    result = kmeans(grid, K, seed=47)
    expected = grid.astype(np.float64) - result.centroids.astype(np.float64)[result.assignments]
    np.testing.assert_allclose(residuals(grid, result.centroids, result.assignments), expected, atol=1e-5)


def test_residuals_have_a_smaller_dynamic_range_than_the_vectors() -> None:
    """The reason to store residuals at all: quantising them costs far less."""
    grid = blobs(60)
    result = kmeans(grid, K, seed=53)
    spread = residuals(grid, result.centroids, result.assignments)
    assert float(np.abs(spread).max()) < float(np.abs(grid).max())


def test_residuals_reject_a_label_outside_the_centroids() -> None:
    grid = blobs(10)
    cores = initialise_centroids(grid, 2, seed=59)
    with pytest.raises(ValueError, match="centroid that does not exist"):
        residuals(grid, cores, np.full(grid.shape[0], 5))


def test_residuals_reject_a_mismatched_label_count() -> None:
    grid = blobs(10)
    cores = initialise_centroids(grid, 2, seed=61)
    with pytest.raises(ValueError, match="one entry per row"):
        residuals(grid, cores, np.zeros(3, dtype=np.int64))


# -- input validation ------------------------------------------------------


def test_an_empty_matrix_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty matrix"):
        kmeans(np.empty((0, DIM), dtype="<f4"), 2)


def test_a_non_positive_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        initialise_centroids(blobs(10), 0)


def test_k_beyond_the_row_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds the"):
        initialise_centroids(blobs(6), 100)


def test_high_dimensions_cluster_without_overflow() -> None:
    grid = (np.random.default_rng(67).normal(size=(60, 1536)) * 1e2).astype("<f4")
    result = kmeans(grid, 3, seed=67)
    assert np.isfinite(result.centroids).all()
    assert result.centroids.shape == (3, 1536)
    assert result.inertia_history[-1] <= result.inertia_history[0] + 1e-6


def test_the_result_is_immutable_state() -> None:
    result = kmeans(blobs(20), K, seed=71)
    assert isinstance(result, Clustering)
    with pytest.raises(AttributeError):
        result.inertia = 0.0  # type: ignore[misc]
