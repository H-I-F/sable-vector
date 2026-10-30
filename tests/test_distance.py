"""Accuracy and shape contracts for the similarity kernels.

These tests assert *numerical* behaviour, not just shape. A kernel that returns
the right array size with the wrong numbers is the failure mode this layer
actually risks: the Gram expansion in :func:`l2_distances` trades an exact
subtractive form for the absence of an ``(n, d)`` temporary, and the only thing
standing between that trade and silent mis-ranking is a bound on the error it
introduces.
"""

from __future__ import annotations

import numpy as np
import pytest

from sable_vector.distance import (
    SCORE_DTYPE,
    VECTOR_DTYPE,
    as_matrix,
    as_query,
    cosine_similarities,
    distance_matrix,
    dot_scores,
    l2_distances,
    normalise,
    normalise_matrix,
    score_matrix,
    top_k,
)
from sable_vector.types import DistanceMetric

DIM = 64


def matrix(rows: int, dim: int = DIM, *, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(rows, dim)).astype(VECTOR_DTYPE)


def query(dim: int = DIM, *, seed: int = 11) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=dim).astype(VECTOR_DTYPE)


# -- layout coercion ------------------------------------------------------


def test_as_matrix_returns_contiguous_float32() -> None:
    stacked = as_matrix([np.zeros(4, dtype=np.float64), np.ones(4, dtype=np.float64)])
    assert stacked.shape == (2, 4)
    assert stacked.dtype == VECTOR_DTYPE
    assert stacked.flags["C_CONTIGUOUS"]


def test_as_matrix_accepts_a_fortran_ordered_view() -> None:
    source = matrix(5, 8)
    column_major = np.asfortranarray(source)
    assert not column_major.flags["C_CONTIGUOUS"]
    coerced = as_matrix(column_major)
    assert coerced.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(coerced, source)


def test_as_matrix_rejects_inconsistent_dimensions() -> None:
    with pytest.raises(ValueError, match="inconsistent dimensions"):
        as_matrix([np.zeros(4), np.zeros(5)])


def test_as_query_rejects_a_mismatched_dimension() -> None:
    with pytest.raises(ValueError, match="expected dimension 8"):
        as_query(np.zeros(4), dim=8)


def test_as_matrix_of_empty_sequence_is_empty() -> None:
    empty = as_matrix([], dim=DIM)
    assert empty.shape == (0, DIM)
    assert empty.dtype == VECTOR_DTYPE


# -- L2 -------------------------------------------------------------------


def test_l2_matches_the_subtractive_reference() -> None:
    grid, probe = matrix(200), query()
    reference = np.linalg.norm(grid.astype(np.float64) - probe.astype(np.float64), axis=1)
    np.testing.assert_allclose(l2_distances(grid, probe), reference, rtol=1e-5, atol=1e-6)


def test_l2_precise_path_equals_the_reference_exactly() -> None:
    grid, probe = matrix(50), query()
    reference = np.linalg.norm(grid.astype(np.float64) - probe.astype(np.float64), axis=1)
    np.testing.assert_allclose(l2_distances(grid, probe, precise=True), reference, rtol=1e-12)


def test_l2_fast_path_stays_inside_its_documented_error_bound() -> None:
    """Near-identical vectors are where the Gram expansion loses precision.

    A float32 dot product over ``d`` terms carries roughly ``eps * sqrt(d)``
    relative error, and once the three expansion terms cancel that error is all
    that is left of the squared distance — so the distance itself inherits its
    square root. The bound is derived here rather than hard-coded so that a
    regression in the kernel cannot hide behind a loose magic number.
    """
    probe = query()
    grid = np.stack([probe, probe * (1.0 + 1e-6)])
    norm = float(np.linalg.norm(probe.astype(np.float64)))
    bound = float(np.sqrt(2.0 * np.finfo(np.float32).eps * np.sqrt(probe.size)) * norm)

    fast = l2_distances(grid, probe)
    precise = l2_distances(grid, probe, precise=True)
    for index in range(grid.shape[0]):
        assert abs(float(fast[index]) - float(precise[index])) <= bound

    # An identical pair is exact: ||m||² and ||q||² are the same float64 sum, so
    # the only surviving term is 2 * (||q||² - m·q), which cancels outright.
    assert float(fast[0]) == pytest.approx(0.0, abs=1e-12)


def test_l2_is_zero_for_an_identical_vector_and_symmetric() -> None:
    grid = matrix(4)
    distances = l2_distances(grid, grid[2])
    assert distances[2] == pytest.approx(0.0, abs=1e-3)
    assert distances.min() == distances[2]


def test_l2_handles_high_dimensions_without_overflow() -> None:
    grid = (np.random.default_rng(3).normal(size=(64, 1536)) * 1e3).astype(VECTOR_DTYPE)
    probe = (np.random.default_rng(4).normal(size=1536) * 1e3).astype(VECTOR_DTYPE)
    distances = l2_distances(grid, probe)
    assert np.isfinite(distances).all()
    reference = np.linalg.norm(grid.astype(np.float64) - probe.astype(np.float64), axis=1)
    np.testing.assert_allclose(distances, reference, rtol=1e-4)


# -- cosine ---------------------------------------------------------------


def test_cosine_matches_manual_normalisation() -> None:
    grid, probe = matrix(120), query()
    unit_grid = grid.astype(np.float64) / np.linalg.norm(grid.astype(np.float64), axis=1)[:, None]
    unit_probe = probe.astype(np.float64) / np.linalg.norm(probe.astype(np.float64))
    np.testing.assert_allclose(
        cosine_similarities(grid, probe), unit_grid @ unit_probe, rtol=1e-5, atol=1e-6
    )


def test_cosine_satisfies_its_canonical_values() -> None:
    unit = np.array([1.0, 0.0, 0.0], dtype=VECTOR_DTYPE)
    grid = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=VECTOR_DTYPE
    )
    scores = cosine_similarities(grid, unit)
    np.testing.assert_allclose(scores, [1.0, 0.0, -1.0], atol=1e-6)


def test_cosine_is_scale_invariant() -> None:
    grid, probe = matrix(30), query()
    scaled = cosine_similarities(grid * 1000.0, probe)
    np.testing.assert_allclose(scaled, cosine_similarities(grid, probe), atol=1e-5)


def test_cosine_stays_within_the_unit_interval() -> None:
    scores = cosine_similarities(matrix(500, 32), query(32))
    assert scores.min() >= -1.0 - 1e-6
    assert scores.max() <= 1.0 + 1e-6


# -- inner product --------------------------------------------------------


def test_dot_matches_numpy_dot() -> None:
    grid, probe = matrix(80), query()
    # float32 gemv accumulates ~eps * sqrt(d) relative error; 1e-5 is that bound
    # with margin, not a concession to a loose kernel.
    np.testing.assert_allclose(
        dot_scores(grid, probe), grid.astype(np.float64) @ probe.astype(np.float64), rtol=1e-5
    )


def test_dot_is_unbounded_and_signed() -> None:
    grid = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=VECTOR_DTYPE)
    np.testing.assert_allclose(dot_scores(grid, np.array([3.0, 0.0], dtype=VECTOR_DTYPE)), [3.0, -3.0])


# -- normalisation --------------------------------------------------------


def test_normalise_produces_a_unit_vector() -> None:
    unit = normalise(np.array([3.0, 4.0], dtype=VECTOR_DTYPE))
    assert unit.dtype == VECTOR_DTYPE
    assert float(np.linalg.norm(unit)) == pytest.approx(1.0, abs=1e-6)


def test_normalise_leaves_the_zero_vector_zero_instead_of_nan() -> None:
    zero = normalise(np.zeros(8, dtype=VECTOR_DTYPE))
    np.testing.assert_array_equal(zero, np.zeros(8, dtype=VECTOR_DTYPE))
    assert np.isfinite(zero).all()


def test_normalise_matrix_guards_zero_rows() -> None:
    grid = np.zeros((3, 4), dtype=VECTOR_DTYPE)
    grid[1] = 2.0
    unit = normalise_matrix(grid)
    assert np.isfinite(unit).all()
    np.testing.assert_allclose(np.linalg.norm(unit[1]), 1.0, atol=1e-6)
    np.testing.assert_array_equal(unit[0], np.zeros(4))


def test_normalise_matrix_accumulates_row_norms_in_float64() -> None:
    """Large float32 components must not overflow the norm accumulator."""
    grid = np.full((2, 1536), 1e20, dtype=VECTOR_DTYPE)
    unit = normalise_matrix(grid)
    assert np.isfinite(unit).all()
    np.testing.assert_allclose(np.linalg.norm(unit[0]), 1.0, atol=1e-6)


# -- metric orientation ---------------------------------------------------


@pytest.mark.parametrize("metric", list(DistanceMetric))
def test_score_is_the_negation_of_distance(metric: DistanceMetric) -> None:
    grid, probe = matrix(40), query()
    np.testing.assert_allclose(
        score_matrix(grid, probe, metric), -distance_matrix(grid, probe, metric)
    )


def test_metric_orientation_flags() -> None:
    assert DistanceMetric.L2.higher_is_better is False
    assert DistanceMetric.COSINE.higher_is_better is True
    assert DistanceMetric.DOT.higher_is_better is True
    assert DistanceMetric.L2.worst_score == float("inf")
    assert DistanceMetric.COSINE.worst_score == float("-inf")
    assert DistanceMetric.COSINE.normalised is True
    assert DistanceMetric.DOT.normalised is False


def test_every_metric_ranks_the_nearest_row_first() -> None:
    grid, probe = matrix(120), query()
    for metric in DistanceMetric:
        scores = score_matrix(grid, probe, metric)
        best = int(np.argmax(scores))
        reference = l2_distances(grid, probe, precise=True)
        if metric is DistanceMetric.L2:
            assert best == int(np.argmin(reference))
        else:
            assert best == int(np.argmin(reference))


# -- top-k selection ------------------------------------------------------


def test_top_k_returns_descending_scores() -> None:
    scores = np.array([0.1, 0.9, 0.5, 0.7, 0.3])
    indices, values = top_k(scores, 3)
    np.testing.assert_array_equal(indices, [1, 3, 2])
    np.testing.assert_allclose(values, [0.9, 0.7, 0.5])


def test_top_k_supports_the_lower_is_better_direction() -> None:
    indices, _ = top_k(np.array([5.0, 1.0, 3.0]), 2, larger_is_better=False)
    np.testing.assert_array_equal(indices, [1, 2])


def test_top_k_with_k_beyond_the_collection_is_a_full_stable_sort() -> None:
    scores = np.array([0.5, 0.5, 0.9])
    indices, _ = top_k(scores, 10)
    np.testing.assert_array_equal(indices, [2, 0, 1])


def test_top_k_is_empty_for_non_positive_k() -> None:
    indices, values = top_k(np.array([1.0, 2.0]), 0)
    assert indices.size == 0
    assert values.size == 0


def test_top_k_survives_an_empty_collection() -> None:
    indices, values = top_k(np.empty(0), 5)
    assert indices.size == 0 and values.size == 0


def test_top_k_scores_are_float64() -> None:
    _, values = top_k(np.array([0.2, 0.8], dtype=VECTOR_DTYPE), 2)
    assert values.dtype == SCORE_DTYPE


def test_top_k_matches_a_full_sort_on_a_large_collection() -> None:
    scores = np.random.default_rng(19).normal(size=5000)
    indices, values = top_k(scores, 10)
    expected = np.argsort(-scores, kind="stable")[:10]
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_allclose(values, scores[expected])
