"""Exhaustive-recall contracts for the Flat index.

Flat's defining property is that it is never wrong: it scores every vector, so
its output *is* the definition of correct recall and every approximate index
added later is measured against it. That makes the tests here less about the
index and more about protecting the reference — an optimisation that changes
which rows are scored, or in what order, would quietly redefine correctness for
the whole project.

The recall assertions are therefore computed independently in float64 from the
raw vectors, never from the index's own kernels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sable_vector.codec import encode_vector
from sable_vector.index import flat as flat_module
from sable_vector.index.flat import FlatIndex
from sable_vector.storage import VectorStorage
from sable_vector.types import DistanceMetric, VectorId

DIM = 16
ROWS = 120


def vectors(rows: int = ROWS, dim: int = DIM, *, seed: int = 17) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(rows, dim)).astype("<f4")


def ids(rows: int = ROWS) -> list[VectorId]:
    return [f"doc-{index:04d}" for index in range(rows)]


@pytest.fixture()
def storage(tmp_path: Path) -> VectorStorage:
    handle = VectorStorage.open(tmp_path / "db")
    for label, row in zip(ids(), vectors(), strict=False):
        handle.put_vector(label, encode_vector(row))
    yield handle
    handle.close()


def reference(
    matrix: np.ndarray,
    query: np.ndarray,
    metric: DistanceMetric,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent float64 ranking, used as the definition of correct recall."""
    grid = matrix.astype(np.float64)
    probe = query.astype(np.float64)
    if metric is DistanceMetric.L2:
        scores = -np.linalg.norm(grid - probe, axis=1)
    elif metric is DistanceMetric.COSINE:
        unit = grid / np.linalg.norm(grid, axis=1)[:, None]
        scores = unit @ (probe / np.linalg.norm(probe))
    else:
        scores = grid @ probe
    order = np.argsort(-scores, kind="stable")[:k]
    return order, scores[order]


# -- materialisation -------------------------------------------------------


def test_materialise_stacks_every_stored_vector(storage: VectorStorage) -> None:
    index = FlatIndex.materialise(storage, metric=DistanceMetric.COSINE, dim=DIM)
    assert index.size == ROWS
    assert index.ids == ids()
    np.testing.assert_allclose(index.matrix, vectors(), rtol=1e-6)


def test_materialise_of_an_empty_store_is_empty(tmp_path: Path) -> None:
    handle = VectorStorage.open(tmp_path / "db")
    index = FlatIndex.materialise(handle, metric=DistanceMetric.L2, dim=DIM)
    assert index.size == 0
    assert index.matrix.shape == (0, DIM)
    assert index.search(vectors(1)[0], 5) == []
    handle.close()


def test_materialise_decodes_escaped_ids(tmp_path: Path) -> None:
    handle = VectorStorage.open(tmp_path / "db")
    handle.put_vector("a:b", encode_vector(vectors(1)[0]))
    index = FlatIndex.materialise(handle, metric=DistanceMetric.COSINE, dim=DIM)
    assert index.ids == ["a:b"]
    handle.close()


def test_materialise_rejects_a_wrong_row_width(storage: VectorStorage) -> None:
    with pytest.raises(ValueError, match="not a whole number"):
        FlatIndex.materialise(storage, metric=DistanceMetric.COSINE, dim=DIM + 1)


# -- exhaustive recall -----------------------------------------------------


@pytest.mark.parametrize("metric", list(DistanceMetric))
def test_search_matches_an_independent_float64_ranking(
    storage: VectorStorage, metric: DistanceMetric
) -> None:
    grid, labels = vectors(), ids()
    index = FlatIndex(metric=metric, dim=DIM, ids=labels, matrix=grid)
    probe = vectors(1, seed=99)[0]
    k = 10

    matches = index.search(probe, k)
    order, scores = reference(grid, probe, metric, k)

    assert [match.id for match in matches] == [labels[int(i)] for i in order]
    np.testing.assert_allclose([match.score for match in matches], scores, rtol=1e-5, atol=1e-6)
    assert matches == sorted(matches, key=lambda match: match.score, reverse=True)


@pytest.mark.parametrize("metric", list(DistanceMetric))
def test_recall_is_exactly_one_even_at_k_equal_to_one(
    storage: VectorStorage, metric: DistanceMetric
) -> None:
    """The nearest neighbour by definition is the nearest neighbour."""
    grid, labels = vectors(), ids()
    index = FlatIndex(metric=metric, dim=DIM, ids=labels, matrix=grid)
    probe = vectors(1, seed=123)[0]

    best = index.search(probe, 1)[0]
    order, _ = reference(grid, probe, metric, 1)
    assert best.id == labels[int(order[0])]


def test_distance_is_the_negation_of_score(storage: VectorStorage) -> None:
    index = FlatIndex(metric=DistanceMetric.L2, dim=DIM, ids=ids(), matrix=vectors())
    for match in index.search(vectors(1, seed=5)[0], 4):
        assert match.distance == pytest.approx(-match.score)
        assert match.distance >= 0.0


def test_a_query_wider_than_the_index_is_rejected(storage: VectorStorage) -> None:
    index = FlatIndex(metric=DistanceMetric.COSINE, dim=DIM, ids=ids(), matrix=vectors())
    with pytest.raises(ValueError, match="expected dimension"):
        index.search(vectors(1, dim=DIM + 1)[0], 3)


def test_k_larger_than_the_corpus_returns_the_whole_corpus(storage: VectorStorage) -> None:
    index = FlatIndex(metric=DistanceMetric.COSINE, dim=DIM, ids=ids(), matrix=vectors())
    assert len(index.search(vectors(1)[0], ROWS * 3)) == ROWS


def test_non_positive_k_returns_nothing(storage: VectorStorage) -> None:
    index = FlatIndex(metric=DistanceMetric.COSINE, dim=DIM, ids=ids(), matrix=vectors())
    assert index.search(vectors(1)[0], 0) == []
    assert index.search(vectors(1)[0], -1) == []


def test_payloads_are_left_empty_for_the_query_layer(storage: VectorStorage) -> None:
    index = FlatIndex(metric=DistanceMetric.COSINE, dim=DIM, ids=ids(), matrix=vectors())
    assert all(match.payload == {} for match in index.search(vectors(1)[0], 5))


# -- candidate restriction -------------------------------------------------


def test_candidates_are_the_only_rows_scored(
    storage: VectorStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A filtered query must do less arithmetic, not filter afterwards."""
    grid, labels = vectors(), ids()
    index = FlatIndex(metric=DistanceMetric.COSINE, dim=DIM, ids=labels, matrix=grid)
    scored: list[int] = []
    real = flat_module.score_matrix

    def spy(matrix: np.ndarray, query: np.ndarray, metric: DistanceMetric) -> np.ndarray:
        scored.append(int(matrix.shape[0]))
        return real(matrix, query, metric)

    monkeypatch.setattr(flat_module, "score_matrix", spy)

    allowed = labels[:40]
    matches = index.search(vectors(1, seed=7)[0], 5, candidate_ids=allowed)

    assert scored == [40]
    assert {match.id for match in matches} <= set(allowed)


def test_candidate_results_match_the_ranking_over_that_subset(storage: VectorStorage) -> None:
    grid, labels = vectors(), ids()
    index = FlatIndex(metric=DistanceMetric.COSINE, dim=DIM, ids=labels, matrix=grid)
    rows = list(range(10, 60))
    allowed = [labels[row] for row in rows]
    probe = vectors(1, seed=31)[0]

    matches = index.search(probe, 7, candidate_ids=allowed)
    order, _ = reference(grid[rows], probe, DistanceMetric.COSINE, 7)
    assert [match.id for match in matches] == [labels[rows[int(i)]] for i in order]


def test_unknown_candidates_are_ignored(storage: VectorStorage) -> None:
    index = FlatIndex(metric=DistanceMetric.COSINE, dim=DIM, ids=ids(), matrix=vectors())
    assert index.search(vectors(1)[0], 3, candidate_ids=["nope"]) == []
    assert len(index.search(vectors(1)[0], 3, candidate_ids=["doc-0001", "nope"])) == 1


def test_an_empty_candidate_set_returns_nothing(storage: VectorStorage) -> None:
    index = FlatIndex(metric=DistanceMetric.COSINE, dim=DIM, ids=ids(), matrix=vectors())
    assert index.search(vectors(1)[0], 3, candidate_ids=[]) == []


# -- streaming parity ------------------------------------------------------


def test_streaming_matches_the_resident_path_exactly(storage: VectorStorage) -> None:
    index = FlatIndex.materialise(storage, metric=DistanceMetric.COSINE, dim=DIM)
    probe = vectors(1, seed=41)[0]

    resident = index.search(probe, 8)
    streamed = index.search_streaming(storage, probe, 8, block_rows=16)

    assert [match.id for match in streamed] == [match.id for match in resident]
    np.testing.assert_allclose(
        [match.score for match in streamed], [match.score for match in resident], rtol=1e-6
    )


def test_streaming_holds_only_one_block_resident(storage: VectorStorage) -> None:
    index = FlatIndex(metric=DistanceMetric.COSINE, dim=DIM, ids=ids(), matrix=vectors())
    seen: list[int] = []
    real = flat_module.score_matrix

    def spy(matrix: np.ndarray, query: np.ndarray, metric: DistanceMetric) -> np.ndarray:
        seen.append(int(matrix.shape[0]))
        return real(matrix, query, metric)

    import sable_vector.index.flat as module

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module, "score_matrix", spy)
    try:
        index.search_streaming(storage, vectors(1)[0], 5, block_rows=32)
    finally:
        monkeypatch.undo()

    assert max(seen) <= 32
    assert sum(seen) == ROWS


def test_streaming_skips_the_io_of_rejected_candidates(
    storage: VectorStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = FlatIndex(metric=DistanceMetric.COSINE, dim=DIM, ids=ids(), matrix=vectors())
    seen: list[int] = []
    real = flat_module.score_matrix

    def spy(matrix: np.ndarray, query: np.ndarray, metric: DistanceMetric) -> np.ndarray:
        seen.append(int(matrix.shape[0]))
        return real(matrix, query, metric)

    monkeypatch.setattr(flat_module, "score_matrix", spy)
    index.search_streaming(storage, vectors(1)[0], 5, block_rows=64, candidate_ids=ids()[:25])

    assert sum(seen) == 25


def test_streaming_of_an_empty_store_returns_nothing(tmp_path: Path) -> None:
    handle = VectorStorage.open(tmp_path / "db")
    index = FlatIndex.empty(metric=DistanceMetric.COSINE, dim=DIM)
    assert index.search_streaming(handle, vectors(1)[0], 5) == []
    handle.close()


# -- mutation --------------------------------------------------------------


def test_add_appends_and_remove_drops_rows(storage: VectorStorage) -> None:
    index = FlatIndex.empty(metric=DistanceMetric.L2, dim=DIM)
    index.add("a", vectors(1, seed=1)[0])
    index.add("b", vectors(1, seed=2)[0])
    assert index.ids == ["a", "b"]
    assert index.size == 2

    assert index.remove("a") is True
    assert index.remove("a") is False
    assert index.ids == ["b"]
    assert index.matrix.shape == (1, DIM)


def test_add_replaces_an_existing_row_in_place_of_position(storage: VectorStorage) -> None:
    index = FlatIndex.empty(metric=DistanceMetric.L2, dim=DIM)
    index.add("a", vectors(1, seed=1)[0])
    index.add("b", vectors(1, seed=2)[0])
    replacement = vectors(1, seed=3)[0]
    index.add("a", replacement)

    assert index.ids == ["a", "b"]
    np.testing.assert_allclose(index.matrix[0], replacement)


def test_add_rejects_a_mismatched_width(storage: VectorStorage) -> None:
    index = FlatIndex.empty(metric=DistanceMetric.L2, dim=DIM)
    with pytest.raises(ValueError, match="expected dimension"):
        index.add("a", vectors(1, dim=DIM + 1)[0])


def test_the_matrix_is_replaced_not_mutated(storage: VectorStorage) -> None:
    """A reader holding the previous array must keep a consistent view."""
    index = FlatIndex.empty(metric=DistanceMetric.L2, dim=DIM)
    index.add("a", vectors(1, seed=1)[0])
    captured = index.matrix
    snapshot = captured.copy()

    index.add("b", vectors(1, seed=2)[0])

    assert index.matrix is not captured
    np.testing.assert_array_equal(captured, snapshot), "the captured view was mutated"


def test_iter_rows_walks_ids_and_matrix_together(storage: VectorStorage) -> None:
    grid, labels = vectors(4), ids(4)
    index = FlatIndex(metric=DistanceMetric.DOT, dim=DIM, ids=labels, matrix=grid)
    walked = list(index.iter_rows())
    assert [label for label, _ in walked] == labels
    np.testing.assert_allclose(np.stack([row for _, row in walked]), grid)


def test_a_mismatched_id_and_matrix_is_rejected(storage: VectorStorage) -> None:
    with pytest.raises(ValueError, match="same number of rows"):
        FlatIndex(metric=DistanceMetric.DOT, dim=DIM, ids=ids(3), matrix=vectors(2))


def test_empty_index_reports_zero_length(storage: VectorStorage) -> None:
    index = FlatIndex.empty(metric=DistanceMetric.DOT, dim=DIM)
    assert len(index) == 0
    assert index.metric is DistanceMetric.DOT
    assert index.dim == DIM
