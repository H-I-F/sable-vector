"""Pre-filtering: the predicate must shrink the arithmetic, not decorate it.

The whole reason this executor exists is that a selective predicate makes the
vector work smaller. An implementation that scores every row and then discards
the non-matching ones returns exactly the same answer and is therefore
indistinguishable from a correct one *unless* the test measures the work — so
every result here is checked twice: against an independent float64 ranking for
correctness, and against a spy on the kernel for how many rows it was handed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sable_vector.engine import VectorEngine
from sable_vector.index import flat as flat_module
from sable_vector.index.flat import FlatIndex
from sable_vector.index.scalar import ScalarIndex
from sable_vector.query.prefilter import PrefilterExecutor
from sable_vector.types import DistanceMetric

DIM = 12
FIELDS = ("tenant", "tier")

TENANTS = ("acme", "globex", "initech")
TIERS = (1, 5, 9)


def corpus() -> tuple[list[str], np.ndarray, list[dict]]:
    """A deterministic corpus with a known tenant/tier distribution."""
    rng = np.random.default_rng(23)
    labels: list[str] = []
    rows: list[np.ndarray] = []
    payloads: list[dict] = []
    index = 0
    for tenant in TENANTS:
        for tier in TIERS:
            for _ in range(6):
                labels.append(f"doc-{index:03d}")
                rows.append(rng.normal(size=DIM).astype("<f4"))
                payloads.append({"tenant": tenant, "tier": tier})
                index += 1
    return labels, np.stack(rows), payloads


@pytest.fixture()
def engine(tmp_path: Path) -> VectorEngine:
    labels, matrix, payloads = corpus()
    handle = VectorEngine.open(
        tmp_path / "db", metric=DistanceMetric.COSINE, indexed_fields=FIELDS
    )
    for label, row, payload in zip(labels, matrix, payloads):
        handle.upsert(label, row, payload)
    return handle


@pytest.fixture()
def executor(engine: VectorEngine) -> PrefilterExecutor:
    flat = FlatIndex.materialise(engine.storage, metric=engine.metric, dim=DIM)
    return PrefilterExecutor(engine.storage, flat, ScalarIndex(engine.storage, FIELDS))


def reference(
    labels: list[str], matrix: np.ndarray, query: np.ndarray, allowed: set[str], k: int
) -> list[str]:
    """Independent float64 ranking restricted to the allowed ids."""
    rows = [position for position, label in enumerate(labels) if label in allowed]
    grid = matrix[rows].astype(np.float64)
    probe = query.astype(np.float64)
    unit = grid / np.linalg.norm(grid, axis=1)[:, None]
    scores = unit @ (probe / np.linalg.norm(probe))
    order = np.argsort(-scores, kind="stable")[:k]
    return [labels[rows[int(position)]] for position in order]


def spy_on_rows(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record how many rows the vector kernel is actually handed."""
    seen: list[int] = []
    real = flat_module.score_matrix

    def spy(matrix: np.ndarray, query: np.ndarray, metric: DistanceMetric) -> np.ndarray:
        seen.append(int(matrix.shape[0]))
        return real(matrix, query, metric)

    monkeypatch.setattr(flat_module, "score_matrix", spy)
    return seen


# -- correctness -----------------------------------------------------------


def test_only_records_matching_the_predicate_are_returned(executor: PrefilterExecutor) -> None:
    outcome = executor.search(np.zeros(DIM, dtype="<f4") + 1.0, 5, predicate={"tenant": "acme"})
    assert outcome.matches
    assert all(match.payload.get("tenant") == "acme" for match in outcome.matches)


def test_results_match_an_independent_ranking_over_the_candidates(
    executor: PrefilterExecutor, engine: VectorEngine
) -> None:
    labels, matrix, _ = corpus()
    probe = np.random.default_rng(77).normal(size=DIM).astype("<f4")
    allowed = set(executor.candidate_ids({"tenant": "globex"}) or set())

    outcome = executor.search(probe, 7, predicate={"tenant": "globex"})
    assert [match.id for match in outcome.matches] == reference(labels, matrix, probe, allowed, 7)


def test_a_conjunction_narrows_the_result_further(executor: PrefilterExecutor) -> None:
    broad = executor.search(np.ones(DIM, dtype="<f4"), 50, predicate={"tenant": "acme"})
    narrow = executor.search(
        np.ones(DIM, dtype="<f4"), 50, predicate={"tenant": "acme", "tier": 5}
    )
    assert len(narrow.matches) <= len(broad.matches)
    assert all(match.payload.get("tier") == 5 for match in narrow.matches)


def test_payloads_are_attached_to_every_match(executor: PrefilterExecutor) -> None:
    outcome = executor.search(np.ones(DIM, dtype="<f4"), 4, predicate={"tier": 9})
    assert all(set(match.payload) == {"tenant", "tier"} for match in outcome.matches)


def test_results_are_ordered_by_score(executor: PrefilterExecutor) -> None:
    outcome = executor.search(np.ones(DIM, dtype="<f4"), 6, predicate={"tenant": "initech"})
    scores = [match.score for match in outcome.matches]
    assert scores == sorted(scores, reverse=True)


def test_k_larger_than_the_candidates_returns_all_of_them(executor: PrefilterExecutor) -> None:
    outcome = executor.search(np.ones(DIM, dtype="<f4"), 500, predicate={"tier": 1})
    assert len(outcome.matches) == 18
    assert outcome.candidates == 18


# -- the arithmetic is actually smaller ------------------------------------


def test_similarity_is_computed_only_for_candidates(
    executor: PrefilterExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The kernel must be handed the candidate block, not the whole matrix."""
    seen = spy_on_rows(monkeypatch)
    outcome = executor.search(np.ones(DIM, dtype="<f4"), 5, predicate={"tenant": "acme"})

    assert seen == [18]
    assert outcome.scored == 18
    assert executor.flat.size == 54
    assert outcome.scored < executor.flat.size


def test_a_narrower_predicate_scores_fewer_rows(
    executor: PrefilterExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = spy_on_rows(monkeypatch)
    executor.search(np.ones(DIM, dtype="<f4"), 3, predicate={"tenant": "acme", "tier": 5})
    assert seen == [6]


def test_an_empty_candidate_set_never_reaches_the_kernel(
    executor: PrefilterExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = spy_on_rows(monkeypatch)
    outcome = executor.search(np.ones(DIM, dtype="<f4"), 5, predicate={"tenant": "nobody"})

    assert seen == []
    assert outcome.matches == ()
    assert outcome.candidates == 0
    assert outcome.scored == 0


def test_an_unrestricted_query_scores_the_whole_corpus(
    executor: PrefilterExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = spy_on_rows(monkeypatch)
    outcome = executor.search(np.ones(DIM, dtype="<f4"), 5)

    assert seen == [54]
    assert outcome.candidates is None
    assert outcome.scored == 54


def test_an_empty_predicate_is_treated_as_unrestricted(
    executor: PrefilterExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = spy_on_rows(monkeypatch)
    outcome = executor.search(np.ones(DIM, dtype="<f4"), 5, predicate={})
    assert seen == [54]
    assert outcome.candidates is None


def test_a_range_predicate_scores_only_the_rows_inside_it(
    executor: PrefilterExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = spy_on_rows(monkeypatch)
    outcome = executor.search(np.ones(DIM, dtype="<f4"), 3, predicate={"tier": {"$gte": 5}})
    assert outcome.candidates == 36
    assert seen == [36]


# -- edges -----------------------------------------------------------------


def test_an_unindexed_field_is_rejected(executor: PrefilterExecutor) -> None:
    with pytest.raises(ValueError, match="is not indexed"):
        executor.search(np.ones(DIM, dtype="<f4"), 3, predicate={"colour": "red"})


def test_an_unsupported_operator_is_rejected(executor: PrefilterExecutor) -> None:
    with pytest.raises(ValueError, match="unsupported filter operators"):
        executor.search(np.ones(DIM, dtype="<f4"), 3, predicate={"tenant": {"$regex": "a"}})


def test_non_positive_k_returns_nothing(executor: PrefilterExecutor) -> None:
    outcome = executor.search(np.ones(DIM, dtype="<f4"), 0, predicate={"tenant": "acme"})
    assert outcome.matches == ()
    assert outcome.candidates == 18


def test_a_deleted_vector_among_the_candidates_is_skipped(
    executor: PrefilterExecutor, engine: VectorEngine
) -> None:
    """A candidate the vector index no longer holds must not crash the query."""
    victim = next(iter(executor.candidate_ids({"tenant": "acme"}) or set()))
    engine.delete(victim)
    executor.flat.remove(victim)

    outcome = executor.search(np.ones(DIM, dtype="<f4"), 50, predicate={"tenant": "acme"})
    assert victim not in {match.id for match in outcome.matches}


def test_the_executor_exposes_the_indexes_it_was_built_with(
    executor: PrefilterExecutor
) -> None:
    assert executor.flat.metric is DistanceMetric.COSINE
    assert executor.scalar.fields == FIELDS


def test_an_empty_corpus_returns_nothing(tmp_path: Path) -> None:
    handle = VectorEngine.open(tmp_path / "db", indexed_fields=FIELDS)
    flat = FlatIndex.materialise(handle.storage, metric=handle.metric, dim=DIM)
    executor = PrefilterExecutor(handle.storage, flat, ScalarIndex(handle.storage, FIELDS))
    outcome = executor.search(np.ones(DIM, dtype="<f4"), 3, predicate={"tenant": "acme"})
    assert outcome.matches == ()
    assert outcome.scored == 0
    handle.close()
