"""Post-filtering: same answer as pre-filtering, opposite cost profile.

The load-bearing assertion here is the *equivalence* of the two orderings: for
the same query and predicate, post-filtering must return exactly the same top-k
as pre-filtering. A divergence would mean one of the two executors is wrong
about what a hybrid query means, and nothing else in the suite would catch it.

The second assertion is the cost profile, because that is the only thing that
distinguishes a good implementation from one that accidentally does the other
job: a permissive predicate should stop after roughly *k* payload reads, and a
selective one should be visibly expensive. Both are measured, not asserted by
argument.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sable_vector.engine import VectorEngine
from sable_vector.index.flat import FlatIndex
from sable_vector.index.scalar import ScalarIndex
from sable_vector.query.postfilter import PostfilterExecutor
from sable_vector.query.prefilter import PrefilterExecutor
from sable_vector.types import DistanceMetric

DIM = 12
FIELDS = ("tenant", "tier")


def corpus() -> tuple[list[str], np.ndarray, list[dict]]:
    rng = np.random.default_rng(31)
    labels: list[str] = []
    rows: list[np.ndarray] = []
    payloads: list[dict] = []
    index = 0
    for tenant in ("acme", "globex", "initech"):
        for tier in (1, 5, 9):
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
def flat(engine: VectorEngine) -> FlatIndex:
    return FlatIndex.materialise(engine.storage, metric=engine.metric, dim=DIM)


@pytest.fixture()
def executor(engine: VectorEngine, flat: FlatIndex) -> PostfilterExecutor:
    return PostfilterExecutor(engine.storage, flat)


def probe(seed: int = 5) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=DIM).astype("<f4")


# -- equivalence with pre-filtering ----------------------------------------


@pytest.mark.parametrize(
    "predicate",
    [
        {"tenant": "acme"},
        {"tenant": "globex", "tier": 5},
        {"tier": {"$gte": 5}},
        {"tenant": {"$in": ["acme", "initech"]}},
        {"tier": {"$ne": 5}},
    ],
)
def test_post_filtering_agrees_with_pre_filtering(
    engine: VectorEngine, flat: FlatIndex, predicate: dict
) -> None:
    """Both orderings must return the same top-k for the same question."""
    pre = PrefilterExecutor(engine.storage, flat, ScalarIndex(engine.storage, FIELDS))
    post = PostfilterExecutor(engine.storage, flat)
    query = probe(7)

    expected = [match.id for match in pre.search(query, 6, predicate=predicate).matches]
    outcome = post.search(query, 6, predicate=predicate)

    assert [match.id for match in outcome.matches] == expected
    assert all(match.payload for match in outcome.matches)


def test_scores_agree_as_well_as_ids(engine: VectorEngine, flat: FlatIndex) -> None:
    pre = PrefilterExecutor(engine.storage, flat, ScalarIndex(engine.storage, FIELDS))
    post = PostfilterExecutor(engine.storage, flat)
    query = probe(11)

    first = [match.score for match in pre.search(query, 5, predicate={"tenant": "acme"}).matches]
    second = [match.score for match in post.search(query, 5, predicate={"tenant": "acme"}).matches]
    np.testing.assert_allclose(first, second, rtol=1e-6)


# -- cost profile ----------------------------------------------------------


def test_a_permissive_predicate_stops_early(executor: PostfilterExecutor) -> None:
    """Most rows match, so the survivors sit at the top of the ranking."""
    outcome = executor.search(probe(), 5, predicate={"tier": {"$gte": 1}})
    assert outcome.examined == 5
    assert outcome.complete is True


def test_a_selective_predicate_walks_the_ranking(executor: PostfilterExecutor) -> None:
    """18 of 54 rows match: filling k=10 costs far more than 10 payload reads."""
    outcome = executor.search(probe(), 10, predicate={"tenant": "acme"})
    assert len(outcome.matches) == 10
    assert outcome.examined > 10
    assert outcome.scored == 54


def test_the_whole_corpus_is_scored_regardless_of_selectivity(
    executor: PostfilterExecutor
) -> None:
    outcome = executor.search(probe(), 3, predicate={"tenant": "acme"})
    assert outcome.scored == 54


def test_an_examine_budget_stops_the_walk(executor: PostfilterExecutor) -> None:
    outcome = executor.search(probe(), 10, predicate={"tenant": "acme"}, examine_budget=4)
    assert outcome.examined == 4
    assert outcome.complete is False
    assert len(outcome.matches) <= 4


def test_a_budget_that_fits_still_completes(executor: PostfilterExecutor) -> None:
    outcome = executor.search(probe(), 3, predicate={"tenant": "acme"}, examine_budget=54)
    assert outcome.complete is True
    assert len(outcome.matches) == 3


# -- no index required -----------------------------------------------------


def test_a_field_that_was_never_indexed_is_still_queryable(
    engine: VectorEngine, flat: FlatIndex
) -> None:
    """The predicate is evaluated against the payload, not the key space."""
    executor = PostfilterExecutor(engine.storage, flat)
    outcome = executor.search(probe(), 4, predicate={"colour": "red"})
    assert outcome.matches == ()
    assert outcome.examined == 54


def test_an_unindexed_field_that_does_exist_matches(
    engine: VectorEngine, flat: FlatIndex
) -> None:
    engine.upsert("doc-red", probe(2), {"tenant": "acme", "tier": 1, "colour": "red"})
    rebuilt = FlatIndex.materialise(engine.storage, metric=engine.metric, dim=DIM)
    executor = PostfilterExecutor(engine.storage, rebuilt)

    outcome = executor.search(probe(), 4, predicate={"colour": "red"})
    assert [match.id for match in outcome.matches] == ["doc-red"]


def test_pre_filtering_rejects_what_post_filtering_accepts(
    engine: VectorEngine, flat: FlatIndex
) -> None:
    """The asymmetry is deliberate: one reads the index, the other the payload."""
    pre = PrefilterExecutor(engine.storage, flat, ScalarIndex(engine.storage, FIELDS))
    post = PostfilterExecutor(engine.storage, flat)
    query = probe(3)

    assert post.search(query, 3, predicate={"colour": "red"}).matches == ()
    with pytest.raises(ValueError, match="is not indexed"):
        pre.search(query, 3, predicate={"colour": "red"})


# -- edges -----------------------------------------------------------------


def test_no_predicate_degenerates_to_a_plain_ranking(executor: PostfilterExecutor) -> None:
    outcome = executor.search(probe(), 5)
    assert len(outcome.matches) == 5
    assert outcome.complete is True
    scores = [match.score for match in outcome.matches]
    assert scores == sorted(scores, reverse=True)


def test_an_empty_predicate_is_unrestricted(executor: PostfilterExecutor) -> None:
    assert len(executor.search(probe(), 5, predicate={}).matches) == 5


def test_k_beyond_the_matches_returns_them_all(executor: PostfilterExecutor) -> None:
    outcome = executor.search(probe(), 500, predicate={"tenant": "acme"})
    assert len(outcome.matches) == 18
    assert outcome.complete is False


def test_non_positive_k_examines_nothing(executor: PostfilterExecutor) -> None:
    outcome = executor.search(probe(), 0, predicate={"tenant": "acme"})
    assert outcome.matches == ()
    assert outcome.examined == 0
    assert outcome.complete is True


def test_an_empty_corpus_returns_nothing(tmp_path: Path) -> None:
    handle = VectorEngine.open(tmp_path / "db", indexed_fields=FIELDS)
    flat = FlatIndex.materialise(handle.storage, metric=handle.metric, dim=DIM)
    outcome = PostfilterExecutor(handle.storage, flat).search(probe(), 3, predicate={"tier": 1})
    assert outcome.matches == ()
    assert outcome.scored == 0
    handle.close()


def test_a_dimension_mismatch_is_rejected(executor: PostfilterExecutor) -> None:
    with pytest.raises(ValueError, match="expected dimension"):
        executor.search(np.zeros(DIM + 1, dtype="<f4"), 3, predicate={"tier": 1})


def test_payloads_are_attached(executor: PostfilterExecutor) -> None:
    outcome = executor.search(probe(), 4, predicate={"tier": 9})
    assert all(set(match.payload) == {"tenant", "tier"} for match in outcome.matches)
