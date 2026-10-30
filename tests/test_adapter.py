"""Contracts for the framework-facing ``VectorStore`` adapter.

The adapter is where a framework-shaped request becomes an engine-shaped one, and
the tests here are about that translation rather than about plumbing. Three
things can go wrong in a translation, and each is checked:

* **the answer changes** — a filtered search must return exactly the documents an
  unfiltered search would, minus the ones the predicate rejects. So filtered
  results are compared against an independently computed expectation, not
  against whatever the store last returned.
* **the optimisation silently stops happening** — the adapter's entire reason to
  exist is that a filter is answered *before* the search. `last_search.scored`
  is asserted to drop below the corpus size, because an adapter that filters
  after the fact still returns correct answers and is therefore invisible to a
  correctness test.
* **the interface's defaults lie** — text has to survive a round trip through a
  payload slot it does not own, ids have to be stable enough that re-ingesting
  replaces rather than duplicates, and a filter on a field nobody declared has to
  degrade to a slower correct answer instead of raising.

The embedding model is a signed hashing embedder defined in this file: it ranks
texts by shared vocabulary, needs no network, and is deterministic, so a
relevance assertion is reproducible rather than a model version's opinion.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from sable_vector.adapter.langchain import (
    MAX_INDEXED_FIELDS,
    TEXT_FIELD,
    SableVectorStore,
)
from sable_vector.engine import VectorEngine
from sable_vector.types import DistanceMetric, VectorRecord

DIM = 96
_TOKEN = re.compile(r"[a-z0-9]+")

CORPUS: list[tuple[str, dict[str, object]]] = [
    ("the write-ahead log is fsynced once per record", {"tenant": "acme", "year": 2024}),
    ("a compaction merges immutable tables", {"tenant": "acme", "year": 2025}),
    ("the inverted index renders values as sortable text", {"tenant": "acme", "year": 2025}),
    ("residuals have a smaller dynamic range", {"tenant": "globex", "year": 2024}),
    ("pre-filtering resolves the predicate first", {"tenant": "globex", "year": 2025}),
    ("post-filtering ranks everything and walks back", {"tenant": "globex", "year": 2025}),
    ("a torn write leaves a vector with no payload", {"tenant": "initech", "year": 2024}),
    ("snapshot reads pin tables rather than copying them", {"tenant": "initech", "year": 2025}),
]


class HashingEmbedding:
    """Signed hashing embedder: deterministic, offline, and good enough to rank."""

    def __init__(self, dim: int = DIM) -> None:
        self._dim = dim
        self.calls: list[list[str]] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        buckets = np.zeros(self._dim, dtype=np.float64)
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            buckets[value % self._dim] += 1.0 if value >> 63 else -1.0
        norm = float(np.linalg.norm(buckets))
        if norm > 0.0:
            buckets /= norm
        return [float(item) for item in buckets]


def texts() -> list[str]:
    return [text for text, _ in CORPUS]


def metadatas() -> list[dict[str, object]]:
    return [dict(metadata) for _, metadata in CORPUS]


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    return tmp_path / "collection"


@pytest.fixture()
def store(home: Path) -> Iterator[SableVectorStore]:
    handle = SableVectorStore.from_texts(
        texts(),
        HashingEmbedding(),
        metadatas(),
        home=home,
        metric=DistanceMetric.COSINE,
        dim=DIM,
    )
    yield handle
    handle.close()


# -- construction ----------------------------------------------------------


def _expect_rejected(**kwargs: object) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        SableVectorStore(HashingEmbedding(), **kwargs)  # type: ignore[arg-type]


def test_exactly_one_of_engine_or_home_is_required(home: Path) -> None:
    engine = VectorEngine.open(home, metric=DistanceMetric.COSINE, dim=DIM)
    try:
        _expect_rejected()
        _expect_rejected(engine=engine, home=home)
    finally:
        engine.close()


def test_from_texts_derives_the_indexed_fields(home: Path) -> None:
    with SableVectorStore.from_texts(
        texts(), HashingEmbedding(), metadatas(), home=home, dim=DIM
    ) as built:
        assert set(built.indexed_fields) == {"tenant", "year"}


def test_the_derived_field_set_is_capped(home: Path) -> None:
    wide = [{f"f{index}": index for index in range(MAX_INDEXED_FIELDS + 8)}]
    with SableVectorStore.from_texts(
        ["one document"], HashingEmbedding(), wide, home=home, dim=DIM
    ) as built:
        assert len(built.indexed_fields) == MAX_INDEXED_FIELDS


def test_an_adopted_engine_is_not_closed_by_the_store(home: Path) -> None:
    engine = VectorEngine.open(home, metric=DistanceMetric.COSINE, dim=DIM)
    wrapper = SableVectorStore(HashingEmbedding(), engine=engine)
    wrapper.add_texts(["a document"], [{"tenant": "acme"}])
    wrapper.close()
    # The store does not own the handle, so closing it must not close the engine.
    engine.upsert("still-open", np.zeros(DIM, dtype="<f4"), {})
    engine.close()


def test_the_embedding_model_is_exposed(store: SableVectorStore) -> None:
    assert isinstance(store.embeddings, HashingEmbedding)


# -- writes ----------------------------------------------------------------


def test_add_texts_returns_content_derived_ids(home: Path) -> None:
    with SableVectorStore(HashingEmbedding(), home=home, dim=DIM) as built:
        ids = built.add_texts(["alpha", "beta"])
        assert len(set(ids)) == 2
        assert all(identifier.startswith("lc-") for identifier in ids)


def test_re_adding_the_same_text_replaces_rather_than_duplicates(
    store: SableVectorStore,
) -> None:
    before = sum(1 for _ in store.engine.storage.iter_payloads())
    store.add_texts(texts()[:2], metadatas()[:2])
    after = sum(1 for _ in store.engine.storage.iter_payloads())
    assert after == before


def test_add_texts_with_explicit_ids(home: Path) -> None:
    with SableVectorStore(HashingEmbedding(), home=home, dim=DIM) as built:
        assert built.add_texts(["alpha", "beta"], ids=["a", "b"]) == ["a", "b"]
        assert set(built.engine.storage.iter_vector_ids()) == {"a", "b"}


def test_two_rows_can_share_text_when_ids_are_given(home: Path) -> None:
    with SableVectorStore(HashingEmbedding(), home=home, dim=DIM) as built:
        built.add_texts(["same", "same"], ids=["one", "two"])
        assert sum(1 for _ in built.engine.storage.iter_payloads()) == 2


def _mismatched(home: Path, *args: object, **kwargs: object) -> None:
    """Build a throwaway store and make one rejected call, to see why it failed."""
    built = SableVectorStore(HashingEmbedding(), home=home, dim=DIM)
    try:
        built.add_texts(*args, **kwargs)  # type: ignore[arg-type]
    finally:
        built.close()


def test_a_metadata_length_mismatch_is_rejected(home: Path) -> None:
    with pytest.raises(ValueError, match="metadatas for"):
        _mismatched(home, ["a", "b"], [{"tenant": "acme"}])


def test_an_id_length_mismatch_is_rejected(home: Path) -> None:
    with pytest.raises(ValueError, match="ids for"):
        _mismatched(home, ["a", "b"], ids=["only-one"])


def test_add_texts_on_empty_input_does_nothing(home: Path) -> None:
    with SableVectorStore(HashingEmbedding(), home=home, dim=DIM) as built:
        assert built.add_texts([]) == []


def test_add_documents_round_trips_metadata(home: Path) -> None:
    with SableVectorStore(HashingEmbedding(), home=home, dim=DIM) as built:
        from sable_vector.adapter.langchain import Document

        built.add_documents(
            [
                Document(page_content="first", metadata={"tenant": "acme"}),
                Document(page_content="second", metadata={"tenant": "globex"}),
            ],
            ids=["d1", "d2"],
        )
        assert built.engine.storage.get_payload("d1")["tenant"] == "acme"
        assert built.engine.storage.get_payload("d2")["tenant"] == "globex"


def test_delete_removes_the_rows(store: SableVectorStore) -> None:
    identifiers = [f"del-{index}" for index in range(3)]
    store.add_texts(["x", "y", "z"], ids=identifiers)
    assert store.delete(identifiers) is True
    assert store.delete([]) is False


def test_delete_reports_false_for_unknown_ids(store: SableVectorStore) -> None:
    assert store.delete(["never-seen"]) is False


# -- reads -----------------------------------------------------------------


def test_a_document_round_trips_its_text_and_metadata(store: SableVectorStore) -> None:
    documents = store.similarity_search("the inverted index renders sortable text", k=8)
    assert documents
    for document in documents:
        assert document.page_content
        assert TEXT_FIELD not in document.metadata
        assert "tenant" in document.metadata


def test_scores_are_descending_and_higher_means_closer(
    store: SableVectorStore,
) -> None:
    ranked = store.similarity_search_with_score("pre-filtering the predicate", k=5)
    assert len(ranked) == 5
    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True)


def test_k_is_respected(store: SableVectorStore) -> None:
    assert len(store.similarity_search("anything", k=2)) == 2


def test_zero_k_returns_nothing(store: SableVectorStore) -> None:
    assert store.similarity_search("anything", k=0) == []


def test_search_by_vector_matches_search_by_text(store: SableVectorStore) -> None:
    model = store.embeddings
    assert isinstance(model, HashingEmbedding)
    question = "snapshot reads pin tables"
    by_text = [doc.page_content for doc in store.similarity_search(question, k=3)]
    by_vector = [
        doc.page_content
        for doc in store.similarity_search_by_vector(model.embed_query(question), k=3)
    ]
    assert by_text == by_vector


def test_a_stored_collection_survives_reopening(store: SableVectorStore, home: Path) -> None:
    store.close()
    reopened = SableVectorStore(HashingEmbedding(), home=home, dim=DIM)
    try:
        assert sum(1 for _ in reopened.engine.storage.iter_payloads()) == len(CORPUS)
        assert reopened.similarity_search("compaction merges tables", k=1)
    finally:
        reopened.close()


# -- the filter is the point -----------------------------------------------


def test_a_filter_admits_only_matching_documents(store: SableVectorStore) -> None:
    documents = store.similarity_search("anything", k=8, filter={"tenant": "globex"})
    assert documents
    assert {document.metadata["tenant"] for document in documents} == {"globex"}


def test_a_compound_filter_is_conjunctive(store: SableVectorStore) -> None:
    documents = store.similarity_search(
        "anything", k=8, filter={"tenant": "acme", "year": {"$gte": 2025}}
    )
    assert documents
    for document in documents:
        assert document.metadata["tenant"] == "acme"
        assert document.metadata["year"] >= 2025


def test_the_filter_reduces_the_work_not_just_the_result(store: SableVectorStore) -> None:
    store.similarity_search("predicate first", k=3)
    baseline = store.last_search
    assert baseline is not None
    assert baseline.scored == len(CORPUS)

    store.similarity_search("predicate first", k=3, filter={"tenant": "globex"})
    narrowed = store.last_search
    assert narrowed is not None
    assert narrowed.mode == "prefilter"
    assert narrowed.candidates == 3
    assert narrowed.scored == 3 < len(CORPUS)


def test_the_filtered_answer_is_the_unfiltered_answer_minus_the_rejected(
    store: SableVectorStore,
) -> None:
    question = "the inverted index renders values as sortable text"
    everything = store.similarity_search_with_score(question, k=len(CORPUS))
    surviving = [item for item in everything if item[0].metadata["tenant"] == "acme"]
    filtered = store.similarity_search_with_score(
        question, k=len(CORPUS), filter={"tenant": "acme"}
    )
    # Compared as sets, not as orderings: documents that share no vocabulary
    # with the question score exactly 0.0, and their relative order is a
    # rounding artefact rather than an answer. What must hold is that the
    # filtered search returns the same documents with the same scores.
    assert {document.page_content for document, _ in filtered} == {
        document.page_content for document, _ in surviving
    }
    assert sorted(score for _, score in filtered) == pytest.approx(
        sorted(score for _, score in surviving), abs=1e-6
    )


def test_an_unindexed_field_falls_back_instead_of_raising(
    store: SableVectorStore,
) -> None:
    documents = store.similarity_search("anything", k=8, filter={"owner": "nobody"})
    assert store.last_search is not None
    assert store.last_search.mode == "postfilter"
    # The predicate still holds: no row has that field, so nothing can match.
    assert documents == []


def test_the_fallback_still_filters_when_the_field_exists_but_is_unindexed(
    home: Path,
) -> None:
    with SableVectorStore(HashingEmbedding(), home=home, dim=DIM) as built:
        built.add_texts(
            ["alpha", "beta", "gamma"],
            [{"tenant": "acme"}, {"tenant": "globex"}, {"tenant": "globex"}],
            ids=["a", "b", "c"],
        )
        documents = built.similarity_search("alpha", k=3, filter={"tenant": "globex"})
        assert built.last_search is not None
        assert built.last_search.mode == "postfilter"
        assert {document.metadata["tenant"] for document in documents} == {"globex"}


def test_an_empty_store_answers_without_error(home: Path) -> None:
    with SableVectorStore(HashingEmbedding(), home=home, dim=DIM) as built:
        assert built.similarity_search("anything", k=3) == []
        assert built.last_search is not None
        assert built.last_search.scored == 0


def test_the_demo_runs_offline_and_exits_cleanly() -> None:
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "examples" / "langchain_rag_demo.py")],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(root),
    )
    assert result.returncode == 0, result.stderr
    assert "OK: the filter reduced the work" in result.stdout


def test_a_record_written_directly_is_visible_to_the_adapter(home: Path) -> None:
    engine = VectorEngine.open(home, metric=DistanceMetric.COSINE, dim=DIM)
    engine.upsert_many(
        [
            VectorRecord(
                "raw",
                np.asarray(HashingEmbedding().embed_query("snapshot reads"), dtype="<f4"),
                {"tenant": "acme", TEXT_FIELD: "snapshot reads pin tables"},
            )
        ]
    )
    wrapper = SableVectorStore(HashingEmbedding(), engine=engine)
    try:
        documents = wrapper.similarity_search("snapshot reads pin tables", k=1)
        assert documents[0].page_content == "snapshot reads pin tables"
        assert documents[0].metadata == {"tenant": "acme"}
    finally:
        engine.close()
