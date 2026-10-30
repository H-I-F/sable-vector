"""A ``VectorStore`` that answers a metadata filter before it scores anything.

Most vector store adapters are shaped the same way: embed the query, ask the
store for ``k`` neighbours, then drop the ones the caller's filter rejects. That
is correct and it is also where the waste is — a filter matching three percent
of a corpus still costs a full search, and the caller cannot ask for "the
nearest ten that match" because the store only knows how to answer "the nearest
ten".

This adapter does not do that, because the engine underneath it does not have
to. A filter becomes :class:`~sable_vector.query.prefilter.PrefilterExecutor`,
which resolves the predicate through the scalar inverted index first and hands
the surviving ids to the kernel as a candidate block — so a selective predicate
makes the query *cheaper*, not just narrower.

Three decisions are worth stating because they are the ones a caller can trip
over.

**Text lives in the payload.** There is no second storage slot beside the
vector, so ``page_content`` is stored under :data:`TEXT_FIELD` inside the
payload and stripped back out when a document is reconstructed. It is the schema
cost of having one record rather than two, and it is visible rather than
cleverly hidden.

**Ids are content-derived by default.** :meth:`_content_id` hashes the text, so
re-ingesting the same document replaces it instead of duplicating it. A caller
who wants two rows with identical text supplies ids explicitly.

**Indexed fields are declared, not inferred on the fly.** The inverted index is
built per field at construction, so a field must be known before it can be
filtered cheaply. :meth:`SableVectorStore.from_texts` therefore derives the
field set from the metadata it is handed, and a later filter naming a field that
was never declared falls back to post-filtering — which is slower but returns
the same answer, and is reported through :attr:`last_search` rather than
surfacing as an error.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from sable_vector.distance import VECTOR_DTYPE, as_matrix, as_query
from sable_vector.engine import VectorEngine
from sable_vector.index.flat import FlatIndex
from sable_vector.query.postfilter import PostfilterExecutor
from sable_vector.query.prefilter import PrefilterExecutor
from sable_vector.types import DistanceMetric, Match, MetadataFilter, VectorId

__all__ = [
    "MAX_INDEXED_FIELDS",
    "SableVectorStore",
    "TEXT_FIELD",
    "Embedder",
]

TEXT_FIELD = "__text__"
"""Payload key holding ``page_content``; stripped when a document is rebuilt."""

MAX_INDEXED_FIELDS = 32
"""Ceiling on derived indexed fields, so a wild corpus cannot inflate writes."""


try:  # pragma: no cover - depends on what the host application has installed
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
except ImportError:  # pragma: no cover

    @dataclass(slots=True)
    class Document:  # type: ignore[no-redef]
        """Minimal stand-in so the adapter is usable without LangChain installed."""

        page_content: str
        metadata: dict[str, Any] = field(default_factory=dict)

    class Embeddings(Protocol):  # type: ignore[no-redef]
        """The two calls this adapter makes of an embedding model."""

        def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

        def embed_query(self, text: str) -> list[float]: ...


class Embedder(Protocol):
    """Anything that can embed a batch of texts and a single query."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class SearchReport:
    """How the last filtered search was actually answered.

    Attributes:
        mode: ``"prefilter"`` when the predicate was resolved through the
            inverted index, ``"postfilter"`` when it named a field that was
            never indexed, ``"unfiltered"`` when there was no predicate.
        candidates: Ids the predicate admitted, when it was applied first.
        scored: Vectors handed to the kernel. The number that says whether the
            filter bought anything.
    """

    mode: str
    candidates: int | None
    scored: int


class SableVectorStore:
    """A durable, filter-first ``VectorStore`` over one collection.

    Args:
        embedding: Anything with ``embed_documents`` and ``embed_query``.
        engine: An open collection. Mutually exclusive with ``home``.
        home: Directory to open (or recover). Ignored when ``engine`` is given.
        metric: Geometry for scoring; cosine is the framework default.
        dim: Vector width; adopted from the first write when omitted.
        indexed_fields: Payload attributes to maintain inverted entries for.
        options: Storage-engine tuning, when this store opens the directory.

    Raises:
        ValueError: Neither ``engine`` nor ``home`` was given, or both were.
    """

    def __init__(
        self,
        embedding: Embedder | Embeddings,
        *,
        engine: VectorEngine | None = None,
        home: Path | str | None = None,
        metric: DistanceMetric = DistanceMetric.COSINE,
        dim: int | None = None,
        indexed_fields: Sequence[str] = (),
        options: Any | None = None,
    ) -> None:
        if (engine is None) == (home is None):
            raise ValueError("give exactly one of engine or home")
        self._embedding = embedding
        self._owns_engine = engine is None
        self._engine = engine or VectorEngine.open(
            Path(str(home)),
            metric=metric,
            dim=dim,
            indexed_fields=tuple(indexed_fields),
            options=options,
        )
        self._last: SearchReport | None = None

    # -- introspection -----------------------------------------------------

    @property
    def embeddings(self) -> Embedder | Embeddings:
        """The embedding model, as the interface requires it be exposed."""
        return self._embedding

    @property
    def engine(self) -> VectorEngine:
        """The underlying collection, for callers that need more than the protocol."""
        return self._engine

    @property
    def indexed_fields(self) -> tuple[str, ...]:
        return self._engine.indexed_fields

    @property
    def last_search(self) -> SearchReport | None:
        """How the most recent search was answered, for callers tuning a filter."""
        return self._last

    # -- writes ------------------------------------------------------------

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict[str, Any]] | None = None,
        *,
        ids: list[VectorId] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Embed, store and index a batch of texts.

        Args:
            texts: Documents' text.
            metadatas: Parallel attribute dicts; defaults to empty.
            ids: Identifiers to use; defaults to a hash of the text, which
                makes re-adding the same text idempotent.

        Returns:
            The ids the rows were written under.
        """
        rows = list(texts)
        if not rows:
            return []
        if metadatas is None:
            metadatas = [{} for _ in rows]
        if len(metadatas) != len(rows):
            raise ValueError(
                f"{len(metadatas)} metadatas for {len(rows)} texts"
            )
        if ids is None:
            ids = [_content_id(text) for text in rows]
        elif len(ids) != len(rows):
            raise ValueError(f"{len(ids)} ids for {len(rows)} texts")

        matrix = self._embed_documents(rows)
        for index, (vector_id, text, metadata) in enumerate(
            zip(ids, rows, metadatas, strict=True)
        ):
            payload = dict(metadata)
            payload[TEXT_FIELD] = text
            self._engine.upsert(vector_id, matrix[index], payload)
        return list(ids)

    def add_documents(
        self,
        documents: Sequence[Document],
        *,
        ids: list[VectorId] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Store already-built documents."""
        return self.add_texts(
            [document.page_content for document in documents],
            [dict(document.metadata) for document in documents],
            ids=ids,
            **kwargs,
        )

    def delete(self, ids: Sequence[VectorId] | None = None, **kwargs: Any) -> bool:
        """Remove rows by id.

        Returns:
            ``True`` when at least one row was removed.
        """
        if not ids:
            return False
        return any(self._engine.delete(vector_id) for vector_id in ids)

    # -- reads -------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        *,
        filter: MetadataFilter | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """The nearest ``k`` documents satisfying ``filter``, best first."""
        return [
            document
            for document, _ in self.similarity_search_with_score(
                query, k, filter=filter, **kwargs
            )
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        *,
        filter: MetadataFilter | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Nearest ``k`` documents with their scores, higher meaning closer."""
        return [
            (_document(match), match.score)
            for match in self._search_matches(query, k, filter=filter)
        ]

    def similarity_search_by_vector(
        self,
        embedding: Sequence[float] | np.ndarray,
        k: int = 4,
        *,
        filter: MetadataFilter | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Search with an already-computed embedding."""
        probe = np.asarray(embedding, dtype=VECTOR_DTYPE)
        return [_document(match) for match in self._rank(probe, k, filter)]

    # -- construction ------------------------------------------------------

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embedder | Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        *,
        home: Path | str,
        ids: list[VectorId] | None = None,
        metric: DistanceMetric = DistanceMetric.COSINE,
        dim: int | None = None,
        max_indexed_fields: int = MAX_INDEXED_FIELDS,
        options: Any | None = None,
        **kwargs: Any,
    ) -> SableVectorStore:
        """Build a store and fill it in one call.

        The indexed field set is derived from the metadata handed in, because a
        field can only be filtered cheaply if the inverted index was built for
        it — and this is the only moment the whole corpus is visible at once.

        Args:
            texts: Documents' text.
            embedding: Embedding model to bind.
            metadatas: Parallel attribute dicts.
            home: Directory to open or recover.
            ids: Identifiers to store under.
            metric: Geometry for scoring.
            dim: Vector width; adopted from the first write when omitted.
            max_indexed_fields: Cap on how many fields are derived.
            options: Storage-engine tuning.

        Returns:
            A store holding every text.
        """
        declared: list[str] = []
        for metadata in metadatas or ():
            for key in metadata:
                if key not in declared and len(declared) < max_indexed_fields:
                    declared.append(key)
        store = cls(
            embedding,
            home=home,
            metric=metric,
            dim=dim,
            indexed_fields=tuple(declared),
            options=options,
        )
        store.add_texts(texts, metadatas, ids=ids, **kwargs)
        return store

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the collection if this store opened it."""
        if self._owns_engine:
            self._engine.close()

    def __enter__(self) -> SableVectorStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _search_matches(
        self, query: str, k: int, *, filter: MetadataFilter | None
    ) -> list[Match]:
        return self._rank(self._embed_query(query), k, filter)

    def _rank(
        self, probe: np.ndarray, k: int, predicate: MetadataFilter | None
    ) -> list[Match]:
        """Resolve the predicate, then score only what survived it."""
        if k <= 0:
            self._last = SearchReport("unfiltered", None, 0)
            return []

        query = as_query(probe)
        width = int(query.shape[0])
        flat = FlatIndex.materialise(
            self._engine.storage, metric=self._engine.metric, dim=width
        )
        if flat.size == 0:
            self._last = SearchReport("unfiltered", None, 0)
            return []

        known = set(self._engine.indexed_fields)
        indexed = predicate is not None and all(
            key in known for key in predicate
        )
        if predicate and not indexed:
            # The field was never declared, so there is no posting list to read.
            # Walking the ranking costs more and answers the same question; the
            # alternative — raising — would make a filter a landmine.
            outcome = PostfilterExecutor(self._engine.storage, flat).search(
                query, k, predicate=predicate
            )
            self._last = SearchReport("postfilter", None, outcome.scored)
            return list(outcome.matches)

        executor = PrefilterExecutor(self._engine.storage, flat, self._engine.scalar)
        result = executor.search(query, k, predicate=predicate)
        self._last = SearchReport(
            "prefilter" if predicate else "unfiltered",
            result.candidates,
            result.scored,
        )
        return list(result.matches)

    def _embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = self._embedding.embed_documents(texts)
        return as_matrix(np.asarray(vectors, dtype=np.float32))

    def _embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self._embedding.embed_query(text), dtype=VECTOR_DTYPE)


def _content_id(text: str) -> VectorId:
    """A stable id derived from the text, so re-adding replaces rather than duplicates."""
    return "lc-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _document(match: Match) -> Document:
    """Rebuild a document from a payload, lifting the text back out of it."""
    payload = dict(match.payload)
    text = payload.pop(TEXT_FIELD, "")
    return Document(page_content=str(text), metadata=payload)
