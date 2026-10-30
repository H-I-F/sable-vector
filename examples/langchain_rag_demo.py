#!/usr/bin/env python3
"""Offline retrieval demo: a filter-first ``VectorStore`` answering a question.

Run it with nothing installed but this package:

    python examples/langchain_rag_demo.py

There is no model download and no network call. The embedder is the **hashing
trick** — every token is hashed to one of ``dim`` buckets with a signed weight,
which is not semantic but is deterministic, so the demo proves the part that
matters: that a metadata filter is answered *before* the search, and that the
number of vectors actually scored drops when the filter is selective.

Two searches are run against the same corpus and question:

1. **unfiltered** — the whole corpus is scored.
2. **filtered** to one tenant — the predicate is resolved through the inverted
   index, and only the survivors are scored.

The report prints ``scored`` for both. The second number being much smaller is
the entire claim of this engine, shown rather than asserted.

A third search runs with a filter on a field that was never indexed, to show
the fallback: it still returns the right documents, costs a full ranking, and
says so instead of raising.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from sable_vector.adapter.langchain import SableVectorStore
from sable_vector.types import DistanceMetric

DIM = 256
_TOKEN = re.compile(r"[a-z0-9]+")

CORPUS: list[tuple[str, dict[str, Any]]] = [
    (
        "The write-ahead log is fsynced once per record, which is what makes an "
        "acknowledged write durable.",
        {"tenant": "acme", "topic": "durability", "year": 2024},
    ),
    (
        "A compaction merges immutable tables and retires the inputs, deferring "
        "deletion until no snapshot pins them.",
        {"tenant": "acme", "topic": "compaction", "year": 2024},
    ),
    (
        "The inverted index stores each value as sortable text, so a range "
        "predicate is one contiguous prefix scan.",
        {"tenant": "acme", "topic": "indexing", "year": 2025},
    ),
    (
        "Residuals are stored instead of raw vectors because they have a smaller "
        "dynamic range and cost the same bytes.",
        {"tenant": "acme", "topic": "quantisation", "year": 2025},
    ),
    (
        "Pre-filtering resolves the predicate first and hands the surviving ids "
        "to the kernel as a candidate block.",
        {"tenant": "globex", "topic": "query", "year": 2024},
    ),
    (
        "Post-filtering ranks everything and walks back over the payloads, which "
        "is correct and slower under a selective predicate.",
        {"tenant": "globex", "topic": "query", "year": 2025},
    ),
    (
        "A torn write leaves a vector with no payload, never a payload with no "
        "vector; that direction is repaired at open.",
        {"tenant": "globex", "topic": "durability", "year": 2025},
    ),
    (
        "Snapshot reads pin tables rather than copying them, so a reader holds no "
        "lock and compaction runs underneath.",
        {"tenant": "globex", "topic": "concurrency", "year": 2025},
    ),
    (
        "k-means++ spreads the initial centroids by squared distance, which is "
        "what keeps two partitions from landing in one basin.",
        {"tenant": "initech", "topic": "clustering", "year": 2024},
    ),
    (
        "Empty clusters are reseeded onto the worst-served row rather than left "
        "as NaN, which keeps inertia monotone.",
        {"tenant": "initech", "topic": "clustering", "year": 2025},
    ),
    (
        "Partition pruning means nprobe, not corpus size, governs the cost of a "
        "query.",
        {"tenant": "initech", "topic": "query", "year": 2025},
    ),
    (
        "Payloads are the commit record: a record exists exactly when its payload "
        "does.",
        {"tenant": "initech", "topic": "durability", "year": 2024},
    ),
]

QUESTION = "how does a filter make a query cheaper?"


class HashingEmbedding:
    """Signed hashing embedder: deterministic, offline, and good enough to rank.

    Each token is hashed to one bucket and contributes a signed weight, so two
    texts sharing vocabulary land close together. It captures no semantics —
    that is fine, because the demo is about the filter, not the embedding.
    """

    def __init__(self, dim: int = DIM) -> None:
        self._dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        buckets = np.zeros(self._dim, dtype=np.float64)
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            bucket = value % self._dim
            # One bit of the digest picks the sign, which is what keeps
            # unrelated tokens from all pushing in the same direction.
            sign = 1.0 if value >> 63 else -1.0
            buckets[bucket] += sign
        norm = float(np.linalg.norm(buckets))
        if norm > 0.0:
            buckets /= norm
        return [float(value) for value in buckets]


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def show(label: str, documents: list[tuple[Any, float]]) -> None:
    for rank, (document, score) in enumerate(documents, start=1):
        metadata = ", ".join(f"{key}={value}" for key, value in document.metadata.items())
        print(f"  {rank}. [{score:+.4f}] {document.page_content[:66]}...  ({metadata})")


def main() -> int:
    home = Path(tempfile.mkdtemp(prefix="sable-vector-rag-")) / "collection"
    try:
        texts = [text for text, _ in CORPUS]
        metadatas = [metadata for _, metadata in CORPUS]

        store = SableVectorStore.from_texts(
            texts,
            HashingEmbedding(),
            metadatas,
            home=home,
            metric=DistanceMetric.COSINE,
            dim=DIM,
        )
        print(f"collection: {home}")
        print(f"indexed fields: {', '.join(store.indexed_fields)}")
        print(f"documents ingested: {len(texts)}")
        print(f"question: {QUESTION}")

        rule("1. unfiltered")
        everything = store.similarity_search_with_score(QUESTION, k=3)
        show("everything", everything)
        print(f"  scored {store.last_search.scored} of {len(texts)} vectors")

        rule("2. filtered to tenant=acme")
        filtered = store.similarity_search_with_score(
            QUESTION, k=3, filter={"tenant": "acme", "year": {"$gte": 2025}}
        )
        show("acme 2025+", filtered)
        print(f"  candidates admitted: {store.last_search.candidates}")
        print(f"  scored {store.last_search.scored} of {len(texts)} vectors")

        rule("3. filter on a field that was never indexed")
        # ``topic`` was declared, ``owner`` was not — the fallback must answer
        # the same question at full cost rather than refuse.
        fallback = store.similarity_search_with_score(
            QUESTION, k=3, filter={"owner": "platform"}
        )
        print(f"  mode: {store.last_search.mode}")
        print(f"  documents returned: {len(fallback)}")

        rule("verdict")
        first = store.last_search
        store.similarity_search(QUESTION, k=3)
        baseline = store.last_search.scored
        store.similarity_search(QUESTION, k=3, filter={"tenant": "acme"})
        narrow = store.last_search.scored
        print(f"  unfiltered scored {baseline}, filtered scored {narrow}")
        if narrow >= baseline:
            print("  FAILED: the filter did not reduce the vectors scored")
            return 1
        print("  OK: the filter reduced the work, not just the result set")
        print(f"  last mode observed: {first.mode}")
        store.close()
        return 0
    finally:
        shutil.rmtree(home.parent, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
