# sable-vector v0.1.0

**Embedded hybrid vector search for local AI agent runtimes.** 31 October 2026.

Most teams that build an agent eventually hit the same wall: the agent remembers
things, but the memory dies with the process. Vectors live in RAM, the index is a
`dict`, and a restart means rebuilding embeddings you have already paid for.
Reaching for a hosted vector database fixes durability but hands your corpus to a
service that needs to be run, secured, and paid for — usually for a workload that
is one laptop and a few hundred thousand embeddings.

`sable-vector` takes the third option: keep the embeddings local, make them
**durable by construction**, and let metadata participate in retrieval instead of
being applied afterwards in Python.

## Why hybrid retrieval matters here

Dense similarity answers "what looks like this?" But an agent almost never wants
"the nearest neighbours" — it wants **the nearest neighbours within my tenant, in
this language, of this document kind, excluding the ones I have already
revoked**. Pure vector search cannot express that; the usual workaround is to
retrieve a large buffer and filter in application code, which quietly degrades
recall and makes latency depend on how selective your predicate happens to be.

sable-vector treats the metadata filter as a first-class part of the query, with
two explicitly separate strategies — `PrefilterExecutor` (candidate set pruned
by the inverted index before scoring) and `PostfilterExecutor` (global top-N,
then predicate check). They are not interchangeable: each has its own cost curve
and its own failure mode, and which one you want depends on how selective the
predicate is. Exposing both, with their trade-offs documented, is the point.

## What ships in v0.1.0

**Durability.** Built on [`sable`](https://github.com/MKA930/sable), a pure-Python
LSM engine. A vector, its payload and its inverted entries are committed as one
unit; every mutation reaches a write-ahead log before it becomes visible, so a
killed process loses unacknowledged writes and never committed data. Reopening a
collection rebuilds the in-memory indexes from the WAL and SSTables, and a SIGKILL
chaos suite verifies exactly that property at several injection points.

**Geometry.** Similarity is a matrix–vector product over float32 vectors. The
kernels never loop over rows in Python, so recall is exact rather than
approximate, and there is no training step before the first query.

**Filtering.** Metadata is stored as ordinary sorted keys, so a union timeline is a prefix scan rather than a table walk. Indexed fields are declared per field and become usable by the pre-filter path.

**Scale-out path.** When exact search stops being affordable, train an IVF index
and buy Recall back with `nprobe`. The benchmark prints the latency/recall curve
so the operating point is a measurement on your own data, not a guess.

**Integration.** A `VectorStore`-style adapter implements `add_texts` /
`similarity_search` and runs fully offline with a local embedding callable — no
API keys, no network. Being a library, it also means no server: nothing opens a
socket or spawns a thread behind your back.

**Tooling.** Continuous integration across Python 3.10 and 3.11 (ruff + pytest),
a self-validating multi-stage container image whose default command runs the test
suite, a seeded 1536-dimension benchmark, and a single-step evolution ledger that
records one tick per merged PR.

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install "sable @ git+https://github.com/MKA930/sable.git@v0.1.1"
pytest -q
```

```python
import numpy as np
from sable_vector.engine import VectorEngine
from sable_vector.index.flat import FlatIndex
from sable_vector.index.scalar import ScalarIndex
from sable_vector.query.prefilter import PrefilterExecutor

engine = VectorEngine.open("collection.sv", metric="cosine",
                          indexed_fields=("tenant",))
engine.upsert("doc-1", np.random.randn(1536).astype("float32"),
              payload={"tenant": "acme"})

executor = PrefilterExecutor(
    engine.storage,
    FlatIndex.materialise(engine, metric="cosine"),
    ScalarIndex(engine.storage, ("tenant",)),
)
outcome = executor.search(np.random.randn(1536).astype("float32"), k=5,
                          predicate={"tenant": "acme"})
```

## Honest limits

- **Single process.** One writer at a time; this is an embedded engine, not a
  distributed store.
- **Pre-1.0.** Layout and API may still change; the CHANGELOG says so explicitly
  rather than letting you discover it during recovery.
- **Pre-filtering only sees indexed fields.** Rows written before a field was
  declared are invisible to that predicate until the corpus is rebuilt.
- **No built-in embedding model.** Bring your own callable. That is deliberate.

## Thanks

v0.1.0 also carries an upstream contribution: investigating a concurrency failure
during sable-vector's snapshot testing traced back to
[`sable`](https://github.com/MKA930/sable), whose memtable point lookups could
disagree with range scans about what a pinned sequence number resolves to. That
is fixed in `sable` v0.1.1, and this release depends on it.

MIT licensed.
