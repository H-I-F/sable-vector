[English](adr-004-float32-codec-and-score-orientation.md) | [简体中文](adr-004-float32-codec-and-score-orientation.zh-CN.md)

# ADR-004: float32 zero-copy images, higher-is-closer scores

**Status:** Accepted — implemented in `src/sable_vector/codec.py` and `distance.py`.

## Context

Vectors have to survive in an LSM tree as opaque bytes and be fed back to NumPy as a
matrix. Copying on every read would dominate the cost of a point lookup. Separately,
`cosine` and `dot` naturally rank higher-is-better while `l2` ranks lower-is-better;
if that difference leaked out of the kernel, every downstream sort would need to know
which metric is in play.

## Decision

**On disk:** little-endian float32 with no framing, no header and no delimiter.
`encode_vector()` is `as_query(vector).tobytes()`; `decode_vector()` is a
`np.frombuffer` view onto the bytes the storage layer already returned, so there is
no second copy. `decode_matrix()` reshapes a concatenated block into `(n, d)`.
Endianness and dtype are pinned at both ends; `decode_vector(..., writeable=True)`
returns a private copy when a caller needs to mutate.

**In memory:** `score_matrix()` orients all three metrics so **higher always means
closer** — cosine similarity and dot product are used as-is, L2 becomes the negated
distance. Scores accumulate in float64 even though storage is float32, and `top_k()`
uses a stable `argsort` so ties resolve deterministically.

## Consequences

A decoded vector is a read-only view; code that mutates one must ask for a copy. A
float32 image is not portable across a byte-order change without an explicit
conversion, which is why the dtype constant lives in one place and is asserted on
decode.

Downstream code can compare, sort and truncate without branching on `metric=`. The
float32-storage / float64-scoring split is deliberate: it keeps the durable image
compact without letting accumulation error reorder near-tied neighbours. The
benchmark's recall figures are measured against this exhaustive kernel, so it is
also the ground truth for every approximate index.
