[English](zero-copy-codec.md) | [简体中文](zero-copy-codec.zh-CN.md)

# Zero-copy vectors: the codec and the score orientation

> Two representations cross the storage boundary, and they want opposite things.
> A vector is read on every candidate of every query; a payload is read once per
> surviving hit. This guide walks through `codec.py` and `distance.py`, the
> invariants they enforce, and the failure modes that show up when a caller
> assumes either one is more forgiving than it is.

## 1. The format is the layout, not a header

A 1536-dimension embedding is 6144 bytes of little-endian float32:

```python
ITEM_SIZE: int = VECTOR_DTYPE.itemsize      # 4
def vector_bytes(dim: int) -> int:
    return dim * ITEM_SIZE
```

There is no framing, no length, no delimiter, no version byte. The reasons are
boring and decisive: a header has to be parsed on every one of the rows a scan
touches, and a delimiter has to be escaped. Endianness and dtype are pinned at
both ends (`VECTOR_DTYPE` is `<f4`), so an image written on one machine decodes on
another without a header explaining the layout. The layout is part of the format.

`VectorRecord.__post_init__` coerces on the way in, so every downstream kernel
sees the same thing:

```python
array = np.ascontiguousarray(self.vector, dtype="<f4")
if array.ndim != 1:
    raise ValueError(f"vector must be one-dimensional, got shape {array.shape}")
if array.size == 0:
    raise ValueError("vector must have at least one dimension")
```

A scalar, a Fortran-ordered slice and a float64 list all become one contiguous
`<f4` vector here. Doing this once at the boundary is what lets `encode_vector`
be a single `tobytes()`.

## 2. Encode is a memcpy; decode is a view

```python
def encode_vector(vector: np.ndarray) -> bytes:
    return as_query(vector).tobytes()

def decode_vector(blob: bytes, *, dim: int | None = None,
                  writeable: bool = False) -> np.ndarray:
    _check_length(blob, dim, single_row=True)
    view = np.frombuffer(blob, dtype=VECTOR_DTYPE)
    return view.copy() if writeable else view
```

`encode_vector` is the array's own memory in canonical order. `decode_vector` is
`np.frombuffer`: it reinterprets the bytes the storage layer already returned as
an array, with no second allocation. That is the whole meaning of "zero-copy" in
this project, and it has one hard condition — **the default view is read-only**.

The storage layer treats the bytes it hands out as immutable. A writable view onto
them would let a caller mutate a buffer that other readers may still be pinning.
So mutation is opt-in:

| Call | Result | When |
| --- | --- | --- |
| `decode_vector(blob)` | read-only view, aliases `blob` | scoring, comparison, matrix assembly |
| `decode_vector(blob, writeable=True)` | private copy | residual arithmetic, anything that writes |

`ivf_cluster.py` computes residuals against centroids, which needs a mutable
array; it asks for the copy. Everything else keeps the view.

### Length is checked before reinterpretation

`_check_length(blob, dim, single_row=...)` runs first. A blob whose length is not
a whole number of float32 items, or does not match `dim`, raises `ValueError`.
It is deliberately not a reshape with a guessed dimension: a truncated SSTable
block and a mis-typed 512-dim vector would otherwise both become a plausible-
looking array with the wrong shape, and the bug would surface much later as a
nonsensical ranking.

## 3. A matrix is one image, not N blobs

```python
def encode_matrix(matrix: np.ndarray) -> bytes:
    return as_matrix(matrix).tobytes()

def decode_matrix(blob: bytes, *, dim: int, writeable: bool = False) -> np.ndarray:
    _check_length(blob, dim, single_row=False)
    view = np.frombuffer(blob, dtype=VECTOR_DTYPE).reshape(-1, dim)
    return view.copy() if writeable else view
```

`n` is derived from the length; `dim` is required. The resident `FlatIndex` holds
an `(n, d)` array that came from this path, so a query is one gemv over a
contiguous block rather than N Python-level decodes. `search_streaming()` reads
the SSTables in blocks of `block_rows`, decoding one block at a time — same
`frombuffer` trick, bounded memory.

## 4. Payloads take the opposite bargain

```python
def encode_payload(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

def decode_payload(blob: bytes) -> dict[str, Any]:
    document = json.loads(blob.decode("utf-8"))
    if not isinstance(document, dict):
        raise ValueError("payload must decode to a JSON object")
    return document
```

Sorted keys and tight separators make equal payloads produce equal bytes, which
is what lets a payload be compared, hashed, or used as part of an index key
without re-parsing it. `decode_payload` insists on a JSON object: a stored list
or string is corruption, not a payload with unusual shape. `ensure_ascii=False`
keeps non-ASCII attributes readable from a shell.

The trade is explicit in the module docstring: payloads pay for a parse, vectors
do not, because payloads are read once per surviving hit while vectors are read
once per candidate.

## 5. One score orientation for all three metrics

L2 is a distance; cosine and dot are similarities. Rather than branching on
`metric=` in every consumer, `distance.py` exposes two consistently-oriented
views:

```python
def distance_matrix(matrix, query, metric):   # lower is closer, always
    if metric is DistanceMetric.L2:
        return l2_distances(matrix, query)
    if metric is DistanceMetric.COSINE:
        return -cosine_similarities(matrix, query)
    return -dot_scores(matrix, query)

def score_matrix(matrix, query, metric):      # higher is closer, always
    if metric is DistanceMetric.L2:
        return -l2_distances(matrix, query)
    if metric is DistanceMetric.COSINE:
        return cosine_similarities(matrix, query)
    return dot_scores(matrix, query)
```

`DistanceMetric` is a `str`-valued enum so a metric survives JSON configuration
intact — a reopened collection keeps the geometry it was created with. Three
properties encode the rest:

| Property | `l2` | `cosine` / `dot` | Used for |
| --- | --- | --- | --- |
| `higher_is_better` | `False` | `True` | `top_k` direction |
| `worst_score` | `+inf` | `-inf` | accumulator neutral element |
| `normalised` | `False` | `True` for cosine | project onto unit sphere |

`Match` keeps both numbers on purpose: `score` is the oriented ranking key,
`distance` is the raw geometric quantity for callers that need it. Mixing them up
is the single easiest way to get a reversed ranking that still looks plausible.

### Why float64 scoring over float32 storage

Storage is float32 to keep the durable image compact. But `top_k()` casts scores
to `SCORE_DTYPE` (float64) before selecting:

```python
values = np.asarray(scores, dtype=SCORE_DTYPE)
directed = -values if larger_is_better else values
if k >= values.size:
    order = np.argsort(directed, kind="stable")
    return order, values[order]
candidates = np.argpartition(directed, k - 1)[:k]
ranked = candidates[np.argsort(directed[candidates], kind="stable")]
```

`argpartition` is O(n) rather than O(n log n) — that matters when n is the whole
collection and k is ten. Only the surviving k are fully ordered, and the ordering
is **stable** so ties keep insertion order. Accumulating in float64 keeps a
near-tie from being decided by float32 rounding; the stable sort keeps an exact
tie from being decided by the hash seed. [ADR-004](../decisions/adr-004-float32-codec-and-score-orientation.md)
records both as deliberate choices.

## 6. Failure modes

| Mistake | What actually happens |
| --- | --- |
| Write through a decoded view | The array is read-only; assignment raises. Ask for `writeable=True`. |
| Guess `dim` on decode | `_check_length` raises rather than reshaping into a wrong-shaped matrix. |
| Store a non-dict payload | `decode_payload` raises `ValueError` on reopen. |
| Sort by `distance` expecting best-first | `distance` is lower-is-closer; ranking uses `score`. |
| Use an unstable sort on ties | Two identical runs can produce different orderings; bug reports stop reproducing. |
| Change the dtype constant on one side | Images written before the change decode as garbage; the dtype is asserted on decode. |

## 7. Checklist for a new codec path

1. Coerce to contiguous `<f4` once, at the boundary.
2. Check length before `frombuffer`; never infer `dim` from a truncated blob.
3. Return read-only views by default; make copies opt-in and explicit.
4. Keep the payload JSON canonical — sorted keys, tight separators, object-only.
5. Orient scores once in `score_matrix`; let every consumer be geometry-agnostic.
6. Accumulate in float64, select with `argpartition`, order the survivors stably.
7. Assert the dtype on decode so a layout change fails loudly.

## Where to read next

| Concern | Source |
| --- | --- |
| The codec | [`codec.py`](../../src/sable_vector/codec.py) |
| Kernels and top-k | [`distance.py`](../../src/sable_vector/distance.py) |
| Types and `Match` | [`types.py`](../../src/sable_vector/types.py) |
| The decision | [ADR-004](../decisions/adr-004-float32-codec-and-score-orientation.md) |
| Byte-image and dtype tests | [`tests/test_codec.py`](../../tests/test_codec.py), [`tests/test_distance.py`](../../tests/test_distance.py) |
| Why the codec defines recall ground truth | [benchmark report](../reports/benchmark-2026-11.md) |
