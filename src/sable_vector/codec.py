"""Byte-level codec for embeddings and payloads.

Two representations cross the storage boundary, and they have opposite
requirements:

* **Vectors are fixed-width binary.** A 1536-dimension embedding is 6144 bytes
  of little-endian float32 with no framing, no header and no delimiter, because
  a header would have to be parsed on every one of the 50k rows a scan touches
  and a delimiter would have to be escaped. Continuity is what makes
  :func:`decode_vector` a ``np.frombuffer`` call — a view onto the bytes the
  storage layer already returned, not a second copy of them.
* **Payloads are JSON.** They are read once per surviving hit rather than once
  per candidate, so they pay for the parse; in exchange they stay inspectable
  from a shell, which is worth more than the bytes saved by a binary format.

Endianness is pinned to little-endian and dtype to float32 at both ends. A
vector written on one machine therefore decodes on any other without a header
saying which layout it was written in — the layout is part of the format, not
part of the data.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from sable_vector.distance import VECTOR_DTYPE, as_matrix, as_query

__all__ = [
    "ITEM_SIZE",
    "decode_matrix",
    "decode_payload",
    "decode_vector",
    "encode_matrix",
    "encode_payload",
    "encode_vector",
    "vector_bytes",
]

ITEM_SIZE: int = VECTOR_DTYPE.itemsize


def vector_bytes(dim: int) -> int:
    """Exact on-disk size of one ``dim``-dimension vector."""
    return dim * ITEM_SIZE


def encode_vector(vector: np.ndarray) -> bytes:
    """Serialise one embedding to little-endian float32 bytes.

    Args:
        vector: One-dimensional embedding of any float dtype.

    Returns:
        ``4 * dim`` bytes, the vector's memory in canonical order.
    """
    return as_query(vector).tobytes()


def decode_vector(
    blob: bytes, *, dim: int | None = None, writeable: bool = False
) -> np.ndarray:
    """Reinterpret ``blob`` as a float32 vector without copying it.

    Args:
        blob: Bytes produced by :func:`encode_vector` (or any little-endian
            float32 image of the same shape).
        dim: Expected width; a mismatch is an error rather than a reshape.
        writeable: Return a private copy instead of a read-only view. Needed
            only by callers that mutate the result, such as residual arithmetic.

    Returns:
        A one-dimensional array. Read-only unless ``writeable`` — the default
        view aliases ``blob``, so writing through it would mutate buffers the
        storage layer considers immutable.

    Raises:
        ValueError: The length is not a whole number of float32 items, or does
            not match ``dim``.
    """
    _check_length(blob, dim, single_row=True)
    view = np.frombuffer(blob, dtype=VECTOR_DTYPE)
    return view.copy() if writeable else view


def encode_matrix(matrix: np.ndarray) -> bytes:
    """Serialise an ``(n, d)`` block of embeddings as one contiguous image."""
    return as_matrix(matrix).tobytes()


def decode_matrix(blob: bytes, *, dim: int, writeable: bool = False) -> np.ndarray:
    """Reinterpret ``blob`` as an ``(n, d)`` block of float32 vectors.

    Args:
        blob: Concatenated little-endian float32 rows.
        dim: Row width; ``n`` is derived from the length.
        writeable: Return a private copy instead of a read-only view.

    Returns:
        ``(len(blob) // (4 * dim), dim)`` array.

    Raises:
        ValueError: The length is not a whole number of ``dim``-wide rows.
    """
    _check_length(blob, dim, single_row=False)
    view = np.frombuffer(blob, dtype=VECTOR_DTYPE).reshape(-1, dim)
    return view.copy() if writeable else view


def encode_payload(payload: Mapping[str, Any]) -> bytes:
    """Serialise a payload to canonical JSON.

    Keys are sorted and separators tightened so that equal payloads produce
    equal bytes — which is what lets a payload be compared, hashed or used as
    part of an index key without re-parsing it.
    """
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def decode_payload(blob: bytes) -> dict[str, Any]:
    """Parse a payload written by :func:`encode_payload`.

    Raises:
        ValueError: The bytes are not a JSON object.
    """
    document = json.loads(blob.decode("utf-8"))
    if not isinstance(document, dict):
        raise ValueError("payload must decode to a JSON object")
    return document


def _check_length(blob: bytes, dim: int | None, *, single_row: bool) -> None:
    """Validate a float32 image before it is reinterpreted.

    Args:
        blob: Candidate image.
        dim: Expected width, or ``None`` when any width is acceptable.
        single_row: Require exactly ``dim`` items rather than a multiple of
            them. A vector is one row; a block is however many rows fit.

    Raises:
        ValueError: Length is not a whole number of items, or not ``dim`` wide.
    """
    if len(blob) % ITEM_SIZE != 0:
        raise ValueError(
            f"vector blob of {len(blob)} bytes is not a multiple of {ITEM_SIZE}"
        )
    if dim is None:
        return
    if dim <= 0:
        raise ValueError(f"dimension must be positive, got {dim}")
    items = len(blob) // ITEM_SIZE
    if single_row:
        if items != dim:
            raise ValueError(f"blob holds {items} float32 items, expected exactly {dim}")
    elif items % dim != 0:
        raise ValueError(
            f"blob holds {items} float32 items, which is not a whole number of {dim}"
        )
