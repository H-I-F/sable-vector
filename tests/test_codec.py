"""Byte-layout contracts for the embedding codec.

The point of a fixed-width, headerless, little-endian float32 image is that it
can be reinterpreted rather than parsed. These tests pin the three properties
that make that true — exact byte layout, true zero-copy decoding, and strict
rejection of anything that is not a whole number of rows — because each of them
fails silently if it fails at all: an accidentally big-endian image decodes to
garbage that still *looks* like an array.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from sable_vector.codec import (
    ITEM_SIZE,
    decode_matrix,
    decode_payload,
    decode_vector,
    encode_matrix,
    encode_payload,
    encode_vector,
    vector_bytes,
)
from sable_vector.distance import VECTOR_DTYPE

DIM = 1536


def sample(dim: int = 32, *, seed: int = 5) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=dim).astype(VECTOR_DTYPE)


# -- exact wire format ----------------------------------------------------


def test_encoded_bytes_match_struct_packing() -> None:
    vector = sample(16)
    assert encode_vector(vector) == struct.pack(f"<{vector.size}f", *vector.tolist())


def test_float32_one_is_little_endian_on_the_wire() -> None:
    """1.0f is 0x3F800000; big-endian would put those bytes in reverse."""
    assert encode_vector(np.array([1.0], dtype=VECTOR_DTYPE)) == b"\x00\x00\x80\x3f"


def test_encoded_length_is_four_bytes_per_dimension() -> None:
    assert len(encode_vector(sample(DIM))) == vector_bytes(DIM) == DIM * ITEM_SIZE == 6144


def test_a_1536_dimension_vector_round_trips_bit_exactly() -> None:
    vector = sample(DIM)
    np.testing.assert_array_equal(decode_vector(encode_vector(vector), dim=DIM), vector)


def test_float64_input_is_narrowed_to_float32() -> None:
    wide = np.array([1.0, 1e-9, 1.0 / 3.0], dtype=np.float64)
    decoded = decode_vector(encode_vector(wide), dim=3)
    assert decoded.dtype == VECTOR_DTYPE
    np.testing.assert_array_equal(decoded, wide.astype(VECTOR_DTYPE))


def test_non_contiguous_input_is_encoded_by_value() -> None:
    source = np.arange(8, dtype=VECTOR_DTYPE).reshape(4, 2)
    stride_view = source[::2, ::-1]
    np.testing.assert_array_equal(decode_vector(encode_vector(stride_view[0]), dim=2), [1.0, 0.0])


# -- zero-copy decoding ---------------------------------------------------


def test_decode_is_a_view_onto_the_source_bytes() -> None:
    blob = encode_vector(sample(64))
    decoded = decode_vector(blob, dim=64)
    assert isinstance(decoded.base, bytes)
    assert decoded.flags["WRITEABLE"] is False
    assert np.shares_memory(decoded, np.frombuffer(blob, dtype=np.uint8))


def test_decode_matrix_is_a_view_without_a_row_copy() -> None:
    """A reshaped view keeps aliasing its buffer; only the reshape is new."""
    blob = encode_matrix(np.arange(24, dtype=VECTOR_DTYPE).reshape(3, 8))
    block = decode_matrix(blob, dim=8)
    assert block.shape == (3, 8)
    assert block.flags["WRITEABLE"] is False
    assert np.shares_memory(block, np.frombuffer(blob, dtype=np.uint8))
    np.testing.assert_array_equal(block[2], np.arange(16, 24, dtype=VECTOR_DTYPE))


def test_writeable_decode_returns_an_independent_copy() -> None:
    blob = encode_vector(sample(8))
    decoded = decode_vector(blob, dim=8, writeable=True)
    assert decoded.flags["WRITEABLE"] is True
    decoded[0] = 123.0
    # The source image is untouched: the copy is real, not a shared buffer.
    assert decode_vector(blob, dim=8)[0] != 123.0


def test_decoding_the_same_blob_twice_aliases_one_buffer() -> None:
    blob = encode_vector(sample(8))
    first = decode_vector(blob, dim=8)
    second = decode_vector(blob, dim=8)
    assert np.shares_memory(first, second)


# -- round trips ----------------------------------------------------------


def test_matrix_round_trip_preserves_row_order() -> None:
    block = np.arange(40, dtype=VECTOR_DTYPE).reshape(5, 8)
    np.testing.assert_array_equal(decode_matrix(encode_matrix(block), dim=8), block)


def test_matrix_round_trip_handles_a_single_row() -> None:
    block = np.arange(8, dtype=VECTOR_DTYPE).reshape(1, 8)
    assert decode_matrix(encode_matrix(block), dim=8).shape == (1, 8)


def test_empty_matrix_round_trip() -> None:
    empty = np.empty((0, 8), dtype=VECTOR_DTYPE)
    assert encode_matrix(empty) == b""
    assert decode_matrix(b"", dim=8).shape == (0, 8)


# -- rejection ------------------------------------------------------------


def test_a_truncated_blob_is_rejected() -> None:
    blob = encode_vector(sample(16))[:-1]
    with pytest.raises(ValueError, match="not a multiple of"):
        decode_vector(blob)


def test_a_dimension_mismatch_is_rejected() -> None:
    blob = encode_vector(sample(16))
    with pytest.raises(ValueError, match="expected exactly 32"):
        decode_vector(blob, dim=32)


def test_a_partial_final_matrix_row_is_rejected() -> None:
    blob = encode_vector(sample(16)) + encode_vector(sample(9))
    with pytest.raises(ValueError, match="not a whole number of 16"):
        decode_matrix(blob, dim=16)


def test_a_non_positive_dimension_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        decode_vector(b"\x00" * 8, dim=0)


# -- payloads -------------------------------------------------------------


def test_payload_round_trip_preserves_types() -> None:
    payload = {"text": "héllo", "tenant": "acme", "score": 1.5, "tags": ["a", "b"], "n": 3}
    assert decode_payload(encode_payload(payload)) == payload


def test_payload_encoding_is_canonical_across_key_order() -> None:
    first = {"a": 1, "b": 2}
    second = {"b": 2, "a": 1}
    assert encode_payload(first) == encode_payload(second)


def test_payload_encoding_is_compact() -> None:
    assert encode_payload({"a": 1}) == b'{"a":1}'


def test_non_object_json_is_rejected() -> None:
    with pytest.raises(ValueError, match="must decode to a JSON object"):
        decode_payload(b"[1, 2, 3]")


def test_empty_payload_round_trip() -> None:
    assert decode_payload(encode_payload({})) == {}
