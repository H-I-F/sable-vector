[English](adr-001-single-keyspace.md) | [简体中文](adr-001-single-keyspace.zh-CN.md)

# ADR-001: One key space, separated by prefix

**Status:** Accepted — implemented in `src/sable_vector/storage.py`.

## Context

A collection has more than vectors to persist: the JSON payload, one inverted entry
per indexed field value, IVF centroids, IVF posting-list members, and a small amount
of metadata. Giving each of those its own store would give the project several WALs,
several manifests and several recovery orderings. Worse, the parts of a single record
would then live in different trees, and the claim that "a vector, its payload and its
inverted entries commit as one unit" would have nothing to enforce it.

`sable` exposes exactly the primitive that avoids this: an ordered key space with
atomic writes and range scans. The question was only how to lay the parts out.

## Decision

Everything lives in **one `sable` LSM tree**, separated by byte prefixes that are
mutually disjoint by construction:

| Prefix | Holds |
| --- | --- |
| `v:` | raw little-endian float32 image |
| `p:` | canonical JSON payload — the commit record |
| `idx:` | scalar inverted entry, `idx:<field>:<value>:<id>`, empty value |
| `c:` | IVF centroid |
| `b:` | IVF posting-list member |
| `m:` | metadata |

Three rules make this safe, all centralised in `storage.py` so no caller has to
reimplement them:

- `escape_segment()` escapes a segment once. Without it `idx:a:b:c` is ambiguous
  about where the field ends and the value begins.
- `prefix_upper_bound()` returns the smallest key greater than every key under a
  prefix, turning a predicate into a half-open scan.
- The prefixes are disjoint: `v:` is not a prefix of `p:`, so a namespace scan can
  never leak another namespace's keys.

## Consequences

One WAL, one manifest, one crash-recovery story. A metadata predicate becomes a
contiguous prefix scan whose cost is proportional to the answer rather than to the
corpus — this is what makes `ScalarIndex.matching_ids` cheap.

The cost is coupling. Every derived structure shares the tree's lifetime, so
`reclaim()` and `rebuild_ivf()` must clear whole namespaces rather than drop files.
Adding a prefix is a contract change: it must be disjoint from all existing ones and
escaped through the same helper. The layout is documented in the module docstring and
enforced by the prefix constants, not by convention.
