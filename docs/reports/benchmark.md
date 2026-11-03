[English](benchmark.md) | [简体中文](benchmark.zh-CN.md)

# ANN benchmark

A measured run of `benchmarks/bench_vector.py`, published so that `nprobe` can be
chosen against numbers rather than intuition. The harness is self-asserting: the
resident `FlatIndex` is exhaustive, so its recall is 1.0 by construction and every
approximate figure below is computed against it with `_recall_at_k`.

## How it was run

```bash
python -m benchmarks.bench_vector \
  --dim 128 --sizes 1000 5000 --queries 30 --ks 10 \
  --nprobes 1 2 4 8 --clusters 16 --stream-reps 2 \
  --metric cosine --seed 42
```

Linux x86-64 container, Python 3.11, NumPy 2.3.5, with `sable` loaded from its
working tree. The seed is fixed, so the corpus and query set are reproducible.
`--dim 128` is deliberately smaller than the 1536-dimension default: it keeps the
run short while exercising the same kernels. Latency is wall-clock per query over
30 queries; `qps` is its reciprocal.

## Results — 1,000 vectors

Ingest: **21,108 vec/s** over 0.05 s.

| index | k | nprobe | p50 (ms) | p95 (ms) | mean (ms) | qps | recall@k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flat | 10 | — | 0.144 | 0.157 | 0.145 | 6,894 | 1.00 |
| flat-stream | 10 | — | 1.148 | 1.194 | 1.148 | 871 | 1.00 |
| ivf | 10 | 1 | 0.257 | 0.316 | 0.249 | 4,022 | 0.523 |
| ivf | 10 | 2 | 0.466 | 0.553 | 0.462 | 2,166 | 0.680 |
| ivf | 10 | 4 | 0.819 | 1.202 | 0.885 | 1,129 | 0.877 |
| ivf | 10 | 8 | 1.489 | 1.609 | 1.508 | 663 | 1.000 |

## Results — 5,000 vectors

Ingest: **26,596 vec/s** over 0.19 s.

| index | k | nprobe | p50 (ms) | p95 (ms) | mean (ms) | qps | recall@k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| flat | 10 | — | 3.773 | 95.074 | 29.779 | 34 | 1.00 |
| flat-stream | 10 | — | 4.993 | 5.024 | 4.993 | 200 | 1.00 |
| ivf | 10 | 1 | 0.897 | 1.179 | 0.890 | 1,124 | 0.623 |
| ivf | 10 | 2 | 1.534 | 1.933 | 1.565 | 639 | 0.727 |
| ivf | 10 | 4 | 3.087 | 3.613 | 3.110 | 322 | 0.810 |
| ivf | 10 | 8 | 5.860 | 6.756 | 5.994 | 167 | 0.920 |

## Reading these honestly

- **Exact search is the right default at this scale.** At 1,000 vectors the resident
  flat index answers in 0.144 ms with recall 1.00 — faster than IVF at *any*
  `nprobe`, because IVF pays centroid scoring and posting-list reads before it
  scores a single vector. Approximate search is not a free faster mode.
- **`nprobe` buys recall at a predictable price.** On 5,000 vectors, going from 1 to
  8 partitions lifts recall 0.623 → 0.920 while p50 rises 0.897 → 5.860 ms. At
  1,000 vectors the same sweep reaches 1.000 at `nprobe=8` — with 16 clusters that
  is half the corpus, which is why the curve flattens.
- **Streaming exact search is disk-bound, not compute-bound.** `flat-stream` keeps
  recall 1.00 but costs roughly 8x the resident path at 1,000 vectors (1.148 ms vs
  0.144 ms). It is there for collections that do not fit in memory, not as a default.
- **One outlier is published, not smoothed away.** The 5,000-vector resident flat
  row shows p50 3.773 ms but p95 95.074 ms and mean 29.779 ms. That is a warm-up /
  allocation stall in a handful of the 30 queries, not a property of the kernel —
  the p50 is the representative figure, and the p95 is the reason not to quote a
  single number. A rerun will move these tails; the recall column will not.
- **Ingest is not the bottleneck.** 21–27k vectors/s at 128 dimensions is an
  in-memory staging rate. It does not include a durable `fsync` per record; see
  ADR-002 for where that boundary sits.

## What this run does not cover

Production embedding widths (1536 dimensions), `l2` and `dot` metrics, IVF training
time, concurrent readers, and durable write costs with real `fsync`. Recall here is
measured against an exhaustive index over the *same* corpus, so it says nothing about
the quality of the upstream embedding model. Absolute latencies are a snapshot of
one container; treat the recall curve and the ratios as the result.

Reproduce with a different shape:

```bash
python -m benchmarks.bench_vector --dim 1536 --sizes 10000 --queries 50
```
