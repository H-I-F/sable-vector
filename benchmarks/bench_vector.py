"""ANN micro-benchmark for sable-vector at embedding scale.

The headline dimension is 1536, the width of the embedding models this engine
targets, but it is a parameter. The harness builds a *structured* corpus — points
drawn around a small set of cluster centres — because a uniform random cloud
makes every approximate index look perfect and every recall number meaningless.
Structured data is what exercises the IVF partition-pruning trade-off.

What it measures, per corpus size:

* **Ingest throughput** — vectors per second through the durable write path
  (``VectorEngine.upsert_many``), so the number reflects the WAL and the codec,
  not an in-memory append.
* **Exact recall baseline** — the resident ``FlatIndex`` and the streaming
  ``FlatIndex.search_streaming`` are both exhaustive, so their recall is 1.0 by
  construction; they exist to bound latency.
* **Approximate recall + latency** — ``IvfIndex`` at several ``nprobe`` values,
  with recall@k measured against the resident flat index as ground truth.

Run it with ``python -m benchmarks.bench_vector``. Every size, seed and query is
deterministic, so a rerun is directly comparable to a previous one.

Examples
--------
    python -m benchmarks.bench_vector
    python -m benchmarks.bench_vector --sizes 2000 10000 --dim 1536 --queries 300
    python -m benchmarks.bench_vector --metric l2 --no-ivf
"""

from __future__ import annotations

import argparse
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sable_vector.distance import DistanceMetric
from sable_vector.engine import VectorEngine
from sable_vector.index.flat import FlatIndex
from sable_vector.index.ivf_storage import IvfIndex
from sable_vector.types import VectorRecord

DEFAULT_DIM = 1536
DEFAULT_SIZES = (1_000, 5_000)
DEFAULT_KS = (1, 10)
DEFAULT_NPROBES = (1, 4, 16)
DEFAULT_QUERIES = 100
DEFAULT_WARMUP = 20
DEFAULT_CLUSTERS = 32
DEFAULT_PARTITIONS_CAP = 64
# The streaming search reads the whole corpus off disk per query, so its cost
# scales with n. Sampling it with fewer repetitions keeps the sweep honest
# without letting one O(n) path dominate the runtime at large corpora.
DEFAULT_STREAM_REPS = 40


@dataclass(slots=True)
class BenchmarkConfig:
    """Everything that varies a run."""

    dim: int = DEFAULT_DIM
    sizes: tuple[int, ...] = DEFAULT_SIZES
    ks: tuple[int, ...] = DEFAULT_KS
    nprobes: tuple[int, ...] = DEFAULT_NPROBES
    queries: int = DEFAULT_QUERIES
    warmup: int = DEFAULT_WARMUP
    clusters: int = DEFAULT_CLUSTERS
    metric: DistanceMetric = DistanceMetric.COSINE
    seed: int = 0
    partitions_cap: int = DEFAULT_PARTITIONS_CAP
    stream_reps: int = DEFAULT_STREAM_REPS
    run_ivf: bool = True

    @property
    def normalise(self) -> bool:
        """Cosine distances need unit vectors; L2 / dot do not."""
        return self.metric is DistanceMetric.COSINE


@dataclass(slots=True)
class Row:
    """One measured configuration: a size, an index kind, its latency and recall."""

    size: int
    index: str
    k: int
    nprobe: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    qps: float
    recall: float


@dataclass(slots=True)
class SizeReport:
    """Ingest timing plus the rows measured for one corpus size."""

    size: int
    dim: int
    ingest_seconds: float
    ingest_qps: float
    rows: list[Row] = field(default_factory=list)


def generate_corpus(
    n: int, dim: int, *, clusters: int, seed: int, normalise: bool
) -> np.ndarray:
    """Draw ``n`` unit-ish vectors clustered around ``clusters`` centres.

    Clustering makes the corpus behave like real embeddings, where neighbours are
    not uniformly scattered, so IVF's partition pruning has something to trade.
    """
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((clusters, dim)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    labels = rng.integers(0, clusters, size=n)
    noise = rng.standard_normal((n, dim)).astype(np.float32) * 0.25
    vectors = centres[labels] + noise
    if normalise:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.where(norms > 0.0, norms, 1.0)
    return vectors.astype(np.float32)


def _recall_at_k(ground: list[list[str]], found: list[list[str]], k: int) -> float:
    """Mean fraction of the ground-truth top-k present in the candidate set."""
    if not ground:
        return 1.0
    total = 0.0
    for truth, guess in zip(ground, found, strict=False):
        truth_set = set(truth[:k])
        if not truth_set:
            continue
        hit = len(truth_set & set(guess[:k]))
        total += hit / k
    return total / len(ground)


def _run_search(search_fn, queries, warmup: int, repeats: int):
    """Time ``repeats`` calls (after ``warmup``) and keep the returned matches."""
    for query in queries[:warmup]:
        search_fn(query)
    durations: list[float] = []
    results: list[object] = []
    for query in queries[:repeats]:
        start = time.perf_counter()
        result = search_fn(query)
        durations.append(time.perf_counter() - start)
        results.append(result)
    return np.asarray(durations, dtype=np.float64), results


def _partitions_for(size: int, cap: int) -> int:
    """Pick a partition count that scales with the corpus but stays bounded."""
    return max(8, min(cap, size // 250))


def benchmark_size(engine: VectorEngine, vectors: np.ndarray, cfg: BenchmarkConfig) -> SizeReport:
    """Ingest ``vectors`` and measure every index kind at every ``k``/``nprobe``."""
    size = vectors.shape[0]
    dim = vectors.shape[1]

    start = time.perf_counter()
    records = [VectorRecord(id=f"v{i:08d}", vector=vectors[i]) for i in range(size)]
    engine.upsert_many(records)
    ingest_seconds = time.perf_counter() - start
    ingest_qps = size / ingest_seconds

    flat = FlatIndex.materialise(engine.storage, metric=cfg.metric, dim=dim)

    rng = np.random.default_rng(cfg.seed + size)
    chosen = rng.choice(size, size=min(cfg.queries, size), replace=False)
    query_vectors = [vectors[int(i)] for i in chosen]

    # Ground truth from the exhaustive resident index.
    ground_truth: list[list[str]] = []
    for q in query_vectors:
        ground_truth.append([match.id for match in flat.search(q, k=max(cfg.ks))])

    report = SizeReport(
        size=size, dim=dim, ingest_seconds=ingest_seconds, ingest_qps=ingest_qps
    )

    for k in cfg.ks:
        # Resident exhaustive search: the latency floor and the recall reference.
        durations, results = _run_search(
            lambda q, k=k: flat.search(q, k=k), query_vectors, cfg.warmup, cfg.queries
        )
        found = [[m.id for m in r] for r in results]
        report.rows.append(
            Row(
                size=size,
                index="flat",
                k=k,
                nprobe=0,
                p50_ms=float(np.percentile(durations, 50) * 1e3),
                p95_ms=float(np.percentile(durations, 95) * 1e3),
                mean_ms=float(durations.mean() * 1e3),
                qps=float(1.0 / durations.mean()),
                recall=1.0,
            )
        )

        # Streaming search: same maths, block-at-a-time I/O profile. Its cost
        # grows with the corpus, so it is sampled with fewer repetitions than the
        # resident and IVF paths.
        durations, results = _run_search(
            lambda q, k=k: flat.search_streaming(engine.storage, q, k=k),
            query_vectors,
            min(cfg.warmup, 10),
            min(cfg.queries, cfg.stream_reps),
        )
        report.rows.append(
            Row(
                size=size,
                index="flat-stream",
                k=k,
                nprobe=0,
                p50_ms=float(np.percentile(durations, 50) * 1e3),
                p95_ms=float(np.percentile(durations, 95) * 1e3),
                mean_ms=float(durations.mean() * 1e3),
                qps=float(1.0 / durations.mean()),
                recall=1.0,
            )
        )

    if cfg.run_ivf:
        partitions = _partitions_for(size, cfg.partitions_cap)
        ivf = IvfIndex.train(
            engine.storage,
            metric=cfg.metric,
            dim=dim,
            partitions=partitions,
            seed=cfg.seed,
        )
        for nprobe in cfg.nprobes:
            if nprobe > partitions:
                continue
            for k in cfg.ks:
                durations, results = _run_search(
                    lambda q, k=k, nprobe=nprobe: ivf.search(
                        q, k=k, nprobe=nprobe
                    ).matches,
                    query_vectors,
                    cfg.warmup,
                    cfg.queries,
                )
                found = [[m.id for m in r] for r in results]
                report.rows.append(
                    Row(
                        size=size,
                        index="ivf",
                        k=k,
                        nprobe=nprobe,
                        p50_ms=float(np.percentile(durations, 50) * 1e3),
                        p95_ms=float(np.percentile(durations, 95) * 1e3),
                        mean_ms=float(durations.mean() * 1e3),
                        qps=float(1.0 / durations.mean()),
                        recall=_recall_at_k(ground_truth, found, k),
                    )
                )

    return report


def run_benchmark(cfg: BenchmarkConfig) -> list[SizeReport]:
    """Run every configured size in an isolated, discarded storage directory."""
    reports: list[SizeReport] = []
    with tempfile.TemporaryDirectory(prefix="sable-bench-") as home:
        for size in cfg.sizes:
            vectors = generate_corpus(
                size,
                cfg.dim,
                clusters=cfg.clusters,
                seed=cfg.seed,
                normalise=cfg.normalise,
            )
            engine = VectorEngine.open(
                Path(home) / f"size-{size}", metric=cfg.metric, dim=cfg.dim
            )
            try:
                reports.append(benchmark_size(engine, vectors, cfg))
            finally:
                engine.close()
    return reports


def _print_report(cfg: BenchmarkConfig, reports: list[SizeReport]) -> None:
    """Render the results as a Markdown table to stdout."""
    print(
        f"# sable-vector ANN benchmark — dim={cfg.dim} "
        f"metric={cfg.metric.value} seed={cfg.seed} queries={cfg.queries}"
    )
    print()
    for report in reports:
        print(f"## corpus size = {report.size:,}  (dim {report.dim})")
        print(
            f"ingest: {report.ingest_qps:,.0f} vec/s "
            f"over {report.ingest_seconds:.2f}s"
        )
        print()
        print(
            "| index | k | nprobe | p50 (ms) | p95 (ms) | mean (ms) | "
            "qps | recall@k |"
        )
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in report.rows:
            nprobe = "—" if row.index != "ivf" else str(row.nprobe)
            recall = "1.00" if row.index != "ivf" else f"{row.recall:.3f}"
            print(
                f"| {row.index} | {row.k} | {nprobe} | "
                f"{row.p50_ms:.3f} | {row.p95_ms:.3f} | {row.mean_ms:.3f} | "
                f"{row.qps:,.0f} | {recall} |"
            )
        print()


def _parse_args(argv: list[str] | None = None) -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        description="Benchmark sable-vector at embedding scale (default 1536-dim)."
    )
    parser.add_argument(
        "--dim", type=int, default=DEFAULT_DIM, help="embedding width (default 1536)"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_SIZES),
        help="corpus sizes to sweep",
    )
    parser.add_argument(
        "--ks", type=int, nargs="+", default=list(DEFAULT_KS), help="top-k values"
    )
    parser.add_argument(
        "--nprobes",
        type=int,
        nargs="+",
        default=list(DEFAULT_NPROBES),
        help="IVF nprobe values",
    )
    parser.add_argument("--queries", type=int, default=DEFAULT_QUERIES)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument(
        "--stream-reps",
        type=int,
        default=DEFAULT_STREAM_REPS,
        help="repetitions for the disk-bound streaming search (kept low; it is O(n))",
    )
    parser.add_argument("--clusters", type=int, default=DEFAULT_CLUSTERS)
    parser.add_argument(
        "--metric",
        choices=[m.value for m in DistanceMetric],
        default=DistanceMetric.COSINE.value,
        help="distance metric",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-ivf", action="store_true", help="skip the IVF sweep"
    )
    args = parser.parse_args(argv)
    return BenchmarkConfig(
        dim=args.dim,
        sizes=tuple(args.sizes),
        ks=tuple(args.ks),
        nprobes=tuple(args.nprobes),
        queries=args.queries,
        warmup=args.warmup,
        clusters=args.clusters,
        metric=DistanceMetric(args.metric),
        seed=args.seed,
        stream_reps=args.stream_reps,
        run_ivf=not args.no_ivf,
    )


def main(argv: list[str] | None = None) -> None:
    cfg = _parse_args(argv)
    reports = run_benchmark(cfg)
    _print_report(cfg, reports)


if __name__ == "__main__":
    main()
