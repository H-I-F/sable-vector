# sable-vector — 单步演进状态机 / Single-step evolution ledger

One box per pull request. A box is ticked only after that PR has been merged on
`main`; the tick itself is part of the very commit that PR carries, so the
ledger cannot drift ahead of the repository it describes.

每个复选框对应一个 PR。只有该 PR 在 `main` 上合并后才打勾，且打勾动作本身就在
该 PR 的提交里，因此这份台账不可能跑到它所描述的仓库前面去。

| PR | Branch | Deliverable | Acceptance |
| --- | --- | --- | --- |
| 01 | `feat/types-distance` | `types.py`, `distance.py`, `tests/test_distance.py` | `pytest tests/test_distance.py -v` |
| 02 | `feat/codec-bytes` | `codec.py`, `tests/test_codec.py` | `pytest tests/test_codec.py -v` |
| 03 | `feat/storage-adapter` | `storage.py`, `tests/test_storage.py` | `pytest tests/test_storage.py -v` |
| 04 | `feat/atomic-upsert` | `engine.py` write path, `tests/test_upsert.py` | `pytest tests/test_upsert.py -v` |
| 05 | `feat/flat-index` | `index/flat.py`, `tests/test_flat.py` | `pytest tests/test_flat.py -v` |
| 06 | `feat/scalar-index` | `index/scalar.py`, `tests/test_scalar.py` | `pytest tests/test_scalar.py -v` |
| 07 | `feat/pre-filtering` | `query/prefilter.py`, `tests/test_prefilter.py` | `pytest tests/test_prefilter.py -v` |
| 08 | `feat/post-filtering` | `query/postfilter.py`, `tests/test_postfilter.py` | `pytest tests/test_postfilter.py -v` |
| 09 | `feat/ivf-clustering` | `index/ivf_cluster.py`, `tests/test_ivf_cluster.py` | `pytest tests/test_ivf_cluster.py -v` |
| 10 | `feat/ivf-posting-lists` | `index/ivf_storage.py`, `tests/test_ivf_scan.py` | `pytest tests/test_ivf_scan.py -v` |
| 11 | `feat/crash-rebuild` | `recovery.py`, `tests/test_recovery.py` | `pytest tests/test_recovery.py -v` |
| 12 | `feat/concurrent-read` | `tests/test_concurrent.py` | `pytest tests/test_concurrent.py -v` |
| 13 | `feat/langchain-adapter` | `adapter/langchain.py`, `examples/langchain_rag_demo.py` | `python examples/langchain_rag_demo.py` |
| 14 | `test/chaos-injection` | `tests/test_chaos.py` | `pytest tests/test_chaos.py -v` |
| 15 | `docs/portal-readme` | `Dockerfile`, `.dockerignore`, `README.md`, `README.zh-CN.md` | `docker build -t sable-vector . && docker run --rm sable-vector pytest -q` |
| 16 | `bench/ann-benchmarks` | `benchmarks/bench_vector.py` | `python -m benchmarks.bench_vector` |

## Checklist

- [x] PR 01: 领域类型与 NumPy 距离计算 / Domain types and NumPy distance kernels
- [x] PR 02: float32 连续二进制零拷贝编解码 / Zero-copy float32 codec
- [x] PR 03: sable 物理命名空间适配器 / Physical namespace adapter over sable
- [x] PR 04: WAL 级多键原子 Upsert / WAL-level atomic multi-key upsert
- [x] PR 05: NumPy 矩阵化 Flat 爆搜 / Matrix-vectorised Flat brute force
- [x] PR 06: 标量倒排前缀索引 / Scalar inverted prefix index
- [ ] PR 07: 前置过滤执行器 / Pre-filtering executor
- [ ] PR 08: 后置过滤执行器 / Post-filtering executor
- [ ] PR 09: IVF 质心聚类状态机 / IVF centroid clustering state machine
- [ ] PR 10: 倒排桶落盘与分区修剪 / Disk-backed posting lists with partition pruning
- [ ] PR 11: 崩溃恢复与索引重建 / Crash recovery and index rebuild
- [ ] PR 12: 无锁快照并发读 / Lock-free snapshot concurrent reads
- [ ] PR 13: VectorStore 协议适配器 / VectorStore protocol adapter
- [ ] PR 14: SIGKILL 混沌故障注入 / SIGKILL chaos injection
- [ ] PR 15: 容器化与双语架构 README / Container image and bilingual README
- [ ] PR 16: 1536 维基准矩阵 / 1536-dimension benchmark matrix
