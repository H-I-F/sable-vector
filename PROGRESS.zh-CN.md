[English](./PROGRESS.md) | 简体中文

# sable-vector — 单步演进台账

一个 PR 一个复选框。只有这个 PR 在 `main` 上合并了才打勾，而且打勾动作本身就在该 PR 带的提交里，
所以这份台账不可能跑到它所描述的仓库前面去。

| PR | 分支 | 交付物 | 验收方式 |
| --- | --- | --- | --- |
| 01 | `feat/types-distance` | `types.py`、`distance.py`、`tests/test_distance.py` | `pytest tests/test_distance.py -v` |
| 02 | `feat/codec-bytes` | `codec.py`、`tests/test_codec.py` | `pytest tests/test_codec.py -v` |
| 03 | `feat/storage-adapter` | `storage.py`、`tests/test_storage.py` | `pytest tests/test_storage.py -v` |
| 04 | `feat/atomic-upsert` | `engine.py` 写路径、`tests/test_upsert.py` | `pytest tests/test_upsert.py -v` |
| 05 | `feat/flat-index` | `index/flat.py`、`tests/test_flat.py` | `pytest tests/test_flat.py -v` |
| 06 | `feat/scalar-index` | `index/scalar.py`、`tests/test_scalar.py` | `pytest tests/test_scalar.py -v` |
| 07 | `feat/pre-filtering` | `query/prefilter.py`、`tests/test_prefilter.py` | `pytest tests/test_prefilter.py -v` |
| 08 | `feat/post-filtering` | `query/postfilter.py`、`tests/test_postfilter.py` | `pytest tests/test_postfilter.py -v` |
| 09 | `feat/ivf-clustering` | `index/ivf_cluster.py`、`tests/test_ivf_cluster.py` | `pytest tests/test_ivf_cluster.py -v` |
| 10 | `feat/ivf-posting-lists` | `index/ivf_storage.py`、`tests/test_ivf_scan.py` | `pytest tests/test_ivf_scan.py -v` |
| 11 | `feat/crash-rebuild` | `recovery.py`、`tests/test_recovery.py` | `pytest tests/test_recovery.py -v` |
| 12 | `feat/concurrent-read` | `tests/test_concurrent.py` | `pytest tests/test_concurrent.py -v` |
| 13 | `feat/langchain-adapter` | `adapter/langchain.py`、`examples/langchain_rag_demo.py` | `python examples/langchain_rag_demo.py` |
| 14 | `test/chaos-injection` | `tests/test_chaos.py` | `pytest tests/test_chaos.py -v` |
| 15 | `docs/portal-readme` | `Dockerfile`、`.dockerignore`、`README.md`、`README.zh-CN.md` | `docker build -t sable-vector . && docker run --rm sable-vector pytest -q` |
| 16 | `bench/ann-benchmarks` | `benchmarks/bench_vector.py` | `python -m benchmarks.bench_vector` |
| 17 | `ci(actions)` | `.github/workflows/ci.yml`，ruff + pytest，Py 3.10/3.11 | `ruff check . && pytest -q` |
| 18 | `build(docker)` | 多阶段 `Dockerfile`、`docker` CI job | `docker build -t sable-vector . && docker run --rm sable-vector pytest -q`（云端） |
| 19 | `docs(governance)` | `CONTRIBUTING.md`、`SECURITY.md`、`CHANGELOG.md` | 文件齐备，README 里链接接好 |
| 20 | `docs(portal)` | README 重构：CI 徽章、分层图、用法指南、文档导航 | Markdown 语法与表格校验 |
| 21 | `chore(release)` | `docs/RELEASE_NOTES_v0.1.0.md` | `gh release create v0.1.0 --notes-file docs/RELEASE_NOTES_v0.1.0.md` |
| 22 | `docs(case-study)` | `docs/case-studies/cross-layer-bugfix.md`，README 高亮 | 链到 `H-I-F/sable`，两个仓库互相引用 |
| 23 | `feat(mcp)` | `adapter/mcp_server.py`、`cli.py`（`serve-mcp`） | `sable-vector serve-mcp --help` + stdio 握手 |
| 24 | `chore(release)` | 版本 `0.1.1`、`docs/RELEASE_NOTES_v0.1.1.md`、CHANGELOG | `gh release create v0.1.1 --notes-file docs/RELEASE_NOTES_v0.1.1.md` |

## 清单

- [x] PR 01：领域类型与 NumPy 距离计算
- [x] PR 02：float32 连续二进制零拷贝编解码
- [x] PR 03：sable 物理命名空间适配器
- [x] PR 04：WAL 级多键原子 Upsert
- [x] PR 05：NumPy 矩阵化 Flat 爆搜
- [x] PR 06：标量倒排前缀索引
- [x] PR 07：前置过滤执行器
- [x] PR 08：后置过滤执行器
- [x] PR 09：IVF 质心聚类状态机
- [x] PR 10：倒排桶落盘与分区修剪
- [x] PR 11：崩溃恢复与索引重建
- [x] PR 12：无锁快照并发读
- [x] PR 13：VectorStore 协议适配器
- [x] PR 14：SIGKILL 混沌故障注入
- [x] PR 15：容器化与双语架构 README
- [x] PR 16：1536 维基准矩阵
- [x] PR 17：自动化 CI 矩阵（ruff + pytest，Py 3.10/3.11）
- [x] PR 18：多阶段 Docker 镜像
- [x] PR 19：治理文档（CONTRIBUTING / SECURITY / CHANGELOG）
- [x] PR 20：README 最终重构与徽章
- [x] PR 21：v0.1.0 发版公告与 Release
- [x] PR 22：跨层 Bug 排查案例
- [x] PR 23：MCP 服务端与 CLI 子命令
- [x] PR 24：v0.1.1 发版（含 MCP）与 Release
