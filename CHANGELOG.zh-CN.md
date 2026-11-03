[English](./CHANGELOG.md) | 简体中文

# 变更日志

项目里所有值得记一笔的变更都在这里。格式沿用
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/)，从 1.0.0 起按
[语义化版本](https://semver.org/spec/v2.0.0.html)走。0.1.x 还是初始版本，API 没定稳。

## [未发布]

### 新增

- **PR 评审报告改成确定性的了。** 每个 PR 会收到一条固定评论，里面是变更地图、CI 门禁的实时
  状态，以及针对本仓库的几条较真提示：`mcp <2` 的装饰器 API、Dependabot 管不了的
  `sable @ git+...@v0.1.1` 直引用、recall@k 门禁、基准测试的 ground truth。顺带打上
  `area/`、`size/`、`risk/` 标签。它只提建议，不合并、不关闭、不批准，不调模型、不连外部服务，
  也没有运行成本。

### 变更

- **仓库迁到 [H.I.F](https://github.com/H-I-F) 组织下。** 规范地址现在是
  `https://github.com/H-I-F/sable-vector`，原来的 `MKA930/sable-vector` 还能访问，GitHub 会重定向
  迁移过的仓库。文档、打包元数据、容器构建里的引用都改了，包括那条固定依赖，现在写成
  `sable @ git+https://github.com/H-I-F/sable.git@v0.1.1`。

  代码、存储布局、版本号一个没动，已有集合照常打开，不用迁移。之前从旧 URL 装过 `sable` extra
  的环境靠重定向还能用，只是记录下来的源地址过时了，重装一次就刷新：

  ```sh
  pip install -e ".[sable,mcp]"
  ```

## [0.1.2] — 2026-11-03

一次文档发版。工程笔记现在中英双语覆盖到项目的每个角落，每篇都能从双语索引直达。键空间布局、
索引格式、公开 API、依赖固定、查询行为都没变：0.1.1 写出来的集合，用 0.1.2 打开不需要迁移。

`sable` 依赖仍固定在 `v0.1.1`。那个上游 tag 正是修复本项目的并发测试暴露出的快照读缺陷的版本，
而 0.1.2 没有任何行为变化——挪动固定值没有收益，反倒会损失可复现性。

### 新增

- **双语工程笔记索引** —— `docs/README.md` 与 `docs/README.zh-CN.md`，含板块表、源码地图和
  推荐阅读顺序。
- **五篇双语架构决策记录** —— 单一键空间、payload 提交记录、两种过滤策略、float32 编解码与
  分数方向、依赖固定。
- **FAQ、三篇工程实践指南与两篇实践配方**，均为双语 —— `docs/faq/vector-search-qa.md`、
  `docs/skills/` 那组（零拷贝编解码、持久写入、混合查询策略）、`docs/recipes/` 那组
  （混合检索、选择 `nprobe`）。
- **实测报告、发版公告与既有笔记的中文版** —— `docs/reports/benchmark-2026-11.md`、v0.1.0 与
  v0.1.1 公告、跨层案例复盘，以及 `CHANGELOG`、`CONTRIBUTING`、`SECURITY`、`PROGRESS` 的中文版。
- **两份 README 的双语入口** —— 文档表格与治理清单里每条都成对给出 English 和简体中文。

### 变更

- `pyproject.toml` 与 `sable_vector.__version__` 的版本从 `0.1.1` 提到 `0.1.2`。

## [0.1.1] — 2026-10-31

首个 tag 之后的跟进版本，补上 MCP 接入面，也把那次跨层排障记了下来。

### 新增

**Model Context Protocol**

- stdio MCP 服务端（`adapter/mcp_server.py`），在现有公开接口上暴露三个工具：
  `sable_vector_search`（向量查询，可选 payload 等值过滤，走 `PrefilterExecutor`）、
  `sable_vector_upsert`（一条持久记录：向量加 JSON 属性）、`sable_vector_describe`
  （记录数、向量维度、距离度量）。(#23)
- `sable-vector serve-mcp` 子命令（`cli.py`），注册成 console script。MCP 是异步的、Typer 是同步的，
  所以传输循环单独用一次 `asyncio.run` 驱动，没往命令里混。(#23)
- `mcp` 可选 extra 和 `typer` 核心依赖。extra 固定在 `<2`：服务端用的是
  `Server.list_tools` / `Server.call_tool` 装饰器 API，2.x 换成了构造器回调。(#23)
- 容器镜像、CI 矩阵、贡献者指南都装 `mcp` extra，适配器是被真正覆盖到的，不是假定它存在。(#23)

**文档**

- `docs/case-studies/cross-layer-bugfix.md` —— 跨层快照读 bug 的排查全过程，从症状到纯 `sable`
  复现，再到上游修复。两份 README 都加上了 *「Built on sable — with critical upstream bugfixes
  contributed」* 的说明。(#22)

### 修复

- `FlatIndex.materialise` 收的是 **storage** 而不是 engine，而且必须给 `dim`，所以两份 README 和
  v0.1.0 公告里的 quickstart 照抄会抛 `TypeError`。三处都改了。(#23)
- Mermaid 节点标签里带括号又没加引号，会把 GitHub 上的架构图弄坏；两份 README 都补了引号。

### 变更

- 版本 `0.1.0` → `0.1.1`。

## [0.1.0] — 2026-10-31

嵌入式混合向量引擎的第一个 tag。

### 新增

**核心引擎**

- 领域类型与 NumPy 距离核，float32 向量上的 L2 和 cosine（`types.py`、`distance.py`）。(#01)
- float32 ↔ 连续字节的零拷贝编解码，用于向量持久化（`codec.py`）。(#02)
- 物理命名空间适配器，把向量键空间映射到 `sable` LSM 存储引擎（`storage.py`）。(#03)
- WAL 级多键原子 upsert，让 payload 和向量一起落盘（`engine.py`）。(#04)

**索引与查询**

- 矩阵化 Flat 暴力检索，支持常驻索引和磁盘流式索引（`index/flat.py`）。(#05)
- 标量倒排前缀索引，用于元数据过滤（`index/scalar.py`）。(#06)
- 前置过滤、后置过滤两个执行器（`query/prefilter.py`、`query/postfilter.py`）。(#07、#08)
- IVF 聚类与倒排桶落盘，给近似检索用（`index/ivf_cluster.py`、`index/ivf_storage.py`）。(#09、#10)

**持久性与集成**

- 崩溃后从持久化状态重建内存结构（`recovery.py`）。(#11)
- 并发读的快照隔离，并在并行 reader 下验证引擎（`tests/test_concurrent.py`）。(#12)
- 离线 LangChain 适配器和一份能跑的 RAG demo（`adapter/langchain.py`、
  `examples/langchain_rag_demo.py`）。(#13)
- SIGKILL 混沌注入测试，检查突然终止后的恢复（`tests/test_chaos.py`）。(#14)

**打包、文档与 CI**

- 自校验容器镜像，以及双语（English / 简体中文）门户 README（`Dockerfile`、`.dockerignore`、
  `README.md`、`README.zh-CN.md`）。(#15)
- 1536 维 ANN 基准矩阵（`benchmarks/bench_vector.py`）。(#16)
- GitHub Actions CI：ruff + pytest，覆盖 Python 3.10 / 3.11（`.github/workflows/ci.yml`）。(#17)
- 多阶段 Docker 构建，CI 里的 `docker` job 会构建镜像并自测。(#18)

### 依赖

- 构建在 `sable` 存储引擎之上，固定到 `v0.1.1`（快照读一致性修复，顺带解决了并发读的两个回归测试）。

[未发布]: https://github.com/H-I-F/sable-vector/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/H-I-F/sable-vector/releases/tag/v0.1.2
[0.1.1]: https://github.com/H-I-F/sable-vector/releases/tag/v0.1.1
[0.1.0]: https://github.com/H-I-F/sable-vector/releases/tag/v0.1.0
