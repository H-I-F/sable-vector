[English](README.md) | [简体中文](README.zh-CN.md)

# sable-vector — 工程笔记索引

> [项目主文档](../README.md)写给用这个库的人。这个目录写给想搞清楚**为什么**一条记录要这样排布、
> 一种查询策略到底要付多少代价、哪些约束是刻意选的。下面每条结论都指向 `src/sable_vector/` 里
> 真实存在的符号。

## 这里都有什么

| 板块 | 面向 | 形式 |
| --- | --- | --- |
| [发版公告](#发版公告) | 版本之间升级 | 每个 tag 一篇，中英双语 |
| [案例研究](#案例研究) | 复盘、设计评审 | 带复现过程的叙事 |
| [架构决策记录](#架构决策记录) | 要改键空间或写入契约的人 | ADR-001 … ADR-005，中英双语 |
| [工程实践 Skills](#工程实践-skills底层实现指南) | 要实现或移植这个引擎的人 | 带代码与表格的叙事指南 |
| [硬核问答 FAQ](#硬核问答-faq) | 深度追问、设计评审、「干嘛不直接……」的追问 | 中英双语问答，配 Mermaid 图 |
| [实践配方 Recipes](#实践配方-recipes) | 用引擎解决具体问题 | 可运行的片段、代价表 |
| [基准测试](#基准测试) | 挑工作点 | 可运行脚本 + 实测报告 |
| [源码地图](#源码地图) | 从概念反查代码 | 模块对照表 |

## 发版公告

- [v0.1.0](release-notes-v0.1.0.md) / [简体中文](release-notes-v0.1.0.zh-CN.md) —— 首个 tag：
  持久化的混合检索、前置/后置过滤、IVF、离线适配器。
- [v0.1.1](release-notes-v0.1.1.md) / [简体中文](release-notes-v0.1.1.zh-CN.md) —— MCP stdio
  服务端、`mcp<2` 固定，以及 quickstart 的几处更正。
- [v0.1.2](release-notes-v0.1.2.md) / [简体中文](release-notes-v0.1.2.zh-CN.md) —— 补齐中英双语的
  FAQ、工程实践、实践配方与笔记，引擎没有改动。

## 案例研究

- [跨层排障](case-studies/cross-layer-bugfix.md) /
  [简体中文](case-studies/cross-layer-bugfix.zh-CN.md) —— 本项目两个稳定失败的测试，一路追到
  `sable` 的 memtable：它和自己的范围扫描对「一个固定 sequence 该解析出什么」给出了不同答案。
  内含纯上游复现，以及为什么这事在本仓库修不了。

## 架构决策记录

决定磁盘与内存契约的五个决策，都按 **状态 / 背景 / 决策 / 后果** 写。要改键前缀、提交边界或依赖
固定之前，先读这些。

- [ADR-001：单一键空间，用前缀分隔](decisions/adr-001-single-keyspace.md) /
  [简体中文](decisions/adr-001-single-keyspace.zh-CN.md) —— 向量、payload、倒排条目和 IVF 状态
  共用一棵 `sable` 树，于是只有一份 WAL、一套恢复逻辑。
- [ADR-002：payload 就是提交记录](decisions/adr-002-payload-commit-record.md) /
  [简体中文](decisions/adr-002-payload-commit-record.zh-CN.md) —— 其余键一律 `sync=False` 暂存，
  只在 payload 上刷一次盘。
- [ADR-003：两种过滤策略都发](decisions/adr-003-dual-filter-strategies.md) /
  [简体中文](decisions/adr-003-dual-filter-strategies.zh-CN.md) —— 前置与后置过滤返回同样的行，
  但代价曲线和失效方式各不相同。
- [ADR-004：float32 零拷贝镜像、分数一律越大越近](decisions/adr-004-float32-codec-and-score-orientation.md) /
  [简体中文](decisions/adr-004-float32-codec-and-score-orientation.zh-CN.md) —— 不加帧字节、
  不在 Python 里逐行循环、三种度量共用一种分数方向。
- [ADR-005：把定义 API 的依赖固定住](decisions/adr-005-pinned-dependencies.md) /
  [简体中文](decisions/adr-005-pinned-dependencies.zh-CN.md) —— `mcp<2` 与 `sable@v0.1.1` 直引用，
  以及为什么让 Dependabot 别碰它们。

## 工程实践 Skills：底层实现指南

三篇按不变量要求的顺序读源码的指南，每篇都带代码、失效方式表格和检查清单。

- [零拷贝向量：编解码与分数方向](skills/zero-copy-codec.md) /
  [简体中文](skills/zero-copy-codec.zh-CN.md) —— 无帧头的 `<f4` 镜像、`np.frombuffer`
  视图、`writeable=True` 副本，以及三种度量为什么共用一种分数方向。
- [持久写入：暂存、提交记录与崩溃重建](skills/durable-upsert.md) /
  [简体中文](skills/durable-upsert.zh-CN.md) —— `_write_batch` 把每个键 unsynced 暂存、
  只在 payload 上提交一次；`reclaim()` 与 `recover()` 按已提交集合的纯函数重建派生状态。
- [混合查询策略：两个执行器，两条代价曲线](skills/hybrid-query-strategies.md) /
  [简体中文](skills/hybrid-query-strategies.zh-CN.md) —— `matching_ids`、排序后的候选
  gather、后置过滤的遍历，以及用 `threading.Barrier` 写确定性竞态测试。

## 硬核问答 FAQ

八句「干嘛不直接……」，分五组：单一键空间、提交边界、字节与分数、两条过滤路径、
照实测数字谈近似检索，以及钉死的依赖。每个回答都指向真实存在的符号。

- English: [Hybrid retrieval, answered honestly](faq/vector-search-qa.md)
- 简体中文: [混合检索：把追问回答清楚](faq/vector-search-qa.zh-CN.md)

## 实践配方 Recipes

- [按租户收窄的混合检索](recipes/hybrid-retrieval.md) /
  [简体中文](recipes/hybrid-retrieval.zh-CN.md) —— open 时声明 `indexed_fields`、
  打分前解析谓词、看 `scored` 而不只看结果列表。含三种会悄悄给错答案的情况。
- [照实测数字选 `nprobe`](recipes/ivf-nprobe-tuning.md) /
  [简体中文](recipes/ivf-nprobe-tuning.zh-CN.md) —— 基于那次实测运行的决策步骤，
  包括为什么 1000 条时精确检索才是默认值，以及为什么 p95 离群值要留在表里。

## 基准测试

[`benchmarks/bench_vector.py`](../benchmarks/bench_vector.py) 自带断言：常驻的 `FlatIndex` 是穷举的，
把 recall 定义为 1.0，所以每个近似数字都是拿它当 ground truth 量出来的，不是假定出来的。
跑法：`python -m benchmarks.bench_vector --dim 1536 --sizes 10000`。

- [实测报告](reports/benchmark.md) /
  [简体中文](reports/benchmark.zh-CN.md) —— 两档语料规模下的延迟、吞吐和
  recall / `nprobe` 曲线，采集机器与这次运行刻意不覆盖的部分都写明。

## 源码地图

| 模块 | 职责 | 关键符号 |
| --- | --- | --- |
| [`types.py`](../src/sable_vector/types.py) | id、度量、记录、结果 | `VectorId`、`DistanceMetric`、`VectorRecord`、`Match` |
| [`codec.py`](../src/sable_vector/codec.py) | float32 ↔ 字节、payload JSON | `encode_vector`、`decode_vector`、`decode_matrix` |
| [`distance.py`](../src/sable_vector/distance.py) | NumPy 核、分数方向、top-k | `score_matrix`、`top_k`、`normalise_matrix` |
| [`storage.py`](../src/sable_vector/storage.py) | 带前缀的命名空间适配器 | `VectorStorage`、`prefix_upper_bound`、`escape_segment` |
| [`engine.py`](../src/sable_vector/engine.py) | 持久 upsert / delete / reclaim | `VectorEngine`、`RepairReport` |
| [`recovery.py`](../src/sable_vector/recovery.py) | 崩溃重建、索引健康 | `recover`、`rebuild_ivf`、`IndexHealth` |
| [`index/flat.py`](../src/sable_vector/index/flat.py) | 穷举矩阵检索 | `FlatIndex`、`search_streaming` |
| [`index/scalar.py`](../src/sable_vector/index/scalar.py) | 倒排前缀索引 | `ScalarIndex`、`matching_ids` |
| [`index/ivf_cluster.py`](../src/sable_vector/index/ivf_cluster.py) | 质心训练 | `kmeans`、`assign`、`residuals` |
| [`index/ivf_storage.py`](../src/sable_vector/index/ivf_storage.py) | 倒排桶、`nprobe` 修剪 | `IvfIndex`、`IvfOutcome` |
| [`query/prefilter.py`](../src/sable_vector/query/prefilter.py) | 谓词在前 | `PrefilterExecutor`、`PrefilterOutcome` |
| [`query/postfilter.py`](../src/sable_vector/query/postfilter.py) | 几何在前 | `PostfilterExecutor`、`PostfilterOutcome` |
| [`adapter/langchain.py`](../src/sable_vector/adapter/langchain.py) | 离线 `VectorStore` | `SableVectorStore` |
| [`adapter/mcp_server.py`](../src/sable_vector/adapter/mcp_server.py) | stdio MCP 工具 | `list_tools`、`call_tool` |
| [`cli.py`](../src/sable_vector/cli.py) | `serve-mcp` 与 CLI 入口 | `app` |

## 从哪儿读起

1. [`README.md`](../README.md) —— 一页纸的心智模型和三张 Mermaid 图。
2. [ADR-001](decisions/adr-001-single-keyspace.md) 然后
   [ADR-002](decisions/adr-002-payload-commit-record.md) —— 所有持久化承诺都压在这两条上。
3. [Skills → 持久写入](skills/durable-upsert.zh-CN.md) 然后
   [Skills → 零拷贝编解码](skills/zero-copy-codec.zh-CN.md) —— 这两条决策落到源码里长什么样。
4. [ADR-003](decisions/adr-003-dual-filter-strategies.md) 与
   [Skills → 混合查询策略](skills/hybrid-query-strategies.zh-CN.md) —— 为什么是两个执行器，
   而不是一个更聪明的。
5. [案例研究](case-studies/cross-layer-bugfix.md) —— 上游不变量只在一条读路径上实现时会出什么事。
6. [硬核问答](faq/vector-search-qa.zh-CN.md) —— 「干嘛不直接……」的追问，先答一遍。
7. [Recipes → 选 `nprobe`](recipes/ivf-nprobe-tuning.zh-CN.md) 与
   [实测报告](reports/benchmark.zh-CN.md) —— 定工作点之前先看这两份。

---

[← 返回项目主文档](../README.zh-CN.md)
