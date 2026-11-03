[English](adr-003-dual-filter-strategies.md) | [简体中文](adr-003-dual-filter-strategies.zh-CN.md)

# ADR-003：两种过滤策略都发

**状态：** 已接受 —— 实现在 `src/sable_vector/query/`。

## 背景

元数据谓词的选择性可以差得很远（`tenant = acme` 很窄，也可能几乎匹配全部）。单一执行顺序不可能
两头都最省：给候选块打分要做一次矩阵–向量乘，代价随块大小走；读 payload 则是每个候选一次存储读。
退回到「取一大块缓冲再在 Python 里过滤」会悄悄拉低 recall，还让延迟取决于谓词碰巧有多窄。

## 决策

暴露**两个执行器**，返回同样的 `Match` 列表，并且刻意做成**不可互换**：

- `PrefilterExecutor` 先由 `ScalarIndex.matching_ids()` 解出谓词，再把幸存的 id 交给
  `FlatIndex.search(query, k, candidate_ids=...)`。谓词里出现没在
  `VectorEngine.open(indexed_fields=...)` 声明过的字段，会抛 `ValueError`。
- `PostfilterExecutor` 先算全局 top-N，按排名顺序往下走，逐个读 payload，凑够 `k` 个幸存者就停。
  它完全不需要索引——谓词直接对着 payload 求值。

两者共用 `search(query, k, predicate)` 这个签名，于是调用方选的是失效方式，不是另一套 API。

## 后果

谓词很窄时，把 50k 行收窄到 400 行，就省掉了 99% 的算术。谓词很宽时，幸存者本来就排在前面，
后置过滤大约读 `k` 次 payload 就能停，倒排索引反而是纯开销——`PostfilterOutcome` 会报告它一路
走到排名的第几位。

两者的失效方式不同，而且是写明而不是藏起来的：前置过滤看不到某字段被声明为 indexed 之前写进去的
行；后置过滤可能返回少于 `k`，因为全局 top-N 里可能全是没通过谓词的近邻。既然两个实现承诺同样的
行，测试套件就拿穷举的 flat 索引当 ground truth 校验它们等价；对不上就是 bug，不是调参差异。
