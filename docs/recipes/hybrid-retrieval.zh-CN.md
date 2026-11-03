[English](hybrid-retrieval.md) | [简体中文](hybrid-retrieval.zh-CN.md)

# 实践配方：按租户收窄的混合检索

> 一个 collection 一旦不止一个 owner，很快就会碰到这个问题：「在全库里搜，但只限这个租户，
> 而且只要最近的记录」。这篇用库 API 把这个查询搭出来，说明它要付多少代价，再列出三种会
> 悄悄给错答案的情况。所有片段都来自仓库里自带的代码，demo 自带断言。

## 先给能跑的那份

**在 open 的时候**就要声明将来要过滤的字段，把它们写进 payload，然后在打分之前解析谓词：

```python
import numpy as np
from sable_vector.engine import VectorEngine
from sable_vector.query.prefilter import PrefilterExecutor
from sable_vector.index.scalar import ScalarIndex
from sable_vector.index.flat import FlatIndex

engine = VectorEngine.open(
    "collection.sv",
    metric="cosine",
    indexed_fields=("tenant", "lang", "year"),
)

engine.upsert("doc-1", np.random.randn(1536).astype("float32"),
              payload={"tenant": "acme", "lang": "en", "year": 2024})
engine.upsert("doc-2", np.random.randn(1536).astype("float32"),
              payload={"tenant": "acme", "lang": "en", "year": 2025})
engine.upsert("doc-3", np.random.randn(1536).astype("float32"),
              payload={"tenant": "globex", "lang": "en", "year": 2025})

flat = FlatIndex.materialise(engine.storage, metric="cosine", dim=engine.dim)
scalar = ScalarIndex(engine.storage, ("tenant", "lang", "year"))
executor = PrefilterExecutor(engine.storage, flat, scalar)

outcome = executor.search(query, k=5,
                          predicate={"tenant": "acme", "year": {"$gte": 2025}})
print(outcome.candidates, outcome.scored, len(outcome.matches))
```

要盯的是 `outcome.scored`。三行数据看不出什么；换成 50,000 行、再配一个选择性强的租户条件，
它就是「给 50,000 条向量打分」和「给 400 条打分」的区别。

## 为什么 `indexed_fields` 是写时决策

倒排索引不是 schema。`ScalarIndex.reindex()` 只会给
`VectorEngine.open(indexed_fields=...)` 里列过的字段写 `idx:` 条目；没声明的字段，磁盘上一个
`idx:` 键都不会有。所以 `matching_ids()` 遇到没声明过的字段会抛 `ValueError`，而不是给一份
不完整的答案：

```
ValueError: field 'owner' is not indexed; indexed fields are ['tenant', 'lang', 'year']
```

事后补字段必须连同已有语料一起重建——已经写进去的那些行，磁盘上并不存在对应条目。
哪些属性要能过滤，得在第一次灌数据之前定下来。

## 三种会出错的样子

| 现象 | 原因 | 怎么办 |
| --- | --- | --- |
| 明明匹配的行，过滤后却搜不到 | 声明该字段之前就写进去了，没有 `idx:` 条目 | 重建语料，或改用后置过滤 |
| payload 里明明有这个字段，却抛 `ValueError` | 没在 `indexed_fields` 里声明 | 声明它，然后重建 |
| 有匹配，却返回少于 `k` 条 | 走的是后置过滤，全局 top-N 里全是被谓词刷掉的行 | 查 `outcome.complete`；放宽 `examine_budget`，或改用前置过滤 |

前两种是前置过滤独有的。[ADR-003](../decisions/adr-003-dual-filter-strategies.zh-CN.md) 把它们
记作「更便宜的代价曲线」的价码。后置过滤没有盲区——它直接读 payload——但可能少于 `k`，
而且永远给全库打分。

## 要看代价，不能只看结果

| 字段 | `PrefilterOutcome` | `PostfilterOutcome` |
| --- | --- | --- |
| `candidates` / `scored` | 放行的 id 数 / 实际打分行数——应明显小于全库 | — / 永远等于全库大小 |
| `examined` | — | 读了多少次 payload；谓词宽约等于 `k`，谓词强就走得很深 |
| `complete` | — | `False` 表示凑够 `k` 个之前，排名（或预算）用完了 |

像 `{"year": {"$gte": 2000}}` 这种很宽的谓词就不适合前置过滤：候选集几乎就是全库，索引查了
一次却什么也没省。它反而适合后置过滤，大约读 `k` 次 payload 就收工——
`tests/test_postfilter.py` 断言的正是这件事（`k=5` 时 `examined == 5`）。

操作符就是那一套：`$eq`、`$ne`、`$in`、`$nin`、`$gt`、`$gte`、`$lt`、`$lte`，裸标量表示等值。
数值由 `render_value()` 渲染成定宽文本，所以 `$gte` 是真的范围比较，不会撞上字典序的意外。

## 仓库里已经有一份离线 demo

[`examples/langchain_rag_demo.py`](../../examples/langchain_rag_demo.py) 不下载模型、不发网络请求
就能把整条链跑完。它的 embedder 是带符号的 hashing trick——不语义化，但确定——并且把三次
搜索的 `scored` 都打出来：不过滤、限一个租户加年份范围、以及过滤一个从未建索引的字段。
第三次专门演示 fallback：它照样返回正确的文档，代价是一次完整排名，并且报出 mode 而不是抛错。

```bash
python examples/langchain_rag_demo.py
```

结尾它自己断言自己的结论——`narrow >= baseline` 就算失败——所以这是一份能跑的检查，验证
过滤减少的是**工作量**，不只是结果集：

```python
store.similarity_search(QUESTION, k=3)
baseline = store.last_search.scored
store.similarity_search(QUESTION, k=3, filter={"tenant": "acme"})
narrow = store.last_search.scored
if narrow >= baseline:
    return 1
```

适配器那边用 `filter=` 传同样形状的谓词：

```python
store.similarity_search_with_score(
    QUESTION, k=3, filter={"tenant": "acme", "year": {"$gte": 2025}}
)
```

## 运维提示

- 把 `indexed_fields` 放进配置，别散落在各个调用点。灌数据时声明的，和查询时传进去的如果对
  不上，写侧是静默的，读侧才会响。
- 别把短结果列表当成「没有匹配」。先看 `complete`。
- 谓词形状经常变（很多临时属性）时，后置过滤才是更诚实的默认值——总不能给每个属性都建一份
  索引。
- 后置过滤的 payload 读取是按幸存候选算的。有延迟 SLA 就设 `examine_budget`，并在调用方
  处理 `complete=False`。
- 两个执行器必须返回同样的行。如果对不上，那是 bug——拿不带谓词的 `FlatIndex.search()`
  比一比。

## 接着读哪儿

| 想看的事 | 出处 |
| --- | --- |
| 完整机制与确定性 gather | [混合查询策略](../skills/hybrid-query-strategies.zh-CN.md) |
| 两个执行器背后的决策 | [ADR-003](../decisions/adr-003-dual-filter-strategies.zh-CN.md) |
| 前缀键与值的渲染 | [`storage.py`](../../src/sable_vector/storage.py) |
| 离线 `VectorStore` 适配器 | [`adapter/langchain.py`](../../src/sable_vector/adapter/langchain.py) |
| 能跑的前后对比 | [`examples/langchain_rag_demo.py`](../../examples/langchain_rag_demo.py) |
