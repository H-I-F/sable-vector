[English](ivf-nprobe-tuning.md) | [简体中文](ivf-nprobe-tuning.zh-CN.md)

# 实践配方：照实测数字选 `nprobe`

> 关于这个引擎里的 IVF，最有用的问题不是「`nprobe` 该设多少」，而是「到底该不该上 IVF」。
> 这篇以 [实测报告](../reports/benchmark.zh-CN.md) 为据——下面的数字
> 全部引自那次运行，不是为这一页现跑的——把它整理成一套决策步骤。本文没有重跑压测。

## 第 0 步：先弄清索引在打分之前就要付什么

`IvfIndex.search()` 先用一次 gemm 给所有分区按质心排个序，然后只读胜出的那几个倒排桶：

```python
distances = squared_distances(probe.reshape(1, self._dim), self._centroids)[0]
order = np.argsort(distances, kind="stable")[: min(nprobe, self.partitions)]
```

给质心打分、读倒排桶，都发生在**给第一条向量打分之前**。语料小的时候，常驻 flat 索引没有这层
开销——它就是在连续的 `(n, d)` 块上做一次 gemv。所以近似检索不是免费的加速档，它是一笔赌注：
赌「跳过大部分语料」值得为此付一次路由的代价。

训练还有前提：`IvfIndex.train()` 会拒绝空语料，也会拒绝比 `partitions` 还小的语料。
`tests/test_ivf_scan.py` 把这两条都断言了，还断言训练只用一次 fsync 提交。

## 第 1 步：决策表

| 语料规模 | 数字怎么说 | 该用什么 |
| --- | --- | --- |
| ~1,000 条 | flat p50 **0.144 ms**、recall 1.00，比 IVF 在任何 `nprobe` 下都快 | 常驻 `FlatIndex` |
| ~5,000 条，内存吃紧 | flat p50 3.773 ms；`flat-stream` 4.993 ms、recall 1.00、约为常驻路径 8 倍 | `search_streaming()` |
| ~5,000 条，recall 可以谈 | IVF `nprobe=8` 给 p50 5.860 ms、recall 0.920 | IVF，前提是 0.920 够用 |
| 大到「跳过多数行」明显盖过路由成本 | — | IVF，且要在自己的数据上重新定 |

第一行恰恰是大家最容易跳过的。16 个簇、1000 条语料时，`nprobe=8` 已经是半个语料库——曲线平到
recall 1.000，p50 却涨到 1.489 ms，是 flat 的六倍。这个规模上精确检索才是正确的默认值，报告
里也是这么直说的。

`search_streaming()` 针对的是另一种约束，不是另一种速度：它只在内存里留 `block_rows × d` 个
float 加 `n` 个分数，所以它是「语料放不进内存」时该用的模式。recall 保持 1.00，代价是受磁盘
约束——这正是它在 1000 条时约为常驻路径 8 倍的原因。

## 第 2 步：看 recall / 延迟曲线，别只看一个数

下面两张表都照抄自报告（128 维、16 个簇、30 条查询、`k=10`、cosine、seed 42，Linux x86-64 /
Python 3.11 / NumPy 2.3.5）。

**1,000 条** —— ingest 21,108 vec/s：

| 索引 | nprobe | p50 (ms) | p95 (ms) | qps | recall@k |
| --- | ---: | ---: | ---: | ---: | ---: |
| flat | — | 0.144 | 0.157 | 6,894 | 1.00 |
| flat-stream | — | 1.148 | 1.194 | 871 | 1.00 |
| ivf | 1 | 0.257 | 0.316 | 4,022 | 0.523 |
| ivf | 2 | 0.466 | 0.553 | 2,166 | 0.680 |
| ivf | 4 | 0.819 | 1.202 | 1,129 | 0.877 |
| ivf | 8 | 1.489 | 1.609 | 663 | 1.000 |

**5,000 条** —— ingest 26,596 vec/s：

| 索引 | nprobe | p50 (ms) | p95 (ms) | qps | recall@k |
| --- | ---: | ---: | ---: | ---: | ---: |
| flat | — | 3.773 | 95.074 | 34 | 1.00 |
| flat-stream | — | 4.993 | 5.024 | 200 | 1.00 |
| ivf | 1 | 0.897 | 1.179 | 1,124 | 0.623 |
| ivf | 2 | 1.534 | 1.933 | 639 | 0.727 |
| ivf | 4 | 3.087 | 3.613 | 322 | 0.810 |
| ivf | 8 | 5.860 | 6.756 | 167 | 0.920 |

`nprobe` 拿延迟换 recall，价格可预期。5000 条下从 1 个分区加到 8 个，recall 0.623 → 0.920，
p50 从 0.897 涨到 5.860 ms，qps 从 1,124 掉到 167。先锚定应用真正需要的 recall，再从同一行
读出延迟；别先定一个延迟预算，然后指望 recall 刚好够用。

`nprobe == partitions` 退化成全扫、recall 精确。知道这个上限是有用的，把它当成工作点去跑则
毫无意义——到那一步，你是在全库之上又多付了一次质心打分。

### 有个离群值是照发的，不是抹平的

5000 档的常驻 flat 行 p50 是 3.773 ms，p95 却是 **95.074 ms**，mean 29.779 ms。那是 30 条查询里
少数几条撞上预热／分配停顿，不是核的性质。p50 是有代表性的那个数；p95 正是「不该只报一个数」
的理由。重跑一次，这些尾部会动，recall 那一列不会。把 recall 曲线和比值当作结论，把绝对延迟
当作一台容器的快照。

ingest 也不是持久写基准：128 维下 21–27k vectors/s 是内存暂存速率，不含每条记录那次
`fsync`。那条边界到底在哪儿，见 [ADR-002](../decisions/adr-002-payload-commit-record.zh-CN.md)。

## 第 3 步：在自己的形状上复现

压测脚本自带断言——常驻 flat 索引是穷举的，recall 按定义为 1.0，每个近似数字都拿它当
ground truth、用 `_recall_at_k` 量出来。发布的那次跑法是：

```bash
python -m benchmarks.bench_vector \
  --dim 128 --sizes 1000 5000 --queries 30 --ks 10 \
  --nprobes 1 2 4 8 --clusters 16 --stream-reps 2 \
  --metric cosine --seed 42
```

要贴近生产形状，改的是形状，不是拿外推去信：

```bash
python -m benchmarks.bench_vector --dim 1536 --sizes 10000 --queries 50
```

这次发布**没有**覆盖：1536 维、`l2` 与 `dot`、IVF 训练耗时、并发读，以及带真实 `fsync` 的写
成本。这里的 recall 是对**同一份语料**的穷举索引量出来的，所以它对上游 embedding 模型的质量
不置一词。只要你的工作点依赖上面任何一项，就去实测它。

## 第 4 步：崩过一次之后，先让索引恢复健康

近似索引是派生数据。`recover()` 会调 `inspect_ivf()`，再问
`IndexHealth.needs_retrain(rebuild_ratio)` 该重算分区还是只补差量。`drift` 是
`missing + stale + duplicated`；镜像 `malformed`、元数据缺失、`dim`／`k` 对不上、出现任何
`duplicated`，或者漂移超过阈值，都会强制重训。否则就地删掉 stale 的桶内项、补上 missing 的项。
对健康的库调 `recover()` 会给出一份全零报告，这正是它可以每次 open 都跑的原因。

这一点对调参很要紧：漂移的索引不会响亮地失败，它会返回看着挺合理的邻居，recall 却悄悄掉了。
在把一个 recall 数字当成 `nprobe` 的属性之前，先看一眼 `IndexHealth.healthy`。

## 检查清单

1. 先问精确检索是不是已经满足延迟预算。1000 条时不仅满足，还富余很多。
2. 语料放不进内存就用 `search_streaming()`——它把 recall 保持在 1.00。
3. 先定需要的 recall，再从那一行读 p50 / qps；p95 的尾部也要看。
4. `nprobe` 明显小于 `partitions`；顶到上限，等于在全扫之上白付一次路由。
5. 定下数字之前，在自己的维度与度量上复现一遍。
6. 崩过之后，先验索引健康，再信 recall 测量结果。

## 接着读哪儿

| 想看的事 | 出处 |
| --- | --- |
| 实测报告 | [benchmark.md](../reports/benchmark.zh-CN.md) |
| 能跑的压测脚本 | [`benchmarks/bench_vector.py`](../../benchmarks/bench_vector.py) |
| `nprobe` 修剪与倒排桶 | [`index/ivf_storage.py`](../../src/sable_vector/index/ivf_storage.py) |
| 质心训练 | [`index/ivf_cluster.py`](../../src/sable_vector/index/ivf_cluster.py) |
| 崩溃重建与索引健康 | [`recovery.py`](../../src/sable_vector/recovery.py)、[持久写入](../skills/durable-upsert.zh-CN.md) |
| 基准背后的分数方向 | [零拷贝编解码](../skills/zero-copy-codec.zh-CN.md) |
