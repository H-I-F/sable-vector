[English](hybrid-query-strategies.md) | [简体中文](hybrid-query-strategies.zh-CN.md)

# 混合查询策略：两个执行器，两条代价曲线

> 两个执行器都实现 `search(query, k, predicate)`，也都承诺返回同样的 `Match` 列表。但它们不能
> 互换。一个把谓词放在几何之前，一个放在几何之后；就这一个选择，决定了查询要付多少代价，
> 以及它被允许怎样失效。[ADR-003](../decisions/adr-003-dual-filter-strategies.zh-CN.md)
> 是故意把两个都留下的。

## 一、共用的契约

```python
PrefilterExecutor(storage, flat, scalar).search(query, k, predicate=...)
PostfilterExecutor(storage, flat).search(query, k, predicate=..., examine_budget=...)
```

两者都会给结果挂上 payload，返回 best-first 的排名。差别在你看第一行结果之前，从 outcome
对象里就能读出来：

| | `PrefilterOutcome` | `PostfilterOutcome` |
| --- | --- | --- |
| `candidates` | 通过谓词的 id 数；无约束时为 `None` | — |
| `scored` | 真正交给核去打分的行数 | 永远是全库 |
| `examined` | — | 读了多少次 payload |
| `complete` | — | 有没有凑够 `k` 个幸存者 |

`scored` 是回答「前置过滤到底省没省」的那个数；`examined` 是回答「后置过滤的遍历走了多远」
的那个数。

## 二、前置过滤：把谓词解析成候选集

```python
def search(self, query, k, *, predicate=None) -> PrefilterOutcome:
    allowed = self._scalar.matching_ids(predicate)
    candidates = None if allowed is None else len(allowed)
    matches = self._flat.search(query, k, candidate_ids=allowed)
    scored = candidates if candidates is not None else self._flat.size
    return PrefilterOutcome(matches=tuple(self._with_payloads(matches)),
                            candidates=candidates, scored=scored)
```

活儿在 `ScalarIndex.matching_ids()` 里：

```python
if not predicate:
    return None                       # 「不设限制」，跟空集是两回事
for field in predicate:
    if field not in self._fields:
        raise ValueError(f"field {field!r} is not indexed; indexed fields are ...")
ordered = sorted(predicate.items(),
                 key=lambda item: len(_probe_values(item[1]) or []))
result = None
for field, condition in ordered:
    ids = self.field_ids(field, condition)
    result = ids if result is None else (result & ids)
    if not result:
        return set()
return result
```

有三处值得慢一点读：

- **`None` 与空集。** 空谓词表示「完全不限制」，于是 `candidate_ids=None`，执行器退化成一次
  普通 flat 检索；一个什么都没匹配上的谓词是 `set()`——空的候选块，零行打分。把这两者混为一谈，
  会悄悄把「没过滤」变成「什么都不返回」。
- **选择性最强的字段先算。** `_probe_values` 估出某个条件要查几次索引，按它排序让交集从一个
  小集合开始；一旦中途为空就直接短路。
- **没声明过的字段直接抛错。** 索引字段是写路径上的元数据，由
  `VectorEngine.open(indexed_fields=...)` 定死。谓词里出现没声明过的字段，倒排索引答不了，
  于是响亮地失败，而不是悄悄给一份不完整的答案。

### 谓词比较的是渲染后的文本

`render_value()` 把值做成可排序的文本，于是等值探测是一次前缀扫描，范围条件是一段连续扫描：

| 输入 | 渲染结果 |
| --- | --- |
| `True` / `False` | `true` / `false` |
| 数字 | `f"{value:020.6f}"` —— 字典序与数值序一致 |
| 字符串 | 原样 UTF-8 |

`evaluate_condition()` 认 `FILTER_OPERATORS`：`$eq`、`$ne`、`$in`、`$nin`、`$gt`、`$gte`、
`$lt`、`$lte`。裸标量就是等值。表达式里出现未知操作符会在**读任何数据之前**抛 `ValueError`，
所以格式不对的查询，哪怕对着一个空字段也会被拒。demo 里用的是
`{"tenant": "acme", "year": {"$gte": 2025}}`。

### 候选块为什么要先排序再 gather

Python 的 id 集合按哈希序迭代，而哈希种子每个进程随机。按这个次序 gather 出来的矩阵块不同，
gemv 里的浮点舍入也就不同；两个分数恰好相等的候选，名次会随哈希种子互换。
`FlatIndex.search()` 先把行号排好，再做一次 gather：

```python
rows = sorted(self._positions[label] for label in candidate_ids
              if label in self._positions)
grid = self._matrix[rows]
```

这是 gather 而不是 mask——候选块交给的是与未过滤路径同一个核。确定性在这里的开销，只是对
候选表排一次序。

## 三、后置过滤：先给全库排名，再逐个读 payload

```python
def search(self, query, k, *, predicate=None, examine_budget=None):
    if not predicate:
        matches = flat.search(query, k)
        return PostfilterOutcome(..., examined=len(matches), scored=flat.size,
                                 complete=len(matches) >= k)
    scores = score_matrix(flat.matrix, probe, flat.metric)
    order = np.argsort(-scores, kind="stable")   # 全排序，不是挑选
    limit = len(order) if examine_budget is None else min(len(order), examine_budget)
    accepted = []
    for position in order[:limit]:
        payload = self._storage.get_payload(labels[int(position)]) or {}
        examined += 1
        if not self._satisfies(payload, predicate):
            continue
        accepted.append(Match(...))
        if len(accepted) >= k:
            break
    return PostfilterOutcome(..., examined=examined, scored=flat.size,
                             complete=len(accepted) >= k)
```

这里**必须**是全排序：遍历要从排名顶端一行行往下吃，部分排序没法告诉它下一行是谁。
`_satisfies()` 用的是同一套 `render_value()` / `evaluate_condition()` 规则，只是对着 payload 求值，
所以两个执行器对「谓词什么意思」的理解是一致的。payload 里没有的字段不可能满足条件——这跟
倒排索引的行为对得上，那里只有真正出现过的值才有条目。

代价曲线正好与前置过滤相反。谓词宽的时候，幸存者就排在前面，遍历大约读 `k` 次 payload 就停，
倒排索引纯属多余；`tests/test_postfilter.py` 对一个宽谓词断言 `examined == 5`（`k=5`）。谓词
选择性强的时候，遍历会走得很深——`examine_budget` 给它设上限，`complete=False` 则告诉调用方
拿到的是一段前缀而不是完整答案。设预算是在延迟与完整性之间选了延迟；默认 `None` 会走完整个
排名，对正确性来说这才是诚实的那个默认值。

## 四、失效方式不同，而且两种都写明了

| | `PrefilterExecutor` | `PostfilterExecutor` |
| --- | --- | --- |
| 盲区 | 声明字段之前写进去的行，对它不可见 | 没有——它直接读 payload |
| 会不会少于 `k` | 只在候选集确实更小时 | 会，近邻被谓词刷掉时 |
| 最坏情况 | 字段没建索引就抛错 | 谓词强时几乎走遍全库 |
| 前提 | 需要 `indexed_fields`，改字段要重建语料 | 除了 payload 什么都不需要 |

[ADR-003](../decisions/adr-003-dual-filter-strategies.zh-CN.md) 把这笔账算得很清楚：选择性强的
谓词把 50k 行收窄到 400，省掉 99% 的算术；谓词宽的时候幸存者本来就在顶端，后置过滤早早收工。
既然两者承诺同样的行，测试套件就拿穷举的 flat 索引核对它们是否等价——不一致是 bug，不是
调参差异。

## 五、查询路径上的确定性竞态测试

`tests/test_concurrent.py` 是这里测并发读的范本，开头那段话本身就是条通用规矩：光是「起几个
线程跑到出事」证明不了哪些状态合法；而在该出现竞态的地方放一个 `sleep` 更糟，因为 sleep 恰好
把它本想暴露的交错藏起来了。

- `threading.Barrier` 让所有参与者同时放行，交错于是成为测试的属性。
- 读者跑固定轮数；用 `Event` 而不是计时器来判定写后状态。
- 一层守卫记下线程里抛出的任何异常并打破 barrier——否则 Python 只报个 warning，测试会因为
  错误的原因通过。
- 断言的不变量是方向性的：在并发 upsert **和**并发 delete 之下，快照读者都不会看到「payload
  在、向量不在」；快照是冻结的；pins 在释放时归还。这套测试刻意不让两个线程改同一个 id——
  这里的原子性是单个写进程内、按记录算的。

[跨层排障记](../case-studies/cross-layer-bugfix.zh-CN.md) 讲的就是：上游不变量只在一条读路径上
实现时会出什么事——这边两个稳定失败的测试，一路追到 `sable` 的 memtable，它和自己的范围扫描
对「一个固定 sequence 该解析出什么」给出了不同答案。

## 六、挑执行器的检查清单

1. 谓词选择性强吗？它点到的字段是不是都声明过 indexed？→ 前置过滤，并且看一眼
   `outcome.scored` 是否真的降下来了。
2. 谓词宽、库小，或者字段是语料建好之后才加的？→ 后置过滤，检查 `outcome.examined`
   与 `outcome.complete`。
3. 有延迟 SLA 就设 `examine_budget`，并显式处理 `complete=False`；别把短列表当成「没有匹配」。
4. 新增索引字段，一定要连同已有语料一起重建。
5. 候选 gather 保持排序——按集合序 gather 会毁掉可复现性。
6. 拿不准就把两个执行器都对着 flat 索引跑一遍比一比；测试套件就是这么干的，代价很小。

## 接着读哪儿

| 想看的事 | 出处 |
| --- | --- |
| 谓词 → 候选 id | [`index/scalar.py`](../../src/sable_vector/index/scalar.py) |
| 候选 gather 与穷举核 | [`index/flat.py`](../../src/sable_vector/index/flat.py) |
| 两个执行器 | [`query/prefilter.py`](../../src/sable_vector/query/prefilter.py)、[`query/postfilter.py`](../../src/sable_vector/query/postfilter.py) |
| 决策记录 | [ADR-003](../decisions/adr-003-dual-filter-strategies.zh-CN.md) |
| 等价性与代价断言 | [`tests/test_prefilter.py`](../../tests/test_prefilter.py)、[`tests/test_postfilter.py`](../../tests/test_postfilter.py) |
| 一份能跑的前后对比 | [混合检索配方](../recipes/hybrid-retrieval.zh-CN.md) |
