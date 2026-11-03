[English](durable-upsert.md) | [简体中文](durable-upsert.zh-CN.md)

# 持久写入：暂存、提交记录与崩溃重建

> 这个引擎的写路径，说到底是三行顺序加一条可见性规则。其余一切——`reclaim()`、`recover()`、
> 混沌测试、并发测试断言的那条方向性不变量——都是从这条规则推出来的。这篇按 `engine.py`、
> `recovery.py` 的顺序读一遍。

## 一、一条不变量：记录就是它的 payload

`p:<id>` 存在，记录才算提交。[ADR-002](../decisions/adr-002-payload-commit-record.zh-CN.md)
把它定为唯一的可见性判据。看代码之前，先说清它换来了什么：读的人不必把向量键、倒排条目和
payload 三方核对一遍才能判断某行算不算数；恢复时可以把其余所有键都当成派生数据，而派生数据
是已提交 payload 的纯函数。

这条不变量是**有方向**的。向量先于 payload 写入、晚于 payload 删除，所以整个序列里不存在
「payload 在、向量不在」的时刻。反过来「向量在、payload 不在」恰恰就是写残的样子，也是回收器
存在要清理的那个方向。

## 二、`_write_batch`：全部暂存，一次提交

```python
def _write_batch(self, items):
    if not items:
        return
    prepared = [(vid, self._admit(vec), payload) for vid, vec, payload in items]

    # 1. 又大又便宜的派生数据：float32 镜像。
    for vector_id, vector, _ in prepared:
        self._storage.put_vector(vector_id, encode_vector(vector), sync=False)

    # 2. 倒排条目，按它要替换的 payload 逐条算增量。
    for vector_id, _, payload in prepared:
        previous = self._storage.get_payload(vector_id)
        self._apply_index_delta(vector_id, previous or {}, payload, sync=False)

    # 3. 提交记录。只有最后一条把日志强制落到稳定存储。
    last = len(prepared) - 1
    for index, (vector_id, _, payload) in enumerate(prepared):
        self._storage.put_payload(vector_id, payload, sync=index == last)
```

`sync=False` 把帧留在 WAL 缓冲区里。最后一条 payload 的 `sync=True` 把整段一次性刷盘并
`fsync`，不管它跨了多少键、多少条记录。`upsert()` 和 `upsert_many()` 都走这个函数；一批 200 条
记录是一个持久单元，不是 600 次 fsync。

由此得出一个后果：持久边界是**按调用**划的，不是按记录。想让两条记录有不同的持久级别，就得
发两次调用。同样，批次执行到一半没有任何东西可见——帧都还在缓冲里，一条 `p:` 都还没落地。

### 维度也是这个批次的一部分

```python
def _admit(self, vector):
    query = as_query(vector)
    width = int(query.shape[0])
    if self._dim is None:
        self._dim = width
        self._storage.put_raw(_DIM_KEY, str(width).encode("utf-8"), sync=False)
        return query
    if self._dim != width:
        raise ValueError(
            f"vector of width {width} does not match the collection width {self._dim}"
        )
    return query
```

collection 里第一条向量把宽度定下来，`_DIM_KEY` 以 unsynced 写进同一批。批次没提交，宽度也就
没确立——内存里的 `_dim` 和磁盘上的元数据不会各说各话。此后每条向量都必须对上这个宽度。

## 三、`delete()` 是镜像写法

```python
def delete(self, vector_id):
    payload = self._storage.get_payload(vector_id)
    if payload is None and self._storage.get_vector_blob(vector_id) is None:
        return False
    if payload is not None:
        self._storage.delete_payload(vector_id, sync=False)
        self._drop_index_keys(vector_id, payload, sync=False)
    else:
        self._drop_unknown_index_keys(vector_id, sync=False)
    self._storage.delete_vector(vector_id, sync=True)
    return True
```

先 payload，再倒排条目，最后才以 `sync=True` 删向量。这个顺序是挑过的：删到一半被打断，留下的
是**有向量没 payload**——一个 `reclaim()` 认得的、已知的写残状态——而不是「有 payload 没向量」，
后者按写入契约根本不该出现。若 payload 已经没了，`_drop_unknown_index_keys` 会按字段遍历，
而不是试图从缺失的 payload 里反推值段。

`_apply_index_delta` 落到 `ScalarIndex.reindex()`，diff 新旧 payload：改一个属性，就只删掉那几条
旧 `idx:`、只写那几条新的，全程 unsynced。

## 四、`reclaim()`：把树拉回「记录就是 payload」

```python
def reclaim(self) -> RepairReport:
    vector_ids = set(self._storage.iter_vector_ids())
    payloads = dict(self._storage.iter_payloads())
    payload_ids = set(payloads)

    report = RepairReport(
        orphan_vectors=len(vector_ids - payload_ids),
        orphan_payloads=len(payload_ids - vector_ids),
    )
    for orphan in sorted(vector_ids - payload_ids):
        self._storage.delete_vector(orphan)
        report = _bump(report, stale_index_keys=self._drop_unknown_index_keys(orphan))
    for orphan in sorted(payload_ids - vector_ids):
        self._storage.delete_payload(orphan)
        report = _bump(report, stale_index_keys=self._drop_unknown_index_keys(orphan))

    audit = self._scalar.repair()
    return _bump(report,
                 stale_index_keys=len(audit.orphaned),
                 rebuilt_index_keys=len(audit.missing))
```

`RepairReport` 有四个计数和一个 `total`：

| 计数 | 含义 |
| --- | --- |
| `orphan_vectors` | 写残——有 `v:` 没 `p:` |
| `orphan_payloads` | 有 `p:` 没 `v:`；按写入顺序不可能，但照样报出来 |
| `stale_index_keys` | 指向已不再提交的 id 的 `idx:` 条目 |
| `rebuilt_index_keys` | 某条已提交记录本该有、却缺了的条目 |

倒排索引是已提交 payload 的纯函数，所以写残清掉之后，`ScalarIndex.repair()` 会做**双向** diff：
多余的条目删掉，缺的补回来。这里没有「磁盘上有什么就信什么」的分支。`VectorEngine.open()`
在交出句柄之前先调 `reclaim()`，因此调用方永远看不到写残的记录。

## 五、`recover()`：先回收，再决定 IVF 索引还值不值得留

`recover()` 以 `repair = engine.reclaim()` 开头，把计数带进 `RecoveryReport`，然后不写任何东西地
把 IVF 索引量一遍：

```python
health = inspect_ivf(storage, metric=engine.metric, dim=dim)
if health.needs_retrain(rebuild_ratio):
    if health.vectors == 0:
        clear_namespace(storage, POSTING_PREFIX)
        clear_namespace(storage, CENTROID_PREFIX)
        storage.delete_raw(IVF_META_KEY, sync=False)
        return _bump(report, retrained=True)
    rebuild_ivf(storage, metric=engine.metric, dim=dim,
                partitions=min(partitions, health.vectors), seed=seed)
    ...
```

`IndexHealth` 只报告哪里不对，不动手：

| 字段 / 属性 | 含义 |
| --- | --- |
| `trained`、`partitions`、`vectors`、`postings` | 索引和语料各有什么 |
| `missing` | 已提交、却在任何桶里都找不到的 id |
| `stale` | 没被提交的桶内 id——删到一半或重建被打断留下的 |
| `duplicated` | 出现在多个分区的 id；一律触发重建 |
| `malformed` | 存下来的镜像不是 `dim` 个 float，或元数据解不开 |
| `drift` | `missing + stale + duplicated` |
| `healthy` | 无损坏、无漂移；空语料只有在**没有**索引时才算健康 |
| `needs_retrain(ratio)` | 漂移大到该重算，而不是补一补 |

`needs_retrain` 卡得很严：`malformed`、元数据缺失、`dim` 或 `k` 对不上、出现任何 `duplicated`，
或者漂移超过 `rebuild_ratio * vectors`，都会强制整体重训。否则 `recover()` 加载索引，只把差量
补上——`stale` 逐个 `remove()`，`missing` 逐个 `add()`。

有两个细节让它可以每次 open 都安全调用。健康的库给出一份**全零**的 `RecoveryReport`；而
`train=False` 时，没训练过或漂移严重的索引会被原样留着，一次只读检查不会触发重建。

## 六、测试怎么把这条不变量钉住

「起几个线程一直跑到出事」这种测试，证明不了哪些中间状态是合法的。`tests/test_concurrent.py`
先把不变量写出来，再强制交错：

- **用 Barrier，不用 sleep。** `threading.Barrier` 让所有参与者同一瞬间放行，于是「写在读者看之前
  就开始了」是测试的属性，而不是对调度器的一厢情愿。
- **工作量有界。** 读者跑固定轮数，写者做固定次数的变更；需要看到写后状态的读者等在 `Event`
  上，那是同步，不是拖时间。
- **线程里抛异常要算失败。** Python 会把线程里的异常报成 warning 然后让测试因为错误的原因
  通过，所以每个参与者都套一层守卫：记下失败并打破 barrier。

`tests/test_chaos.py` 在五个结构不同的位置杀进程——两次确认过的记录之间、一个 `upsert_many`
内部（整批共用一条 fsync 边界）、IVF 重建中、payload-first 与 vector-last 之间、以及恢复过程
本身——然后在**同一个库上连杀五次**，因为「第一次崩能恢复、第四次不能」的库不算库。每次都是
真的 `SIGKILL`：没有 handler，没有 `finally`，没有运行时刷缓冲。断言很窄也很硬：库能重开、
断点前已确认的没丢、每条活下来的向量都有 payload 且每条 payload 都有向量、倒排索引与 payload
一致、派生索引收敛。不断言具体剩几条——剩多少由存储引擎说了算，但「每一条都是完整的」不由它
说了算。

[ADR-002](../decisions/adr-002-payload-commit-record.zh-CN.md) 点明了为什么要这么测：任何一处
把 payload 写成不带 `sync=True`，或者让别的键来决定可见性，都会悄悄破坏不变量。返回值抓不到
这种错误，这些测试抓得到。

## 七、失效方式

| 踩法 | 实际发生什么 |
| --- | --- |
| 每个暂存的键都 `sync=True` | 结果没错，但 fsync 多了三倍，整批也不再是一个持久单元。 |
| 最后一条 payload 用 `sync=False` | 调用返回时什么都不持久，崩一次就丢掉已确认的写。 |
| 拿 `v:` 当可见性判据 | 写残会变成可见记录，`reclaim()` 再也分不清它们。 |
| 先删向量再删 payload | 删到一半会留下有 `p:` 没 `v:`，正是契约禁止的那个状态。 |
| open 时跳过 `reclaim()` | 调用方能看到写残，`RepairReport` 也没机会收敛。 |
| 把「健康」当成「索引已训练」 | 空语料恰恰是在没有索引时才算健康。 |

## 八、新增持久写路径的检查清单

1. 先定下哪个键代表「已提交」，然后让所有人都用它。
2. 派生键 unsynced 暂存；只在提交记录上刷一次盘。
3. 首次写入顺带确立的元数据，放进同一批，同样 unsynced。
4. delete 把顺序镜像过来，让写残删操作落到已知的孤儿状态，而不是造出新状态。
5. 派生索引按已提交集合的纯函数重建，两个方向都 diff。
6. 不可能出现的状态要报出来，不要当成不可能就略过。
7. 用 Barrier 和真 `SIGKILL` 测，断言的是完整性而不是条数。

## 接着读哪儿

| 想看的事 | 出处 |
| --- | --- |
| 写路径与删除路径 | [`engine.py`](../../src/sable_vector/engine.py) |
| 提交边界的决策 | [ADR-002](../decisions/adr-002-payload-commit-record.zh-CN.md) |
| 重建与索引健康 | [`recovery.py`](../../src/sable_vector/recovery.py) |
| 倒排索引双向 diff | [`index/scalar.py`](../../src/sable_vector/index/scalar.py) |
| 线程下的方向性不变量 | [`tests/test_concurrent.py`](../../tests/test_concurrent.py) |
| 五个断点、连崩五次 | [`tests/test_chaos.py`](../../tests/test_chaos.py) |
| 上游 WAL 与截断恢复 | [`sable`](https://github.com/H-I-F/sable) `docs/skills/wal-and-recovery.md` |
