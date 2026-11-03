[English](zero-copy-codec.md) | [简体中文](zero-copy-codec.zh-CN.md)

# 零拷贝向量：编解码与分数方向

> 越过存储边界的有两种表示，想要的东西正好相反：向量在每次查询的每个候选上都要读，
> payload 只在每个幸存结果上读一次。这篇顺着 `codec.py` 和 `distance.py` 走一遍，讲清楚
> 它们守住的那些不变量，以及调用方把其中任何一个想得太宽容时会撞上的失效方式。

## 一、格式就是布局，不是一段头部

1536 维 embedding 是 6144 字节的小端 float32：

```python
ITEM_SIZE: int = VECTOR_DTYPE.itemsize      # 4
def vector_bytes(dim: int) -> int:
    return dim * ITEM_SIZE
```

没有帧、没有长度、没有分隔符，也没有版本字节。理由朴素但很硬：头部要在扫描碰到的每一行上
解析一次，分隔符还得转义。字节序和 dtype 在两端都钉死（`VECTOR_DTYPE` 是 `<f4`），所以一台机器上
写出的镜像，在另一台机器上不需要头部来解释布局就能解码。布局属于格式本身。

`VectorRecord.__post_init__` 在入口处就做归一化，下游每个核看到的都是同一种东西：

```python
array = np.ascontiguousarray(self.vector, dtype="<f4")
if array.ndim != 1:
    raise ValueError(f"vector must be one-dimensional, got shape {array.shape}")
if array.size == 0:
    raise ValueError("vector must have at least one dimension")
```

标量、Fortran 序的切片、float64 列表，到这里统统变成一段连续的 `<f4` 向量。在边界上做这一次，
`encode_vector` 才能只是单独一句 `tobytes()`。

## 二、编码是一次 memcpy，解码是一个视图

```python
def encode_vector(vector: np.ndarray) -> bytes:
    return as_query(vector).tobytes()

def decode_vector(blob: bytes, *, dim: int | None = None,
                  writeable: bool = False) -> np.ndarray:
    _check_length(blob, dim, single_row=True)
    view = np.frombuffer(blob, dtype=VECTOR_DTYPE)
    return view.copy() if writeable else view
```

`encode_vector` 交出的就是数组自己那块内存，按规范序排好。`decode_vector` 是 `np.frombuffer`：
把 storage 层已经交回来的字节重新解释成数组，没有第二次分配。这就是本项目里「零拷贝」的全部
含义，而它有一个硬性前提——**默认视图是只读的**。

storage 层把它交出去的字节当成不可变。一个可写视图会让调用方改掉别的读者可能还钉着的缓冲区。
所以修改是显式开启的：

| 调用 | 结果 | 用在哪儿 |
| --- | --- | --- |
| `decode_vector(blob)` | 只读视图，指向 `blob` | 打分、比较、组装矩阵 |
| `decode_vector(blob, writeable=True)` | 私有副本 | 残差运算等要写入的场合 |

`ivf_cluster.py` 拿向量减质心算残差，需要可写数组，就传 `writeable=True`。其余各处都留着视图。

### 长度校验在重解释之前

`_check_length(blob, dim, single_row=...)` 先跑。字节数不是 float32 的整数倍，或者跟 `dim` 对不上，
一律 `ValueError`。这里刻意不做「猜一个维度再 reshape」：否则一个被截断的 SSTable 块，和一个
维度写错的 512 维向量，都会变成形状看着还挺合理的数组，直到很久以后以「排名莫名其妙」的形式
才暴露出来。

## 三、矩阵是一整块镜像，不是 N 份 blob

```python
def encode_matrix(matrix: np.ndarray) -> bytes:
    return as_matrix(matrix).tobytes()

def decode_matrix(blob: bytes, *, dim: int, writeable: bool = False) -> np.ndarray:
    _check_length(blob, dim, single_row=False)
    view = np.frombuffer(blob, dtype=VECTOR_DTYPE).reshape(-1, dim)
    return view.copy() if writeable else view
```

`n` 由长度推出来，`dim` 必须给。常驻的 `FlatIndex` 持有的 `(n, d)` 数组就是这条路来的，所以一次
查询是在连续块上做一次 gemv，而不是 N 次 Python 层解码。`search_streaming()` 按 `block_rows`
分块读 SSTable，一次解一块——同一个 `frombuffer` 技巧，只是内存有上界。

## 四、payload 走的是相反的那笔交易

```python
def encode_payload(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

def decode_payload(blob: bytes) -> dict[str, Any]:
    document = json.loads(blob.decode("utf-8"))
    if not isinstance(document, dict):
        raise ValueError("payload must decode to a JSON object")
    return document
```

键排序、分隔符压紧，让相等的 payload 产出相等的字节；这样 payload 可以比较、可以哈希、可以
当作索引键的一部分，而不必先解析回来。`decode_payload` 认死必须是 JSON 对象：存进去一个列表
或字符串属于损坏，不是「形状不太常见的 payload」。`ensure_ascii=False` 让非 ASCII 属性在 shell 里
直接读得懂。

模块 docstring 把这笔交易写得很直白：payload 付得起一次解析，向量付不起，因为 payload 每个
幸存结果才读一次，向量每个候选都要读一次。

## 五、三种度量共用一种分数方向

L2 是距离，cosine 与 dot 是相似度。与其让每个消费者都按 `metric=` 分一支，`distance.py`
直接给出两种方向恒定的视图：

```python
def distance_matrix(matrix, query, metric):   # 恒为越小越近
    if metric is DistanceMetric.L2:
        return l2_distances(matrix, query)
    if metric is DistanceMetric.COSINE:
        return -cosine_similarities(matrix, query)
    return -dot_scores(matrix, query)

def score_matrix(matrix, query, metric):      # 恒为越大越近
    if metric is DistanceMetric.L2:
        return -l2_distances(matrix, query)
    if metric is DistanceMetric.COSINE:
        return cosine_similarities(matrix, query)
    return dot_scores(matrix, query)
```

`DistanceMetric` 是 `str` 值的枚举，度量穿过 JSON 配置不会走样——重新打开的 collection 保持着
创建时的几何。三个属性把其余规则编码进去：

| 属性 | `l2` | `cosine` / `dot` | 用途 |
| --- | --- | --- | --- |
| `higher_is_better` | `False` | `True` | `top_k` 的方向 |
| `worst_score` | `+inf` | `-inf` | 累加器的中性初值 |
| `normalised` | `False` | cosine 为 `True` | 是否投影到单位球面 |

`Match` 刻意同时留着两个数：`score` 是定向后的排名键，`distance` 是给需要几何量的调用方的
原始值。把两者搞混，是最容易造出「看着还挺合理、其实反了」的排名的方式。

### 存储 float32，为什么计分用 float64

存储用 float32 是为了让持久化镜像紧凑。但 `top_k()` 在挑选之前会把分数转成 `SCORE_DTYPE`
（float64）：

```python
values = np.asarray(scores, dtype=SCORE_DTYPE)
directed = -values if larger_is_better else values
if k >= values.size:
    order = np.argsort(directed, kind="stable")
    return order, values[order]
candidates = np.argpartition(directed, k - 1)[:k]
ranked = candidates[np.argsort(directed[candidates], kind="stable")]
```

`argpartition` 是 O(n) 而不是 O(n log n)——n 是全库、k 是十的时候，这个差别很实在。只有幸存的
k 个被完整排序，而且排序是**稳定**的，同分保持插入序。累加放在 float64 上，避免咬得很近的
两个分数被 float32 舍入判了胜负；稳定排序再保证完全同分不会被哈希种子判了胜负。
[ADR-004](../decisions/adr-004-float32-codec-and-score-orientation.zh-CN.md) 把这两处都记作刻意选择。

## 六、失效方式

| 踩法 | 实际发生什么 |
| --- | --- |
| 往解码出来的视图里写 | 数组只读，赋值直接抛；要写就传 `writeable=True`。 |
| 解码时猜 `dim` | `_check_length` 抛错，不会 reshape 成形状不对的矩阵。 |
| 存了非 dict 的 payload | 重新打开时 `decode_payload` 抛 `ValueError`。 |
| 拿 `distance` 排序当成 best-first | `distance` 是越小越近；排名用的是 `score`。 |
| 对同分用不稳定排序 | 两次相同的跑法可能给出不同次序，bug 报告就复现不出来了。 |
| 只改一端的 dtype 常量 | 改动前写的镜像解出来是乱码；decode 时会对 dtype 做断言。 |

## 七、新增编解码路径的检查清单

1. 只在边界处归一化成连续 `<f4`，一次就够。
2. `frombuffer` 之前先校长度，绝不从被截断的 blob 里推 `dim`。
3. 默认返回只读视图；要副本就显式、单独开口子。
4. payload 的 JSON 保持规范化——键排序、分隔符压紧、只认对象。
5. 分数方向只在 `score_matrix` 里定一次，让所有消费者都跟几何无关。
6. float64 累加、`argpartition` 挑选，幸存者再做稳定排序。
7. decode 时断言 dtype，让布局变更响亮地失败。

## 接着读哪儿

| 想看的事 | 出处 |
| --- | --- |
| 编解码 | [`codec.py`](../../src/sable_vector/codec.py) |
| 距离核与 top-k | [`distance.py`](../../src/sable_vector/distance.py) |
| 类型与 `Match` | [`types.py`](../../src/sable_vector/types.py) |
| 决策记录 | [ADR-004](../decisions/adr-004-float32-codec-and-score-orientation.zh-CN.md) |
| 字节镜像与 dtype 测试 | [`tests/test_codec.py`](../../tests/test_codec.py)、[`tests/test_distance.py`](../../tests/test_distance.py) |
| 为什么编解码定义了 recall 的基准 | [实测报告](../reports/benchmark-2026-11.zh-CN.md) |
