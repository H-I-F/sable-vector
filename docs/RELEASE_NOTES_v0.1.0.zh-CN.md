[English](./RELEASE_NOTES_v0.1.0.md) | 简体中文

# sable-vector v0.1.0

**给本地 AI agent runtime 用的嵌入式混合向量检索。** 2026 年 10 月 31 日。

做 agent 的团队大多会撞上同一堵墙：agent 记得住东西，但记忆跟着进程一起死。向量放在内存里，
索引是个 `dict`，一重启就得把已经花钱算过的 embedding 重算一遍。换成托管向量数据库确实解决了
持久化，代价是把语料交给一个需要运维、加固、付钱的服务——而你的负载往往只是一台笔记本加几十万
条向量。

`sable-vector` 走的是第三条路：embedding 留在本地，**从构造上就保证持久**，并且让元数据参与检索，
而不是等检索完了再在 Python 里过滤一遍。

## 为什么这里非得做混合检索

稠密相似度回答的是「什么东西长得像这个」。但 agent 几乎从来不想要「最近的邻居」，它想要的是
**在我的租户里、这个语言、这类文档、并且排除掉我已经撤销的那些之后，最近的邻居**。纯向量检索表达
不了这个；常见的绕法是取一个很大的缓冲集，再在应用代码里过滤，这会悄悄拉低 recall，还让延迟
取决于你的谓词到底有多窄。

sable-vector 把元数据过滤当成查询的一等公民，给了两个明明白白分开的策略——`PrefilterExecutor`
（打分开先用倒排索引剪掉候选集）和 `PostfilterExecutor`（先取全局 top-N，再查谓词）。两者不能互换：
各有各的代价曲线，也各有各的失效方式，该用哪个取决于谓词的选择性。把两个都摆出来、把取舍写清楚，
就是这个设计的用意。

## v0.1.0 里有什么

**持久性。** 建在 [`sable`](https://github.com/H-I-F/sable) 这个纯 Python LSM 引擎之上。一条向量、
它的 payload 和倒排条目作为一个整体提交；每次变更前都会先落到预写日志才可见，所以被杀掉的进程
只会丢掉尚未确认的写入，绝不会丢已提交的数据。重新打开集合时，内存索引从 WAL 和 SSTable 重建，
SIGKILL 混沌套件在多个注入点上验证的正是这条性质。

**几何计算。** 相似度就是 float32 向量上的矩阵–向量乘。核函数从不在 Python 里逐行循环，因此 recall
是精确值而非近似值，第一次查询之前也不需要训练步骤。

**过滤。** 元数据以普通有序键存放，所以一次集合归并就是前缀扫描，不必走全表。索引字段按字段声明，
声明后才能被前置过滤路径用上。

**扩容路径。** 当精确检索的代价吃不消时，训练一个 IVF 索引，用 `nprobe` 把 recall 买回来。基准脚本
会打印延迟 / recall 曲线，于是工作点是对自己数据的测量，而不是拍脑袋。

**集成。** 一个 `VectorStore` 风格的适配器实现 `add_texts` / `similarity_search`，配一个本地
embedding callable 就能完全离线跑通——不需要 API key，不联网。作为库，它也就意味着没有服务端：
背后不会偷偷开 socket 或起线程。

**工具链。** Python 3.10 与 3.11 上的 CI（ruff + pytest）、一个自校验的多阶段容器镜像（默认命令
就跑测试套件）、一份带固定种子的 1536 维基准，以及每合并一个 PR 打一个勾的单步演进台账。

## 上手

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install "sable @ git+https://github.com/H-I-F/sable.git@v0.1.1"
pytest -q
```

```python
import numpy as np
from sable_vector.engine import VectorEngine
from sable_vector.index.flat import FlatIndex
from sable_vector.index.scalar import ScalarIndex
from sable_vector.query.prefilter import PrefilterExecutor

engine = VectorEngine.open("collection.sv", metric="cosine",
                          indexed_fields=("tenant",))
engine.upsert("doc-1", np.random.randn(1536).astype("float32"),
              payload={"tenant": "acme"})

executor = PrefilterExecutor(
    engine.storage,
    FlatIndex.materialise(engine.storage, metric="cosine", dim=engine.dim),
    ScalarIndex(engine.storage, ("tenant",)),
)
outcome = executor.search(np.random.randn(1536).astype("float32"), k=5,
                          predicate={"tenant": "acme"})
```

## 老实说清楚的限制

- **单进程。** 同一时间只有一个 writer；这是嵌入式引擎，不是分布式存储。
- **1.0 之前。** 布局和 API 仍可能变；CHANGELOG 里写明了这一点，不让你在恢复时才撞上。
- **前置过滤只看得到已索引字段。** 字段声明之前写进去的行，在语料重建前对该谓词不可见。
- **不内置 embedding 模型。** 自己带 callable 来，这是刻意的。

## 致谢

v0.1.0 还捎带了一次上游贡献：排查 sable-vector 快照测试里的并发失败时，一路追到了
[`sable`](https://github.com/H-I-F/sable) ——它的 memtable 点查与范围扫描对「一个固定的 sequence
到底该解析出什么」会给出不同答案。该问题已在 `sable` v0.1.1 修复，本版本依赖的就是它。

MIT 许可。
