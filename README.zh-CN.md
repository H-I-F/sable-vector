<p align="center">
  <a href="./README.md"><img src="https://img.shields.io/badge/English-docs-blue.svg" alt="English docs"></a>
  <a href="./README.zh-CN.md"><img src="https://img.shields.io/badge/简体中文-docs-blue.svg" alt="简体中文文档"></a>
</p>

<h1 align="center">sable-vector</h1>

<p align="center">
  <b>面向本地 AI Agent 运行时的嵌入式混合检索向量引擎。</b><br/>
  持久化由 <a href="https://github.com/MKA930/sable">sable</a> LSM 提供崩溃安全 · NumPy 矩阵运算保证精确召回 · 标量倒排索引实现前置/后置过滤。
</p>

<p align="center">
  <a href="https://github.com/MKA930/sable-vector/releases"><img src="https://img.shields.io/github/v/release/MKA930/sable-vector" alt="Release"></a>
  <a href="https://github.com/MKA930/sable-vector/actions/workflows/ci.yml"><img src="https://github.com/MKA930/sable-vector/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/storage-sable%20LSM-blue.svg" alt="Storage">
  <a href="https://github.com/MKA930/sable"><img src="https://img.shields.io/badge/built%20on-sable-8A2BE2.svg" alt="Built on sable"></a>
</p>

---

sable-vector 是一个单进程、轻依赖的向量引擎，面向**已经拥有自有 embedding 模型**、并希望向量能**在产出它们的进程崩溃后依然存活**的 Agent。它把三件事严格拆开：

- **持久化**交给 [`sable`](https://github.com/MKA930/sable)——一个纯 Python 的 LSM 引擎。每次写入先落 WAL 再可见，因此被强杀的进程只会丢未确认的写，不会丢已提交的数据。
- **几何计算**交给 NumPy。相似度就是矩阵–向量乘积，内核从不在 Python 层对向量做循环。
- **过滤**交给标量倒排索引，它被实现成普通的排序键，于是元数据谓词变成一次前缀扫描，而非全表遍历。

它不开 socket、不在你背后起线程、也不内嵌任何模型。它是库，不是服务。

## 特性

- **混合检索**——通过两种显式策略把稠密向量相似度与标量元数据过滤结合：*前置过滤*（先由倒排索引裁剪候选集再打分）与*后置过滤*（先取全局 Top-N 再做谓词判定）。
- **崩溃安全写入**——向量、payload 与倒排条目作为一个持久化单元写入；恢复时在 `open` 阶段从 WAL/SSTable 重建内存索引。
- **精确而非近似**——Flat 索引对全部候选打分，因此召回率按构造为 100%；同时提供 IVF 聚类用于对更大规模集合做分区剪枝。
- **标准适配器**——实现 LangChain 风格的 `add_texts` / `similarity_search` 协议，可完全离线运行（使用本地 NumPy embedding）。
- **零运行时服务**——无服务器、无网络、无外部向量库。

## 架构

引擎是单棵 `sable` LSM 树之上的一层薄而带类型的门面。向量、payload、标量索引、IVF 质心与倒排列表都存在于同一个键空间，仅靠键前缀区分——因此只有一份 WAL、一份 manifest、一套崩溃恢复逻辑。

### 组件分层

```mermaid
flowchart TB
    APP[Python 调用方] --> ADP["adapter/ — VectorStore 协议"]
    APP --> QRY["query/ — 前置 / 后置过滤执行器"]
    ADP --> QRY
    QRY --> IDX["index/ — Flat · 标量倒排 · IVF"]
    QRY --> STO["storage.py — 命名空间适配器 v:/p:/idx:/c:/b:"]
    IDX --> STO
    STO --> ENG["engine.py — 持久化 upsert / delete / reclaim"]
    ENG --> SABLE[("sable LSM — WAL + memtable + SSTables")]
    REC["recovery.py — 崩溃重建"] -.-> IDX
    REC -.-> ENG
```

### 写入与读取路径

```mermaid
flowchart TB
    subgraph write["写入路径 (VectorEngine)"]
        direction TB
        U["upsert / delete"] --> A["写入向量 v:<id> · sync=False"]
        A --> B["写入索引 idx:<field>:<val>:<id> · sync=False"]
        B --> C["提交 payload p:<id> · sync=True"]
        C --> WAL[("sable WAL + memtable")]
    end
    
    WAL --> SST[("SSTables")]
    WAL --> IDX[("VectorStorage 命名空间适配器")]
    
    %% 连线标签含特殊字符，需加双引号
    IDX -->|"v:"| VEC["float32 向量"]
    IDX -->|"p:"| PAY["JSON payload"]
    IDX -->|"idx:"| INV["标量倒排索引"]
    IDX -->|"c:/b:"| IVF["IVF 质心 / 倒排列表"]

    subgraph read["读取路径"]
        direction TB
        Q["query 向量 + 谓词"] --> PF{"前置 还是 后置 过滤?"}
        PF -->|前置| PRE["ScalarIndex.matching_ids → 候选集"]
        PF -->|后置| POS["FlatIndex top-N → 谓词判定"]
        PRE --> FLAT["FlatIndex.search(candidate_ids)"]
        POS --> ATT["回填 payload"]
        FLAT --> M["Matches: score + distance + payload"]
    end
    
    %% 跨子图虚线连接
    VEC -.-> FLAT
    PAY -.-> ATT
    INV -.-> PRE
```

### 混合检索执行计划

两种过滤策略不可互换；引擎同时暴露两者，由调用方选择自己能容忍其失败模式的那种。

```mermaid
sequenceDiagram
    participant App
    participant Exec as 查询执行器
    participant Scalar as ScalarIndex
    participant Flat as FlatIndex
    participant Store as VectorStorage

    App->>Exec: search(query, k, predicate)
    alt 前置过滤（谓词较窄）
        Exec->>Scalar: matching_ids(predicate)
        Scalar-->>Exec: 候选集
        Exec->>Flat: search(query, k, candidate_ids)
        Flat->>Store: 读取候选的 v: 行
        Flat-->>Exec: 打分后的邻居
    else 后置过滤（谓词较宽 / 无索引）
        Exec->>Flat: search(query, k=N)
        Flat-->>Exec: 全局 Top-N
        Exec->>Store: 逐个取 p: payload
        Exec->>Exec: 丢弃不满足谓词的行
    end
    Exec-->>App: PrefilterOutcome / PostfilterOutcome (Matches)
```

## 快速开始

```python
import numpy as np
from sable_vector.engine import VectorEngine
from sable_vector.query.prefilter import PrefilterExecutor
from sable_vector.index.scalar import ScalarIndex
from sable_vector.index.flat import FlatIndex

engine = VectorEngine.open(
    "collection.sv",
    metric="cosine",
    indexed_fields=("tenant", "lang"),
)
engine.upsert("doc-1", np.random.randn(1536).astype("float32"),
              payload={"tenant": "acme", "lang": "en"})
engine.upsert("doc-2", np.random.randn(1536).astype("float32"),
              payload={"tenant": "acme", "lang": "zh"})

flat = FlatIndex.materialise(engine, metric="cosine")
scalar = ScalarIndex(engine.storage, ("tenant", "lang"))
executor = PrefilterExecutor(engine.storage, flat, scalar)

outcome = executor.search(np.random.randn(1536).astype("float32"), k=5,
                          predicate={"tenant": "acme"})
for match in outcome.matches:
    print(match.id, round(match.score, 4), match.payload)
```

### LangChain 风格适配器（离线）

```python
from sable_vector.adapter.langchain import SableVectorStore

store = SableVectorStore.from_texts(
    texts=["local retrieval", "hybrid search"],
    embedding=lambda texts: [np.random.randn(1536).tolist() for _ in texts],
    collection_home="rag.sv",
    metric="cosine",
    indexed_fields=("kind",),
)
hits = store.similarity_search("local retrieval", k=2)
```

适配器从不调用远程 embedding API：传入一个本地可调用对象返回 NumPy / List[float] 向量，完整的 `add_texts` → `similarity_search` 闭环即可在单机端到端跑通。

## 使用指南

### 选择距离度量

`metric=` 接受 `cosine`、`l2` 或 `dot`。所有内核都会把分数统一调整为**越大越相近**，因此下游代码无需知道当前用的是哪种几何即可比较与排序。

| `metric=` | 度量 | 典型场景 |
| --- | --- | --- |
| `cosine` | 单位球面上的角度距离 | 已归一化的文本 embedding（最常见） |
| `l2` | 欧氏距离 | 未归一化的原始特征向量 |
| `dot` | 负内积 | 最大内积检索 / 幅值本身有语义的检索 |

### 选择过滤策略

两个执行器入参与返回值（均为 `Matches` 列表）一致，区别只在谓词施加的位置，而这决定了各自的失败模式。

| | `PrefilterExecutor` | `PostfilterExecutor` |
| --- | --- | --- |
| 做法 | 先由 `ScalarIndex` 收窄候选集，再由 `FlatIndex` 只对幸存者打分 | 先由 `FlatIndex` 算全局 Top-N，再用谓词丢弃行 |
| 适用 | 谓词**选择性强**（`tenant`、`kind`、`lang`）且字段已建索引 | 谓词**较宽**，或语料小到可以承受更宽的扫描 |
| 失败模式 | 只认带索引前缀的键——缺少索引条目的行对它不可见 | 可能返回少于 `k` 条，因为全局 Top-N 里的近邻可能全部被谓词淘汰 |

字段必须先传入 `VectorEngine.open(indexed_fields=...)`，前置过滤才能使用它。索引字段是写路径上的元数据而非 schema：后续新增字段需要对已有语料做一次重建。

### 用 IVF 做近似检索

Flat 索引对全部候选打分，结果精确，在数万条以内完全够用。超过这个量级就训练一个 IVF 索引，把检索剪枝到离查询最近的若干分区，并逐步加大 `nprobe` 直到召回可接受：

```python
from sable_vector.index.ivf_storage import IvfIndex
from sable_vector.types import DistanceMetric

index = IvfIndex.train(
    engine.storage,
    metric=DistanceMetric.COSINE,
    dim=1536,
    partitions=32,
)
outcome = index.search(query, k=10, nprobe=4)
print(outcome.matches, outcome.buckets_scanned)
```

`nprobe=1` 只读一个分区、速度最快；`nprobe == partitions` 则退化为全量扫描。基准脚本会输出延迟/召回曲线，便于你用自己的数据选工作点：

```bash
python -m benchmarks.bench_vector --dim 1536 --sizes 10000
```

### 持久化与维护

每次变更都以 unsynced 方式暂存，最后靠 payload 提交完成持久化，因此 upsert 过程中被杀只会留下"什么都没有"或"完整记录"，绝不会留下没有 payload 的孤儿向量。打开集合时会自动从 WAL 与 SSTable 重建内存索引。

如果你希望手工对账派生数据（撕裂写入、上一次崩溃残留的条目），一次扫描即可：

```python
report = engine.reclaim()
print(report.orphan_vectors, report.orphan_payloads)
```

## 安装

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install "sable @ git+https://github.com/MKA930/sable.git@v0.1.1"
pytest -q
```

## 容器

```bash
docker build -t sable-vector .
docker run --rm sable-vector pytest -q          # 校验镜像构建
```

## 目录结构

| 路径 | 职责 |
| --- | --- |
| `src/sable_vector/engine.py` | 持久化 upsert / delete / reclaim |
| `src/sable_vector/storage.py` | `sable` 命名空间适配器（`v:` / `p:` / `idx:`） |
| `src/sable_vector/distance.py` | NumPy L2 / cosine / dot 内核 |
| `src/sable_vector/codec.py` | float32 零拷贝编解码 |
| `src/sable_vector/index/` | Flat、标量倒排、IVF 聚类与倒排列表 |
| `src/sable_vector/query/` | 前置与后置过滤执行器 |
| `src/sable_vector/adapter/` | `VectorStore` 协议适配器 |
| `src/sable_vector/recovery.py` | 崩溃重建与索引健康 |

## 文档与治理

- 架构与用法：本中文文档与 [English](./README.md)
- 贡献指南：[CONTRIBUTING.md](./CONTRIBUTING.md)
- 安全策略：[SECURITY.md](./SECURITY.md)
- 发布历史：[CHANGELOG.md](./CHANGELOG.md)
- 演进台账（每个 PR 一个勾）：[PROGRESS.md](./PROGRESS.md)
- v0.1.0 发布公告：[docs/RELEASE_NOTES_v0.1.0.md](./docs/RELEASE_NOTES_v0.1.0.md)

## 许可证

[MIT](./LICENSE)。
