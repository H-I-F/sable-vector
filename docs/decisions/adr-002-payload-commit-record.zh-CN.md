[English](adr-002-payload-commit-record.md) | [简体中文](adr-002-payload-commit-record.zh-CN.md)

# ADR-002：payload 就是提交记录

**状态：** 已接受 —— 实现在 `src/sable_vector/engine.py`。

## 背景

`sable` 没有原子多键 batch：**每次 `put` 都拿到自己的 sequence**。而 sable-vector 的一条记录是
好几个键（见 ADR-001），reader 绝不能看到半条。所以引擎得用存储引擎唯一给得出手的那个东西——
每次写入的 `sync` 标志，决定 WAL 帧是否刷盘 `fsync`——自己造一个提交边界。

## 决策

**payload 就是提交记录。** `p:<id>` 存在，记录才算已提交。写路径是：

```
v:<id>       float32 镜像     (sync=False)
idx:...      倒排条目         (sync=False)
p:<id>       payload          (sync=True)   <- 记录在这里变得可见
```

`sync=False` 把帧留在 WAL buffer 里；payload 的 `sync=True` 用一次 `fsync(2)` 把整段刷下去，不管
这段跨了多少个键。`upsert_many()` 在批次尺度上同理：把每条记录的每个键都暂存好，再提交 payload，
只让最后一个把日志推到稳定存储。

`delete()` 是镜像写法：不带 sync 地删掉 payload 和倒排条目，最后以 `delete_vector(sync=True)` 收尾。

`reclaim()` 也用同一条规则定义撕裂写——有向量没 payload，或有倒排条目却没有已提交的 id，都算
孤儿派生数据。

## 后果

upsert 中途被杀，留下的要么什么都没有，要么是一条完整记录。reader 可以把「payload 在不在」当成
唯一的可见性判据，`recovery.py` 则把派生索引当作已提交 payload 的纯函数重建出来。

边界是「每条记录」或「每批」，不是「每个键」。想给不同记录不同持久性，就得分别调用。任何不带
`sync=True` 写 payload、或者让别的键来决定可见性的改法，都会悄悄废掉这条不变量——所以
`tests/test_upsert.py` 和 SIGKILL 混沌套件是直接断言它，而不是查一个返回值。
