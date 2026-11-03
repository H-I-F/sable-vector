[English](./cross-layer-bugfix.md) | 简体中文

# 跨层排障：一次快照读悄悄丢掉了向量

一个关于 [`sable`](https://github.com/H-I-F/sable)（本项目的存储引擎）的调试故事，讲的是
sable-vector 里两个挂掉的测试，最后怎么落到别人家 memtable 里一段话的修复上。

## 一句话概括

| | |
| --- | --- |
| **症状** | `tests/test_concurrent.py` 里两个测试稳定失败（100% 复现，不是 flaky） |
| **根因** | `sable` 的 memtable 点查只返回某个键*最新*的那个版本，于是被快照钉住的读没法回退到更旧、但仍然可见的版本 |
| **修复** | `MemTable.get_visible()` —— 沿着版本链走，停在第一个不高于视图 sequence 的版本上，跟范围扫描本来就在做的事保持一致 |
| **发布为** | [`sable` v0.1.1](https://github.com/H-I-F/sable/releases/tag/v0.1.1)，由 sable-vector 固定引用 |
| **为什么要紧** | 这不是测试写错了，下游也绕不过去——一个偶尔找不到已提交键的持久化引擎，就不叫持久化 |

## 症状

sable-vector 的两个并发测试每次都挂：

- `test_readers_never_see_a_payload_without_its_vector` —— 并发 reader 看到了某条已提交记录的
  payload，它的向量却不见了。
- `test_snapshot_pins_are_returned_on_release` —— 引用了一个并不存在的可观测属性。

第二个只是表面问题（引擎得把那个属性暴露出来）。第一个是真的，而且值得为它把手头所有事都停下来：
这个项目的整个前提就是一条记录的 payload 和向量作为一个整体提交，所以 reader 要么什么也看不到，
要么看到一条完整记录。看到*半条*，就等于持久化承诺是假的。

测试是稳定失败的，这一点反而有用——flaky 的测试很容易被顺手隔离掉，而 100% 失败就是一份现成的
可复现 bug。

## 缩小范围

第一个要定的问题永远是「这是谁的 bug」。两个项目都在场，猜错方向要白耗好几天。于是把失败行为放到
**纯 sable** 里复现，完全不掺 sable-vector 的代码：

```python
# put K=A  -> 分到 sequence 1
# put K=B  -> 分到 sequence 4（同一个键，更新的版本）
# 快照钉在 sequence 2
view.get(b"K")   # 期望 b"A"（不高于 2 的最新版本）
                 # 实际：None
```

复现就这么多，结论毫不含糊：附近没有任何向量代码，这是存储引擎自己的行为。

另外两个观察把影响范围钉死了：

1. **只有 memtable 路径是错的。** `SSTableReader.get_entry` 早就会返回第一个
   `sequence <= visible_sequence` 的版本。flush 或 compaction 之后，同样的读就正常了。所以 bug
   只活在那张内存表里。
2. **只有在版本共存时才会触发。** 单版本的键解析得好好的。你得让同一个键的较新版本，和视图仍然
   有权读取的较旧版本并排待着——而这正是并发 writer 会造出来的局面。

## 根因

`sable` 的 memtable 按 `(user_key, -sequence)` 排序，所以一个键的最新版本排在它版本链的最前面。
这让「新的赢」成了结构性质，而不是一条约定。但读路径有两条，只有一条守了可见性规则：

- **范围扫描**（`scan` / `_versioned_scan`）会走遍*每一个*版本，按 `sequence <= visible_sequence`
  过滤，再取第一个幸存者。对的。
- **点查**（`MemTable.get`）直奔版本链的头，可见就返回，不可见就直接给 `None`——它停下了，
  没有继续往链下面走。

于是在快照钉在 sequence 2、而链头是个 sequence 4 的新版本、后面跟着本该可见的 sequence 1 版本时，
点查看见 sequence 4 那条、判定不可见、停下、把键报成不存在。对同一个键做扫描却会高高兴兴地返回
`A`。**两条读路径对一个固定 sequence 该解析出什么给出了不同答案**，这才是「坏了」的真正定义。

sable-vector 的症状之所以长成那样，原因也在这里：引擎把一条记录的各个部分写在不同 sequence 上，
而持有固定视图的并发 reader 刚好会落在它们中间。

## 修复

把 SSTable reader 早就有的可见性规则教给 memtable。在
[`sable`](https://github.com/H-I-F/sable) 里，`MemTable.get_visible`：

```python
def get_visible(self, user_key, visible_sequence):
    with self._mutation:
        node = self._descend_to(user_key)
        while node is not None:
            candidate = node.entry
            if candidate.user_key != user_key:
                break
            if visible_sequence is None or candidate.sequence <= visible_sequence:
                return candidate
            node = node.next[0]
    return None
```

从新到旧走版本链，返回第一个不高于视图 sequence 的条目。`visible_sequence is None` 仍然表示
「最新者胜」，所以普通的不带快照的读完全没变。

然后把点查指过去——`Engine._lookup_entry` 对 memtable 改调
`table.get_visible(user_key, visible_sequence)`，不再调 `table.get(user_key)`。

再补上另一个测试想要的可观测性，`Engine.open_snapshots`：

```python
@property
def open_snapshots(self) -> int:
    """Number of unreleased views, so callers can assert acquire/release balance."""
```

这条很小，但不是装饰：快照是一种会钉住文件、留住版本的资源。能断言一次 acquire/release 配平为零，
而不必伸手进引擎内部，正是让泄漏不至于在六周后变成一桩悬案的东西。

**验证。** 原本失败的两个 sable-vector 测试转绿，纯 sable 复现通过，`sable` 自己的套件保持
183 个测试通过、五项 CI 检查全绿，`sable-vector` 保持 374 passed / 1 xpassed——那个 `xpassed`
正是以前被预期为失败的并发回归用例。

## 为什么下游修不了

这点值得说透，因为「在你自己代码里兜一下」是最诱人的答案。`sable` 没有原子多键 batch：**每次
`put` 都拿到自己的 sequence**。sable-vector 先暂存向量、再暂存倒排条目、最后提交 payload，靠存储
引擎的可见性规则让这一串在 reader 眼里看起来是原子的。调用方没有任何 sequence 算术能绕开 memtable
的行为。诚实的选项只有两个：修引擎，或者文档里写明「并发快照读可能丢已提交的键」然后照样发版。

## 两个仓库之间怎么把它发出去

这一段最容易出错，因为两个项目会各自往前走：

1. 在 [`sable`](https://github.com/H-I-F/sable) 落地修复（PR #37），并写两个引擎级的回归测试，
   这样即便 sable-vector 哪天没了，不变量仍有人守着。
2. 把 `sable` 提到 **v0.1.1** 并打 tag——行为变了就得有版本号，CHANGELOG 里也点名了是哪个下游
   项目把它揪出来的。
3. 在 `sable-vector` 里把依赖从 `@main` 挪到已发布的 tag：

   ```toml
   sable = ["sable @ git+https://github.com/H-I-F/sable.git@v0.1.1"]
   ```

   对着 `@main` 构建，等于说「碰巧存在的那个提交」，这正是发布之后复现不了的原因。
4. 给 `sable` 的两份 README 加一节 `## Ecosystem` 指回这里，让两边都能发现这层关系。

当下那份完整记录也挂在 v0.1.0 公告里：[`docs/RELEASE_NOTES_v0.1.0.md`](../RELEASE_NOTES_v0.1.0.zh-CN.md)。

## 几点教训

- **同一个谓词实现两遍，早晚会对不上。** 这里是 `get` 和 `scan`，两者本该共同遵守的不变量是
  「不高于视图 sequence 的最大 sequence」。只有一边实现了，你就给同一个问题造了两个答案。
- **结构不等于语义。** 按 `(user_key, -sequence)` 排序让最新版本*排在前面*，这跟让它*可见*
  不是一回事。在有人来要一个既不是最新、也不是不存在的版本之前，那个捷径一直都好使。
- **去依赖里复现，别在自己的项目里复现。** 三行纯 sable 就定了归属。不这么做的话，最自然的
  猜测会是向量引擎的分段写入有 bug，调查就会从错误的仓库开始。
- **100% 失败的测试是份礼物。** 它是别人已经替你写好的确定性复现。修掉它，是去掉一笔长期的税；
  把它压下去不是。

## 链接

- [`sable`](https://github.com/H-I-F/sable) —— 本项目构建于其上的存储引擎
- [`sable` v0.1.1 发布页](https://github.com/H-I-F/sable/releases/tag/v0.1.1)
- [`sable-vector`](https://github.com/H-I-F/sable-vector) —— 本项目
