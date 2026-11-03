[English](./RELEASE_NOTES_v0.1.1.md) | 简体中文

# sable-vector v0.1.1

**给本地 AI agent runtime 用的嵌入式混合向量检索——现在可以通过 Model Context Protocol 直接对话。**
2026 年 10 月 31 日。

v0.1.0 让记忆变得持久：一条向量、它的 payload 和倒排条目作为一个整体提交，扛得住 `SIGKILL`，
重新打开也不需要重建。它没做到的是让这份记忆**被 Python 之外的东西访问**。如果你的 agent runtime
不是你自己的进程——比如一个通过 MCP 跟工具对话的宿主——那么每个 sable-vector 集合前面都还挡着
一层你得自己写、自己维护的胶水脚本。

v0.1.1 把这层胶水拿掉了。集合现在是一等的 MCP 资源：通过 stdio 发布三个工具，任何支持 MCP 的宿主
都能读写它，完全不必 import 这个库。

## v0.1.1 里有什么

### Model Context Protocol 服务端

```bash
pip install "sable-vector[sable,mcp]"
sable-vector serve-mcp
```

命令会在传输保持打开期间一直阻塞，客户端断开时才返回——所以它该出现在客户端的 stdio server
配置里，而不是一个你在等它结束的 shell 里。

| 工具 | 用途 |
| --- | --- |
| `sable_vector_search` | 按向量查询，可选地收窄到 payload 字段等于给定值的记录 |
| `sable_vector_upsert` | 写一条记录——embedding 加 JSON 属性——作为一个持久单元 |
| `sable_vector_describe` | 报告集合的记录数、向量维度和距离度量 |

有两个细节值得说直白些，因为最容易搞错的就是它们：

**embedding 由调用方给。** 服务端从不计算向量、不调模型、不开 socket。它就是一个前面套了协议的
索引；谁拥有 embedding 模型，谁就仍然拥有 embedding。

**过滤是真过滤。** 带 `filters` 的 `search` 会走 `PrefilterExecutor`，候选集在打分*之前*就被倒排
索引剪掉，而不是先全局取回再丢掉。这一点在响应里能看出来：前置过滤路径会报告它考虑了几个候选、
实际给几个打了分。一个什么都匹配不上的谓词返回零候选，而不是一堆近似但不符合条件的 top-N。

### 关于 `mcp` 的固定版本

服务端用的是底层的 `Server.list_tools` / `Server.call_tool` **装饰器** API。`mcp` 2.x 把这些装饰器
换成了构造器回调，所以 extra 固定在 `<2`。这是有意做的兼容性选择，不是疏忽：固定的这条线正是文档里
代码所针对的版本，悄悄解开会拿一个能用的服务端换第一次握手时的 `AttributeError`。

### 这个版本顺带修掉的文档问题

- 两份 README 和 v0.1.0 公告里的 quickstart 写的是
  `FlatIndex.materialise(engine, metric=...)`。但 `materialise` 收的是 **storage**，不是 engine，
  而且必须给 `dim`——那段代码照抄就会抛 `TypeError`。三处都改了。
- Mermaid 节点标签里含括号又没加引号，会弄坏 GitHub 上的架构图；两份 README 现在都加了引号。

## 从 v0.1.0 升级

存储布局没变。`0.1.0` 就标了 pre-stable，不存在迁移步骤：用 v0.1.1 打开已有集合直接就能用。
行为上唯一的变化是加法——多了一个 `serve-mcp` 命令和一个别的部分都不依赖的可选 `mcp` extra。

```bash
pip install --upgrade "sable-vector[sable,mcp]"
```

## 已知限制

和 v0.1.0 一样，还是那份老实清单：每个集合单 writer 进程；磁盘布局是 pre-1.0、尚未冻结；
前置过滤只看得到显式声明为 indexed 的字段；不捆绑 embedding 模型。

MCP 这边再补两条：服务端是单客户端 stdio（没有 HTTP 传输，也没有多租户鉴权层——这是本地库，
信任边界就是启动它的那个进程的边界）；每次工具调用都会为它的集合打开并物化状态，所以对吞吐
敏感的循环应该直接用库 API，不要走协议。

## 致谢

这个版本也把 v0.1.0 开发期间那次跨层排障记了下来——一个并发失败，最后查出是存储引擎的缺陷，
已在 [`sable`](https://github.com/H-I-F/sable) v0.1.1 上游修复。完整过程见
[案例研究](./case-studies/cross-layer-bugfix.zh-CN.md)；两份 README 都带着
*「Built on sable — with critical upstream bugfixes contributed」*。

## 链接

- 仓库：<https://github.com/H-I-F/sable-vector>
- 发布页：<https://github.com/H-I-F/sable-vector/releases/tag/v0.1.1>
- 完整历史：[CHANGELOG.md](../CHANGELOG.zh-CN.md)
- 上游存储引擎：[`sable`](https://github.com/H-I-F/sable)
