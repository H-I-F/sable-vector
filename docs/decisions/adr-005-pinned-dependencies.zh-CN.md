[English](adr-005-pinned-dependencies.md) | [简体中文](adr-005-pinned-dependencies.zh-CN.md)

# ADR-005：把定义 API 的依赖固定住

**状态：** 已接受 —— 实现在 `pyproject.toml` 与 `.github/dependabot.yml`。

## 背景

有两条依赖属于公开契约的一部分，不只是环境：

- `adapter/mcp_server.py` 建立在底层的 `Server.list_tools` / `Server.call_tool`
  **装饰器** API 上。`mcp` 2.x 把这些装饰器换成了构造器回调。曾经有一个 Dependabot PR 把约束
  放宽到 `<3`，结果三个 CI job 全红，报
  `AttributeError: 'Server' object has no attribute 'list_tools'`；那个 PR 被关掉，规则改成了永久的。
- `sable` 没发到 PyPI，而本项目依赖的快照读一致性修复要到 `v0.1.1` 才有（见
  [案例研究](../case-studies/cross-layer-bugfix.zh-CN.md)）。对着 `@main` 构建，等于说「碰巧存在的
  那个提交」，发布之后就复现不了了。

## 决策

```toml
mcp   = ["mcp>=1.9,<2"]
sable = ["sable @ git+https://github.com/H-I-F/sable.git@v0.1.1"]
```

`.github/dependabot.yml` 里对 `mcp` 的 `>=2` 写了永久 `ignore`。PEP 508 直引用本来就不在
Dependabot 的管辖范围内，所以上游这个固定版本是手动、连同一次发版一起提升的，不会由 bot 动。

## 后果

悄悄解开 `mcp` 的固定，等于拿一个能用的 stdio 服务端换第一次握手时的异常；现在要动它，得先把
服务端迁到构造器回调 API 并单独发一次变更。上游修复以「提升 tag + 发一个新的 `0.1.x`」吸纳，
`SECURITY.md` 里也是这么写的。

代价是变旧：两个固定版本都会慢慢落后于上游。这个代价可以接受，因为另一个选项——一个悄悄改掉
代码所针对的 API 的自动 PR——已经在 CI 里失败过一次。PR 评审报告会在每个 PR 上检查 `mcp<2`
规则和那条直引用，免得约束被顺手改掉。
