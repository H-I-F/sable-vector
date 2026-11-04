[English](./release-notes-v0.1.3.md) | 简体中文

# sable-vector v0.1.3

MCP 接入面上的一次小版本。2026 年 11 月 4 日。

`serve-mcp` 暴露的三个工具，现在会在宿主请人批准调用之前，先把自己是什么说清楚。每个工具都
显式给出 `readOnlyHint`、`destructiveHint`、`idempotentHint`、`openWorldHint`。检索和 describe
只读且幂等；upsert 会覆盖已有的 id，属于有破坏性、但逻辑上幂等的写入。三个工具都只操作本地
集合目录，所以 `openWorldHint` 一律为 false。

少写一个 hint，就等于把 SDK 的默认值交给客户端，而其中三个默认值偏偏落在不安全的一侧。
现在有一条契约测试把工具的 wire name 和四个布尔值都钉住，工具改名或者漏写 hint 都会让测试挂掉。

## 这一版里有什么

- **三个 MCP 工具各补齐四个行为标注。** 宿主不用猜就能分清读和写。`ToolAnnotations` 在整个
  `mcp>=1.9,<2` 范围内都有，依赖固定没动。
- **一条契约测试**，按 wire name 而不是内部常量，锁定对外声明的工具集合和每个工具的四个 hint。
- **CHANGELOG 的欠账清了。** 确定性的 PR 评审报告和迁到 H.I.F 组织这两条，从 v0.1.2 起一直挂在
  `[未发布]` 段，这次归进正式的版本历史。

## 这一版里没有什么

引擎没动。键空间、索引格式、公开 API、依赖固定、查询行为，都和 0.1.2 一模一样——0.1.2 写出来的
集合，用 0.1.3 打开不需要迁移。这些 hint 只是工具清单上的提示性元数据，改变的是宿主事先能知道
什么，不是工具实际做什么。`sable` 依赖仍固定在 `v0.1.1`。

没有重跑基准，也没有重写任何指南：它们描述的东西一样没变。

## 升级

```sh
pip install --upgrade "sable-vector[sable,mcp] @ git+https://github.com/H-I-F/sable-vector.git@v0.1.3"
```

- 仓库：<https://github.com/H-I-F/sable-vector>
- 发布页：<https://github.com/H-I-F/sable-vector/releases/tag/v0.1.3>
- 完整变更：[CHANGELOG.zh-CN.md](../CHANGELOG.zh-CN.md) · [English](../CHANGELOG.md)
- 工程笔记：[索引](README.zh-CN.md) · [English](README.md)
