[English](./RELEASE_NOTES_v0.1.2.md) | 简体中文

# sable-vector v0.1.2

一次文档发版。2026 年 11 月 3 日。

检索引擎、键空间、编解码、每一种索引，都和 0.1.1 一模一样。公开 API 没动，磁盘格式没动，
0.1.1 写出来的集合照常打开。`sable` 依赖仍固定在 `v0.1.1`：那个 tag 装的是本项目的并发测试
逼出来的快照读修复，而 0.1.2 没有任何行为变化，没有理由去挪它。

这个版本号只为一件事存在——`docs/` 下的工程笔记补齐了，中英双语，从一个入口就能走到。

## 这一版里有什么

- **两份索引。** `docs/README.md` 与 `docs/README.zh-CN.md` 列出每一篇，附源码地图，以及从
  ADR-001 经案例复盘读到实测报告的阅读顺序。
- **五篇双语 ADR** —— 单一键空间、payload 提交记录、两种过滤策略、float32 编解码与分数方向、
  依赖固定。
- **一篇 FAQ、三篇工程实践指南、两篇实践配方**，全部双语：`vector-search-qa`、
  `skills/` 那组（零拷贝编解码、持久写入、混合查询策略）、`recipes/` 那组（按租户收窄的混合
  检索、选择 `nprobe`）。
- **一份带实测数字的报告**（`docs/reports/benchmark-2026-11.md`）、v0.1.0 与 v0.1.1 公告、
  跨层案例复盘，都各有中文配套；根目录里 `CHANGELOG`、`CONTRIBUTING`、`SECURITY`、`PROGRESS`
  的中文版也一并补齐。
- **README 接好了线。** 文档表格，以及每条都成对给出 English 与简体中文的治理清单。

## 这一版里没有什么

没有重跑基准。配方里引用的 recall、延迟和 `nprobe` 数字，全部直接取自 2026 年 11 月那份报告，
包括那个没有抹平、原样留着的 p95 离群值。指南里的代码片段都从 `src/sable_vector/` 摘出，
不是为写文档另外编的。

CHANGELOG 的 `[未发布]` 段仍然挂着确定性的 PR 评审报告和迁到 H.I.F 组织这两条。那是工具与治理
层面按自己节奏走的事，这个 tag 讲的是文档。

## 升级

```sh
pip install --upgrade "sable-vector[sable,mcp] @ git+https://github.com/H-I-F/sable-vector.git@v0.1.2"
```

- 仓库：<https://github.com/H-I-F/sable-vector>
- 发布页：<https://github.com/H-I-F/sable-vector/releases/tag/v0.1.2>
- 完整变更：[CHANGELOG.zh-CN.md](../CHANGELOG.zh-CN.md) · [English](../CHANGELOG.md)
- 工程笔记：[索引](README.zh-CN.md) · [English](README.md)
