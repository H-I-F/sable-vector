[English](./release-notes-v0.1.4.md) | 简体中文

# sable-vector v0.1.4

一次维护发版。2026 年 11 月 11 日。

库本身没有任何改动。引擎、键空间、索引格式、公开 API、查询行为、每一条依赖固定，都和 0.1.3
一模一样——0.1.3 写出来的集合用 0.1.4 打开不需要迁移。`mcp<2` 的固定，以及
`sable @ git+...@v0.1.1` 这条直引用，都还在原处。

## 这一版里有什么

这一版记下的是仓库打理自己的方式变了。

- **依赖更新会自己合并。** 一个 `workflow_run` workflow 会在 Dependabot 的 PR 转绿之后给它挂上
  `auto-merge`。真正决定合不合的还是分支保护：必需检查、线性历史、所有对话已解决，都得过了才落地，
  所以 CI 挂了绝不会进 `main`。它在默认分支上跑，不需要额外 token 或第三方 action，也只碰
  Dependabot 开的 PR。
- **Dependabot 改成每月扫、分组提。** pip、GitHub Actions、Docker 三个生态从每周挪到每月，
  各自并成一个 PR。`mcp>=2` 的排除规则没动——Dependabot 依然不碰这一族，因为 2.x 会破坏
  `@server.list_tools` 的 API。
- **自动化安全修复打开了。** 常规更新慢下来，不代表报出来的漏洞也慢，它照样会有自己那条修复 PR。

## 以后什么才发版

不碰引擎的改动——依赖升级、CI 与 workflow 调整、bot 配置、文档——以后都走自动合并，不再逐个人工
复核，也不再单独发版。只有核心逻辑的调整，或者其他大到值得让装包的人专门读一眼的变更，才打 tag。

这一版就是把这条约定写下来的那一版，也是最后一次为纯维护改动发版。从它往后，版本号只在引擎动的时候
才动。

## 这一版里没有什么

引擎没动，没有新增运行时依赖，格式没变，也没有重跑基准。从这个 tag 装到的库，和 0.1.3 一样，只是
版本字符串不同。没有重写任何指南：它们描述的东西一样没变。

## 升级

```sh
pip install --upgrade "sable-vector[sable,mcp] @ git+https://github.com/H-I-F/sable-vector.git@v0.1.4"
```

- 仓库：<https://github.com/H-I-F/sable-vector>
- 发布页：<https://github.com/H-I-F/sable-vector/releases/tag/v0.1.4>
- 完整变更：[CHANGELOG.zh-CN.md](../CHANGELOG.zh-CN.md) · [English](../CHANGELOG.md)
- 工程笔记：[索引](README.zh-CN.md) · [English](README.md)
