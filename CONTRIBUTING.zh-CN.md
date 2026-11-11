[English](./CONTRIBUTING.md) | 简体中文

# 给 sable-vector 提代码

sable-vector 是给本地 AI agent runtime 用的嵌入式混合（稠密向量 + 标量过滤）检索引擎，底下压着
[`sable`](https://github.com/H-I-F/sable) 这个 LSM 存储引擎。这篇讲怎么把开发环境搭起来，以及项目
遵守哪些约定，好让补丁能顺顺当当地合进去。

## 开发环境

目标版本是 **Python 3.10+**。`sable` 没发到 PyPI，得从仓库 tag 装。

```bash
git clone https://github.com/H-I-F/sable-vector.git
cd sable-vector
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[sable,mcp,test,lint]"
```

`[sable]` 从已发布的 git tag 拉存储引擎，`[test]` 装 `pytest`，`[lint]` 装 `ruff`。装完之后
`sable_vector` 直接从工作树 import。

## 本地自检

推之前跑一遍，CI 跑的就是同一套命令：

```bash
ruff check .
pytest -q
python -m benchmarks.bench_vector --sizes 1000 --queries 50 --no-ivf
```

容器镜像自带校验：`docker build -t sable-vector .`，然后 `docker run --rm sable-vector pytest -q`。
镜像构建由 CI 的 `docker` job 负责，贡献代码不需要自己本地 build。

## 分支与提交约定

分支按 `<type>/<short-slug>` 命名，跟合并提交的前缀对得上：

| 类型 | 例子 |
| --- | --- |
| `feat` | `feat/pq-compression` |
| `fix` | `fix/seek-overflow` |
| `docs` | `docs/portal-readme` |
| `bench` | `bench/ann-benchmarks` |
| `ci` | `ci/actions` |
| `build` | `build/docker` |
| `chore` | `chore/release` |

提交信息用 [Conventional Commits](https://www.conventionalcommits.org/)。标题别超过 72 个字符，
不好懂的地方在正文里写清楚*为什么*。本项目每条提交都带：

```
Co-authored-by: Leisurely Winston <winston_cn@yeah.net>
```

## 拉取请求

- PR 开到 `main`。
- 写清楚改了什么；如果改了行为，附上怎么验证。
- 指明它关掉的是 `PROGRESS.md` 里哪一行（交付进度记在那儿，不用 issue tracker）。
- CI 必须全绿：ruff、Python 3.10/3.11 上的 pytest，还有 docker 构建。

## 目录结构

| 路径 | 职责 |
| --- | --- |
| `src/sable_vector/` | 引擎、索引（flat / IVF）、查询、距离计算、适配器 |
| `benchmarks/` | ANN 基准套件（`python -m benchmarks.bench_vector`） |
| `examples/` | 能跑的集成 demo（例如 LangChain RAG） |
| `tests/` | pytest 套件 |

## 代码风格

- `ruff` 是 lint 和格式门禁，本地跑 `ruff check .`。
- 目标 Python 是 3.10，别用 3.11 才有的语法，除非做了保护。
- 合进来的代码里不许留 `TODO` / `FIXME` 占位；没做完的部分放到 PR 里讨论。

## 维护与发版

这里大多数改动根本不碰引擎：依赖升级、CI 与 workflow 调整、bot 配置、文档、dev 工具
pin。这些算 `routine`。Dependabot 提的 `routine` PR，CI 绿了之后自动合并，不用你人工复核、
也不用点合并，更不会为它单独发一个版本。

怎么分档不是评审时临时商量的，是算出来的。`.github/scripts/change_policy.py` 是唯一的
定义：读改动文件，返回三档之一——`core`（`src/` 下任何改动，或运行时依赖；唯一会发版的一档）、
`review`（基础镜像或 action 引用跨版本跳转，留给人工看）、`routine`。自动合并 workflow、
PR 报告、发版工具读的都是同一个判定，所以规则要改只改一处。改动文件列表为空或格式坏掉，
一律算错误，绝不当成 `routine`。

`policy-check.yml` 在每次 push 和 PR 上跑两个自检加一次发版一致性检查，它本身是必需状态
检查。完整约定、包括发一次版的步骤，见
[RELEASING.md](./RELEASING.md) · [简体中文](./RELEASING.zh-CN.md)。
