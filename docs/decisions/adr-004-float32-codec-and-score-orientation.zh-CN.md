[English](adr-004-float32-codec-and-score-orientation.md) | [简体中文](adr-004-float32-codec-and-score-orientation.zh-CN.md)

# ADR-004：float32 零拷贝镜像，分数一律越大越近

**状态：** 已接受 —— 实现在 `src/sable_vector/codec.py` 与 `distance.py`。

## 背景

向量既要以不透明字节的形式活在 LSM 树里，又要喂回 NumPy 当矩阵用。每次读都复制一遍，点查的
成本大头就变成复制了。另一头，`cosine` 和 `dot` 天然是越大越好，`l2` 却是越小越好；这个差别
一旦从核函数里漏出去，下游每一处排序都得知道当前用的是哪种度量。

## 决策

**落盘：** 小端 float32，无帧头、无分隔符、无定界。`encode_vector()` 就是
`as_query(vector).tobytes()`；`decode_vector()` 用 `np.frombuffer` 返回 storage 已经给回来的
字节上的一个 view，不做第二次复制。`decode_matrix()` 把拼接的块 reshape 成 `(n, d)`。字节序
和 dtype 两端都钉死；调用方确实要改，才用 `decode_vector(..., writeable=True)` 拿一份私有副本。

**内存里：** `score_matrix()` 把三种度量统一成**越大越近**——cosine 相似度和 dot 积原样使用，
L2 取距离的负值。即便存储是 float32，分数也用 float64 累加；`top_k()` 用稳定的 `argsort`，
让并列分数有确定的次序。

## 后果

解出来的向量是只读 view，要改就得显式要副本。float32 镜像在字节序变化时不具备可移植性，
需要显式转换——这也是 dtype 常量只放一处、解码时要断言的原因。

下游比较、排序、截断都不必对 `metric=` 分支。float32 存储 / float64 计分是有意的精度分配：
既让落盘镜像紧凑，又不让累加误差把分数咬得很近的邻居排错序。基准里的 recall 都是拿这个穷举核
当基准量出来的，所以它同时也是所有近似索引的 ground truth。
