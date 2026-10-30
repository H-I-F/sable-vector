# sable-vector

> Embedded hybrid vector search for local AI agent runtimes, built on the
> [`sable`](https://github.com/MKA930/sable) LSM storage engine.

The full architecture document is written in [English](README.md) and
[简体中文](README.zh-CN.md); it lands with PR 15 together with the container
image. Until then, `PROGRESS.md` is the authoritative view of what exists.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . && pip install "sable @ git+https://github.com/MKA930/sable.git@main"
pytest -q
```
