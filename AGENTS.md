# lagranto_tracking – 项目说明（给人/Agent）

## Python 环境

- 本项目使用仓库内置的虚拟环境：`.venv/`
- 当前机器上的可执行目录为：`/home/sunco/Projects/lagranto_tracking/.venv/bin`（等同于相对路径 `.venv/bin`）
- 为避免误用系统 Python，建议命令里显式使用：`.venv/bin/python`、`.venv/bin/pip`

## 依赖

- 依赖列表在：`requirements.txt`
- 安装方式（二选一）：
  - 激活环境后安装：`source .venv/bin/activate && pip install -r requirements.txt`
  - 不激活，直接用 venv 的 pip：`.venv/bin/pip install -r requirements.txt`

## 运行（可选）

- 运行脚本：`.venv/bin/python lagranto_track.py`
- 运行 Notebook：在 VS Code / Jupyter 里选择 `.venv` 作为 Python 解释器 / Kernel
