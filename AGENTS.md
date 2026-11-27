# 项目协作规则

用中文沟通。先阅读 `README.md` 与 `docs/plans/2026-11-23-独立客户端重做-design.md`。

- **定位**：独立桌面音效浏览器（Python + PySide6 + SQLite FTS5）＋ 本机声卡直出试听 ＋ 可选 AI 语义/音频旁路检索。
- **目录约定**：产品方案放 `docs/plans/`，测试音效放 `samples/audio/`，本地数据库、缓存、备份与打包输出放 `.local/`（不提交到 Git）。
- **核心包边界**：
  - `src/audiosearch/library/`：文件夹管理、增量扫描、SQLite + FTS5 检索、峰值生成与 R5 元数据写回。
  - `src/audiosearch/browser/`：独立 PySide6 桌面界面、虚拟列表、本地试听控制与文件拖放。
  - `src/audiosearch/audition/`、`audioio/`、`catalog/embedded.py`：解码、声卡输出、封面链与只读内嵌元数据解析。
  - `src/audiosearch/inference/`、`retrieval/`、`indexing/`、`integrations/reaper/`：保留供后续轮次接入的 AI 检索与 REAPER 桥接基础模块。
- **数据安全边界**：**音频数据永不修改**；元数据仅在 R5 用户显式写回操作下由写回模块写入（需带音频数据块写前后哈希校验、写前备份与原子替换）。ME 数据库与缓存只读，不把私有素材、路径日志、索引或模型提交到 Git。
- **依赖与环境**：使用 `uv sync --locked`；跨模块时间均为原文件秒数，区间 `[start, end)`，声道从 0 开始。
- **验证原则**：保留对核心行为（增量扫描、FTS 语法、音频缓冲、UI 交互）的自动化行为测试与真实数据实测，以可运行版本与实际使用体验为准；不写只检查源码字符串的测试。
- **Git 分支、Worktree 与发版隔离**：
  - 禁止直接在 `main` 上进行新功能开发或未验收改动；动工前先核对当前分支与 `git status`，单任务切独立分支（如 `feat/<name>`、`fix/<name>`），多对话并行修改同一仓库时使用独立 `git worktree`（如 `git worktree add ..\AudioSearchTool-wt-<name> -b feat/<name> main`）。
  - 提交与发版前**禁止无脑 `git add -A`**，必须先核对 `git status` 与改动文件清单并仅暂存当前任务相关文件；若工作区混有其他任务未提交改动，必须先停下确认或隔离，严禁顺手带入 `main` 或 `v*` Release tag。
  - Worktree 复用环境技巧：未变更依赖时可通过 `$env:UV_PROJECT_ENVIRONMENT` 复用主仓库 `.venv`，或通过目录联接（Junction）共享 `.local/` 缓存；涉及依赖新增时在对应 worktree 单独执行 `uv sync --locked`。
