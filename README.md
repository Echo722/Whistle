# Whistle

独立桌面音效浏览器（包目录位于 `src/audiosearch/`）：交互手感对标 REAPER Media Explorer，元数据管理对标 Soundminer，并内建本地 Qwen3-0.6B + CLAP2-Large 中文语义检索与音频找同类。

设计方案见 [独立客户端重做设计方案](./docs/plans/2026-11-23-独立客户端重做-design.md)，日常使用请看 [快速上手指南](./docs/快速上手指南.md)。

## 核心能力（R1–R6）

- **百万级本地浏览与检索（R1）**：SQLite WAL + FTS5（`trigram`）即时检索（空格 `AND` / `OR` / `-排除` / `"短语"` / `*` 通配符）、文件夹树与路径历史、虚拟列表排序与自定义列、离线盘灰显及路径重定位。
- **低延迟试听与波形编辑（R2 + Spectral Peaks）**：WASAPI/PortAudio 帧计数直出、`Space` 播停 / `↑↓` 切行 / 自动播放 / 循环、实时变调（±24 st）/ 变速（0.25x–4.00x）/ `StreamingWSOLA` 保留音高、全宽可缩放波形（含 REAPER Spectral Peaks 频谱着色）、选区微调与多格式烘培拖出（`CF_HDROP`）。
- **AI 中文语义搜索与以音找音（R3）**：Qwen3-0.6B（文本）+ CLAP2-Large（声学）双塔旁路检索、中文自然语言与排除规则解析、基于整文件或波形选区的音频找同类、命中原因解释与后台增量向量建库。
- **素材库整理（R4）**：收藏（★）、评分与备注、自定义合集分组、最近试听、搜索历史固定、重复文件识别与不删原文件隐藏、`.ReaperFileList` 只读导入。
- **Soundminer 级元数据管线（R5）**：内置 UCS v8.2 双语分类、列表内 `F2` 编辑与多选批量处理（设值/前缀/后缀/正则）、模板批量改名（含环状冲突两阶段交换与批次撤销）、CSV 导入导出、AI 自动补全 UCS / 描述 / 中英互译，以及四重安全写回源文件（BWF `bext` / `LIST-INFO` / `iXML`、ID3、Vorbis，写前后音频块 SHA-256 校验 + 写前备份 + 同盘原子替换 + 批次撤销）。
- **REAPER 一键桥接与独立打包（R6）**：状态栏一键部署 `audiosearch_spot_bridge.lua`（支持便携版 REAPER）、一键导入 REAPER Media Explorer 快捷方式分组、双击 / `Enter` 无破坏插入编辑光标及多选连续顺插、Windows 独立可执行打包，以及基于 GitHub Releases 的 `< 1 MB` 增量代码热更新（自动回滚保护）。

## 目录导航

```text
Whistle/
  src/audiosearch/
    library/          本地库管理、后台增量扫描、FTS5 查询、UCS、改名、CSV 与安全写回
    browser/          PySide6 桌面端（树、虚拟表、检查器、波形、试听、拖放、REAPER 桥配置）
    audition/         本机解码、帧计数播放器、WASAPI/PortAudio 输出、MediaEye 封面链
    audioio/          基础 WAV 解析与重采样
    catalog/          WAV 内嵌元数据解析（embedded.py）
    inference/        Qwen3-Embedding-0.6B (MRL) / CLAP2-Large (BF16) 双塔推理封装
    ai/               增量 float16 向量库、R3 中英规则、双塔检索与 AI 元数据补全
    retrieval/        查询规则解释、向量检索与 RRF 融合
    integrations/     REAPER 文件通道协议与安全守卫
  reaper/             REAPER Spot Lua 桥脚本
  packaging/          PyInstaller 全量 AI 打包配置、图标生成与一键构建脚本
  tests/              自动化行为与 UI 测试
  docs/               设计方案与快速上手指南
  samples/audio/      本机测试音效（不入 Git）
  .local/             本地库、模型缓存、向量索引、备份与构建产物（不入 Git）
```

## 开发期启动与命令行

```powershell
uv sync --locked --group desktop --group playback --group ai
uv run --locked --group desktop --group playback --group ai python -m audiosearch.browser
```

也可用 CLI 启动、预扫描目录或增量构建 AI 向量索引：

```powershell
uv run --locked --group desktop --group playback --group ai audiosearch browser
uv run --locked --group playback audiosearch scan samples/audio
uv run --locked --group playback --group ai audiosearch index --db .local/library.sqlite3
```

> 如 Windows 开发机开启过低完整性沙箱导致无法与 REAPER 拖拽或写通道，可执行 `icacls ".\.venv\Scripts\python.exe" /setintegritylevel Medium` 恢复普通权限后启动。

## 打包独立可执行版本（Windows 全量 AI CUDA 版）

关闭正在运行的桌面端后，在仓库根目录 PowerShell 执行（带模型打包实现开箱即用）：

```powershell
.\packaging\build_windows.ps1 -IncludeModels -Zip
```

- 默认产出：`.local\dist\Whistle\Whistle.exe`（便携目录，支持 `--smoke-test` 自检）与 `.local\dist\Whistle-win64.zip`。
- `-IncludeModels` 仅精准复制正式版双塔模型（`Qwen3-Embedding-0.6B` + `clap2-large-audioxl-s1B`，约 `3.66 GB`）；不加该参数时可在首次启用 AI 时按需自动下载（国内网络可配置 `HF_ENDPOINT=https://hf-mirror.com`），也可离线放到 `Whistle\models\hf\`，或设置环境变量 `AUDIOSEARCH_MODELS_DIR` 指向本机已有模型目录。

### 增量热更新补丁打包

便携版支持从 GitHub Releases 拉取 `< 1 MB` 的代码补丁包并原地重启升级（不覆盖 `data/` 与 `models/`）：

```powershell
.\packaging\build_patch.ps1
```

默认产出 `.local\dist\patches\whistle-patch-vX.Y.Z.zip` 与 `.local\dist\patches\latest.json`，随对应 tag 上传到 GitHub Releases 即可推送在线更新。

## 检查与测试

```powershell
uv run --locked ruff check src tests
uv run --locked --group desktop --group playback --group ai pytest
```
