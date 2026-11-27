param(
    [switch]$NoAI,
    [switch]$SkipSync,
    [switch]$IncludeModels,
    [switch]$Zip,
    [switch]$IsolatedEnv,
    [string]$OutDir = ".local\dist"
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$TargetVenv = if ($IsolatedEnv) {
    Join-Path $env:TEMP "Whistle-pack-venv"
} else {
    Join-Path $Root ".venv"
}

if (-not $SkipSync) {
    if (-not $IsolatedEnv) {
        $running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
            Where-Object {
                $_.CommandLine -match "audiosearch\.browser" -and
                $_.ExecutablePath -like "*AudioSearchTool\.venv*"
            }
        if ($running) {
            Write-Host "[ERROR] 检测到 Whistle 正在从 .venv 运行，为避免占用 DLL 锁导致同步失败，请先关闭客户端后再执行打包（或加 -IsolatedEnv / -SkipSync）。" -ForegroundColor Yellow
            exit 2
        }
    } else {
        $env:UV_PROJECT_ENVIRONMENT = $TargetVenv
    }

    $syncArgs = @("sync", "--locked", "--group", "desktop", "--group", "playback", "--group", "packaging")
    if (-not $NoAI) {
        $syncArgs += @("--group", "ai")
    }
    Write-Host "[1/4] 同步打包环境 ($TargetVenv): uv $($syncArgs -join ' ')" -ForegroundColor Cyan
    & uv @syncArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$VenvPy = Join-Path $TargetVenv "Scripts\python.exe"
if (-not (Test-Path $VenvPy)) {
    Write-Host "[ERROR] 未找到 $VenvPy，请先运行不带 -SkipSync 的打包命令。" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path (Join-Path $Root "src\audiosearch\browser\resources\audiosearch.ico"))) {
    Write-Host "[1.5/4] 生成应用图标..." -ForegroundColor Cyan
    $env:QT_QPA_PLATFORM = "offscreen"
    & $VenvPy (Join-Path $Root "packaging\generate_icon.py")
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$OutAbs = if ([System.IO.Path]::IsPathRooted($OutDir)) { $OutDir } else { Join-Path $Root $OutDir }
$WorkAbs = if ($IsolatedEnv) { Join-Path $env:TEMP "Whistle-pack-work" } else { Join-Path $Root ".local\build\pyinstaller" }
$BuildDistAbs = if ($IsolatedEnv) { Join-Path $env:TEMP "Whistle-pack-dist" } else { $OutAbs }
New-Item -ItemType Directory -Force -Path $OutAbs | Out-Null
New-Item -ItemType Directory -Force -Path $WorkAbs | Out-Null
if ($IsolatedEnv -and (Test-Path $BuildDistAbs)) {
    Remove-Item -Recurse -Force $BuildDistAbs
}
New-Item -ItemType Directory -Force -Path $BuildDistAbs | Out-Null

$env:AUDIOSEARCH_BUILD_AI = if ($NoAI) { "0" } else { "1" }
Write-Host "[2/4] 执行 PyInstaller 打包 (AUDIOSEARCH_BUILD_AI=$($env:AUDIOSEARCH_BUILD_AI))..." -ForegroundColor Cyan
& $VenvPy -m PyInstaller `
    --noconfirm `
    --clean `
    --distpath $BuildDistAbs `
    --workpath $WorkAbs `
    (Join-Path $Root "packaging\audiosearch.spec")
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$BundleDir = Join-Path $BuildDistAbs "Whistle"
$ExePath = Join-Path $BundleDir "Whistle.exe"
if (-not (Test-Path $ExePath)) {
    Write-Host "[ERROR] 打包完成但未找到 $ExePath" -ForegroundColor Red
    exit 1
}

Set-Content -Path (Join-Path $BundleDir "portable.flag") -Value "portable=1" -Encoding UTF8

$QuickReadme = @"
Whistle（口哨）独立桌面版 · 快速启动说明
========================================

1. 双击 Whistle.exe 即可运行（本目录旁自动生成 data\ 存放本地曲库、设置与向量索引）。
2. 如需离线启用 AI 中文语义搜索与找同类：
   - 已预置双塔离线模型（Qwen3-Embedding-0.6B + CLAP2-large）时可直接建库与搜索；
   - 或将共享的模型缓存（含 hub\ 目录）放到本目录的 models\hf\ 下，也可设置环境变量 AUDIOSEARCH_MODELS_DIR。
3. REAPER 联动与快捷方式迁移：
   - 点击底部状态栏「REAPER 未连接」→「一键安装 / 更新脚本」，即可启用双击 / Enter 无破坏插入（Spot）；
   - 在左侧栏空白处或状态栏 REAPER 弹窗里点击「从 REAPER 导入快捷方式…」，可将 REAPER Media Explorer 分组与快捷方式一键迁入；
   - 平时也可直接从列表或波形选区拖拽音频文件到 REAPER、剪映、Premiere 等软件。
4. 在线增量更新：
   - 点击右下角状态栏版本号（如 v0.1.0）可检查 GitHub 最新版本并下载 <1 MB 热更新补丁，升级全程不改动本地 data\ 与 models\。
"@
Set-Content -Path (Join-Path $BundleDir "先读我-快速启动.txt") -Value $QuickReadme -Encoding UTF8

if ($IncludeModels) {
    if ($NoAI) {
        Write-Host "[WARN] 已指定 -NoAI，忽略 -IncludeModels。" -ForegroundColor Yellow
    } else {
        $RequiredModels = @(
            "models--Qwen--Qwen3-Embedding-0.6B",
            "models--laion--clap2-large-audioxl-s1B"
        )
        $SrcHub = Join-Path $Root ".local\models\hf\hub"
        $DstHub = Join-Path $BundleDir "models\hf\hub"
        New-Item -ItemType Directory -Force -Path $DstHub | Out-Null
        foreach ($m in $RequiredModels) {
            $src = Join-Path $SrcHub $m
            $dst = Join-Path $DstHub $m
            if (-not (Test-Path $src)) {
                Write-Host "[ERROR] 缺少必需离线模型缓存: $src" -ForegroundColor Red
                exit 1
            }
            Write-Host "[2.5/4] 复制模型缓存: $m ..." -ForegroundColor Cyan
            New-Item -ItemType Directory -Force -Path $dst | Out-Null
            & robocopy $src $dst /E /R:2 /W:1 /NP /NFL /NDL | Out-Null
            if ($LASTEXITCODE -ge 8) {
                Write-Host "[ERROR] robocopy 复制模型 $m 失败 (exit=$LASTEXITCODE)" -ForegroundColor Red
                exit $LASTEXITCODE
            }
        }
    }
}

Write-Host "[3/4] 运行可执行文件自检 (--smoke-test)..." -ForegroundColor Cyan
$SmokeId = [System.Guid]::NewGuid().ToString("N")
$SmokeDb = Join-Path $env:TEMP ("whistle-smoke-$SmokeId.sqlite3")
$SmokeOut = Join-Path $env:TEMP ("whistle-smoke-out-$SmokeId.txt")
$SmokeErr = Join-Path $env:TEMP ("whistle-smoke-err-$SmokeId.txt")
try {
    $smokeProc = Start-Process `
        -FilePath $ExePath `
        -ArgumentList @("--smoke-test", "--db", $SmokeDb) `
        -Wait `
        -PassThru `
        -RedirectStandardOutput $SmokeOut `
        -RedirectStandardError $SmokeErr
    $outRaw = if (Test-Path $SmokeOut) { Get-Content $SmokeOut -Raw } else { $null }
    $errRaw = if (Test-Path $SmokeErr) { Get-Content $SmokeErr -Raw } else { $null }
    $outText = if ($outRaw) { $outRaw.Trim() } else { "" }
    $errText = if ($errRaw) { $errRaw.Trim() } else { "" }
    if ($outText) { Write-Host $outText }
    if ($errText) { Write-Host $errText -ForegroundColor Yellow }
    if ($smokeProc.ExitCode -ne 0) {
        Write-Host "[ERROR] 可执行文件自检失败 (exit=$($smokeProc.ExitCode))" -ForegroundColor Red
        exit $smokeProc.ExitCode
    }
    $reportLine = ($outText -split "`r?`n" | Where-Object { $_.Trim().StartsWith("{") } | Select-Object -Last 1)
    if (-not $reportLine) {
        Write-Host "[ERROR] 自检未输出 JSON 报告。" -ForegroundColor Red
        exit 1
    }
    $report = $reportLine | ConvertFrom-Json
    if (-not ($report.store_ok -and $report.reaper_script -and $report.app_icon)) {
        Write-Host "[ERROR] 核心资源检查未全通过：$reportLine" -ForegroundColor Red
        exit 1
    }
    if (-not $NoAI) {
        if (-not $report.ai.torch -or $report.ai.transformers -ne $true -or $report.ai.open_clip -ne $true -or $report.ai.soxr -ne $true) {
            Write-Host "[ERROR] AI 运行时自检失败：$reportLine" -ForegroundColor Red
            exit 1
        }
    }
} finally {
    Remove-Item -Force -ErrorAction SilentlyContinue $SmokeDb, "$SmokeDb-shm", "$SmokeDb-wal", $SmokeOut, $SmokeErr
}

if ($Zip) {
    $ZipPath = Join-Path $OutAbs "Whistle-win64.zip"
    Write-Host "[4/4] 压缩分发包: $ZipPath ..." -ForegroundColor Cyan
    if (Test-Path $ZipPath) {
        Remove-Item -Force $ZipPath
    }
    $TarExe = Join-Path $env:SystemRoot "System32\tar.exe"
    if (Test-Path $TarExe) {
        & $TarExe -a -c -C $BuildDistAbs -f $ZipPath "Whistle"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] tar.exe 压缩失败 (exit=$LASTEXITCODE)" -ForegroundColor Red
            exit $LASTEXITCODE
        }
    } else {
        Compress-Archive -Path $BundleDir -DestinationPath $ZipPath -CompressionLevel Optimal
    }
}

$FinalBundleDir = Join-Path $OutAbs "Whistle"
if ($IsolatedEnv) {
    if (Test-Path $FinalBundleDir) {
        Remove-Item -Recurse -Force $FinalBundleDir
    }
    Move-Item -Path $BundleDir -Destination $FinalBundleDir
    Remove-Item -Recurse -Force $BuildDistAbs -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force $WorkAbs -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "[OK] 打包成功：$FinalBundleDir" -ForegroundColor Green
