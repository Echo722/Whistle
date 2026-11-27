param(
    [Parameter(Mandatory = $true)][string]$Version,
    [string]$Notes = "",
    [string]$Repo = "Echo722/Whistle",
    [int]$MinRuntimeAbi = 0,
    [string]$OutDir = ".local\dist\patches"
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ($MinRuntimeAbi -le 0) {
    $MinRuntimeAbi = 1
    $UpdaterPy = Join-Path $Root "src\audiosearch\browser\updater.py"
    if (Test-Path $UpdaterPy) {
        $UpdaterSrc = Get-Content -Path $UpdaterPy -Raw -Encoding UTF8
        if ($UpdaterSrc -match '(?m)^RUNTIME_ABI\s*=\s*(\d+)') {
            $MinRuntimeAbi = [int]$Matches[1]
        }
    }
}
Set-Location $Root

$CleanVer = $Version.Trim().TrimStart("v", "V")
if (-not $CleanVer) {
    Write-Host "[ERROR] Version 不能为空。" -ForegroundColor Red
    exit 1
}

$OutAbs = if ([System.IO.Path]::IsPathRooted($OutDir)) { $OutDir } else { Join-Path $Root $OutDir }
New-Item -ItemType Directory -Force -Path $OutAbs | Out-Null

$StageDir = Join-Path $OutAbs (".whistle-patch-stage-" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $StageDir | Out-Null

try {
    $SrcPkg = Join-Path $Root "src\audiosearch"
    $DstPkg = Join-Path $StageDir "audiosearch"
    New-Item -ItemType Directory -Force -Path $DstPkg | Out-Null
    & robocopy $SrcPkg $DstPkg /E /XD __pycache__ .pytest_cache /XF *.pyc *.pyo /R:2 /W:1 /NP /NFL /NDL | Out-Null
    if ($LASTEXITCODE -ge 8) {
        Write-Host "[ERROR] 复制 src\audiosearch 失败 (robocopy exit=$LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }

    $SrcLua = Join-Path $Root "reaper\audiosearch_spot_bridge.lua"
    if (Test-Path $SrcLua) {
        $DstReaper = Join-Path $StageDir "reaper"
        New-Item -ItemType Directory -Force -Path $DstReaper | Out-Null
        Copy-Item -Force $SrcLua (Join-Path $DstReaper "audiosearch_spot_bridge.lua")
    }

    $BuiltAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    $PatchManifest = [ordered]@{
        version         = $CleanVer
        min_runtime_abi = $MinRuntimeAbi
        notes           = $Notes
        built_at        = $BuiltAt
    } | ConvertTo-Json -Depth 4
    $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText((Join-Path $StageDir "patch_manifest.json"), $PatchManifest, $Utf8NoBom)

    $ZipName = "whistle-patch-v$CleanVer.zip"
    $ZipPath = Join-Path $OutAbs $ZipName
    if (Test-Path $ZipPath) {
        Remove-Item -Force $ZipPath
    }

    $ItemsToPack = @("audiosearch", "patch_manifest.json")
    if (Test-Path (Join-Path $StageDir "reaper")) {
        $ItemsToPack += "reaper"
    }

    $TarExe = Join-Path $env:SystemRoot "System32\tar.exe"
    if (Test-Path $TarExe) {
        & $TarExe -a -c -C $StageDir -f $ZipPath @ItemsToPack
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[ERROR] tar.exe 创建补丁包失败 (exit=$LASTEXITCODE)" -ForegroundColor Red
            exit $LASTEXITCODE
        }
    } else {
        Compress-Archive -Path (Join-Path $StageDir "*") -DestinationPath $ZipPath -CompressionLevel Optimal
    }

    $Hash = (Get-FileHash -Algorithm SHA256 -Path $ZipPath).Hash.ToLowerInvariant()
    $SizeBytes = (Get-Item $ZipPath).Length
    $CleanRepo = $Repo.Trim().Trim("/")

    $LatestJson = [ordered]@{
        version         = $CleanVer
        min_runtime_abi = $MinRuntimeAbi
        patch_url       = "https://github.com/$CleanRepo/releases/download/v$CleanVer/$ZipName"
        patch_sha256    = $Hash
        patch_size      = $SizeBytes
        release_url     = "https://github.com/$CleanRepo/releases/tag/v$CleanVer"
        notes           = $Notes
        published_at    = $BuiltAt
    } | ConvertTo-Json -Depth 4
    $LatestPath = Join-Path $OutAbs "latest.json"
    [System.IO.File]::WriteAllText($LatestPath, $LatestJson, $Utf8NoBom)

    $SizeKb = [Math]::Round($SizeBytes / 1024.0, 1)
    Write-Host "[OK] 增量补丁已生成：$ZipPath ($SizeKb KB)" -ForegroundColor Green
    Write-Host "[OK] 更新清单已生成：$LatestPath" -ForegroundColor Green
    Write-Host "     SHA256: $Hash" -ForegroundColor DarkGray
} finally {
    Remove-Item -Recurse -Force $StageDir -ErrorAction SilentlyContinue
}
