<#
.SYNOPSIS
    Download the Qwen2.5-0.5B model from ModelScope into the project's .cache directory.

.DESCRIPTION
    This script clones the Qwen2.5-0.5B model repository from ModelScope using
    git + git-lfs (shallow clone) and places it under:
        <ProjectRoot>\.cache\Qwen2.5-0.5B

    It is the first step of the C-language LLM inference engine project,
    preparing the experimental model weights for later loading in C.

.PARAMETER ModelRepo
    ModelScope repository path in "Owner/Name" format.
    Default: "Qwen/Qwen2.5-0.5B"

.PARAMETER SkipExisting
    If set, reuse an already-downloaded model directory without re-cloning.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\download_model.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\download_model.ps1 -SkipExisting
#>

[CmdletBinding()]
param(
    [string]$ModelRepo = "Qwen/Qwen2.5-0.5B",
    [switch]$SkipExisting
)

# ---------------------------------------------------------------------------
# 0. Helpers
# ---------------------------------------------------------------------------
$ErrorActionPreference = "Stop"

function Write-Step([string]$msg) {
    Write-Host ""
    Write-Host "[>>] $msg" -ForegroundColor Cyan
}

function Write-Ok([string]$msg) {
    Write-Host "[ OK] $msg" -ForegroundColor Green
}

function Write-Warn([string]$msg) {
    Write-Host "[WARN] $msg" -ForegroundColor Yellow
}

function Write-Err([string]$msg) {
    Write-Host "[FAIL] $msg" -ForegroundColor Red
}

# ---------------------------------------------------------------------------
# 1. Locate project root
#    The script lives in <root>/scripts/, so the project root is the parent
#    directory of the directory containing this script.
# ---------------------------------------------------------------------------
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$CacheDir    = Join-Path $ProjectRoot ".cache"
$ModelDir    = Join-Path $CacheDir "Qwen2.5-0.5B"

Write-Host "==========================================================" -ForegroundColor DarkGray
Write-Host "  Qwen2.5-0.5B Model Downloader (ModelScope)" -ForegroundColor White
Write-Host "==========================================================" -ForegroundColor DarkGray
Write-Host "  Project root : $ProjectRoot"
Write-Host "  Cache dir    : $CacheDir"
Write-Host "  Model dir    : $ModelDir"
Write-Host "  Model repo   : $ModelRepo"
Write-Host "==========================================================" -ForegroundColor DarkGray

# ---------------------------------------------------------------------------
# 2. Reuse existing download
# ---------------------------------------------------------------------------
$keyFiles = @("config.json")
function ModelLooksComplete([string]$dir) {
    if (-not (Test-Path $dir)) { return $false }
    foreach ($f in $keyFiles) {
        if (-not (Test-Path (Join-Path $dir $f))) { return $false }
    }
    # At least one weight file should be present
    $weights = Get-ChildItem -Path $dir -Recurse -Include *.safetensors, *.bin -File -ErrorAction SilentlyContinue
    return ($weights -and $weights.Count -gt 0)
}

if ($SkipExisting -and (Test-Path $ModelDir)) {
    if (ModelLooksComplete $ModelDir) {
        Write-Ok "Model already downloaded, reusing existing directory (SkipExisting)."
        Write-Host ""
        Write-Host "Model files under: $ModelDir" -ForegroundColor Cyan
        exit 0
    } else {
        Write-Warn "Model directory exists but looks incomplete, will re-clone."
        Remove-Item -Path $ModelDir -Recurse -Force
    }
}

if (Test-Path $ModelDir) {
    Write-Warn "Target directory already exists: $ModelDir"
    Write-Warn "Remove it manually or re-run with -SkipExisting, aborting to avoid conflict."
    exit 1
}

# ---------------------------------------------------------------------------
# 3. Verify required tools: git + git-lfs
# ---------------------------------------------------------------------------
Write-Step "Checking tool availability..."

$gitBin = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitBin) {
    Write-Err "git not found in PATH. Install Git from https://git-scm.com and re-run."
    exit 1
}
Write-Ok "git found: $($gitBin.Source)"

$gitLfsBin = Get-Command git-lfs -ErrorAction SilentlyContinue
if (-not $gitLfsBin) {
    Write-Err "git-lfs not found in PATH. Install Git LFS from https://git-lfs.com and re-run."
    exit 1
}
Write-Ok "git-lfs found: $($gitLfsBin.Source)"

# Confirm git-lfs is installed (not just present on disk)
$lfsVersion = & git lfs version 2>$null
if (-not $lfsVersion) {
    Write-Err "git-lfs is not installed. Run 'git lfs install' first."
    exit 1
}
Write-Ok "git-lfs installed: $lfsVersion"

# ---------------------------------------------------------------------------
# 4. Create .cache directory
# ---------------------------------------------------------------------------
Write-Step "Creating cache directory..."
New-Item -ItemType Directory -Path $CacheDir -Force | Out-Null
Write-Ok "Cache directory ready: $CacheDir"

# ---------------------------------------------------------------------------
# 5. Clone the model (shallow, with LFS)
# ---------------------------------------------------------------------------
$cloneUrl = "https://www.modelscope.cn/$ModelRepo.git"

Write-Step "Cloning model from ModelScope (shallow clone)..."
Write-Host "  URL : $cloneUrl"
Write-Host "  DIR : $ModelDir"
Write-Host "  This may take a while depending on your network and model size (~1 GB)."
Write-Host ""

try {
    # --depth 1 : shallow clone (we only need the latest snapshot)
    # --no-tags : skip tags to save time
    git clone --depth 1 --no-tags --progress $cloneUrl $ModelDir
} catch {
    Write-Err "git clone failed: $_"
    exit 1
}

Write-Ok "Clone completed."

# ---------------------------------------------------------------------------
# 6. Verify LFS files were actually pulled
# ---------------------------------------------------------------------------
Write-Step "Verifying downloaded files..."

if (-not (Test-Path (Join-Path $ModelDir "config.json"))) {
    Write-Err "config.json missing. The download may be incomplete."
    exit 1
}

# Check that LFS weight files are real files (not pointer files)
$weightFiles = Get-ChildItem -Path $ModelDir -Recurse -Include *.safetensors, *.bin -File -ErrorAction SilentlyContinue
if (-not $weightFiles -or $weightFiles.Count -eq 0) {
    Write-Err "No weight files (.safetensors / .bin) found. git-lfs files may not have been fetched."
    Write-Warn "Try running: git -C $ModelDir lfs pull"
    exit 1
}

# Detect LFS pointer files (they are tiny text files with "version https://git-lfs" header)
$pointerFiles = @()
foreach ($wf in $weightFiles) {
    if ($wf.Length -lt 1024) {
        # A real model weight file is always much larger than 1KB.
        # A pointer file is a small text file.
        $pointerFiles += $wf
    }
}
if ($pointerFiles.Count -gt 0) {
    Write-Err "Found $($pointerFiles.Count) LFS pointer file(s) instead of real weights:"
    foreach ($pf in $pointerFiles) { Write-Host "    - $($pf.FullName)" }
    Write-Warn "Run: git -C $ModelDir lfs pull --force"
    exit 1
}

Write-Ok "Weight files verified (real LFS content, not pointers)."

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
$allFiles = Get-ChildItem -Path $ModelDir -Recurse -File -ErrorAction SilentlyContinue
$totalSize = ($allFiles | Measure-Object -Property Length -Sum).Sum

Write-Host ""
Write-Host "==========================================================" -ForegroundColor DarkGray
Write-Host "  Download complete" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor DarkGray
Write-Host "  Location     : $ModelDir"
Write-Host "  Files        : $($allFiles.Count)"
Write-Host "  Total size   : $([math]::Round($totalSize / 1MB, 2)) MB"
Write-Host ""
Write-Host "  Key files:" -ForegroundColor Cyan
$keyPatterns = @(
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "*.safetensors",
    "*.bin"
)
foreach ($pat in $keyPatterns) {
    Get-ChildItem -Path $ModelDir -Recurse -Filter $pat -File -ErrorAction SilentlyContinue |
        ForEach-Object {
            Write-Host ("    {0,-30} {1,8} KB" -f $_.FullName.Substring($ModelDir.Length + 1), [math]::Round($_.Length / 1KB, 1))
        }
}
Write-Host "==========================================================" -ForegroundColor DarkGray
Write-Host ""
Write-Host "Next step: implement the C inference engine that loads these weights." -ForegroundColor White
