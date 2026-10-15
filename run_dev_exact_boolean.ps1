$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = Join-Path $root "backend\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
& $python -m pip install -r (Join-Path $root "backend\requirements-optional.txt")
if ($LASTEXITCODE -ne 0) { throw "manifold3d 安装失败" }
& (Join-Path $root "run_dev.ps1")
