$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = Join-Path $root "backend\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

if (-not (Test-Path (Join-Path $root "frontend\dist\index.html"))) {
    Push-Location (Join-Path $root "frontend")
    try {
        if (-not (Test-Path "node_modules")) { npm install --no-audit --no-fund; if ($LASTEXITCODE -ne 0) { throw "npm install 失败" } }
        npm run build
        if ($LASTEXITCODE -ne 0) { throw "前端构建失败" }
    } finally { Pop-Location }
}

Push-Location (Join-Path $root "backend")
try {
    & $python -m uvicorn app.main:app --host 0.0.0.0 --port 3277
    if ($LASTEXITCODE -ne 0) { throw "后端启动失败；请安装 backend/requirements.txt" }
} finally { Pop-Location }
