param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$env:PYTHONUTF8 = '1'
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.cache/uv'
$env:TEMP = Join-Path (Get-Location) '.cache/tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'Install uv first. See docs/hai-runbook-2026-11-11.md.'
}
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    uv venv --python $Python --no-python-downloads .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create CPU environment from an installed Python' }
}
# CPU only. Reuse the existing verified lock; never resolve the GPU extras here.
uv pip sync --python .venv/Scripts/python.exe requirements-dev.lock.txt
if ($LASTEXITCODE -ne 0) { throw 'CPU dependency sync failed' }
uv pip install --python .venv/Scripts/python.exe --no-deps -e .
if ($LASTEXITCODE -ne 0) { throw 'Editable source install failed' }
New-Item -ItemType Directory -Force -Path 'outputs/runs/hai','outputs/runs/thinh','outputs/packages' | Out-Null
Write-Output 'CPU ready. .\.venv\Scripts\python.exe -m pytest -q --basetemp .cache/pytest'
