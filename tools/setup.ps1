param([switch]$LLM)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$env:PYTHONUTF8 = '1'
$env:PIP_CACHE_DIR = Join-Path (Get-Location) '.cache/pip'
$env:TEMP = Join-Path (Get-Location) '.cache/tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) { python -m venv .venv }
if ($LLM) { & '.venv/Scripts/python.exe' -m pip install -e '.[llm,dev]' }
else { & '.venv/Scripts/python.exe' -m pip install -e '.[dev]' }
if ($LASTEXITCODE -ne 0) { throw 'Cài đặt thất bại' }
