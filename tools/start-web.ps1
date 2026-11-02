param([int]$Port = 7860)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$env:PYTHONUTF8 = '1'
$env:HF_HOME = Join-Path (Get-Location) '.cache/huggingface'
$env:TORCH_HOME = Join-Path (Get-Location) '.cache/torch'
& '.venv/Scripts/python.exe' -m datamate serve --workspace . --port $Port
