param([switch]$ProtocolTests)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$env:PYTHONUTF8 = '1'
$env:PIP_CACHE_DIR = Join-Path (Get-Location) '.cache/pip'
$env:TEMP = Join-Path (Get-Location) '.cache/tmp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
New-Item -ItemType Directory -Force -Path 'outputs/runs/hai','outputs/runs/thinh','outputs/packages','models' | Out-Null
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) { python -m venv .venv }
& '.venv/Scripts/python.exe' -m pip install -r requirements-dev.lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Cài dependency CPU thất bại' }
& '.venv/Scripts/python.exe' -m pip install -e . --no-deps
if ($LASTEXITCODE -ne 0) { throw 'Cài source thất bại' }
if ($ProtocolTests) {
    # Tiny random models and tokenizer fixtures only; no business inference.
    & '.venv/Scripts/python.exe' -m pip install 'torch>=2.7,<3' --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'Cài PyTorch CPU thất bại' }
    & '.venv/Scripts/python.exe' -m pip install -e '.[test-llm]'
    if ($LASTEXITCODE -ne 0) { throw 'Cài dependency kiểm thử giao thức thất bại' }
}
Write-Output 'CPU dev ready. Run full-checkpoint inference on Colab/Kaggle/rented GPU.'
