$ErrorActionPreference = 'Stop'

$python = 'python'
if (Test-Path '.venv\Scripts\python.exe') {
  $python = '.venv\Scripts\python.exe'
}

Write-Host "Using Python: $python"
& $python scripts/smoke_real_rag.py --model-path . --top-k 3 --max-new-tokens 80 --strict
