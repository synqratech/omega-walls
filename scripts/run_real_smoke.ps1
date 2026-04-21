$ErrorActionPreference = 'Stop'

$python = '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
  throw "Expected existing project venv interpreter at '$python'. Do not fallback to a random system python."
}

Write-Host "Using Python: $python"
& $python scripts/smoke_real_rag.py --model-path . --top-k 3 --max-new-tokens 80 --strict
