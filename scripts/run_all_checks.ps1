$ErrorActionPreference = 'Stop'

$python = 'python'
if (Test-Path '.venv\\Scripts\\python.exe') {
  $python = '.venv\\Scripts\\python.exe'
}

Write-Host 'Running release gate...'
& $python scripts/run_release_gate.py --profile dev --strict
if ($LASTEXITCODE -ne 0) {
  Write-Host "Release gate failed with exit code $LASTEXITCODE."
  exit $LASTEXITCODE
}

Write-Host 'Release gate passed.'
