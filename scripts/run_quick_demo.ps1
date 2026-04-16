param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$PassThroughArgs
)

$ErrorActionPreference = "Stop"

$python = ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
  throw "Expected existing project venv interpreter at '$python'."
}

Write-Host "Using Python: $python"
& $python scripts/quick_demo.py @PassThroughArgs
exit $LASTEXITCODE
