$ErrorActionPreference = 'Stop'

 $python = 'python'
 if (Test-Path '.venv\\Scripts\\python.exe') {
   $python = '.venv\\Scripts\\python.exe'
 }

Write-Host 'Running pytest...'
& $python -m pytest

Write-Host 'Running quick OSS eval...'
& $python -m omega eval --suite quick --strict

Write-Host 'All local checks passed.'
