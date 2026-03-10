param(
  [string]$PythonBin = "python",
  [string]$RunId = "cloud_run_001",
  [string]$BaselineReport = "artifacts/deepset_eval/latest_deepset_report.json"
)

$ErrorActionPreference = "Stop"
$dataDir = "artifacts/pitheta_data/$RunId"
$trainDir = "artifacts/pitheta_train/$RunId/best"

& $PythonBin scripts/eval_pitheta.py `
  --checkpoint $trainDir `
  --data-dir $dataDir `
  --baseline-report $BaselineReport `
  --strict-gates
