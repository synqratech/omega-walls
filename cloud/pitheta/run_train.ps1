param(
  [string]$PythonBin = "python",
  [string]$RunId = "cloud_run_001"
)

$ErrorActionPreference = "Stop"
$dataDir = "artifacts/pitheta_data/$RunId"
$trainDir = "artifacts/pitheta_train/$RunId"

& $PythonBin scripts/build_pitheta_dataset.py `
  --registry cloud/pitheta/dataset_registry.yml `
  --output-dir $dataDir `
  --seed 41 `
  --strict

& $PythonBin scripts/train_pitheta_lora.py `
  --train-config cloud/pitheta/train_config.yml `
  --data-dir $dataDir `
  --output-dir $trainDir
