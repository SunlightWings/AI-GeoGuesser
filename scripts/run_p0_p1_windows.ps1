param(
    [string]$Device = "cuda",
    [int]$BatchSize = 32,
    [int]$NumWorkers = 4,
    [int]$GeoClipEpochs = 30
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

function Require-Path([string]$Path, [string]$Description) {
    if (-not (Test-Path $Path)) {
        throw "Missing $Description`: $Path"
    }
}

function Run-Python([string]$Name, [string[]]$Arguments) {
    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

$DataTrain = (Resolve-Path (Join-Path $ProjectRoot "..\..\geo_dataset\train")).Path
$RunRoot = Join-Path $ProjectRoot "outputs\efficientnet_b0_leaderboard_restart"
$FoldsCsv = Join-Path $ProjectRoot "leaderboard\folds.csv"
$Fold0 = Join-Path $RunRoot "fold_0"
$Fold1 = Join-Path $RunRoot "fold_1"

Require-Path $DataTrain "training image directory"
Require-Path (Join-Path $Fold0 "best_model.pt") "fold-0 checkpoint"
Require-Path (Join-Path $Fold1 "best_model.pt") "fold-1 checkpoint"
Require-Path (Join-Path $Fold0 "reference_bank.npz") "fold-0 reference bank"
Require-Path $FoldsCsv "canonical folds CSV"

Run-Python "Cache fold 0 outputs" @(
    "scripts/cache_fold_outputs.py",
    "--checkpoint", (Join-Path $Fold0 "best_model.pt"),
    "--validation-csv", (Join-Path $Fold0 "validation_rows.csv"),
    "--image-dir", $DataTrain,
    "--output-dir", "outputs/p1/fold_0/cache",
    "--tta", "none", "rot180",
    "--batch-size", "$BatchSize",
    "--num-workers", "$NumWorkers",
    "--device", $Device
)

Run-Python "Tune fine and CPlaNet decoders on fold 0" @(
    "scripts/run_decoder_ablations.py",
    "--caches",
    "outputs/p1/fold_0/cache/validation_none.npz",
    "outputs/p1/fold_0/cache/validation_rot180.npz",
    "--train-csv", (Join-Path $Fold0 "train_rows.csv"),
    "--output-dir", "outputs/p1/fold_0/dev"
)

Run-Python "Honest frozen-image GPS encoder screen on fold 0" @(
    "scripts/screen_frozen_geoclip.py",
    "--reference-bank", (Join-Path $Fold0 "reference_bank.npz"),
    "--validation-cache", "outputs/p1/fold_0/cache/validation_rot180.npz",
    "--folds-csv", $FoldsCsv,
    "--output-dir", "outputs/frozen_geoclip_honest_fold0",
    "--epochs", "$GeoClipEpochs",
    "--methods", "xyz", "rff", "rbf"
)

Run-Python "Cache fold 1 outputs" @(
    "scripts/cache_fold_outputs.py",
    "--checkpoint", (Join-Path $Fold1 "best_model.pt"),
    "--validation-csv", (Join-Path $Fold1 "validation_rows.csv"),
    "--image-dir", $DataTrain,
    "--output-dir", "outputs/p1/fold_1/cache",
    "--tta", "none", "rot180",
    "--batch-size", "$BatchSize",
    "--num-workers", "$NumWorkers",
    "--device", $Device
)

Run-Python "Confirm the frozen CPlaNet setting on fold 1" @(
    "scripts/run_decoder_ablations.py",
    "--caches",
    "outputs/p1/fold_1/cache/validation_none.npz",
    "outputs/p1/fold_1/cache/validation_rot180.npz",
    "--train-csv", (Join-Path $Fold1 "train_rows.csv"),
    "--output-dir", "outputs/p1/fold_1/confirmation",
    "--selected-config", "outputs/p1/fold_0/dev/cplanet_selected.json"
)

Write-Host ""
Write-Host "Finished. Read these three files:" -ForegroundColor Green
Write-Host "  outputs/p1/fold_0/dev/cplanet_ablation.csv"
Write-Host "  outputs/p1/fold_1/confirmation/cplanet_confirmation.csv"
Write-Host "  outputs/frozen_geoclip_honest_fold0/results.csv"
