# Install and run the P0/P1 add-on on Windows

## Correct location

Keep the dataset outside the project and keep all project code together:

```text
GEOGUESSER/
|-- .venv/
|-- geo_dataset/
|   |-- train/
|   |-- holdout_public/
|   `-- train_labels.csv
`-- geoguesser_reproducible_project/
    `-- geoguesser_final/       <-- run every command here
```

Do not paste anything into `train.py`. Do not run these files from a separate project.
The add-on contains only new filenames and is designed to be extracted directly into the
existing `geoguesser_final` directory.

## 1. Install the ZIP

Put `geoguesser_P0_P1_ADDON.zip` in the `GEOGUESSER` directory. In PowerShell:

```powershell
cd C:\path\to\GEOGUESSER\geoguesser_reproducible_project\geoguesser_final
Expand-Archive ..\..\geoguesser_P0_P1_ADDON.zip -DestinationPath . -Force
```

Verify that these return `True`:

```powershell
Test-Path .\scripts\cache_fold_outputs.py
Test-Path .\scripts\run_decoder_ablations.py
Test-Path .\src\geoguesser\advanced_decoders.py
Test-Path .\src\geoguesser\frozen_geoclip.py
```

## 2. Activate Python

The project requires Python 3.10--3.12. Python 3.11 is recommended.

```powershell
cd C:\path\to\GEOGUESSER
.\.venv\Scripts\Activate.ps1
python --version
```

If this prints Python 3.8, create a new environment instead of using it:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
```

Install the project from its root:

```powershell
cd .\geoguesser_reproducible_project\geoguesser_final
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -c "from geoguesser.advanced_decoders import decode_cplanet_numpy; from geoguesser.frozen_geoclip import TinyGpsTower; print('ADD-ON OK')"
```

## 3. Check the required existing files

```powershell
Test-Path .\outputs\efficientnet_b0_leaderboard_restart\fold_0\best_model.pt
Test-Path .\outputs\efficientnet_b0_leaderboard_restart\fold_0\reference_bank.npz
Test-Path .\outputs\efficientnet_b0_leaderboard_restart\fold_1\best_model.pt
Test-Path .\leaderboard\folds.csv
Test-Path ..\..\geo_dataset\train
```

All five checks must print `True`. If a checkpoint or reference bank is missing, stop: the
decoder experiments cannot be reconstructed from a CSV of final coordinates.

## 4. Run the recommended experiments

No EfficientNet training is performed. Fold 0 is used for selection; one frozen CPlaNet
configuration is then tested on fold 1.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_p0_p1_windows.ps1
```

For a smaller GPU batch:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_p0_p1_windows.ps1 -BatchSize 16
```

The script performs:

1. Fold-0 inference cache with no TTA and 180-degree TTA.
2. The complete P0 decoder grid and P1 fine/coarse/country intersection grid.
3. An honest frozen-image XYZ/RFF/RBF-64 screen on fold 0.
4. Fold-1 caching and confirmation of the single fold-0-selected CPlaNet setting.

Read these outputs after it finishes:

```text
outputs/p1/fold_0/dev/fine_decoder_ablation.csv
outputs/p1/fold_0/dev/cplanet_ablation.csv
outputs/p1/fold_0/dev/cplanet_selected.json
outputs/p1/fold_1/confirmation/cplanet_confirmation.csv
outputs/frozen_geoclip_honest_fold0/results.csv
```

Do not run `train.py` yet. These experiments are designed to decide whether CPlaNet or an
RBF contrastive objective deserves an expensive backbone-training run.
