# Unofficial leaderboard workflow

The unofficial board uses the supplied training set differently from the official challenge:

- five country-stratified folds from `leaderboard/folds.csv` (seed 42);
- five separately trained models;
- each model predicts only the fold it did **not** train on;
- metrics are computed once over all 11,758 pooled out-of-fold (OOF) predictions.

The live protocol is at <https://geo-location-leaderboard.netlify.app/>. The checked-in
`leaderboard/folds.csv` was verified byte-for-byte against the site's download on
2026-08-18 (SHA-256 `b96837109ef7afbba95b7ed661a1a9e250873b6018f6a1e85930396a31f2af47`).

## Why the old result was about 8%

`leaderboard/predictions_from_model.csv` is not a valid five-fold OOF result. The old
evaluator loaded one checkpoint and ran it over every training image, including rows that
checkpoint had trained on. The training script also saved an exponential moving average
(EMA) whose averaged convolution weights were incompatible with its BatchNorm statistics.
In evaluation mode, that checkpoint produced effectively the same feature for every image
and predicted Iceland almost everywhere. Its pooled `<750 km` rate is 8.16% only because
Iceland makes up roughly that fraction of the data.

The repaired configs disable EMA, use the shared folds by filename, use only 0/180-degree
orientation augmentation, use the same TTA during validation and prediction, and use a
smaller geographically meaningful residual scale. The evaluator now rejects missing folds,
wrong-fold rows, duplicate filenames, invalid coordinates, non-compliant checkpoints, and
models over the 5M parameter cap.

## 1. Set up and verify CUDA

Run every command below from `geoguesser_final`:

```bash
cd /home/prabin/Documents/GeoGuesser/geoguesser_reproducible_project/geoguesser_final
source ../../.venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.version.cuda)"
python scripts/audit_dataset.py --data-root ../../geo_dataset
sha256sum leaderboard/folds.csv
```

Expected checks are 14 passing tests, 11,758 training images, 2,400 holdout images, and the
fold checksum printed above. Stop before training if CUDA is unavailable or the checksum is
different.

## 2. Run one fold as a sanity check

MobileNet is the quicker diagnostic run:

```bash
python scripts/train.py \
  --config configs/mobilenet_v3_small_scratch.yaml \
  --data-root ../../geo_dataset \
  --fold 0 \
  --device cuda
```

Inspect `outputs/mobilenet_v3_small_leaderboard/fold_0/run_summary.json` and `history.csv`.
Country accuracy and coordinate predictions should no longer be constant. This is only a
single-fold diagnostic and must not be posted as the five-fold score.

## 3. Train and score all five folds

For the faster baseline:

```bash
python scripts/cross_validate.py \
  --config configs/mobilenet_v3_small_scratch.yaml \
  --data-root ../../geo_dataset \
  --device cuda
```

For the higher-capacity model under the 5M limit:

```bash
python scripts/cross_validate.py \
  --config configs/efficientnet_b0_scratch.yaml \
  --data-root ../../geo_dataset \
  --device cuda
```

The command trains folds 0 through 4 sequentially and then automatically writes:

- `outputs/<experiment>/oof_predictions.csv` — all 11,758 OOF rows;
- `outputs/<experiment>/leaderboard_metrics.json` — pooled and per-fold metrics;
- `outputs/<experiment>/cross_validation.csv` — individual fold best metrics;
- `outputs/<experiment>/fold_N/` — checkpoint, history, exact row split, and fold OOF CSV.

Re-evaluate existing completed fold directories without using the GPU:

```bash
python scripts/evaluate_leaderboard.py \
  --run-dir outputs/efficientnet_b0_leaderboard \
  --leaderboard-csv leaderboard/folds.csv \
  --output outputs/efficientnet_b0_leaderboard/oof_predictions.csv \
  --metrics-output outputs/efficientnet_b0_leaderboard/leaderboard_metrics.json
```

Enter the four printed **Overall** values on the website. The two rates are already printed
as percentages. Use the printed per-fold model parameter count, not five times that value.

If a long run is interrupted, train only the missing folds, then run the evaluator command:

```bash
python scripts/cross_validate.py \
  --config configs/efficientnet_b0_scratch.yaml \
  --data-root ../../geo_dataset \
  --device cuda \
  --folds 2 3 4
```

Do not use `leaderboard/predictions_from_model.csv` or the old checkpoints for a score.

## 4. Optional decoder tuning

Tune on a development fold, transfer the stable settings into the YAML, and retrain all five
folds with that one frozen configuration:

```bash
python scripts/tune_inference.py \
  --checkpoint outputs/efficientnet_b0_leaderboard/fold_0/best_model.pt \
  --validation-csv outputs/efficientnet_b0_leaderboard/fold_0/validation_rows.csv \
  --image-dir ../../geo_dataset/train \
  --output outputs/efficientnet_b0_leaderboard/fold_0/inference_tuning.csv \
  --device cuda
```

Do not select separate decoder settings for each fold; that would fit the leaderboard folds.

## 5. Official holdout later

After method selection, train one model on every labelled row and predict the 2,400-image
official holdout:

```bash
python scripts/train.py \
  --config configs/efficientnet_b0_scratch.yaml \
  --data-root ../../geo_dataset \
  --fold full \
  --run-dir outputs/final \
  --device cuda

python scripts/predict.py \
  --checkpoint outputs/final/final_model.pt \
  --data-root ../../geo_dataset \
  --output outputs/final/predictions.csv \
  --device cuda

python scripts/validate_submission.py \
  --predictions outputs/final/predictions.csv \
  --holdout-dir ../../geo_dataset/holdout_public
```

That official `predictions.csv` has 2,400 rows. It is a different artifact from the
11,758-row unofficial OOF file.
