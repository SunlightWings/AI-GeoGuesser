<<<<<<< HEAD
# Compact Europe Image Geolocation

A reproducible PyTorch project for the course challenge: predict `(latitude, longitude)` from 512×512 European road-scene images while keeping the submitted network at or below **5,000,000 parameters** and training only on the supplied data.

For the friends' five-fold board, start with **[LEADERBOARD_GUIDE.md](LEADERBOARD_GUIDE.md)**. It gives the exact shared-fold protocol, explains the old 8.16% result, and lists the CUDA commands for a clean run.

The default submission-safe model has **4,876,618 parameters** when fitted on all 11,758 labels (493 fine cells, 119 coarse cells). It uses an EfficientNet-B0 feature extractor with the original 1,000-class layer removed, then adds country, hierarchical geocell, spherical-regression, and cell-residual heads. Exact counts are recomputed and enforced before every run.

## What matters most

- The target is a coordinate, not a country. Country classification is only an auxiliary signal.
- Selection metric is lexicographic `(median haversine km, mean haversine km)`, matching the challenge.
- Longitude is never averaged directly; coordinates are represented as 3D unit vectors on the sphere.
- Fine/coarse geocells handle multimodal uncertainty better than plain latitude/longitude regression.
- The loader applies EXIF orientation correction. Optional 180-degree training/inference robustness addresses visibly inverted samples.
- The unofficial leaderboard uses the checked-in seed-42 `leaderboard/folds.csv` exactly and reports pooled OOF metrics.
- A spatially grouped split is supplied separately as a leakage stress test.
- EMA is disabled because the old EMA/BatchNorm mismatch collapsed evaluation features to an almost constant vector.
- External ImageNet weights are blocked by default. The clearly named research-only config requires an explicit acknowledgement.

## Repository map

```text
configs/                    experiment definitions
src/geoguesser/             data, geometry, cells, model, losses, engine, inference
scripts/train.py            fold or full-data training
scripts/cross_validate.py   reproducible K-fold loop
scripts/evaluate_leaderboard.py  strict pooled OOF validation and scoring
scripts/benchmark.py        fair config comparison on the same fold
scripts/pretrain_ssl.py     SimCLR on train/ only (rule-safe)
scripts/tune_inference.py   decoder/TTA calibration on a validation fold
scripts/predict.py          holdout inference and optional spherical ensemble
scripts/build_reference_bank.py / tune_retrieval.py / predict_retrieval.py
                            guarded train-only retrieval path, if approved
scripts/validate_submission.py  exact CSV contract checks
tests/                      geometry, split, cell, model, and loss tests
notebooks/                  readable walkthrough (thin wrapper over tested code)
report/main.tex             1–2 page write-up template
EXPERIMENTS_README.md       experiment rationale and completed checks
experiments.csv             append-only experiment table
```

## 1. Environment

Python 3.10–3.12 is supported. On an NVIDIA machine, first use the official [PyTorch installer](https://pytorch.org/get-started/locally/) to select the wheel matching the installed driver. Then:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

Expected test result: `14 passed`.

In this workspace the data is two levels above this project:

```text
GeoGuesser/
├── geo_dataset/
│   ├── train/                  # 11,758 .jpg files
│   ├── train_labels.csv
│   └── holdout_public/         # 2,400 .jpg files
└── geoguesser_reproducible_project/geoguesser_final/
```

Audit before spending GPU hours:

```bash
python scripts/audit_dataset.py --data-root ../../geo_dataset
```

## 2. First complete run

The MobileNet experiment is the fastest serious baseline:

```bash
python scripts/train.py \
  --config configs/mobilenet_v3_small_scratch.yaml \
  --data-root ../../geo_dataset \
  --fold 0
```

The main submission-safe model is:

```bash
python scripts/train.py \
  --config configs/efficientnet_b0_scratch.yaml \
  --data-root ../../geo_dataset \
  --fold 0
```

Each run writes `history.csv`, its exact train/validation rows, out-of-fold predictions, a guarded checkpoint, and `run_summary.json`. Override a value without copying a config:

```bash
python scripts/train.py \
  --config configs/efficientnet_b0_scratch.yaml \
  --fold 0 \
  --set train.stages.0.epochs=5
```

For list elements, editing a copied YAML config is clearer than command-line overrides.

## 3. Unofficial leaderboard cross-validation

The leaderboard configs use the shared fold file by filename. Use the same fold for early decisions:

```bash
python scripts/benchmark.py \
  --configs \
    configs/mobilenet_v3_small_scratch.yaml \
    configs/efficientnet_b0_scratch.yaml \
  --data-root ../../geo_dataset \
  --fold 0 \
  --output-dir outputs/benchmark
```

Then run five folds for a finalist. The command automatically creates and scores all 11,758 OOF predictions:

```bash
python scripts/cross_validate.py \
  --config configs/efficientnet_b0_scratch.yaml \
  --data-root ../../geo_dataset \
  --device cuda
```

See [LEADERBOARD_GUIDE.md](LEADERBOARD_GUIDE.md) for generated files, manual re-evaluation, and restart instructions.

Run the spatial stress test on at least fold 0:

```bash
python scripts/train.py \
  --config configs/spatial_validation.yaml \
  --data-root ../../geo_dataset \
  --fold 0
```

Record every completed run in `experiments.csv`. Do not choose by country accuracy; choose the lowest cross-validated median km, breaking ties with mean km.

## 4. Rule-safe self-supervised initialization

Because external pretraining is risky under the stated rule, the strongest compliant initialization available here is SimCLR using only supplied images. For honest leaderboard CV, pretrain separately with each fold's `train_rows.csv`; pretraining once on all rows would expose validation images to representation learning.

```bash
python scripts/pretrain_ssl.py \
  --data-root ../../geo_dataset \
  --labels outputs/efficientnet_b0_leaderboard/fold_0/train_rows.csv \
  --backbone efficientnet_b0 \
  --epochs 100 \
  --batch-size 128 \
  --output outputs/ssl/fold_0/efficientnet_b0_backbone.pt

python scripts/train.py \
  --config configs/efficientnet_b0_ssl.yaml \
  --data-root ../../geo_dataset \
  --fold 0 \
  --set model.backbone_checkpoint=outputs/ssl/fold_0/efficientnet_b0_backbone.pt
```

Repeat with the matching `fold_N/train_rows.csv` and a separate SSL checkpoint for every fold. If memory is limited, reduce SimCLR batch size; contrastive learning generally benefits from more negatives.

## 5. Calibrate decoding without touching holdout labels

After training fold 0:

```bash
python scripts/tune_inference.py \
  --checkpoint outputs/efficientnet_b0_leaderboard/fold_0/best_model.pt \
  --validation-csv outputs/efficientnet_b0_leaderboard/fold_0/validation_rows.csv \
  --image-dir ../../geo_dataset/train \
  --output outputs/efficientnet_b0_leaderboard/fold_0/inference_tuning.csv
```

Repeat on multiple folds and prefer settings that are stable, not the single lucky minimum. Transfer the selected decoder values into the final YAML or pass them to `predict.py`.

### Optional train-only retrieval (ask first)

A non-parametric embedding bank can exploit visually similar routes without adding learned parameters, and it uses only supplied training images. The wording “submitted model” may nevertheless treat that bank as part of the system. Ask the instructors: **“May the submitted inference pipeline include a fixed nearest-neighbour embedding bank built only from the labelled training images, if the neural network remains below 5M parameters?”**

If approved, build a fold bank from that fold's `train_rows.csv`, tune only on its validation rows, and then build the final bank from all labels:

```bash
python scripts/build_reference_bank.py \
  --checkpoint outputs/efficientnet_b0_leaderboard/fold_0/best_model.pt \
  --labels outputs/efficientnet_b0_leaderboard/fold_0/train_rows.csv \
  --image-dir ../../geo_dataset/train \
  --output outputs/efficientnet_b0_leaderboard/fold_0/reference_bank.npz \
  --acknowledge-reference-bank-rule

python scripts/tune_retrieval.py \
  --checkpoint outputs/efficientnet_b0_leaderboard/fold_0/best_model.pt \
  --reference-bank outputs/efficientnet_b0_leaderboard/fold_0/reference_bank.npz \
  --validation-csv outputs/efficientnet_b0_leaderboard/fold_0/validation_rows.csv \
  --image-dir ../../geo_dataset/train \
  --output outputs/efficientnet_b0_leaderboard/fold_0/retrieval_tuning.csv \
  --acknowledge-reference-bank-rule
```

The default prediction path deliberately remains neural-only until permission is explicit.

## 6. Train the final model and create predictions

After model selection, set the epoch schedule to the cross-validated choice and fit all labels:

```bash
python scripts/train.py \
  --config configs/efficientnet_b0_scratch.yaml \
  --data-root ../../geo_dataset \
  --fold full \
  --run-dir outputs/final

python scripts/predict.py \
  --checkpoint outputs/final/final_model.pt \
  --data-root ../../geo_dataset \
  --output outputs/final/predictions.csv \
  --tta rot180

python scripts/validate_submission.py \
  --predictions outputs/final/predictions.csv \
  --holdout-dir ../../geo_dataset/holdout_public
```

The validator requires the exact header `filename,pred_lat,pred_lng`, exactly 2,400 unique filenames, finite values, legal coordinate ranges, and an exact filename match.

After completing the `TODO` fields in `report/main.tex` (and optionally compiling it), package the exact deliverables:

```bash
python scripts/package_submission.py \
  --checkpoint outputs/final/final_model.pt \
  --predictions outputs/final/predictions.csv \
  --report report/main.tex \
  --output submission.zip
```

Packaging refuses an oversized model, an externally pretrained checkpoint, a malformed row count, or an unfinished report.

## 7. Pretrained-CNN baseline (not submission-safe by default)

The requested pretrained baseline is runnable, but ImageNet is external training data. The command is deliberately noisy:

```bash
python scripts/train.py \
  --config configs/efficientnet_b0_imagenet_research_only.yaml \
  --data-root ../../geo_dataset \
  --fold 0 \
  --acknowledge-noncompliant-experiment
```

Its checkpoint is tagged `rule_compliant=false`, and inference refuses it unless `--allow-noncompliant-checkpoint` is supplied. Do not submit it without written instructor approval.

## Recommended experiment order

1. MobileNet scratch, fold 0: pipeline and cheap baseline.
2. EfficientNet scratch, fold 0: capacity gain under the cap.
3. Fine-cell sizes 20, 24, 32: resolution versus classification difficulty.
4. `tta=none` versus `rot180`: test the rotation artifact hypothesis.
5. EfficientNet + supplied-data SimCLR.
6. If approved, cell-gated retrieval.
7. Five-fold comparison of the two best settings.
8. Spatial fold 0 as the generalization warning light.
9. Full-data refit and one final rule-compliant checkpoint.

Three seeds for the finalists are ideal if compute permits. The holdout set is never used for training, pseudo-labeling, hyperparameter selection, or SSL.

## Runtime and reproducibility

The code records hardware and wall time in every `run_summary.json`. The final README submitted to the course should replace this paragraph with your measured hardware, exact runtime, selected checkpoint, and metrics. Runtime cannot be truthfully estimated from the attached samples alone; image size, GPU, and storage throughput dominate it.

Deterministic mode can be enabled in a config, but exact bitwise reproducibility across GPU models is not guaranteed. Seeds, folds, package versions, data rows, model metadata, and output validation are all persisted.

## Scientific basis

The method follows the geocell-classification insight of [PlaNet](https://arxiv.org/abs/1602.05314), adds a compact hierarchical coordinate decoder inspired by modern geolocation systems such as [PIGEON](https://openaccess.thecvf.com/content/CVPR2024/html/Haas_PIGEON_Predicting_Image_Geolocations_CVPR_2024_paper.html), and offers supplied-data-only [SimCLR](https://proceedings.mlr.press/v119/chen20j.html) initialization. The backbone implementation and official parameter reference come from [Torchvision EfficientNet-B0](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html).
=======
Contains AI geoguesser model
>>>>>>> origin/main
