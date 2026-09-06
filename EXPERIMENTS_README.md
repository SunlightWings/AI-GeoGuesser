# Experiment log and rationale

This file separates verified facts from experiments that still require the full image corpus and the student's GPU. Blank leaderboard claims are scientifically preferable to imaginary victories.

## Completed during codebase construction

### E00 — Specification and data audit

- Parsed the Markdown and PDF challenge specifications; they agree.
- Confirmed 11,758 unique label rows, 12 countries, valid coordinate ranges, and near-balanced country counts (864–1,000 images each).
- Confirmed the real objective: median haversine distance; mean is the tie-break.
- Confirmed hard rules: at most 5,000,000 submitted model parameters; no external or generated training data.
- Inspected the supplied samples. At least one image is inverted, motivating EXIF correction and a controlled rotation experiment.

Decision: optimize coordinates on the sphere, keep country as an auxiliary head, and explicitly guard rule-sensitive paths.

### E01 — Historical label-only sanity baselines (random fold 0, seed 2026)

These predate the shared leaderboard fold audit. They use labels only and are diagnostics, not deployable image models or leaderboard results.

| Diagnostic | Median km | Mean km | <200 km | <750 km | Interpretation |
|---|---:|---:|---:|---:|---|
| Global training medoid | 1203.43 | 1321.57 | 2.59% | 20.49% | A coordinate-blind floor |
| Oracle true-country medoid | 291.58 | 336.72 | 26.23% | 96.81% | Country recognition alone cannot win |

Decision: direct country classification is necessary but radically insufficient.

### E02 — Historical geocell quantization oracle (random fold 0)

The validation coordinate is assigned to its nearest training-fitted cell **using its true country**. This measures quantization resolution only, not image-recognition performance.

| Target samples/cell | Fine cells | Median km | Mean km | <200 km |
|---:|---:|---:|---:|---:|
| 16 | 588 | 23.45 | 26.58 | 100.00% |
| 24 | 389 | 30.32 | 33.83 | 100.00% |
| 32 | 294 | 36.99 | 40.48 | 100.00% |
| 45 | 211 | 45.11 | 49.05 | 99.96% |
| 64 | 143 | 59.13 | 63.10 | 99.57% |
| 96 | 95 | 78.31 | 82.25 | 98.09% |

Decision: 24 samples per fine cell gives a strong 30 km quantization ceiling without making the classifier as data-starved as the 16-sample partition. A learned cell residual can refine within each cell.

### E03 — Architecture budget

Fitting cells on all 11,758 labels produces 493 fine and 119 coarse cells.

| Backbone | Exact submitted parameters | Cap status |
|---|---:|---|
| MobileNet-V3-Small hierarchical model | 1,615,854 | Pass |
| EfficientNet-B0 hierarchical model | 4,876,618 | Pass, 123,382 spare |

The counts include the complete backbone and every inference head. The SimCLR projector is discarded and is not part of the submitted model.

Decision: MobileNet is the fast debugging baseline; EfficientNet uses the available budget more fully.

### E04 — End-to-end smoke experiment

- Data: 10 attached labelled images and 2 attached images absent from the label CSV.
- Model: MobileNet-V3-Small, 32-dimensional head, 947,135 parameters.
- Run: one 64×64 CPU epoch; 0.70 s training in the validation container.
- Result: checkpoint saved, both coordinates predicted, exact two-row CSV validation passed.
- Unit tests: 10/10 passed.

This validates software flow only. The tiny loss and predictions say nothing about leaderboard accuracy.

### E05 — Review of the earlier walkthrough notebook

Useful ideas retained: country-conditioned K-means cells, multi-scale heads, spherical decoding, progressive resolution, and a cell-aware batch sampler. EMA was removed after its BatchNorm-state mismatch was shown to collapse validation features.

Changes made in this repository:

- Modularized notebook code into testable scripts and package modules.
- Added learned within-cell residuals and a differentiable decoded-coordinate loss.
- Added strict pretrained-data compliance metadata and inference blocks.
- Removed retrieval from the default submission path because a stored embedding bank may be interpreted as part of the model and needs instructor clarification; a guarded, train-only implementation is included for use if approved.
- Added spatial validation, decoder calibration, exact submission validation, SSL pretraining on train-only images, and a reproducible benchmark driver.
- Added strict use of the downloaded seed-42 folds, held-out-only fold artifacts, and pooled OOF evaluation.

## Experiments to run on the full images

Fill `experiments.csv` after each run. Do not replace failed runs with a polished retrospective; the grading rubric explicitly values reasoning about failures.

### E10 — MobileNet scratch baseline

Hypothesis: quickest reliable reference, probably underfits fine geographic texture. Run fold 0 first.

### E11 — EfficientNet scratch

Hypothesis: higher capacity improves vegetation, road furniture, camera-source, and marking cues. If loss remains near random for many epochs, scratch training is data-limited; continue to E13.

### E12 — Cell granularity

Compare target sizes 20, 24, and 32 on the same fold. Smaller cells reduce quantization error but increase class confusion and parameter count. Reject any configuration over the cap before training.

### E13 — Train-only SimCLR initialization

Hypothesis: representation pretraining on the supplied road scenes improves optimization without violating the data rule. For leaderboard CV, pretrain separately on each fold's training rows so the validation images are not seen by SSL. Compare against E11 with identical supervised settings.

### E14 — Rotation robustness

Compare no TTA, 180-degree TTA, and (only if useful) four-rotation TTA using `tune_inference.py`. A rotated example exists, but averaging implausible orientations may still hurt normal images.

### E15 — Shared leaderboard versus spatial validation

Large degradation on the spatial split indicates route/location memorization. Report both; use the downloaded seed-42 split for the unofficial board and describe the spatial result honestly.

### E16 — External pretrained research-only baseline

Purpose: quantify the value of external representation learning. This is not submission-eligible under the literal rule. If it dominates, ask the instructors whether pretrained weights are allowed; do not infer permission from performance.

### E17 — Cell-gated train-only retrieval (permission required)

Build a bank from the fold's training rows, never its validation rows. Tune neighbour count and blend on validation. This may exploit route similarity, but submit it only if the instructors confirm that a non-parametric train-only bank is allowed under the model-size rule.

## Stop conditions

- Never inspect or pseudo-label holdout images for training decisions.
- Never average latitude/longitude arithmetically.
- Never submit an external-pretrained checkpoint unless permission is explicit.
- Never ensemble checkpoints whose aggregate interpretation could exceed the parameter cap without written clarification.
- Prefer a stable five-fold gain over a tiny single-fold win.
