# P0/P1 and frozen-GeoCLIP report

This report separates valid out-of-fold results from development-fold and diagnostic
screens. Only the first row is currently a leaderboard-valid five-fold score.

## Verified checkpoint result

The uploaded full archive contained five complete EfficientNet-B0 fold runs under
`outputs/efficientnet_b0_leaderboard_restart`. Their train/validation membership matched
`leaderboard/folds.csv`, the pooled file contained all 11,758 filenames exactly once, and
the ten best/final checkpoint containers passed integrity checks. Each model has 4,763,666
parameters.

| Experiment | Scope | Median km | Mean km | <200 km | <750 km |
|---|---|---:|---:|---:|---:|
| Original EfficientNet-B0 | True five-fold OOF | 317.834 | 665.563 | 40.42% | 69.56% |
| Best fine decoder grid | Fold-0 development | 300.114 | 677.362 | 40.60% | 68.84% |
| Frozen image retrieval | Fold-0 development | 269.089 | 658.861 | 44.52% | 68.75% |
| XYZ GPS tower, 12 epochs | Diagnostic | 425.163 | 653.063 | 19.73% | 73.64% |
| RFF GPS tower, 12 epochs | Diagnostic | 278.604 | 533.218 | 40.39% | 77.76% |
| RBF-64 GPS tower, 12 epochs | Diagnostic | 228.569 | 522.584 | 44.56% | 79.42% |
| XYZ GPS tower, 30 epochs | Diagnostic | 378.593 | 613.159 | 23.47% | 76.79% |
| RFF GPS tower, 30 epochs | Diagnostic | 251.615 | 515.847 | 43.41% | 78.74% |
| RBF-64 GPS tower, 30 epochs | Diagnostic | **224.340** | 520.649 | 47.32% | 79.25% |
| Frozen image k-NN control | Diagnostic | 227.503 | 541.252 | 47.32% | 74.49% |

The GPS-tower diagnostic froze all image embeddings and trained only 16.9k--20.8k
parameters. It is not honest OOF: the fold-0 image encoder had already trained on the
subset used to validate GPS-tower encodings. It falsifies raw XYZ as the leading encoding,
but RBF-64 must still be checked against genuinely unseen fold-0 embeddings.

## Best P0 setting

```text
TTA=rot180, top_cells=1, temperature=0.65, country_gate=0.5,
residual_scale=0.0, global_blend=0.05
```

This setting is essentially tied with the original fold-0 score. Decoder tuning alone is
not a plausible route from 318 km to 50 km.

## CPlaNet status

The fine/coarse/country product-of-experts decoder is implemented. It scores each nonempty
fine/coarse intersection using:

```text
log p(fine) + coarse_weight * log p(coarse) + country_gate * log p(country)
```

Real tuning could not be run in the remote environment because the archive did not contain
the source images or cached validation logits. `scripts/run_p0_p1_windows.ps1` now creates
the caches on the user's PC, tunes only fold 0, and confirms the frozen selection on fold 1.

## Holdout predictions

There was no full-data checkpoint in the archive and the holdout images were absent.
Therefore no new 2,400-row submission was generated. The existing `my_predictions.csv`
and `leaderboard/predictions_from_model.csv` were collapsed near Iceland and should not be
submitted.

## Recommendation

Run the supplied PowerShell workflow. If CPlaNet improves both fold 0 and fold 1, preserve
it as a decoder. If RBF-64 wins the honest fold-0 GPS screen, the next retraining ablation
should add an RBF image/GPS contrastive objective without changing the backbone.
