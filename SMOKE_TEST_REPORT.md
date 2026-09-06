# Verification report

Latest run date: 2026-08-18

- Static compilation: passed for every package module and script.
- Unit tests: 14 passed, including strict shared-fold and pooled-OOF checks.
- Full-label architecture construction: EfficientNet model has 4,876,618 parameters; MobileNet model has 1,615,854.
- Tiny integration data: 24 labelled images and 2 holdout images.
- Tiny training: one CPU epoch completed and wrote a 947,982-parameter checkpoint with the repaired non-EMA path.
- Inference: two finite spherical coordinates produced.
- CSV contract: exact filenames, columns, coordinate ranges, and row count passed.
- Optional retrieval path: a 10-row train-only embedding bank was built, queried, and produced a second valid two-row CSV.
- Retrieval grid-search script completed all decoder combinations. Its tiny-data distances are deliberately not treated as accuracy evidence.

This report establishes executability, not full-dataset accuracy. The full 11,758-image dataset was audited, but a new five-fold CUDA run is intentionally left to the user because it is a multi-hour experiment.
