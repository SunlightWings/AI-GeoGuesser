# P0/P1 add-on manifest

This package is additive. It does not contain or replace `train.py`, model definitions,
configs, checkpoints, or existing OOF predictions.

## New code

- `src/geoguesser/advanced_decoders.py`
- `src/geoguesser/frozen_geoclip.py`
- `scripts/cache_fold_outputs.py`
- `scripts/run_decoder_ablations.py`
- `scripts/screen_frozen_geoclip.py`
- `scripts/summarize_p0_grid.py`
- `scripts/run_p0_p1_windows.ps1`
- `tests/test_advanced_decoders.py`

## Documentation and recorded results

- `P0_P1_INSTALL_AND_RUN_WINDOWS.md`
- `P0_P1_GEOCLIP_REPORT.md`
- `outputs/p0_existing_fold0/best_decoder.json`
- `outputs/frozen_geoclip_screen_cached_bank_30ep/results.csv`

Extract the archive into the existing `geoguesser_final` directory. Directory merging is
expected; manual line-by-line source-code merging is not.
