# ONE project, TWO machines, 12-hour deadline

## Read this first

There is ONE ZIP for BOTH computers. No external image training, teacher models never in
the submitted neural model. All final neural Parameters (including frozen ones) are counted.
Do NOT paste code into train.py. Use the new commands below instead of old train.py.

The latest 358 MB uploaded project is no longer available in this remote workspace. This
ZIP includes the earlier core as a starter for university and an additive deadline_upgrade
folder for your newer home project. Existing architecture, loss, checkpoint and inference
APIs are reused. No supplied checkpoint or fold manifest is fabricated or included.

**Home installation:** extract the ZIP into a temporary location. Copy ONLY its
`geoguesser_final/deadline_upgrade` folder into your CURRENT `geoguesser_final` directory.
Do NOT overwrite your src/, scripts/, configs/, outputs/, or leaderboard/ directories.

**University installation:** extract the entire ZIP. Use its `geoguesser_final` as the
project. Bring the ORIGINAL supplied dataset and copy `leaderboard/folds.csv` from HOME
into the university `geoguesser_final/leaderboard/` directory. This tiny file is essential.
Do not generate substitute folds, even with the same seed.

On either PC, run commands from the directory containing pyproject.toml and deadline_upgrade.
The dataset can be anywhere: the examples use a PowerShell variable $Data.

## Setup (PowerShell; Windows)

Use your working CUDA environment at home. On university, create a Python 3.11 environment.
Do not copy virtual environments between PCs. If the university uses Linux, use
`python3 -m venv .venv` and `source .venv/bin/activate`, and replace $Data with its path.

```powershell
# Start inside geoguesser_final on either PC
$Data = "C:\path\to\geo_dataset"
python --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
# If these imports already work, preserve the working CUDA installation.
python -m pip install --no-deps -e .
python deadline_upgrade/test_contracts.py
python deadline_upgrade/smoke_torch.py
```

On a FRESH university environment, before those tests:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e .
```

If the driver cannot support that CUDA build, use the university's existing working
PyTorch installation rather than spending the deadline upgrading drivers. Tests must pass
before expensive runs. Both GPUs should use the same core/dependency versions where possible.

## Schedule and decision rules

Time estimates are planning ranges, NOT measurements from these new scripts. Your uploaded
old fold-0 log reported 55 epochs in about 46 minutes, but resolution, deterministic kernels,
GPU power and disk I/O differ. Measure the first TWO epochs and extrapolate. Reserve the
LAST THREE HOURS for full-data training, inference, validation and packaging.

| Time | University | Home |
|---|---|---|
| 0--1.5 h | Prepare plans; A pretrained control, first 8 epochs | Dataset transfer checks; oracle; smoke tests |
| 1.5--3 h | GeoCLIP 256-image preview; cache all only if useful | Continue transferred A, or evaluate A/retrieval when available |
| 3--6 h | B 8-epoch screen; C only if justified | Retrieval grid on A; optionally A continuation |
| 6--9 h | Continue winning B/C; D only if time/evidence | Untouched fold-1 confirmation of selected architecture/decoder |
| 9--12 h | FULL-data run of frozen winner, then predict | Validate/package; no new model searches |

P0 A is the FIRST training control. A/B/C/D all start from the same ImageNet weights and
seed, and reuse the SAME fold plan. Do not warm-start B/C/D from A: that confounds KD with
extra training. Eight epochs are a screen, not a final accuracy verdict. Resume survivors
to 24 total epochs using --resume; optimizer, scaler, scheduler and RNG are restored.

Decision gates (heuristics, not statistical claims):

- A: continue beyond 8 epochs if validation is improving. If still near chance, stop and
  inspect paths, normalization and learning curves, not another architecture.
- Teacher preview: compare on the SAME filenames as A's OOF CSV; do not compare a 256-row
  teacher score with the full-fold median. Cache all if teacher shows useful geographic
  signal or a material improvement; never use preview results as a leaderboard score.
- B/C: stop a variant if it is >10% worse than A after the SAME 8 epochs and its last few
  validation epochs are flat. Borderline variants may need a longer run; don't kill on one epoch.
- D: run only when B and C independently help. At most one modest weight setting initially.
- Selection: choose on fold 0 by median then mean. Freeze configuration before fold 1.
  Do not repeatedly tune on fold 1. A 0.5 km change is not evidence of improvement.
- The 30--40 km goal is an aspiration, not a promised result. Add <40 km to all reports.
  A teacher's success is not guaranteed to survive distillation.

Typical ranges: A/B/C 8 epochs 15--60 min; 24 epochs 45--150 min. GeoCLIP preview 5--20 min
plus downloads; all 11.7k teacher images 20--90 min. StreetCLIP cache 30--120 min. PCA/targets
1--10 min. Retrieval grid 5--30 min CPU depending BLAS. Full data 1--3 h; holdout 5--20 min.
Use batch 8 if student batch 16 OOMs; use teacher batch 1 if batch 4 OOMs. Larger GPU can
use larger inference batch, but keep student training batch identical for fair A/B/C runs.

## University commands in priority order

### P0: create portable fold plans, then run A

```powershell
python deadline_upgrade/student.py prepare --data "$Data" --folds leaderboard/folds.csv --fold 0 --out deadline/plans/fold0
python deadline_upgrade/student.py prepare --data "$Data" --folds leaderboard/folds.csv --fold 1 --out deadline/plans/fold1
python deadline_upgrade/student.py train --data "$Data" --plan deadline/plans/fold0 --variant A --epochs 8 --out deadline/runs/A0
```

This loads ImageNet weights, removes rotations/mirroring, uses full-image resizing with
light colour jitter and ImageNet normalization, trains existing heads, and asserts <=5M.
The controlled first stage uses 320px. Do not change resolution between A/B/C/D screens.
Plans refuse overwrites. Create once and COPY them to the other machine; do not refit there.

### P1: teacher environment and preview (no student training here)

Keep teacher dependencies isolated so they cannot break your student environment.

```powershell
py -3.11 -m venv .teacher_env
.\.teacher_env\Scripts\Activate.ps1
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r deadline_upgrade/teacher_requirements.txt
python -m pip freeze > teacher_environment.txt
python deadline_upgrade/teachers.py cache --teacher geoclip --plan deadline/plans/fold0 --data "$Data" --cache deadline/teachers/geoclip --limit 256
```

GeoCLIP uses pretrained image/location encoders and the plan's training-derived fine-cell
medoids. Its external 100k GPS gallery is NOT used. Metrics are in
`deadline/plans/fold0/geoclip_preview_metrics.json`, with the exact preview prediction rows.
Compare the preview against A on exactly the same filenames (CPU command):

```powershell
python deadline_upgrade/evaluate.py compare --plan deadline/plans/fold0 --first deadline/runs/A0/oof_predictions.csv --second deadline/plans/fold0/geoclip_preview_predictions.csv
```

If the preview is useful:

```powershell
python deadline_upgrade/teachers.py cache --teacher geoclip --plan deadline/plans/fold0 --data "$Data" --cache deadline/teachers/geoclip --all
python deadline_upgrade/teachers.py targets --plan deadline/plans/fold0 --geoclip deadline/teachers/geoclip --features deadline/teachers/geoclip --out deadline/targets/fold0_geoclip.npz
```

Re-running cache reuses every completed image; interrupted work resumes. Actual teacher
weights and image bytes are fingerprinted. Targets are keyed by filename AND exact fold
plan hash. Feature PCA fits ONLY fold-training rows, then pads to 256 dimensions if needed.
Teacher caches may include validation images for evaluation, but student training only
looks up targets for its TRAIN rows. No holdout student training is performed.

Optional StreetCLIP: only if GeoCLIP disappoints or feature KD needs a better teacher.
Its preview uses 256 validation queries and up to 1024 TRAIN reference images; this is a
screen, not a fair full-gallery comparison. --all expands the gallery to the full train fold.

```powershell
python deadline_upgrade/teachers.py cache --teacher streetclip --plan deadline/plans/fold0 --data "$Data" --cache deadline/teachers/streetclip --limit 256
# ONLY if promising:
python deadline_upgrade/teachers.py cache --teacher streetclip --plan deadline/plans/fold0 --data "$Data" --cache deadline/teachers/streetclip --all
python deadline_upgrade/teachers.py targets --plan deadline/plans/fold0 --geoclip deadline/teachers/geoclip --features deadline/teachers/streetclip --out deadline/targets/fold0_mixed.npz
```

Do not install PLONK in this deadline path. It is optional and deferred.

### P2/P3: return to STUDENT environment; controlled KD screens

```powershell
.\.venv\Scripts\Activate.ps1
# B: probability KD only
python deadline_upgrade/student.py train --data "$Data" --plan deadline/plans/fold0 --variant B --targets deadline/targets/fold0_geoclip.npz --epochs 8 --out deadline/runs/B0
# C: feature KD only; run after inspecting B, not blindly
python deadline_upgrade/student.py train --data "$Data" --plan deadline/plans/fold0 --variant C --targets deadline/targets/fold0_geoclip.npz --epochs 8 --out deadline/runs/C0
# D: only if both mechanisms helped and time remains
python deadline_upgrade/student.py train --data "$Data" --plan deadline/plans/fold0 --variant D --targets deadline/targets/fold0_geoclip.npz --epochs 8 --out deadline/runs/D0
```

Defaults: KD weight .3, feature weight .3, temperature 2. Probability KD is
T^2 KL(softmax(teacher_logits/T) || softmax(student_fine_logits/T)). Teacher logits use its
pretrained logit scale. Feature KD is cosine against train-fold PCA targets. Change with
--kd-weight, --feature-weight and --temperature, but keep equal compute for comparisons.

To continue A to 24 TOTAL epochs (not 24 additional):

```powershell
python deadline_upgrade/student.py train --data "$Data" --plan deadline/plans/fold0 --variant A --epochs 24 --out deadline/runs/A0 --resume
```

For B/C/D continuation, retain the same --variant, --targets and KD arguments and add
--resume --epochs 24. Do not run the same output folder on both PCs concurrently.

## Home commands

Your normal .venv is at the top-level GEOGUESSER in the screenshot; activate that one.

```powershell
..\..\.venv\Scripts\Activate.ps1
$Data = "C:\path\to\geo_dataset"
python -m pip install --no-deps -e .
python deadline_upgrade/evaluate.py oracle --data "$Data" --folds leaderboard/folds.csv --fold 0 --out deadline/oracle_fold0.json
```

The oracle is a diagnostic lower bound for selecting ONE training coordinate, not an
accuracy estimate. If its median exceeds 40 km, candidate-only retrieval cannot reach
40 km; interpolation or continuous refinement is needed. It does NOT bound all models.

Receive `deadline/plans/` and `deadline/runs/A0/` from university. You can continue A0
at home with its exact resume command while university runs teachers and B0.
Send the complete resumed A0 directory back only after stopping the home run.

For new-student retrieval (example A0; replace run name with actual winner):

```powershell
python deadline_upgrade/student.py export --checkpoint deadline/runs/A0/best_model.pt --rows deadline/runs/A0/train_rows.csv --data "$Data" --out deadline/runs/A0/train_bank.npz
python deadline_upgrade/evaluate.py retrieval --plan deadline/plans/fold0 --bank deadline/runs/A0/train_bank.npz --query deadline/runs/A0/validation.npz --out deadline/retrieval/A0
```

This tests k=1,3,5,10; similarity-weighted spherical means; predicted-country gating vs
unrestricted; and neural/retrieval blend 0,.25,.5,1. No validation image enters the bank.
The bank and validation cache MUST match the same checkpoint hash.

### Confirmation on fold 1: example if B wins

The frozen teacher can be reused: no need to infer the same image twice. On university's
teacher environment, create fold-1 candidate encodings and fold-specific targets:

```powershell
python deadline_upgrade/teachers.py cache --teacher geoclip --plan deadline/plans/fold1 --data "$Data" --cache deadline/teachers/geoclip --all
python deadline_upgrade/teachers.py targets --plan deadline/plans/fold1 --geoclip deadline/teachers/geoclip --features deadline/teachers/geoclip --out deadline/targets/fold1_geoclip.npz
```

Copy `plans/fold1` and `targets/fold1_geoclip.npz` home. Student environment:

```powershell
python deadline_upgrade/student.py train --data "$Data" --plan deadline/plans/fold1 --variant B --targets deadline/targets/fold1_geoclip.npz --epochs 24 --out deadline/runs/B1 --confirm
# For a paired KD comparison, also run A1 if time allows (otherwise don't claim confirmed KD gain).
python deadline_upgrade/student.py train --data "$Data" --plan deadline/plans/fold1 --variant A --epochs 24 --out deadline/runs/A1 --confirm
```

If A wins instead, omit targets/teacher steps and run A1. If C/D wins, use that variant.
Freeze epoch budget, weights and resolution from fold 0. Do not select a new architecture
on fold 1. --confirm retains the LAST fixed-budget epoch (the filename remains best_model.pt
for API compatibility), not a checkpoint selected by fold-1 labels. Use results only after
all planned epochs complete. Do not average the two medians or claim five-fold OOF.

If you tuned retrieval on B0, confirm that frozen setting on B1:

```powershell
python deadline_upgrade/student.py export --checkpoint deadline/runs/B1/best_model.pt --rows deadline/runs/B1/train_rows.csv --data "$Data" --out deadline/runs/B1/train_bank.npz
python deadline_upgrade/evaluate.py retrieval --plan deadline/plans/fold1 --bank deadline/runs/B1/train_bank.npz --query deadline/runs/B1/validation.npz --selected deadline/retrieval/B0/selected.json --out deadline/retrieval/B1_confirmation
```

Replace B0/B1 consistently with A0/A1 if A was selected. Don't reuse A's tuned settings to
claim B's confirmatory result without declaring the transfer. Retrieval submission remains
optional: the TA email does not explicitly settle the status of non-parametric banks.
The guaranteed simple final command below submits ONE neural model, no retrieval or teacher.

## Final full-data training and predictions (reserve 3 hours)

Freeze a winner no later than hour 9. A is the safe fallback. Replace 24 with the preselected
epoch budget if changed based on development data; never choose it using holdout labels.
Full data MUST start from pretrained initialization, not from a fold checkpoint with an
incompatible fine-head class ordering.

```powershell
python deadline_upgrade/student.py prepare --data "$Data" --folds leaderboard/folds.csv --fold full --out deadline/plans/full
# If A wins, this is the complete final training command:
python deadline_upgrade/student.py train --data "$Data" --plan deadline/plans/full --variant A --epochs 24 --out deadline/final/A
python deadline_upgrade/student.py predict --checkpoint deadline/final/A/final_model.pt --data "$Data" --out deadline/final/A/predictions.csv
python deadline_upgrade/evaluate.py validate --data "$Data" --csv deadline/final/A/predictions.csv
```

If B/C/D wins, regenerate FULL-data targets in the teacher environment first (the full
fine-cell ordering differs from every fold), then train the selected variant:

```powershell
python deadline_upgrade/teachers.py cache --teacher geoclip --plan deadline/plans/full --data "$Data" --cache deadline/teachers/geoclip --all
python deadline_upgrade/teachers.py targets --plan deadline/plans/full --geoclip deadline/teachers/geoclip --features deadline/teachers/geoclip --out deadline/targets/full_geoclip.npz
# Switch to student environment. Example B winner:
python deadline_upgrade/student.py train --data "$Data" --plan deadline/plans/full --variant B --targets deadline/targets/full_geoclip.npz --epochs 24 --out deadline/final/B
python deadline_upgrade/student.py predict --checkpoint deadline/final/B/final_model.pt --data "$Data" --out deadline/final/B/predictions.csv
python deadline_upgrade/evaluate.py validate --data "$Data" --csv deadline/final/B/predictions.csv
```

For StreetCLIP feature KD, replace --features with its cache path and preserve that choice
in confirmation/full targets. A single --features cache controls C/D's feature teacher.
No teacher model is loaded by student.py train or predict.

## What to transfer between machines

| Artifact | Transfer rule |
|---|---|
| `leaderboard/folds.csv` | HOME -> UNIVERSITY before anything; preserve bytes |
| Original `geo_dataset/` | Available independently on both; do not copy augmentations/external images |
| `deadline/plans/fold0/`, `fold1/`, `full/` | Create ONCE on university; copy whole folders; do not refit |
| `deadline/targets/*.npz` | Portable, filename-aligned KD targets; small compared with teacher weights |
| `deadline/runs/A0/`, `B0/`, etc. | Copy whole completed/stopped run directory, never individual best weights alone for resume |
| `deadline/teachers/` | Keep at university normally; optional archive for reproducibility, not needed for student training |
| `deadline/retrieval/` | Home -> university only if you need the frozen decoder records |
| `deadline/final/<winner>/` | University -> home: final weights, config and predictions |

Every run contains best_model.pt (validation) or final_model.pt (full data),
last_training.pt (resumable state), config.json, run.json, history.csv, training/validation
row CSVs, and (validation only) metrics.json, oof_predictions.csv, validation.npz.
Teacher/target files and training resumable state should NOT be required for inference.

For the official submission copy final_model.pt, predictions.csv, config.json, the code,
requirements and a short README/report. Do not bundle the huge teachers. Do not submit
the 256-row preview predictions or old collapsed my_predictions.csv.

Automatic neural-only final ZIP (example A winner):

```powershell
python deadline_upgrade/package_final.py --run deadline/final/A --plan deadline/plans/full --data "$Data" --out deadline/SUBMISSION_A.zip
```

The packager reloads the model on CPU, checks the cap and full-plan provenance, validates
all 2400 rows and includes code/config/environment. Add your required course report before
uploading. It deliberately excludes teacher weights, optimizer resume state and reference
banks. It is NOT the older package_submission.py with the obsolete pretrained prohibition.

## Verification limits and sources

Six dependency-light unit tests passed here. All new Python files compile. The remote
runtime's bundled torch crashes on import and full images/checkpoints are unavailable;
the CUDA training, teacher downloads and real-data scores have NOT been run here.
Run smoke_torch.py on either PC before training. It uses random tensors solely for a
non-scored unit test, never as challenge training images.

Teacher interfaces checked against the official GeoCLIP implementation:
https://github.com/VicenteVivan/geo-clip/blob/main/geoclip/model/image_encoder.py
https://github.com/VicenteVivan/geo-clip/blob/main/geoclip/model/GeoCLIP.py
StreetCLIP: https://huggingface.co/geolocal/StreetCLIP
Use StreetCLIP according to its CC-BY-NC-4.0 model license and cite teachers in the report.
