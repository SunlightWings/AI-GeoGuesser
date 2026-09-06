# Verification performed in this session

- All new Python sources: compileall PASS.
- Six dependency-light unit tests: PASS (metrics, canonical fold alignment/rejection,
  filename cache round-trip, strict submission rejection cases, retrieval gate and blend,
  k greater than the number of same-country candidates).
- End-to-end synthetic cached retrieval CLI: PASS, 64 development configurations and
  exactly one frozen confirmation configuration; no images trained or generated.
- Teacher model APIs inspected from official source, but weights NOT downloaded/run here.
- PyTorch smoke NOT RUN successfully here: bundled torch exits with a bus error. This is
  a limitation of this workspace, not proof the package works on CUDA. The provided
  smoke_torch.py checks supervised + both KD losses, backward, model count and checkpoint
  reload on your PC without external downloads.
- No full-data images, latest checkpoints, or canonical folds are bundled. The earlier
  core is included as a starter only. The additive entry points need the user's canonical
  manifest and supplied dataset. No numerical performance claims for the new model.

Keep the working old project/checkpoints untouched until the new control is validated.
The training CLI saves last_training.pt every completed epoch. Resume replays from the last
completed epoch; it is NOT a mid-minibatch checkpoint. Teacher caching resumes image-wise.
Bitwise reproducibility across different GPUs/CUDA versions is not promised.
