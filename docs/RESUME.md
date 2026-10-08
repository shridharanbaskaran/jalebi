# Stage checkpoints and resume (0.22)

`fit.resume: auto` (the default) makes a per-disk run restartable: every stage writes its result into the
disk's output folder as soon as it is done, and a run that starts in a folder with valid checkpoints continues
from the last one instead of starting over.  A killed survey job (wall-clock limit, node failure, Ctrl-C)
costs only the stage it was in.

## What is written, and when

| file | written after | content |
| --- | --- | --- |
| `detection.json` | auto-detect (`fit.auto_detect`) | the candidate table, the components / windows / ordering it wrote |
| `grid.json` | the grid stage | the parameter vector after the sequential grid, the best cell per component |
| `de_pass1.json`, `de_pass2.json`, … | each optimiser pass | θ, χ², ln P, the resolved parameters (areas), runtime, every DE start (0.22 multi-start) |
| `continuum_refined.npz` (`continuum_refined2.npz`, …) | each model-aware continuum refinement pass | the refined continuum and mask, whether it was accepted (χ² improved) |
| `chain.h5` + `chain.key.json` | during the MCMC (emcee's HDF backend, `fit.mcmc.checkpoint: true`) | the chain so far; groups `mcmc` (joint) or `mcmc_block{i}` (`fit.mcmc.blocks: auto`); works for `linear: sample`, `profile` and `marginalise` |
| `dynesty.save` + `chain.key.json` | during nested sampling (`fit.dynesty.checkpoint: true`, every `checkpoint_every` s) | dynesty's own save file |

JSON stages are written to a temporary name and renamed into place (`os.replace`), so a crash during the write
never leaves a half-written file under the final name; a file that cannot be parsed, or whose `complete` flag is
missing, is ignored.

## The key

Every checkpoint carries a key: a SHA-256 of

* the data: file names, sizes and modification times of the target file or of every file in the target folder
  (not the bytes — cubes are large);
* the config with the run-only keys removed: `output`, `fit.resume`, `fit.mcmc.processes`, `fit.mcmc.checkpoint`,
  `fit.mcmc.nsteps`, `fit.optimise.workers`, `fit.grid.n_jobs`, `fit.dynesty.processes` / `.checkpoint`,
  `fit.emulator.cache_dir` / `.read_only` / `.rebuild` / `.spot_check`, `linedata.data_dir`, `report`;
* the model version: the shared-table model version and LSF version (`jalebi.emulator_shared`), the line
  truncation and a stage-format version (`jalebi.resume.STAGE_FORMAT_VERSION`).

(0.22 also seeds emcee's own random state from `fit.mcmc.seed`; before, it came from process entropy, so no
two runs gave the same chain and no continuation could be bit-identical.)  A checkpoint whose key differs from the current run's is ignored and the stage runs again; `chain.h5` /
`dynesty.save` with another key are deleted before the sampler starts.  Because `fit.mcmc.nsteps` is *not* in the
key, raising it continues an existing chain to the new length (emcee's random state is restored from the
backend, so the continued chain is identical to an uninterrupted run of the full length — the tests check
this bit for bit).

## Logging

Lines that start with `[stage]` say what happened, e.g.

```
[stage] de_pass1 resumed from checkpoint (chi2_red=1.066)
[stage] continuum refinement pass 1 resumed from checkpoint (accepted)
[stage] de_pass2 done, checkpoint written
[stage] mcmc resumed from checkpoint (4000 stored steps, target 10000)
[stage] chain.h5 belongs to another data/config/model version (or fit.resume off): starting the sampler over
```

`diagnostics.json` gets a `resume` block (`key`, `resumed`, `written`).  `jalebi.resume.status(folder, key)`
lists the stage files of a folder and whether they are valid for a key.

## Options

```yaml
fit:
  resume: auto        # auto (default) | off: ignore existing checkpoints (they are still written)
  mcmc: {checkpoint: true}
  dynesty: {checkpoint: true, checkpoint_every: 60.0}
```

`jalebi fit CONFIG --resume off` overrides the config.  The `jalebi batch --only-failed` switch still skips
folders that hold a `summary.csv`; with resume the natural way to finish a survey is simply to re-run the batch:
finished disks are read back from their chain without sampling, unfinished ones continue.

## Limits

* **A `kill -9` (SIGKILL) during the MCMC does not give a bit-identical continuation.**  emcee's HDF backend
  writes each step as coords → log-prob → blobs → random state → iteration counter in one open/close of the
  file; a SIGKILL inside that write leaves a snapshot whose random state can be one step ahead of (or behind)
  its coords.  The resumed chain is then a different, equally valid realisation of the same posterior (the
  smoke test checks the medians agree within 1 σ) and the stored steps are kept.  A continuation without a
  kill — a run that finished, then a larger `nsteps` — is bit-identical (tested).  The DE stages are
  bit-identical in every case: their files are written atomically.

* The grid maps (`grid_*.png` / `.npz`) are not re-drawn when the grid stage is resumed.
* dynesty's save file pickles the likelihood, i.e. the whole problem including attached emulator tables; it
  can be large.  Restoring a save file written by another jalebi version may fail; the run then starts over
  (logged).
* The data signature is by name / size / mtime: copying the data elsewhere (different mtimes) changes the key.
