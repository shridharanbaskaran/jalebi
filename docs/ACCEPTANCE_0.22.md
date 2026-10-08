# 0.22.0 acceptance checklist

What each RUN_ME must show for 0.22.0 to count as validated.  Everything below was *not* run in the sandbox
(2 cores; the heavy runs are yours); the sandbox evidence is the test suite and the synthetic cases quoted in
`claude_jalebi_STATUS.md`.

## RUN_ME_0.22_smoke.sh (~40 min)

- [ ] `pytest -q`: 0 failed.  Skips are fine (`JALEBI_EMULATOR_FULL` campaigns, optional packages).
- [ ] FZ Tau quick fit (emulator, shared tables, marginalise, 3 starts): the log shows `optimise: de, … 3 starts`,
      three `start k/3` blocks, `best start: …`, and `corner check: …` (or a `warning: … pinned` line).
- [ ] The killed run restarts with at least one `[stage] … resumed from checkpoint` line; if it was killed
      during the MCMC, `[stage] mcmc resumed from checkpoint (N stored steps, target 1500)`.
- [ ] `RESUME CHECK: passed`: the DE stage of the killed run equals run A bit for bit; the clean continuation
      (run C: 1000 steps, then continued to 1500 without a kill) equals run A bit for bit; the killed/resumed
      run's posterior medians agree with run A within 1 σ_post.  After a `kill -9` inside emcee's per-step HDF5
      write the stored random state can be one step ahead of the stored coords, so that continuation is a
      different, equally valid realisation and is *not* required to be bit-identical (docs/RESUME.md, Limits).

## RUN_ME_validation_0.22.sh (~1 day; QUICK=1 ~2 h)

- [ ] `check-sync`: `in sync: only output differs` for `validation_blind_0.22.yaml` vs `survey_0.22.yaml`.
- [ ] `jalebi emulator list survey_0.22.yaml`: every molecule of the survey candidates has a shared table
      (build them first: `jalebi emulator build --survey runs/survey_0.22.yaml -j 8`).
- [ ] All 19 known runs and 17 blind disks finish; no `FAILED.txt` except J160532 in the known set until a
      C6H6 list is imported (docs/LINELISTS.md).
- [ ] Known set, `validation_report.md` line `All scored quantities, known: strict a/n, 0.20 rule b/n, lenient
      c/n`: **b/n ≥ 45 %** (0.20's value with the old manifest; with the corrected manifest and the 0.20 results it
      is 84/125 = 67 %, so b/n should be near or above 67 %).  Record strict and lenient next to it.
- [ ] Tier-A sources in the known set (0.20 rule) ≥ 2 (FT Tau, DN Tau were tier A with the corrected manifest).
- [ ] **No new double-pinned components**: `grep -l "warning: .*pinned" results/validation_known_0.22/*/log.txt`
      lists at most the 8 known / 7 blind disks of 0.20 (`diagnostics.json["corner"]` has the details); fewer is
      the goal of the multi-start.
- [ ] `compare_validation_results.py` known 0.20 vs 0.22: most parameters within |z| < 2; every parameter that
      moved by more is explained by (a) a different optimum found by the extra starts (lower χ² in
      `diagnostics.json["optimise"]`), or (b) marginalised instead of profiled areas.  AS 209 and DR Tau, where
      the 0.20 DE missed the better minimum, should now reach χ²_red 1.91 and 3.02 or lower.
- [ ] Convergence: `steps/tau ≥ 50` (`diagnostics.json`) on the disks that needed 10000 steps in 0.20.
- [ ] Timing: wall clock per disk not more than ~3× the 0.20 run (three DE starts; the MCMC is the same cost per
      step).  Note the per-disk times in the status file.

## RUN_ME_continuum_study.sh (days; QUICK=1 on a subset ~1 h)

- [ ] `continuum_study_ranking.md` exists with one row per setting and `n` = the number of sources run.
- [ ] The survey continuum (`irsqr_q0.1_k25`) reproduces the validation numbers above (same pass counts).
- [ ] Decision: the setting with the highest 0.20-rule pass count whose corner-flag count is not above the
      survey continuum's.  If it is a `+offset` / `+spline` setting, `diagnostics.json["continuum_fit"]
      ["max_abs_correction_over_continuum"]` stays below ~0.1 on every source (a larger correction means the
      prior is too loose and line flux leaks into it) — then set `fit.continuum_fit` in `survey_0.22.yaml`
      and `validation_blind_0.22.yaml` and rerun the validation.

## RUN_ME_co_bound.sh (~6 h; QUICK=1 ~1 h)

- [ ] For the CO sources, log N·A of CO agrees between the 1500 / 2000 / 3000 K bounds to within 0.3 dex.
- [ ] If T and log N pass the 0.20 rule only with the 1500 K bound: lower `fit.bounds_by_molecule.CO.T` in the
      survey config; otherwise keep 3000 K and set `report.co: NA_only` for the survey summary.

## RUN_ME_detection_prob.sh (hours; with MCMC a day)

- [ ] `detection_probability_survey.csv` has one row per disk × unit; the class cross-table prints.
- [ ] Classes are sensible: the water components and CO2 of the bright disks (FZ Tau, DR Tau, Sz 114, GW Lup)
      are `robust`; isotopologues and the weak organics split between `continuum-dependent` and `not detected`;
      nothing that the papers detect at high S/N (CO2 in GW Lup, C2H2 in Sz 114, HCN in DF Tau) comes out
      `not detected`.
- [ ] The T / log N·A spreads across continua are of the order of the MCMC errors or larger for the weak
      species and much smaller for the strong ones.

## Sign-off

0.22.0 is validated when the smoke test and the validation run pass their boxes; the continuum study, CO bound
and detection probability are experiments whose outcome decides the 0.23 defaults, not acceptance criteria.
