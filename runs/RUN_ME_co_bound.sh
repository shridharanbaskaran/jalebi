#!/usr/bin/env bash
# jalebi 0.22.0: the CO temperature prior -- T <= 1500, 2000 and 3000 K on the known validation sources that have CO
# (FZ Tau, DR Tau, GW Lup, Sz 114, DF Tau, CX Tau ...), via continuum_study.py with runs/co_bound_settings.yaml.
#
#   bash runs/RUN_ME_co_bound.sh                          # 3 settings x the CO sources: ~6 h on 17 workers at 10000 steps
#   QUICK=1 bash runs/RUN_ME_co_bound.sh                  # 1500 steps: ~1 h
#
# What to check: results/co_bound/continuum_study_ranking.md and, per setting, the CO rows of
# results/co_bound/<setting>/validation_scores.csv:
#   * with T <= 3000 K (0.20): CO hot and thin (2394 K, log N 15 on FZ Tau) against the published 1400 K / 18.5
#   * log(N.A) of CO should agree between the three bounds (same lines, same flux); if it does, report.co: NA_only
#     is the honest summary for the survey and the bound can stay at 3000 K; if T and N pass only at 1500 K, lower
#     fit.bounds_by_molecule.CO.T in survey_0.22.yaml
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
WORKERS=${WORKERS:-17}
ONLY=${ONLY:-FZ_Tau_P24,DR_Tau,CX_Tau,GW_Lup,Sz_114,DF_Tau}
MAN=validation_known.yaml
if [ "${QUICK:-0}" = "1" ]; then
  python - <<'PY'
import yaml
m = yaml.safe_load(open("validation_known.yaml"))
d = m.setdefault("defaults", {}).setdefault("fit", {})
d.setdefault("mcmc", {})["nsteps"] = 1500; d.setdefault("optimise", {})["maxiter"] = 60
yaml.safe_dump(m, open("validation_known_quick.yaml", "w"), sort_keys=False, allow_unicode=True)
PY
  MAN=validation_known_quick.yaml
fi
time python continuum_study.py run "$MAN" --settings-file co_bound_settings.yaml --root results/co_bound --only "$ONLY" --workers "$WORKERS"
python continuum_study.py rank "$MAN" --settings-file co_bound_settings.yaml --root results/co_bound --only "$ONLY"
echo "== CO rows per bound"
for s in CO_T1500 CO_T2000 CO_T3000; do echo "-- $s"; grep -E "^[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,CO," results/co_bound/$s/validation_scores.csv | cut -d, -f1,6,7,8,9,10,11,12 || true; done
