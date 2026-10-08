#!/usr/bin/env bash
# jalebi 0.22.0 continuum study: the known validation set under every built-in continuum setting (IRSQR variants,
# median_sg, aspls, asls, and the joint continuum correction offset / spline), scored and ranked.
#
#   bash runs/RUN_ME_continuum_study.sh                   # 15 settings x 19 sources: ~2-3 days on 17 workers at 10000 steps
#   QUICK=1 bash runs/RUN_ME_continuum_study.sh           # 1500 steps, 60 DE generations (ranking by the optimum): ~8 h
#   SETTINGS=irsqr_q0.1_k25,irsqr_q0.1_k25+spline1um ONLY=DR_Tau,GW_Lup,Sz_114,DF_Tau bash runs/RUN_ME_continuum_study.sh   # ~1 h quick
#
# Needs the same runs/ files as RUN_ME_validation_0.22.sh.  Every setting writes to results/continuum_study/<setting>/;
# runs resume, so the script can be restarted.
# What to check: results/continuum_study/continuum_study_ranking.md -- the setting with the highest 0.20-rule pass
# count whose corner-flag count is not above the survey continuum's (irsqr_q0.1_k25).  If a +offset / +spline
# setting wins, set fit.continuum_fit in survey_0.22.yaml and validation_blind_0.22.yaml (keep them in sync) and
# rerun RUN_ME_validation_0.22.sh.
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
WORKERS=${WORKERS:-17}
ARGS=(--workers "$WORKERS" --root results/continuum_study)
[ -n "${SETTINGS:-}" ] && ARGS+=(--settings "$SETTINGS")
[ -n "${ONLY:-}" ] && ARGS+=(--only "$ONLY")
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
python continuum_study.py list "${ARGS[@]:2}" | grep -E "^[A-Za-z]" | tr -d ':' | xargs echo "settings:"
time python continuum_study.py run "$MAN" "${ARGS[@]}"
python continuum_study.py rank "$MAN" "${ARGS[@]}"
echo "ranking: results/continuum_study/continuum_study_ranking.md"
