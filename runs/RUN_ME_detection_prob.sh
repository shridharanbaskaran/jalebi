#!/usr/bin/env bash
# jalebi 0.22.0 detection probability (ensemble mode) on the blind validation disks: 30 continua per disk, refits from
# the de_pass2 checkpoint, optional short MCMC; then the survey table.
#
#   bash runs/RUN_ME_detection_prob.sh                    # 17 disks x 30 variants, optimum only: ~2-4 h on 17 workers (emulator)
#   MCMC=300 bash runs/RUN_ME_detection_prob.sh           # + 300-step MCMC per variant for the parameter spread: ~1 day
#   N=50 ONLY=DR_Tau,Sz_114 bash runs/RUN_ME_detection_prob.sh
#
# Needs: results/validation_blind_0.22/<disk>/config.yaml and de_pass2.json from RUN_ME_validation_0.22.sh.
# What to check: results/validation_blind_0.22/detection_probability_survey.csv
#   * H2O components and CO2 should be "robust" in the bright disks; the classes of HCN / C2H2 / 13CO2 are the result:
#     count robust / continuum-dependent / not detected per molecule and compare with the papers' detections
#   * the T / logNA spreads are the continuum systematic error -- compare with the MCMC errors of summary.csv
#   * a molecule "detected" (dBIC_nominal > 10) but continuum-dependent is one to report with a caveat
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JALEBI_EMULATOR_READONLY=${JALEBI_EMULATOR_READONLY:-1}
ROOT=${ROOT:-results/validation_blind_0.22}
N=${N:-30}; MCMC=${MCMC:-0}; WORKERS=${WORKERS:-17}
ls -d "$ROOT"/*/ >/dev/null 2>&1 || { echo "no disk folders under $ROOT: run RUN_ME_validation_0.22.sh first"; exit 1; }
DISKS=$(ls -d "$ROOT"/*/ | xargs -n1 basename | grep -v "^_")
[ -n "${ONLY:-}" ] && DISKS=$(echo "$ONLY" | tr ',' '\n')
echo "$DISKS" | xargs -P "$WORKERS" -I{} sh -c 'jalebi detect-prob "'"$ROOT"'/{}/config.yaml" --n '"$N"' --mcmc-steps '"$MCMC"' > "'"$ROOT"'/{}/detect_prob.log" 2>&1 && echo "done {}" || echo "FAILED {} (see '"$ROOT"'/{}/detect_prob.log)"'
jalebi detect-prob --survey "$ROOT"
python - "$ROOT" <<'PY'
import sys, pandas as pd
t = pd.read_csv(f"{sys.argv[1]}/detection_probability_survey.csv")
t["molecule"] = t.unit.str.split("_").str[0]
print(pd.crosstab(t.molecule, t["class"]))
PY
