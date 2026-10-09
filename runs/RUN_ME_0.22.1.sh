#!/usr/bin/env bash
# jalebi 0.22.1 on the server: check the two fixes, then start the corner study (controls + published disks).
#
#   bash runs/RUN_ME_0.22.1.sh                       # tests + the five 0.21 failures re-run + corner study (dry run)
#   LAUNCH=1 WORKERS=30 bash runs/RUN_ME_0.22.1.sh   # ... and launch the corner study (5 settings x ~80 targets)
#
# Needs run_survey_server.py + runner_core.py in runs/ (local-only runner scripts, not in git).
# Expected: tests/test_0221_fixes.py 17 passed (~10 s); the five targets that failed in 0.21 finish (a log line
# "MCMC start: N of M walkers had ln P = -inf" shows where the old code would have crashed); in the logs,
# "emulator <unit>: kept: ..." lines where 0.21 fell back to the exact model.
# Time: the five failed targets ~1-3 h on 5 workers; the corner study ~80 targets x 5 settings at ~10-20 min
# ~ 70-130 core-h, i.e. 3-5 h on 30 workers.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
export JALEBI_EMULATOR_READONLY=${JALEBI_EMULATOR_READONLY:-1}
CONFIG=${CONFIG:-runs/survey_0.22.yaml}
TARGETS=${TARGETS:-runs/survey_targets.csv}
WORKERS=${WORKERS:-30}

python -c "import jalebi; assert jalebi.__version__ == '0.22.1', jalebi.__version__; print('jalebi', jalebi.__version__)"
python -m pytest -q tests/test_0221_fixes.py

# 1. the five 0.21 failures (NaN in the DE snooker move), own output folder
python runs/rerun_targets.py --config "$CONFIG" --targets "$TARGETS" --name failed_0.21_rerun \
       808-50020 881-50220 HD-142666 HD-15407 IRAS-17178-2600
python runs/run_survey_server.py run --config runs/results/failed_0.21_rerun/config.yaml \
       --targets runs/results/failed_0.21_rerun/targets.csv --skip-preflight --workers 5 --timeout-h 12 || true
grep -h "MCMC start\|NaN" runs/results/failed_0.21_rerun/*/log.txt 2>/dev/null | head -20 || true

# 2. the corner study
python runs/corner_study.py prepare --base "$CONFIG" --targets "$TARGETS" --random 20
if [[ "${LAUNCH:-0}" == "1" ]]; then
  nohup python runs/corner_study.py run --workers "$WORKERS" > runs/results/corner_study/study.log 2>&1 &
  echo "corner study started (pid $!): tail -f runs/results/corner_study/study.log"
  echo "when finished: python runs/corner_study.py analyse --targets $TARGETS   -> runs/results/corner_study/RANKING.md"
else
  python runs/corner_study.py run --workers "$WORKERS" --dry-run
  echo "dry run only: LAUNCH=1 bash runs/RUN_ME_0.22.1.sh to start"
fi
