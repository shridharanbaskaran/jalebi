#!/usr/bin/env bash
# jalebi 0.24 on the server: H I accretion + continuum robustness on the finished survey_0.22 fits (nothing refitted).
#
#   cd ~/LTE_run/jalebi && conda activate jalebi
#   export PY=$(python -c "import sys; print(sys.executable)")
#   setsid nohup bash runs/RUN_ME_0.24.sh > results/run_0.24.log 2>&1 < /dev/null &
#
# Run it AFTER results/extend_and_report.sh has finished (that job imports jalebi in new worker processes).
# 1. tests;  2. jalebi accretion --survey (H I lines with a local water fit, ~10 s per disk);
# 3. jalebi robustness --survey (~20 continuum variants per disk, local refits with the emulator, minutes per disk);
# 4. merged tables + tarball.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PATH="$(dirname "$("$PY" -c 'import sys; print(sys.executable)')"):$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
export JALEBI_EMULATOR_READONLY=${JALEBI_EMULATOR_READONLY:-1}
ROOT=${ROOT:-results/survey_0.22}
WORKERS=${WORKERS:-30}

"$PY" -c "import jalebi, numpy; v=tuple(int(x) for x in jalebi.__version__.split('.')[:3]); assert v >= (0, 24, 0), jalebi.__version__; print('jalebi', jalebi.__version__, 'python', __import__('sys').executable)" || exit 1
"$PY" -m pytest -q -W ignore tests/test_0240_robustness_accretion.py -k "not app" || { echo "tests failed"; exit 1; }

echo "== accretion ($(date))"
"$PY" -m jalebi accretion "$ROOT" --survey --workers "$WORKERS"
echo "== continuum robustness ($(date))"
"$PY" -m jalebi robustness "$ROOT" --survey --workers "$WORKERS" --backend emulator
echo "== done ($(date))"

base=$(basename "$ROOT")
tar czf ~/"${base}_0.24.tar.gz" -C "$(dirname "$ROOT")" \
    $(cd "$(dirname "$ROOT")" && ls -d "$base"/accretion_survey.csv "$base"/continuum_robustness_survey.csv \
      "$base"/*/accretion.csv "$base"/*/accretion.json "$base"/*/accretion.png \
      "$base"/*/continuum_robustness.csv "$base"/*/continuum_robustness.png 2>/dev/null)
echo "copy back: scp new_server:~/${base}_0.24.tar.gz ~/Desktop/Work/LTE_fitting/ && tar xzf ${base}_0.24.tar.gz"
