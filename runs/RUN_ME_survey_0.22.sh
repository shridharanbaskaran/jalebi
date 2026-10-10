#!/usr/bin/env bash
# survey_0.22 production run (jalebi >= 0.23.1): every spectrum in runs/survey_targets.csv with runs/survey_0.22.yaml.
#
#   setsid nohup bash runs/RUN_ME_survey_0.22.sh > results/survey_0.22.log 2>&1 < /dev/null &
#   WORKERS=30 NO_SEED=1 bash runs/RUN_ME_survey_0.22.sh        # without reusing the corner-study fits
#
# Setting: survey_0.22.yaml as it is (= corner-study "base": continuum_fit none, H2O log N up to 21). The 0.23
# corner study ranked it first: 1 control with shift-null detections (HD 23514, aperture check pending) against
# 2-6 for the others, the narrowest pooled null (FAP 1.8e-3 at S > 5), the best water T agreement with published
# fits (median |dT| 61 K) and 66 % converged chains (spline: 31 %).  Hot-corner water (T ~1490 K, log N ~21) still
# appears in most fits, but the shift-null test rejects 101 of 109 such units.
#
# Steps: tests -> preflight -> seed the 74 corner-study "base" fits (identical config, so they resume from their
# checkpoints and only re-save with the 0.23 code, incl. shift_null.csv) -> run every disk (finished disks are
# skipped on a restart) -> report -> collect -> pooled shift-null calibration per class -> tarball.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
export JALEBI_EMULATOR_READONLY=${JALEBI_EMULATOR_READONLY:-1}
CONFIG=${CONFIG:-runs/survey_0.22.yaml}
TARGETS=${TARGETS:-runs/survey_targets.csv}
WORKERS=${WORKERS:-30}
TIMEOUT_H=${TIMEOUT_H:-48}
CLASSES=${CLASSES:-runs/census_classes_0.21.csv}
SEED=${SEED:-runs/results/corner_study/base}
ROOT=$(python -c "from jalebi.config import ProjectConfig; print(ProjectConfig.load('$CONFIG').output_root())")

python -c "import jalebi; v=tuple(int(x) for x in jalebi.__version__.split('.')[:3]); assert v >= (0, 23, 1), jalebi.__version__; print('jalebi', jalebi.__version__)"
python -m pytest -q tests/test_0230_shift_null.py tests/test_0222_fixes.py
python runs/run_survey_server.py preflight --config "$CONFIG" --targets "$TARGETS"

mkdir -p "$ROOT"
n=0
if [[ "${NO_SEED:-0}" != 1 && -d "$SEED" ]]; then
  for d in "$SEED"/*/; do
    t=$(basename "$d")
    [[ $t == _* || ! -f "$d/chain.h5" || -e "$ROOT/$t" ]] && continue
    cp -a "$d" "$ROOT/$t"
    rm -f "$ROOT/$t/DONE" "$ROOT/$t/FAILED.txt" "$ROOT/$t/NO_DETECTION"
    n=$((n + 1))
  done
fi
echo "seeded $n disks from $SEED (they should log 'mcmc resumed from checkpoint')"

echo "== run ($WORKERS workers, ${TIMEOUT_H} h per disk; started $(date))"
python runs/run_survey_server.py run --config "$CONFIG" --targets "$TARGETS" --workers "$WORKERS" \
       --timeout-h "$TIMEOUT_H" --skip-preflight || echo "runner exited with $? -- continuing to the report"
echo "== finished $(date)"
if [[ $n -gt 0 ]]; then
  n_res=$(grep -l "mcmc resumed from checkpoint" "$ROOT"/*/log.txt 2>/dev/null | wc -l)
  echo "logs with a resumed chain: $n_res (seeded: $n)"
fi

python runs/run_survey_server.py report --config "$CONFIG" --targets "$TARGETS" || true
python runs/collect_results.py --run-dir "$ROOT" --workers 16 || true
if [[ -f "$CLASSES" ]]; then
  python runs/shift_null_calibrate.py "$ROOT" --classes "$CLASSES"
else
  python runs/shift_null_calibrate.py "$ROOT"
fi

base=$(basename "$ROOT")
tar czf ~/"${base}_results.tar.gz" -C "$(dirname "$ROOT")" \
    $(cd "$(dirname "$ROOT")" && ls -d "$base"/population.csv "$base"/survey_report.md "$base"/*.png \
      "$base"/_shift_null_calibration "$base"/runner_status.csv "compiled_$base" 2>/dev/null)
echo "copy back: scp SERVER:~/${base}_results.tar.gz ~/Desktop/Work/LTE_fitting/ && tar xzf ${base}_results.tar.gz"
