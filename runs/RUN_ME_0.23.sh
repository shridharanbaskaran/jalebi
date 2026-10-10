#!/usr/bin/env bash
# jalebi 0.23 on the server: score the corner study with the shifted-template null test (docs/SHIFT_NULL.md).
#
#   bash runs/RUN_ME_0.23.sh                     # after RUN_ME_0.22.2.sh has finished
#   WORKERS=30 bash runs/RUN_ME_0.23.sh
#
# 1. tests;  2. jalebi shift-null on every corner-study setting (nothing refitted: each fit is rebuilt from its
# config.yaml + prep.csv + best_fit.json; ~10-60 s per target with the emulator);  3. corner_study.py analyse
# (now ranked by shift-null detections in the controls, then pinned water, then water |dT|);  4. pooled-null
# calibration per setting;  5. tarball.  The survey_0.22 run itself gets the test automatically (fit.shift_null
# is on by default in 0.23 and is not part of the resume key).
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
export JALEBI_EMULATOR_READONLY=${JALEBI_EMULATOR_READONLY:-1}
TARGETS=${TARGETS:-runs/survey_targets.csv}
WORKERS=${WORKERS:-30}
SETTINGS=${SETTINGS:-"base h2o_logN20 h2o_cap spline spline_h2o20"}
STUDY=runs/results/corner_study

python -c "import jalebi; assert jalebi.__version__.startswith('0.23.'), jalebi.__version__; print('jalebi', jalebi.__version__)"
if [[ -f "$STUDY/rescore_0.22.2.log" ]] && ! grep -q "copy back:" "$STUDY/rescore_0.22.2.log"; then
  echo "RUN_ME_0.22.2.sh has not finished yet (no 'copy back:' line in $STUDY/rescore_0.22.2.log) -- wait for it"; exit 1
fi
python -m pytest -q tests/test_0230_shift_null.py tests/test_0222_fixes.py

for s in $SETTINGS; do
  [[ -d "$STUDY/$s" ]] || { echo "$s: no folder, skipped"; continue; }
  echo "== shift-null: $s"
  jalebi shift-null "$STUDY/$s" --survey --workers "$WORKERS" --backend emulator --force 2>&1 | tail -n 5
  python runs/shift_null_calibrate.py "$STUDY/$s" 2>&1 | sed -n '1,16p'
done

python runs/corner_study.py analyse --targets "$TARGETS"
cat "$STUDY/RANKING.md"

tar czf ~/corner_study_0.23.tar.gz -C runs/results corner_study/RANKING.md corner_study/ranking.csv \
    $(cd runs/results && for s in $SETTINGS; do ls -d corner_study/$s/shift_null_survey.csv corner_study/$s/_shift_null_calibration corner_study/$s/_compiled 2>/dev/null; done)
echo "copy back: scp SERVER:~/corner_study_0.23.tar.gz ~/Desktop/Work/LTE_fitting/ && tar xzf corner_study_0.23.tar.gz"
