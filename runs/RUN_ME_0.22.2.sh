#!/usr/bin/env bash
# jalebi 0.22.2 on the server: re-score the corner study's spline settings with the fixed significance test.
#
#   bash runs/RUN_ME_0.22.2.sh                 # tests + re-save spline / spline_h2o20 + collect + analyse + tarball
#   WORKERS=30 SETTINGS="spline spline_h2o20" bash runs/RUN_ME_0.22.2.sh
#
# Why: up to 0.22.1 component_significance compared the model without a component against the raw data while the
# full model saw the continuum-corrected data, so with fit.continuum_fit: spline every component got (nearly) the
# same, large delta chi^2 (24 of 74 corner-study targets) and 28 of 36 no-disk controls looked like detections.
# Nothing is refitted: --force with fit.resume: auto reuses the grid / optimiser / continuum stages and the
# finished chain.h5 (0 new MCMC steps); the pipeline only re-saves best_fit / detections / summaries / plots.
# Expected: "[stage] mcmc resumed from checkpoint (... stored steps, target N)" in every log, a few minutes per
# target; the base / h2o settings (continuum_fit: none) are unaffected and are only re-analysed.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
export JALEBI_EMULATOR_READONLY=${JALEBI_EMULATOR_READONLY:-1}
TARGETS=${TARGETS:-runs/survey_targets.csv}
WORKERS=${WORKERS:-30}
SETTINGS=${SETTINGS:-"spline spline_h2o20"}
STUDY=runs/results/corner_study

python -c "import jalebi; assert jalebi.__version__ == '0.22.2', jalebi.__version__; print('jalebi', jalebi.__version__)"
python -m pytest -q tests/test_0222_fixes.py tests/test_0221_fixes.py

for s in $SETTINGS; do
  grep -q "continuum_fit: spline\|continuum_fit: offset" "$STUDY/$s/config.yaml" || { echo "$s: no continuum correction, skipped"; continue; }
  grep -q "resume: off" "$STUDY/$s/config.yaml" && { echo "$s: fit.resume is off -- this would refit everything; stop"; exit 1; }
  echo "== re-saving $s (resume, no refit)"
  python runs/run_survey_server.py run --config "$STUDY/$s/config.yaml" --targets "$STUDY/targets_subset.csv" \
         --force --skip-preflight --workers "$WORKERS" --timeout-h 6 || true
  n_res=$(grep -l "mcmc resumed from checkpoint" "$STUDY/$s"/*/log.txt 2>/dev/null | wc -l)
  n_all=$(ls -d "$STUDY/$s"/*/ 2>/dev/null | wc -l)
  echo "$s: $n_res of $n_all logs show a resumed chain (should be all)"
done

python runs/corner_study.py analyse --targets "$TARGETS"
cat "$STUDY/RANKING.md"
for s in $SETTINGS; do
  python runs/collect_results.py --run-dir "$STUDY/$s" --with-files --workers 16
done
tar czf ~/corner_study_0.22.2.tar.gz -C runs/results corner_study/RANKING.md corner_study/ranking.csv \
    $(cd runs/results && for s in $SETTINGS; do ls -d corner_study/compiled_$s 2>/dev/null; done)
echo "copy back: scp SERVER:~/corner_study_0.22.2.tar.gz ~/Desktop/Work/LTE_fitting/ && tar xzf corner_study_0.22.2.tar.gz"
