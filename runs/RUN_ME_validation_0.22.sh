#!/usr/bin/env bash
# jalebi 0.22.0 validation on your machine: the known set (config 3) and the blind set (config 2) with the 0.22 survey
# settings (runs/survey_0.22.yaml: emulator + shared tables, marginalised areas, 10000 steps, 3 DE starts, resume),
# scored strict / 0.20-rule / lenient with N.A scoring, then compared parameter by parameter with your 0.20 results.
#
#   bash runs/RUN_ME_validation_0.22.sh                      # ~1 day on 17 workers (10000 steps; 5 disks need it for 50 tau)
#   QUICK=1 bash runs/RUN_ME_validation_0.22.sh              # 500 steps, 60 DE generations: a first look, ~1-2 h
#   ONLY=DR_Tau,GW_Lup bash runs/RUN_ME_validation_0.22.sh
#
# Needs, next to this script in runs/ (copy them from the delivered runs_local/ folder; they are not in git):
#   jalebi_runs.py, run_validation_local.py, compare_validation_results.py, runner_core.py (yours),
#   validation_known.yaml (corrected values + scoring block), validation_blind_0.22.yaml, validation_targets.csv
# and the shared tables: jalebi emulator build --survey runs/survey_0.22.yaml -j 8  (once; ~4 core-hours)
#
# What to check (docs/ACCEPTANCE_0.22.md):
#   * check-sync prints "in sync: only `output` differs" (validation_blind_0.22.yaml == survey_0.22.yaml)
#   * every source finishes (no FAILED.txt); killed sources resume ("[stage] ... resumed")
#   * results/validation_report.md: "All scored quantities, known: ... 0.20 rule X/Y": X/Y must not be below 45 %
#     (the 0.20 result with the old manifest was 51/114 = 45 %; with the corrected manifest it was 84/125 = 67 %)
#   * no new double-pinned components: grep -c pinned results/validation_known_0.22/*/log.txt against the 0.20 run
#   * compare_validation_results.py: |z| < 2 for most parameters; list the ones that moved and why
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JALEBI_EMULATOR_READONLY=${JALEBI_EMULATOR_READONLY:-0}
WORKERS=${WORKERS:-17}
ARGS=(--blind validation_blind_0.22.yaml --survey survey_0.22.yaml --workers "$WORKERS")
[ "${QUICK:-0}" = "1" ] && ARGS+=(--quick)
[ -n "${ONLY:-}" ] && ARGS+=(--only "$ONLY")
python -c "import jalebi; assert jalebi.__version__ == '0.22.0'; print('jalebi', jalebi.__version__)"
for f in jalebi_runs.py run_validation_local.py compare_validation_results.py validation_known.yaml validation_blind_0.22.yaml validation_targets.csv; do
  [ -f "$f" ] || { echo "missing runs/$f (copy it from the delivered runs_local/ folder)"; exit 1; }
done
python jalebi_runs.py check-sync survey_0.22.yaml validation_blind_0.22.yaml
jalebi emulator list survey_0.22.yaml | tail -20
time python run_validation_local.py "${ARGS[@]}"
echo "== comparison with the 0.20 results (z = (new - old) / sqrt(sig_old^2 + sig_new^2))"
[ -d results/validation_known ] && python compare_validation_results.py results/validation_known results/validation_known_0.22 --out results/known_0.20_vs_0.22.csv --z 2 | tail -40 || true
[ -d results/validation_blind_v2 ] && python compare_validation_results.py results/validation_blind_v2 results/validation_blind_0.22 --out results/blind_0.20_vs_0.22.csv --z 2 | tail -40 || true
echo "== corner flags (0.22 diagnostic) per known source"
grep -l "warning: .*pinned" results/validation_known_0.22/*/log.txt 2>/dev/null | wc -l | xargs echo "sources with a pinned component:"
echo "report: results/validation_report.md"
