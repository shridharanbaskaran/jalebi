#!/usr/bin/env bash
# One-time pre-build of the shared emulator tables (jalebi 0.21, fit.emulator.cache: shared) for the survey and
# the validation configs: every molecule / line list / (T, log N) box the configs could need, including all
# auto-detect candidates.  Tables land in $JALEBI_EMULATOR_DIR/shared (default ~/.jalebi/emulator/shared) and are
# resampled onto each disk's pixels when a fit loads them.  Re-running only verifies the cache (nothing is rebuilt
# unless the key changed); one table is built by exactly one process (lock file), so this can run while fits are
# already queued -- those wait for the table or, in read-only mode, use the exact model.
#
#   bash runs/RUN_ME_build_emulator.sh                 # survey + validation configs, J tables in parallel
#   J=16 bash runs/RUN_ME_build_emulator.sh
#   CONFIGS="runs/survey_300_autodetect.yaml" bash runs/RUN_ME_build_emulator.sh
#
# Memory: ~1.5-2.5 GB per worker for the H2O HITEMP table (the fine grid of the whole MRS range at oversample 6,
# 95 000 lines); the others need < 1 GB.  Pick J accordingly.
# Needs: pip install -e ".[app,dev]"
set -euo pipefail
cd "$(dirname "$0")/.."
J=${J:-8}
CONFIGS=${CONFIGS:-"runs/survey_300_autodetect.yaml runs/validation_blind.yaml examples/configs/FZ_Tau_quick.yaml"}
export JALEBI_EMULATOR_DIR=${JALEBI_EMULATOR_DIR:-$HOME/.jalebi/emulator}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c "import jalebi; print('jalebi', jalebi.__version__)"
echo "tables -> $JALEBI_EMULATOR_DIR/shared  (J=$J)"
time jalebi emulator build --survey $CONFIGS -j "$J"
jalebi emulator list runs/survey_300_autodetect.yaml
