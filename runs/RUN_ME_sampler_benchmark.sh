#!/usr/bin/env bash
# Sampler benchmark (jalebi 0.20): emcee vs dynesty on AS 209 and FZ Tau, exact model and emulator, linear sample/profile.
# The emulator cells were run in the sandbox (see runs/results/sampler_benchmark/REPORT.md); the exact-model cells take
# hours, so they run here.  Every finished cell is skipped on a rerun (delete its folder or pass --force to redo it).
#
#   bash runs/RUN_ME_sampler_benchmark.sh            # everything still missing, P processes per run
#   P=16 bash runs/RUN_ME_sampler_benchmark.sh
#   CELLS='FZ_Tau/dynesty/exact/*' bash runs/RUN_ME_sampler_benchmark.sh
#
# Needs: pip install -e ".[app,dev]" dynesty   (and runs/results/validation_blind/AS_209 from the blind validation)
set -euo pipefail
cd "$(dirname "$0")/.."
P=${P:-8}
OUT=${OUT:-runs/results/sampler_benchmark}
EMU=${EMU:-$HOME/.jalebi/emulator}
CELLS=${CELLS:-'*/*/*/*'}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c "import dynesty, jalebi; print('jalebi', jalebi.__version__, 'dynesty', dynesty.__version__)"
# cheap cells first (emulator), then the exact model; dynesty runs 3 seeds per cell
python runs/sampler_benchmark.py run --cells '*/*/emulator/*' --out "$OUT" --emulator-dir "$EMU" --processes "$P"
python runs/sampler_benchmark.py run --cells "$CELLS" --out "$OUT" --emulator-dir "$EMU" --processes "$P"
python runs/sampler_benchmark.py report --out "$OUT"
echo "report: $OUT/REPORT.md"
