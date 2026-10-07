#!/usr/bin/env bash
# SLURM array version of RUN_ME_sampler_benchmark.sh: one cell per task (16 cells; dynesty cells run their 3 seeds).
#   sbatch runs/slurm_sampler_benchmark.sh            then, when all tasks are done:
#   python runs/sampler_benchmark.py report --out runs/results/sampler_benchmark
#SBATCH --job-name=jalebi-samplers
#SBATCH --array=0-15
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=runs/results/sampler_benchmark/slurm_%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mapfile -t CELLS < <(python runs/sampler_benchmark.py list)
CELL=${CELLS[$SLURM_ARRAY_TASK_ID]}
echo "cell $CELL on $(hostname), $SLURM_CPUS_PER_TASK cores"
python runs/sampler_benchmark.py run --cells "$CELL" --out runs/results/sampler_benchmark \
       --emulator-dir "${JALEBI_EMULATOR_DIR:-$HOME/.jalebi/emulator}" --processes "$SLURM_CPUS_PER_TASK"
