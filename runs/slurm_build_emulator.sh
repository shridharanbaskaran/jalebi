#!/usr/bin/env bash
# One-time pre-build of the shared emulator tables (jalebi 0.21) on the cluster, before the survey array.
# One task; the tables are built in parallel over molecules (joblib, cpus-per-task workers) into the shared cache
# every compute node reads ($JALEBI_EMULATOR_DIR/shared, default ~/.jalebi/emulator/shared).
#
#   sbatch runs/slurm_build_emulator.sh                    # then: sbatch --array=0-29 runs/slurm_survey.sh
#   JALEBI_EMULATOR_DIR=/scratch/$USER/jalebi_emulator sbatch runs/slurm_build_emulator.sh
#
# The survey tasks should run read-only (export JALEBI_EMULATOR_READONLY=1 in slurm_survey.sh, or
# fit.emulator.read_only: true): a disk whose table is missing then uses the exact model for that unit, with a
# warning, instead of 300 tasks trying to build it.  A table that is being built is locked; a fit that needs it
# while the build runs waits for the lock (unless read-only).
#SBATCH --job-name=jalebi-emulator
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3G
#SBATCH --time=06:00:00
#SBATCH --output=logs/jalebi_emulator_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
export JALEBI_DATA=${JALEBI_DATA:-$HOME/.jalebi/linedata}
export JALEBI_EMULATOR_DIR=${JALEBI_EMULATOR_DIR:-$HOME/.jalebi/emulator}
CONFIGS=${CONFIGS:-"runs/survey_300_autodetect.yaml runs/validation_blind.yaml"}
echo "jalebi $(python -c 'import jalebi; print(jalebi.__version__)') on $(hostname): $SLURM_CPUS_PER_TASK workers -> $JALEBI_EMULATOR_DIR/shared"
time jalebi emulator build --survey $CONFIGS -j "$SLURM_CPUS_PER_TASK"
jalebi emulator list runs/survey_300_autodetect.yaml
