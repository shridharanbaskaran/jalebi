#!/usr/bin/env bash
# Example 6 — many disks with one config: every row of targets.csv (name,path,distance_pc,rv_kms) is
# fitted in parallel; one population table collects the results.
#
#   bash 06_batch.sh                       # quick: grid + optimiser, 2 targets
#
# For a real sample: put your targets in a CSV (path = folder of Level3_ch*_x1d.fits files), then
#   jalebi batch configs/FZ_Tau_quick.yaml my_targets.csv --workers 16
#   jalebi batch configs/FZ_Tau_quick.yaml my_targets.csv --auto-detect      # molecules chosen per target
#   jalebi batch configs/FZ_Tau_quick.yaml my_targets.csv --only-failed      # re-run what failed
set -euo pipefail
cd "$(dirname "$0")"
jalebi batch configs/FZ_Tau_quick.yaml configs/targets.csv --workers 2 --stages grid,optimise
echo "population table: results/FZ_Tau/population.csv (one folder per target next to it)"
