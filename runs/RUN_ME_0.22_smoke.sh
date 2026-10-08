#!/usr/bin/env bash
# jalebi 0.22.0 smoke test on your machine: (1) the test suite, (2) one quick disk end to end with the shared tables,
# (3) a resume test that kills a run halfway and restarts it, (4) a clean continuation; both compared with the uninterrupted run.
#
#   bash runs/RUN_ME_0.22_smoke.sh                 # ~30-40 min: pytest ~25 min (2 cores) + 2 x FZ Tau quick + the resumed run
#   FAST=1 bash runs/RUN_ME_0.22_smoke.sh          # only the 0.22 tests (~5 min) + the fits
#
# Expected / what to check (acceptance checklist, docs/ACCEPTANCE_0.22.md):
#   * pytest: 0 failed (skips are fine); the 0.22 suites test_resume / test_multistart / test_continuum_fit /
#     test_corner / test_detection_prob / test_linelist_import / test_components_output all pass
#   * the quick fit prints "optimise: de, ... 3 starts", "best start: ...", "corner check: ..." and ends with summary.csv
#   * the killed run restarts with "[stage] de_pass1 resumed from checkpoint" (or later stages) and
#     "[stage] mcmc resumed from checkpoint (N stored steps, target 1500)" when it was killed during the MCMC
#   * RESUME CHECK: passed -- printed at the end: the DE stage of the killed run equals run A bit for bit, a clean
#     continuation (1000 -> 1500 steps, no kill) equals run A bit for bit, and the killed/resumed run's posterior
#     medians agree with A within 1 sigma_post.  (After a kill -9 inside emcee's per-step HDF5 write the stored
#     random state can be one step ahead of the stored coords, so that continuation is an equally valid but
#     different realisation -- it is not required to be bit-identical.)
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export JALEBI_EMULATOR_READONLY=${JALEBI_EMULATOR_READONLY:-0}
OUT=${OUT:-runs/results/smoke_0.22}
python -c "import jalebi; assert jalebi.__version__ == '0.22.0', jalebi.__version__; print('jalebi', jalebi.__version__)" || exit 1

echo "== 1. tests"
if [ "${FAST:-0}" = "1" ]; then
  python -m pytest -q tests/test_resume.py tests/test_multistart.py tests/test_components_output.py tests/test_continuum_fit.py \
      tests/test_corner.py tests/test_detection_prob.py tests/test_linelist_import.py || { echo "TESTS FAILED"; exit 1; }
else
  python -m pytest -q || { echo "TESTS FAILED"; exit 1; }
fi

echo "== 2. one quick disk end to end (FZ Tau, bundled data, shared tables if built, 1500 steps)"
rm -rf "$OUT"; mkdir -p "$OUT"
CFG=$OUT/fz_quick.yaml
python - "$CFG" <<'EOF'
import sys
from jalebi.config import ProjectConfig
cfg = ProjectConfig.load("examples/configs/FZ_Tau_quick.yaml")
cfg.fit.model_backend = "emulator"; cfg.fit.emulator.cache = "shared"
cfg.fit.mcmc.nsteps = 1500; cfg.fit.mcmc.processes = 1; cfg.fit.mcmc.linear = "marginalise"; cfg.fit.mcmc.moves = "de"
cfg.fit.mcmc.init = "scaled"; cfg.fit.mcmc.vectorize = True
cfg.fit.optimise.workers = 1; cfg.fit.optimise.n_starts = 3
cfg.fit.optimise.seed_points = {"H2O_hot": {"T": 900.0, "logN": 18.5}, "H2O_warm": {"T": 500.0, "logN": 18.0}}
cfg.fit.stages = ["optimise", "mcmc"]
cfg.output = sys.argv[1].rsplit("/", 1)[0] + "/A/{target}"
cfg.save(sys.argv[1])
EOF
T0=$(date +%s)
jalebi fit "$CFG" 2>&1 | tee "$OUT/fit_A.log" | grep -E "starts|best start|corner|\[stage\]|done:|multimodal|warning" || true
T1=$(date +%s); DT=$((T1 - T0)); echo "uninterrupted run: ${DT}s"
test -f "$OUT/A/FZ_Tau/summary.csv" || { echo "no summary.csv from the uninterrupted run"; exit 1; }

echo "== 3. resume test: kill a second run at ~60 % of that time, then restart it"
sed "s#/A/{target}#/B/{target}#" "$CFG" > "$OUT/fz_quick_B.yaml"
KILL=$(( DT * 6 / 10 )); [ "$KILL" -lt 20 ] && KILL=20
( jalebi fit "$OUT/fz_quick_B.yaml" > "$OUT/fit_B_killed.log" 2>&1 ) & PID=$!
sleep "$KILL"; kill -9 "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
echo "killed after ${KILL}s; stage files present:"; ls "$OUT/B/FZ_Tau" 2>/dev/null | grep -E "json|npz|h5" || true
jalebi fit "$OUT/fz_quick_B.yaml" 2>&1 | tee "$OUT/fit_B_resumed.log" | grep -E "\[stage\]|done:" || true

echo "== 4. clean continuation: a run stopped at 1000 steps and continued to 1500 (no kill) must equal run A bit for bit"
sed "s#/A/{target}#/C/{target}#" "$CFG" > "$OUT/fz_quick_C.yaml"
python - "$OUT/fz_quick_C.yaml" 1000 <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])); d["fit"]["mcmc"]["nsteps"] = int(sys.argv[2]); yaml.safe_dump(d, open(sys.argv[1], "w"), sort_keys=False)
PY
jalebi fit "$OUT/fz_quick_C.yaml" > "$OUT/fit_C_1000.log" 2>&1
python - "$OUT/fz_quick_C.yaml" 1500 <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])); d["fit"]["mcmc"]["nsteps"] = int(sys.argv[2]); yaml.safe_dump(d, open(sys.argv[1], "w"), sort_keys=False)
PY
jalebi fit "$OUT/fz_quick_C.yaml" 2>&1 | tee "$OUT/fit_C_1500.log" | grep -E "\[stage\]" || true

echo "== 5. compare"
python - "$OUT" <<'PY'
import json, re, sys
import numpy as np, pandas as pd
out = sys.argv[1]
ok = True
def theta(run, name):
    return np.asarray(json.load(open(f"{out}/{run}/FZ_Tau/{name}"))["payload"]["theta"], float)
same_de = np.array_equal(theta("A", "de_pass1.json"), theta("B", "de_pass1.json"))
print("de_pass1 identical (A vs killed/resumed B):", same_de); ok &= same_de
ca = np.load(f"{out}/A/FZ_Tau/chain.npz")["chain"]; cb = np.load(f"{out}/B/FZ_Tau/chain.npz")["chain"]; cc = np.load(f"{out}/C/FZ_Tau/chain.npz")["chain"]
print("chain lengths A/B/C:", ca.shape[0], cb.shape[0], cc.shape[0]); ok &= (ca.shape == cb.shape == cc.shape)
same_c = np.array_equal(ca, cc)
print("clean continuation 1000 -> 1500 identical to A:", same_c); ok &= same_c
log = open(f"{out}/fit_B_resumed.log").read()
m = re.search(r"mcmc resumed from checkpoint \((\d+) stored steps", log)
n_res = int(m.group(1)) if m else 0
if n_res:
    print(f"killed run: first {n_res} steps identical to A:", np.array_equal(ca[:n_res], cb[:n_res]),
          "(False = the HDF5 snapshot at kill -9 was inconsistent; allowed)")
sa = pd.read_csv(f"{out}/A/FZ_Tau/summary.csv").set_index("parameter"); sb = pd.read_csv(f"{out}/B/FZ_Tau/summary.csv").set_index("parameter")
sig = 0.5 * (sa["minus"] + sa["plus"]).clip(lower=1e-12)
z = ((sb["median"] - sa["median"]) / sig).abs()
print("killed/resumed B vs A posterior medians: max |dmedian| / sigma_post = %.2f (%s)" % (z.max(), z.idxmax()))
ok &= bool(z.max() < 1.0)
print("RESUME CHECK:", "passed" if ok else "FAILED -- report this (attach runs/results/smoke_0.22/*.log)")
PY
