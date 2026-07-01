#!/bin/bash
# Runs INSIDE the Vanda pytorch_2.5 container (called by run_p3.pbs, stage B).
# Reuses project 2's proven MACE user-site (~/macepkg) and layers the project-3 extras
# (~/p3pkg: catboost for the project-1 model). MACE-MP-0 weights were pre-downloaded by
# setup_screen.sh. Runs 01(from-results) -> 02(screen) -> 03(score) -> 04(rank).
# Params arrive as env vars (run_p3.pbs via SINGULARITYENV_*): EHULL TOP.
set -e
export PYTHONUSERBASE="$HOME/p3pkg"
export PYTHONPATH="$HOME/macepkg/lib/python3.10/site-packages:$HOME/p3pkg/lib/python3.10/site-packages:$PYTHONPATH"
cd "$HOME/AI4SSB/project3_generative"

# fail fast if the A40 isn't visible (e.g. a stray --nv, or a CPU node) -- relaxing ~160
# structures on CPU would silently burn the whole walltime.
python3 -c "import torch,sys; ok=torch.cuda.is_available(); print('[gpu]', ok, \
torch.cuda.get_device_name(0) if ok else 'NO-GPU'); sys.exit(0 if ok else 1)"
python3 -c "import mace, catboost, pymatgen; print('[pkgs] mace/catboost/pymatgen ok')"

# Clean stale runtime products so a real run never inherits laptop LJ fixtures (the cloud
# analogue of the notebook's clean-up cell). data/ref_entries.pkl is kept (MP structures are
# calculator-agnostic -> reused for the hull, so no MP_API_KEY needed here).
rm -rf data/generated/* data/relaxed/* data/screened.csv data/scored.csv \
       data/candidates_final.csv data/hull_energy_cache*.json figures/0*.png 2>/dev/null || true

python3 01_generate.py --source from-results --results-path data/mattergen_run
python3 02_screen_stability.py --calc mace --device cuda --ehull-cutoff "${EHULL:-0.1}" --fmax 0.05 --steps 200
python3 03_score_conductivity.py --stable-only
python3 04_rank_candidates.py --top "${TOP:-5}" --ehull-cutoff "${EHULL:-0.1}" \
  --chemsys "${CHEMSYS:-Li-P-S}" --batch-size "${BATCH:-16}" --num-batches "${NBATCH:-4}" --guidance "${GUIDANCE:-2.0}"
echo "[screen] done -> data/candidates_final.csv + figures/* + data/p3_runs.json"
