#!/bin/bash
# Run ON THE LOGIN NODE (has internet) inside the Vanda pytorch_2.5 container:
#   module load singularity
#   singularity exec /app1/common/singularity-img/vanda/pytorch_2.5_cuda_12.4_unsloth.sif \
#       bash ~/AI4SSB/project3_generative/setup_screen.sh
#
# Stage B (screen + score) REUSES project 2's proven MACE user-site (~/macepkg). Here we
# only add the project-3 deltas into an ISOLATED ~/p3pkg -- catboost (the project-1 model)
# and mp-api (only used if data/ref_entries.pkl is absent) -- then pre-download MACE-MP-0
# weights so the offline compute node can relax. Layered, so macepkg is never clobbered.
set -e
export PYTHONUSERBASE="$HOME/p3pkg"; unset PIP_PREFIX; mkdir -p "$HOME/p3pkg"

echo "=== install screen extras into ~/p3pkg ==="
# pymatgen is listed defensively (no-op if macepkg already provides it; macepkg wins on PATH).
python3 -m pip install --user --no-cache-dir catboost mp-api pymatgen 2>&1 | tail -30

export PYTHONPATH="$HOME/macepkg/lib/python3.10/site-packages:$HOME/p3pkg/lib/python3.10/site-packages:$PYTHONPATH"
echo "=== sanity (mace from macepkg, catboost+pymatgen resolvable; eyeball torch cu build) ==="
# torch comes from macepkg (container's 2.5+cu124, the A40-proven build); print its CUDA build
# so a stray cu130/cpu torch in p3pkg never sneaks in (project-2 gotcha).
python3 -c "import torch, mace, catboost, pymatgen; print('mace/catboost/pymatgen ok | torch', torch.__version__, 'cuda', torch.version.cuda)"

echo "=== pre-download MACE-MP-0 weights (CPU init, no GPU needed) ==="
python3 -c "from mace.calculators import mace_mp; mace_mp(model='medium', dispersion=False, default_dtype='float32', device='cpu'); print('MACE_WEIGHTS_OK')" 2>&1 | tail -8
echo "=== setup_screen DONE ==="
