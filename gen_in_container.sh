#!/bin/bash
# Runs INSIDE the Vanda pytorch_2.5 container (called by run_p3.pbs, stage A).
# MatterGen lives in an isolated uv venv at ~/mattergen/.venv (built by setup_mattergen.sh on
# the login node). The chemical_system checkpoint was pre-fetched into $HF_HOME, so this runs
# offline. Params arrive as env vars (run_p3.pbs via SINGULARITYENV_*): BATCH NBATCH GUIDANCE CHEMSYS.
set -e
MG="$HOME/mattergen"
RESULTS="$MG/results/chemical_system"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"   # pre-populated checkpoint cache (offline)
cd "$MG"

# fail fast if the A40 isn't visible (e.g. a stray --nv, or a CPU node) -- MatterGen
# diffusion sampling on CPU would run for hours and waste the whole walltime.
.venv/bin/python -c "import torch,sys; ok=torch.cuda.is_available(); print('[gpu]', ok, \
torch.cuda.get_device_name(0) if ok else 'NO-GPU'); sys.exit(0 if ok else 1)"

# Chemical-system conditioned generation (single-quoted dict is required; no shell here, so the
# braces/quotes reach the CLI literally). Matches src/generate.py + the notebook.
.venv/bin/mattergen-generate "$RESULTS" \
  --pretrained-name=chemical_system \
  --batch_size="${BATCH:-16}" \
  --num_batches="${NBATCH:-4}" \
  --properties_to_condition_on="{'chemical_system': '${CHEMSYS:-Li-P-S}'}" \
  --diffusion_guidance_factor="${GUIDANCE:-2.0}"

# hand the output to project 3's downstream seam (01_generate --source from-results)
DST="$HOME/AI4SSB/project3_generative/data/mattergen_run"
mkdir -p "$DST"
cp -f "$RESULTS/generated_crystals_cif.zip"  "$DST/" 2>/dev/null || true
cp -f "$RESULTS/generated_crystals.extxyz"   "$DST/" 2>/dev/null || true
echo "[gen] copied MatterGen output -> $DST"; ls -la "$DST"
