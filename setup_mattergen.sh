#!/bin/bash
# Run ON THE LOGIN NODE (has internet) inside the Vanda pytorch_2.5 container:
#   module load singularity
#   singularity exec /app1/common/singularity-img/vanda/pytorch_2.5_cuda_12.4_unsloth.sif \
#       bash ~/AI4SSB/project3_generative/setup_mattergen.sh
#
# Builds an ISOLATED uv venv for MatterGen (mirrors the proven Colab recipe: py3.10 +
# MatterGen's own resolved torch, so it never fights the container's torch 2.5) and
# pre-downloads the chemical_system checkpoint into a persistent HF cache, so the OFFLINE
# GPU compute node can generate without internet. Analogue of project 2's setup_mattersim.sh.
set -e
MG="$HOME/mattergen"
export HF_HOME="$HOME/hf_cache"; mkdir -p "$HF_HOME"

echo "=== clone mattergen (code only; checkpoints come via HF cache below) ==="
# GIT_LFS_SKIP_SMUDGE keeps the clone light -- we fetch the one checkpoint we need via HF.
[ -d "$MG" ] || GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/microsoft/mattergen.git "$MG"
cd "$MG"

echo "=== build py3.10 venv + editable install (uv) ==="
python3 -m pip install --user -q uv
UV="$HOME/.local/bin/uv"; command -v uv >/dev/null && UV="uv"
[ -d .venv ] || "$UV" venv .venv --python 3.10      # idempotent: reuse an existing venv
"$UV" pip install -e . --python .venv/bin/python

echo "=== verify venv torch CUDA build vs A40 driver (12.4) ==="
# Project-2 gotcha 2: a torch wheel built for a CUDA NEWER than the node driver (e.g. cu130 on a
# 12.4 driver) won't run and silently falls back to CPU. The reverse is fine -- the A40's 12.4
# driver runs any OLDER toolkit (cu118/cu121) via backward compatibility, so MatterGen's pinned
# cu118 stack (torch + its cu118-built torch_scatter etc.) is internally consistent and OK here.
# We only reject a cpu-only wheel or a build newer than the driver. Check on the login node
# (torch.version.cuda is the wheel's CUDA, independent of GPU presence).
.venv/bin/python - <<'PY'
import sys, torch
cu = torch.version.cuda
print("venv torch", torch.__version__, "| cuda build", cu)
ok = cu is not None and tuple(int(x) for x in cu.split(".")) <= (12, 4)
if not ok:
    sys.exit("FATAL: venv torch cuda build %r is incompatible with the A40 driver (12.4): need a "
             "CUDA build <= 12.4 (a cpu/None or cu13x wheel silently falls back to CPU)." % cu)
print("TORCH_CUDA_OK")
PY

echo "=== pre-download chemical_system checkpoint to \$HF_HOME ($HF_HOME) ==="
# Repo id confirmed by the upstream README; --pretrained-name=chemical_system resolves to
# checkpoints/chemical_system inside microsoft/mattergen on Hugging Face. Pulling it now
# means the offline compute node finds it in cache. If the path moves upstream, this is the
# one line to re-check (run with the same HF_HOME the PBS job uses).
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="microsoft/mattergen",
                      allow_patterns=["checkpoints/chemical_system/*"])
print("MATTERGEN_WEIGHTS_OK", p)
PY
echo "=== setup_mattergen DONE (venv: $MG/.venv ; weights: $HF_HOME) ==="
