"""Generate 01_mattergen_pipeline.ipynb (valid nbformat-4 JSON, no nbformat dep).

Colab-first orchestration of Project 3's inverse-design demo. Two stages:
  A) MatterGen conditional generation in an isolated uv/py3.10 env (CLI subprocess);
  B) stability screen (MACE hull) + vendored Project-1 conductivity score in the kernel env.
Re-run this script to regenerate the notebook after editing the cell sources below.
"""
import json
import os

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip("\n").splitlines(keepends=True)}

cells = [
md("""# Inverse design — MatterGen → MLIP stability screen → conductivity score (Project 3, bonus)

**Concept validation**: demonstrates the inverse-design paradigm and a *generate → screen → validate*
loop. **No synthesizability is claimed** — MatterGen's own paper validated a single synthesis.

Position in the portfolio: Project 1 (screening) → **Project 3 (generate new candidates)** →
Project 2 (high-accuracy MLIP-MD validation) → future electrochemical experiment.

---
**Two stages** (MatterGen needs an isolated Python 3.10 environment, kept apart from the screen):
- **A — generate**: in a uv-built py3.10 venv, run the official `mattergen-generate` CLI (condition =
  chemical system Li-P-S) → `generated_crystals_cif.zip`.
- **B — screen + score**: back in the kernel, use a MACE-MP-0 self-consistent hull to compute
  `e_above_hull` and keep stable candidates → score with the vendored Project-1 CatBoost model → rank.

> **Vanda HPC alternative** (the preferred path): run stage A in the Vanda container with an isolated
> venv and stage B by reusing the installed `~/macepkg`. See `HPC_VANDA.md` — same commands, just run
> inside the container instead of via `!`-shell."""),

md("""## 0) Environment + get the code

This repo is **self-contained** (the Project-1 conductivity model is vendored under `vendor/`), so
cloning it alone is enough — no need for the Project 1 repo alongside."""),

code("""# GPU check
import torch, sys
print("python", sys.version.split()[0], "| torch", torch.__version__, "| CUDA", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only (generation will be slow)")"""),

code("""# Get the code: clone this (public, self-contained) repo.
import os, subprocess
REPO_URL = "https://github.com/E1582271-dotcom/Solid-Electrolyte-Inverse-Design.git"
ROOT = "/content/project3"
if not os.path.isdir(ROOT):
    subprocess.run(["git", "clone", REPO_URL, ROOT], check=True)
assert os.path.isfile(os.path.join(ROOT, "vendor", "catboost_model.cbm")), \\
    "vendored conductivity model missing -- clone the full repo"
os.chdir(ROOT)
print("working dir:", os.getcwd())"""),

code("""# Clean any committed run products so this notebook produces its own. Keep the portable
# reference cache data/ref_structures.json (calculator-independent, avoids needing an MP API key).
import os, glob, shutil
for pat in ["data/generated/*", "data/relaxed/*", "data/mattergen_run/*",
            "data/screened.csv", "data/scored.csv", "data/candidates_final.csv", "data/p3_runs.json",
            "data/hull_energy_cache*.json", "figures/0*.png", "figures/fig_inverse_design.*"]:
    for p in glob.glob(pat):
        shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
print("cleaned committed run products (kept ref_structures.json for the offline hull)")"""),

code("""# MP_API_KEY -- OPTIONAL. The shipped data/ref_structures.json already provides the 96 reference
# phases, so the hull builds offline. Only set this if you delete that cache and want to re-fetch.
import os
try:
    from google.colab import userdata
    os.environ["MP_API_KEY"] = userdata.get("MP_API_KEY")
    print("MP_API_KEY loaded from Colab Secret")
except Exception as e:
    print("no MP_API_KEY secret (fine -- ref_structures.json is shipped):", e)"""),

md("""## A) Generate — MatterGen conditional generation of Li-P-S candidates

Install MatterGen in an isolated uv/py3.10 venv and run its CLI (to avoid clashing with the kernel's
torch/e3nn). **Follow the official [microsoft/mattergen](https://github.com/microsoft/mattergen)
README for the install/CLI (versions change).** The `chemical_system` pretrained weights are on
HuggingFace and download automatically on first use."""),

code("""# A1) Install MatterGen in an isolated env (git clone + uv, py3.10). ~5-10 min the first time.
import os, subprocess
MG = "/content/mattergen"
if not os.path.isdir(MG):
    subprocess.run(["git", "clone", "https://github.com/microsoft/mattergen.git", MG], check=True)
subprocess.run(["pip", "install", "-q", "uv"], check=True)
# build a py3.10 venv in the mattergen dir + editable install; then call the venv's executables
subprocess.run(["uv", "venv", ".venv", "--python", "3.10"], cwd=MG, check=True)
subprocess.run(["uv", "pip", "install", "-e", ".", "--python", ".venv/bin/python"], cwd=MG, check=True)
print("MatterGen environment ready:", MG)"""),

code("""# A2) Conditional generation: sample candidates conditioned on the chemical system Li-P-S.
#     Thin run: batch_size=16, num_batches=2 (~32 total). Raise num_batches for more.
import os, shutil, subprocess
MG = "/content/mattergen"
RESULTS = os.path.join(MG, "results", "chemical_system")
cmd = [".venv/bin/mattergen-generate", RESULTS,
       "--pretrained-name=chemical_system",
       "--batch_size=16", "--num_batches=2",
       "--properties_to_condition_on={'chemical_system': 'Li-P-S'}",
       "--diffusion_guidance_factor=2.0"]
print("$", " ".join(cmd))
subprocess.run(cmd, cwd=MG, check=True)

# copy the MatterGen output into the project's data/ for the from-results reader
import os
DST = os.path.join(os.getcwd(), "data", "mattergen_run")
os.makedirs(DST, exist_ok=True)
for fn in ("generated_crystals_cif.zip", "generated_crystals.extxyz"):
    src = os.path.join(RESULTS, fn)
    if os.path.exists(src):
        shutil.copy(src, DST)
print("copied MatterGen output to", DST, "->", os.listdir(DST))"""),

code("""# A3) Normalise to a single manifest + per-structure CIFs (pure pymatgen, runs in the kernel)
!python 01_generate.py --source from-results --results-path data/mattergen_run"""),

md("""## B) Screen + score — MACE self-consistent hull + conductivity score

- **Stability**: reference phases and candidates are relaxed with the **same** MACE-MP-0, and the
  hull is built self-consistently from those MLIP energies (avoids the bias from mixing MLIP
  candidate energies with PBE+U reference energies; see `src/stability.py`). Reference energies are
  cached by material_id, so re-runs are fast.
- **S.U.N.**: Stable (`e_above_hull` ≤ cutoff) + Unique (de-dup among candidates) + Novel (no match to
  any known MP phase) — the standard lens MatterGen reports.
- **Score**: the vendored Project-1 OBELiX-CatBoost model gives survivors a **coarse conductivity
  prior (for ranking, not a quantitative σ)**."""),

code("""# B1) Install the screen/score deps into the kernel (MACE + pymatgen stack). ~2-4 min.
!pip install -q mace-torch pymatgen mp-api catboost"""),

code("""# B2) Stability screen: MACE self-consistent hull -> e_above_hull + de-dup + novelty.
#     Downloads MACE-MP-0 weights on first use; reference energies cache to data/hull_energy_cache_mace.json
!python 02_screen_stability.py --calc mace --ehull-cutoff 0.1 --fmax 0.05 --steps 200"""),

code("""# B3) Score the survivors with the vendored Project-1 model (stable ones only)
!python 03_score_conductivity.py --stable-only"""),

code("""# B4) Merge + rank + plot (landscape + final shortlist)
!python 04_rank_candidates.py --top 5"""),

code("""# B5) Show the result figures
from IPython.display import Image, display
for f in ["figures/01_landscape.png", "figures/02_shortlist.png"]:
    print(f); display(Image(f))"""),

md("""## After it runs / honest disclosure

- **Most generated structures are unstable / unsynthesizable** → they must pass the MLIP stability
  screen; that is the demo's core signal.
- **e_above_hull comes from a universal MLIP** (not fine-tuned for sulfides): a relative-stability
  indicator, not DFT-quantitative. The self-consistent hull removes the reference-state mismatch, but
  the MLIP's own bias remains.
- **The conductivity score is a coarse ranking prior** (Project-1 model); generated stoichiometries
  score with `Family='unknown'` and are often P1-symmetry — the score is for ranking and handoff to
  Project 2 MD only.
- **Positioning**: concept validation + closed-loop demo, with no over-promise on synthesizability.
- **Next**: hand the S.U.N. shortlist head to Project 2 MLIP-MD for a real σ/Eₐ; draw the unified
  pipeline diagram (W11)."""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_mattergen_pipeline.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote", out, "with", len(cells), "cells")
