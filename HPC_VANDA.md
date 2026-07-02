# Running Project 3 on NUS Vanda (A40) — the main path

**Vanda HPC is Project 3's main path** (same machinery as Project 2: PBS + Singularity +
user-site package isolation). Colab is only a fallback for stage A (see the end).

> **Context.** Project 2's **MACE baseline has already run on Vanda A40** (50 ps × 600/800/1000 K,
> σ₃₀₀ ≈ 29.6 mS/cm, Ea ≈ 0.198 eV — ~9.4× over experiment, as expected for an un-fine-tuned
> potential). So stage B (MACE relaxation on the A40, reusing `~/macepkg`) runs a **proven path**.
> **Only stage A (MatterGen) is new** — it had never run on Vanda, so it is the real check on the
> first run (see Risks). Numbers not backed by the scripts (quota GB, scratch paths, …) are left
> unstated here — take them from Vanda itself.

## Environment facts (verified)

- **GPU = NVIDIA A40** (Ampere, compute capability 8.6, 48 GB VRAM).
- Image `pytorch_2.5_cuda_12.4_unsloth.sif` (py3.10 + torch 2.5 + **CUDA 12.4**) → no A40
  compatibility issue (not Blackwell, no need for CUDA ≥ 12.8).
- MLIP / generation both run in **FP32**; the A40 is strong at single precision, weak at FP64 (this
  workflow never touches FP64). The container only has `python3` (no `python`).
- ⚠️ **torch's CUDA build must be ≤ the A40 driver (12.4).** A 12.4 driver runs any **older** toolkit
  (cu118/cu121/cu124 all fine), but not a **newer** one (e.g. `cu130`) — that silently falls back to
  CPU on the A40 (a known trap). MatterGen pins `torch 2.2.1+cu118`, which is fine on the 12.4 driver;
  `setup_mattergen.sh`'s assertion only rejects a build newer than the driver, or a `cpu` build — a
  `TORCH_CUDA_OK` line means it passed. **Do not manually change torch's cu build in the venv** — its
  compiled extensions (`torch_scatter`, etc.) are built for cu118, and swapping the torch cu build
  makes them fail to find `libcudart`. Stage B reuses the container torch from macepkg, unaffected.

Two stages, two package trees (same image, mutually non-clobbering):
- **A — generate**: MatterGen in an isolated uv venv `~/mattergen/.venv` (its own resolved torch,
  independent of the container's 2.5).
- **B — screen + score**: **reuse Project 2's already-installed `~/macepkg`** (mace + ase); only add
  catboost + pymatgen + mp-api into `~/p3pkg`.

## 0) Prerequisite: get the code onto Vanda

The scoring model is **vendored** in the repo (`vendor/`), so Project 3 is **self-contained** — you
only need this project on Vanda:

```
~/AI4SSB/project3_generative/     # this project (includes vendor/ + data/ref_structures.json)
```

Bring `data/ref_structures.json` along → the hull uses the 96 cached MP reference phases directly,
so **no MP_API_KEY is needed** (it is a portable structure cache, readable across pymatgen versions;
do not use a locally pickled `ref_entries.pkl` — a newer pymatgen cannot unpickle it).

## 1) One-time environment setup (login node, has internet)

> Compute nodes are offline, so package installs + weight pre-downloads must happen on the **login
> node** (which has internet).

```bash
module load singularity
IMG=/app1/common/singularity-img/vanda/pytorch_2.5_cuda_12.4_unsloth.sif

# Stage A: build the MatterGen venv + pre-download the chemical_system weights into ~/hf_cache
singularity exec "$IMG" bash ~/AI4SSB/project3_generative/setup_mattergen.sh

# Stage B: add catboost+pymatgen into ~/p3pkg (reusing ~/macepkg) + pre-download MACE-MP-0 weights
singularity exec "$IMG" bash ~/AI4SSB/project3_generative/setup_screen.sh
```

`MATTERGEN_WEIGHTS_OK` and `mace/catboost/pymatgen ok` + `MACE_WEIGHTS_OK` mean it is set up (the
latter also confirms the "reuse macepkg + layer p3pkg" assumption holds).

## 2) Submit the job (compute node, offline)

One GPU job, two `singularity exec` steps (A → B). **Do not pass `--nv` on NUS** (the container
exposes the GPU automatically; `--nv` actually hides CUDA).

```bash
cd ~/AI4SSB/project3_generative
# smoke first (a few minutes, validates both stages)
qsub -v BATCH=4,NBATCH=1,EHULL=0.1,TOP=5 run_p3.pbs
# then the full run (~64 candidates)
qsub -v BATCH=16,NBATCH=4,EHULL=0.1,TOP=5 run_p3.pbs
```

Parameters (`qsub -v`, comma-separated; none of these values contain commas, so no need for
Project 2's `_`→`,` trick): `BATCH` structures per batch, `NBATCH` number of batches (total = their
product), `GUIDANCE` diffusion guidance factor, `CHEMSYS` chemical system (default Li-P-S), `EHULL`
stability cutoff in eV/atom, `TOP` shortlist size. Runs on the personal free budget — **no `-P`**.

## 3) Submit / monitor / collect (all doable from the Mac, asynchronously)

```bash
# one-line submit (needs passwordless ssh to vanda set up; off-campus, connect NUS nVPN first)
ssh vanda "cd ~/AI4SSB/project3_generative && qsub -v BATCH=4,NBATCH=1 run_p3.pbs"

# monitor
qstat -u $USER         # all my jobs
qstat -f <jobid>       # one job in detail
qdel  <jobid>          # cancel

# collect results back to the laptop
rsync -az vanda:~/AI4SSB/project3_generative/{data,figures}/ ./
```

> Once `qsub` queues a job it runs independently on the server — **losing the network / closing the
> Mac does not affect it**. The GPU queue caps each job at **2 GPU + 2 nodes** (per job, not per
> user); to parallelise over chemical systems / parameters, **submit several jobs**. No `-P` = the
> personal free budget, lower priority, so it may queue when GPUs are busy.

## 4) Outputs

- `data/candidates_final.csv` — every candidate, ranked by predicted log₁₀σ, with e_above_hull /
  S.U.N. flags.
- `figures/01_landscape.png` / `figures/02_shortlist.png` — raw diagnostic figures from
  `04_rank_candidates.py` (regenerated on every run; not tracked). The tracked deliverable is
  `figures/fig_inverse_design.{png,svg}`, consolidated from both by
  `figures/make_publication_figure.py`.

Hand the shortlist head to Project 2 MLIP-MD for a real σ/Eₐ (W11), then backfill the README results
table.

## Risks & fallback

| stage | status |
|---|---|
| Stage B (MACE screen + score + rank) | reuses the **MACE path Project 2 already ran on the A40** (baseline σ₃₀₀ ≈ 29.6 mS/cm) + the installed macepkg; the local CPU plumbing is also verified → low risk |
| Stage A (MatterGen install + run) | commands verified against the upstream README; MatterGen is new to this project, so three things to confirm on the first Vanda run (below) |

**Three things to watch on stage A's first run:**
1. **Weight path.** `setup_mattergen.sh` uses `snapshot_download(microsoft/mattergen,
   checkpoints/chemical_system/*)` to pre-download into `~/hf_cache`, and the PBS job passes the same
   `HF_HOME` into the container. If the upstream checkpoint path moves, the error is on this line —
   adjust `allow_patterns`.
2. **Building the venv needs internet.** `setup_mattergen.sh` must run on the **login node** (compute
   nodes are offline); the venv and weights live under `$HOME`, so compute nodes can use them directly.
3. **The venv torch CUDA build must be ≤ 12.4** (a known trap). `setup_mattergen.sh`'s assertion
   rejects a build newer than the driver (`cu130`) or a `cpu` build (either falls back to CPU on the
   A40); MatterGen's `cu118` is fine on a 12.4 driver — a `TORCH_CUDA_OK` line means it passed.
   ⚠️ Do not manually change torch's cu build in the venv — extensions like `torch_scatter` are built
   for cu118 and will crash (`libcudart.so.11.0` not found).

**Fallback**: if stage A is hard to install on Vanda, run stage A on free Colab T4 instead (stage A
of the notebook `01_mattergen_pipeline.ipynb`, already set up), download `generated_crystals_cif.zip`
into `data/mattergen_run/`, and submit only stage B on Vanda (comment out the stage-A line
`singularity exec ... gen_in_container.sh` in `run_p3.pbs`).

> Optional upgrade: the official `chemical_system_energy_above_hull` multi-property model conditions
> on **chemical system + stability** at once
> (`--properties_to_condition_on="{'energy_above_hull': 0.05, 'chemical_system': 'Li-P-S'}"`),
> yielding candidates closer to the hull. Swap this `--pretrained-name` when a stronger signal is wanted.
