# Methods

The production run was made on NUS Vanda (NVIDIA A40) on 2026-07-01; the 50 stable candidates were
rescored on 2026-10-19 with Project 1's current conductivity ensemble. Cluster setup and job
submission: [hpc_vanda.md](hpc_vanda.md).

## Pipeline

1. **Generate.** Candidate structures are sampled from a pretrained **MatterGen** diffusion model
   conditioned on the chemical system Li-P-S (`chemical_system` checkpoint, batch 16 × 4 = 64,
   guidance 2.0). Candidates without Li are dropped (`01_generate.py --require-elements Li`, on by
   default).
2. **Stability screen.** Each candidate is relaxed with a universal MLIP (**MACE-MP-0**), its
   energy above the convex hull (`e_above_hull`) is computed on a self-consistent MLIP hull of 96
   Materials Project reference phases, and candidates with `e_above_hull` ≤ 0.1 eV/atom are kept.
   Each candidate is flagged **S.U.N.** (Stable + Unique + Novel), the lens MatterGen reports for
   itself: unique = not a StructureMatcher duplicate of another candidate, novel = not matched to a
   known MP phase.
3. **Score.** Project 1's OBELiX-trained CatBoost conductivity ensemble (formula-only) gives the
   survivors a coarse, descriptive prior ([vendor/README.md](../vendor/README.md)).
4. **Output.** The S.U.N. set of new and relatively stable candidates (43 structures). All of them
   are the next batch for the Solid-Electrolyte-MLIP-MD census protocol; the prior annotates them but
   is not used to cut the list.

## Scripts

Numbered pipeline scripts with reusable code in `src/`:

| Script | Role | Runs on |
|---|---|---|
| `src/generate.py` + `01_generate.py` | `--source mattergen` runs the official conditional-generation CLI; `--source from-results` reads an existing `*_cif.zip`; `--source mp-demo` pulls real Li-P-S phases as an **offline plumbing fixture** (not generative output; `--rattle` displaces atoms to mimic unrelaxed cells) | mattergen = GPU; else CPU |
| `src/stability.py` + `02_screen_stability.py` | reference phases **and** candidates relaxed with the **same** MLIP → self-consistent convex hull → `e_above_hull`; `--calc mace\|chgnet` (real) / `lj` (CPU plumbing fixture, energies meaningless) | GPU (except lj) |
| `src/novelty.py` | StructureMatcher de-duplication among candidates (Unique) and comparison against known MP phases (Novel) | CPU |
| `src/score.py` + `03_score_conductivity.py` | applies the vendored Project 1 production ensemble (`vendor/p1_conductivity.py` + 15 `vendor/p1_models/*.cbm`, checksums in `vendor/p1_manifest.json`) to each candidate's formula → log₁₀σ prior = ensemble mean, with ensemble SD and applicability-domain distance | CPU |
| `04_rank_candidates.py` | merges screen and score → `candidates_final.csv` + `p3_runs.json` + two quick-look figures | CPU |
| `figures/make_publication_figure.py`, `figures/make_generation_funnel_figure.py` | the two tracked figures ([figures.md](figures.md)) | CPU |
| `notebooks/01_mattergen_pipeline.ipynb` | cloud orchestration: stage A (isolated py3.10 venv runs MatterGen) → stage B (MACE screen + score) | Colab T4 / Vanda |

## Conductivity prior

Project 1's production conductivity ranker is a 15-member CatBoost ensemble on 134 formula-only
Magpie descriptors, trained on all 599 OBELiX records. It is **vendored** under `vendor/` (15
`.cbm` members + `p1_conductivity.py`, sha256 of every member in `vendor/p1_manifest.json`, matching
Project 1's release manifest), so this repository clones and runs on its own and `score.py` always
uses the vendored copy. Per candidate it reports the ensemble mean (`pred_log10_sigma`), the
ensemble SD, the applicability-domain distance, whether the composition is already in OBELiX, and
whether spread or AD distance reaches Project 1's frozen q75 route thresholds (0.385 / 12.15). To
pick up a new [Project 1](https://github.com/E1582271-dotcom/Solid-Electrolyte-Screening) release,
run `python vendor/refresh_from_p1.py --p1 <path-to-Solid-Electrolyte-Screening>`
([vendor/README.md](../vendor/README.md)).

## Design decisions

- **Two-stage isolation.** MatterGen pins Python 3.10 with its own torch/lightning stack, which
  clashes with the MACE screening environment. Generation runs as a CLI subprocess in a dedicated
  venv; screening and scoring run in the kernel; the `--source from-results` seam joins them, so the
  screening environment does not depend on the mattergen package.
- **Self-consistent MLIP hull** (not mixed MP/PBE energies). Reference phases and candidates are
  relaxed with the same universal MLIP before building the hull. This avoids the systematic
  `e_above_hull` bias of comparing an MLIP candidate energy with a PBE+U reference energy
  (inconsistent reference states) and matches MatterGen's own evaluator. Reference energies are
  cached by `material_id`, so re-runs are fast.
- **CPU plumbing fixtures** (`mp-demo` + `lj`). Following the Lennard-Jones fixture of
  Solid-Electrolyte-MLIP-MD, the whole downstream pipeline can be developed and checked on a laptop
  without a GPU; those outputs are labelled "plumbing check, non-physical".
- **Require the mobile ion Li** (`01_generate.py --require-elements Li`, on by default). Conditioned
  on `chemical_system=Li-P-S`, MatterGen also emits Li-free P–S binaries, which the Project 1 model
  still scores, so a Li-ion-conductor screen must first require Li or the top of the ranking fills
  with meaningless Li-free phases (this run dropped 3 Li-free phases, keeping 61).
- **Portable reference cache** (`data/ref_structures.json`, not a pickle). Reference phases are
  cached as pymatgen `Structure` (`as_dict`), readable across pymatgen versions; a directly pickled
  MP entry fails to unpickle when the two ends run different pymatgen versions.

## Limitations

- **Most generated structures are unstable or unsynthesizable**, so they must pass the MLIP
  stability screen. This is concept validation; synthesizability is not claimed. MatterGen's own
  paper validated a single synthesis.
- `e_above_hull` comes from an **un-fine-tuned** universal MLIP: a relative-stability indicator, not
  DFT-quantitative. The self-consistent hull removes the reference-state mismatch, but the MLIP's
  own bias remains.
- The conductivity score is a **weak, descriptive prior**. It is formula-only and never sees the
  generated structure, so every polymorph of a composition receives the same score (ties are broken
  by `e_above_hull`). On this run it largely tracks Li fraction (Spearman ρ = 0.93 against Li atomic
  fraction over the 26 compositions; `Li_frac` is itself one of the 134 features) and favours
  Li-rich formulas that mostly do not charge-balance. For compositions already in OBELiX (Li₃PS₄,
  Li₂PS₃, LiPS₃; 18 of the 50 scored structures) it largely echoes measured values. It describes
  candidates but is not used to select them, and it is **not a quantitative σ**.
- No generated structure has a simulated conductivity yet: the 43 S.U.N. structures have not been
  through the Solid-Electrolyte-MLIP-MD census.
