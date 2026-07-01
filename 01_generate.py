"""
Project 3 -- step 1: generate candidate Li-P-S structures.

Three sources (analogous to project 1's --demo and project 2's LJ check):

  --source mattergen     Run MatterGen's official CLI conditioned on the chemical system.
                         Needs the mattergen package + a GPU; only works where mattergen
                         is importable (i.e. inside its own venv), so usually the cloud.

  --source from-results  Read MatterGen output that ALREADY exists in --results-path
                         (generated_crystals_cif.zip / .extxyz). This is what the notebook
                         uses: generation runs as a CLI in MatterGen's dedicated py3.10
                         venv, then we just normalise its output here -- no re-run, no
                         dependency on mattergen in the screening env.

  --source mp-demo       Pull a handful of REAL, known Li-P-S phases from Materials Project
                         as stand-in "generated" candidates, so the whole downstream
                         screen->score->rank pipeline can be built and verified OFFLINE on a
                         CPU. NOT generative output -- a deliberate plumbing fixture (with
                         --rattle they are displaced to mimic raw, unrelaxed output).

Output: data/generated/gen_*.cif + data/generated/manifest.csv (label, formula, n_atoms, source).

Run:
    ~/Code/AI4SSB/.venv/bin/python 01_generate.py --source mp-demo --max-demo 6 --rattle 0.1
    python 01_generate.py --source mattergen --chemsys Li-P-S --batch-size 16 --num-batches 4   # cloud
    python 01_generate.py --source from-results --results-path results/chemical_system/        # cloud
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from src import generate as G

GEN_DIR = os.path.join(HERE, "data", "generated")


def main():
    ap = argparse.ArgumentParser(description="Generate / fetch candidate structures.")
    ap.add_argument("--source", choices=["mattergen", "from-results", "mp-demo"], default="mp-demo")
    ap.add_argument("--chemsys", default="Li-P-S", help="chemical system, e.g. Li-P-S")
    # MatterGen knobs
    ap.add_argument("--model", default=G.DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-batches", type=int, default=1)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--results-path", default=None, help="MatterGen results dir (default data/mattergen_run)")
    # mp-demo knobs
    ap.add_argument("--max-demo", type=int, default=6)
    ap.add_argument("--rattle", type=float, default=0.0, help="random displacement (A) for mp-demo")
    ap.add_argument("--require-elements", default="Li",
                    help="drop generated structures missing any of these elements (comma-sep). "
                         "Default 'Li': a Li-ion-conductor screen keeps only Li-bearing candidates "
                         "(MatterGen's chemical-system conditioning also emits Li-free P-S binaries). "
                         "Pass '' to disable.")
    args = ap.parse_args()

    os.makedirs(GEN_DIR, exist_ok=True)
    for f in os.listdir(GEN_DIR):                # clear stale candidates from prior runs
        if (f.startswith("gen_") and f.endswith(".cif")) or f == "manifest.csv":
            os.remove(os.path.join(GEN_DIR, f))

    if args.source == "mattergen":
        results_path = args.results_path or os.path.join(HERE, "data", "mattergen_run")
        G.run_mattergen_cli(
            results_path, chemsys=args.chemsys, model_name=args.model,
            batch_size=args.batch_size, num_batches=args.num_batches,
            guidance_factor=args.guidance,
        )
        structures = G.read_generated(results_path)
        source = f"mattergen:{args.model}"
        print(f"[generate] MatterGen produced {len(structures)} structures.")
    elif args.source == "from-results":
        results_path = args.results_path or os.path.join(HERE, "data", "mattergen_run")
        structures = G.read_generated(results_path)
        source = f"mattergen-results:{os.path.basename(results_path.rstrip('/'))}"
        print(f"[generate] read {len(structures)} MatterGen structures from {results_path}.")
    else:
        structures = G.mp_demo_candidates(
            chemsys=args.chemsys.split("-"), max_n=args.max_demo, rattle=args.rattle,
        )
        source = "mp-demo" + (f"+rattle{args.rattle}" if args.rattle else "")
        print(f"[generate] mp-demo fetched {len(structures)} real Li-P-S phases "
              f"(stand-in candidates, NOT generative output).")

    # A Li-ion-conductor screen must keep the mobile ion: drop candidates missing required
    # elements (default Li). MatterGen conditioned on chemical_system=Li-P-S also emits Li-free
    # P-S binaries, which the project-1 model would otherwise rank as "conductors".
    req = [e.strip() for e in args.require_elements.split(",") if e.strip()]
    if req:
        before = len(structures)
        structures = [s for s in structures
                      if set(req) <= {str(el) for el in s.composition.elements}]
        print(f"[generate] require-elements {req}: kept {len(structures)}/{before} "
              f"candidates (dropped ones missing {req}).")

    labels = [f"gen_{i:03d}" for i in range(len(structures))]
    paths = G.save_structures(structures, GEN_DIR, prefix="gen")
    manifest = pd.DataFrame({
        "label": labels,
        "formula": [s.composition.reduced_formula for s in structures],
        "n_atoms": [len(s) for s in structures],
        "source": source,
        "cif": [os.path.relpath(p, HERE) for p in paths],
    })
    mpath = os.path.join(GEN_DIR, "manifest.csv")
    manifest.to_csv(mpath, index=False)
    print(manifest.to_string(index=False))
    print(f"\nSaved {len(structures)} CIFs + {os.path.relpath(mpath, HERE)}")


if __name__ == "__main__":
    main()
