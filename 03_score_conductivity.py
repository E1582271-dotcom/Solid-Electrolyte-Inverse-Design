"""
Project 3 -- step 3: score survivors with the Project 1 conductivity model.

Applies the OBELiX-trained CatBoost ranker (project 1) to the MLIP-relaxed candidates,
giving each a coarse log10(ionic conductivity) prior. This is the seam that turns three
separate projects into one pipeline: screen (P1) -> generate (P3) -> rank by conductivity
(P1 model) -> validate by MD (P2).

Honest, same as project 1: a RANK, not a quantitative sigma. Generated chemistries score
on composition + structure only (Family heuristic / 'unknown').

Reads data/screened.csv + data/relaxed/relaxed_<label>.cif; writes data/scored.csv.

Run:
    ~/Code/AI4SSB/.venv/bin/python 03_score_conductivity.py [--stable-only]
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from pymatgen.core import Structure

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from src import score as SC

DATA = os.path.join(HERE, "data")
RELAX_DIR = os.path.join(DATA, "relaxed")


def main():
    ap = argparse.ArgumentParser(description="Score candidates with the project-1 model.")
    ap.add_argument("--stable-only", action="store_true",
                    help="only score candidates flagged stable in the screen")
    args = ap.parse_args()

    screened = pd.read_csv(os.path.join(DATA, "screened.csv"))
    if args.stable_only:
        screened = screened[screened["stable"]].copy()
    if screened.empty:
        print("No candidates to score (try without --stable-only)."); return

    labels = screened["label"].tolist()
    structures = [Structure.from_file(os.path.join(RELAX_DIR, f"relaxed_{lab}.cif"))
                  for lab in labels]
    print(f"[score] scoring {len(structures)} relaxed candidates with the project-1 model ...")

    ranked = SC.score_structures(structures, labels=labels)
    keep = ["label", "formula", "Family", "spacegroup_no",
            "pred_log10_sigma", "pred_sigma_S_cm"]
    out = ranked[keep].copy()
    out.to_csv(os.path.join(DATA, "scored.csv"), index=False)

    show = out.copy()
    show["pred_log10_sigma"] = show["pred_log10_sigma"].round(2)
    show["pred_sigma_S_cm"] = show["pred_sigma_S_cm"].map(lambda v: f"{v:.1e}")
    print("\n=== candidates ranked by predicted log10 sigma ===")
    print(show.to_string(index=False))
    print("\nReminder: RANK, not quantitative sigma. Validate the head with project 2 MD.")
    print(f"Saved {os.path.relpath(os.path.join(DATA, 'scored.csv'), HERE)}")


if __name__ == "__main__":
    main()
