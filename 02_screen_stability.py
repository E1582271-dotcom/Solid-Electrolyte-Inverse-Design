"""
Project 3 -- step 2: stability screen via a self-consistent MLIP convex hull.

For each generated candidate: relax with a universal MLIP, then compute energy above the
hull (eV/atom). The hull is built from MP reference phases relaxed with the SAME MLIP
(self-consistent -- see src/stability.py for why). Also flag uniqueness (de-dup among
candidates) and novelty (no matching MP phase) -> the S.U.N. lens.

Calculators:
    --calc mace    MACE-MP-0   (GPU; the real screen -- run in the cloud notebook)
    --calc chgnet  CHGNet      (GPU)
    --calc lj      Lennard-Jones -- CPU PLUMBING ONLY (meaningless energies), to verify
                   the relax->hull->e_above_hull->novelty wiring end-to-end on a laptop.

Output: data/screened.csv + relaxed CIFs in data/relaxed/.

Run:
    ~/Code/AI4SSB/.venv/bin/python 02_screen_stability.py --calc lj --max-refs 6 --steps 3 --no-relax-cell   # laptop plumbing
    python 02_screen_stability.py --calc mace --ehull-cutoff 0.1   # cloud, real
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from src import generate as G
from src import stability as ST
from src import novelty as NV

DATA = os.path.join(HERE, "data")
GEN_DIR = os.path.join(DATA, "generated")
RELAX_DIR = os.path.join(DATA, "relaxed")
REF_CACHE = os.path.join(DATA, "ref_structures.json")   # portable structure cache (see stability.py)
# NB: the hull energy cache is per-calculator (built in main() from args.calc) so an LJ
# plumbing cache can never feed a MACE hull -- mixing reference states would silently
# bias e_above_hull, the very thing the self-consistent hull exists to prevent.


def main():
    ap = argparse.ArgumentParser(description="Stability screen (self-consistent MLIP hull).")
    ap.add_argument("--calc", choices=ST.SUPPORTED, default="mace")
    ap.add_argument("--chemsys", default="Li-P-S")
    ap.add_argument("--ehull-cutoff", type=float, default=0.1, help="keep e_above_hull <= this (eV/atom)")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--relax-cell", dest="relax_cell", action="store_true", default=True)
    ap.add_argument("--no-relax-cell", dest="relax_cell", action="store_false")
    ap.add_argument("--max-refs", type=int, default=None, help="cap reference phases (plumbing)")
    ap.add_argument("--max-cands", type=int, default=None, help="cap candidates (plumbing)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    os.makedirs(RELAX_DIR, exist_ok=True)
    for f in os.listdir(RELAX_DIR):              # clear stale relaxed CIFs from prior runs
        if f.startswith("relaxed_") and f.endswith(".cif"):
            os.remove(os.path.join(RELAX_DIR, f))
    chemsys = args.chemsys.split("-")

    # candidates
    manifest = pd.read_csv(os.path.join(GEN_DIR, "manifest.csv"))
    if args.max_cands:
        manifest = manifest.head(args.max_cands)
    cands = G.load_structures(GEN_DIR, prefix="gen")[: len(manifest)]
    labels = manifest["label"].tolist()
    print(f"[screen] {len(cands)} candidates, calc={args.calc}, ehull cutoff {args.ehull_cutoff} eV/atom")

    calc = ST.load_calculator(args.calc, device=args.device)

    # reference phases -> self-consistent MLIP hull. Cache FILE is per-calculator so an LJ
    # plumbing cache can never be reused as MACE energies (entries are keyed by material_id,
    # which is calculator-agnostic -- the filename is what keeps the calculators apart).
    import json
    energy_cache_path = os.path.join(DATA, f"hull_energy_cache_{args.calc}.json")
    energy_cache = json.load(open(energy_cache_path)) if os.path.exists(energy_cache_path) else {}
    refs = ST.fetch_reference_entries(chemsys, cache_path=REF_CACHE)
    print(f"[screen] {len(refs)} MP reference phases for {args.chemsys}; building MLIP hull ...")
    hull = ST.build_mlip_hull(refs, calc, energy_cache=energy_cache,
                              fmax=args.fmax, steps=args.steps, max_refs=args.max_refs)
    json.dump(hull["energy_cache"], open(energy_cache_path, "w"))
    pd_hull = hull["pd"]
    ref_structs = NV.reference_structures_from_entries(refs)

    # screen each candidate
    rows, relaxed = [], []
    for lab, s in zip(labels, cands):
        r = ST.e_above_hull(s, calc, pd_hull, fmax=args.fmax, steps=args.steps)
        relaxed.append(r["relaxed"])
        novel = NV.is_novel(r["relaxed"], ref_structs)
        rows.append({
            "label": lab,
            "formula": r["relaxed"].composition.reduced_formula,
            "n_atoms": r["n_atoms"],
            "e_above_hull": round(r["e_above_hull"], 4),
            "converged": r["converged"],
            "novel": novel,
            "decomp": "+".join(r["decomp"].keys()),
        })
        rp = os.path.join(RELAX_DIR, f"relaxed_{lab}.cif")
        r["relaxed"].to(filename=rp)

    # uniqueness among candidates
    reps, groups = NV.dedupe(relaxed)
    rep_set = set(reps)
    for i, row in enumerate(rows):
        row["unique"] = i in rep_set

    df = pd.DataFrame(rows)
    df["stable"] = df["e_above_hull"] <= args.ehull_cutoff
    df["SUN"] = df["stable"] & df["unique"] & df["novel"]
    out = os.path.join(DATA, "screened.csv")
    df.to_csv(out, index=False)

    show = df[["label", "formula", "e_above_hull", "converged", "stable", "unique", "novel", "SUN"]]
    print("\n=== screened candidates ===")
    print(show.to_string(index=False))
    n_stable = int(df["stable"].sum())
    n_sun = int(df["SUN"].sum())
    print(f"\n{n_stable}/{len(df)} within {args.ehull_cutoff} eV/atom of hull; "
          f"{n_sun} are Stable+Unique+Novel (S.U.N.).")
    if args.calc == "lj":
        print("NOTE: --calc lj is a PLUMBING check only -- e_above_hull values are meaningless.")
    print(f"Saved {os.path.relpath(out, HERE)} + relaxed CIFs in data/relaxed/")


if __name__ == "__main__":
    main()
