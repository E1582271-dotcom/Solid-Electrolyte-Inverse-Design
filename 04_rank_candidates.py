"""
Project 3 -- step 4: combine the screen + conductivity score into a final shortlist.

Joins data/screened.csv (e_above_hull, S.U.N. flags) with data/scored.csv (predicted
conductivity) and produces:
  - data/candidates_final.csv : every candidate, all columns, ranked by predicted sigma.
  - figures/01_landscape.png  : e_above_hull vs predicted log10 sigma (the whole batch),
                                the "stable AND conductive" corner being what we want.
  - figures/02_shortlist.png  : the final S.U.N. shortlist, ranked by predicted sigma.

The shortlist = Stable (e_above_hull <= cutoff) AND Unique AND Novel, then ranked by the
project-1 conductivity prior. These are the "new and relatively stable" candidates the
demo set out to produce -- explicitly a concept-validation handoff to project 2 MD, NOT
a synthesizability claim.

Run:
    ~/Code/AI4SSB/.venv/bin/python 04_rank_candidates.py --top 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")


def main():
    ap = argparse.ArgumentParser(description="Rank + plot the final candidate shortlist.")
    ap.add_argument("--top", type=int, default=5, help="size of the printed/plotted shortlist")
    ap.add_argument("--ehull-cutoff", type=float, default=0.1)
    # passthrough run params -- only for the p3_runs.json audit log (mirrors project 2's md_runs.json)
    ap.add_argument("--chemsys", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-batches", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)

    screened = pd.read_csv(os.path.join(DATA, "screened.csv"))
    scored_path = os.path.join(DATA, "scored.csv")
    scored = (pd.read_csv(scored_path) if os.path.exists(scored_path)
              else pd.DataFrame(columns=["label", "pred_log10_sigma", "pred_sigma_S_cm"]))
    for c in ["label", "pred_log10_sigma", "pred_sigma_S_cm"]:   # tolerate an empty scored.csv
        if c not in scored.columns:
            scored[c] = pd.Series(dtype="float64")
    df = screened.merge(scored[["label", "pred_log10_sigma", "pred_sigma_S_cm"]],
                        on="label", how="left")
    df = df.sort_values("pred_log10_sigma", ascending=False).reset_index(drop=True)
    df.to_csv(os.path.join(DATA, "candidates_final.csv"), index=False)

    # --- figure 1: stability vs conductivity landscape ---
    fig, ax = plt.subplots(figsize=(6.4, 5))
    scored_mask = df["pred_log10_sigma"].notna()
    sub = df[scored_mask]
    for flag, color, marker, name in [
        (sub["SUN"], "#1F4E79", "*", "S.U.N. (stable+unique+novel)"),
        (~sub["SUN"] & sub["stable"], "#2E75B6", "o", "stable, not S.U.N."),
        (~sub["stable"], "#BFBFBF", "x", "unstable"),
    ]:
        s = sub[flag]
        if len(s):
            ax.scatter(s["e_above_hull"], s["pred_log10_sigma"],
                       c=color, marker=marker, s=90 if marker == "*" else 55,
                       label=name, edgecolors="k" if marker != "x" else None, linewidths=0.5)
    ax.axvline(args.ehull_cutoff, ls="--", c="crimson", lw=1,
               label=f"stability cutoff {args.ehull_cutoff} eV/atom")
    ax.set_xlabel("e above hull (eV/atom)  -- lower = more stable")
    ax.set_ylabel("predicted log10 ionic conductivity (S/cm)")
    ax.set_title("Generated Li-P-S candidates: stability vs conductivity")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "01_landscape.png"), dpi=600)
    plt.close(fig)

    # --- figure 2: final S.U.N. shortlist ---
    sun = df[df["SUN"] & df["pred_log10_sigma"].notna()].head(args.top).iloc[::-1]
    fallback = ""
    if sun.empty:                       # nothing fully S.U.N. -> show stable as fallback
        sun = df[df["stable"] & df["pred_log10_sigma"].notna()].head(args.top).iloc[::-1]
        fallback = " (no S.U.N. hits -- showing stable candidates)"
    if not sun.empty:
        fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(sun) + 1)))
        ypos = range(len(sun))
        ax.barh(list(ypos), sun["pred_log10_sigma"], color="#1F4E79")
        ax.set_yticks(list(ypos))
        ax.set_yticklabels([f"{r.formula}\n(Ehull={r.e_above_hull:.2f})"
                            for r in sun.itertuples()])
        ax.set_xlabel("predicted log10 ionic conductivity (S/cm)")
        ax.set_title(f"Final shortlist{fallback}")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "02_shortlist.png"), dpi=600)
        plt.close(fig)

    # --- print ---
    cols = ["label", "formula", "e_above_hull", "stable", "unique", "novel", "SUN",
            "pred_log10_sigma", "pred_sigma_S_cm"]
    show = df[cols].copy()
    show["pred_log10_sigma"] = show["pred_log10_sigma"].round(2)
    print("=== all candidates (ranked by predicted log10 sigma) ===")
    print(show.to_string(index=False))
    n_sun = int(df["SUN"].sum())
    print(f"\nFinal shortlist: {n_sun} Stable+Unique+Novel candidate(s).")
    print("These are the generate->screen->score outputs to hand to project 2 MD. "
          "Concept-validation only -- not a synthesizability claim.")
    print(f"Saved data/candidates_final.csv + figures/01_landscape.png, 02_shortlist.png")

    # --- audit log (mirrors project 2's data/md_runs.json): run params + S.U.N. counts ---
    manifest_path = os.path.join(DATA, "generated", "manifest.csv")
    n_generated = len(pd.read_csv(manifest_path)) if os.path.exists(manifest_path) else len(df)
    run_meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": {
            "chemsys": args.chemsys, "batch_size": args.batch_size,
            "num_batches": args.num_batches, "guidance": args.guidance,
            "ehull_cutoff": args.ehull_cutoff, "top": args.top,
        },
        "counts": {
            "n_generated": int(n_generated), "n_screened": int(len(df)),
            "n_stable": int(df["stable"].sum()), "n_unique": int(df["unique"].sum()),
            "n_novel": int(df["novel"].sum()), "n_sun": int(df["SUN"].sum()),
            "n_scored": int(df["pred_log10_sigma"].notna().sum()),
        },
        "outputs": ["data/candidates_final.csv", "figures/01_landscape.png", "figures/02_shortlist.png"],
    }
    with open(os.path.join(DATA, "p3_runs.json"), "w") as f:
        json.dump(run_meta, f, indent=2)
    print("Saved data/p3_runs.json (audit log: params + S.U.N. counts).")


if __name__ == "__main__":
    main()
