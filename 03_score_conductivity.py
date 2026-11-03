"""
Project 3 -- step 3: score screened candidates with Project 1's conductivity ranker.

Applies Project 1's production model (vendored in vendor/: 15-member CatBoost ensemble on 134
formula-only Magpie descriptors, trained on all 599 OBELiX records) to each candidate's relaxed
reduced formula, giving a coarse log10(ionic conductivity) prior. Project 3 is an independent
generative track: it does not consume Project 1's screening output and only borrows its trained
model as an off-the-shelf scorer before the Project 2 MD hand-off.

Columns written: pred_log10_sigma (ensemble mean; the ranking score), pred_sigma_S_cm,
pred_log10_sigma_spread (ensemble SD), ad_distance (Project 1 applicability-domain distance),
obelix_composition_match (composition measured in OBELiX), p1_extrapolative (spread or AD at or
above Project 1's frozen route thresholds).

Honest, same as Project 1: a RANK, not a quantitative sigma. The descriptors are formula-only,
so every polymorph of a composition gets the same score (step 4 breaks ties by e_above_hull).

Reads data/screened.csv (formula column; no CIFs needed); writes data/scored.csv.

Run:
    python 03_score_conductivity.py [--stable-only]
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from src import score as SC

DATA = os.path.join(HERE, "data")
KEEP = ["label", "formula"] + SC.SCORE_COLUMNS


def main():
    ap = argparse.ArgumentParser(description="Score candidates with the Project 1 ensemble.")
    ap.add_argument("--stable-only", action="store_true",
                    help="only score candidates flagged stable in the screen")
    args = ap.parse_args()

    screened = pd.read_csv(os.path.join(DATA, "screened.csv"))
    if args.stable_only:
        screened = screened[screened["stable"]].copy()
    out_path = os.path.join(DATA, "scored.csv")
    if screened.empty:
        # still (over)write an empty scored.csv so step 4 never merges a stale one
        pd.DataFrame(columns=KEEP).to_csv(out_path, index=False)
        print("No stable candidates to score; wrote empty scored.csv "
              "(loosen 02's --ehull-cutoff, or run 03 without --stable-only).")
        return

    print(f"[score] scoring {len(screened)} candidates with the vendored Project 1 ensemble ...")
    out = SC.score_formulas(screened["formula"].tolist(), labels=screened["label"].tolist())[KEEP]
    out.to_csv(out_path, index=False)

    show = out[["label", "formula", "pred_log10_sigma", "pred_log10_sigma_spread",
                "ad_distance", "obelix_composition_match", "p1_extrapolative"]].copy()
    for c in ("pred_log10_sigma", "pred_log10_sigma_spread", "ad_distance"):
        show[c] = show[c].round(2)
    print("\n=== candidates ranked by predicted log10 sigma (ensemble mean) ===")
    print(show.to_string(index=False))
    print("\nReminder: RANK, not quantitative sigma; formula-only, so polymorphs tie. "
          "Validate the head with Project 2 MD.")
    print(f"Saved {os.path.relpath(out_path, HERE)}")


if __name__ == "__main__":
    main()
