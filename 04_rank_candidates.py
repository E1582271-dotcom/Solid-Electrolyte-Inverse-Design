"""
Project 3 -- step 4: combine the screen + conductivity score into a final shortlist.

Joins data/screened.csv (e_above_hull, S.U.N. flags, the ehull_cutoff 02 used) with
data/scored.csv (Project 1 ensemble score + spread) and produces:
  - data/candidates_final.csv : every candidate, all columns, ranked by predicted sigma, with
                                score_rank (among scored) and sun_rank (among scored S.U.N.).
  - figures/01_landscape.png  : quick-look e_above_hull vs predicted log10 sigma.
  - figures/02_shortlist.png  : quick-look S.U.N. shortlist (bars = ensemble mean, whiskers =
                                ensemble SD). Untracked; the tracked figures are built by
                                figures/make_*.py from candidates_final.csv + p3_runs.json.
  - data/p3_runs.json         : run log (params, funnel counts with their sources, scorer
                                provenance); earlier runs are kept under "previous_runs".

Ranking: ensemble mean descending. The scorer is formula-only, so polymorphs of one composition
tie; ties are broken by e_above_hull (more stable first), then label.

The shortlist = Stable (e_above_hull <= 02's cutoff) AND Unique AND Novel, then ranked by the
Project 1 conductivity prior -- a concept-validation hand-off to Project 2 MD, NOT a
synthesizability claim.

Run:
    python 04_rank_candidates.py --top 8 --chemsys Li-P-S --batch-size 16 --num-batches 4 --guidance 2.0
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
sys.path.insert(0, HERE)
from src import score as SC

DATA = os.path.join(HERE, "data")
FIG = os.path.join(HERE, "figures")
GEN_DIR = os.path.join(DATA, "generated")


def screen_cutoff(screened: pd.DataFrame) -> float:
    """The e_above_hull cutoff 02_screen_stability.py applied (its ehull_cutoff column)."""
    if "ehull_cutoff" not in screened.columns:
        raise SystemExit("data/screened.csv has no ehull_cutoff column -- re-run "
                         "02_screen_stability.py (it records the cutoff it applied).")
    values = screened["ehull_cutoff"].dropna().unique()
    if len(values) != 1:
        raise SystemExit(f"data/screened.csv carries {len(values)} different ehull_cutoff values")
    cut = float(values[0])
    if not (screened["stable"].astype(bool) == (screened["e_above_hull"] <= cut)).all():
        raise SystemExit(f"stable flags in data/screened.csv disagree with ehull_cutoff={cut}")
    return cut


def funnel_counts(screened: pd.DataFrame, args) -> tuple[dict, dict]:
    """n_generated (raw generator output, before the Li filter) and n_li_bearing (after it),
    each from the most direct record available, with the source written alongside."""
    counts, sources = {}, {}
    summary_path = os.path.join(GEN_DIR, "generation_summary.json")
    manifest_path = os.path.join(GEN_DIR, "manifest.csv")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else None
    if summary is not None:
        counts["n_generated"] = int(summary["n_generated"])
        sources["n_generated"] = "data/generated/generation_summary.json"
    elif args.batch_size is not None and args.num_batches is not None:
        counts["n_generated"] = int(args.batch_size) * int(args.num_batches)
        sources["n_generated"] = ("batch_size x num_batches (MatterGen emits batch_size structures "
                                  "per batch; generation_summary.json absent)")
    else:
        counts["n_generated"] = None
        sources["n_generated"] = "unknown (no generation_summary.json, no --batch-size/--num-batches)"
    if os.path.exists(manifest_path):
        counts["n_li_bearing"] = int(len(pd.read_csv(manifest_path)))
        sources["n_li_bearing"] = "data/generated/manifest.csv (post element filter)"
    elif summary is not None:
        counts["n_li_bearing"] = int(summary["n_kept"])
        sources["n_li_bearing"] = "data/generated/generation_summary.json n_kept"
    else:
        counts["n_li_bearing"] = int(len(screened))
        sources["n_li_bearing"] = "rows of data/screened.csv (02 screens every post-filter candidate)"
    return counts, sources


def main():
    ap = argparse.ArgumentParser(description="Rank + plot the final candidate shortlist.")
    ap.add_argument("--top", type=int, default=5, help="size of the printed/plotted shortlist")
    # passthrough run params -- only for the p3_runs.json audit log (mirrors project 2's md_runs.json)
    ap.add_argument("--chemsys", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-batches", type=int, default=None)
    ap.add_argument("--guidance", type=float, default=None)
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)

    screened = pd.read_csv(os.path.join(DATA, "screened.csv"))
    cut = screen_cutoff(screened)
    scored_path = os.path.join(DATA, "scored.csv")
    score_cols = ["label"] + SC.SCORE_COLUMNS
    scored = (pd.read_csv(scored_path) if os.path.exists(scored_path)
              else pd.DataFrame(columns=score_cols))
    for c in score_cols:                       # tolerate an empty scored.csv
        if c not in scored.columns:
            scored[c] = pd.Series(dtype="float64")
    df = screened.merge(scored[score_cols], on="label", how="left", validate="one_to_one")
    df = df.sort_values(["pred_log10_sigma", "e_above_hull", "label"],
                        ascending=[False, True, True], na_position="last",
                        kind="stable").reset_index(drop=True)
    has_score = df["pred_log10_sigma"].notna()
    df["score_rank"] = pd.Series(np.where(has_score, has_score.cumsum(), np.nan)).astype("Int64")
    sun_scored = df["SUN"].astype(bool) & has_score
    df["sun_rank"] = pd.Series(np.where(sun_scored, sun_scored.cumsum(), np.nan)).astype("Int64")
    df.to_csv(os.path.join(DATA, "candidates_final.csv"), index=False)

    # --- figure 1: stability vs conductivity landscape (quick look) ---
    fig, ax = plt.subplots(figsize=(6.4, 5))
    sub = df[has_score]
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
    ax.axvline(cut, ls="--", c="crimson", lw=1, label=f"stability cutoff {cut} eV/atom (from 02)")
    ax.set_xlabel("e above hull (eV/atom)  -- lower = more stable")
    ax.set_ylabel("predicted log10 ionic conductivity (S/cm), P1 ensemble mean")
    ax.set_title("Generated Li-P-S candidates: stability vs conductivity")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "01_landscape.png"), dpi=600)
    plt.close(fig)

    # --- figure 2: final S.U.N. shortlist (quick look) ---
    sun = df[sun_scored].head(args.top).iloc[::-1]
    fallback = ""
    if sun.empty:                       # nothing fully S.U.N. -> show stable as fallback
        sun = df[df["stable"] & has_score].head(args.top).iloc[::-1]
        fallback = " (no S.U.N. hits -- showing stable candidates)"
    if not sun.empty:
        fig, ax = plt.subplots(figsize=(7, max(3, 0.5 * len(sun) + 1)))
        ypos = range(len(sun))
        ax.barh(list(ypos), sun["pred_log10_sigma"], xerr=sun["pred_log10_sigma_spread"],
                color="#1F4E79", ecolor="#444444", capsize=2)
        ax.set_yticks(list(ypos))
        ax.set_yticklabels([f"{r.formula} {r.label}\n(Ehull={r.e_above_hull:.3f})"
                            for r in sun.itertuples()])
        ax.set_xlabel("predicted log10 ionic conductivity (S/cm); whiskers = ensemble SD")
        ax.set_title(f"Final shortlist{fallback}")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "02_shortlist.png"), dpi=600)
        plt.close(fig)

    # --- print ---
    cols = ["label", "formula", "e_above_hull", "stable", "unique", "novel", "SUN",
            "pred_log10_sigma", "pred_log10_sigma_spread", "score_rank", "sun_rank"]
    show = df[cols].copy()
    show["pred_log10_sigma"] = show["pred_log10_sigma"].round(2)
    show["pred_log10_sigma_spread"] = show["pred_log10_sigma_spread"].round(2)
    print("=== all candidates (ranked by predicted log10 sigma; ties by e_above_hull) ===")
    print(show.to_string(index=False))
    n_sun = int(df["SUN"].sum())
    print(f"\nFinal shortlist: {n_sun} Stable+Unique+Novel candidate(s) (cutoff {cut} eV/atom, from 02).")
    print("These are the generate->screen->score outputs to hand to project 2 MD. "
          "Concept-validation only -- not a synthesizability claim.")
    print("Saved data/candidates_final.csv + figures/01_landscape.png, 02_shortlist.png")

    # --- audit log (mirrors project 2's data/md_runs.json): run params + S.U.N. counts ---
    counts, sources = funnel_counts(screened, args)
    counts.update({
        "n_screened": int(len(df)),
        "n_stable": int(df["stable"].sum()), "n_unique": int(df["unique"].sum()),
        "n_novel": int(df["novel"].sum()), "n_sun": n_sun,
        "n_scored": int(has_score.sum()),
    })
    sources.update({"ehull_cutoff": "ehull_cutoff column of data/screened.csv (written by 02)"})
    run_meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": {
            "chemsys": args.chemsys, "batch_size": args.batch_size,
            "num_batches": args.num_batches, "guidance": args.guidance,
            "ehull_cutoff": cut, "top": args.top,
        },
        "counts": counts,
        "count_sources": sources,
        "scorer": SC.provenance(),
        "ranking": "pred_log10_sigma (ensemble mean) desc; ties by e_above_hull asc, then label",
        "outputs": ["data/candidates_final.csv", "figures/01_landscape.png",
                    "figures/02_shortlist.png"],
    }
    log_path = os.path.join(DATA, "p3_runs.json")
    if os.path.exists(log_path):
        with open(log_path) as f:
            prev = json.load(f)
        history = prev.pop("previous_runs", [])
        run_meta["previous_runs"] = history + [prev]
    with open(log_path, "w") as f:
        json.dump(run_meta, f, indent=2)
    print("Saved data/p3_runs.json (audit log: params + funnel counts + scorer provenance).")


if __name__ == "__main__":
    main()
