"""
Generation-step funnel figure for Project 3 (fills the "generate" visualization gap:
fig_inverse_design.png already covers screen+output, nothing previously visualized the
raw MatterGen output before filtering).

Reads data/candidates_final.csv (one row per post-Li-filter candidate) + data/p3_runs.json
and renders a single-panel e_above_hull distribution histogram, stacked/coloured by S.U.N.
status, with the stability cutoff 02 applied (p3_runs.json params.ehull_cutoff, the same value
fig_inverse_design uses) marked and the funnel annotated from p3_runs.json counts:
n_generated (raw MatterGen output, before the Li filter) -> n_li_bearing -> n_stable -> n_sun.
The per-category counts are cross-checked against candidates_final.csv.

Run:
    python figures/make_generation_funnel_figure.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
from src import plotstyle as ps  # noqa: E402
from src.plotstyle import C_SUN, C_STABLE, C_UNSTABLE, C_CUT  # noqa: E402

DATA = os.path.join(REPO_ROOT, "data", "candidates_final.csv")
RUNS = os.path.join(REPO_ROOT, "data", "p3_runs.json")
OUT = os.path.join(HERE, "fig_generation_funnel")


def load_run(d: pd.DataFrame) -> tuple[dict, float]:
    """Funnel counts + stability cutoff from data/p3_runs.json (written by 04 from 01/02's
    records), cross-checked against the candidates table."""
    with open(RUNS) as f:
        run = json.load(f)
    counts, cut = run["counts"], float(run["params"]["ehull_cutoff"])
    checks = {"n_li_bearing": len(d), "n_stable": int(d["stable"].sum()),
              "n_sun": int(d["SUN"].sum())}
    for key, value in checks.items():
        if int(counts[key]) != value:
            raise SystemExit(f"p3_runs.json {key}={counts[key]} but candidates_final.csv has {value}")
    if not (d["stable"] == (d["e_above_hull"] <= cut)).all():
        raise SystemExit(f"stable flags disagree with p3_runs.json ehull_cutoff={cut}")
    if counts.get("n_generated") is None:
        raise SystemExit("p3_runs.json has no n_generated -- re-run 04 with generation records")
    return counts, cut


def main():
    ps.apply_publication_style()
    d = pd.read_csv(DATA)
    counts, cut = load_run(d)
    d["cat"] = np.where(d["SUN"], "sun", np.where(d["stable"], "stable", "unstable"))

    fig, ax = plt.subplots(figsize=(ps.COL_SINGLE_IN, 3.0))
    bins = np.linspace(d["e_above_hull"].min() - 0.01, d["e_above_hull"].max() + 0.01, 22)
    cats = ["unstable", "stable", "sun"]
    colors_ = [C_UNSTABLE, C_STABLE, C_SUN]
    counts_ = {c: int((d["cat"] == c).sum()) for c in cats}
    labels_ = [f"unstable ({counts_['unstable']})",
              f"stable, not S.U.N. ({counts_['stable']})", f"S.U.N. ({counts_['sun']})"]
    ax.hist([d.loc[d["cat"] == c, "e_above_hull"] for c in cats], bins=bins,
           stacked=True, color=colors_, edgecolor="white", linewidth=0.4,
           label=labels_, zorder=2)

    ax.axvline(cut, color=C_CUT, ls="--", lw=0.9, zorder=4)
    ax.text(cut + 0.004, ax.get_ylim()[1] * 0.96, f"stability\ncutoff {cut:g}", color=C_CUT,
           fontsize=ps.FS_ANNOT, va="top", ha="left", linespacing=1.1)

    ax.set_xlabel("Energy above hull, $E_{\\mathrm{hull}}$ (eV atom$^{-1}$)")
    ax.set_ylabel("Candidates")
    ax.legend(loc="upper right", fontsize=ps.FS_LEGEND, handletextpad=0.4, borderpad=0.3,
             labelspacing=0.35)
    ax.text(0.5, 1.1, f"{counts['n_generated']} generated → {counts['n_li_bearing']} Li-bearing → "
           f"{counts['n_stable']} stable → {counts['n_sun']} S.U.N.",
           transform=ax.transAxes, fontsize=ps.FS_ANNOT, color=ps.PALETTE["neutral_dark"],
           ha="center", va="bottom")

    rows = [[r.label, r.formula, r.e_above_hull, r.cat, r.stable, r.SUN]
           for r in d.itertuples()]
    ps.save_source_data(f"{OUT}.png", ["label", "formula", "e_above_hull", "category",
                                        "stable", "SUN"], rows)
    ps.finalize_figure(fig, f"{OUT}.png", formats=("png", "svg"), pad=1.2)
    print(f"wrote {OUT}.{{png,svg}}, source_data/fig_generation_funnel.csv")


if __name__ == "__main__":
    main()
