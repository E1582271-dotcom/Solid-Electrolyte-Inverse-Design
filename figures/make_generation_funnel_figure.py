"""
Generation-step funnel figure for Project 3 (fills the "generate" visualization gap:
fig_inverse_design.png already covers screen+output, nothing previously visualized the
raw MatterGen output before filtering).

Reads data/candidates_final.csv (61 rows, post Li-filter) + data/p3_runs.json (counts dict)
and renders a single-panel e_above_hull distribution histogram, stacked/coloured by S.U.N.
status, with the stability cutoff (0.1 eV/atom, matching fig_inverse_design.py) marked and
the full funnel count annotated: 64 generated -> 61 Li-bearing -> 50 stable -> 43 S.U.N.
(the "64" and the 3 dropped Li-free P-S binaries are not in candidates_final.csv -- they are
recorded only in README.md's pipeline description, cited here as a literal, commented count).

Run:
    python figures/make_generation_funnel_figure.py
"""
from __future__ import annotations

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
OUT = os.path.join(HERE, "fig_generation_funnel")
CUT = 0.1  # stability cutoff, eV/atom -- same value as make_publication_figure.py

# README.md pipeline description ("Li-P-S (batch 16x4=64) -> Li filter (dropped 3
# Li-free P-S binaries, keeping 61)"): not present in candidates_final.csv, which
# already starts post-filter at 61 rows.
N_RAW_GENERATED = 64
N_DROPPED_LI_FREE = 3


def main():
    ps.apply_publication_style()
    d = pd.read_csv(DATA)
    d["cat"] = np.where(d["SUN"], "sun", np.where(d["stable"], "stable", "unstable"))
    n_li_bearing = len(d)
    n_stable = int(d["stable"].sum())
    n_sun = int(d["SUN"].sum())

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

    ax.axvline(CUT, color=C_CUT, ls="--", lw=0.9, zorder=4)
    ax.text(CUT + 0.004, ax.get_ylim()[1] * 0.96, "stability\ncutoff 0.1", color=C_CUT,
           fontsize=ps.FS_ANNOT, va="top", ha="left", linespacing=1.1)

    ax.set_xlabel("Energy above hull, $E_{\\mathrm{hull}}$ (eV atom$^{-1}$)")
    ax.set_ylabel("Candidates")
    ax.legend(loc="upper right", fontsize=ps.FS_LEGEND, handletextpad=0.4, borderpad=0.3,
             labelspacing=0.35)
    ax.text(0.5, 1.1, f"{N_RAW_GENERATED} generated → {n_li_bearing} Li-bearing → "
           f"{n_stable} stable → {n_sun} S.U.N.",
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
