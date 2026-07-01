"""
Publication-grade composite figure for the Project 3 inverse-design demo.

Two panels (Nature double-column, ~180 mm, editable SVG):
  a  Stability-conductivity landscape: E_above_hull vs predicted log10 sigma for every
     screened candidate, coloured by S.U.N. status; stability cutoff + the screened-out
     unstable candidates shown as a bottom rug -> the generate->screen funnel geometry.
  b  Final S.U.N. shortlist ranked by the project-1 conductivity prior.

Reads data/candidates_final.csv. Backend: matplotlib only. Run:
    python figures/make_publication_figure.py
"""
from __future__ import annotations

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "candidates_final.csv")
OUT = os.path.join(HERE, "fig_inverse_design")

# ---- Nature-style rcParams: editable text, thin spines, 7 pt base ----------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none",   # keep text as <text> nodes (editable in Illustrator/Inkscape)
    "pdf.fonttype": 42,       # editable TrueType in PDF
    "font.size": 7,
    "axes.linewidth": 0.6,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
})

# Restrained palette: one blue signal family (two shades) + neutral grey; red reserved
# for the reference cutoff line only.
C_SUN = "#0F4D92"      # stable + unique + novel  (the hero signal)
C_STABLE = "#8FB2D6"   # stable but not S.U.N.    (lighter blue)
C_UNSTABLE = "#BEBEBE"  # screened out             (neutral grey)
C_CUT = "#B64342"      # stability cutoff          (reference cue)
C_TARGET = "#0F4D92"   # faint "kept" region shade


def fml(s: str) -> str:
    """Chemical formula with subscripted stoichiometry, e.g. Li3PS4 -> $\\mathrm{Li_{3}PS_{4}}$."""
    return "$\\mathrm{" + re.sub(r"(\d+)", r"_{\1}", str(s)) + "}$"


def main():
    d = pd.read_csv(DATA)
    cut = 0.1
    d["cat"] = np.where(d["SUN"], "sun",
                        np.where(d["stable"], "stable", "unstable"))
    scored = d[d["pred_log10_sigma"].notna()]
    sun = scored[scored["cat"] == "sun"]
    stab = scored[scored["cat"] == "stable"]
    unstable = d[d["cat"] == "unstable"]

    fig = plt.figure(figsize=(7.1, 3.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.34,
                          left=0.075, right=0.975, bottom=0.16, top=0.9)

    # ================= Panel a: stability-conductivity landscape (hero) =================
    ax = fig.add_subplot(gs[0, 0])
    y0 = scored["pred_log10_sigma"].min() - 0.30
    y1 = scored["pred_log10_sigma"].max() + 0.25
    # cap the x-axis just past the cutoff so the scored candidates are not squished into the
    # left third; the 11 unstable ones (E_hull up to ~0.34) sit off-panel and are counted.
    xlo, xhi = -0.05, 0.15
    ax.set_xlim(xlo, xhi)

    # faint "kept (stable)" region left of the cutoff
    ax.axvspan(xlo, cut, color=C_TARGET, alpha=0.05, lw=0, zorder=0)
    ax.axvline(cut, color=C_CUT, ls="--", lw=0.9, zorder=1)
    ax.text(cut + 0.004, y1 - 0.10, "stability\ncutoff 0.1", color=C_CUT, fontsize=6,
            va="top", ha="left", linespacing=1.1)

    # S.U.N. filled; stable-not-S.U.N. as open diamonds ON TOP so the 7 stay visible
    ax.scatter(sun["e_above_hull"], sun["pred_log10_sigma"], s=17, c=C_SUN,
               alpha=0.88, edgecolors="white", linewidths=0.3, zorder=3,
               label=f"S.U.N. ({len(sun)})")
    ax.scatter(stab["e_above_hull"], stab["pred_log10_sigma"], s=24, marker="D",
               facecolors="none", edgecolors="#3F5E86", linewidths=0.9, zorder=4,
               label=f"stable, not S.U.N. ({len(stab)})")

    ax.text(cut + 0.004, y0 + 0.28, f"{len(unstable)} unstable →\nscreened out",
            color="#8A8A8A", fontsize=6, ha="left", va="bottom", linespacing=1.2)

    # annotate the hero candidate only (LiPS3 below-hull is shown in panel b)
    r0 = sun[(sun["formula"] == "Li3PS4") & (np.isclose(sun["e_above_hull"], 0.006, atol=1e-3))]
    if len(r0):
        x0, yv0 = r0["e_above_hull"].iloc[0], r0["pred_log10_sigma"].iloc[0]
        # highlight ring: a stable-not-S.U.N. Li3PS4 polymorph sits at nearly the same point,
        # so ring the S.U.N. hero to make the callout unambiguous.
        ax.scatter([x0], [yv0], s=44, facecolors=C_SUN, edgecolors="#E0A100",
                   linewidths=1.3, zorder=6)
        ax.annotate(fml("Li3PS4") + " novel β-polymorph", xy=(x0, yv0),
                    xytext=(-0.045, y1 - 0.10), fontsize=6, ha="left", va="center",
                    color="#272727",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#5A5A5A",
                                    shrinkA=0, shrinkB=4))

    ax.set_ylim(y0, y1)
    ax.set_xlabel("Energy above hull, $E_{\\mathrm{hull}}$ (eV atom$^{-1}$)")
    ax.set_ylabel("Predicted log$_{10}\\,\\sigma$ (S cm$^{-1}$)")
    ax.legend(loc="lower left", fontsize=6, handletextpad=0.3, borderpad=0.3,
              labelspacing=0.4)
    ax.text(0.0, 1.02, "a", transform=ax.transAxes, fontsize=9, fontweight="bold",
            ha="left", va="bottom")
    # funnel summary
    ax.text(0.5, 1.02, "64 generated → 61 Li-bearing → 50 stable → 43 S.U.N.",
            transform=ax.transAxes, fontsize=6, color="#5A5A5A", ha="center", va="bottom")

    # ================= Panel b: S.U.N. shortlist ranked by predicted sigma =================
    axb = fig.add_subplot(gs[0, 1])
    top = sun.sort_values("pred_log10_sigma", ascending=False).head(8).iloc[::-1]
    ypos = np.arange(len(top))
    base = top["pred_log10_sigma"].min() - 0.45
    widths = top["pred_log10_sigma"].values - base
    axb.barh(ypos, widths, left=base, height=0.66, color=C_SUN,
             edgecolor="white", linewidth=0.4, zorder=3)
    axb.set_xlim(base, top["pred_log10_sigma"].max() + 0.06)
    axb.set_ylim(-0.6, len(top) - 0.4)

    labels = [f"{fml(r.formula)}\n$E_{{\\mathrm{{hull}}}}$ {r.e_above_hull:.3f}"
              for r in top.itertuples()]
    axb.set_yticks(ypos)
    axb.set_yticklabels(labels, fontsize=6, linespacing=1.15)
    axb.set_xlabel("Predicted log$_{10}\\,\\sigma$ (S cm$^{-1}$)")
    axb.tick_params(axis="y", length=0)
    axb.text(0.0, 1.02, "b", transform=axb.transAxes, fontsize=9, fontweight="bold",
             ha="left", va="bottom")
    axb.text(1.0, 1.02, "top-8 S.U.N.", transform=axb.transAxes, fontsize=6,
             color="#5A5A5A", ha="right", va="bottom")

    fig.savefig(f"{OUT}.svg", bbox_inches="tight")   # vector, editable text
    plt.close(fig)
    print("wrote", OUT + ".svg")


if __name__ == "__main__":
    main()
