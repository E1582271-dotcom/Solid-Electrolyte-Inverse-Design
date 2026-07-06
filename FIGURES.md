# Figures — Generative Inverse Design (Project 3)

Publication figure set with legends + statistics/provenance, in the project's logical order. The
repo tracks **PNG (600 dpi) + SVG** (editable text, `svg.fonttype='none'`); underlying numbers are
in [`source_data/`](source_data/). Styling is centralised in
[`src/plotstyle.py`](src/plotstyle.py) — the same Nature-family palette/sizing as Projects 1 & 2
(`apply_publication_style()`, `finalize_figure()`, `save_source_data()`), plus this project's own
S.U.N.-status palette (`C_SUN`/`C_STABLE`/`C_UNSTABLE`/`C_CUT`).

## The story (why each figure exists)

What does raw MatterGen output actually look like before any filtering? (Fig. 1, generate) →
after self-consistent-MLIP-hull screening and Project-1 conductivity scoring, where do the
survivors land, and which ones make the final cut? (Fig. 2, screen + score + output).

---

**Fig. 1 | Generation-step funnel.** `figures/fig_generation_funnel.*` — *[generate]*
Distribution of `e_above_hull` for all 61 Li-bearing MatterGen candidates (of 64 raw generations;
3 Li-free P–S binaries dropped before this point, per README), stacked by final status: unstable
(screened out) / stable-not-S.U.N. / S.U.N., with the 0.1 eV/atom stability cutoff marked (same
value as Fig. 2). Funnel counts annotated: 64 generated → 61 Li-bearing → 50 stable → 43 S.U.N.
*Stats:* n=61 scored candidates (deterministic MLIP relaxation, no repeats). *Source:*
`source_data/fig_generation_funnel.csv`.

**Fig. 2 | Stability–conductivity landscape and S.U.N. shortlist.** `figures/fig_inverse_design.*`
— *[screen + score + output]*
(**a**) `e_above_hull` vs the Project-1 CatBoost conductivity prior (`log₁₀σ`) for every scored
candidate, coloured by S.U.N. status; stability cutoff (0.1 eV/atom) and the screened-out unstable
tail shown as a bottom rug. The hero candidate — a novel β-Li₃PS₄ polymorph, e_above_hull≈0.006 —
is ringed. (**b**) Final S.U.N. shortlist (top 8), ranked by the Project-1 prior. *Stats:*
`e_above_hull` from a **self-consistent** MLIP hull (candidates and the 96 MP reference phases
relaxed with the same un-fine-tuned MACE-MP-0) — a relative-stability indicator, not DFT-
quantitative; `log₁₀σ` is a coarse ranking prior (Project-1 model), not a quantitative σ — see
[README § Known limitations](README.md#known-limitations-disclosed). *Source:*
`source_data/fig_inverse_design.csv` (all scored candidates; a `shortlist_rank` column marks the
top-8 S.U.N. subset plotted in panel b).
