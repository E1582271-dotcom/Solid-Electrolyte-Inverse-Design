# Results

Production run on NUS Vanda (A40), 2026-07-01; conductivity prior rescored 2026-10-19. Full table:
[data/candidates_final.csv](../data/candidates_final.csv); counts:
[data/p3_runs.json](../data/p3_runs.json).

## Generation and stability screen

**64 generated → 61 Li-bearing → 50 stable → 43 S.U.N.** (`ehull_cutoff` = 0.1 eV/atom). MatterGen
conditioned on Li-P-S (batch 16 × 4 = 64); the Li filter dropped 3 Li-free P–S binaries, keeping 61;
MACE-MP-0 self-consistent hull screen against 96 reference phases. Uniqueness and novelty are
independent flags over the 61-candidate pool (54/61 unique; **61/61 novel**: every survivor is a new
polymorph relative to the known MP phases), so the 43 S.U.N. structures are the stable ∧ unique ∧
novel intersection. Figures: [figures.md](figures.md).

## Conductivity prior

Scored with Project 1's current 15-member ensemble (vendored; rescored 2026-10-19). The 50 stable
candidates cover 26 distinct compositions and span 7.0 log units (−2.78 to −9.81 log₁₀ S/cm). These
scores replace those of the retired July 2026 single-CatBoost model; the two rankings are
essentially unrelated (Spearman ρ = −0.19 over the 50 scored candidates), so score-based statements
made on the old scale are withdrawn. The new prior largely rewards Li fraction (Spearman ρ = 0.93
against Li atomic fraction over the 26 compositions; `Li_frac` is itself one of the 134 features),
and none of the 50 is flagged extrapolative at Project 1's frozen q75 route thresholds (ensemble
spread 0.385, AD distance 12.15).

Top S.U.N. structures by the prior. This is **descriptive only, not a shortlist**; all 43 S.U.N.
structures are the next batch for the Solid-Electrolyte-MLIP-MD census (not yet run).

| S.U.N. rank | Candidate | e_above_hull (eV/atom) | Predicted log₁₀σ ± ensemble SD (S/cm) | AD distance | Note |
|---|---|---|---|---|---|
| 1 | Li8PS2 (gen_040) | 0.019 | −2.78 ± 0.33 | 5.41 | Li excess: decomposes on the MLIP hull to Li₃P + Li₂S + Li metal; not charge-balanced |
| 2 | Li6PS (gen_027) | 0.035 | −3.78 ± 0.26 | 5.48 | Li excess: decomposes to Li₃P + Li₂S + Li metal; not charge-balanced |
| 3 | Li7PS2 (gen_038) | 0.049 | −3.83 ± 0.28 | 5.29 | Li₃P + 2 Li₂S stoichiometry; balances only with P³⁻ (a phosphide-sulfide, not a thiophosphate) |
| 4 | Li4PS5 (gen_006) | 0.080 | −4.28 ± 0.18 | 0.85 | decomposes to Li₂S + Li₃PS₄ + S |
| 5 | Li9P2S9 (gen_035) | 0.046 | −4.79 ± 0.20 | 4.74 | decomposes to Li₂S + Li₃PS₄ + P |
| 6 | Li9P2S5 (gen_056) | 0.097 | −5.01 ± 0.28 | 4.89 | decomposes to Li₂S + Li₃PS₄ + P |
| 7 | Li5PS4 (gen_049) | 0.087 | −5.14 ± 0.29 | 4.78 | decomposes to Li₂S + Li₃PS₄ + P |
| 8 | Li7PS7 (gen_043) | 0.074 | −5.47 ± 0.38 | 4.74 | decomposes to Li₂S + Li₃PS₄ + S |
| … | | | | | |
| 13 | Li3PS4 (gen_016) | 0.006 | −6.05 ± 0.15 | 0.45 | β-Li₃PS₄ is a known fast-ion conductor and this is a generated novel polymorph, essentially on the hull; Li₃PS₄ is in OBELiX, so its score largely echoes measured values; the three S.U.N. Li₃PS₄ polymorphs tie at #12–14 |
| 37 | LiPS3 (gen_021) | −0.031 | −9.81 ± 0.21 | 3.03 | `e_above_hull` < 0: an MLIP-predicted new ground state; LiPS₃ is in OBELiX; its seven S.U.N. polymorphs tie at #37–43 (last) |

Of the top eight, only Li₇PS₂ charge-balances with Li⁺, S²⁻ and P³⁻ or P⁵⁺. The head of the prior is
Li-rich and S-poor, so the score describes the candidates but selects none.

## Historical pilot (July 2026)

Two structures went to single-seed MACE-MP-0 MD in the July 2026 four-lead pilot of
Solid-Electrolyte-MLIP-MD (two leads from each track; the pilot predates that repository's census):
Li₃PS₄ gen_016 (e_above_hull 0.006, a novel β-polymorph essentially on the hull;
5.5×10⁻³ mS/cm) and LiPS₃ gen_021 (e_above_hull −0.031, an MLIP-predicted new ground state;
~10 mS/cm, the second highest of the four). On Project 1's current ensemble the four pilot leads
score Li₂₀Si₃P₃S₂₃Cl −2.26 and Li₈TiS₆ −6.63 (Project 1's own scores), Li₃PS₄ gen_016 −6.05 and
LiPS₃ gen_021 −9.81. They span 7.6 log units, against ~4.5 orders of magnitude in the single-seed
pilot MD (29 to 9×10⁻⁴ mS/cm). Both put Li₂₀Si₃P₃S₂₃Cl first and gen_016 above Li₈TiS₆, but the
prior puts LiPS₃ last where the pilot MD put it second. The pilot is historical and single-seed, so
this comparison validates or falsifies neither the prior nor any of the four materials.

## Status

- **Local CPU plumbing check** (`mp-demo`, 6 real Li-P-S phases + `lj` hull): generate → screen
  (`e_above_hull` + de-duplication + novelty, 96 MP reference phases including the Li/P/S
  endpoints) → score → rank + 2 figures, all wired. `lj` energies are non-physical and `mp-demo`
  fixtures are known phases, not generative output, so `novel=False` for all of them, which
  confirms the novelty test. At the time, scoring bridged into a sibling Project 1 checkout with the
  cross-project `src` package-name clash isolated; since 2026-10-19 the ensemble is vendored under
  `vendor/` and loaded under its own module name, so no clash arises.
- **Production run** (NUS Vanda A40, 2026-07-01): MatterGen generation → Li filter → MACE-MP-0
  self-consistent hull screen → Project 1 CatBoost scoring → ranked figures. PBS + Singularity,
  reusing the `~/macepkg` environment of Solid-Electrolyte-MLIP-MD, at no cost
  ([hpc_vanda.md](hpc_vanda.md)).
- **Rescore** (2026-10-19): the 50 stable candidates rescored with Project 1's current vendored
  15-member ensemble; `data/candidates_final.csv`, `data/p3_runs.json`, the figures and
  `source_data/` were regenerated. The retired July 2026 single-model scores are superseded.
- **Next:** all 43 S.U.N. structures through the Solid-Electrolyte-MLIP-MD census and replicate
  protocol (not yet run).
