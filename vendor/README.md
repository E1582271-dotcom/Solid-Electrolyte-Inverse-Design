# vendor/ — bundled Project 1 conductivity ensemble

A **vendored, checksum-verified copy** of Project 1's production conductivity ranker
([Solid-Electrolyte-Screening](https://github.com/E1582271-dotcom/Solid-Electrolyte-Screening),
`project1_screening/`), so this repo clones and scores candidates on its own. `src/score.py`
always loads this copy; it never reads a sibling Project 1 checkout.

## Source

| field | value |
|---|---|
| Project 1 version | `p1screen` 0.1.0 |
| Project 1 commit at refresh | `099731e0d00d836fed263b5f020d7c433adce338` |
| commit that last changed the models / `data/model_manifest.json` | `de72aa90bd8ad070a761e0d1a55f285e34a96b5d` |
| Project 1 release manifest checked against | `data/release_manifest.json`, `verified_utc` 2026-10-14T15:57:48Z |
| refreshed | 2026-10-19 (exact time in `p1_manifest.json`) |

All 15 member sha256 values equal both Project 1's `data/model_manifest.json` and its
`data/release_manifest.json`; the loader re-checks every file's sha256 before scoring and
refuses on any mismatch.

## What is copied

- `p1_models/*.cbm` — the 15 CatBoost members (`compact` 400 trees/depth 5, `base` 600/6,
  `deep_slow` 800/7; seeds 0–4 each), trained by Project 1 on **all 599 OBELiX records**
  (target log₁₀ σ / S cm⁻¹). Verbatim copies of `project1_screening/artifacts/models/`.
- `p1_manifest.json` — provenance, per-member sha256/bytes/config/seed, the 134-feature contract
  (132 matminer Magpie statistics + `n_elements` + `Li_frac`), applicability-domain settings and
  Project 1's frozen route thresholds.
- `p1_conductivity.py` — the production scoring path, copied from Project 1's
  `src/p1screen/` (`composition.py` formula normalisation + `composition_key`;
  `features.composition_features`/`formula_matrix`; `modeling.load_ensemble`/`ensemble_predictions`;
  `domain.ApplicabilityDomain`; the OBELiX catalogue match from `screening._disposition`).
- `p1_obelix_formulas.csv` — the 599 OBELiX training formulas (record ID, split, nominal
  `Reduced Composition`, and the feature formula: `True Composition`, falling back to
  `Reduced Composition`, exactly as Project 1 resolves it). No conductivity values. Used to refit
  the applicability domain and to flag compositions already measured in OBELiX.
- `refresh_from_p1.py` — rebuilds everything above from a Project 1 checkout.

## What the scorer reports (per candidate formula)

| column | meaning |
|---|---|
| `pred_log10_sigma` | **ranking score** = mean of the 15 members' predicted log₁₀ σ |
| `pred_log10_sigma_spread` | standard deviation across the 15 members (seed + configuration variation) |
| `ad_distance` | Project 1 applicability-domain distance (mean distance to the 5 nearest OBELiX records in standardised-PCA descriptor space) |
| `obelix_composition_match` | composition (nominal or feature formula) is in OBELiX, so the score largely echoes measured values of that composition |
| `p1_extrapolative` | spread or AD at/above Project 1's frozen 75th-percentile route thresholds (0.385 / 12.15) |

Formula-only by design: Project 1's METHODS states that the raw-lattice models (M1/M2) never
score catalogue formulas, so structure, space group and polymorph do not enter and every
polymorph of one composition gets the same score. Parity: on Project 1's own 62-composition
queue this copy reproduces `ranking_score`, `ensemble_spread` and `ad_distance` to ≤ 2×10⁻¹⁵
(`tests/test_vendor_p1.py` keeps a frozen subset of those values).

The output is a **coarse ranking prior**, not a quantitative σ.

## How to refresh

```bash
python vendor/refresh_from_p1.py --p1 ../Solid-Electrolyte-Screening   # needs pymatgen
python -m pytest tests                                          # parity + checksum tests
python 03_score_conductivity.py --stable-only && python 04_rank_candidates.py --top 8 ...
python figures/make_publication_figure.py && python figures/make_generation_funnel_figure.py
```

The refresh aborts without writing anything if Project 1's production representation is not the
formula-only `M0_formula`, if any member checksum disagrees between disk, the model manifest and
the release manifest, or if the OBELiX record count differs from the model manifest. If Project 1
retrains, update the frozen parity values in `tests/test_vendor_p1.py` from its new
`candidate_queue.csv` and update the table above.

## Licences

The model files and code are derived from Project 1 (MIT). `p1_obelix_formulas.csv` is derived
from **OBELiX** and is redistributed under its **CC BY 4.0** terms, not this repository's MIT
licence: F. Therrien et al., "OBELiX: a curated dataset of experimentally measured ionic
conductivities," *Digital Discovery* (2026), doi:10.1039/D5DD00441A;
upstream <https://github.com/NRC-Mila/OBELiX>.
