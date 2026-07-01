# vendor/ — bundled Project 1 conductivity model

These files are a **vendored copy** of Project 1 (OBELiX ionic-conductivity screening),
bundled so this repo clones and runs on its own without Project 1 present:

- `catboost_model.cbm` — the trained CatBoost regressor (target = log₁₀ σ).
- `p1_conductivity.py` — self-contained feature-row builder + predictor (trimmed from
  Project 1's `src/featurize.py` + `04_screen_mp.py`; no plotting/data-loading).

`src/score.py` prefers these; if the full `project1_screening/` repo sits alongside
(monorepo / side-by-side layout) it falls back to importing from there instead.

The conductivity output is a **rank prior** (composition + structure), not a quantitative σ.
