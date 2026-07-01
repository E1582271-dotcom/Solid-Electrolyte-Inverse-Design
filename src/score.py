"""
Conductivity scoring bridge -- reuse the Project 1 CatBoost model (Project 3, W10).

This is the seam that makes the portfolio a *pipeline*: the cheap composition+structure
ionic-conductivity ranker trained in project 1 (OBELiX) is applied to the survivors of
the stability screen, giving each a coarse conductivity prior. Same honest caveat as
project 1: this is a RANK, not a quantitative sigma; novel chemistries score on
composition + structure only ('Family' defaults to the heuristic / 'unknown').

We import project 1's own featurizer + model loader (no re-implementation), so the two
projects stay in lockstep. import-only.
"""
from __future__ import annotations

import importlib.util
import os
from typing import Optional

import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.normpath(os.path.join(_HERE, "..", "vendor"))
_P1 = os.path.normpath(os.path.join(_HERE, "..", "..", "project1_screening"))


def _load_from_path(name, path):
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_scorer():
    """Load the conductivity scorer + model. Prefer the vendored self-contained copy
    (vendor/p1_conductivity.py + vendor/catboost_model.cbm) so this repo runs standalone;
    fall back to the sibling project1_screening/ repo if the vendored files are absent."""
    vmod = os.path.join(_VENDOR, "p1_conductivity.py")
    vmodel = os.path.join(_VENDOR, "catboost_model.cbm")
    if os.path.exists(vmod) and os.path.exists(vmodel):
        screen = _load_from_path("p1_conductivity", vmod)
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(vmodel)
        return screen, model
    return _load_project1_sibling()


def _load_project1_sibling():
    """Fallback: import project 1's 04_screen_mp helpers (build_feature_row / predict) + the
    model from a sibling project1_screening/ repo (monorepo / side-by-side layout).
    Returns (screen_module, CatBoostRegressor instance).

    project 1's ``04_screen_mp.py`` does ``from src.featurize import ...`` -- but ``src``
    is ALSO project 3's own package name. We temporarily install a synthetic ``src``
    package pointing at project 1's src dir for the duration of the load, then restore
    project 3's ``src`` modules, so the two never collide."""
    import sys
    import types

    feat_path = os.path.join(_P1, "src", "featurize.py")
    screen_path = os.path.join(_P1, "04_screen_mp.py")
    saved = {k: sys.modules[k] for k in list(sys.modules)
             if k == "src" or k.startswith("src.")}
    try:
        for k in saved:
            del sys.modules[k]
        pkg = types.ModuleType("src")
        pkg.__path__ = [os.path.join(_P1, "src")]
        sys.modules["src"] = pkg
        _load_from_path("src.featurize", feat_path)
        screen = _load_from_path("p1_screen", screen_path)  # its `from src.featurize` now resolves
    finally:
        for k in list(sys.modules):
            if k == "src" or k.startswith("src."):
                del sys.modules[k]
        sys.modules.update(saved)                            # restore project 3's `src`

    model_path = os.path.join(_P1, "catboost_model.cbm")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Project 1 model not found at {model_path} -- run project1_screening/01_train_eval.py first."
        )
    from catboost import CatBoostRegressor
    model = CatBoostRegressor()
    model.load_model(model_path)
    return screen, model


def _row_from_structure(screen, structure: Structure) -> dict:
    """Build a project-1 feature row from a (relaxed) pymatgen Structure."""
    formula = structure.composition.reduced_formula
    a, b, c = structure.lattice.abc
    alpha, beta, gamma = structure.lattice.angles
    try:
        sg = SpacegroupAnalyzer(structure, symprec=0.1).get_space_group_number()
    except Exception:
        sg = 1                              # generated cells can be low-symmetry / P1
    _, z = structure.composition.get_reduced_formula_and_factor()
    return screen.build_feature_row(formula, a, b, c, alpha, beta, gamma, int(sg), int(z))


def score_structures(structures: list[Structure],
                     labels: Optional[list[str]] = None) -> pd.DataFrame:
    """Predict log10(ionic conductivity) for each structure with the project-1 model.

    Returns a DataFrame (formula, Family, spacegroup_no, pred_log10_sigma,
    pred_sigma_S_cm, label) sorted by predicted conductivity, descending.
    """
    screen, model = _load_scorer()
    rows = []
    labels = labels or [f"cand_{i:03d}" for i in range(len(structures))]
    for lab, s in zip(labels, structures):
        row = _row_from_structure(screen, s)
        row = {"label": lab, "formula": s.composition.reduced_formula, **row}
        rows.append(row)
    ranked = screen.predict(model, rows)     # adds pred_log10_sigma / pred_sigma_S_cm, sorts
    return ranked
