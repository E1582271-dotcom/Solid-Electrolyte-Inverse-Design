"""
Conductivity scoring bridge -- Project 1's production ranker applied to Project 3 candidates.

Project 3 is an independent generative track; it only borrows Project 1's trained model as an
off-the-shelf scorer. The model is the vendored copy in ``vendor/`` (15-member CatBoost
ensemble on 134 formula-only Magpie descriptors, trained on all 599 OBELiX records; checksums
and source commit in ``vendor/p1_manifest.json``, refreshed with ``vendor/refresh_from_p1.py``).

Score = ensemble mean predicted log10(sigma / S cm^-1); the ensemble standard deviation is
reported as ``pred_log10_sigma_spread``, together with Project 1's applicability-domain
distance and an OBELiX catalogue-match flag. The descriptors are formula-only, so all
polymorphs of a composition share one score. As in Project 1 this is a coarse RANKING prior,
not a quantitative sigma. import-only.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

_VENDOR_MODULE = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vendor", "p1_conductivity.py"))

SCORE_COLUMNS = ["pred_log10_sigma", "pred_sigma_S_cm", "pred_log10_sigma_spread",
                 "ad_distance", "obelix_composition_match", "p1_extrapolative"]


def _vendor():
    name = "p3_vendor_p1_conductivity"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _VENDOR_MODULE)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def score_formulas(formulas: list[str], labels: Optional[list[str]] = None) -> pd.DataFrame:
    """Score each formula with the vendored Project 1 ensemble.

    Returns one row per input (label, formula, pred_log10_sigma, pred_sigma_S_cm,
    pred_log10_sigma_spread, ad_distance, obelix_composition_match, p1_extrapolative), sorted by
    pred_log10_sigma descending; ties keep input order.
    """
    formulas = [str(f) for f in formulas]
    labels = list(labels) if labels is not None else [f"cand_{i:03d}" for i in range(len(formulas))]
    if len(labels) != len(formulas):
        raise ValueError("labels and formulas differ in length")
    scored = _vendor().score_formulas(formulas)
    scored.insert(0, "label", labels)
    scored.insert(3, "pred_sigma_S_cm", np.power(10.0, scored["pred_log10_sigma"].astype(float)))
    return scored.sort_values("pred_log10_sigma", ascending=False, kind="stable").reset_index(drop=True)


def score_structures(structures, labels: Optional[list[str]] = None) -> pd.DataFrame:
    """Convenience wrapper: score pymatgen Structures by their reduced formula (the model is
    formula-only; lattice and symmetry are deliberately not used)."""
    return score_formulas([s.composition.reduced_formula for s in structures], labels=labels)


def provenance() -> dict:
    """JSON-ready description of the scorer (source commit, members, manifest checksum)."""
    return _vendor().provenance()
