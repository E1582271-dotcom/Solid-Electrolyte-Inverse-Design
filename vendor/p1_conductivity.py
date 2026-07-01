"""
Vendored conductivity scorer -- a self-contained copy of Project 1's ionic-conductivity
model helpers, bundled so Project 3 clones and runs on its own (Project 1 need not be
present alongside). Source of truth: `project1_screening/src/featurize.py` +
`project1_screening/04_screen_mp.py`; the trained model is `vendor/catboost_model.cbm`.
Kept minimal -- only the feature-row builder + predictor, no plotting/data-loading.

The output is a RANK (composition + structure prior), not a quantitative sigma. See README.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element

# Element properties averaged over a composition (from project-1 featurize.py).
_ELEM_PROPS = {
    "Z": lambda e: e.Z,
    "mass": lambda e: float(e.atomic_mass),
    "X": lambda e: e.X,                      # Pauling electronegativity
    "row": lambda e: e.row,
    "group": lambda e: e.group,
    "mendeleev": lambda e: e.mendeleev_no,
    "radius": lambda e: float(e.atomic_radius) if e.atomic_radius else np.nan,
}

CAT_FEATURES = ["Family", "crystal_system"]


def _crystal_system(sg_number: int) -> str:
    n = int(sg_number)
    bounds = [(2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"),
              (142, "tetragonal"), (167, "trigonal"), (194, "hexagonal"),
              (230, "cubic")]
    for hi, name in bounds:
        if n <= hi:
            return name
    return "unknown"


def _cell_volume(a, b, c, al, be, ga) -> float:
    al, be, ga = np.radians([al, be, ga])
    return float(a * b * c * np.sqrt(
        1 - np.cos(al) ** 2 - np.cos(be) ** 2 - np.cos(ga) ** 2
        + 2 * np.cos(al) * np.cos(be) * np.cos(ga)))


def _composition_features(comp_str: str) -> dict:
    """Fraction-weighted mean/std/min/max + Li fraction + n_elements."""
    try:
        comp = Composition(comp_str)
    except Exception:
        return {}
    fracs = comp.fractional_composition.get_el_amt_dict()
    els = {Element(sym): w for sym, w in fracs.items()}
    feat = {
        "n_elements": len(els),
        "Li_frac": fracs.get("Li", 0.0),
    }
    for pname, getter in _ELEM_PROPS.items():
        vals = np.array([getter(e) for e in els], dtype=float)
        wts = np.array(list(els.values()), dtype=float)
        ok = ~np.isnan(vals)
        if ok.sum() == 0:
            feat[f"{pname}_mean"] = feat[f"{pname}_std"] = np.nan
            feat[f"{pname}_min"] = feat[f"{pname}_max"] = feat[f"{pname}_range"] = np.nan
            continue
        v, w = vals[ok], wts[ok]
        w = w / w.sum()
        mean = float((v * w).sum())
        feat[f"{pname}_mean"] = mean
        feat[f"{pname}_std"] = float(np.sqrt((w * (v - mean) ** 2).sum()))
        feat[f"{pname}_min"] = float(v.min())
        feat[f"{pname}_max"] = float(v.max())
        feat[f"{pname}_range"] = float(v.max() - v.min())
    return feat


def guess_family(formula: str) -> str:
    """Map a composition to an OBELiX family by stoichiometry. Conservative:
    returns 'unknown' unless the pattern is a well-known sulfide class."""
    try:
        comp = Composition(formula)
    except Exception:
        return "unknown"
    el = {str(e) for e in comp.elements}
    d = comp.get_el_amt_dict()
    if "S" not in el:
        return "unknown"
    if {"Li", "P", "S"} <= el and el & {"Cl", "Br", "I"}:
        x = sum(d.get(h, 0) for h in ("Cl", "Br", "I"))
        if 0.5 <= x and 4.0 <= d.get("S", 0) / max(x, 1e-9) <= 7.0:
            return "argyrodites"
    if {"Li", "P", "S"} <= el and el & {"Ge", "Sn", "Si"}:
        return "LGPS"
    if "P" not in el and el & {"Ge", "Sn", "Si", "Al", "Ga"} and "Li" in el:
        return "thio-LISICON"
    if el <= {"Li", "P", "S"} and "Li" in el and "P" in el:
        return "sulfides"
    return "unknown"


def build_feature_row(formula, a, b, c, alpha, beta, gamma, sg_no, Z,
                      family=None) -> dict:
    feat = _composition_features(formula)
    feat.update({
        "a": a, "b": b, "c": c, "alpha": alpha, "beta": beta, "gamma": gamma,
        "cell_volume": _cell_volume(a, b, c, alpha, beta, gamma),
        "spacegroup_no": sg_no, "Z": Z,
        "crystal_system": _crystal_system(sg_no),
        "Family": family if family is not None else guess_family(formula),
    })
    return feat


def predict(model, rows: list[dict]) -> pd.DataFrame:
    """rows -> DataFrame reindexed to the model's own feature order, then predicted.
    Returns rows + pred_log10_sigma + pred_sigma_S_cm, ranked (descending)."""
    names = list(model.feature_names_)
    X = pd.DataFrame(rows).reindex(columns=names)
    for col in names:
        if col in CAT_FEATURES:
            X[col] = X[col].astype(str)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    yhat = model.predict(X)
    out = pd.DataFrame(rows)
    out["pred_log10_sigma"] = yhat
    out["pred_sigma_S_cm"] = np.power(10.0, yhat)
    return out.sort_values("pred_log10_sigma", ascending=False).reset_index(drop=True)
