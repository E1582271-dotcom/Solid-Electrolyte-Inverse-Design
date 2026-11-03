"""
Vendored Project 1 conductivity ranker (production path), so Project 3 scores candidates with
exactly the model Project 1 ships, without Project 1 present.

What it reproduces (Project 1 ``src/p1screen``):
  - features.composition_features / formula_matrix : 132 matminer Magpie statistics +
    n_elements + Li_frac (134 formula-only descriptors, representation "M0_formula").
  - modeling.load_ensemble / ensemble_predictions  : 15 CatBoost members (3 configurations x
    5 seeds) trained on all 599 OBELiX records; score = member mean of log10(sigma / S cm^-1),
    spread = member standard deviation (ddof=0).
  - domain.ApplicabilityDomain                      : median-impute -> standardise -> PCA (95 %
    variance) -> mean distance to the 5 nearest OBELiX training records.
  - screening._disposition (catalogue match only)   : whether the composition was measured in
    OBELiX (nominal or feature formula), in which case a formula-only score mostly echoes the
    measured values of that composition.

Formula-only by design: Project 1's METHODS forbids scoring catalogue formulas with the
raw-lattice ablations (M1/M2), so structure, space group and polymorph do not enter. Every
polymorph of one composition therefore receives the same score.

Model files and provenance live in ``vendor/p1_manifest.json`` (sha256 per member, checked
before every load); refresh them with ``vendor/refresh_from_p1.py``. The output is a coarse
RANKING prior, not a quantitative sigma.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

VENDOR = Path(__file__).resolve().parent
MANIFEST_PATH = VENDOR / "p1_manifest.json"


# --------------------------------------------------------------------------- #
# Manifest + models
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def load_ensemble() -> tuple:
    """Load the 15 members after verifying each file's sha256 against the manifest."""
    from catboost import CatBoostRegressor

    manifest = load_manifest()
    models = []
    for member in manifest["members"]:
        path = VENDOR / member["path"]
        if _sha256(path) != member["sha256"]:
            raise RuntimeError(f"model checksum mismatch: {path} (re-run vendor/refresh_from_p1.py)")
        model = CatBoostRegressor()
        model.load_model(str(path))
        if list(model.feature_names_) != manifest["feature_names"]:
            raise RuntimeError(f"{path.name}: feature order differs from the manifest")
        models.append((member["config"]["name"], int(member["seed"]), model))
    if len(models) != int(manifest["n_members"]):
        raise RuntimeError(f"expected {manifest['n_members']} members, loaded {len(models)}")
    return tuple(models)


# --------------------------------------------------------------------------- #
# Formula handling + features (Project 1 composition.py / features.py)
# --------------------------------------------------------------------------- #
def normalize_formula_text(value: object) -> str:
    text = str(value).strip().replace(" ", "")
    return text.replace("*", "").replace("·", "")


def as_composition(value: object):
    from pymatgen.core import Composition

    return Composition(normalize_formula_text(value))


def composition_key(value: object, decimals: int = 8) -> str:
    """Scale- and ordering-invariant elemental-fraction key (Project 1 composition_key)."""
    fractional = as_composition(value).fractional_composition
    return "|".join(
        f"{element}:{float(amount):.{decimals}f}"
        for element, amount in sorted(fractional.items(), key=lambda item: str(item[0]))
        if float(amount) > 10 ** (-(decimals + 1))
    )


@lru_cache(maxsize=1)
def _magpie():
    from matminer.featurizers.composition import ElementProperty

    return ElementProperty.from_preset("magpie")


def feature_names() -> list[str]:
    names = list(_magpie().feature_labels()) + ["n_elements", "Li_frac"]
    if names != load_manifest()["feature_names"]:
        raise RuntimeError("installed matminer Magpie labels differ from the vendored feature contract")
    return names


def composition_features(formula: object) -> dict[str, float]:
    comp = as_composition(formula)
    featurizer = _magpie()
    result = dict(zip(featurizer.feature_labels(), featurizer.featurize(comp), strict=True))
    fractions = comp.fractional_composition.get_el_amt_dict()
    result["n_elements"] = float(len(fractions))
    result["Li_frac"] = float(fractions.get("Li", 0.0))
    return result


def formula_matrix(formulas) -> pd.DataFrame:
    return pd.DataFrame([composition_features(f) for f in formulas]).reindex(columns=feature_names())


def ensemble_predictions(models, X: pd.DataFrame):
    predictions = np.stack([model.predict(X) for _, _, model in models])
    return predictions.mean(axis=0), predictions.std(axis=0), predictions


# --------------------------------------------------------------------------- #
# Applicability domain + OBELiX catalogue match (Project 1 domain.py / screening.py)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _obelix_reference() -> pd.DataFrame:
    info = load_manifest()["obelix_formulas"]
    path = VENDOR / info["path"]
    if _sha256(path) != info["sha256"]:
        raise RuntimeError(f"checksum mismatch: {path} (re-run vendor/refresh_from_p1.py)")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if len(frame) != int(info["n_records"]):
        raise RuntimeError(f"{path.name}: expected {info['n_records']} records, found {len(frame)}")
    return frame


class ApplicabilityDomain:
    """Copy of Project 1's ApplicabilityDomain (same defaults, same distance)."""

    def __init__(self, variance: float, neighbors: int):
        self.variance = variance
        self.neighbors = neighbors

    def fit(self, X: pd.DataFrame) -> "ApplicabilityDomain":
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler

        self.imputer = SimpleImputer(strategy="median").fit(X)
        imputed = self.imputer.transform(X)
        self.scaler = StandardScaler().fit(imputed)
        scaled = self.scaler.transform(imputed)
        self.pca = PCA(n_components=self.variance, svd_solver="full").fit(scaled)
        self.nn = NearestNeighbors(n_neighbors=self.neighbors + 1).fit(self.pca.transform(scaled))
        return self

    def distance(self, X: pd.DataFrame) -> np.ndarray:
        projected = self.pca.transform(self.scaler.transform(self.imputer.transform(X)))
        distances = self.nn.kneighbors(projected, n_neighbors=self.neighbors,
                                       return_distance=True)[0]
        return distances.mean(axis=1)


@lru_cache(maxsize=1)
def applicability_domain() -> ApplicabilityDomain:
    ad = load_manifest()["ad"]
    domain = ApplicabilityDomain(variance=float(ad["variance_retained"]),
                                 neighbors=int(ad["neighbors"]))
    domain.fit(formula_matrix(_obelix_reference()["feature_formula"].tolist()))
    if int(domain.pca.n_components_) != int(ad["n_components"]):
        raise RuntimeError(f"AD refit kept {domain.pca.n_components_} components, "
                           f"Project 1 recorded {ad['n_components']}")
    return domain


@lru_cache(maxsize=1)
def _obelix_keys() -> frozenset:
    ref = _obelix_reference()
    return frozenset(composition_key(f) for f in ref["nominal_formula"]) | frozenset(
        composition_key(f) for f in ref["feature_formula"])


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def score_formulas(formulas) -> pd.DataFrame:
    """Score formulas with the vendored Project 1 ensemble. Returns one row per input, in input
    order: formula, pred_log10_sigma (ensemble mean), pred_log10_sigma_spread (ensemble SD),
    ad_distance, obelix_composition_match, p1_extrapolative (spread or AD at/above Project 1's
    frozen route thresholds)."""
    formulas = [str(f) for f in formulas]
    if not formulas:
        return pd.DataFrame(columns=["formula", "pred_log10_sigma", "pred_log10_sigma_spread",
                                     "ad_distance", "obelix_composition_match", "p1_extrapolative"])
    X = formula_matrix(formulas)
    mean, spread, _ = ensemble_predictions(load_ensemble(), X)
    ad = applicability_domain().distance(X)
    thresholds = load_manifest()["route_thresholds"]
    keys = _obelix_keys()
    return pd.DataFrame({
        "formula": formulas,
        "pred_log10_sigma": mean,
        "pred_log10_sigma_spread": spread,
        "ad_distance": ad,
        "obelix_composition_match": [composition_key(f) in keys for f in formulas],
        "p1_extrapolative": (spread >= float(thresholds["spread"])) | (ad >= float(thresholds["ad"])),
    })


def provenance() -> dict:
    """Small, JSON-ready description of the vendored model (for run logs)."""
    m = load_manifest()
    return {
        "model": "Project 1 production CatBoost ensemble (formula-only M0, vendored)",
        "n_members": m["n_members"],
        "n_features": m["n_features"],
        "training_records": m["training_records"],
        "p1_project_version": m["source"]["project_version"],
        "p1_git_head": m["source"]["git_head"],
        "p1_models_commit": m["source"]["models_last_changed_commit"],
        "manifest_sha256": _sha256(MANIFEST_PATH),
        "score": "ensemble mean of log10(sigma / S cm^-1)",
        "spread": "ensemble standard deviation (ddof=0)",
        "route_thresholds": m["route_thresholds"],
    }
