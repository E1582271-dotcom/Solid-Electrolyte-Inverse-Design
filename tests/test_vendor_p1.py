"""The vendored Project 1 ensemble is intact and reproduces Project 1's own outputs."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import score as SC  # noqa: E402

VENDOR = ROOT / "vendor"
P1 = ROOT.parent / "Solid-Electrolyte-Screening"

# Frozen from project1_screening/candidate_queue.csv (P1 commit 099731e, models de72aa9):
# formula, ranking_score, ensemble_spread, ad_distance, evidence_state == "extrapolative"
P1_QUEUE_SUBSET = [
    ("Li20Si3P3S23Cl", -2.2616938324596543, 0.1161909853777895, 4.717161061048966, False),
    ("Li9S3N", -3.475747659133966, 0.3959674219528381, 5.900351654965452, True),
    ("Li6AsS5I", -5.466361544081773, 0.2522016618426946, 3.671871873969864, False),
    ("Li5SbS3I2", -6.01690858725057, 0.210521347368254, 6.217472029882716, False),
    ("Li8TiS6", -6.625391072063249, 0.2586492829719566, 5.4187095033241395, False),
    ("Li3BS3", -7.92696373734181, 0.2306492495593855, 3.017218706953638, False),
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_and_checksums():
    manifest = json.loads((VENDOR / "p1_manifest.json").read_text())
    assert manifest["representation"] == "M0_formula"
    assert manifest["n_members"] == len(manifest["members"]) == 15
    assert manifest["n_features"] == len(manifest["feature_names"]) == 134
    assert manifest["training_records"] == 599
    for member in manifest["members"]:
        assert _sha(VENDOR / member["path"]) == member["sha256"], member["path"]
    assert sorted(p.name for p in (VENDOR / "p1_models").glob("*.cbm")) == sorted(
        Path(m["path"]).name for m in manifest["members"])
    obelix = manifest["obelix_formulas"]
    assert _sha(VENDOR / obelix["path"]) == obelix["sha256"]
    assert not (VENDOR / "catboost_model.cbm").exists(), "stale single-model copy is back"


def test_reproduces_project1_queue_values():
    formulas = [row[0] for row in P1_QUEUE_SUBSET]
    scored = SC.score_formulas(formulas, labels=formulas).set_index("label").loc[formulas]
    expected = np.array([row[1:4] for row in P1_QUEUE_SUBSET], dtype=float)
    np.testing.assert_allclose(scored["pred_log10_sigma"], expected[:, 0], rtol=0, atol=1e-9)
    np.testing.assert_allclose(scored["pred_log10_sigma_spread"], expected[:, 1], rtol=0, atol=1e-9)
    np.testing.assert_allclose(scored["ad_distance"], expected[:, 2], rtol=0, atol=1e-9)
    assert scored["p1_extrapolative"].tolist() == [row[4] for row in P1_QUEUE_SUBSET]
    np.testing.assert_allclose(scored["pred_sigma_S_cm"], 10 ** expected[:, 0], rtol=1e-12)


@pytest.mark.skipif(not (P1 / "candidate_queue.csv").exists(),
                    reason="sibling Solid-Electrolyte-Screening checkout not present")
def test_matches_live_project1_checkout():
    queue = pd.read_csv(P1 / "candidate_queue.csv")
    p1_manifest = json.loads((P1 / "data" / "model_manifest.json").read_text())
    ours = json.loads((VENDOR / "p1_manifest.json").read_text())
    if [m["sha256"] for m in p1_manifest["members"]] != [m["sha256"] for m in ours["members"]]:
        pytest.skip("Project 1 has retrained since the last vendor refresh")
    scored = SC._vendor().score_formulas(queue["formula"].tolist())
    np.testing.assert_allclose(scored["pred_log10_sigma"], queue["ranking_score"], atol=1e-9, rtol=0)
    np.testing.assert_allclose(scored["pred_log10_sigma_spread"], queue["ensemble_spread"],
                               atol=1e-9, rtol=0)
    np.testing.assert_allclose(scored["ad_distance"], queue["ad_distance"], atol=1e-9, rtol=0)
    assert (scored["p1_extrapolative"].to_numpy()
            == queue["evidence_state"].eq("extrapolative").to_numpy()).all()


def test_formula_only_scale_invariant_and_sorted():
    scored = SC.score_formulas(["LiPS3", "Li3PS4", "Li6P2S8", "Li8PS2"],
                               labels=["a", "b", "c", "d"])
    assert list(scored.columns) == ["label", "formula"] + SC.SCORE_COLUMNS
    assert scored["pred_log10_sigma"].is_monotonic_decreasing
    by = scored.set_index("label")
    assert by.loc["b", "pred_log10_sigma"] == pytest.approx(by.loc["c", "pred_log10_sigma"], abs=1e-12)
    # Li3PS4 and LiPS3 are measured in OBELiX; Li8PS2 is not
    assert bool(by.loc["b", "obelix_composition_match"]) and not bool(by.loc["d", "obelix_composition_match"])
