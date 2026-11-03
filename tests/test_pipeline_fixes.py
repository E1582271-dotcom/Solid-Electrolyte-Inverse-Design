"""Regression tests for the screen/rank/log fixes (no MLIP, no network)."""
from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import generate as G  # noqa: E402
from src import novelty as NV  # noqa: E402
from src import stability as ST  # noqa: E402


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RANK = _load("p3_rank", "04_rank_candidates.py")


def _rocksalt(a: float) -> Structure:
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), ["Li", "S"],
                                     [[0, 0, 0], [0.5, 0.5, 0.5]])


# --- 02: an unstable twin listed first must not make the stable twin non-unique ---
def test_dedupe_prefers_stable_representative():
    structs = [_rocksalt(5.70), _rocksalt(5.71), _rocksalt(5.72)]   # all one structure
    stable = [False, True, True]
    reps, groups = NV.dedupe(structs, order=NV.stable_first_order(stable))
    assert reps == [1] and sorted(groups[0]) == [0, 1, 2]
    reps_old, _ = NV.dedupe(structs)                                 # input order: first wins
    assert reps_old == [0]
    with pytest.raises(ValueError):
        NV.dedupe(structs, order=[0, 0, 1])


# --- stability.py: the reference cache is tied to one chemical system ---
def test_reference_cache_refuses_other_chemsys(tmp_path):
    shipped = ROOT / "data" / "ref_structures.json"
    refs = ST.fetch_reference_entries(["Li", "P", "S"], cache_path=str(shipped))
    assert len(refs) == 96
    with pytest.raises(RuntimeError, match="Li-P-S-Si"):
        ST.fetch_reference_entries(["Li", "Si", "P", "S"], cache_path=str(shipped))
    tagged = tmp_path / "ref_LiS.json"
    entry = {"entry_id": "x", "structure": _rocksalt(5.7).as_dict()}
    tagged.write_text(json.dumps({"chemsys": ["Li", "S"], "entries": [entry]}))
    assert len(ST.fetch_reference_entries(["S", "Li"], cache_path=str(tagged))) == 1
    with pytest.raises(RuntimeError):
        ST.fetch_reference_entries(["Li", "P", "S"], cache_path=str(tagged))


# --- generate.py: MP key from the environment only ---
def test_mp_api_key_env_only(monkeypatch):
    monkeypatch.delenv("MP_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="MP_API_KEY"):
        G.mp_api_key()
    monkeypatch.setenv("MP_API_KEY", "  abc123 ")
    assert G.mp_api_key() == "abc123"


# --- 04: cutoff comes from 02's output; funnel counts from the right records ---
def _screened(cut=0.1):
    df = pd.DataFrame({"label": ["gen_000", "gen_001"], "e_above_hull": [0.05, 0.2],
                       "stable": [True, False]})
    if cut is not None:
        df["ehull_cutoff"] = cut
    return df


def test_rank_reads_cutoff_from_screen():
    assert RANK.screen_cutoff(_screened(0.1)) == 0.1
    with pytest.raises(SystemExit):
        RANK.screen_cutoff(_screened(None))
    with pytest.raises(SystemExit):
        RANK.screen_cutoff(_screened(0.3))         # flags disagree with the recorded cutoff


def test_funnel_counts_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(RANK, "GEN_DIR", str(tmp_path))
    args = Namespace(batch_size=16, num_batches=4)
    counts, sources = RANK.funnel_counts(_screened(), args)
    assert counts == {"n_generated": 64, "n_li_bearing": 2}
    assert "batch_size" in sources["n_generated"]
    (tmp_path / "generation_summary.json").write_text(json.dumps({"n_generated": 64, "n_kept": 61}))
    pd.DataFrame({"label": [f"gen_{i:03d}" for i in range(61)]}).to_csv(
        tmp_path / "manifest.csv", index=False)
    counts, sources = RANK.funnel_counts(_screened(), Namespace(batch_size=None, num_batches=None))
    assert counts == {"n_generated": 64, "n_li_bearing": 61}
    assert sources["n_generated"].endswith("generation_summary.json")


# --- shipped deliverables agree with each other ---
def test_shipped_run_log_matches_candidates():
    runs = json.loads((ROOT / "data" / "p3_runs.json").read_text())
    final = pd.read_csv(ROOT / "data" / "candidates_final.csv")
    c = runs["counts"]
    assert (c["n_generated"], c["n_li_bearing"]) == (64, 61)
    assert c["n_li_bearing"] == len(final)
    assert c["n_stable"] == int(final["stable"].sum()) and c["n_sun"] == int(final["SUN"].sum())
    assert c["n_scored"] == int(final["pred_log10_sigma"].notna().sum())
    cut = runs["params"]["ehull_cutoff"]
    assert (final["stable"] == (final["e_above_hull"] <= cut)).all()
    assert runs["scorer"]["n_members"] == 15
    scored = final[final["pred_log10_sigma"].notna()]
    assert scored["pred_log10_sigma"].is_monotonic_decreasing
    assert scored["score_rank"].tolist() == list(range(1, len(scored) + 1))
