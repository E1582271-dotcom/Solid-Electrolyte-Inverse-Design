"""
Refresh the vendored Project 1 conductivity ensemble from a Project 1 checkout.

Copies the 15 production CatBoost members (``artifacts/models/*.cbm``), writes
``vendor/p1_manifest.json`` (provenance, per-member sha256, feature contract, P1's frozen
routing thresholds) and ``vendor/p1_obelix_formulas.csv`` (the 599 OBELiX formulas the
production ensemble was trained on; used for the applicability-domain distance and the
"composition already measured in OBELiX" flag).

Every member checksum is checked three ways before anything is written: the bytes on disk,
Project 1's ``data/model_manifest.json`` and Project 1's ``data/release_manifest.json``. Any
mismatch aborts the refresh, so the vendored copy is always a verified Project 1 release.

Run (from anywhere; only reads the Project 1 checkout):
    python vendor/refresh_from_p1.py --p1 ../Solid-Electrolyte-Screening
Then re-run 03_score_conductivity.py + 04_rank_candidates.py and the figure scripts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

VENDOR = Path(__file__).resolve().parent
MODELS_DIR = VENDOR / "p1_models"
MANIFEST = VENDOR / "p1_manifest.json"
FORMULAS = VENDOR / "p1_obelix_formulas.csv"
P1_REPO_URL = "https://github.com/E1582271-dotcom/Solid-Electrolyte-Screening"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(p1: Path, *args: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(p1), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parseable(text: str) -> bool:
    """Same rule as Project 1's data.resolve_feature_formula: pymatgen must parse it."""
    from pymatgen.core import Composition

    try:
        Composition(str(text).strip().replace(" ", "").replace("*", "").replace("·", ""))
    except Exception:
        return False
    return str(text).strip() not in {"", "nan", "None"}


def _obelix_formulas(p1: Path) -> list[dict]:
    """Feature formula per OBELiX record: 'True Composition' when parseable, otherwise
    'Reduced Composition' (Project 1's data.resolve_feature_formula)."""
    rows = []
    for split, name in (("official_train", "train.csv"), ("official_test", "test.csv")):
        with open(p1 / "data" / name, newline="", encoding="utf-8") as f:
            for rec in csv.DictReader(f):
                for column in ("True Composition", "Reduced Composition"):
                    if _parseable(rec.get(column, "")):
                        feature, source = rec[column], column
                        break
                else:
                    raise ValueError(f"OBELiX record {rec.get('ID')} has no parseable formula")
                rows.append({"record_id": rec["ID"], "source_split": split,
                             "nominal_formula": rec["Reduced Composition"],
                             "feature_formula": feature, "feature_formula_source": source})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--p1", default=str(VENDOR.parent.parent / "Solid-Electrolyte-Screening"),
                    help="path to a Project 1 (Solid-Electrolyte-Screening) checkout")
    args = ap.parse_args()
    p1 = Path(args.p1).resolve()

    model_manifest = json.loads((p1 / "data" / "model_manifest.json").read_text())
    release = json.loads((p1 / "data" / "release_manifest.json").read_text())
    release_sha = {f["path"]: f["sha256"] for f in release["files"]}
    if model_manifest["representation"] != "M0_formula":
        raise SystemExit(f"refusing: P1 production representation is "
                         f"{model_manifest['representation']!r}, not the formula-only M0")
    if release_sha.get("data/model_manifest.json") != _sha256(p1 / "data" / "model_manifest.json"):
        raise SystemExit("refusing: P1 data/model_manifest.json does not match its release manifest")

    feature_names = list(model_manifest["feature_names"])
    members = []
    for m in model_manifest["members"]:
        src = p1 / "artifacts" / "models" / m["path"]
        digest = _sha256(src)
        rel = f"artifacts/models/{m['path']}"
        if digest != m["sha256"] or digest != release_sha.get(rel):
            raise SystemExit(f"refusing: checksum mismatch for {rel} "
                             f"(disk {digest[:12]}, model_manifest {m['sha256'][:12]}, "
                             f"release_manifest {str(release_sha.get(rel))[:12]})")
        if list(m["feature_names"]) != feature_names:
            raise SystemExit(f"refusing: {m['path']} feature order differs from the manifest")
        members.append({"path": f"p1_models/{m['path']}", "source_path": rel,
                        "sha256": digest, "bytes": src.stat().st_size,
                        "seed": int(m["seed"]), "config": m["config"]})
    if len(members) != int(release["contracts"]["model_members"]):
        raise SystemExit(f"refusing: {len(members)} members vs release contract "
                         f"{release['contracts']['model_members']}")

    formulas = _obelix_formulas(p1)
    if len(formulas) != int(model_manifest["training_records"]):
        raise SystemExit(f"refusing: {len(formulas)} OBELiX records vs "
                         f"{model_manifest['training_records']} in the model manifest")

    # --- write (only after every check passed) ---
    if MODELS_DIR.exists():
        shutil.rmtree(MODELS_DIR)
    MODELS_DIR.mkdir(parents=True)
    for m in members:
        shutil.copy2(p1 / m["source_path"], VENDOR / m["path"])
        assert _sha256(VENDOR / m["path"]) == m["sha256"]
    with open(FORMULAS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(formulas[0]))
        w.writeheader()
        w.writerows(formulas)

    head = _git(p1, "rev-parse", "HEAD")
    models_commit = _git(p1, "log", "-1", "--format=%H", "--", "artifacts/models",
                         "data/model_manifest.json")
    dirty = _git(p1, "status", "--porcelain", "--", "artifacts/models", "data/model_manifest.json")
    manifest = {
        "description": ("Vendored copy of Project 1's production conductivity ranker: a 15-member "
                        "CatBoost perturbation ensemble on 134 formula-only descriptors (132 Magpie "
                        "statistics + n_elements + Li_frac), trained on all 599 OBELiX records. "
                        "Score = ensemble mean log10(sigma / S cm^-1); spread = ensemble SD."),
        "source": {
            "project": "Project 1 (project1_screening / Solid-Electrolyte-Screening)",
            "repository": P1_REPO_URL,
            "project_version": release.get("project_version"),
            "git_head": head,
            "models_last_changed_commit": models_commit,
            "model_files_clean_in_source_checkout": (dirty == "") if dirty is not None else None,
            "release_manifest_verified_utc": release.get("verified_utc"),
            "release_manifest_python": release.get("python"),
            "release_manifest_package_versions": release.get("package_versions"),
            "model_manifest_sha256": _sha256(p1 / "data" / "model_manifest.json"),
        },
        "refreshed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checksum_policy": ("every member sha256 below equals Project 1's data/model_manifest.json "
                            "and data/release_manifest.json at refresh time; the loader re-checks "
                            "the bytes before scoring"),
        "representation": model_manifest["representation"],
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "n_members": len(members),
        "members": members,
        "training_records": model_manifest["training_records"],
        "training_compositions": model_manifest["training_compositions"],
        "training_weighting": model_manifest["training_weighting"],
        "spread_note": model_manifest.get("variance_decomposition_note"),
        "ad": model_manifest["ad"],
        "route_quantile": model_manifest["route_quantile"],
        "route_thresholds": model_manifest["route_thresholds"],
        "route_thresholds_note": ("Project 1's frozen 75th-percentile thresholds over its own "
                                  "62-composition queue; Project 3 reuses the spread and AD "
                                  "thresholds only to flag extrapolative scores on P1's scale"),
        "obelix_formulas": {"path": FORMULAS.name, "sha256": _sha256(FORMULAS),
                            "n_records": len(formulas),
                            "license": "CC BY 4.0 (OBELiX; see vendor/README.md)"},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"vendored {len(members)} members + {len(formulas)} OBELiX formulas from {p1}")
    print(f"P1 HEAD {head}; models last changed in {models_commit}")


if __name__ == "__main__":
    main()
