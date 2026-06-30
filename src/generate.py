"""
MatterGen conditional generation wrapper for the inverse-design demo (Project 3, W10).

Positioning (repeat in README): this is a *concept-validation* of the inverse-design
paradigm -- generate candidates conditioned on a chemical system, then screen them.
We do NOT claim synthesizability; MatterGen's own paper validated a single synthesis.

Split of labour (mirrors projects 1 & 2: heavy on GPU, light on the laptop):
- ``run_mattergen_cli`` shells out to MatterGen's official ``mattergen-generate`` CLI.
  This needs the mattergen package + a GPU, so it runs in the cloud notebook, NOT on a
  laptop. It is a thin, defensive subprocess wrapper -- the exact flags live in the
  notebook so they track the upstream README (versions move).
- ``read_generated`` / ``save_structures`` / ``mp_demo_candidates`` are pure pymatgen and
  run anywhere. They let the whole *downstream* pipeline (screen -> score -> rank) be
  developed and verified on a CPU-only laptop against either real MatterGen output or a
  handful of real MP structures used as stand-in "generated" candidates.

import-only.
"""
from __future__ import annotations

import os
import subprocess
import zipfile
from typing import Optional

from pymatgen.core import Structure

# Official MatterGen artefacts (see notebooks/01_mattergen_pipeline.ipynb):
#   mattergen-generate $OUT --pretrained-name=chemical_system --batch_size=16 \
#       --properties_to_condition_on="{'chemical_system': 'Li-P-S'}" \
#       --diffusion_guidance_factor=2.0
# writes generated_crystals.extxyz + generated_crystals_cif.zip into $OUT.
EXTXYZ_NAME = "generated_crystals.extxyz"
CIF_ZIP_NAME = "generated_crystals_cif.zip"
DEFAULT_MODEL = "chemical_system"
DEFAULT_CHEMSYS = "Li-P-S"


# --------------------------------------------------------------------------- #
# Heavy path: drive the official CLI (cloud / GPU only)
# --------------------------------------------------------------------------- #
def run_mattergen_cli(
    results_path: str,
    chemsys: str = DEFAULT_CHEMSYS,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 16,
    num_batches: int = 1,
    guidance_factor: float = 2.0,
    extra_args: Optional[list[str]] = None,
) -> str:
    """Invoke ``mattergen-generate`` for chemical-system-conditioned generation.

    Returns ``results_path``. Raises if the CLI is missing (run in the MatterGen env)
    or exits non-zero. Kept deliberately thin: see the notebook for the canonical call
    and for installing MatterGen (``uv venv --python 3.10`` + ``uv pip install -e .``).
    """
    os.makedirs(results_path, exist_ok=True)
    cmd = [
        "mattergen-generate", results_path,
        f"--pretrained-name={model_name}",
        f"--batch_size={int(batch_size)}",
        f"--num_batches={int(num_batches)}",
        # MatterGen parses a python-dict-style string; single quotes inside are required.
        f"--properties_to_condition_on={{'chemical_system': '{chemsys}'}}",
        f"--diffusion_guidance_factor={float(guidance_factor)}",
    ]
    if extra_args:
        cmd += list(extra_args)
    print("[generate] $", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return results_path


# --------------------------------------------------------------------------- #
# Light path: read / write / stand-in candidates (CPU, anywhere)
# --------------------------------------------------------------------------- #
def read_generated(results_path: str) -> list[Structure]:
    """Load MatterGen output from a results dir into pymatgen Structures.

    Prefers the per-structure CIF zip (lattice + species are unambiguous); falls back
    to the multi-frame extxyz via ASE.
    """
    cif_zip = os.path.join(results_path, CIF_ZIP_NAME)
    extxyz = os.path.join(results_path, EXTXYZ_NAME)
    if os.path.exists(cif_zip):
        out = []
        with zipfile.ZipFile(cif_zip) as zf:
            for name in sorted(zf.namelist()):
                if name.endswith(".cif"):
                    out.append(Structure.from_str(zf.read(name).decode(), fmt="cif"))
        if out:
            return out
    if os.path.exists(extxyz):
        from ase.io import read as ase_read
        from pymatgen.io.ase import AseAtomsAdaptor

        frames = ase_read(extxyz, index=":")
        return [AseAtomsAdaptor.get_structure(a) for a in frames]
    raise FileNotFoundError(
        f"No MatterGen output in {results_path} "
        f"(expected {CIF_ZIP_NAME} or {EXTXYZ_NAME})."
    )


def save_structures(structures: list[Structure], out_dir: str, prefix: str = "gen") -> list[str]:
    """Write each structure to ``out_dir/<prefix>_<i>.cif``. Returns the paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, s in enumerate(structures):
        p = os.path.join(out_dir, f"{prefix}_{i:03d}.cif")
        s.to(filename=p)
        paths.append(p)
    return paths


def load_structures(in_dir: str, prefix: str = "gen") -> list[Structure]:
    """Re-load CIFs previously written by ``save_structures`` (sorted by name)."""
    files = sorted(f for f in os.listdir(in_dir) if f.startswith(prefix) and f.endswith(".cif"))
    return [Structure.from_file(os.path.join(in_dir, f)) for f in files]


def mp_demo_candidates(
    chemsys: list[str] | None = None,
    api_key: Optional[str] = None,
    max_n: int = 6,
    rattle: float = 0.0,
    seed: int = 0,
) -> list[Structure]:
    """Stand-in "generated" candidates pulled from Materials Project, for developing and
    verifying the downstream screen->score->rank pipeline OFFLINE (no GPU / no MatterGen).

    HONEST: these are real, known phases -- NOT generative output. With ``rattle`` > 0 the
    atoms are randomly displaced (A) so they sit slightly off the hull, mimicking a raw
    generated structure that still needs relaxation. Used only by ``01_generate --source
    mp-demo``; the real run uses ``run_mattergen_cli``. This mirrors project 1's ``--demo``
    and project 2's Lennard-Jones plumbing check.
    """
    import numpy as np

    chemsys = chemsys or ["Li", "P", "S"]
    key = _find_api_key(api_key)
    if not key:
        raise RuntimeError("mp-demo needs an MP API key ($MP_API_KEY or project-1 key file).")
    from mp_api.client import MPRester

    with MPRester(key) as mpr:
        entries = mpr.get_entries_in_chemsys(chemsys, inc_structure=True)
    # We are standing in for a Li-ion-conductor generator, so keep only Li-bearing phases
    # (a fixture with no Li would make the downstream conductivity score meaningless), and
    # prefer the Li-P-S ternaries -- the actual target class -- over Li-S / Li-P binaries.
    # energy_per_atom is a cheap stability proxy (lower first) to pick a sensible handful.
    mobile = "Li"
    entries = [e for e in entries if mobile in {str(el) for el in e.composition.elements}
               and len(e.composition.elements) >= 2]
    n_target = len(set(chemsys))            # 3 for Li-P-S -> ternaries rank ahead of binaries
    entries = sorted(entries, key=lambda e: (n_target - len(e.composition.elements),
                                             e.energy_per_atom))
    structs = [e.structure for e in entries[: max_n * 3]]
    # de-dup-ish: keep distinct reduced formulas, take first max_n
    seen, picked = set(), []
    for s in structs:
        f = s.composition.reduced_formula
        if f in seen:
            continue
        seen.add(f)
        picked.append(s)
        if len(picked) >= max_n:
            break
    if rattle > 0:
        rng = np.random.default_rng(seed)
        out = []
        for s in picked:
            s = s.copy()
            s.perturb(distance=rattle)  # uniform random displacement of magnitude `rattle`
            out.append(s)
        picked = out
    return picked


def _find_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """MP API key from (1) arg, (2) $MP_API_KEY, (3) the shared project-1 key file."""
    if explicit:
        return explicit.strip()
    if os.environ.get("MP_API_KEY"):
        return os.environ["MP_API_KEY"].strip()
    here = os.path.dirname(os.path.abspath(__file__))
    shared = os.path.normpath(
        os.path.join(here, "..", "..", "project1_screening", "mp_api_key.txt")
    )
    if os.path.exists(shared):
        return open(shared).read().strip()
    return None
