"""
Stability screening for generated candidates (Project 3, W10).

The crux of an inverse-design demo is the *filter*: most generated crystals are
unstable, so we relax each one with a universal MLIP and compute its energy above the
convex hull (e_above_hull), keeping only near-stable candidates.

Design decision -- a SELF-CONSISTENT MLIP hull (documented in the README):
    Both the reference phases (pulled from Materials Project for the chemical system)
    AND the generated candidates are relaxed with the *same* universal MLIP, and the
    hull is built from those MLIP energies. This avoids mixing MLIP candidate energies
    with PBE+U reference energies (different reference states -> a systematically biased
    e_above_hull). It is the same principle MatterGen's own evaluator uses (it relaxes a
    reference set consistently). The cost is relaxing the reference phases once; results
    are cached by material_id so re-runs are cheap.

Calculator-agnostic, exactly like project 2's ``md.py``: heavy deps (torch / mace /
chgnet) are imported lazily inside ``load_calculator`` so this module imports on a
laptop with only pymatgen + ASE. A toy ``lj`` calculator is provided purely to exercise
the relax -> hull -> e_above_hull plumbing on CPU (energies are meaningless -- it is the
analogue of project 2's Lennard-Jones trajectory check, never a physics result).

import-only.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.io.ase import AseAtomsAdaptor

SUPPORTED = ("mace", "chgnet", "lj")


# --------------------------------------------------------------------------- #
# Calculators (lazy heavy imports)
# --------------------------------------------------------------------------- #
def pick_device(prefer: Optional[str] = None) -> str:
    """'cuda' if available, else Apple 'mps', else 'cpu'. ``prefer`` overrides."""
    if prefer:
        return prefer
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_calculator(mlip: str, device: Optional[str] = None, mace_model: str = "medium",
                    dtype: str = "float32"):
    """ASE calculator for ``mlip`` in {'mace','chgnet','lj'}.

    - mace   : MACE-MP-0 universal potential (``mace_model`` 'small'|'medium'|'large').
    - chgnet : CHGNet universal potential.
    - lj     : ASE Lennard-Jones -- a CPU plumbing stand-in ONLY (meaningless energies).
    """
    mlip = mlip.lower()
    if mlip == "lj":
        from ase.calculators.lj import LennardJones
        return LennardJones()
    device = pick_device(device)
    if mlip == "mace":
        from mace.calculators import mace_mp
        return mace_mp(model=mace_model, dispersion=False, default_dtype=dtype, device=device)
    if mlip == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator
        return CHGNetCalculator(use_device=device)
    raise ValueError(f"unknown mlip {mlip!r}; supported: {SUPPORTED}")


# --------------------------------------------------------------------------- #
# Relaxation
# --------------------------------------------------------------------------- #
def relax(structure: Structure, calc, fmax: float = 0.05, steps: int = 200,
          relax_cell: bool = True) -> dict:
    """Relax a structure with ``calc`` (ASE FIRE). Optionally relax the cell too.

    Returns {relaxed (Structure), energy_eV (total), energy_per_atom, n_atoms,
    converged (bool), n_steps}. Never raises on non-convergence -- the caller decides
    what to do with a high-energy / unconverged candidate.
    """
    from ase.optimize import FIRE

    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = calc
    target = atoms
    if relax_cell:
        try:
            from ase.filters import FrechetCellFilter as CellFilter
        except ImportError:                       # older ASE
            from ase.constraints import ExpCellFilter as CellFilter
        target = CellFilter(atoms)

    opt = FIRE(target, logfile=None)
    converged = opt.run(fmax=fmax, steps=steps)
    e = float(atoms.get_potential_energy())
    relaxed = AseAtomsAdaptor.get_structure(atoms)
    return {
        "relaxed": relaxed,
        "energy_eV": e,
        "energy_per_atom": e / len(atoms),
        "n_atoms": len(atoms),
        "converged": bool(converged),
        "n_steps": int(opt.get_number_of_steps()),
    }


# --------------------------------------------------------------------------- #
# Reference phases from Materials Project (cached)
# --------------------------------------------------------------------------- #
def _find_api_key(explicit: Optional[str] = None) -> Optional[str]:
    from .generate import _find_api_key as f
    return f(explicit)


def fetch_reference_entries(chemsys: list[str], api_key: Optional[str] = None,
                            cache_path: Optional[str] = None) -> list:
    """All MP ComputedStructureEntries spanning ``chemsys`` (e.g. ['Li','P','S']),
    including the elemental endpoints. Cached to ``cache_path`` (pickle) so later runs
    are offline. These supply the *structures* we relax to build the MLIP hull.

    (Pickle, not JSON: MP entries carry an ``oxidation_states`` dict keyed by Element
    objects that the JSON encoder cannot serialise as dict keys.)"""
    import pickle

    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    key = _find_api_key(api_key)
    if not key:
        raise RuntimeError("No MP API key for reference phases ($MP_API_KEY or key file).")
    from mp_api.client import MPRester

    with MPRester(key) as mpr:
        entries = mpr.get_entries_in_chemsys(chemsys, inc_structure=True)
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(entries, f)
    return entries


# --------------------------------------------------------------------------- #
# Self-consistent MLIP convex hull
# --------------------------------------------------------------------------- #
def build_mlip_hull(reference_entries: list, calc, energy_cache: Optional[dict] = None,
                    fmax: float = 0.05, steps: int = 200, max_refs: Optional[int] = None,
                    verbose: bool = True) -> dict:
    """Relax every reference structure with ``calc`` and build a PhaseDiagram from the
    resulting MLIP energies (a self-consistent hull).

    ``energy_cache`` maps entry_id -> relaxed total energy (eV); pass one in (and read it
    back from the return value) to skip re-relaxing references across runs.

    Returns {pd (PhaseDiagram), entries (list[PDEntry]), energy_cache, n_refs}.
    """
    energy_cache = dict(energy_cache or {})
    refs = list(reference_entries)
    if max_refs:
        refs = refs[:max_refs]
    pd_entries = []
    for i, e in enumerate(refs):
        eid = str(getattr(e, "entry_id", None) or e.composition.reduced_formula + f"_{i}")
        if eid in energy_cache:
            energy = energy_cache[eid]
        else:
            res = relax(e.structure, calc, fmax=fmax, steps=steps, relax_cell=True)
            energy = res["energy_eV"]
            energy_cache[eid] = energy
            if verbose:
                print(f"  [hull] {eid:>16s} {e.composition.reduced_formula:>12s} "
                      f"E={energy:10.3f} eV ({i+1}/{len(refs)})")
        pd_entries.append(PDEntry(e.composition, energy, name=eid))
    pd = PhaseDiagram(pd_entries)
    return {"pd": pd, "entries": pd_entries, "energy_cache": energy_cache, "n_refs": len(pd_entries)}


def e_above_hull(structure: Structure, calc, pd: PhaseDiagram, fmax: float = 0.05,
                 steps: int = 200) -> dict:
    """Relax a candidate and return its energy above the (MLIP) hull, eV/atom.

    Returns the relax dict augmented with {e_above_hull (eV/atom), formation-style
    decomposition products}. ``allow_negative`` lets a candidate that beats the hull
    report a negative value (it would be a new ground state) instead of crashing.
    """
    r = relax(structure, calc, fmax=fmax, steps=steps, relax_cell=True)
    entry = PDEntry(r["relaxed"].composition, r["energy_eV"])
    decomp, ehull = pd.get_decomp_and_e_above_hull(entry, allow_negative=True)
    r["e_above_hull"] = float(ehull)
    r["decomp"] = {str(k.composition.reduced_formula): round(float(v), 3)
                   for k, v in decomp.items()}
    return r
