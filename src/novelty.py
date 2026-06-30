"""
Uniqueness + novelty for generated candidates (Project 3, W10).

A *generative* project lives or dies on whether the outputs are actually new -- a model
that re-draws Li6PS5Cl is useless. So beyond stability we report:
  - uniqueness : collapse duplicate generations among themselves (StructureMatcher).
  - novelty    : flag candidates that do NOT match any known MP phase in the chemsys.
Combined with the e_above_hull filter from ``stability.py`` this is the standard
**S.U.N.** lens (Stable, Unique, Novel) that MatterGen itself reports.

Pure pymatgen -- runs anywhere. import-only.
"""
from __future__ import annotations

from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

# Default matcher tolerances: pymatgen's defaults are tuned for "same material up to
# DFT/relaxation noise", which is what we want for both de-dup and novelty.
def _matcher() -> StructureMatcher:
    return StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)


def dedupe(structures: list[Structure]) -> tuple[list[int], list[list[int]]]:
    """Group structurally-equivalent candidates. Returns (representatives, groups):
    ``representatives`` are the indices to keep (one per distinct structure, first wins);
    ``groups`` lists the member indices of each kept representative."""
    sm = _matcher()
    reps: list[int] = []
    groups: list[list[int]] = []
    for i, s in enumerate(structures):
        placed = False
        for gi, rep in enumerate(reps):
            if structures[rep].composition.reduced_formula == s.composition.reduced_formula \
               and sm.fit(structures[rep], s):
                groups[gi].append(i)
                placed = True
                break
        if not placed:
            reps.append(i)
            groups.append([i])
    return reps, groups


def is_novel(structure: Structure, reference_structures: list[Structure]) -> bool:
    """True if ``structure`` matches NO reference (a structure unseen in MP). Compares
    only against references of the same reduced formula first (cheap), then structurally."""
    sm = _matcher()
    f = structure.composition.reduced_formula
    for ref in reference_structures:
        if ref.composition.reduced_formula == f and sm.fit(ref, structure):
            return False
    return True


def reference_structures_from_entries(reference_entries: list) -> list[Structure]:
    """Pull the Structure out of each MP ComputedStructureEntry (for novelty matching)."""
    return [e.structure for e in reference_entries]
