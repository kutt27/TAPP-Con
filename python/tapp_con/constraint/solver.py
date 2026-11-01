"""Block enumeration, mirroring ConstraintSolver::generate_blocks
(src/constraint/constraint_solver.cpp).

The "backtracking" name in the C++ source is a leftover: the real algorithm is a
full Cartesian enumeration over each leg's sectors, filtered by the constraints.
There is no pruning.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List, Sequence

from .parser import ConstraintSystem, LinearConstraint


@dataclass
class BlockSpec:
    row_offset: int
    col_offset: int
    num_rows: int
    num_cols: int
    u1_qnum: int
    su2_spin_times_two: int


def satisfies_constraints(
    constraints: Sequence[LinearConstraint], leg_u1_qnums: Sequence[int]
) -> bool:
    for lc in constraints:
        if not lc.evaluate(leg_u1_qnums):
            return False
    return True


def _leg_entries(leg_sector):
    """Per-sector (charge, spin) pairs for one leg.

    If the charge and spin lists have different lengths the longer one sets the
    count and the shorter is zero-padded, matching the C++ behaviour.
    """
    n = max(len(leg_sector.allowed_charges), len(leg_sector.allowed_spins_t2))
    if n == 0:
        return []
    entries = []
    for s in range(n):
        charge = leg_sector.allowed_charges[s] if s < len(leg_sector.allowed_charges) else 0
        spin = leg_sector.allowed_spins_t2[s] if s < len(leg_sector.allowed_spins_t2) else 0
        entries.append((charge, spin))
    return entries


def generate_blocks(
    system: ConstraintSystem,
    row_legs: Sequence[int],
    col_legs: Sequence[int],
    leg_dims: Sequence[int],
) -> List[BlockSpec]:
    n_legs = len(system.leg_sectors)
    entries = [_leg_entries(ls) for ls in system.leg_sectors]
    if any(len(e) == 0 for e in entries):
        return []

    # Integer division: the physical leg dimension is split evenly across the
    # leg's sectors; any remainder is dropped.
    sector_dim = [leg_dims[leg] // max(1, len(entries[leg])) for leg in range(n_legs)]

    blocks: List[BlockSpec] = []
    ranges = [range(len(entries[leg])) for leg in range(n_legs)]

    for combo in itertools.product(*ranges):
        u1_vals = [entries[leg][combo[leg]][0] for leg in range(n_legs)]

        if not satisfies_constraints(system.u1_constraints, u1_vals):
            continue

        # Row offset / height over the row legs, traversed last -> first.
        # FAITHFUL-QUIRK: the running multiplier grows by sector_dim[leg] of the
        # current leg (constraint_solver.cpp:233-250) rather than by the product
        # of the already-visited legs' sizes; reproduce as-is.
        r_off = 0
        stride = 1
        for rl in reversed(list(row_legs)):
            r_off += combo[rl] * sector_dim[rl] * stride
            stride *= sector_dim[rl]
        r_dim = stride

        c_off = 0
        stride = 1
        for cl in reversed(list(col_legs)):
            c_off += combo[cl] * sector_dim[cl] * stride
            stride *= sector_dim[cl]
        c_dim = stride

        fused_u1 = sum(u1_vals)
        # FAITHFUL-QUIRK: the C++ code stores u1_vals[0] in the SU(2) spin field
        # (constraint_solver.cpp:253-257). Reproduce exactly.
        blocks.append(
            BlockSpec(
                row_offset=r_off,
                col_offset=c_off,
                num_rows=r_dim,
                num_cols=c_dim,
                u1_qnum=fused_u1,
                su2_spin_times_two=u1_vals[0],
            )
        )

    blocks.sort(key=lambda b: (b.row_offset, b.col_offset))
    return blocks
