"""Tensor construction from constraints, mirroring
src/constraint/constraint_engine.cpp."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from .._types import DataType, SymmetryType
from ..errors import InvalidParameterError
from ..tensor import ConstrainedTensor
from .parser import ConstraintSystem, parse_system
from .solver import generate_blocks


def build_tensor(
    system: ConstraintSystem,
    row_legs: Sequence[int],
    col_legs: Sequence[int],
    leg_dims: Sequence[int],
    dtype: DataType,
) -> ConstrainedTensor:
    total_rows = 1
    for rl in row_legs:
        if rl < len(leg_dims):
            total_rows *= leg_dims[rl]

    total_cols = 1
    for cl in col_legs:
        if cl < len(leg_dims):
            total_cols *= leg_dims[cl]

    tensor = ConstrainedTensor(total_rows, total_cols, dtype, system.sym_type)

    for spec in generate_blocks(system, row_legs, col_legs, leg_dims):
        tensor.add_block(
            spec.row_offset,
            spec.col_offset,
            spec.num_rows,
            spec.num_cols,
            spec.u1_qnum,
            spec.su2_spin_times_two,
            None,  # zero-initialized, exactly like the C++ path
        )
    return tensor


def build_tensor_from_strings(
    n_legs: int,
    u1_exprs: Sequence[str],
    sector_sizes: Sequence[int],
    row_legs: Sequence[int],
    col_legs: Sequence[int],
    leg_dimensions: Sequence[int],
    dtype: DataType = DataType.FLOAT,
    symmetry: SymmetryType = SymmetryType.U1,
    charge_min: int = -1,
    charge_max: int = 1,
) -> ConstrainedTensor:
    result = parse_system(
        n_legs, u1_exprs, sector_sizes, symmetry, charge_min, charge_max
    )
    if not result.success:
        raise InvalidParameterError(f"Constraint parsing failed: {result.error_message}")
    return build_tensor(result.system, row_legs, col_legs, leg_dimensions, dtype)
