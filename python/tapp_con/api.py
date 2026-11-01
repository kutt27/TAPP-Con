"""C-API mirrors (include/tapp_constrained.h).

Each function returns the same ``Status`` codes as the C library and, where the C
signature has an out-parameter, a ``(status, value)`` tuple. Exceptions are caught
and mapped to status codes exactly like the C wrapper's ``map_exception``.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ._types import DataType, Status, SymmetryType
from .backend import BlockGEMM, BlockTriple, PlanData
from .constraint.engine import build_tensor_from_strings
from .errors import status_for
from .tensor import Block, ConstrainedTensor

_ERROR_STRINGS = {
    Status.SUCCESS: "Success",
    Status.INVALID_TENSOR: "Invalid tensor handle",
    Status.INVALID_PLAN: "Invalid plan handle",
    Status.INCOMPATIBLE_SHAPES: "Incompatible tensor shapes",
    Status.SYMMETRY_MISMATCH: "Symmetry type mismatch",
    Status.BLOCK_OVERLAP: "Block overlap detected",
    Status.MEMORY_ERROR: "Memory allocation failed",
    Status.BACKEND_ERROR: "Backend computation error",
    Status.INVALID_PARAMETER: "Invalid parameter",
}


def error_string(status) -> str:
    try:
        return _ERROR_STRINGS[Status(status)]
    except ValueError:
        return "Unknown error"


# ── Tensor lifecycle ───────────────────────────────────


def tensor_create(
    dtype, num_rows: int, num_cols: int, symmetry
) -> Tuple[Status, Optional[ConstrainedTensor]]:
    if num_rows == 0 or num_cols == 0:
        return Status.INVALID_PARAMETER, None
    try:
        return Status.SUCCESS, ConstrainedTensor(num_rows, num_cols, dtype, symmetry)
    except Exception as exc:  # noqa: BLE001 - mirrors map_exception
        return status_for(exc), None


def tensor_destroy(tensor) -> Status:
    if tensor is None:
        return Status.INVALID_TENSOR
    return Status.SUCCESS


def tensor_add_block(
    tensor,
    row_offset: int,
    col_offset: int,
    block_rows: int,
    block_cols: int,
    u1_qnum: int,
    su2_spin_times_two: int,
    data=None,
) -> Status:
    if tensor is None:
        return Status.INVALID_TENSOR
    try:
        tensor.add_block(
            row_offset,
            col_offset,
            block_rows,
            block_cols,
            u1_qnum,
            su2_spin_times_two,
            data,
        )
        return Status.SUCCESS
    except Exception as exc:  # noqa: BLE001
        return status_for(exc)


def tensor_get_block(
    tensor, block_index: int
) -> Tuple[Status, Optional[Block], Optional[np.ndarray]]:
    if tensor is None:
        return Status.INVALID_TENSOR, None, None
    try:
        blk, data = tensor.get_block(block_index)
        return Status.SUCCESS, blk, data
    except IndexError:
        return Status.INVALID_PARAMETER, None, None
    except Exception as exc:  # noqa: BLE001
        return status_for(exc), None, None


def tensor_num_blocks(tensor) -> Tuple[Status, int]:
    if tensor is None:
        return Status.INVALID_PARAMETER, 0
    return Status.SUCCESS, tensor.num_blocks


def tensor_validate(tensor) -> Status:
    if tensor is None:
        return Status.INVALID_TENSOR
    return Status.SUCCESS if tensor.validate() else Status.BLOCK_OVERLAP


# ── Build from constraints ─────────────────────────────


def build_from_constraints(
    n_legs: int,
    u1_constraints: Sequence[str],
    sector_sizes: Sequence[int],
    row_legs: Sequence[int],
    col_legs: Sequence[int],
    leg_dimensions: Sequence[int],
    dtype=DataType.FLOAT,
    symmetry=SymmetryType.U1,
    charge_min: int = -1,
    charge_max: int = 1,
) -> Tuple[Status, Optional[ConstrainedTensor]]:
    if n_legs == 0 or not sector_sizes or not row_legs or not col_legs or not leg_dimensions:
        return Status.INVALID_PARAMETER, None
    try:
        tensor = build_tensor_from_strings(
            n_legs,
            list(u1_constraints) if u1_constraints else [],
            list(sector_sizes),
            list(row_legs),
            list(col_legs),
            list(leg_dimensions),
            dtype,
            symmetry,
            charge_min,
            charge_max,
        )
        return Status.SUCCESS, tensor
    except Exception as exc:  # noqa: BLE001
        return status_for(exc), None


# ── Plan ───────────────────────────────────────────────


def plan_create(
    a,
    b,
    c,
    dtype=DataType.FLOAT,
    transpose_a: int = 0,
    transpose_b: int = 0,
) -> Tuple[Status, Optional[PlanData]]:
    if a is None or b is None or c is None:
        return Status.INVALID_PARAMETER, None
    try:
        plan = PlanData(a, b, c, dtype, bool(transpose_a), bool(transpose_b))
        plan.compute_block_map()
        return Status.SUCCESS, plan
    except Exception as exc:  # noqa: BLE001
        return status_for(exc), None


def plan_destroy(plan) -> Status:
    if plan is None:
        return Status.INVALID_PLAN
    return Status.SUCCESS


# ── Compute ────────────────────────────────────────────


def gemm(plan, alpha: float, beta: float, threads: Optional[int] = None) -> Status:
    if plan is None:
        return Status.INVALID_PLAN
    try:
        if plan.triples_valid:
            BlockGEMM.multiply_parallel(
                plan.c,
                plan.a,
                plan.b,
                plan.compatible_triples,
                plan.c_block_work,
                alpha,
                beta,
                threads=threads,
            )
        else:
            BlockGEMM.multiply(
                plan.c, plan.a, plan.b, alpha, beta, plan.trans_a, plan.trans_b
            )
        return Status.SUCCESS
    except Exception as exc:  # noqa: BLE001
        return status_for(exc)


# ── 1:1 aliases using the C symbol names ───────────────

tapp_constrained_tensor_create = tensor_create
tapp_constrained_tensor_destroy = tensor_destroy
tapp_constrained_tensor_add_block = tensor_add_block
tapp_constrained_tensor_get_block = tensor_get_block
tapp_constrained_tensor_num_blocks = tensor_num_blocks
tapp_constrained_tensor_validate = tensor_validate
tapp_constrained_build_from_constraints = build_from_constraints
tapp_constrained_plan_create = plan_create
tapp_constrained_plan_destroy = plan_destroy
tapp_constrained_gemm = gemm
tapp_constrained_error_string = error_string

# C enum aliases
TAPP_CON_SUCCESS = Status.SUCCESS
TAPP_CON_INVALID_TENSOR = Status.INVALID_TENSOR
TAPP_CON_INVALID_PLAN = Status.INVALID_PLAN
TAPP_CON_INCOMPATIBLE_SHAPES = Status.INCOMPATIBLE_SHAPES
TAPP_CON_SYMMETRY_MISMATCH = Status.SYMMETRY_MISMATCH
TAPP_CON_BLOCK_OVERLAP = Status.BLOCK_OVERLAP
TAPP_CON_MEMORY_ERROR = Status.MEMORY_ERROR
TAPP_CON_BACKEND_ERROR = Status.BACKEND_ERROR
TAPP_CON_INVALID_PARAMETER = Status.INVALID_PARAMETER

TAPP_CON_SYMMETRY_NONE = SymmetryType.NONE
TAPP_CON_SYMMETRY_U1 = SymmetryType.U1
TAPP_CON_SYMMETRY_SU2 = SymmetryType.SU2
TAPP_CON_SYMMETRY_U1XSU2 = SymmetryType.U1XSU2

TAPP_CON_FLOAT = DataType.FLOAT
TAPP_CON_DOUBLE = DataType.DOUBLE
TAPP_CON_COMPLEX_FLOAT = DataType.COMPLEX_FLOAT
TAPP_CON_COMPLEX_DOUBLE = DataType.COMPLEX_DOUBLE
