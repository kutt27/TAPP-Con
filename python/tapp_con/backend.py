"""Block-sparse GEMM and the precomputed plan, mirroring
src/backend/block_gemm.cpp and PlanData in src/tapp_constrained.cpp.

Contract (per block, exactly as the C++ code)::

    C = alpha * op(A) @ op(B) + beta * C

If ``beta == 0`` the whole C buffer is zeroed first. The ``beta`` term is applied
**once per contributing triple** (that is what ``multiply_parallel`` does), so for
a C block fed by several A/B pairs, ``beta != 0`` is not plain BLAS semantics.
FAITHFUL-QUIRK.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from ._types import SymmetryType
from .errors import InvalidParameterError
from .symmetry import SU2Handler, U1Handler
from .tensor import ConstrainedTensor


@dataclass(frozen=True)
class BlockTriple:
    """Indices into A/B/C block lists: C[c_idx] += A[a_idx] @ B[b_idx]."""

    a_idx: int
    b_idx: int
    c_idx: int


def _symmetry_ok(
    sym: SymmetryType, blk_a, blk_b, blk_c
) -> bool:
    bits = int(sym)
    if bits & int(SymmetryType.U1):
        if not U1Handler.compatible(blk_a.u1_qnum, blk_b.u1_qnum, blk_c.u1_qnum):
            return False
    if bits & int(SymmetryType.SU2):
        if not SU2Handler.compatible(
            blk_a.su2_spin_times_two,
            blk_b.su2_spin_times_two,
            blk_c.su2_spin_times_two,
        ):
            return False
    return True


def _accumulate(target: np.ndarray, a_op: np.ndarray, b_op: np.ndarray,
                alpha: float, beta: float) -> None:
    prod = a_op @ b_op
    if beta == 0.0:
        target[...] = alpha * prod
    else:
        target[...] = alpha * prod + beta * target


class BlockGEMM:
    @staticmethod
    def multiply(
        c: ConstrainedTensor,
        a: ConstrainedTensor,
        b: ConstrainedTensor,
        alpha: float,
        beta: float,
        trans_a: bool = False,
        trans_b: bool = False,
    ) -> None:
        """Serial scan-based GEMM (honours transpose), mirrors BlockGEMM::multiply."""
        if a.dtype != b.dtype or a.dtype != c.dtype:
            raise InvalidParameterError("Type mismatch between tensors")

        if beta == 0.0:
            c.zero_buffer()

        for i in range(a.num_blocks):
            blk_a = a.block(i)
            for j in range(b.num_blocks):
                blk_b = b.block(j)

                blk_a_rows = blk_a.num_cols if trans_a else blk_a.num_rows
                blk_a_cols = blk_a.num_rows if trans_a else blk_a.num_cols
                blk_b_rows = blk_b.num_cols if trans_b else blk_b.num_rows
                blk_b_cols = blk_b.num_rows if trans_b else blk_b.num_cols

                blk_a_r_off = blk_a.col_offset if trans_a else blk_a.row_offset
                blk_a_c_off = blk_a.row_offset if trans_a else blk_a.col_offset
                blk_b_r_off = blk_b.col_offset if trans_b else blk_b.row_offset
                blk_b_c_off = blk_b.row_offset if trans_b else blk_b.col_offset

                if blk_a_cols != blk_b_rows:
                    continue

                for k in range(c.num_blocks):
                    blk_c = c.block(k)
                    if blk_c.row_offset != blk_a_r_off or blk_c.col_offset != blk_b_c_off:
                        continue
                    if blk_c.num_rows != blk_a_rows or blk_c.num_cols != blk_b_cols:
                        continue
                    if not _symmetry_ok(a.symmetry_type, blk_a, blk_b, blk_c):
                        continue

                    _, c_view = c.get_block(k)
                    _, a_view = a.get_block(i)
                    _, b_view = b.get_block(j)
                    _accumulate(
                        c_view,
                        a_view.T if trans_a else a_view,
                        b_view.T if trans_b else b_view,
                        alpha,
                        beta,
                    )

    @staticmethod
    def multiply_parallel(
        c: ConstrainedTensor,
        a: ConstrainedTensor,
        b: ConstrainedTensor,
        triples: Sequence[BlockTriple],
        c_block_work: Sequence[Sequence[BlockTriple]],
        alpha: float,
        beta: float,
        threads: Optional[int] = None,
    ) -> None:
        """Plan-driven GEMM.

        Partitions by outer C-block index, mirroring the OpenMP loop in
        ``multiply_parallel``; each C block is written by exactly one worker, so
        there are no races. Transpose is ignored here — ``multiply_parallel``
        hardcodes ``false, false``. FAITHFUL-QUIRK.
        """
        if a.dtype != b.dtype or a.dtype != c.dtype:
            raise InvalidParameterError("Type mismatch between tensors")

        if beta == 0.0:
            c.zero_buffer()

        def work(ck: int) -> None:
            for tr in c_block_work[ck]:
                _, c_view = c.get_block(tr.c_idx)
                _, a_view = a.get_block(tr.a_idx)
                _, b_view = b.get_block(tr.b_idx)
                _accumulate(c_view, a_view, b_view, alpha, beta)

        n = len(c_block_work)
        if threads == 1 or n <= 1:
            for ck in range(n):
                work(ck)
        else:
            with ThreadPoolExecutor(max_workers=threads) as pool:
                list(pool.map(work, range(n)))


class PlanData:
    """Precomputed block compatibility map (PlanData in src/tapp_constrained.cpp)."""

    def __init__(
        self,
        a: ConstrainedTensor,
        b: ConstrainedTensor,
        c: ConstrainedTensor,
        dtype,
        trans_a: bool = False,
        trans_b: bool = False,
    ) -> None:
        self.a = a
        self.b = b
        self.c = c
        self.dtype = dtype
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.compatible_triples: List[BlockTriple] = []
        self.c_block_work: List[List[BlockTriple]] = []
        self.triples_valid = False

    def compute_block_map(self) -> None:
        a, b, c = self.a, self.b, self.c
        trans_a, trans_b = self.trans_a, self.trans_b
        self.compatible_triples = []
        self.c_block_work = []

        a_k = a.total_rows if trans_a else a.total_cols
        b_k = b.total_rows if trans_b else b.total_cols
        if a_k != b_k:
            return

        bits = int(a.symmetry_type)
        has_u1 = bool(bits & int(SymmetryType.U1))
        has_su2 = bool(bits & int(SymmetryType.SU2))

        for i in range(a.num_blocks):
            blk_a = a.block(i)
            for j in range(b.num_blocks):
                blk_b = b.block(j)

                blk_a_rows = blk_a.num_cols if trans_a else blk_a.num_rows
                blk_a_cols = blk_a.num_rows if trans_a else blk_a.num_cols
                blk_b_rows = blk_b.num_cols if trans_b else blk_b.num_rows
                blk_b_cols = blk_b.num_rows if trans_b else blk_b.num_cols

                blk_a_r_off = blk_a.col_offset if trans_a else blk_a.row_offset
                blk_a_c_off = blk_a.row_offset if trans_a else blk_a.col_offset
                blk_b_r_off = blk_b.col_offset if trans_b else blk_b.row_offset
                blk_b_c_off = blk_b.row_offset if trans_b else blk_b.col_offset

                if blk_a_cols != blk_b_rows:
                    continue

                for k in range(c.num_blocks):
                    blk_c = c.block(k)
                    if blk_c.row_offset != blk_a_r_off or blk_c.col_offset != blk_b_c_off:
                        continue
                    if blk_c.num_rows != blk_a_rows or blk_c.num_cols != blk_b_cols:
                        continue

                    if has_u1 and blk_a.u1_qnum + blk_b.u1_qnum != blk_c.u1_qnum:
                        continue
                    if has_su2:
                        diff = abs(blk_a.su2_spin_times_two - blk_b.su2_spin_times_two)
                        total = blk_a.su2_spin_times_two + blk_b.su2_spin_times_two
                        s_c = blk_c.su2_spin_times_two
                        if s_c < diff or s_c > total:
                            continue
                        if (blk_a.su2_spin_times_two + blk_b.su2_spin_times_two + s_c) % 2 != 0:
                            continue

                    self.compatible_triples.append(BlockTriple(i, j, k))
                    break

        self.c_block_work = [[] for _ in range(c.num_blocks)]
        for tr in self.compatible_triples:
            self.c_block_work[tr.c_idx].append(tr)

        self.triples_valid = True
