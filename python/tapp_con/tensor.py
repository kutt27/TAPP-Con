"""ConstrainedTensor — flat packed buffer + block descriptors.

Mirrors src/constrained_tensor.{h,cpp}. Blocks are appended contiguously into a
single byte buffer; a block's ``data_offset`` is a **byte** offset into that
buffer, and block payloads are dense row-major with stride ``num_cols``.

``total_rows`` / ``total_cols`` are the constructor dimensions, *not* the sum of
block extents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

from ._types import DataType, SymmetryType, data_type_size, numpy_dtype
from .errors import InvalidParameterError, OutOfRangeError


@dataclass(frozen=True)
class Block:
    """Mirrors BlockDescriptor (src/block/block_descriptor.h)."""

    row_offset: int
    col_offset: int
    num_rows: int
    num_cols: int
    u1_qnum: int
    su2_spin_times_two: int
    data_offset: int  # byte offset into the tensor buffer


ArrayLike = Union[np.ndarray, list, tuple]


class ConstrainedTensor:
    def __init__(
        self,
        total_rows: int,
        total_cols: int,
        dtype: DataType,
        symmetry: SymmetryType,
    ) -> None:
        if total_rows == 0 or total_cols == 0:
            raise InvalidParameterError("Tensor dimensions must be non-zero")
        self._total_rows = int(total_rows)
        self._total_cols = int(total_cols)
        self._dtype = DataType(dtype)
        self._symmetry = SymmetryType(symmetry)
        self._buffer = bytearray()
        self._blocks: List[Block] = []

    # ── basic accessors ────────────────────────────────

    @property
    def total_rows(self) -> int:
        return self._total_rows

    @property
    def total_cols(self) -> int:
        return self._total_cols

    @property
    def dtype(self) -> DataType:
        return self._dtype

    @property
    def symmetry_type(self) -> SymmetryType:
        return self._symmetry

    @property
    def num_blocks(self) -> int:
        return len(self._blocks)

    @property
    def blocks(self) -> Tuple[Block, ...]:
        return tuple(self._blocks)

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    @property
    def element_size(self) -> int:
        return data_type_size(self._dtype)

    @property
    def raw_buffer(self) -> memoryview:
        return memoryview(self._buffer)

    def block(self, index: int) -> Block:
        try:
            return self._blocks[index]
        except IndexError:
            raise OutOfRangeError(f"Block index {index} out of range") from None

    # ── construction ───────────────────────────────────

    def _payload_bytes(self, data: ArrayLike, count: int) -> bytes:
        arr = np.ascontiguousarray(data, dtype=numpy_dtype(self._dtype))
        arr = arr.reshape(-1)
        if arr.size != count:
            raise InvalidParameterError(
                f"Block data has {arr.size} elements, expected {count}"
            )
        return arr.tobytes()

    def add_block(
        self,
        row_offset: int,
        col_offset: int,
        block_rows: int,
        block_cols: int,
        u1_qnum: int,
        su2_spin_times_two: int,
        data: Optional[ArrayLike] = None,
    ) -> int:
        if (
            row_offset + block_rows > self._total_rows
            or col_offset + block_cols > self._total_cols
        ):
            raise OutOfRangeError("Block extends beyond tensor bounds")

        count = block_rows * block_cols
        nbytes = count * data_type_size(self._dtype)
        offset = len(self._buffer)

        if data is None:
            # std::vector<uint8_t> resize value-initializes new bytes to zero.
            self._buffer.extend(b"\x00" * nbytes)
        else:
            self._buffer.extend(self._payload_bytes(data, count))

        self._blocks.append(
            Block(
                row_offset=row_offset,
                col_offset=col_offset,
                num_rows=block_rows,
                num_cols=block_cols,
                u1_qnum=u1_qnum,
                su2_spin_times_two=su2_spin_times_two,
                data_offset=offset,
            )
        )
        return len(self._blocks) - 1

    # ── block data access ──────────────────────────────

    def get_block(self, index: int) -> Tuple[Block, np.ndarray]:
        """Return (descriptor, data) where data is a writable (rows, cols) view."""
        blk = self.block(index)
        arr = np.frombuffer(
            self._buffer,
            dtype=numpy_dtype(self._dtype),
            count=blk.num_rows * blk.num_cols,
            offset=blk.data_offset,
        ).reshape(blk.num_rows, blk.num_cols)
        return blk, arr

    def zero_buffer(self) -> None:
        self._buffer[:] = b"\x00" * len(self._buffer)

    def to_dense(self) -> np.ndarray:
        """Expand the block-sparse tensor into a dense (rows, cols) matrix."""
        out = np.zeros((self._total_rows, self._total_cols), dtype=numpy_dtype(self._dtype))
        for i in range(len(self._blocks)):
            blk, data = self.get_block(i)
            out[
                blk.row_offset : blk.row_offset + blk.num_rows,
                blk.col_offset : blk.col_offset + blk.num_cols,
            ] = data
        return out

    # ── validation ─────────────────────────────────────

    def validate(self) -> bool:
        """In-bounds + pairwise non-overlapping (half-open rectangles)."""
        for b in self._blocks:
            if b.row_offset + b.num_rows > self._total_rows:
                return False
            if b.col_offset + b.num_cols > self._total_cols:
                return False
        for i in range(len(self._blocks)):
            a = self._blocks[i]
            for j in range(i + 1, len(self._blocks)):
                b = self._blocks[j]
                overlap_rows = (
                    a.row_offset < b.row_offset + b.num_rows
                    and b.row_offset < a.row_offset + a.num_rows
                )
                overlap_cols = (
                    a.col_offset < b.col_offset + b.num_cols
                    and b.col_offset < a.col_offset + a.num_cols
                )
                if overlap_rows and overlap_cols:
                    return False
        return True

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ConstrainedTensor({self._total_rows}x{self._total_cols}, "
            f"dtype={self._dtype.name}, sym={self._symmetry.name}, "
            f"blocks={len(self._blocks)}, bytes={len(self._buffer)})"
        )
