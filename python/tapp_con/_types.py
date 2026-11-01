"""Core enumerations and element sizes, mirroring src/constrained_tensor.h and
include/tapp_constrained.h."""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class SymmetryType(IntEnum):
    """Bitmask: bit 0 = U(1), bit 1 = SU(2). See src/constrained_tensor.h."""

    NONE = 0
    U1 = 1
    SU2 = 2
    U1XSU2 = 3


class DataType(IntEnum):
    FLOAT = 0
    DOUBLE = 1
    COMPLEX_FLOAT = 2
    COMPLEX_DOUBLE = 3


class Status(IntEnum):
    """Status codes returned by the C-API mirrors (include/tapp_constrained.h)."""

    SUCCESS = 0
    INVALID_TENSOR = -1
    INVALID_PLAN = -2
    INCOMPATIBLE_SHAPES = -3
    SYMMETRY_MISMATCH = -4
    BLOCK_OVERLAP = -5
    MEMORY_ERROR = -6
    BACKEND_ERROR = -7
    INVALID_PARAMETER = -8


_ELEMENT_SIZE = {
    DataType.FLOAT: 4,
    DataType.DOUBLE: 8,
    DataType.COMPLEX_FLOAT: 8,
    DataType.COMPLEX_DOUBLE: 16,
}

_NUMPY_DTYPE = {
    DataType.FLOAT: np.float32,
    DataType.DOUBLE: np.float64,
    DataType.COMPLEX_FLOAT: np.complex64,
    DataType.COMPLEX_DOUBLE: np.complex128,
}


def data_type_size(dtype: DataType) -> int:
    """Element size in bytes (src/constrained_tensor.h:data_type_size)."""
    return _ELEMENT_SIZE[DataType(dtype)]


def numpy_dtype(dtype: DataType):
    return _NUMPY_DTYPE[DataType(dtype)]
