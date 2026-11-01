"""tapp_con — a NumPy port of the TAPP-Con constrained block-sparse GEMM library.

Faithful to the C++ implementation in this repository, including its documented
quirks (see ``python/README.md``). Exposes both a ``ConstrainedTensor`` class and
module-level functions mirroring the C API in ``include/tapp_constrained.h``.
"""

from ._types import DataType, Status, SymmetryType, data_type_size, numpy_dtype
from .api import (
    TAPP_CON_BACKEND_ERROR,
    TAPP_CON_BLOCK_OVERLAP,
    TAPP_CON_COMPLEX_DOUBLE,
    TAPP_CON_COMPLEX_FLOAT,
    TAPP_CON_DOUBLE,
    TAPP_CON_FLOAT,
    TAPP_CON_INCOMPATIBLE_SHAPES,
    TAPP_CON_INVALID_PARAMETER,
    TAPP_CON_INVALID_PLAN,
    TAPP_CON_INVALID_TENSOR,
    TAPP_CON_MEMORY_ERROR,
    TAPP_CON_SUCCESS,
    TAPP_CON_SYMMETRY_NONE,
    TAPP_CON_SYMMETRY_SU2,
    TAPP_CON_SYMMETRY_U1,
    TAPP_CON_SYMMETRY_U1XSU2,
    build_from_constraints,
    error_string,
    gemm,
    plan_create,
    plan_destroy,
    tensor_add_block,
    tensor_create,
    tensor_destroy,
    tensor_get_block,
    tensor_num_blocks,
    tensor_validate,
)
from .api import (
    tapp_constrained_build_from_constraints,
    tapp_constrained_error_string,
    tapp_constrained_gemm,
    tapp_constrained_plan_create,
    tapp_constrained_plan_destroy,
    tapp_constrained_tensor_add_block,
    tapp_constrained_tensor_create,
    tapp_constrained_tensor_destroy,
    tapp_constrained_tensor_get_block,
    tapp_constrained_tensor_num_blocks,
    tapp_constrained_tensor_validate,
)
from .backend import BlockGEMM, BlockTriple, PlanData
from .errors import (
    BackendError,
    BlockOverlapError,
    InvalidParameterError,
    OutOfRangeError,
    TappConError,
)
from .symmetry import SU2Handler, U1Handler, u1_compatible, u1_fuse, su2_compatible
from .tensor import Block, ConstrainedTensor
from .constraint import (
    ConstraintSystem,
    ConstraintTerm,
    LegSector,
    LinearConstraint,
    ParseResult,
    build_tensor,
    build_tensor_from_strings,
    generate_blocks,
    parse_system,
    parse_u1,
    satisfies_constraints,
    tokenize,
)

__version__ = "0.1.0"

__all__ = [
    # core
    "ConstrainedTensor",
    "Block",
    "BlockTriple",
    "PlanData",
    "BlockGEMM",
    # enums / types
    "DataType",
    "SymmetryType",
    "Status",
    "data_type_size",
    "numpy_dtype",
    # symmetry
    "U1Handler",
    "SU2Handler",
    "u1_compatible",
    "u1_fuse",
    "su2_compatible",
    # errors
    "TappConError",
    "InvalidParameterError",
    "OutOfRangeError",
    "BlockOverlapError",
    "BackendError",
    # C-API mirrors
    "tensor_create",
    "tensor_destroy",
    "tensor_add_block",
    "tensor_get_block",
    "tensor_num_blocks",
    "tensor_validate",
    "build_from_constraints",
    "plan_create",
    "plan_destroy",
    "gemm",
    "error_string",
    "tapp_constrained_tensor_create",
    "tapp_constrained_tensor_destroy",
    "tapp_constrained_tensor_add_block",
    "tapp_constrained_tensor_get_block",
    "tapp_constrained_tensor_num_blocks",
    "tapp_constrained_tensor_validate",
    "tapp_constrained_build_from_constraints",
    "tapp_constrained_plan_create",
    "tapp_constrained_plan_destroy",
    "tapp_constrained_gemm",
    "tapp_constrained_error_string",
    # constraint engine
    "ConstraintSystem",
    "ConstraintTerm",
    "LegSector",
    "LinearConstraint",
    "ParseResult",
    "build_tensor",
    "build_tensor_from_strings",
    "generate_blocks",
    "parse_system",
    "parse_u1",
    "satisfies_constraints",
    "tokenize",
    # C constants
    "TAPP_CON_SUCCESS",
    "TAPP_CON_INVALID_TENSOR",
    "TAPP_CON_INVALID_PLAN",
    "TAPP_CON_INCOMPATIBLE_SHAPES",
    "TAPP_CON_BLOCK_OVERLAP",
    "TAPP_CON_MEMORY_ERROR",
    "TAPP_CON_BACKEND_ERROR",
    "TAPP_CON_INVALID_PARAMETER",
    "TAPP_CON_SYMMETRY_NONE",
    "TAPP_CON_SYMMETRY_U1",
    "TAPP_CON_SYMMETRY_SU2",
    "TAPP_CON_SYMMETRY_U1XSU2",
    "TAPP_CON_FLOAT",
    "TAPP_CON_DOUBLE",
    "TAPP_CON_COMPLEX_FLOAT",
    "TAPP_CON_COMPLEX_DOUBLE",
]
