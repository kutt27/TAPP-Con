"""Constraint engine: parser + solver + tensor construction."""

from .engine import build_tensor, build_tensor_from_strings
from .parser import (
    ConstraintSystem,
    ConstraintTerm,
    LegSector,
    LinearConstraint,
    ParseResult,
    Token,
    TokenType,
    parse_system,
    parse_u1,
    tokenize,
)
from .solver import BlockSpec, generate_blocks, satisfies_constraints

__all__ = [
    "BlockSpec",
    "ConstraintSystem",
    "ConstraintTerm",
    "LegSector",
    "LinearConstraint",
    "ParseResult",
    "Token",
    "TokenType",
    "build_tensor",
    "build_tensor_from_strings",
    "generate_blocks",
    "parse_system",
    "parse_u1",
    "satisfies_constraints",
    "tokenize",
]
