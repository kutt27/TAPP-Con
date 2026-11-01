"""Constraint string parser, mirroring src/constraint/constraint_parser.cpp.

Grammar (exactly as the C++ parser accepts it)::

    <line>  := <expr> '=' <int>
    <expr>  := <term> { ('+' | '-') <term> }
    <term>  := [ <int> '*' ] q<digits>

Notes / faithful quirks:

* A coefficient **must** be followed by ``*`` — ``2q0`` and ``2 q0`` are errors.
* The right-hand side is a **non-negative** integer literal; ``q0 = -5`` is an
  error because the tokenizer emits a MINUS token there.
* Identifiers must be ``q`` followed by digits; there is no range check against
  the number of legs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Sequence, Tuple

from .._types import SymmetryType


class TokenType(Enum):
    IDENT = auto()
    NUMBER = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    EQUALS = auto()
    END = auto()


@dataclass
class Token:
    type: TokenType
    text: str = ""
    value: int = 0


@dataclass
class ConstraintTerm:
    leg_index: int
    coefficient: int


@dataclass
class LinearConstraint:
    terms: List[ConstraintTerm] = field(default_factory=list)
    target: int = 0

    def evaluate(self, leg_qnums: Sequence[int]) -> bool:
        total = 0
        for term in self.terms:
            if term.leg_index >= len(leg_qnums):
                return False
            total += term.coefficient * leg_qnums[term.leg_index]
        return total == self.target


@dataclass
class LegSector:
    leg_index: int
    allowed_charges: List[int] = field(default_factory=list)
    allowed_spins_t2: List[int] = field(default_factory=list)


@dataclass
class ConstraintSystem:
    u1_constraints: List[LinearConstraint] = field(default_factory=list)
    leg_sectors: List[LegSector] = field(default_factory=list)
    sym_type: SymmetryType = SymmetryType.NONE


@dataclass
class ParseResult:
    success: bool
    error_message: str
    system: Optional[ConstraintSystem] = None


def tokenize(expr: str) -> Tuple[Optional[List[Token]], str]:
    """Return (tokens, "") on success or (None, error_message) on failure."""
    tokens: List[Token] = []
    i = 0
    n = len(expr)
    while i < n:
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "+":
            tokens.append(Token(TokenType.PLUS, "+"))
            i += 1
            continue
        if ch == "-":
            tokens.append(Token(TokenType.MINUS, "-"))
            i += 1
            continue
        if ch == "*":
            tokens.append(Token(TokenType.STAR, "*"))
            i += 1
            continue
        if ch == "=":
            tokens.append(Token(TokenType.EQUALS, "="))
            i += 1
            continue
        if ch.isdigit():
            value = 0
            while i < n and expr[i].isdigit():
                value = value * 10 + (ord(expr[i]) - ord("0"))
                i += 1
            tokens.append(Token(TokenType.NUMBER, text=str(value), value=value))
            continue
        if ch.isalpha() or ch == "_":
            start = i
            while i < n and (expr[i].isalnum() or expr[i] == "_"):
                i += 1
            tokens.append(Token(TokenType.IDENT, text=expr[start:i]))
            continue
        return None, f"Unexpected character '{ch}'"

    tokens.append(Token(TokenType.END))
    return tokens, ""


def _parse_expr(
    tokens: List[Token], pos: int
) -> Tuple[Optional[LinearConstraint], int, str]:
    lc = LinearConstraint()
    while pos < len(tokens) and tokens[pos].type not in (TokenType.END, TokenType.EQUALS):
        sign = 1
        if tokens[pos].type == TokenType.PLUS:
            pos += 1
        elif tokens[pos].type == TokenType.MINUS:
            sign = -1
            pos += 1

        coefficient = sign
        if pos < len(tokens) and tokens[pos].type == TokenType.NUMBER:
            coefficient = sign * tokens[pos].value
            pos += 1
            if pos >= len(tokens) or tokens[pos].type != TokenType.STAR:
                return None, pos, "Expected '*' after coefficient"
            pos += 1

        if pos >= len(tokens) or tokens[pos].type != TokenType.IDENT:
            return None, pos, "Expected identifier (q<leg>)"
        ident = tokens[pos].text
        pos += 1

        if len(ident) < 2 or ident[0] != "q":
            return None, pos, "Expected identifier of form 'q<leg_index>'"
        digits = ident[1:]
        for c in digits:
            if not c.isdigit():
                return None, pos, f"Invalid leg index in '{ident}'"

        lc.terms.append(ConstraintTerm(leg_index=int(digits), coefficient=coefficient))

    return lc, pos, ""


def parse_u1(expr: str) -> Tuple[Optional[LinearConstraint], str]:
    tokens, err = tokenize(expr)
    if err:
        return None, err
    if not tokens:
        return None, "Expected '='"

    lc, pos, err = _parse_expr(tokens, 0)
    if err:
        return None, err

    if pos >= len(tokens) or tokens[pos].type != TokenType.EQUALS:
        return None, "Expected '='"
    pos += 1

    if pos >= len(tokens) or tokens[pos].type != TokenType.NUMBER:
        return None, "Expected integer target value"
    lc.target = tokens[pos].value
    pos += 1

    if pos < len(tokens) - 1 or (pos < len(tokens) and tokens[pos].type != TokenType.END):
        return None, "Unexpected trailing characters"

    return lc, ""


def _generate_charges(num: int, charge_min: int, charge_max: int) -> List[int]:
    """Two-branch sector generation (constraint_parser.cpp)."""
    if num <= 0:
        return []
    if charge_max - charge_min + 1 < num:
        step = max(1, (charge_max - charge_min) // max(1, num - 1))
        return [charge_min + i * step for i in range(num)]
    denom = max(1, num - 1)
    return [charge_min + (i * (charge_max - charge_min)) // denom for i in range(num)]


def parse_system(
    n_legs: int,
    u1_exprs: Sequence[str],
    sector_sizes: Sequence[int],
    sym_type: SymmetryType,
    charge_min: int = -1,
    charge_max: int = 1,
) -> ParseResult:
    system = ConstraintSystem(u1_constraints=[], leg_sectors=[], sym_type=SymmetryType(sym_type))

    for expr in u1_exprs:
        lc, err = parse_u1(expr)
        if err:
            return ParseResult(False, err, None)
        system.u1_constraints.append(lc)

    has_u1 = bool(int(sym_type) & int(SymmetryType.U1))
    for leg in range(n_legs):
        num = sector_sizes[leg] if leg < len(sector_sizes) else 0
        if has_u1:
            charges = _generate_charges(num, charge_min, charge_max)
        else:
            charges = [0]
        # Spins default to 0 for every charge (build_tensor_from_strings never
        # supplies spin sectors).
        spins = [0] * len(charges)
        system.leg_sectors.append(
            LegSector(leg_index=leg, allowed_charges=charges, allowed_spins_t2=spins)
        )

    return ParseResult(True, "", system)
