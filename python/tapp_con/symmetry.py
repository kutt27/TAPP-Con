"""Symmetry selection rules, mirroring src/symmetry/u1_handler.cpp and
src/symmetry/su2_handler.cpp.

Note: SU(2) spins are stored doubled (spin x 2), so a spin-1/2 block has value 1.
The Clebsch-Gordan coefficients are a Phase-1 placeholder (always 1.0) in the C++
library, so no weighting is applied here either.
"""

from __future__ import annotations

from typing import List


class U1Handler:
    @staticmethod
    def compatible(q_a: int, q_b: int, q_c: int) -> bool:
        return (q_a + q_b) == q_c

    @staticmethod
    def fuse(q_a: int, q_b: int) -> int:
        return q_a + q_b


class SU2Handler:
    @staticmethod
    def compatible(s_a: int, s_b: int, s_c: int) -> bool:
        if (s_a + s_b + s_c) % 2 != 0:
            return False
        diff = abs(s_a - s_b)
        return diff <= s_c <= (s_a + s_b)

    @staticmethod
    def allowed_fusions(s_a: int, s_b: int) -> List[int]:
        diff = abs(s_a - s_b)
        return list(range(diff, s_a + s_b + 1, 2))

    @staticmethod
    def clebsch_gordan(*_args, **_kwargs) -> float:
        # Phase 1 placeholder — see su2_handler.cpp.
        return 1.0


# Module-level aliases (the C++ free functions are static class methods).
u1_compatible = U1Handler.compatible
u1_fuse = U1Handler.fuse
su2_compatible = SU2Handler.compatible
su2_allowed_fusions = SU2Handler.allowed_fusions
