#ifndef TAPP_CON_LINEAR_CONSTRAINT_H
#define TAPP_CON_LINEAR_CONSTRAINT_H

#include <cstdint>
#include <vector>
#include <string>

#include "constrained_tensor.h"

namespace tapp_con {

/// A single term in a linear constraint: coefficient * q(leg_index)
struct ConstraintTerm {
    int leg_index;       // which tensor leg this term refers to
    int32_t coefficient; // typically +1 or -1, but any integer is allowed
};

/// Σ (coefficient_i * q(leg_i)) = target
struct LinearConstraint {
    std::vector<ConstraintTerm> terms;
    int32_t target;

    /// Evaluate the constraint given per-leg quantum numbers.
    bool evaluate(const std::vector<int32_t>& leg_qnums) const;
};

/// Describes the allowed quantum-number sectors for one tensor leg.
struct LegSector {
    int leg_index;
    std::vector<int32_t> allowed_charges;     // U(1) sector values
    std::vector<int32_t> allowed_spins_t2;    // SU(2) spin × 2
};

/// The complete constraint specification for a tensor.
struct ConstraintSystem {
    std::vector<LinearConstraint> u1_constraints;
    std::vector<LegSector> leg_sectors;
    SymmetryType sym_type;
};

} // namespace tapp_con

#endif // TAPP_CON_LINEAR_CONSTRAINT_H
