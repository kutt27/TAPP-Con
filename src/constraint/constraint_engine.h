#ifndef TAPP_CON_CONSTRAINT_ENGINE_H
#define TAPP_CON_CONSTRAINT_ENGINE_H

#include <cstdint>
#include <string>
#include <vector>

#include "constrained_tensor.h"
#include "linear_constraint.h"

namespace tapp_con {

class ConstraintEngine {
public:
    /// Build a ConstrainedTensor from a parsed ConstraintSystem.
    ///
    /// @param system       the constraint system (leg sectors + U(1) constraints)
    /// @param row_legs     leg indices mapped to matrix rows
    /// @param col_legs     leg indices mapped to matrix cols
    /// @param leg_dims     physical dimension per leg
    /// @param dtype        element data type
    /// @return ConstrainedTensor with blocks generated from the constraint system,
    ///         zero-initialized block buffers
    static ConstrainedTensor build_tensor(
        const ConstraintSystem& system,
        const std::vector<int>& row_legs,
        const std::vector<int>& col_legs,
        const std::vector<uint64_t>& leg_dims,
        DataType dtype);

    /// Convenience: build from string constraints.
    static ConstrainedTensor build_tensor_from_strings(
        uint64_t n_legs,
        const std::vector<std::string>& u1_exprs,
        const std::vector<int>& sector_sizes,
        const std::vector<int>& row_legs,
        const std::vector<int>& col_legs,
        const std::vector<uint64_t>& leg_dimensions,
        DataType dtype,
        SymmetryType sym_type,
        int32_t charge_min = -1,
        int32_t charge_max = 1);
};

} // namespace tapp_con

#endif // TAPP_CON_CONSTRAINT_ENGINE_H
