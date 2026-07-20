#ifndef TAPP_CON_CONSTRAINT_PARSER_H
#define TAPP_CON_CONSTRAINT_PARSER_H

#include <cstdint>
#include <string>
#include <vector>

#include "linear_constraint.h"

namespace tapp_con {

struct ParseResult {
    ConstraintSystem system;
    std::string error_message;
    bool success = false;
};

class ConstraintParser {
public:
    /// Parse a single U(1) linear constraint string.
    /// Grammar: <expr> = <int>
    ///   expr  : term { [+|-] term }
    ///   term  : [coefficient *] q<leg_index>
    /// Example: "2*q0 + q1 - q2 = 5"
    static LinearConstraint parse_u1(const std::string& expr,
                                     std::string* error = nullptr);

    /// Parse a full constraint system for a tensor with n_legs.
    /// @param n_legs      number of tensor legs
    /// @param u1_exprs    constraint strings like {"q0 + q1 = 0", "q0 - q2 = 0"}
    /// @param sector_sizes per leg: how many distinct charge values
    /// @param sym_type    symmetry type
    /// @param charge_range (min, max) inclusive for auto-generating sector charges
    /// @param spin_sectors per-leg SU(2) spin values (empty = no SU(2) for that leg)
    static ParseResult parse_system(
        uint64_t n_legs,
        const std::vector<std::string>& u1_exprs,
        const std::vector<int>& sector_sizes,
        SymmetryType sym_type,
        int32_t charge_min = -1,
        int32_t charge_max = 1,
        const std::vector<std::vector<int32_t>>& spin_sectors = {});
};

} // namespace tapp_con

#endif // TAPP_CON_CONSTRAINT_PARSER_H
