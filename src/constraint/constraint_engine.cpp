#include "constraint_engine.h"
#include "constraint_parser.h"
#include "constraint_solver.h"
#include <stdexcept>

namespace tapp_con {

ConstrainedTensor ConstraintEngine::build_tensor(
    const ConstraintSystem& system,
    const std::vector<int>& row_legs,
    const std::vector<int>& col_legs,
    const std::vector<uint64_t>& leg_dims,
    DataType dtype)
{
    // Compute total matrix dimensions from leg dims
    uint64_t total_rows = 1;
    for (int rl : row_legs) {
        if (rl < static_cast<int>(leg_dims.size()))
            total_rows *= leg_dims[rl];
    }
    uint64_t total_cols = 1;
    for (int cl : col_legs) {
        if (cl < static_cast<int>(leg_dims.size()))
            total_cols *= leg_dims[cl];
    }

    ConstrainedTensor tensor(total_rows, total_cols, dtype, system.sym_type);

    // Generate blocks from constraints
    auto block_specs = ConstraintSolver::generate_blocks(
        system, row_legs, col_legs, leg_dims);

    // Add each block as a zero-initialized buffer
    for (const auto& bs : block_specs) {
        tensor.add_block(
            bs.row_offset, bs.col_offset,
            bs.num_rows, bs.num_cols,
            bs.u1_qnum, bs.su2_spin_times_two,
            nullptr); // zero-initialized by add_block
    }

    return tensor;
}

ConstrainedTensor ConstraintEngine::build_tensor_from_strings(
    uint64_t n_legs,
    const std::vector<std::string>& u1_exprs,
    const std::vector<int>& sector_sizes,
    const std::vector<int>& row_legs,
    const std::vector<int>& col_legs,
    const std::vector<uint64_t>& leg_dimensions,
    DataType dtype,
    SymmetryType sym_type,
    int32_t charge_min,
    int32_t charge_max)
{
    auto parse_result = ConstraintParser::parse_system(
        n_legs, u1_exprs, sector_sizes, sym_type, charge_min, charge_max);

    if (!parse_result.success) {
        throw std::invalid_argument(
            "Constraint parsing failed: " + parse_result.error_message);
    }

    return build_tensor(
        parse_result.system,
        row_legs, col_legs,
        leg_dimensions, dtype);
}

} // namespace tapp_con
