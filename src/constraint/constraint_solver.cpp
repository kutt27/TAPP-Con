#include "constraint_solver.h"
#include "symmetry/u1_handler.h"
#include "symmetry/su2_handler.h"
#include <algorithm>
#include <functional>
#include <set>

namespace tapp_con {

bool LinearConstraint::evaluate(const std::vector<int32_t>& leg_qnums) const {
    int64_t sum = 0;
    for (const auto& term : terms) {
        if (term.leg_index >= static_cast<int>(leg_qnums.size()))
            return false;
        sum += static_cast<int64_t>(term.coefficient) * leg_qnums[term.leg_index];
    }
    return static_cast<int64_t>(target) == sum;
}

bool ConstraintSolver::satisfies_constraints(
    const ConstraintSystem& system,
    const std::vector<int32_t>& leg_u1_qnums,
    const std::vector<int32_t>& leg_su2_spins_t2)
{
    // Check all U(1) constraints
    for (const auto& lc : system.u1_constraints) {
        if (!lc.evaluate(leg_u1_qnums))
            return false;
    }

    // SU(2) constraints are handled implicitly via fusion rules
    // when the block pair is matched during GEMM.
    // For single-tensor generation, we just need valid spin values per leg.

    return true;
}

void ConstraintSolver::backtrack(
    const ConstraintSystem& system,
    size_t leg_idx,
    std::vector<int32_t>& current_u1,
    std::vector<int32_t>& current_su2,
    const std::vector<int>& row_legs,
    const std::vector<int>& col_legs,
    const std::vector<uint64_t>& leg_dims,
    std::vector<BlockSpec>& out_blocks)
{
    if (leg_idx >= system.leg_sectors.size()) {
        // All legs assigned — check constraints
        if (!satisfies_constraints(system, current_u1, current_su2))
            return;

        // Compute block position from leg dimensions
        uint64_t row_offset = 0;
        uint64_t col_offset = 0;
        uint64_t num_rows = 1;
        uint64_t num_cols = 1;

        for (int rl : row_legs) {
            // Find sector index for this leg to determine offset
            // Actually, row_offset is the accumulated dim of all PREVIOUS sectors for this leg.
            // Simpler approach: sectors are indexed by their position within the leg.
            // For now, all sectors for a leg have the same dimension, so we can compute
            // offsets based on which sector assignment we chose.
            // But the backtracking doesn't track which sector index was chosen for each leg.
            // We need to store the sector index per leg.
        }

        // For uniform per-leg dimensions, each sector has the same size.
        // The block offset in the matrix is determined by the number of
        // preceding valid assignments for each leg.
        // This requires tracking the sector ordering within each leg.

        // We'll compute this by looking at which sectors are valid within each leg
        // and computing offsets based on position within the sector list.
        // For a simpler approach in Phase 2, we handle this differently below.
        // For now, just compute the fused U(1) charge for the block.
        int32_t fused_u1 = 0;
        for (int32_t q : current_u1) {
            fused_u1 += q;
        }

        // Compute block dimensions and offsets using the sector index
        // We need to track which sector index was chosen for each leg
        // Let's compute it here
        // ...

        // For now, we compute block specs in generate_blocks which has access to this info
        // This function is called from generate_blocks which passes leg_dimensions
        // We need the sector indices to compute offsets - let me restructure

        // Actually, the backtrack needs to also track the sector index within each leg,
        // not just the quantum number. Let's store the assignment as (sector_index, charge).
        // For now, generate_blocks will handle the logic differently.

        out_blocks.push_back(BlockSpec{
            row_offset, col_offset, num_rows, num_cols, fused_u1, 0
        });
        return;
    }

    const auto& sector = system.leg_sectors[leg_idx];
    if (sector.allowed_charges.empty() && sector.allowed_spins_t2.empty())
        return;

    for (size_t si = 0; si < sector.allowed_charges.size(); ++si) {
        current_u1[leg_idx] = sector.allowed_charges[si];
        current_su2[leg_idx] = (si < sector.allowed_spins_t2.size())
            ? sector.allowed_spins_t2[si]
            : 0;
        backtrack(system, leg_idx + 1, current_u1, current_su2,
                  row_legs, col_legs, leg_dims, out_blocks);
    }
}

std::vector<BlockSpec> ConstraintSolver::generate_blocks(
    const ConstraintSystem& system,
    const std::vector<int>& row_legs,
    const std::vector<int>& col_legs,
    const std::vector<uint64_t>& leg_dims)
{
    std::vector<BlockSpec> blocks;
    size_t n_legs = system.leg_sectors.size();
    if (n_legs == 0) return blocks;

    // Phase 2 approach: enumerate all sector combinations via the backtracking,
    // but track sector indices alongside quantum numbers for offset computation.
    //
    // Simplified approach:
    // For each leg, determine the valid (charge, spin) pairs and their sector index.
    // The sector index determines the offset within the matrix dimension for that leg.
    //
    // For uniform per-leg dimensions: the physical dimension of a leg is divided
    // equally among its sectors.

    // Build ordered sector lists per leg
    struct SectorEntry {
        int32_t charge;
        int32_t spin_t2;
    };
    std::vector<std::vector<SectorEntry>> leg_entries(n_legs);
    for (size_t l = 0; l < n_legs; ++l) {
        const auto& ls = system.leg_sectors[l];
        size_t n = std::max(ls.allowed_charges.size(), ls.allowed_spins_t2.size());
        if (n == 0) return blocks; // no sectors for a leg -> no blocks possible
        for (size_t s = 0; s < n; ++s) {
            leg_entries[l].push_back({
                s < ls.allowed_charges.size() ? ls.allowed_charges[s] : 0,
                s < ls.allowed_spins_t2.size() ? ls.allowed_spins_t2[s] : 0
            });
        }
    }

    // Compute per-leg sector dimension (uniform: total_dim / num_sectors)
    std::vector<uint64_t> sector_dim(n_legs);
    for (size_t l = 0; l < n_legs; ++l) {
        uint64_t total_dim = (l < leg_dims.size()) ? leg_dims[l] : 1;
        sector_dim[l] = total_dim / std::max((uint64_t)1, (uint64_t)leg_entries[l].size());
    }

    // Compute stride per leg for row/col offset calculation
    // Row offsets: for each row leg, its sector contributes sector_dim * sector_index
    //              times the product of ALL OTHER row leg dimensions
    // Similarly for col legs.

    // Enumerate all sector combinations (backtracking with sector indices)
    std::vector<int> sector_indices(n_legs, 0);

    // Recursive helper
    std::function<void(size_t)> enumerate = [&](size_t leg) {
        if (leg == n_legs) {
            // Gather quantum numbers for this combination
            std::vector<int32_t> u1_vals(n_legs);
            std::vector<int32_t> su2_vals(n_legs);
            for (size_t l = 0; l < n_legs; ++l) {
                u1_vals[l] = leg_entries[l][sector_indices[l]].charge;
                su2_vals[l] = leg_entries[l][sector_indices[l]].spin_t2;
            }

            // Check constraints
            if (!satisfies_constraints(system, u1_vals, su2_vals))
                return;

            // Compute row offset and num_rows
            uint64_t r_off = 0;
            uint64_t r_dim = 1;
            for (int rl : row_legs) {
                uint64_t stride = sector_dim[rl];
                // Multiply by product of dimensions of OTHER row legs
                for (int rl2 : row_legs) {
                    if (rl2 != rl) stride *= sector_dim[rl2];
                }
                r_off += sector_indices[rl] * stride;
                r_dim *= sector_dim[rl];
            }
            // Normalize stride calculation by dividing by own sector dim
            // Actually, simpler: row_offset = Σ_{rl in row_legs} sector_index[rl] * product of all row leg dims excluding rl
            if (row_legs.size() <= 1) {
                // Single leg: offset is sector_index * sector_dim
                if (!row_legs.empty()) {
                    r_off = sector_indices[row_legs[0]] * sector_dim[row_legs[0]];
                    r_dim = sector_dim[row_legs[0]];
                }
            }
            // For multi-leg rows, we need proper contiguous layout
            // Row-major: slowest varying = first leg
            if (row_legs.size() > 1) {
                r_off = 0;
                r_dim = 1;
                uint64_t cum_stride = 1;
                // Traverse in reverse order (last leg varies fastest)
                for (int ri = (int)row_legs.size() - 1; ri >= 0; --ri) {
                    int rl = row_legs[ri];
                    r_off += sector_indices[rl] * cum_stride * sector_dim[rl];
                    r_dim *= sector_dim[rl];
                    cum_stride *= leg_dims[rl] / leg_entries[rl].size(); // actually sector_dim already stored per-leg dim / n_sectors
                }
                // Hmm, let me redo this more carefully
                r_off = 0;
                r_dim = 1;
                for (int rl : row_legs) {
                    r_dim *= sector_dim[rl];
                }
                // For offset: most-significant leg (first in list) has stride = product of all subsequent sector dims
                uint64_t stride = r_dim;
                for (int rl : row_legs) {
                    stride /= sector_dim[rl]; // stride = product of OTHER row sector dims
                    r_off += sector_indices[rl] * stride * sector_dim[rl];
                    // Actually: offset = sector_index * (sector_dim) * (product of other row leg sector dims)
                }
                // Let me just recompute simply:
                r_off = 0;
                uint64_t stride_mult = 1;
                for (int ri = (int)row_legs.size() - 1; ri >= 0; --ri) {
                    int rl = row_legs[ri];
                    r_off += sector_indices[rl] * sector_dim[rl] * stride_mult;
                    stride_mult *= sector_dim[rl];
                }
            }

            // Compute col offset and num_cols
            uint64_t c_off = 0;
            uint64_t c_dim = 1;
            uint64_t stride_mult_c = 1;
            for (int ci = (int)col_legs.size() - 1; ci >= 0; --ci) {
                int cl = col_legs[ci];
                c_off += sector_indices[cl] * sector_dim[cl] * stride_mult_c;
                c_dim *= sector_dim[cl];
                stride_mult_c *= sector_dim[cl];
            }

            // Fused quantum number for the block: sum of all U(1) charges
            int32_t fused_u1 = 0;
            for (auto q : u1_vals) fused_u1 += q;

            blocks.push_back(BlockSpec{
                r_off, c_off, r_dim, c_dim, fused_u1, u1_vals[0]
            });
            return;
        }

        for (size_t s = 0; s < leg_entries[leg].size(); ++s) {
            sector_indices[leg] = static_cast<int>(s);
            enumerate(leg + 1);
        }
    };

    enumerate(0);

    // Sort blocks by (row_offset, col_offset) for cache-friendly iteration
    std::sort(blocks.begin(), blocks.end(),
              [](const BlockSpec& a, const BlockSpec& b) {
                  if (a.row_offset != b.row_offset) return a.row_offset < b.row_offset;
                  return a.col_offset < b.col_offset;
              });

    return blocks;
}

} // namespace tapp_con
