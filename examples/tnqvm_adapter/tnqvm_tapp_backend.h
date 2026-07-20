#ifndef TNQVM_TAPP_BACKEND_H
#define TNQVM_TAPP_BACKEND_H

#include <string>
#include <vector>
#include <cstdint>

#include "tapp_constrained.h"

namespace tnqvm {

/// TNQVM backend adapter that delegates constrained tensor operations to TAPP-Con.
/// Wraps the tapp_constrained.h C-API for use in Tensor Network Quantum Virtual Machine
/// simulation workflows.
///
/// This adapter is intentionally standalone — it links against the installed tapp_con
/// library and demonstrates real-world integration for DMRG/MPS tensor contractions.
class TappConBackend {
public:
    TappConBackend();
    ~TappConBackend();

    TappConBackend(const TappConBackend&) = delete;
    TappConBackend& operator=(const TappConBackend&) = delete;

    /// Build a constrained tensor from U(1) charge constraints.
    /// The tensor has n_legs legs; row_legs/col_legs specify which legs map
    /// to matrix rows and columns for the GEMM view.
    ///
    /// @param n_legs       number of tensor legs
    /// @param constraints  U(1) constraint strings (e.g., "q0 + q1 = 0")
    /// @param sector_sizes per-leg sector counts
    /// @param row_legs     leg indices mapped to matrix rows
    /// @param col_legs     leg indices mapped to matrix cols
    /// @param leg_dims     physical dimension per leg
    /// @param dtype        element data type (float/double)
    /// @param symmetry     symmetry type (U(1), SU(2), or none)
    /// @param charge_min   minimum charge value for auto-sectors
    /// @param charge_max   maximum charge value for auto-sectors
    /// @return opaque tensor handle, or nullptr on failure
    tapp_constrained_tensor_t* create_constrained_tensor(
        uint64_t n_legs,
        const std::vector<std::string>& constraints,
        const std::vector<int>& sector_sizes,
        const std::vector<int>& row_legs,
        const std::vector<int>& col_legs,
        const std::vector<uint64_t>& leg_dims,
        tapp_constrained_datatype_t dtype,
        tapp_constrained_symmetry_t symmetry,
        int32_t charge_min = -1,
        int32_t charge_max = 1);

    /// Create a tensor with explicit block data (no constraint parsing).
    tapp_constrained_tensor_t* create_tensor(
        uint64_t rows,
        uint64_t cols,
        tapp_constrained_datatype_t dtype,
        tapp_constrained_symmetry_t symmetry);

    /// Add a block to a tensor.
    bool add_block(
        tapp_constrained_tensor_t* tensor,
        uint64_t row_offset,
        uint64_t col_offset,
        uint64_t block_rows,
        uint64_t block_cols,
        int32_t u1_qnum,
        int32_t su2_spin,
        const void* data);

    /// Execute constrained GEMM: C = alpha * A @ B + beta * C
    bool gemm(
        tapp_constrained_tensor_t* c,
        const tapp_constrained_tensor_t* a,
        const tapp_constrained_tensor_t* b,
        double alpha = 1.0,
        double beta = 0.0);

    /// Release a tensor
    void destroy_tensor(tapp_constrained_tensor_t* tensor);

    /// Validate tensor structure
    bool validate(tapp_constrained_tensor_t* tensor);

    /// Get number of blocks in a tensor
    uint64_t num_blocks(const tapp_constrained_tensor_t* tensor);

    /// Get block data (zero-copy pointer into tensor buffer)
    const void* get_block_data(const tapp_constrained_tensor_t* tensor,
                                uint64_t block_index);

    /// Last error message
    const char* last_error() const noexcept;

private:
    std::string last_error_;
    void set_error(const char* msg);
};

} // namespace tnqvm

#endif // TNQVM_TAPP_BACKEND_H
