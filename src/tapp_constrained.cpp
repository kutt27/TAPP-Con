#include "tapp_constrained.h"
#include "constrained_tensor.h"
#include "backend/block_gemm.h"
#ifdef TAPP_CON_USE_CUDA
#include "backend/gpu_backend.h"
#endif
#include "constraint/constraint_engine.h"

#include <new>
#include <stdexcept>
#include <cstring>
#include <vector>
#include <algorithm>

using namespace tapp_con;

// ── Helpers ────────────────────────────────────────────

static inline ConstrainedTensor* to_tensor(tapp_constrained_tensor_t* h) {
    return reinterpret_cast<ConstrainedTensor*>(h);
}
static inline const ConstrainedTensor* to_ctensor(const tapp_constrained_tensor_t* h) {
    return reinterpret_cast<const ConstrainedTensor*>(h);
}

// ── Plan data (pre-computed block compatibility map) ───

struct PlanData {
    const ConstrainedTensor* a;
    const ConstrainedTensor* b;
    ConstrainedTensor* c;
    DataType dtype;
    bool transA;
    bool transB;

    // Phase 2: Pre-computed execution plan
    std::vector<BlockTriple> compatible_triples;
    std::vector<std::vector<BlockTriple>> c_block_work;
    bool triples_valid = false;

    void compute_block_map() {
        compatible_triples.clear();
        c_block_work.clear();

        if (!a || !b || !c) return;

        uint64_t A_k = transA ? a->total_rows() : a->total_cols();
        uint64_t B_k = transB ? b->total_rows() : b->total_cols();
        if (A_k != B_k) return;

        bool has_u1 = (static_cast<uint8_t>(a->symmetry_type()) & static_cast<uint8_t>(SymmetryType::U1)) != 0;
        bool has_su2 = (static_cast<uint8_t>(a->symmetry_type()) & static_cast<uint8_t>(SymmetryType::SU2)) != 0;

        for (size_t i = 0; i < a->num_blocks(); ++i) {
            const auto& blkA = a->block(i);

            for (size_t j = 0; j < b->num_blocks(); ++j) {
                const auto& blkB = b->block(j);

                uint64_t blkA_rows = transA ? blkA.num_cols : blkA.num_rows;
                uint64_t blkA_cols = transA ? blkA.num_rows : blkA.num_cols;
                uint64_t blkB_rows = transB ? blkB.num_cols : blkB.num_rows;
                uint64_t blkB_cols = transB ? blkB.num_rows : blkB.num_cols;

                uint64_t blkA_r_off = transA ? blkA.col_offset : blkA.row_offset;
                uint64_t blkA_c_off = transA ? blkA.row_offset : blkA.col_offset;
                uint64_t blkB_r_off = transB ? blkB.col_offset : blkB.row_offset;
                uint64_t blkB_c_off = transB ? blkB.row_offset : blkB.col_offset;

                if (blkA_cols != blkB_rows) continue;

                for (size_t k = 0; k < c->num_blocks(); ++k) {
                    const auto& blkC = c->block(k);

                    if (blkC.row_offset != blkA_r_off || blkC.col_offset != blkB_c_off)
                        continue;
                    if (blkC.num_rows != blkA_rows || blkC.num_cols != blkB_cols)
                        continue;

                    // Symmetry filtering
                    if (has_u1) {
                        int32_t fused = blkA.u1_qnum + blkB.u1_qnum;
                        if (fused != blkC.u1_qnum) continue;
                    }
                    if (has_su2) {
                        // SU(2) compatibility — triangle inequality
                        int32_t diff = std::abs(blkA.su2_spin_times_two - blkB.su2_spin_times_two);
                        int32_t sum = blkA.su2_spin_times_two + blkB.su2_spin_times_two;
                        int32_t sC = blkC.su2_spin_times_two;
                        if (sC < diff || sC > sum) continue;
                        if ((blkA.su2_spin_times_two + blkB.su2_spin_times_two + sC) % 2 != 0) continue;
                    }

                    compatible_triples.push_back({i, j, k});
                    break;
                }
            }
        }

        // Build per-C-block work lists
        c_block_work.resize(c->num_blocks());
        for (const auto& tr : compatible_triples) {
            c_block_work[tr.c_idx].push_back(tr);
        }

        triples_valid = true;
    }
};

static inline PlanData* to_plan(tapp_constrained_plan_t* h) {
    return reinterpret_cast<PlanData*>(h);
}

static tapp_constrained_status_t map_exception() {
    try {
        throw;
    } catch (const std::bad_alloc&) {
        return TAPP_CON_MEMORY_ERROR;
    } catch (const std::invalid_argument&) {
        return TAPP_CON_INVALID_PARAMETER;
    } catch (const std::out_of_range&) {
        return TAPP_CON_INVALID_PARAMETER;
    } catch (...) {
        return TAPP_CON_BACKEND_ERROR;
    }
}

// ── Tensor lifecycle ───────────────────────────────────

tapp_constrained_status_t tapp_constrained_tensor_create(
    tapp_constrained_tensor_t** tensor,
    tapp_constrained_datatype_t dtype,
    uint64_t num_rows,
    uint64_t num_cols,
    tapp_constrained_symmetry_t symmetry)
{
    if (!tensor || num_rows == 0 || num_cols == 0)
        return TAPP_CON_INVALID_PARAMETER;

    try {
        auto* impl = new ConstrainedTensor(
            num_rows, num_cols,
            static_cast<DataType>(dtype),
            static_cast<SymmetryType>(symmetry));
        *tensor = reinterpret_cast<tapp_constrained_tensor_t*>(impl);
        return TAPP_CON_SUCCESS;
    } catch (...) {
        return map_exception();
    }
}

tapp_constrained_status_t tapp_constrained_tensor_destroy(
    tapp_constrained_tensor_t* tensor)
{
    if (!tensor) return TAPP_CON_INVALID_TENSOR;
    delete to_tensor(tensor);
    return TAPP_CON_SUCCESS;
}

// ── Block management ───────────────────────────────────

tapp_constrained_status_t tapp_constrained_tensor_add_block(
    tapp_constrained_tensor_t* tensor,
    uint64_t row_offset,
    uint64_t col_offset,
    uint64_t block_rows,
    uint64_t block_cols,
    int32_t u1_qnum,
    int32_t su2_spin_times_two,
    const void* data)
{
    if (!tensor) return TAPP_CON_INVALID_TENSOR;

    try {
        to_tensor(tensor)->add_block(
            row_offset, col_offset,
            block_rows, block_cols,
            u1_qnum, su2_spin_times_two,
            data);
        return TAPP_CON_SUCCESS;
    } catch (...) {
        return map_exception();
    }
}

tapp_constrained_status_t tapp_constrained_tensor_get_block(
    const tapp_constrained_tensor_t* tensor,
    uint64_t block_index,
    uint64_t* row_offset,
    uint64_t* col_offset,
    uint64_t* block_rows,
    uint64_t* block_cols,
    int32_t* u1_qnum,
    int32_t* su2_spin_times_two,
    const void** data)
{
    if (!tensor) return TAPP_CON_INVALID_TENSOR;

    try {
        const auto& blk = to_ctensor(tensor)->block(static_cast<size_t>(block_index));

        if (row_offset)          *row_offset = blk.row_offset;
        if (col_offset)          *col_offset = blk.col_offset;
        if (block_rows)          *block_rows = blk.num_rows;
        if (block_cols)          *block_cols = blk.num_cols;
        if (u1_qnum)             *u1_qnum = blk.u1_qnum;
        if (su2_spin_times_two)  *su2_spin_times_two = blk.su2_spin_times_two;
        if (data) {
            const auto* base = static_cast<const uint8_t*>(to_ctensor(tensor)->raw_buffer());
            *data = base + blk.data_offset;
        }

        return TAPP_CON_SUCCESS;
    } catch (const std::out_of_range&) {
        return TAPP_CON_INVALID_PARAMETER;
    } catch (...) {
        return map_exception();
    }
}

tapp_constrained_status_t tapp_constrained_tensor_num_blocks(
    const tapp_constrained_tensor_t* tensor,
    uint64_t* num_blocks)
{
    if (!tensor || !num_blocks) return TAPP_CON_INVALID_PARAMETER;
    *num_blocks = static_cast<uint64_t>(to_ctensor(tensor)->num_blocks());
    return TAPP_CON_SUCCESS;
}

// ── Validation ─────────────────────────────────────────

tapp_constrained_status_t tapp_constrained_tensor_validate(
    const tapp_constrained_tensor_t* tensor)
{
    if (!tensor) return TAPP_CON_INVALID_TENSOR;
    return to_ctensor(tensor)->validate()
        ? TAPP_CON_SUCCESS
        : TAPP_CON_BLOCK_OVERLAP;
}

// ── Build from constraints (Phase 2) ───────────────────

tapp_constrained_status_t tapp_constrained_build_from_constraints(
    tapp_constrained_tensor_t** tensor,
    uint64_t n_legs,
    const char** u1_constraints,
    uint64_t n_constraints,
    const int* sector_sizes,
    const int* row_legs,
    uint64_t n_row_legs,
    const int* col_legs,
    uint64_t n_col_legs,
    const uint64_t* leg_dimensions,
    tapp_constrained_datatype_t dtype,
    tapp_constrained_symmetry_t symmetry,
    int32_t charge_min,
    int32_t charge_max)
{
    if (!tensor || !sector_sizes || !row_legs || !col_legs || !leg_dimensions)
        return TAPP_CON_INVALID_PARAMETER;
    if (n_row_legs == 0 || n_col_legs == 0)
        return TAPP_CON_INVALID_PARAMETER;

    try {
        std::vector<std::string> constraint_strs;
        if (u1_constraints) {
            for (uint64_t i = 0; i < n_constraints; ++i) {
                if (u1_constraints[i])
                    constraint_strs.push_back(u1_constraints[i]);
            }
        }

        std::vector<int> row_legs_vec(row_legs, row_legs + n_row_legs);
        std::vector<int> col_legs_vec(col_legs, col_legs + n_col_legs);
        std::vector<int> sector_sizes_vec(sector_sizes, sector_sizes + n_legs);
        std::vector<uint64_t> leg_dims_vec(leg_dimensions, leg_dimensions + n_legs);

        auto result = ConstraintEngine::build_tensor_from_strings(
            n_legs, constraint_strs, sector_sizes_vec,
            row_legs_vec, col_legs_vec, leg_dims_vec,
            static_cast<DataType>(dtype),
            static_cast<SymmetryType>(symmetry),
            charge_min, charge_max);

        *tensor = reinterpret_cast<tapp_constrained_tensor_t*>(
            new ConstrainedTensor(std::move(result)));
        return TAPP_CON_SUCCESS;
    } catch (const std::invalid_argument& e) {
        return TAPP_CON_INVALID_PARAMETER;
    } catch (const std::bad_alloc&) {
        return TAPP_CON_MEMORY_ERROR;
    } catch (...) {
        return map_exception();
    }
}

// ── Plan creation ──────────────────────────────────────

tapp_constrained_status_t tapp_constrained_plan_create(
    tapp_constrained_plan_t** plan,
    const tapp_constrained_tensor_t* a,
    const tapp_constrained_tensor_t* b,
    tapp_constrained_tensor_t* c,
    tapp_constrained_datatype_t dtype,
    int transpose_a,
    int transpose_b)
{
    if (!plan || !a || !b || !c)
        return TAPP_CON_INVALID_PARAMETER;

    try {
        auto* pd = new PlanData{
            to_ctensor(a),
            to_ctensor(b),
            to_tensor(c),
            static_cast<DataType>(dtype),
            transpose_a != 0,
            transpose_b != 0,
            {}, {}, false
        };
        pd->compute_block_map();
        *plan = reinterpret_cast<tapp_constrained_plan_t*>(pd);
        return TAPP_CON_SUCCESS;
    } catch (...) {
        return map_exception();
    }
}

tapp_constrained_status_t tapp_constrained_plan_destroy(
    tapp_constrained_plan_t* plan)
{
    if (!plan) return TAPP_CON_INVALID_PLAN;
    delete to_plan(plan);
    return TAPP_CON_SUCCESS;
}

// ── Compute ────────────────────────────────────────────

tapp_constrained_status_t tapp_constrained_gemm(
    tapp_constrained_plan_t* plan,
    double alpha,
    double beta)
{
    if (!plan) return TAPP_CON_INVALID_PLAN;

    try {
        PlanData* pd = to_plan(plan);
        if (pd->triples_valid) {
#ifdef TAPP_CON_USE_CUDA
            static GpuBackend gpu_backend;
            BlockGEMM::multiply_gpu(
                *pd->c, *pd->a, *pd->b,
                pd->compatible_triples,
                pd->c_block_work,
                alpha, beta,
                gpu_backend);
#else
            BlockGEMM::multiply_parallel(
                *pd->c, *pd->a, *pd->b,
                pd->compatible_triples,
                pd->c_block_work,
                alpha, beta,
                pd->transA, pd->transB);
#endif
        } else {
            BlockGEMM::multiply(
                *pd->c, *pd->a, *pd->b,
                alpha, beta,
                pd->transA, pd->transB);
        }
        return TAPP_CON_SUCCESS;
    } catch (...) {
        return map_exception();
    }
}

// ── Error reporting ────────────────────────────────────

const char* tapp_constrained_error_string(
    tapp_constrained_status_t status)
{
    switch (status) {
        case TAPP_CON_SUCCESS:             return "Success";
        case TAPP_CON_INVALID_TENSOR:      return "Invalid tensor handle";
        case TAPP_CON_INVALID_PLAN:        return "Invalid plan handle";
        case TAPP_CON_INCOMPATIBLE_SHAPES: return "Incompatible tensor shapes";
        case TAPP_CON_SYMMETRY_MISMATCH:   return "Symmetry type mismatch";
        case TAPP_CON_BLOCK_OVERLAP:       return "Block overlap detected";
        case TAPP_CON_MEMORY_ERROR:        return "Memory allocation failed";
        case TAPP_CON_BACKEND_ERROR:       return "Backend computation error";
        case TAPP_CON_INVALID_PARAMETER:   return "Invalid parameter";
        default:                           return "Unknown error";
    }
}
