#ifndef TAPP_CONSTRAINED_H
#define TAPP_CONSTRAINED_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Opaque handle types ─────────────────────────────── */
typedef struct tapp_constrained_tensor_impl tapp_constrained_tensor_t;
typedef struct tapp_constrained_plan_impl    tapp_constrained_plan_t;

/* ── Error codes ─────────────────────────────────────── */
typedef enum {
    TAPP_CON_SUCCESS              =  0,
    TAPP_CON_INVALID_TENSOR       = -1,
    TAPP_CON_INVALID_PLAN         = -2,
    TAPP_CON_INCOMPATIBLE_SHAPES  = -3,
    TAPP_CON_SYMMETRY_MISMATCH    = -4,
    TAPP_CON_BLOCK_OVERLAP        = -5,
    TAPP_CON_MEMORY_ERROR         = -6,
    TAPP_CON_BACKEND_ERROR        = -7,
    TAPP_CON_INVALID_PARAMETER    = -8
} tapp_constrained_status_t;

/* ── Enumerations ────────────────────────────────────── */
typedef enum {
    TAPP_CON_SYMMETRY_NONE   = 0,
    TAPP_CON_SYMMETRY_U1     = 1,
    TAPP_CON_SYMMETRY_SU2    = 2,
    TAPP_CON_SYMMETRY_U1XSU2 = 3
} tapp_constrained_symmetry_t;

typedef enum {
    TAPP_CON_FLOAT          = 0,
    TAPP_CON_DOUBLE         = 1,
    TAPP_CON_COMPLEX_FLOAT  = 2,
    TAPP_CON_COMPLEX_DOUBLE = 3
} tapp_constrained_datatype_t;

/* ── Tensor lifecycle ────────────────────────────────── */
tapp_constrained_status_t tapp_constrained_tensor_create(
    tapp_constrained_tensor_t** tensor,
    tapp_constrained_datatype_t dtype,
    uint64_t num_rows,
    uint64_t num_cols,
    tapp_constrained_symmetry_t symmetry);

tapp_constrained_status_t tapp_constrained_tensor_destroy(
    tapp_constrained_tensor_t* tensor);

/* ── Block management ────────────────────────────────── */
tapp_constrained_status_t tapp_constrained_tensor_add_block(
    tapp_constrained_tensor_t* tensor,
    uint64_t row_offset,
    uint64_t col_offset,
    uint64_t block_rows,
    uint64_t block_cols,
    int32_t u1_qnum,
    int32_t su2_spin_times_two,
    const void* data);

tapp_constrained_status_t tapp_constrained_tensor_get_block(
    const tapp_constrained_tensor_t* tensor,
    uint64_t block_index,
    uint64_t* row_offset,
    uint64_t* col_offset,
    uint64_t* block_rows,
    uint64_t* block_cols,
    int32_t* u1_qnum,
    int32_t* su2_spin_times_two,
    const void** data);

tapp_constrained_status_t tapp_constrained_tensor_num_blocks(
    const tapp_constrained_tensor_t* tensor,
    uint64_t* num_blocks);

/* ── Validation ──────────────────────────────────────── */
tapp_constrained_status_t tapp_constrained_tensor_validate(
    const tapp_constrained_tensor_t* tensor);

/* ── Constraint engine (Phase 2) ─────────────────────── */
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
    int32_t charge_max);

/* ── Plan creation (pre-compute block compatibility) ─── */
tapp_constrained_status_t tapp_constrained_plan_create(
    tapp_constrained_plan_t** plan,
    const tapp_constrained_tensor_t* a,
    const tapp_constrained_tensor_t* b,
    tapp_constrained_tensor_t* c,
    tapp_constrained_datatype_t dtype,
    int transpose_a,
    int transpose_b);

tapp_constrained_status_t tapp_constrained_plan_destroy(
    tapp_constrained_plan_t* plan);

/* ── Compute ─────────────────────────────────────────── */
tapp_constrained_status_t tapp_constrained_gemm(
    tapp_constrained_plan_t* plan,
    double alpha,
    double beta);

/* ── Error reporting ─────────────────────────────────── */
const char* tapp_constrained_error_string(
    tapp_constrained_status_t status);

#ifdef __cplusplus
}
#endif

#endif /* TAPP_CONSTRAINED_H */
