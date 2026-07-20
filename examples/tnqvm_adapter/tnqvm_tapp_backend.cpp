#include "tnqvm_tapp_backend.h"
#include <cstring>

namespace tnqvm {

TappConBackend::TappConBackend() = default;
TappConBackend::~TappConBackend() = default;

void TappConBackend::set_error(const char* msg) {
    last_error_ = msg ? msg : "Unknown error";
}

const char* TappConBackend::last_error() const noexcept {
    return last_error_.c_str();
}

tapp_constrained_tensor_t* TappConBackend::create_constrained_tensor(
    uint64_t n_legs,
    const std::vector<std::string>& constraints,
    const std::vector<int>& sector_sizes,
    const std::vector<int>& row_legs,
    const std::vector<int>& col_legs,
    const std::vector<uint64_t>& leg_dims,
    tapp_constrained_datatype_t dtype,
    tapp_constrained_symmetry_t symmetry,
    int32_t charge_min,
    int32_t charge_max)
{
    // Build C-style string array for the C-API
    std::vector<const char*> cstrs;
    for (const auto& s : constraints) {
        cstrs.push_back(s.c_str());
    }

    tapp_constrained_tensor_t* tensor = nullptr;
    auto status = tapp_constrained_build_from_constraints(
        &tensor,
        n_legs,
        cstrs.data(),
        cstrs.size(),
        sector_sizes.data(),
        row_legs.data(),
        row_legs.size(),
        col_legs.data(),
        col_legs.size(),
        leg_dims.data(),
        dtype,
        symmetry,
        charge_min,
        charge_max);

    if (status != TAPP_CON_SUCCESS) {
        set_error(tapp_constrained_error_string(status));
        return nullptr;
    }
    return tensor;
}

tapp_constrained_tensor_t* TappConBackend::create_tensor(
    uint64_t rows,
    uint64_t cols,
    tapp_constrained_datatype_t dtype,
    tapp_constrained_symmetry_t symmetry)
{
    tapp_constrained_tensor_t* tensor = nullptr;
    auto status = tapp_constrained_tensor_create(&tensor, dtype, rows, cols, symmetry);
    if (status != TAPP_CON_SUCCESS) {
        set_error(tapp_constrained_error_string(status));
        return nullptr;
    }
    return tensor;
}

bool TappConBackend::add_block(
    tapp_constrained_tensor_t* tensor,
    uint64_t row_offset,
    uint64_t col_offset,
    uint64_t block_rows,
    uint64_t block_cols,
    int32_t u1_qnum,
    int32_t su2_spin,
    const void* data)
{
    auto status = tapp_constrained_tensor_add_block(
        tensor, row_offset, col_offset,
        block_rows, block_cols,
        u1_qnum, su2_spin, data);
    if (status != TAPP_CON_SUCCESS) {
        set_error(tapp_constrained_error_string(status));
        return false;
    }
    return true;
}

bool TappConBackend::gemm(
    tapp_constrained_tensor_t* c,
    const tapp_constrained_tensor_t* a,
    const tapp_constrained_tensor_t* b,
    double alpha,
    double beta)
{
    // Create plan with auto-detected dtype (use double for now)
    auto dtype = TAPP_CON_DOUBLE;

    tapp_constrained_plan_t* plan = nullptr;
    auto status = tapp_constrained_plan_create(&plan, a, b, c, dtype, 0, 0);
    if (status != TAPP_CON_SUCCESS) {
        set_error(tapp_constrained_error_string(status));
        return false;
    }

    status = tapp_constrained_gemm(plan, alpha, beta);
    if (status != TAPP_CON_SUCCESS) {
        set_error(tapp_constrained_error_string(status));
        tapp_constrained_plan_destroy(plan);
        return false;
    }

    tapp_constrained_plan_destroy(plan);
    return true;
}

void TappConBackend::destroy_tensor(tapp_constrained_tensor_t* tensor) {
    if (tensor) {
        tapp_constrained_tensor_destroy(tensor);
    }
}

bool TappConBackend::validate(tapp_constrained_tensor_t* tensor) {
    if (!tensor) {
        set_error("Null tensor handle");
        return false;
    }
    auto status = tapp_constrained_tensor_validate(tensor);
    if (status != TAPP_CON_SUCCESS) {
        set_error(tapp_constrained_error_string(status));
        return false;
    }
    return true;
}

uint64_t TappConBackend::num_blocks(const tapp_constrained_tensor_t* tensor) {
    uint64_t n = 0;
    auto status = tapp_constrained_tensor_num_blocks(tensor, &n);
    if (status != TAPP_CON_SUCCESS) {
        set_error(tapp_constrained_error_string(status));
        return 0;
    }
    return n;
}

const void* TappConBackend::get_block_data(
    const tapp_constrained_tensor_t* tensor,
    uint64_t block_index)
{
    const void* data = nullptr;
    auto status = tapp_constrained_tensor_get_block(
        tensor, block_index,
        nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, &data);
    if (status != TAPP_CON_SUCCESS) {
        set_error(tapp_constrained_error_string(status));
        return nullptr;
    }
    return data;
}

} // namespace tnqvm
