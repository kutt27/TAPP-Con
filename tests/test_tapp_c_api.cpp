#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "tapp_constrained.h"

TEST_CASE("C-API — tensor lifecycle") {
    tapp_constrained_tensor_t* t = nullptr;
    tapp_constrained_status_t s;

    s = tapp_constrained_tensor_create(&t, TAPP_CON_DOUBLE, 4, 4, TAPP_CON_SYMMETRY_U1);
    REQUIRE(s == TAPP_CON_SUCCESS);
    REQUIRE(t != nullptr);

    uint64_t num_blocks;
    s = tapp_constrained_tensor_num_blocks(t, &num_blocks);
    CHECK(s == TAPP_CON_SUCCESS);
    CHECK(num_blocks == 0);

    s = tapp_constrained_tensor_destroy(t);
    CHECK(s == TAPP_CON_SUCCESS);
}

TEST_CASE("C-API — null/invalid parameters") {
    tapp_constrained_tensor_t* t = nullptr;
    tapp_constrained_status_t s;

    // null tensor pointer
    s = tapp_constrained_tensor_create(nullptr, TAPP_CON_FLOAT, 4, 4, TAPP_CON_SYMMETRY_NONE);
    CHECK(s == TAPP_CON_INVALID_PARAMETER);

    // zero dimension
    s = tapp_constrained_tensor_create(&t, TAPP_CON_FLOAT, 0, 4, TAPP_CON_SYMMETRY_NONE);
    CHECK(s == TAPP_CON_INVALID_PARAMETER);
    CHECK(t == nullptr);

    // destroy null
    s = tapp_constrained_tensor_destroy(nullptr);
    CHECK(s == TAPP_CON_INVALID_TENSOR);
}

TEST_CASE("C-API — add and get block") {
    tapp_constrained_tensor_t* t = nullptr;
    tapp_constrained_tensor_create(&t, TAPP_CON_FLOAT, 4, 4, TAPP_CON_SYMMETRY_U1);
    REQUIRE(t != nullptr);

    float block_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    tapp_constrained_status_t s;

    s = tapp_constrained_tensor_add_block(t, 0, 0, 2, 2, 0, 0, block_data);
    CHECK(s == TAPP_CON_SUCCESS);

    uint64_t num;
    tapp_constrained_tensor_num_blocks(t, &num);
    CHECK(num == 1);

    // Retrieve block
    uint64_t row_off, col_off, blk_rows, blk_cols;
    int32_t u1_q, su2_s;
    const void* data_ptr;
    s = tapp_constrained_tensor_get_block(t, 0, &row_off, &col_off, &blk_rows, &blk_cols,
                                           &u1_q, &su2_s, &data_ptr);
    CHECK(s == TAPP_CON_SUCCESS);
    CHECK(row_off == 0);
    CHECK(col_off == 0);
    CHECK(blk_rows == 2);
    CHECK(blk_cols == 2);
    CHECK(u1_q == 0);
    CHECK(data_ptr != nullptr);

    const float* fdata = static_cast<const float*>(data_ptr);
    CHECK(fdata[0] == 1.0f);
    CHECK(fdata[3] == 4.0f);

    tapp_constrained_tensor_destroy(t);
}

TEST_CASE("C-API — validation") {
    tapp_constrained_tensor_t* t = nullptr;
    tapp_constrained_tensor_create(&t, TAPP_CON_FLOAT, 4, 4, TAPP_CON_SYMMETRY_NONE);
    REQUIRE(t != nullptr);

    float data[4] = {1,2,3,4};

    // Non-overlapping
    tapp_constrained_tensor_add_block(t, 0, 0, 2, 2, 0, 0, data);
    tapp_constrained_tensor_add_block(t, 2, 2, 2, 2, 0, 0, data);
    CHECK(tapp_constrained_tensor_validate(t) == TAPP_CON_SUCCESS);

    // Add overlapping block
    tapp_constrained_tensor_add_block(t, 1, 1, 2, 2, 0, 0, data);
    CHECK(tapp_constrained_tensor_validate(t) == TAPP_CON_BLOCK_OVERLAP);

    tapp_constrained_tensor_destroy(t);
}

TEST_CASE("C-API — end-to-end GEMM") {
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]], C = A @ B = [[19,22],[43,50]]
    tapp_constrained_tensor_t *A = nullptr, *B = nullptr, *C = nullptr;

    tapp_constrained_tensor_create(&A, TAPP_CON_FLOAT, 2, 2, TAPP_CON_SYMMETRY_NONE);
    tapp_constrained_tensor_create(&B, TAPP_CON_FLOAT, 2, 2, TAPP_CON_SYMMETRY_NONE);
    tapp_constrained_tensor_create(&C, TAPP_CON_FLOAT, 2, 2, TAPP_CON_SYMMETRY_NONE);

    float dataA[4] = {1,2,3,4};
    float dataB[4] = {5,6,7,8};

    tapp_constrained_tensor_add_block(A, 0, 0, 2, 2, 0, 0, dataA);
    tapp_constrained_tensor_add_block(B, 0, 0, 2, 2, 0, 0, dataB);
    tapp_constrained_tensor_add_block(C, 0, 0, 2, 2, 0, 0, nullptr);

    tapp_constrained_plan_t* plan = nullptr;
    tapp_constrained_plan_create(&plan, A, B, C, TAPP_CON_FLOAT, 0, 0);
    REQUIRE(plan != nullptr);

    tapp_constrained_status_t s = tapp_constrained_gemm(plan, 1.0, 0.0);
    CHECK(s == TAPP_CON_SUCCESS);

    // Verify result
    const void* data_ptr;
    tapp_constrained_tensor_get_block(C, 0, nullptr, nullptr, nullptr, nullptr,
                                       nullptr, nullptr, &data_ptr);
    const float* res = static_cast<const float*>(data_ptr);
    CHECK(res[0] == 19.0f);
    CHECK(res[1] == 22.0f);
    CHECK(res[2] == 43.0f);
    CHECK(res[3] == 50.0f);

    tapp_constrained_plan_destroy(plan);
    tapp_constrained_tensor_destroy(A);
    tapp_constrained_tensor_destroy(B);
    tapp_constrained_tensor_destroy(C);
}

TEST_CASE("C-API — error strings") {
    CHECK(tapp_constrained_error_string(TAPP_CON_SUCCESS) != nullptr);
    CHECK(tapp_constrained_error_string(TAPP_CON_INVALID_TENSOR) != nullptr);
    CHECK(tapp_constrained_error_string(TAPP_CON_MEMORY_ERROR) != nullptr);
    CHECK(tapp_constrained_error_string(static_cast<tapp_constrained_status_t>(-99)) != nullptr);
}
