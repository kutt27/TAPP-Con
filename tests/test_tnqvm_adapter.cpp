#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "tnqvm_tapp_backend.h"

using namespace tnqvm;

TEST_CASE("TNQVM Adapter — create and destroy tensor") {
    TappConBackend backend;

    auto* tensor = backend.create_tensor(4, 4, TAPP_CON_FLOAT, TAPP_CON_SYMMETRY_U1);
    REQUIRE(tensor != nullptr);

    uint64_t n = backend.num_blocks(tensor);
    CHECK(n == 0);

    backend.destroy_tensor(tensor);
}

TEST_CASE("TNQVM Adapter — add block and validate") {
    TappConBackend backend;

    auto* tensor = backend.create_tensor(4, 4, TAPP_CON_FLOAT, TAPP_CON_SYMMETRY_NONE);
    REQUIRE(tensor != nullptr);

    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    bool ok = backend.add_block(tensor, 0, 0, 2, 2, 0, 0, data);
    CHECK(ok);

    ok = backend.validate(tensor);
    CHECK(ok);

    uint64_t n = backend.num_blocks(tensor);
    CHECK(n == 1);

    const void* block_data = backend.get_block_data(tensor, 0);
    REQUIRE(block_data != nullptr);
    const float* fdata = static_cast<const float*>(block_data);
    CHECK(fdata[0] == 1.0f);

    backend.destroy_tensor(tensor);
}

TEST_CASE("TNQVM Adapter — build from constraints") {
    TappConBackend backend;

    auto* tensor = backend.create_constrained_tensor(
        2,
        {"q0 + q1 = 0"},
        {3, 3},
        {0}, {1},
        {6, 6},
        TAPP_CON_FLOAT,
        TAPP_CON_SYMMETRY_U1,
        -1, 1);

    REQUIRE(tensor != nullptr);
    CHECK(backend.validate(tensor));

    uint64_t n = backend.num_blocks(tensor);
    CHECK(n == 3); // q={(-1,1), (0,0), (1,-1)}

    backend.destroy_tensor(tensor);
}

TEST_CASE("TNQVM Adapter — end-to-end GEMM") {
    TappConBackend backend;

    auto* A = backend.create_tensor(2, 2, TAPP_CON_FLOAT, TAPP_CON_SYMMETRY_NONE);
    auto* B = backend.create_tensor(2, 2, TAPP_CON_FLOAT, TAPP_CON_SYMMETRY_NONE);
    auto* C = backend.create_tensor(2, 2, TAPP_CON_FLOAT, TAPP_CON_SYMMETRY_NONE);

    REQUIRE(A != nullptr);
    REQUIRE(B != nullptr);
    REQUIRE(C != nullptr);

    float dataA[4] = {1,2,3,4};
    float dataB[4] = {5,6,7,8};

    backend.add_block(A, 0, 0, 2, 2, 0, 0, dataA);
    backend.add_block(B, 0, 0, 2, 2, 0, 0, dataB);
    backend.add_block(C, 0, 0, 2, 2, 0, 0, nullptr);

    bool ok = backend.gemm(C, A, B);
    CHECK(ok);

    const void* cdata = backend.get_block_data(C, 0);
    REQUIRE(cdata != nullptr);
    const float* res = static_cast<const float*>(cdata);

    // A @ B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
    CHECK(res[0] == 19.0f);
    CHECK(res[1] == 22.0f);
    CHECK(res[2] == 43.0f);
    CHECK(res[3] == 50.0f);

    backend.destroy_tensor(A);
    backend.destroy_tensor(B);
    backend.destroy_tensor(C);
}

TEST_CASE("TNQVM Adapter — error handling") {
    TappConBackend backend;
    const char* err = backend.last_error();
    CHECK(err != nullptr);

    // Null tensor operations should set error
    uint64_t n = backend.num_blocks(nullptr);
    CHECK(n == 0);
    CHECK(backend.last_error() != nullptr);
}

TEST_CASE("TNQVM Adapter — constrained GEMM end-to-end") {
    TappConBackend backend;

    auto* A = backend.create_constrained_tensor(
        2, {"q0 + q1 = 0"}, {3, 3}, {0}, {1}, {6, 6},
        TAPP_CON_FLOAT, TAPP_CON_SYMMETRY_U1, -1, 1);
    auto* B = backend.create_constrained_tensor(
        2, {"q0 + q1 = 0"}, {3, 3}, {0}, {1}, {6, 6},
        TAPP_CON_FLOAT, TAPP_CON_SYMMETRY_U1, -1, 1);
    auto* C = backend.create_constrained_tensor(
        2, {"q0 + q1 = 0"}, {3, 3}, {0}, {1}, {6, 6},
        TAPP_CON_FLOAT, TAPP_CON_SYMMETRY_U1, -1, 1);

    REQUIRE(A != nullptr);
    REQUIRE(B != nullptr);
    REQUIRE(C != nullptr);
    CHECK(backend.num_blocks(A) == 3);
    CHECK(backend.num_blocks(B) == 3);
    CHECK(backend.num_blocks(C) == 3);

    bool ok = backend.gemm(C, A, B);
    CHECK(ok);

    backend.destroy_tensor(A);
    backend.destroy_tensor(B);
    backend.destroy_tensor(C);
}
