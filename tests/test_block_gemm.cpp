#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "constrained_tensor.h"
#include "backend/block_gemm.h"

#include <cmath>
#include <cstring>

using namespace tapp_con;

TEST_CASE("BlockGEMM — simple 2x2 block GEMM (float, no symmetry)") {
    // A = [[1, 2],     B = [[9, 8],
    //      [3, 4]]          [7, 6]]
    // C = A @ B = [[23, 20], [55, 48]]
    // C is 2x2, single block at (0,0)

    ConstrainedTensor A(2, 2, DataType::FLOAT, SymmetryType::NONE);
    ConstrainedTensor B(2, 2, DataType::FLOAT, SymmetryType::NONE);
    ConstrainedTensor C(2, 2, DataType::FLOAT, SymmetryType::NONE);

    float dataA[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float dataB[4] = {9.0f, 8.0f, 7.0f, 6.0f};

    A.add_block(0, 0, 2, 2, 0, 0, dataA);
    B.add_block(0, 0, 2, 2, 0, 0, dataB);
    C.add_block(0, 0, 2, 2, 0, 0, nullptr);

    BlockGEMM::multiply(C, A, B, 1.0, 1.0, false, false);

    const float* result = static_cast<const float*>(C.raw_buffer());
    CHECK(result[0] == 23.0f); // 1*9 + 2*7
    CHECK(result[1] == 20.0f); // 1*8 + 2*6
    CHECK(result[2] == 55.0f); // 3*9 + 4*7
    CHECK(result[3] == 48.0f); // 3*8 + 4*6
}

TEST_CASE("BlockGEMM — double precision") {
    ConstrainedTensor A(2, 2, DataType::DOUBLE, SymmetryType::NONE);
    ConstrainedTensor B(2, 2, DataType::DOUBLE, SymmetryType::NONE);
    ConstrainedTensor C(2, 2, DataType::DOUBLE, SymmetryType::NONE);

    double dataA[4] = {1.5, 2.5, 3.5, 4.5};
    double dataB[4] = {0.5, 1.5, 2.5, 3.5};

    A.add_block(0, 0, 2, 2, 0, 0, dataA);
    B.add_block(0, 0, 2, 2, 0, 0, dataB);
    C.add_block(0, 0, 2, 2, 0, 0, nullptr);

    BlockGEMM::multiply(C, A, B, 1.0, 0.0, false, false);

    const double* result = static_cast<const double*>(C.raw_buffer());
    CHECK(result[0] == doctest::Approx(1.5*0.5 + 2.5*2.5));
    CHECK(result[1] == doctest::Approx(1.5*1.5 + 2.5*3.5));
    CHECK(result[2] == doctest::Approx(3.5*0.5 + 4.5*2.5));
    CHECK(result[3] == doctest::Approx(3.5*1.5 + 4.5*3.5));
}

TEST_CASE("BlockGEMM — U(1) symmetry filters incompatible blocks") {
    //
    // A (4x4), 2 blocks:
    //   Block 0: q=0, rows 0-1, cols 0-1  {{1,2},{3,4}}
    //   Block 1: q=1, rows 2-3, cols 2-3  {{5,6},{7,8}}
    //
    // B (4x4), 2 blocks:
    //   Block 0: q=1, rows 0-1, cols 0-1  {{9,8},{7,6}}
    //   Block 1: q=0, rows 2-3, cols 2-3  {{5,4},{3,2}}
    //
    // U(1) selection:
    //   A.q=0 x B.q=1 -> C.q=1  ✓  (block (0,0) x (0,0) -> C[0,0])
    //   A.q=1 x B.q=0 -> C.q=1  ✓  (block (1,1) x (1,1) -> C[2,2])
    //   A.q=0 x B.q=0 -> C.q=0  ✗  (no C block with q=0)
    //   A.q=1 x B.q=1 -> C.q=2  ✗  (no C block with q=2)
    //

    ConstrainedTensor A(4, 4, DataType::FLOAT, SymmetryType::U1);
    ConstrainedTensor B(4, 4, DataType::FLOAT, SymmetryType::U1);
    ConstrainedTensor C(4, 4, DataType::FLOAT, SymmetryType::U1);

    float dataA0[4] = {1,2,3,4};
    float dataA1[4] = {5,6,7,8};
    float dataB0[4] = {9,8,7,6};
    float dataB1[4] = {5,4,3,2};

    A.add_block(0, 0, 2, 2, 0, 0, dataA0);
    A.add_block(2, 2, 2, 2, 1, 0, dataA1);
    B.add_block(0, 0, 2, 2, 1, 0, dataB0);
    B.add_block(2, 2, 2, 2, 0, 0, dataB1);

    // Output blocks: C[0,0] with q=1 for A0*B0, C[2,2] with q=1 for A1*B1
    C.add_block(0, 0, 2, 2, 1, 0, nullptr);
    C.add_block(2, 2, 2, 2, 1, 0, nullptr);

    BlockGEMM::multiply(C, A, B, 1.0, 0.0, false, false);

    const float* buf = static_cast<const float*>(C.raw_buffer());

    // C[0,0] = A[0,0] @ B[0,0] = [[1,2],[3,4]] @ [[9,8],[7,6]]
    //        = [[1*9+2*7, 1*8+2*6], [3*9+4*7, 3*8+4*6]]
    //        = [[23, 20], [55, 48]]
    CHECK(buf[0] == 23.0f);
    CHECK(buf[1] == 20.0f);
    CHECK(buf[2] == 55.0f);
    CHECK(buf[3] == 48.0f);

    // C[2,2] = A[2,2] @ B[2,2] = [[5,6],[7,8]] @ [[5,4],[3,2]]
    //        = [[5*5+6*3, 5*4+6*2], [7*5+8*3, 7*4+8*2]]
    //        = [[43, 32], [59, 44]]
    CHECK(buf[4] == 43.0f);
    CHECK(buf[5] == 32.0f);
    CHECK(buf[6] == 59.0f);
    CHECK(buf[7] == 44.0f);
}

TEST_CASE("BlockGEMM — alpha/beta scaling") {
    ConstrainedTensor A(2, 2, DataType::FLOAT, SymmetryType::NONE);
    ConstrainedTensor B(2, 2, DataType::FLOAT, SymmetryType::NONE);
    ConstrainedTensor C(2, 2, DataType::FLOAT, SymmetryType::NONE);

    float dataA[4] = {1,1,1,1};
    float dataB[4] = {1,1,1,1};
    float dataC_init[4] = {10,10,10,10};

    A.add_block(0, 0, 2, 2, 0, 0, dataA);
    B.add_block(0, 0, 2, 2, 0, 0, dataB);
    C.add_block(0, 0, 2, 2, 0, 0, dataC_init);

    // C = 2.0 * A @ B + 0.5 * C
    // A@B = [[2,2],[2,2]]
    // C = 2*[[2,2],[2,2]] + 0.5*[[10,10],[10,10]] = [[4+5, 4+5], [4+5, 4+5]] = [[9,9],[9,9]]
    BlockGEMM::multiply(C, A, B, 2.0, 0.5, false, false);

    const float* result = static_cast<const float*>(C.raw_buffer());
    CHECK(result[0] == 9.0f);
    CHECK(result[1] == 9.0f);
    CHECK(result[2] == 9.0f);
    CHECK(result[3] == 9.0f);
}

TEST_CASE("BlockGEMM — transpose A") {
    // A^T = [[1,3],[2,4]]
    // B   = [[5,6],[7,8]]
    // C = A^T @ B = [[1*5+3*7, 1*6+3*8], [2*5+4*7, 2*6+4*8]] = [[26,30],[38,44]]
    ConstrainedTensor A(2, 2, DataType::FLOAT, SymmetryType::NONE);
    ConstrainedTensor B(2, 2, DataType::FLOAT, SymmetryType::NONE);
    ConstrainedTensor C(2, 2, DataType::FLOAT, SymmetryType::NONE);

    float dataA[4] = {1,2,3,4};
    float dataB[4] = {5,6,7,8};

    A.add_block(0, 0, 2, 2, 0, 0, dataA);
    B.add_block(0, 0, 2, 2, 0, 0, dataB);
    C.add_block(0, 0, 2, 2, 0, 0, nullptr);

    BlockGEMM::multiply(C, A, B, 1.0, 0.0, true, false);

    const float* res = static_cast<const float*>(C.raw_buffer());
    CHECK(res[0] == 26.0f);
    CHECK(res[1] == 30.0f);
    CHECK(res[2] == 38.0f);
    CHECK(res[3] == 44.0f);
}
