#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "constraint/constraint_parser.h"
#include "constraint/constraint_solver.h"
#include "constraint/constraint_engine.h"
#include "constraint/linear_constraint.h"
#include "backend/block_gemm.h"

using namespace tapp_con;

// ── Parser tests ───────────────────────────────────────

TEST_CASE("ConstraintParser — single U(1) expressions") {
    std::string err;

    SUBCASE("q0 + q1 = 0") {
        auto lc = ConstraintParser::parse_u1("q0 + q1 = 0", &err);
        CHECK(err.empty());
        CHECK(lc.target == 0);
        REQUIRE(lc.terms.size() == 2);
        CHECK(lc.terms[0].leg_index == 0);
        CHECK(lc.terms[0].coefficient == 1);
        CHECK(lc.terms[1].leg_index == 1);
        CHECK(lc.terms[1].coefficient == 1);
    }

    SUBCASE("q0 = 5") {
        auto lc = ConstraintParser::parse_u1("q0 = 5", &err);
        CHECK(err.empty());
        CHECK(lc.target == 5);
        REQUIRE(lc.terms.size() == 1);
        CHECK(lc.terms[0].leg_index == 0);
        CHECK(lc.terms[0].coefficient == 1);
    }

    SUBCASE("2*q0 + q1 - q2 = 3") {
        auto lc = ConstraintParser::parse_u1("2*q0 + q1 - q2 = 3", &err);
        CHECK(err.empty());
        CHECK(lc.target == 3);
        REQUIRE(lc.terms.size() == 3);
        CHECK(lc.terms[0].leg_index == 0);
        CHECK(lc.terms[0].coefficient == 2);
        CHECK(lc.terms[1].leg_index == 1);
        CHECK(lc.terms[1].coefficient == 1);
        CHECK(lc.terms[2].leg_index == 2);
        CHECK(lc.terms[2].coefficient == -1);
    }

    SUBCASE("malformed — missing equals") {
        auto lc = ConstraintParser::parse_u1("q0 + q1", &err);
        CHECK(!err.empty());
    }

    SUBCASE("malformed — trailing garbage") {
        auto lc = ConstraintParser::parse_u1("q0 = 5 garbage", &err);
        CHECK(!err.empty());
    }
}

TEST_CASE("ConstraintParser — full system") {
    SUBCASE("2-leg U(1) system") {
        auto result = ConstraintParser::parse_system(
            2, {"q0 + q1 = 0"}, {3, 3},
            SymmetryType::U1, -1, 1);

        CHECK(result.success);
        CHECK(result.system.u1_constraints.size() == 1);
        CHECK(result.system.leg_sectors.size() == 2);

        // Each leg should have 3 sectors (from range -1 to 1)
        CHECK(result.system.leg_sectors[0].allowed_charges.size() == 3);
        CHECK(result.system.leg_sectors[0].allowed_charges[0] == -1);
        CHECK(result.system.leg_sectors[0].allowed_charges[1] == 0);
        CHECK(result.system.leg_sectors[0].allowed_charges[2] == 1);
    }
}

// ── LinearConstraint evaluate ──────────────────────────

TEST_CASE("LinearConstraint — evaluate") {
    LinearConstraint lc;
    lc.terms = {{0, 1}, {1, 1}, {2, -1}};
    lc.target = 0;

    CHECK(lc.evaluate({1, -1, 0}));   // 1 + (-1) - 0 = 0 ✓
    CHECK(lc.evaluate({2, 1, 3}));    // 2 + 1 - 3 = 0 ✓
    CHECK_FALSE(lc.evaluate({1, 1, 0})); // 1 + 1 - 0 = 2 ≠ 0
    CHECK_FALSE(lc.evaluate({0, 0, 1})); // 0 + 0 - 1 = -1 ≠ 0
}

// ── ConstraintSolver tests ─────────────────────────────

TEST_CASE("ConstraintSolver — 2-leg U(1), q0+q1=0") {
    ConstraintSystem sys;
    sys.sym_type = SymmetryType::U1;

    // Constraint: q0 + q1 = 0
    LinearConstraint lc;
    lc.terms = {{0, 1}, {1, 1}};
    lc.target = 0;
    sys.u1_constraints.push_back(lc);

    // Leg 0: sectors q = {-1, 0, 1}
    LegSector ls0; ls0.leg_index = 0;
    ls0.allowed_charges = {-1, 0, 1};
    ls0.allowed_spins_t2 = {0, 0, 0};
    sys.leg_sectors.push_back(ls0);

    // Leg 1: sectors q = {-1, 0, 1}
    LegSector ls1; ls1.leg_index = 1;
    ls1.allowed_charges = {-1, 0, 1};
    ls1.allowed_spins_t2 = {0, 0, 0};
    sys.leg_sectors.push_back(ls1);

    // Reshape: leg0 -> rows, leg1 -> cols, each leg dim = 6
    auto blocks = ConstraintSolver::generate_blocks(
        sys, {0}, {1}, {6, 6});

    // Should produce 3 valid blocks:
    // q0=-1, q1=+1 -> off=(0,4), dim=(2,2)
    // q0= 0, q1= 0 -> off=(2,2), dim=(2,2)
    // q0=+1, q1=-1 -> off=(4,0), dim=(2,2)
    REQUIRE(blocks.size() == 3);

    // Sort by row_offset to check independently of order
    std::sort(blocks.begin(), blocks.end(),
              [](const BlockSpec& a, const BlockSpec& b) {
                  if (a.row_offset != b.row_offset) return a.row_offset < b.row_offset;
                  return a.col_offset < b.col_offset;
              });

    CHECK(blocks[0].row_offset == 0);
    CHECK(blocks[0].col_offset == 4);
    CHECK(blocks[0].num_rows == 2);
    CHECK(blocks[0].num_cols == 2);
    CHECK(blocks[0].u1_qnum == 0);

    CHECK(blocks[1].row_offset == 2);
    CHECK(blocks[1].col_offset == 2);
    CHECK(blocks[1].num_rows == 2);
    CHECK(blocks[1].num_cols == 2);
    CHECK(blocks[1].u1_qnum == 0);

    CHECK(blocks[2].row_offset == 4);
    CHECK(blocks[2].col_offset == 0);
    CHECK(blocks[2].num_rows == 2);
    CHECK(blocks[2].num_cols == 2);
    CHECK(blocks[2].u1_qnum == 0);
}

TEST_CASE("ConstraintSolver — 2-leg U(1), no valid blocks") {
    ConstraintSystem sys;
    sys.sym_type = SymmetryType::U1;

    // Constraint: q0 + q1 = 10 (impossible with small sectors)
    LinearConstraint lc;
    lc.terms = {{0, 1}, {1, 1}};
    lc.target = 10;
    sys.u1_constraints.push_back(lc);

    LegSector ls0; ls0.leg_index = 0;
    ls0.allowed_charges = {-1, 0, 1};
    ls0.allowed_spins_t2 = {0, 0, 0};
    sys.leg_sectors.push_back(ls0);

    LegSector ls1; ls1.leg_index = 1;
    ls1.allowed_charges = {-1, 0, 1};
    ls1.allowed_spins_t2 = {0, 0, 0};
    sys.leg_sectors.push_back(ls1);

    auto blocks = ConstraintSolver::generate_blocks(sys, {0}, {1}, {6, 6});
    CHECK(blocks.empty());
}

TEST_CASE("ConstraintSolver — satisfies_constraints") {
    ConstraintSystem sys;
    sys.sym_type = SymmetryType::U1;

    LinearConstraint lc;
    lc.terms = {{0, 2}, {1, -1}};  // 2*q0 - q1 = target
    lc.target = 0;
    sys.u1_constraints.push_back(lc);

    CHECK(ConstraintSolver::satisfies_constraints(sys, {1, 2}, {0, 0}));  // 2*1 - 2 = 0 ✓
    CHECK(ConstraintSolver::satisfies_constraints(sys, {0, 0}, {0, 0}));  // 0 ✓
    CHECK_FALSE(ConstraintSolver::satisfies_constraints(sys, {1, 1}, {0, 0})); // 2 - 1 = 1 ≠ 0
}

// ── ConstraintEngine tests ─────────────────────────────

TEST_CASE("ConstraintEngine — build_tensor from strings") {
    // Build a 2-leg tensor: leg0 -> rows, leg1 -> cols
    // Constraint: q0 + q1 = 0
    // 3 sectors per leg, charge range [-1, 1], leg dims = 6
    auto tensor = ConstraintEngine::build_tensor_from_strings(
        2,                         // n_legs
        {"q0 + q1 = 0"},           // constraints
        {3, 3},                    // sector sizes
        {0},                       // row legs
        {1},                       // col legs
        {6, 6},                    // leg dimensions
        DataType::FLOAT,
        SymmetryType::U1,
        -1, 1);

    CHECK(tensor.total_rows() == 6);
    CHECK(tensor.total_cols() == 6);
    CHECK(tensor.num_blocks() == 3);
    CHECK(tensor.validate());
    CHECK(tensor.buffer_size() == 3 * (2 * 2) * 4); // 3 blocks, each 2x2 float
}

TEST_CASE("ConstraintEngine — end-to-end: build + GEMM") {
    // Build two 6x6 tensors from constraints, multiply them
    // Both use U(1) with q0+q1=0, giving 3 valid 2x2 blocks

    auto A = ConstraintEngine::build_tensor_from_strings(
        2, {"q0 + q1 = 0"}, {3, 3}, {0}, {1}, {6, 6},
        DataType::FLOAT, SymmetryType::U1, -1, 1);

    auto B = ConstraintEngine::build_tensor_from_strings(
        2, {"q0 + q1 = 0"}, {3, 3}, {0}, {1}, {6, 6},
        DataType::FLOAT, SymmetryType::U1, -1, 1);

    auto C = ConstraintEngine::build_tensor_from_strings(
        2, {"q0 + q1 = 0"}, {3, 3}, {0}, {1}, {6, 6},
        DataType::FLOAT, SymmetryType::U1, -1, 1);

    // Fill blocks with simple data: identity-like per block
    for (size_t bi = 0; bi < A.num_blocks(); ++bi) {
        const auto& blk = A.block(bi);
        float* buf = static_cast<float*>(A.raw_buffer()) + blk.data_offset / 4;
        for (uint64_t r = 0; r < blk.num_rows; ++r) {
            for (uint64_t c = 0; c < blk.num_cols; ++c) {
                buf[r * blk.num_cols + c] = 1.0f; // all ones
            }
        }
    }
    for (size_t bi = 0; bi < B.num_blocks(); ++bi) {
        const auto& blk = B.block(bi);
        float* buf = static_cast<float*>(B.raw_buffer()) + blk.data_offset / 4;
        for (uint64_t r = 0; r < blk.num_rows; ++r) {
            for (uint64_t c = 0; c < blk.num_cols; ++c) {
                buf[r * blk.num_cols + c] = (r == c) ? 2.0f : 0.0f; // 2*identity
            }
        }
    }

    // GEMM: C = A @ B (with beta=0)
    BlockGEMM::multiply(C, A, B, 1.0, 0.0, false, false);

    // Each block of C should be A_block @ B_block = all-ones @ 2*identity = all-2s
    for (size_t bi = 0; bi < C.num_blocks(); ++bi) {
        const auto& blk = C.block(bi);
        const float* buf = static_cast<const float*>(C.raw_buffer()) + blk.data_offset / 4;
        for (uint64_t r = 0; r < blk.num_rows; ++r) {
            for (uint64_t c = 0; c < blk.num_cols; ++c) {
                CHECK(buf[r * blk.num_cols + c] == 2.0f);
            }
        }
    }
}
