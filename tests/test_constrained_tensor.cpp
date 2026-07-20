#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "constrained_tensor.h"

using namespace tapp_con;

TEST_CASE("ConstrainedTensor — construction") {
    SUBCASE("valid construction") {
        ConstrainedTensor t(4, 4, DataType::DOUBLE, SymmetryType::U1);
        CHECK(t.total_rows() == 4);
        CHECK(t.total_cols() == 4);
        CHECK(t.num_blocks() == 0);
        CHECK(t.buffer_size() == 0);
        CHECK(t.element_size() == 8);
        CHECK(t.symmetry_type() == SymmetryType::U1);
        CHECK(t.data_type() == DataType::DOUBLE);
        CHECK(t.validate()); // empty tensor is valid
    }

    SUBCASE("zero dimensions throw") {
        CHECK_THROWS_AS(ConstrainedTensor(0, 4, DataType::FLOAT, SymmetryType::NONE),
                        std::invalid_argument);
        CHECK_THROWS_AS(ConstrainedTensor(4, 0, DataType::FLOAT, SymmetryType::NONE),
                        std::invalid_argument);
    }

    SUBCASE("element sizes") {
        CHECK(ConstrainedTensor(1, 1, DataType::FLOAT, SymmetryType::NONE).element_size() == 4);
        CHECK(ConstrainedTensor(1, 1, DataType::DOUBLE, SymmetryType::NONE).element_size() == 8);
    }
}

TEST_CASE("ConstrainedTensor — add_block and access") {
    ConstrainedTensor t(4, 4, DataType::FLOAT, SymmetryType::U1);

    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    SUBCASE("add a single block") {
        size_t idx = t.add_block(0, 0, 2, 2, 0, 0, data);
        CHECK(idx == 0);
        CHECK(t.num_blocks() == 1);

        const auto& blk = t.block(0);
        CHECK(blk.row_offset == 0);
        CHECK(blk.col_offset == 0);
        CHECK(blk.num_rows == 2);
        CHECK(blk.num_cols == 2);
        CHECK(blk.u1_qnum == 0);
        CHECK(blk.data_offset == 0);

        // Verify data was copied
        const float* buf = static_cast<const float*>(t.raw_buffer()) + blk.data_offset / 4;
        CHECK(buf[0] == 1.0f);
        CHECK(buf[1] == 2.0f);
        CHECK(buf[2] == 3.0f);
        CHECK(buf[3] == 4.0f);
    }

    SUBCASE("add multiple blocks") {
        float data_b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
        t.add_block(0, 0, 2, 2, 0, 0, data);
        t.add_block(0, 2, 2, 2, 1, 0, data_b);

        CHECK(t.num_blocks() == 2);

        const auto& b0 = t.block(0);
        const auto& b1 = t.block(1);
        CHECK(b0.num_rows == 2);
        CHECK(b1.u1_qnum == 1);
        CHECK(b1.data_offset == 4 * sizeof(float)); // 16 bytes
    }

    SUBCASE("block out of bounds throws") {
        CHECK_THROWS_AS(t.add_block(3, 3, 2, 2, 0, 0, data), std::out_of_range);
    }
}

TEST_CASE("ConstrainedTensor — validation") {
    ConstrainedTensor t(4, 4, DataType::FLOAT, SymmetryType::NONE);
    float data[4] = {1,2,3,4};

    SUBCASE("non-overlapping blocks are valid") {
        t.add_block(0, 0, 2, 2, 0, 0, data);
        t.add_block(2, 2, 2, 2, 0, 0, data);
        CHECK(t.validate());
    }

    SUBCASE("overlapping blocks are invalid") {
        t.add_block(0, 0, 3, 3, 0, 0, data);
        t.add_block(1, 1, 2, 2, 0, 0, data);
        CHECK_FALSE(t.validate());
    }

    SUBCASE("adjacent blocks are valid") {
        t.add_block(0, 0, 2, 2, 0, 0, data);
        t.add_block(0, 2, 2, 2, 0, 0, data);
        CHECK(t.validate());
    }
}

TEST_CASE("ConstrainedTensor — move semantics") {
    ConstrainedTensor t1(4, 4, DataType::FLOAT, SymmetryType::U1);
    float data[4] = {1,2,3,4};
    t1.add_block(0, 0, 2, 2, 0, 0, data);
    CHECK(t1.num_blocks() == 1);

    ConstrainedTensor t2 = std::move(t1);
    CHECK(t2.num_blocks() == 1);
    CHECK(t1.num_blocks() == 0); // moved-from state
}
