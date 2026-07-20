#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "constrained_tensor.h"
#include "backend/block_gemm.h"

#include <cstring>
#include <vector>
#include <cstdlib>

using namespace tapp_con;

/// Fill a tensor block with deterministic pseudo-random data.
static void fill_block(ConstrainedTensor& t, size_t block_idx, unsigned seed) {
    const auto& blk = t.block(block_idx);
    float* buf = static_cast<float*>(t.raw_buffer()) + blk.data_offset / 4;
    unsigned rng = seed;
    for (uint64_t i = 0; i < blk.num_rows * blk.num_cols; ++i) {
        rng = rng * 1103515245u + 12345u;
        buf[i] = static_cast<float>(rng % 100) / 10.0f;
    }
}

TEST_CASE("Threaded GEMM — single-block, serial vs parallel identical") {
    // Simple 4x4 GEMM: C = A @ B
    // A and B are full 4x4 matrices, single block each.
    ConstrainedTensor A(4, 4, DataType::FLOAT, SymmetryType::NONE);
    ConstrainedTensor B(4, 4, DataType::FLOAT, SymmetryType::NONE);
    ConstrainedTensor C_serial(4, 4, DataType::FLOAT, SymmetryType::NONE);
    ConstrainedTensor C_parallel(4, 4, DataType::FLOAT, SymmetryType::NONE);

    float dataA[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float dataB[16] = {16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1};

    A.add_block(0, 0, 4, 4, 0, 0, dataA);
    B.add_block(0, 0, 4, 4, 0, 0, dataB);
    C_serial.add_block(0, 0, 4, 4, 0, 0, nullptr);
    C_parallel.add_block(0, 0, 4, 4, 0, 0, nullptr);

    // Serial reference
    BlockGEMM::multiply(C_serial, A, B, 1.0, 0.0, false, false);

    // Parallel via pre-computed triples
    std::vector<BlockTriple> triples = {{0, 0, 0}};
    std::vector<std::vector<BlockTriple>> c_work(1);
    c_work[0].push_back({0, 0, 0});

    BlockGEMM::multiply_parallel(C_parallel, A, B, triples, c_work, 1.0, 0.0, false, false);

    const float* s = static_cast<const float*>(C_serial.raw_buffer());
    const float* p = static_cast<const float*>(C_parallel.raw_buffer());
    for (int i = 0; i < 16; ++i) {
        CHECK(s[i] == doctest::Approx(p[i]));
    }
}

TEST_CASE("Threaded GEMM — multi-block diagonal, serial vs parallel") {
    // 4 diagonal blocks, each 3x3. Total matrix 12x12.
    // A has blocks at (0,0), (3,3), (6,6), (9,9) — each 3x3
    // B has blocks at (0,0), (3,3), (6,6), (9,9) — each 3x3
    // Result: C has blocks at same positions
    int n_blocks = 4;
    int block_dim = 3;
    int total = n_blocks * block_dim;

    ConstrainedTensor A(total, total, DataType::FLOAT, SymmetryType::NONE);
    ConstrainedTensor B(total, total, DataType::FLOAT, SymmetryType::NONE);
    ConstrainedTensor C_serial(total, total, DataType::FLOAT, SymmetryType::NONE);
    ConstrainedTensor C_parallel(total, total, DataType::FLOAT, SymmetryType::NONE);

    for (int b = 0; b < n_blocks; ++b) {
        uint64_t off = static_cast<uint64_t>(b) * block_dim;
        A.add_block(off, off, block_dim, block_dim, 0, 0, nullptr);
        B.add_block(off, off, block_dim, block_dim, 0, 0, nullptr);
        C_serial.add_block(off, off, block_dim, block_dim, 0, 0, nullptr);
        C_parallel.add_block(off, off, block_dim, block_dim, 0, 0, nullptr);
        fill_block(A, b, b * 10);
        fill_block(B, b, b * 10 + 5);
    }

    // Serial reference
    BlockGEMM::multiply(C_serial, A, B, 1.0, 0.0, false, false);

    // Build pre-computed triples (diagonal: A[i] @ B[i] -> C[i])
    std::vector<BlockTriple> triples;
    for (size_t i = 0; i < A.num_blocks(); ++i) {
        triples.push_back({i, i, i});
    }

    std::vector<std::vector<BlockTriple>> c_work(C_parallel.num_blocks());
    for (const auto& tr : triples) {
        c_work[tr.c_idx].push_back(tr);
    }

    // Run parallel
    BlockGEMM::multiply_parallel(C_parallel, A, B, triples, c_work, 1.0, 0.0, false, false);

    // Compare buffer — each block pair contributes independently, so results must be identical
    const float* ref = static_cast<const float*>(C_serial.raw_buffer());
    const float* res = static_cast<const float*>(C_parallel.raw_buffer());
    size_t n_floats = C_serial.buffer_size() / 4;
    for (size_t i = 0; i < n_floats; ++i) {
        CHECK(ref[i] == doctest::Approx(res[i]));
    }
}

TEST_CASE("Threaded GEMM — multiple runs identical (race stress test)") {
    int n_blocks = 5;
    int block_dim = 2;
    int total = n_blocks * block_dim;

    ConstrainedTensor A(total, total, DataType::FLOAT, SymmetryType::NONE);
    ConstrainedTensor B(total, total, DataType::FLOAT, SymmetryType::NONE);

    for (int b = 0; b < n_blocks; ++b) {
        uint64_t off = static_cast<uint64_t>(b) * block_dim;
        A.add_block(off, off, block_dim, block_dim, 0, 0, nullptr);
        B.add_block(off, off, block_dim, block_dim, 0, 0, nullptr);
        fill_block(A, b, b * 3);
        fill_block(B, b, b * 3 + 7);
    }

    // Build triples (diagonal: A[i] @ B[i] -> C[i])
    std::vector<BlockTriple> triples;
    for (size_t i = 0; i < A.num_blocks(); ++i) {
        triples.push_back({i, i, i});
    }

    // Run 50 times, all results must be identical
    std::vector<float> reference;
    for (int trial = 0; trial < 50; ++trial) {
        ConstrainedTensor C_trial(total, total, DataType::FLOAT, SymmetryType::NONE);
        for (int b = 0; b < n_blocks; ++b) {
            uint64_t off = static_cast<uint64_t>(b) * block_dim;
            C_trial.add_block(off, off, block_dim, block_dim, 0, 0, nullptr);
        }

        std::vector<std::vector<BlockTriple>> c_work(C_trial.num_blocks());
        for (const auto& tr : triples) c_work[tr.c_idx].push_back(tr);

        BlockGEMM::multiply_parallel(C_trial, A, B, triples, c_work, 1.0, 0.0, false, false);

        const float* data = static_cast<const float*>(C_trial.raw_buffer());
        size_t n_floats = C_trial.buffer_size() / 4;

        if (trial == 0) {
            reference.assign(data, data + n_floats);
        } else {
            for (size_t i = 0; i < n_floats; ++i) {
                CHECK(data[i] == doctest::Approx(reference[i]));
            }
        }
    }
}

TEST_CASE("Threaded GEMM — pre-computed map matches serial iteration") {
    // Build tensors with U(1) symmetry, verify triples from compute_block_map
    // produce same results as serial iteration
    int n_blocks = 3;
    int block_dim = 2;
    int total = n_blocks * block_dim;

    ConstrainedTensor A(total, total, DataType::FLOAT, SymmetryType::U1);
    ConstrainedTensor B(total, total, DataType::FLOAT, SymmetryType::U1);
    ConstrainedTensor C_ref(total, total, DataType::FLOAT, SymmetryType::U1);
    ConstrainedTensor C_par(total, total, DataType::FLOAT, SymmetryType::U1);

    for (int b = 0; b < n_blocks; ++b) {
        uint64_t off = static_cast<uint64_t>(b) * block_dim;
        A.add_block(off, off, block_dim, block_dim, 0, 0, nullptr);
        B.add_block(off, off, block_dim, block_dim, 0, 0, nullptr);
        C_ref.add_block(off, off, block_dim, block_dim, 0, 0, nullptr);
        C_par.add_block(off, off, block_dim, block_dim, 0, 0, nullptr);
        fill_block(A, b, b * 2);
        fill_block(B, b, b * 2 + 1);
    }

    // Serial reference
    BlockGEMM::multiply(C_ref, A, B, 1.0, 0.0, false, false);

    // Build triples the same way PlanData::compute_block_map does
    std::vector<BlockTriple> triples;
    std::vector<std::vector<BlockTriple>> c_work(C_par.num_blocks());

    for (size_t i = 0; i < A.num_blocks(); ++i) {
        const auto& blkA = A.block(i);
        for (size_t j = 0; j < B.num_blocks(); ++j) {
            const auto& blkB = B.block(j);
            if (blkA.num_cols != blkB.num_rows) continue;
            for (size_t k = 0; k < C_par.num_blocks(); ++k) {
                const auto& blkC = C_par.block(k);
                if (blkC.row_offset != blkA.row_offset || blkC.col_offset != blkB.col_offset) continue;
                if (blkC.num_rows != blkA.num_rows || blkC.num_cols != blkB.num_cols) continue;
                // U(1) filter
                if (blkA.u1_qnum + blkB.u1_qnum != blkC.u1_qnum) continue;
                triples.push_back({i, j, k});
                c_work[k].push_back({i, j, k});
                break; // only one matching C block per (A,B) pair
            }
        }
    }

    REQUIRE(!triples.empty());

    BlockGEMM::multiply_parallel(C_par, A, B, triples, c_work, 1.0, 0.0, false, false);

    const float* ref = static_cast<const float*>(C_ref.raw_buffer());
    const float* res = static_cast<const float*>(C_par.raw_buffer());
    size_t n_floats = C_ref.buffer_size() / 4;
    for (size_t i = 0; i < n_floats; ++i) {
        CHECK(ref[i] == doctest::Approx(res[i]));
    }
}


