// GPU Block GEMM tests (Phase 3).
// These tests are only compiled when TAPP_CON_USE_CUDA is defined.
// On systems without CUDA, this file is not compiled.

#ifdef TAPP_CON_USE_CUDA

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include "backend/block_gemm.h"
#include "backend/gpu_backend.h"

using namespace tapp_con;

TEST_CASE("GPU GEMM — single block") {
    GpuBackend gpu;

    ConstrainedTensor A(4, 4, DataType::FLOAT, SymmetryType::NONE);
    ConstrainedTensor B(4, 4, DataType::FLOAT, SymmetryType::NONE);
    ConstrainedTensor C(4, 4, DataType::FLOAT, SymmetryType::NONE);

    float dataA[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float dataB[16] = {16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1};

    A.add_block(0, 0, 4, 4, 0, 0, dataA);
    B.add_block(0, 0, 4, 4, 0, 0, dataB);
    C.add_block(0, 0, 4, 4, 0, 0, nullptr);

    // CPU reference
    ConstrainedTensor C_ref(4, 4, DataType::FLOAT, SymmetryType::NONE);
    C_ref.add_block(0, 0, 4, 4, 0, 0, nullptr);
    BlockGEMM::multiply(C_ref, A, B, 1.0, 0.0, false, false);

    // GPU execution
    std::vector<BlockTriple> triples = {{0, 0, 0}};
    std::vector<std::vector<BlockTriple>> c_work(1);
    c_work[0].push_back({0, 0, 0});

    BlockGEMM::multiply_gpu(C, A, B, triples, c_work, 1.0, 0.0, gpu);

    const float* ref = static_cast<const float*>(C_ref.raw_buffer());
    const float* res = static_cast<const float*>(C.raw_buffer());
    for (int i = 0; i < 16; ++i) {
        CHECK(ref[i] == doctest::Approx(res[i]));
    }
}

TEST_CASE("GPU GEMM — multi-block with streams") {
    GpuBackend gpu;

    int n_blocks = 4;
    int block_dim = 2;
    int total = n_blocks * block_dim;

    ConstrainedTensor A(total, total, DataType::FLOAT, SymmetryType::NONE);
    ConstrainedTensor B(total, total, DataType::FLOAT, SymmetryType::NONE);
    ConstrainedTensor C(total, total, DataType::FLOAT, SymmetryType::NONE);
    ConstrainedTensor C_ref(total, total, DataType::FLOAT, SymmetryType::NONE);

    unsigned rng_seed = 12345;
    for (int b = 0; b < n_blocks; ++b) {
        uint64_t off = static_cast<uint64_t>(b) * block_dim;
        float dataA[4] = {
            static_cast<float>((rng_seed++ * 1103515245u + 12345u) % 100) / 10.0f,
            static_cast<float>((rng_seed++ * 1103515245u + 12345u) % 100) / 10.0f,
            static_cast<float>((rng_seed++ * 1103515245u + 12345u) % 100) / 10.0f,
            static_cast<float>((rng_seed++ * 1103515245u + 12345u) % 100) / 10.0f
        };
        float dataB[4] = {
            static_cast<float>((rng_seed++ * 1103515245u + 12345u) % 100) / 10.0f,
            static_cast<float>((rng_seed++ * 1103515245u + 12345u) % 100) / 10.0f,
            static_cast<float>((rng_seed++ * 1103515245u + 12345u) % 100) / 10.0f,
            static_cast<float>((rng_seed++ * 1103515245u + 12345u) % 100) / 10.0f
        };
        A.add_block(off, off, block_dim, block_dim, 0, 0, dataA);
        B.add_block(off, off, block_dim, block_dim, 0, 0, dataB);
        C.add_block(off, off, block_dim, block_dim, 0, 0, nullptr);
        C_ref.add_block(off, off, block_dim, block_dim, 0, 0, nullptr);
    }

    // CPU reference
    BlockGEMM::multiply(C_ref, A, B, 1.0, 0.0, false, false);

    // GPU execution with pre-computed triples
    std::vector<BlockTriple> triples;
    for (size_t i = 0; i < A.num_blocks(); ++i) triples.push_back({i, i, i});

    std::vector<std::vector<BlockTriple>> c_work(C.num_blocks());
    for (const auto& tr : triples) c_work[tr.c_idx].push_back(tr);

    BlockGEMM::multiply_gpu(C, A, B, triples, c_work, 1.0, 0.0, gpu);

    const float* ref = static_cast<const float*>(C_ref.raw_buffer());
    const float* res = static_cast<const float*>(C.raw_buffer());
    size_t n_floats = C_ref.buffer_size() / 4;
    for (size_t i = 0; i < n_floats; ++i) {
        CHECK(ref[i] == doctest::Approx(res[i]));
    }
}

#endif // TAPP_CON_USE_CUDA
