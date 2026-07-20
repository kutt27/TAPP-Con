#ifndef TAPP_CON_BLOCK_GEMM_H
#define TAPP_CON_BLOCK_GEMM_H

#include "constrained_tensor.h"
#include <vector>

namespace tapp_con {

struct BlockTriple {
    size_t a_idx;
    size_t b_idx;
    size_t c_idx;
};

class GpuBackend;  // forward decl for GPU path

struct BlockGEMM {
    /// Serial block-sparse GEMM (Phase 1, backward-compatible).
    static void multiply(
        ConstrainedTensor& C,
        const ConstrainedTensor& A,
        const ConstrainedTensor& B,
        double alpha,
        double beta,
        bool transA,
        bool transB);

    /// Parallel block-sparse GEMM using pre-computed block compatibility map.
    static void multiply_parallel(
        ConstrainedTensor& C,
        const ConstrainedTensor& A,
        const ConstrainedTensor& B,
        const std::vector<BlockTriple>& triples,
        const std::vector<std::vector<BlockTriple>>& c_block_work,
        double alpha,
        double beta,
        bool transA,
        bool transB);

    /// GPU-accelerated block-sparse GEMM using cuBLAS with async streams.
    /// Requires TAPP_CON_USE_CUDA.
    static void multiply_gpu(
        ConstrainedTensor& C,
        const ConstrainedTensor& A,
        const ConstrainedTensor& B,
        const std::vector<BlockTriple>& triples,
        const std::vector<std::vector<BlockTriple>>& c_block_work,
        double alpha,
        double beta,
        GpuBackend& gpu);
};

} // namespace tapp_con

#endif // TAPP_CON_BLOCK_GEMM_H
