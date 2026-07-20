#include "gpu_backend.h"

namespace tapp_con {

#ifdef TAPP_CON_USE_CUDA

struct BlockGEMM;

/// GPU-accelerated variant of multiply_parallel.
/// Delegates to GpuBackend::gemm_blocks for streamed async execution.
void BlockGEMM::multiply_gpu(
    ConstrainedTensor& C,
    const ConstrainedTensor& A,
    const ConstrainedTensor& B,
    const std::vector<BlockTriple>& triples,
    const std::vector<std::vector<BlockTriple>>& c_block_work,
    double alpha,
    double beta,
    GpuBackend& gpu)
{
    if (A.data_type() != B.data_type() || A.data_type() != C.data_type())
        throw std::invalid_argument("Type mismatch between tensors");

    // Zero C host buffer if beta == 0 so that partial D2H copies work correctly
    if (beta == 0.0) {
        std::memset(C.raw_buffer(), 0, C.buffer_size());
    }

    gpu.gemm_blocks(triples, c_block_work, C, A, B, alpha, beta);
}

#endif // TAPP_CON_USE_CUDA

} // namespace tapp_con
