#ifndef TAPP_CON_GPU_BACKEND_H
#define TAPP_CON_GPU_BACKEND_H

#include "constrained_tensor.h"
#include "block_gemm.h"

#include <vector>
#include <cstddef>

namespace tapp_con {

#ifdef TAPP_CON_USE_CUDA

/// Manages CUDA resources: streams, cuBLAS handles, pinned memory pool.
class GpuBackend {
public:
    explicit GpuBackend(int device_id = 0);
    ~GpuBackend();

    GpuBackend(const GpuBackend&) = delete;
    GpuBackend& operator=(const GpuBackend&) = delete;
    GpuBackend(GpuBackend&&) noexcept;
    GpuBackend& operator=(GpuBackend&&) noexcept;

    /// Stream pool
    cudaStream_t get_stream(int index) const;
    int num_streams() const noexcept;

    /// Pinned (page-locked) host memory
    void* pinned_alloc(size_t bytes);
    void  pinned_free(void* ptr) noexcept;

    /// Device memory
    void* device_alloc(size_t bytes);
    void  device_free(void* ptr) noexcept;

    /// Async H2D transfer
    void h2d_async(void* dst, const void* src, size_t bytes, cudaStream_t stream);

    /// Async D2H transfer
    void d2h_async(void* dst, const void* src, size_t bytes, cudaStream_t stream);

    /// Execute block-sparse GEMM across streams.
    /// Each C-block group is assigned to a stream; streams run concurrently.
    /// Blocks until all streams complete.
    void gemm_blocks(
        const std::vector<BlockTriple>& triples,
        const std::vector<std::vector<BlockTriple>>& c_block_work,
        ConstrainedTensor& C,
        const ConstrainedTensor& A,
        const ConstrainedTensor& B,
        double alpha,
        double beta);

    /// Synchronize all streams
    void sync();

private:
    int device_id_;
    void* cublas_handle_;   // cublasHandle_t (opaque to avoid CUDA headers in .h)
    std::vector<void*> streams_;  // cudaStream_t[]
    int pool_depth_;

    struct PinnedBlock {
        void* ptr;
        size_t size;
        bool in_use;
    };
    std::vector<PinnedBlock> pinned_pool_;
};

#endif // TAPP_CON_USE_CUDA

} // namespace tapp_con

#endif // TAPP_CON_GPU_BACKEND_H
