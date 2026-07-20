#include "gpu_backend.h"

namespace tapp_con {

#ifdef TAPP_CON_USE_CUDA

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <algorithm>

GpuBackend::GpuBackend(int device_id)
    : device_id_(device_id)
    , cublas_handle_(nullptr)
    , pool_depth_(3)
{
    cudaSetDevice(device_id_);

    // Create cuBLAS handle
    cublasHandle_t handle;
    auto status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Failed to create cuBLAS handle");
    cublas_handle_ = handle;

    // Create stream pool
    streams_.resize(pool_depth_);
    for (auto& s : streams_) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        s = stream;
    }
}

GpuBackend::~GpuBackend() {
    if (cublas_handle_) {
        cublasDestroy(static_cast<cublasHandle_t>(cublas_handle_));
    }
    for (auto& s : streams_) {
        if (s) cudaStreamDestroy(static_cast<cudaStream_t>(s));
    }
    for (auto& pb : pinned_pool_) {
        if (pb.ptr) cudaFreeHost(pb.ptr);
    }
}

GpuBackend::GpuBackend(GpuBackend&& other) noexcept
    : device_id_(other.device_id_)
    , cublas_handle_(other.cublas_handle_)
    , streams_(std::move(other.streams_))
    , pool_depth_(other.pool_depth_)
    , pinned_pool_(std::move(other.pinned_pool_))
{
    other.cublas_handle_ = nullptr;
    other.streams_.clear();
}

GpuBackend& GpuBackend::operator=(GpuBackend&& other) noexcept {
    if (this != &other) {
        device_id_ = other.device_id_;
        cublas_handle_ = other.cublas_handle_;
        streams_ = std::move(other.streams_);
        pool_depth_ = other.pool_depth_;
        pinned_pool_ = std::move(other.pinned_pool_);
        other.cublas_handle_ = nullptr;
        other.streams_.clear();
    }
    return *this;
}

cudaStream_t GpuBackend::get_stream(int index) const {
    return static_cast<cudaStream_t>(streams_[index % pool_depth_]);
}

int GpuBackend::num_streams() const noexcept {
    return pool_depth_;
}

void* GpuBackend::pinned_alloc(size_t bytes) {
    // Search for available block in pool
    for (auto& pb : pinned_pool_) {
        if (!pb.in_use && pb.size >= bytes) {
            pb.in_use = true;
            return pb.ptr;
        }
    }
    // Allocate new pinned memory
    void* ptr = nullptr;
    auto err = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
    if (err != cudaSuccess)
        throw std::bad_alloc();
    pinned_pool_.push_back({ptr, bytes, true});
    return ptr;
}

void GpuBackend::pinned_free(void* ptr) noexcept {
    for (auto& pb : pinned_pool_) {
        if (pb.ptr == ptr) {
            pb.in_use = false;
            return;
        }
    }
}

void* GpuBackend::device_alloc(size_t bytes) {
    void* ptr = nullptr;
    auto err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess)
        throw std::bad_alloc();
    return ptr;
}

void GpuBackend::device_free(void* ptr) noexcept {
    if (ptr) cudaFree(ptr);
}

void GpuBackend::h2d_async(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
}

void GpuBackend::d2h_async(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream);
}

void GpuBackend::sync() {
    cudaDeviceSynchronize();
}

void GpuBackend::gemm_blocks(
    const std::vector<BlockTriple>& triples,
    const std::vector<std::vector<BlockTriple>>& c_block_work,
    ConstrainedTensor& C,
    const ConstrainedTensor& A,
    const ConstrainedTensor& B,
    double alpha,
    double beta)
{
    if (A.data_type() != B.data_type() || A.data_type() != C.data_type())
        throw std::invalid_argument("Type mismatch between tensors");

    DataType dtype = A.data_type();
    size_t elem_sz = data_type_size(dtype);

    // Zero out C on device
    if (beta == 0.0) {
        // We'll handle this by beta=0 in cuBLAS calls
    }

    auto hdl = static_cast<cublasHandle_t>(cublas_handle_);

    // Assign work to streams: each C-block group gets a stream
    size_t n_c_blocks = c_block_work.size();
    int n_streams = pool_depth_;

    for (size_t ck = 0; ck < n_c_blocks; ++ck) {
        int sidx = static_cast<int>(ck) % n_streams;
        auto stream = static_cast<cudaStream_t>(streams_[sidx]);
        cublasSetStream(hdl, stream);

        for (const auto& tr : c_block_work[ck]) {
            const auto& blkA = A.block(tr.a_idx);
            const auto& blkB = B.block(tr.b_idx);
            const auto& blkC = C.block(tr.c_idx);

            size_t a_bytes = blkA.num_rows * blkA.num_cols * elem_sz;
            size_t b_bytes = blkB.num_rows * blkB.num_cols * elem_sz;
            size_t c_bytes = blkC.num_rows * blkC.num_cols * elem_sz;

            // Pinned host pointers
            const void* h_a = static_cast<const uint8_t*>(A.raw_buffer()) + blkA.data_offset;
            const void* h_b = static_cast<const uint8_t*>(B.raw_buffer()) + blkB.data_offset;
            void* h_c = static_cast<uint8_t*>(C.raw_buffer()) + blkC.data_offset;

            // Device buffers (allocate per-block for simplicity; could reuse)
            void* d_a = device_alloc(a_bytes);
            void* d_b = device_alloc(b_bytes);
            void* d_c = device_alloc(c_bytes);

            // Pipeline: H2D → GEMM → D2H
            h2d_async(d_a, h_a, a_bytes, stream);
            h2d_async(d_b, h_b, b_bytes, stream);

            // Initialize device C to zero if beta=0
            if (beta == 0.0) {
                cudaMemsetAsync(d_c, 0, c_bytes, stream);
            } else {
                h2d_async(d_c, h_c, c_bytes, stream);
            }

            int m = static_cast<int>(blkC.num_rows);
            int n = static_cast<int>(blkC.num_cols);
            int k = static_cast<int>(blkA.num_cols);
            int lda = static_cast<int>(blkA.num_cols);
            int ldb = static_cast<int>(blkB.num_cols);
            int ldc = static_cast<int>(blkC.num_cols);

            // Dispatch cuBLAS
            if (dtype == DataType::FLOAT) {
                float alf = static_cast<float>(alpha);
                float bet = static_cast<float>(beta);
                cublasSgemm(hdl, CUBLAS_OP_N, CUBLAS_OP_N,
                            m, n, k, &alf,
                            static_cast<const float*>(d_a), lda,
                            static_cast<const float*>(d_b), ldb,
                            &bet,
                            static_cast<float*>(d_c), ldc);
            } else if (dtype == DataType::DOUBLE) {
                double alf = alpha;
                double bet = beta;
                cublasDgemm(hdl, CUBLAS_OP_N, CUBLAS_OP_N,
                            m, n, k, &alf,
                            static_cast<const double*>(d_a), lda,
                            static_cast<const double*>(d_b), ldb,
                            &bet,
                            static_cast<double*>(d_c), ldc);
            } else if (dtype == DataType::COMPLEX_FLOAT) {
                cuComplex alf = make_cuComplex(static_cast<float>(alpha), 0.0f);
                cuComplex bet = make_cuComplex(static_cast<float>(beta), 0.0f);
                cublasCgemm(hdl, CUBLAS_OP_N, CUBLAS_OP_N,
                            m, n, k, &alf,
                            static_cast<const cuComplex*>(d_a), lda,
                            static_cast<const cuComplex*>(d_b), ldb,
                            &bet,
                            static_cast<cuComplex*>(d_c), ldc);
            } else if (dtype == DataType::COMPLEX_DOUBLE) {
                cuDoubleComplex alf = make_cuDoubleComplex(alpha, 0.0);
                cuDoubleComplex bet = make_cuDoubleComplex(beta, 0.0);
                cublasZgemm(hdl, CUBLAS_OP_N, CUBLAS_OP_N,
                            m, n, k, &alf,
                            static_cast<const cuDoubleComplex*>(d_a), lda,
                            static_cast<const cuDoubleComplex*>(d_b), ldb,
                            &bet,
                            static_cast<cuDoubleComplex*>(d_c), ldc);
            }

            // Async D2H: copy result back
            d2h_async(h_c, d_c, c_bytes, stream);

            // Free device buffers (can be deferred to a pool in production)
            device_free(d_a);
            device_free(d_b);
            device_free(d_c);
        }
    }

    // Wait for all streams
    cudaDeviceSynchronize();
}

#endif // TAPP_CON_USE_CUDA

} // namespace tapp_con
