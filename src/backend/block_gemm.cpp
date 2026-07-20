#include "block_gemm.h"
#include "symmetry/u1_handler.h"
#include "symmetry/su2_handler.h"

#include <stdexcept>
#include <cstring>
#include <type_traits>

namespace tapp_con {

// ── Naive GEMM fallback (used when BLAS is unavailable) ──

template <typename T>
static void naive_gemm(int m, int n, int k,
                       T alpha,
                       const T* a, int lda,
                       const T* b, int ldb,
                       T beta,
                       T* c, int ldc,
                       bool transA, bool transB) {
    if (beta == T(0)) {
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                c[i * ldc + j] = T(0);
    }

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            T sum = T(0);
            for (int p = 0; p < k; ++p) {
                T a_val = transA ? a[p * lda + i] : a[i * lda + p];
                T b_val = transB ? b[j * ldb + p] : b[p * ldb + j];
                sum += a_val * b_val;
            }
            c[i * ldc + j] = alpha * sum + beta * c[i * ldc + j];
        }
    }
}

// ── BLAS path ──
#if defined(TAPP_CON_USE_CBLAS)
#include <cblas.h>

static void dispatch_gemm(int m, int n, int k,
                          double alpha,
                          const float* a, int lda,
                          const float* b, int ldb,
                          double beta,
                          float* c, int ldc,
                          bool transA, bool transB) {
    cblas_sgemm(CblasRowMajor,
                transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans,
                m, n, k,
                static_cast<float>(alpha),
                a, lda, b, ldb,
                static_cast<float>(beta),
                c, ldc);
}

static void dispatch_gemm(int m, int n, int k,
                          double alpha,
                          const double* a, int lda,
                          const double* b, int ldb,
                          double beta,
                          double* c, int ldc,
                          bool transA, bool transB) {
    cblas_dgemm(CblasRowMajor,
                transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans,
                m, n, k,
                alpha, a, lda, b, ldb,
                beta, c, ldc);
}

#else
// No BLAS — use naive fallback
template <typename T>
static void dispatch_gemm(int m, int n, int k,
                          double alpha,
                          const T* a, int lda,
                          const T* b, int ldb,
                          double beta,
                          T* c, int ldc,
                          bool transA, bool transB) {
    naive_gemm(m, n, k,
               static_cast<T>(alpha),
               a, lda, b, ldb,
               static_cast<T>(beta),
               c, ldc,
               transA, transB);
}
#endif

// ── Internal helpers ──

static inline void* block_data(ConstrainedTensor& tensor, const BlockDescriptor& blk) {
    return static_cast<uint8_t*>(tensor.raw_buffer()) + blk.data_offset;
}

static inline const void* block_data(const ConstrainedTensor& tensor, const BlockDescriptor& blk) {
    return static_cast<const uint8_t*>(tensor.raw_buffer()) + blk.data_offset;
}

static bool positions_match(uint64_t rA, uint64_t cA, uint64_t rB, uint64_t cB) {
    return rA == rB && cA == cB;
}

void BlockGEMM::multiply(
    ConstrainedTensor& C,
    const ConstrainedTensor& A,
    const ConstrainedTensor& B,
    double alpha,
    double beta,
    bool transA,
    bool transB)
{
    if (A.data_type() != B.data_type() || A.data_type() != C.data_type())
        throw std::invalid_argument("Type mismatch between tensors");

    DataType dtype = A.data_type();

    // Verify dimension compatibility
    uint64_t A_rows = A.total_rows();
    uint64_t A_cols = A.total_cols();
    uint64_t B_rows = B.total_rows();
    uint64_t B_cols = B.total_cols();
    uint64_t C_rows = C.total_rows();
    uint64_t C_cols = C.total_cols();

    uint64_t A_k = transA ? A_rows : A_cols;
    uint64_t B_k = transB ? B_rows : B_cols;
    uint64_t A_m = transA ? A_cols : A_rows;
    uint64_t B_n = transB ? B_cols : B_rows;

    if (A_k != B_k)
        throw std::invalid_argument("Inner dimension mismatch between A and B");
    if (A_m != C_rows || B_n != C_cols)
        throw std::invalid_argument("Output dimension mismatch");

    // Zero out C buffer if beta == 0
    if (beta == 0.0) {
        std::memset(C.raw_buffer(), 0, C.buffer_size());
    }

    // Iterate over all block pairs
    for (size_t i = 0; i < A.num_blocks(); ++i) {
        const auto& blkA = A.block(i);

        for (size_t j = 0; j < B.num_blocks(); ++j) {
            const auto& blkB = B.block(j);

            // Determine effective dimensions accounting for transposition
            uint64_t blkA_rows = transA ? blkA.num_cols : blkA.num_rows;
            uint64_t blkA_cols = transA ? blkA.num_rows : blkA.num_cols;
            uint64_t blkB_rows = transB ? blkB.num_cols : blkB.num_rows;
            uint64_t blkB_cols = transB ? blkB.num_rows : blkB.num_cols;

            uint64_t blkA_r_off = transA ? blkA.col_offset : blkA.row_offset;
            uint64_t blkA_c_off = transA ? blkA.row_offset : blkA.col_offset;
            uint64_t blkB_r_off = transB ? blkB.col_offset : blkB.row_offset;
            uint64_t blkB_c_off = transB ? blkB.row_offset : blkB.col_offset;

            // Inner dimension must match
            if (blkA_cols != blkB_rows)
                continue;

            // Find matching output block
            for (size_t k = 0; k < C.num_blocks(); ++k) {
                const auto& blkC = C.block(k);

                if (!positions_match(blkC.row_offset, blkC.col_offset,
                                     blkA_r_off, blkB_c_off))
                    continue;

                if (blkC.num_rows != blkA_rows || blkC.num_cols != blkB_cols)
                    continue;

                // Symmetry filtering
                bool sym_compatible = true;

                if (static_cast<uint8_t>(A.symmetry_type()) & static_cast<uint8_t>(SymmetryType::U1)) {
                    if (!U1Handler::compatible(blkA.u1_qnum, blkB.u1_qnum, blkC.u1_qnum)) {
                        sym_compatible = false;
                    }
                }

                if (sym_compatible &&
                    (static_cast<uint8_t>(A.symmetry_type()) & static_cast<uint8_t>(SymmetryType::SU2))) {
                    if (!SU2Handler::compatible(blkA.su2_spin_times_two,
                                                 blkB.su2_spin_times_two,
                                                 blkC.su2_spin_times_two)) {
                        sym_compatible = false;
                    }
                }

                if (!sym_compatible)
                    continue;

                // Dispatch dense GEMM on this block pair
                void* dataC = block_data(C, blkC);
                const void* dataA = block_data(A, blkA);
                const void* dataB = block_data(B, blkB);

                int lda = static_cast<int>(transA ? blkA.num_rows : blkA.num_cols);
                int ldb = static_cast<int>(transB ? blkB.num_rows : blkB.num_cols);
                int ldc = static_cast<int>(blkC.num_cols);
                int m = static_cast<int>(blkA_rows);
                int n = static_cast<int>(blkB_cols);
                int k_dim = static_cast<int>(blkA_cols);

                if (dtype == DataType::FLOAT) {
                    dispatch_gemm<float>(m, n, k_dim,
                                         alpha,
                                         static_cast<const float*>(dataA), lda,
                                         static_cast<const float*>(dataB), ldb,
                                         beta,
                                         static_cast<float*>(dataC), ldc,
                                         transA, transB);
                } else {
                    dispatch_gemm<double>(m, n, k_dim,
                                          alpha,
                                          static_cast<const double*>(dataA), lda,
                                          static_cast<const double*>(dataB), ldb,
                                          beta,
                                          static_cast<double*>(dataC), ldc,
                                          transA, transB);
                }
            }
        }
    }
}

// ── Parallel execution (Phase 2) ──────────────────────

void BlockGEMM::multiply_parallel(
    ConstrainedTensor& C,
    const ConstrainedTensor& A,
    const ConstrainedTensor& B,
    const std::vector<BlockTriple>& triples,
    const std::vector<std::vector<BlockTriple>>& c_block_work,
    double alpha,
    double beta,
    bool /*transA*/,
    bool /*transB*/)
{
    if (A.data_type() != B.data_type() || A.data_type() != C.data_type())
        throw std::invalid_argument("Type mismatch between tensors");

    DataType dtype = A.data_type();

    // Zero out C buffer if beta == 0
    if (beta == 0.0) {
        std::memset(C.raw_buffer(), 0, C.buffer_size());
    }

    // Parallelize over C-blocks — each thread owns its C-blocks exclusively
    size_t n_c_blocks = c_block_work.size();

    #ifdef TAPP_CON_USE_OPENMP
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int64_t ck = 0; ck < static_cast<int64_t>(n_c_blocks); ++ck) {
        auto idx = static_cast<size_t>(ck);
        for (const auto& tr : c_block_work[idx]) {
            const auto& blkA = A.block(tr.a_idx);
            const auto& blkB = B.block(tr.b_idx);
            const auto& blkC = C.block(tr.c_idx);

            void* dataC = static_cast<uint8_t*>(C.raw_buffer()) + blkC.data_offset;
            const void* dataA = static_cast<const uint8_t*>(A.raw_buffer()) + blkA.data_offset;
            const void* dataB = static_cast<const uint8_t*>(B.raw_buffer()) + blkB.data_offset;

            int lda = static_cast<int>(blkA.num_cols);
            int ldb = static_cast<int>(blkB.num_cols);
            int ldc = static_cast<int>(blkC.num_cols);
            int m = static_cast<int>(blkC.num_rows);
            int n = static_cast<int>(blkC.num_cols);
            int k_dim = static_cast<int>(blkA.num_cols);

            if (dtype == DataType::FLOAT) {
                dispatch_gemm<float>(m, n, k_dim,
                                     alpha,
                                     static_cast<const float*>(dataA), lda,
                                     static_cast<const float*>(dataB), ldb,
                                     beta,
                                     static_cast<float*>(dataC), ldc,
                                     false, false);
            } else {
                dispatch_gemm<double>(m, n, k_dim,
                                      alpha,
                                      static_cast<const double*>(dataA), lda,
                                      static_cast<const double*>(dataB), ldb,
                                      beta,
                                      static_cast<double*>(dataC), ldc,
                                      false, false);
            }
        }
    }
}

} // namespace tapp_con
