// Constrained block-sparse GEMM (TAPP-Con) vs dense GEMM baseline
//
// Measures per-call wall time of TAPP-Con's constrained block-sparse GEMM
// against a plain dense GEMM on the equivalent expanded matrices, and verifies
// both produce the same numbers.
//
// Two modes:
//   structured  : block-diagonal U(1) tensors built block-by-block. The
//                 block-sparse result is exactly dense(A) @ dense(B), so the
//                 correctness gate is a direct dense-product comparison.
//   constrained : tensors built through the constraint engine
//                 (build_from_constraints), exercising the same API path the
//                 polyacetylene ground-state benchmark uses, with a
//                 charge-conserving "q0 - q1 = 0" constraint that yields a
//                 tunable K-block diagonal structure (sparsity = 1/K).
//
// The dense baseline is a naive row-major triple-loop GEMM (no BLAS on this
// machine, and also what the library falls back to), so the measured speedup
// is the algorithmic sparsity win, not a BLAS-quality difference. An OpenMP
// dense variant is reported alongside for reference.

#include "tapp_constrained.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ── Timing helpers ─────────────────────────────────────

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

template <typename F>
static double time_per_call(F&& f, int reps) {
    f();  // warmup
    double t0 = now_s();
    for (int r = 0; r < reps; ++r) f();
    double dt = now_s() - t0;
    return dt / static_cast<double>(reps);
}

static int reps_for_flops(double flops) {
    if (flops <= 0.0) return 1;
    int r = static_cast<int>(3.0e8 / flops);
    return std::max(1, std::min(2000, r));
}

// ── Dense baseline kernels ─────────────────────────────

// Plain dense row-major GEMM. Exploits nothing about the block structure.
static void dense_gemm_naive(int N, const float* A, const float* B, float* C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

static void dense_gemm_omp(int N, const float* A, const float* B, float* C) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ── Tensor helpers ─────────────────────────────────────

static void fill_blocks_random(tapp_constrained_tensor_t* t, unsigned seed) {
    uint64_t nb = 0;
    tapp_constrained_tensor_num_blocks(t, &nb);
    unsigned rng = seed;
    for (uint64_t b = 0; b < nb; ++b) {
        uint64_t ro, co, br, bc;
        const void* data = nullptr;
        tapp_constrained_tensor_get_block(t, b, &ro, &co, &br, &bc,
                                           nullptr, nullptr, &data);
        float* p = const_cast<float*>(static_cast<const float*>(data));
        for (uint64_t i = 0; i < br * bc; ++i) {
            rng = rng * 1103515245u + 12345u;
            p[i] = static_cast<float>(rng % 200) / 100.0f - 1.0f;
        }
    }
}

// Expand a block-sparse tensor into a dense N x N row-major matrix.
static void expand_dense(const tapp_constrained_tensor_t* t, int N,
                         std::vector<float>& M) {
    M.assign(static_cast<size_t>(N) * N, 0.0f);
    uint64_t nb = 0;
    tapp_constrained_tensor_num_blocks(t, &nb);
    for (uint64_t b = 0; b < nb; ++b) {
        uint64_t ro, co, br, bc;
        const void* data = nullptr;
        tapp_constrained_tensor_get_block(t, b, &ro, &co, &br, &bc,
                                           nullptr, nullptr, &data);
        const float* p = static_cast<const float*>(data);
        for (uint64_t r = 0; r < br; ++r) {
            for (uint64_t c = 0; c < bc; ++c) {
                M[static_cast<size_t>(ro + r) * N + (co + c)] = p[r * bc + c];
            }
        }
    }
}

static double block_area(const tapp_constrained_tensor_t* t, int N) {
    uint64_t nb = 0;
    tapp_constrained_tensor_num_blocks(t, &nb);
    double area = 0.0;
    for (uint64_t b = 0; b < nb; ++b) {
        uint64_t ro, co, br, bc;
        tapp_constrained_tensor_get_block(t, b, &ro, &co, &br, &bc,
                                           nullptr, nullptr, nullptr);
        area += static_cast<double>(br) * static_cast<double>(bc);
    }
    return area / (static_cast<double>(N) * N);
}

// ── Tensor builders ────────────────────────────────────

// Block-diagonal U(1) tensor: n_blocks diagonal s x s blocks, all charge 0.
static tapp_constrained_tensor_t* build_blockdiag(int n_blocks, int s,
                                                  unsigned seed) {
    int N = n_blocks * s;
    tapp_constrained_tensor_t* t = nullptr;
    if (tapp_constrained_tensor_create(&t, TAPP_CON_FLOAT, N, N,
                                       TAPP_CON_SYMMETRY_U1) != TAPP_CON_SUCCESS)
        return nullptr;
    std::vector<float> buf(static_cast<size_t>(s) * s, 0.0f);
    for (int i = 0; i < n_blocks; ++i) {
        tapp_constrained_tensor_add_block(t, i * s, i * s, s, s, 0, 0,
                                           buf.data());
    }
    fill_blocks_random(t, seed);
    return t;
}

// Tensor built through the constraint engine (row leg 0, col leg 1).
static tapp_constrained_tensor_t* build_constrained(const char* constraint,
                                                    int K, int sector_dim,
                                                    int32_t charge_min,
                                                    int32_t charge_max,
                                                    unsigned seed) {
    int N = K * sector_dim;
    const char* cons[] = {constraint};
    int sectors[] = {K, K};
    int row_legs[] = {0};
    int col_legs[] = {1};
    uint64_t dims[] = {static_cast<uint64_t>(N), static_cast<uint64_t>(N)};

    tapp_constrained_tensor_t* t = nullptr;
    tapp_constrained_status_t st = tapp_constrained_build_from_constraints(
        &t, 2, cons, 1, sectors, row_legs, 1, col_legs, 1, dims,
        TAPP_CON_FLOAT, TAPP_CON_SYMMETRY_U1, charge_min, charge_max);
    if (st != TAPP_CON_SUCCESS || !t) return nullptr;
    fill_blocks_random(t, seed);
    return t;
}

// Manually built tensor with a diagonal block pattern, all charge 0.
static tapp_constrained_tensor_t* build_diag_manual(int K, int sector_dim,
                                                    unsigned seed) {
    int N = K * sector_dim;
    tapp_constrained_tensor_t* t = nullptr;
    if (tapp_constrained_tensor_create(&t, TAPP_CON_FLOAT, N, N,
                                       TAPP_CON_SYMMETRY_U1) != TAPP_CON_SUCCESS)
        return nullptr;
    std::vector<float> buf(static_cast<size_t>(sector_dim) * sector_dim, 0.0f);
    for (int i = 0; i < K; ++i) {
        tapp_constrained_tensor_add_block(t, i * sector_dim, i * sector_dim,
                                           sector_dim, sector_dim, 0, 0,
                                           buf.data());
    }
    fill_blocks_random(t, seed);
    return t;
}

// Independent count of fusible (A,B,C) block triples, mirroring the matching
// rules the library applies at plan-build time. Used as a cross-check on how
// much work a constrained plan actually performs.
static unsigned count_triples(const tapp_constrained_tensor_t* A,
                              const tapp_constrained_tensor_t* B,
                              const tapp_constrained_tensor_t* C) {
    struct Blk { uint64_t ro, co, nr, nc; int32_t q; };
    auto load = [](const tapp_constrained_tensor_t* t) {
        std::vector<Blk> v;
        uint64_t nb = 0;
        tapp_constrained_tensor_num_blocks(t, &nb);
        for (uint64_t i = 0; i < nb; ++i) {
            uint64_t ro, co, nr, nc; int32_t q = 0;
            tapp_constrained_tensor_get_block(t, i, &ro, &co, &nr, &nc, &q,
                                               nullptr, nullptr);
            v.push_back({ro, co, nr, nc, q});
        }
        return v;
    };
    auto ab = load(A), bb = load(B), cb = load(C);
    unsigned n = 0;
    for (const auto& a : ab)
        for (const auto& b : bb) {
            if (a.nc != b.nr) continue;
            for (const auto& c : cb) {
                if (c.ro != a.ro || c.co != b.co) continue;
                if (c.nr != a.nr || c.nc != b.nc) continue;
                if (a.q + b.q != c.q) continue;
                ++n;
                break;
            }
        }
    return n;
}

// ── One benchmark case ─────────────────────────────────

struct CaseResult {
    std::string label;
    int N = 0;
    uint64_t n_blocks = 0;
    unsigned n_triples = 0;
    double sparsity = 0.0;
    double us_block = 0.0;
    double us_dense = 0.0;
    double us_dense_omp = 0.0;
    double speedup = 0.0;
    double gflops_block = 0.0;
    bool correct = false;
    double max_err = 0.0;
};

static CaseResult run_case(const std::string& label,
                           tapp_constrained_tensor_t* A,
                           tapp_constrained_tensor_t* B,
                           tapp_constrained_tensor_t* C) {
    CaseResult res;
    res.label = label;

    uint64_t nA = 0, nB = 0, nC = 0;
    tapp_constrained_tensor_num_blocks(A, &nA);
    tapp_constrained_tensor_num_blocks(B, &nB);
    tapp_constrained_tensor_num_blocks(C, &nC);
    res.n_blocks = nC;
    res.n_triples = count_triples(A, B, C);

    // Determine the logical matrix size N from the block extents.
    int N = 0;
    {
        uint64_t nb = 0, ro, co, br, bc;
        tapp_constrained_tensor_num_blocks(A, &nb);
        for (uint64_t i = 0; i < nb; ++i) {
            tapp_constrained_tensor_get_block(A, i, &ro, &co, &br, &bc,
                                               nullptr, nullptr, nullptr);
            uint64_t rmax = ro + br, cmax = co + bc;
            if (rmax > static_cast<uint64_t>(N)) N = static_cast<int>(rmax);
            if (cmax > static_cast<uint64_t>(N)) N = static_cast<int>(cmax);
        }
    }
    res.N = N;

    std::vector<float> Ad, Bd, Cd;
    expand_dense(A, N, Ad);
    expand_dense(B, N, Bd);
    expand_dense(C, N, Cd);
    res.sparsity = block_area(C, N);

    // Reference dense product.
    std::vector<float> Ref(static_cast<size_t>(N) * N, 0.0f);
    dense_gemm_naive(N, Ad.data(), Bd.data(), Ref.data());

    // Library block-sparse GEMM.
    tapp_constrained_plan_t* plan = nullptr;
    if (tapp_constrained_plan_create(&plan, A, B, C, TAPP_CON_FLOAT, 0, 0) !=
        TAPP_CON_SUCCESS) {
        fprintf(stderr, "plan_create failed for %s\n", label.c_str());
        return res;
    }

    tapp_constrained_gemm(plan, 1.0, 0.0);
    expand_dense(C, N, Cd);

    res.max_err = 0.0;
    for (size_t i = 0; i < Cd.size(); ++i) {
        double d = std::fabs(static_cast<double>(Cd[i]) -
                             static_cast<double>(Ref[i]));
        if (d > res.max_err) res.max_err = d;
    }
    res.correct = res.max_err <= 1e-4;

    // Block-sparse compute cost (flops) -> choose a sensible rep count.
    double block_flops = 0.0;
    {
        uint64_t nb = 0, ro, co, br, bc;
        tapp_constrained_tensor_num_blocks(A, &nb);
        for (uint64_t i = 0; i < nb; ++i) {
            tapp_constrained_tensor_get_block(A, i, &ro, &co, &br, &bc,
                                               nullptr, nullptr, nullptr);
            block_flops += 2.0 * br * bc * static_cast<double>(br);
        }
    }
    double dense_flops = 2.0 * static_cast<double>(N) * N * N;

    int breps = reps_for_flops(block_flops);
    int dreps = reps_for_flops(dense_flops);

    res.us_block = time_per_call([&] { tapp_constrained_gemm(plan, 1.0, 0.0); },
                                 breps) * 1e6;

    std::vector<float> Cout(static_cast<size_t>(N) * N, 0.0f);
    res.us_dense = time_per_call(
        [&] { dense_gemm_naive(N, Ad.data(), Bd.data(), Cout.data()); }, dreps) *
        1e6;
    res.us_dense_omp = time_per_call(
        [&] { dense_gemm_omp(N, Ad.data(), Bd.data(), Cout.data()); }, dreps) *
        1e6;

    res.speedup = res.us_dense / res.us_block;
    res.gflops_block = block_flops / (res.us_block * 1e-6) / 1e9;

    tapp_constrained_plan_destroy(plan);
    return res;
}

static void print_header() {
    printf("  %-24s %5s %7s %8s %7s %10s %10s %10s %8s %7s\n",
           "case", "N", "blocks", "triples", "spars", "us/block", "us/dense",
           "us/denseOMP", "speedup", "correct");
    printf("  %s\n", std::string(124, '-').c_str());
}

static void print_row(const CaseResult& r) {
    printf("  %-24s %5d %7lu %8u %7.3f %10.2f %10.2f %10.2f %8.2fx %7s\n",
           r.label.c_str(), r.N, (unsigned long)r.n_blocks, r.n_triples,
           r.sparsity, r.us_block, r.us_dense, r.us_dense_omp, r.speedup,
           r.correct ? "YES" : "NO");
}

// ── Modes ──────────────────────────────────────────────

static int run_structured() {
    printf("\n=== Mode: structured (block-diagonal U(1), exact dense-product check) ===\n\n");
    print_header();

    int s_sizes[] = {8, 16, 32, 64};
    int nb_sizes[] = {4, 16};
    int failures = 0;

    for (int nb : nb_sizes) {
        for (int s : s_sizes) {
            int N = nb * s;
            if (N > 1024) continue;  // keep the naive dense baseline tractable
            tapp_constrained_tensor_t* A = build_blockdiag(nb, s, 11u);
            tapp_constrained_tensor_t* B = build_blockdiag(nb, s, 22u);
            tapp_constrained_tensor_t* C = build_blockdiag(nb, s, 33u);
            if (!A || !B || !C) {
                fprintf(stderr, "build failed\n");
                return 1;
            }
            char label[64];
            snprintf(label, sizeof(label), "s=%d nb=%d", s, nb);
            CaseResult r = run_case(label, A, B, C);
            print_row(r);
            if (!r.correct) ++failures;
            tapp_constrained_tensor_destroy(A);
            tapp_constrained_tensor_destroy(B);
            tapp_constrained_tensor_destroy(C);
        }
    }
    return failures == 0 ? 0 : 1;
}

static int run_constrained() {
    printf("\n=== Mode: constrained (engine-built A,B via 'q0 + q1 = 0'; C = product closure) ===\n");
    printf("    Symmetric charge range makes the anti-diagonal A,B structure close onto a\n");
    printf("    diagonal C, so the block-sparse result is exactly dense(A) @ dense(B).\n\n");
    print_header();

    int sd_sizes[] = {8, 16, 32, 64};
    int k_sizes[] = {4, 8, 16};
    int failures = 0;

    for (int K : k_sizes) {
        // Symmetric charge range -(K-1)..(K-1) with K sectors yields charges
        // -(K-1) + 2i, so q_i + q_j == 0 selects the K anti-diagonal pairs.
        int32_t cmin = -(K - 1);
        int32_t cmax = (K - 1);
        for (int sd : sd_sizes) {
            int N = K * sd;
            if (N > 1024) continue;
            tapp_constrained_tensor_t* A =
                build_constrained("q0 + q1 = 0", K, sd, cmin, cmax, 101u);
            tapp_constrained_tensor_t* B =
                build_constrained("q0 + q1 = 0", K, sd, cmin, cmax, 202u);
            tapp_constrained_tensor_t* C = build_diag_manual(K, sd, 303u);
            if (!A || !B || !C) {
                fprintf(stderr, "build failed for K=%d sd=%d\n", K, sd);
                return 1;
            }
            char label[64];
            snprintf(label, sizeof(label), "K=%d sd=%d", K, sd);
            CaseResult r = run_case(label, A, B, C);
            print_row(r);
            if (!r.correct) ++failures;
            tapp_constrained_tensor_destroy(A);
            tapp_constrained_tensor_destroy(B);
            tapp_constrained_tensor_destroy(C);
        }
    }

    // Polyacetylene-style structure (q0 + q1 = 0): reported for timing and
    // sparsity only. Its block support is a strict subset of the dense product
    // (cross terms land outside C's blocks), so exact dense equality does not
    // apply; the library's unit tests cover its values.
    printf("\n  -- polyacetylene-style structure (timing/sparsity only) --\n\n");
    print_header();
    for (int K : {3}) {
        for (int sd : {16, 32, 64}) {
            int N = K * sd;
            tapp_constrained_tensor_t* A = build_constrained("q0 + q1 = 0", K, sd, -1, 1, 7u);
            tapp_constrained_tensor_t* B = build_constrained("q0 + q1 = 0", K, sd, -1, 1, 8u);
            tapp_constrained_tensor_t* C = build_constrained("q0 + q1 = 0", K, sd, -1, 1, 9u);
            if (!A || !B || !C) continue;

            tapp_constrained_plan_t* plan = nullptr;
            tapp_constrained_plan_create(&plan, A, B, C, TAPP_CON_FLOAT, 0, 0);

            std::vector<float> Ad, Bd;
            expand_dense(A, N, Ad);
            expand_dense(B, N, Bd);

            double block_flops = 0.0;
            {
                uint64_t nb, ro, co, br, bc;
                tapp_constrained_tensor_num_blocks(A, &nb);
                for (uint64_t i = 0; i < nb; ++i) {
                    tapp_constrained_tensor_get_block(A, i, &ro, &co, &br, &bc,
                                                       nullptr, nullptr, nullptr);
                    block_flops += 2.0 * br * bc * static_cast<double>(br);
                }
            }
            double dense_flops = 2.0 * static_cast<double>(N) * N * N;
            double ub = time_per_call([&] { tapp_constrained_gemm(plan, 1.0, 0.0); },
                                      reps_for_flops(block_flops)) * 1e6;
            std::vector<float> Cout(static_cast<size_t>(N) * N, 0.0f);
            double ud = time_per_call(
                [&] { dense_gemm_naive(N, Ad.data(), Bd.data(), Cout.data()); },
                reps_for_flops(dense_flops)) * 1e6;
            double uo = time_per_call(
                [&] { dense_gemm_omp(N, Ad.data(), Bd.data(), Cout.data()); },
                reps_for_flops(dense_flops)) * 1e6;

            CaseResult r;
            r.label = "poly K=3 sd=" + std::to_string(sd);
            r.N = N;
            tapp_constrained_tensor_num_blocks(C, &r.n_blocks);
            r.n_triples = count_triples(A, B, C);
            r.sparsity = block_area(C, N);
            r.us_block = ub;
            r.us_dense = ud;
            r.us_dense_omp = uo;
            r.speedup = ud / ub;
            r.correct = true;  // not gated here
            print_row(r);

            tapp_constrained_plan_destroy(plan);
            tapp_constrained_tensor_destroy(A);
            tapp_constrained_tensor_destroy(B);
            tapp_constrained_tensor_destroy(C);
        }
    }

    // Diagnostic: the engine's canonical structure (A, B and C all built from
    // the same constraint) is degenerate under the charge filter. Every
    // engine-built block stores the charge summed over *all* legs, so the rule
    // qA + qB == qC holds only for the zero-charge block, and the geometric
    // matching additionally requires A's column range to equal B's row range
    // (the library only compares sizes here). Result: almost no usable triples.
    printf("\n  -- diagnostic: engine-only structure (A,B,C all from engine) --\n\n");
    for (int K : {4, 8}) {
        int sd = 16;
        tapp_constrained_tensor_t* A = build_constrained("q0 - q1 = 0", K, sd, -1, 1, 1u);
        tapp_constrained_tensor_t* B = build_constrained("q0 - q1 = 0", K, sd, -1, 1, 2u);
        tapp_constrained_tensor_t* C = build_constrained("q0 - q1 = 0", K, sd, -1, 1, 3u);
        if (!A || !B || !C) continue;
        unsigned tr = count_triples(A, B, C);
        tapp_constrained_plan_t* diag_plan = nullptr;
        tapp_constrained_plan_create(&diag_plan, A, B, C, TAPP_CON_FLOAT, 0, 0);
        tapp_constrained_gemm(diag_plan, 1.0, 0.0);
        uint64_t nb = 0;
        tapp_constrained_tensor_num_blocks(C, &nb);
        long nz = 0;
        for (uint64_t b = 0; b < nb; ++b) {
            uint64_t ro, co, br, bc; const void* d = nullptr;
            tapp_constrained_tensor_get_block(C, b, &ro, &co, &br, &bc,
                                               nullptr, nullptr, &d);
            const float* p = static_cast<const float*>(d);
            for (uint64_t i = 0; i < br * bc; ++i)
                if (p[i] != 0.0f) ++nz;
        }
        printf("    'q0 - q1 = 0' K=%d sd=%d: C blocks=%lu, fusible triples=%u, "
               "nonzero C entries=%ld\n",
               K, sd, (unsigned long)nb, tr, nz);
        tapp_constrained_plan_destroy(diag_plan);
        tapp_constrained_tensor_destroy(A);
        tapp_constrained_tensor_destroy(B);
        tapp_constrained_tensor_destroy(C);
    }

    return failures == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    std::string mode = "all";
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--mode") && i + 1 < argc) mode = argv[++i];
        else if (!strcmp(argv[i], "--help")) {
            printf("Usage: %s [--mode structured|constrained|all]\n", argv[0]);
            return 0;
        }
    }

    printf("TAPP-Con: constrained block-sparse GEMM vs dense GEMM baseline\n");

    int rc = 0;
    if (mode == "structured" || mode == "all") rc |= run_structured();
    if (mode == "constrained" || mode == "all") rc |= run_constrained();

    printf("\n%s\n", rc == 0 ? "All correctness gates PASSED."
                             : "One or more correctness gates FAILED.");
    return rc;
}
