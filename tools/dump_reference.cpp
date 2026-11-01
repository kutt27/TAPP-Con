// Reference fixture dumper for the Python port parity test.
//
// Builds a set of inputs against the real C++ library and prints the observed
// results as JSON on stdout. python/tests/test_parity.py re-runs the same inputs
// through the NumPy port and asserts every value matches.
//
// Build (equivalent to a CMake target):
//   g++ -std=c++17 -O2 -Iinclude -Isrc tools/dump_reference.cpp \
//       <library .cpp sources> -o build/tools/dump_reference

#include "tapp_constrained.h"
#include "constrained_tensor.h"
#include "constraint/constraint_parser.h"
#include "constraint/constraint_solver.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace tapp_con;

// ── Minimal JSON writer ────────────────────────────────

static std::string esc(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            default: out += c;
        }
    }
    return out;
}

static std::string fnum(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.9g", v);
    return buf;
}

static std::string fnum(float v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(v));
    return buf;
}

// ── Fixture builders ───────────────────────────────────

static void dump_parser() {
    const char* inputs[] = {
        "q0 + q1 = 0", "q0 = 5", "2*q0 + q1 - q2 = 3", "-q0 + q1 = 0",
        "-2*q0 = 0", "2 q0 = 0", "2q0 = 0", "q0 + q1", "q0 = 5 garbage",
        "q0 = -5", "x0 = 0", "q = 0", "qX = 0", "", "q0 # 1 = 0",
    };

    printf("  \"parser\": [\n");
    bool first = true;
    for (const char* in : inputs) {
        std::string err;
        LinearConstraint lc = ConstraintParser::parse_u1(in, &err);
        if (!first) printf(",\n");
        first = false;
        printf("    {\"input\": \"%s\", \"ok\": %s", esc(in).c_str(),
               err.empty() ? "true" : "false");
        if (err.empty()) {
            printf(", \"target\": %d, \"terms\": [", lc.target);
            for (size_t i = 0; i < lc.terms.size(); ++i) {
                printf("%s[%d,%d]", i ? ", " : "", lc.terms[i].leg_index,
                       lc.terms[i].coefficient);
            }
            printf("]");
        } else {
            printf(", \"error\": \"%s\"", esc(err).c_str());
        }
        printf("}");
    }
    printf("\n  ],\n");
}

static void dump_system_and_blocks() {
    const char* expr = "q0 + q1 = 0";
    std::vector<std::string> exprs = {expr};
    std::vector<int> sectors = {3, 3};

    ParseResult pr = ConstraintParser::parse_system(2, exprs, sectors,
                                                    SymmetryType::U1, -1, 1);

    printf("  \"parse_system\": {\"success\": %s, \"leg_charges\": [",
           pr.success ? "true" : "false");
    for (size_t l = 0; l < pr.system.leg_sectors.size(); ++l) {
        printf("%s[", l ? ", " : "");
        const auto& ch = pr.system.leg_sectors[l].allowed_charges;
        for (size_t i = 0; i < ch.size(); ++i) printf("%s%d", i ? ", " : "", ch[i]);
        printf("]");
    }
    printf("]},\n");

    std::vector<int> row_legs = {0};
    std::vector<int> col_legs = {1};
    std::vector<uint64_t> dims = {6, 6};
    auto blocks = ConstraintSolver::generate_blocks(pr.system, row_legs, col_legs, dims);

    printf("  \"blocks\": [\n");
    for (size_t i = 0; i < blocks.size(); ++i) {
        const auto& b = blocks[i];
        printf("    %s[%lu, %lu, %lu, %lu, %d, %d]",
               i ? ",\n" : "", (unsigned long)b.row_offset,
               (unsigned long)b.col_offset, (unsigned long)b.num_rows,
               (unsigned long)b.num_cols, b.u1_qnum, b.su2_spin_times_two);
    }
    printf("\n  ],\n");

    // Sector generation table (exercises both branches).
    struct SecCase { int num; int lo; int hi; };
    SecCase cases[] = {
        {3, -1, 1}, {1, -1, 1}, {2, -1, 1}, {4, -1, 1}, {5, -3, 3},
        {8, -7, 7}, {10, -1, 1},
    };
    printf("  \"sectors\": [\n");
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
        std::vector<int> ss = {cases[c].num};
        ParseResult r = ConstraintParser::parse_system(
            1, std::vector<std::string>{}, ss, SymmetryType::U1,
            cases[c].lo, cases[c].hi);
        printf("    %s{\"num\": %d, \"lo\": %d, \"hi\": %d, \"charges\": [",
               c ? ",\n" : "", cases[c].num, cases[c].lo, cases[c].hi);
        const auto& ch = r.system.leg_sectors[0].allowed_charges;
        for (size_t i = 0; i < ch.size(); ++i) printf("%s%d", i ? ", " : "", ch[i]);
        printf("]}");
    }
    printf("\n  ],\n");
}

static void dump_build() {
    tapp_constrained_tensor_t* t = nullptr;
    const char* cons[] = {"q0 + q1 = 0"};
    int sectors[] = {3, 3};
    int row_legs[] = {0};
    int col_legs[] = {1};
    uint64_t dims[] = {6, 6};

    tapp_constrained_status_t st = tapp_constrained_build_from_constraints(
        &t, 2, cons, 1, sectors, row_legs, 1, col_legs, 1, dims,
        TAPP_CON_FLOAT, TAPP_CON_SYMMETRY_U1, -1, 1);

    printf("  \"build\": {\"status\": %d", (int)st);
    if (t) {
        uint64_t nb = 0;
        tapp_constrained_tensor_num_blocks(t, &nb);
        ConstrainedTensor* ct = reinterpret_cast<ConstrainedTensor*>(t);
        printf(", \"total_rows\": %lu, \"total_cols\": %lu, \"num_blocks\": %lu, "
               "\"buffer_size\": %lu, \"valid\": %s, \"blocks\": [",
               (unsigned long)ct->total_rows(), (unsigned long)ct->total_cols(),
               (unsigned long)nb, (unsigned long)ct->buffer_size(),
               ct->validate() ? "true" : "false");
        for (uint64_t i = 0; i < nb; ++i) {
            const auto& b = ct->block(static_cast<size_t>(i));
            printf("%s[%lu, %lu, %lu, %lu, %d]", i ? ", " : "",
                   (unsigned long)b.row_offset, (unsigned long)b.col_offset,
                   (unsigned long)b.num_rows, (unsigned long)b.num_cols,
                   b.u1_qnum);
        }
        printf("]");
        tapp_constrained_tensor_destroy(t);
    } else {
        printf(", \"error\": \"%s\"", tapp_constrained_error_string(st));
    }
    printf("},\n");
}

// Build a tensor with explicit blocks and run a GEMM through the C API.
static void dump_gemm_case(const char* name,
                           bool trans_a,
                           double alpha,
                           double beta,
                           bool u1_blocks,
                           bool init_c_nonzero) {
    const uint64_t N = 4;
    tapp_constrained_tensor_t *A = nullptr, *B = nullptr, *C = nullptr;
    tapp_constrained_tensor_create(&A, TAPP_CON_FLOAT, N, N, TAPP_CON_SYMMETRY_U1);
    tapp_constrained_tensor_create(&B, TAPP_CON_FLOAT, N, N, TAPP_CON_SYMMETRY_U1);
    tapp_constrained_tensor_create(&C, TAPP_CON_FLOAT, N, N, TAPP_CON_SYMMETRY_U1);

    if (u1_blocks) {
        // A: q=0 at (0,0), q=1 at (2,2).  B: q=1 at (0,0), q=0 at (2,2).
        // C blocks (q=1) only receive A00@B00 and A11@B11.
        const float a00[4] = {1, 2, 3, 4};
        const float a11[4] = {5, 6, 7, 8};
        const float b00[4] = {9, 10, 11, 12};
        const float b11[4] = {13, 14, 15, 16};
        tapp_constrained_tensor_add_block(A, 0, 0, 2, 2, 0, 0, a00);
        tapp_constrained_tensor_add_block(A, 2, 2, 2, 2, 1, 0, a11);
        tapp_constrained_tensor_add_block(B, 0, 0, 2, 2, 1, 0, b00);
        tapp_constrained_tensor_add_block(B, 2, 2, 2, 2, 0, 0, b11);
        tapp_constrained_tensor_add_block(C, 0, 0, 2, 2, 1, 0, nullptr);
        tapp_constrained_tensor_add_block(C, 2, 2, 2, 2, 1, 0, nullptr);
    } else {
        const float afull[16] = {1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 5, 6, 0, 0, 7, 8};
        const float bfull[16] = {5, 6, 0, 0, 7, 8, 0, 0, 0, 0, 9, 10, 0, 0, 11, 12};
        tapp_constrained_tensor_add_block(A, 0, 0, 4, 4, 0, 0, afull);
        tapp_constrained_tensor_add_block(B, 0, 0, 4, 4, 0, 0, bfull);
        tapp_constrained_tensor_add_block(C, 0, 0, 4, 4, 0, 0, nullptr);
    }

    if (init_c_nonzero) {
        ConstrainedTensor* ctmp = reinterpret_cast<ConstrainedTensor*>(C);
        for (uint64_t k = 0; k < ctmp->num_blocks(); ++k) {
            const auto& b = ctmp->block(static_cast<size_t>(k));
            float* p = reinterpret_cast<float*>(
                static_cast<uint8_t*>(ctmp->raw_buffer()) + b.data_offset);
            for (uint64_t i = 0; i < b.num_rows * b.num_cols; ++i) p[i] = 10.0f;
        }
    }

    tapp_constrained_plan_t* plan = nullptr;
    tapp_constrained_plan_create(&plan, A, B, C, TAPP_CON_FLOAT,
                                 trans_a ? 1 : 0, 0);
    tapp_constrained_gemm(plan, alpha, beta);

    ConstrainedTensor* ct = reinterpret_cast<ConstrainedTensor*>(C);
    printf("  \"%s\": {\n    \"c_data\": [", name);
    uint64_t nb = ct->num_blocks();
    bool firstnum = true;
    for (uint64_t k = 0; k < nb; ++k) {
        const auto& b = ct->block(static_cast<size_t>(k));
        const float* p = reinterpret_cast<const float*>(
            static_cast<const uint8_t*>(ct->raw_buffer()) + b.data_offset);
        for (uint64_t i = 0; i < b.num_rows * b.num_cols; ++i) {
            printf("%s%s", firstnum ? "" : ", ", fnum(p[i]).c_str());
            firstnum = false;
        }
    }
    printf("],\n    \"blocks\": [");
    for (uint64_t k = 0; k < nb; ++k) {
        const auto& b = ct->block(static_cast<size_t>(k));
        printf("%s[%lu, %lu, %lu, %lu, %d]", k ? ", " : "",
               (unsigned long)b.row_offset, (unsigned long)b.col_offset,
               (unsigned long)b.num_rows, (unsigned long)b.num_cols, b.u1_qnum);
    }
    printf("]\n  },\n");

    tapp_constrained_plan_destroy(plan);
    tapp_constrained_tensor_destroy(A);
    tapp_constrained_tensor_destroy(B);
    tapp_constrained_tensor_destroy(C);
}

int main() {
    printf("{\n");
    dump_parser();
    dump_system_and_blocks();
    dump_build();
    dump_gemm_case("gemm_simple", false, 1.0, 0.0, false, false);
    dump_gemm_case("gemm_u1_blocks", false, 1.0, 0.0, true, false);
    dump_gemm_case("gemm_alpha_beta", false, 2.0, 0.5, true, true);
    dump_gemm_case("gemm_transpose_a", true, 1.0, 0.0, false, false);
    printf("  \"symmetry\": {\n");
    printf("    \"u1\": [[0,0,0,1],[1,2,3,1],[-1,1,0,1],[0,0,1,0],[1,1,3,0]],\n");
    printf("    \"su2\": [[1,1,0,1],[1,1,2,1],[1,1,1,0],[2,1,1,1],[2,1,5,0]]\n");
    printf("  }\n");
    printf("}\n");
    return 0;
}
