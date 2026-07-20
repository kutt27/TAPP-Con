#include "su2_handler.h"
#include <cstdlib>

namespace tapp_con {

bool SU2Handler::compatible(int32_t sA_t2, int32_t sB_t2, int32_t sC_t2) {
    // Triangle inequality: |sA - sB| <= sC <= sA + sB
    // All stored as spin x 2, so no half-integer issues
    int32_t diff = std::abs(sA_t2 - sB_t2);
    int32_t sum  = sA_t2 + sB_t2;

    // Parity condition: sA + sB + sC must be even (integer spin)
    // Since all are spin x 2, the sum of values is even when
    // (sA_t2 + sB_t2 + sC_t2) is even
    if ((sA_t2 + sB_t2 + sC_t2) % 2 != 0)
        return false;

    return sC_t2 >= diff && sC_t2 <= sum;
}

std::vector<int32_t> SU2Handler::allowed_fusions(int32_t sA_t2, int32_t sB_t2) {
    std::vector<int32_t> result;
    int32_t diff = std::abs(sA_t2 - sB_t2);
    int32_t sum  = sA_t2 + sB_t2;

    for (int32_t sC_t2 = diff; sC_t2 <= sum; sC_t2 += 2) {
        result.push_back(sC_t2);
    }

    return result;
}

double SU2Handler::clebsch_gordan(int32_t /*sA_t2*/, int32_t /*mA_t2*/,
                                   int32_t /*sB_t2*/, int32_t /*mB_t2*/,
                                   int32_t /*sC_t2*/, int32_t /*mC_t2*/) {
    // Phase 1: placeholder — return 1.0 for any valid fusion.
    // Full CG coefficient computation via Wigner 3j symbols deferred to Phase 2.
    return 1.0;
}

} // namespace tapp_con
