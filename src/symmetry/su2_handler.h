#ifndef TAPP_CON_SU2_HANDLER_H
#define TAPP_CON_SU2_HANDLER_H

#include <cstdint>
#include <vector>

namespace tapp_con {

struct SU2Handler {
    // Check if spin sC (x2) is in the fusion product sA (x2) x sB (x2)
    static bool compatible(int32_t sA_t2, int32_t sB_t2, int32_t sC_t2);

    // Returns all valid fused spins sC (x2) satisfying triangle inequality
    static std::vector<int32_t> allowed_fusions(int32_t sA_t2, int32_t sB_t2);

    // Clebsch-Gordan coefficient: <sA mA sB mB | sC mC>
    // Phase 1: returns 1.0 for any valid fusion (placeholder)
    static double clebsch_gordan(int32_t sA_t2, int32_t mA_t2,
                                  int32_t sB_t2, int32_t mB_t2,
                                  int32_t sC_t2, int32_t mC_t2);
};

} // namespace tapp_con

#endif // TAPP_CON_SU2_HANDLER_H
