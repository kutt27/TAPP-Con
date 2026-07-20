#ifndef TAPP_CON_U1_HANDLER_H
#define TAPP_CON_U1_HANDLER_H

#include <cstdint>

namespace tapp_con {

struct U1Handler {
    // Returns true if blocks with quantum numbers qA and qB
    // can produce an output block with quantum number qC.
    static bool compatible(int32_t qA, int32_t qB, int32_t qC);

    // Returns the fused (resulting) quantum number: qC = qA + qB
    static int32_t fuse(int32_t qA, int32_t qB);
};

} // namespace tapp_con

#endif // TAPP_CON_U1_HANDLER_H
