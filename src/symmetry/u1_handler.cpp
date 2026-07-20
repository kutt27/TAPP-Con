#include "u1_handler.h"

namespace tapp_con {

bool U1Handler::compatible(int32_t qA, int32_t qB, int32_t qC) {
    return (qA + qB) == qC;
}

int32_t U1Handler::fuse(int32_t qA, int32_t qB) {
    return qA + qB;
}

} // namespace tapp_con
