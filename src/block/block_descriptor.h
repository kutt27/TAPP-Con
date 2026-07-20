#ifndef TAPP_CON_BLOCK_DESCRIPTOR_H
#define TAPP_CON_BLOCK_DESCRIPTOR_H

#include <cstdint>

namespace tapp_con {

struct BlockDescriptor {
    uint64_t row_offset;          // row start in logical matrix
    uint64_t col_offset;          // col start in logical matrix
    uint64_t num_rows;            // block height
    uint64_t num_cols;            // block width
    int32_t  u1_qnum;             // U(1) quantum number
    int32_t  su2_spin_times_two;  // SU(2) spin x 2 (1 = spin-1/2, 2 = spin-1, ...)
    uint64_t data_offset;         // byte offset into raw buffer
};

} // namespace tapp_con

#endif // TAPP_CON_BLOCK_DESCRIPTOR_H
