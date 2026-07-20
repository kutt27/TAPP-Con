#ifndef TAPP_CON_CONSTRAINED_TENSOR_H
#define TAPP_CON_CONSTRAINED_TENSOR_H

#include <cstdint>
#include <vector>
#include <cstring>

#include "block/block_descriptor.h"

namespace tapp_con {

enum class SymmetryType : uint8_t {
    NONE   = 0,
    U1     = 1,
    SU2    = 2,
    U1XSU2 = 3
};

enum class DataType : uint8_t {
    FLOAT          = 0,
    DOUBLE         = 1,
    COMPLEX_FLOAT  = 2,
    COMPLEX_DOUBLE = 3
};

inline uint64_t data_type_size(DataType dtype) noexcept {
    switch (dtype) {
        case DataType::FLOAT:          return 4;
        case DataType::DOUBLE:         return 8;
        case DataType::COMPLEX_FLOAT:  return 8;
        case DataType::COMPLEX_DOUBLE: return 16;
        default:                       return 0;
    }
}

class ConstrainedTensor {
public:
    ConstrainedTensor(uint64_t total_rows, uint64_t total_cols,
                      DataType dtype, SymmetryType sym);
    ~ConstrainedTensor() = default;

    ConstrainedTensor(ConstrainedTensor&&) noexcept = default;
    ConstrainedTensor& operator=(ConstrainedTensor&&) noexcept = default;
    ConstrainedTensor(const ConstrainedTensor&) = delete;
    ConstrainedTensor& operator=(const ConstrainedTensor&) = delete;

    // Block management
    size_t add_block(uint64_t row_offset, uint64_t col_offset,
                     uint64_t block_rows, uint64_t block_cols,
                     int32_t u1_qnum, int32_t su2_spin_times_two,
                     const void* data);

    const BlockDescriptor& block(size_t index) const;
    size_t num_blocks() const noexcept;

    // Queries
    SymmetryType symmetry_type() const noexcept;
    DataType data_type() const noexcept;

    void*       raw_buffer() noexcept;
    const void* raw_buffer() const noexcept;
    uint64_t    total_rows() const noexcept;
    uint64_t    total_cols() const noexcept;
    uint64_t    buffer_size() const noexcept;
    uint64_t    element_size() const noexcept;

    // Validation: returns true if all blocks are within bounds and non-overlapping.
    bool validate() const;

private:
    uint64_t total_rows_;
    uint64_t total_cols_;
    DataType dtype_;
    SymmetryType sym_type_;
    std::vector<uint8_t> raw_buffer_;
    std::vector<BlockDescriptor> blocks_;
};

} // namespace tapp_con

#endif // TAPP_CON_CONSTRAINED_TENSOR_H
