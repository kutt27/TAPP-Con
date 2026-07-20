#include "constrained_tensor.h"
#include <algorithm>
#include <stdexcept>

namespace tapp_con {

ConstrainedTensor::ConstrainedTensor(uint64_t total_rows, uint64_t total_cols,
                                     DataType dtype, SymmetryType sym)
    : total_rows_(total_rows)
    , total_cols_(total_cols)
    , dtype_(dtype)
    , sym_type_(sym)
{
    if (total_rows == 0 || total_cols == 0)
        throw std::invalid_argument("Tensor dimensions must be non-zero");
}

size_t ConstrainedTensor::add_block(uint64_t row_offset, uint64_t col_offset,
                                    uint64_t block_rows, uint64_t block_cols,
                                    int32_t u1_qnum, int32_t su2_spin_times_two,
                                    const void* data)
{
    if (row_offset + block_rows > total_rows_ ||
        col_offset + block_cols > total_cols_)
        throw std::out_of_range("Block extends beyond tensor bounds");

    uint64_t block_bytes = block_rows * block_cols * data_type_size(dtype_);
    uint64_t data_offset = raw_buffer_.size();

    // Extend buffer and copy data
    raw_buffer_.resize(raw_buffer_.size() + block_bytes);
    if (data) {
        std::memcpy(raw_buffer_.data() + data_offset, data, block_bytes);
    }

    blocks_.push_back(BlockDescriptor{
        row_offset,
        col_offset,
        block_rows,
        block_cols,
        u1_qnum,
        su2_spin_times_two,
        data_offset
    });

    return blocks_.size() - 1;
}

const BlockDescriptor& ConstrainedTensor::block(size_t index) const {
    return blocks_.at(index);
}

size_t ConstrainedTensor::num_blocks() const noexcept {
    return blocks_.size();
}

SymmetryType ConstrainedTensor::symmetry_type() const noexcept {
    return sym_type_;
}

DataType ConstrainedTensor::data_type() const noexcept {
    return dtype_;
}

void* ConstrainedTensor::raw_buffer() noexcept {
    return raw_buffer_.data();
}

const void* ConstrainedTensor::raw_buffer() const noexcept {
    return raw_buffer_.data();
}

uint64_t ConstrainedTensor::total_rows() const noexcept {
    return total_rows_;
}

uint64_t ConstrainedTensor::total_cols() const noexcept {
    return total_cols_;
}

uint64_t ConstrainedTensor::buffer_size() const noexcept {
    return raw_buffer_.size();
}

uint64_t ConstrainedTensor::element_size() const noexcept {
    return data_type_size(dtype_);
}

bool ConstrainedTensor::validate() const {
    // Check each block is within bounds
    for (const auto& b : blocks_) {
        if (b.row_offset + b.num_rows > total_rows_)
            return false;
        if (b.col_offset + b.num_cols > total_cols_)
            return false;
    }

    // Check no two blocks overlap
    for (size_t i = 0; i < blocks_.size(); ++i) {
        for (size_t j = i + 1; j < blocks_.size(); ++j) {
            const auto& a = blocks_[i];
            const auto& b = blocks_[j];

            bool overlap_row = a.row_offset < b.row_offset + b.num_rows &&
                               b.row_offset < a.row_offset + a.num_rows;
            bool overlap_col = a.col_offset < b.col_offset + b.num_cols &&
                               b.col_offset < a.col_offset + a.num_cols;

            if (overlap_row && overlap_col)
                return false;
        }
    }

    return true;
}

} // namespace tapp_con
