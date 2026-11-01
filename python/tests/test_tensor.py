import unittest

import numpy as np

import tapp_con as tc
from tapp_con.errors import InvalidParameterError, OutOfRangeError


class TensorTests(unittest.TestCase):
    def test_construction(self):
        t = tc.ConstrainedTensor(4, 4, tc.DataType.DOUBLE, tc.SymmetryType.U1)
        self.assertEqual(t.total_rows, 4)
        self.assertEqual(t.total_cols, 4)
        self.assertEqual(t.num_blocks, 0)
        self.assertEqual(t.buffer_size, 0)
        self.assertEqual(t.element_size, 8)
        self.assertEqual(t.symmetry_type, tc.SymmetryType.U1)
        self.assertEqual(t.dtype, tc.DataType.DOUBLE)
        self.assertTrue(t.validate())

    def test_zero_dims_raise(self):
        with self.assertRaises(InvalidParameterError):
            tc.ConstrainedTensor(0, 4, tc.DataType.FLOAT, tc.SymmetryType.U1)
        with self.assertRaises(InvalidParameterError):
            tc.ConstrainedTensor(4, 0, tc.DataType.FLOAT, tc.SymmetryType.U1)

    def test_element_sizes(self):
        self.assertEqual(
            tc.ConstrainedTensor(2, 2, tc.DataType.FLOAT, tc.SymmetryType.NONE).element_size, 4
        )
        self.assertEqual(
            tc.ConstrainedTensor(2, 2, tc.DataType.DOUBLE, tc.SymmetryType.NONE).element_size, 8
        )

    def test_add_block_returns_index_and_copies(self):
        t = tc.ConstrainedTensor(4, 4, tc.DataType.FLOAT, tc.SymmetryType.U1)
        idx = t.add_block(0, 0, 2, 2, 0, 0, [[1, 2], [3, 4]])
        self.assertEqual(idx, 0)
        self.assertEqual(t.num_blocks, 1)
        blk, data = t.get_block(0)
        self.assertEqual(blk.row_offset, 0)
        self.assertEqual(blk.col_offset, 0)
        self.assertEqual(blk.num_rows, 2)
        self.assertEqual(blk.num_cols, 2)
        self.assertEqual(blk.u1_qnum, 0)
        self.assertEqual(blk.data_offset, 0)
        np.testing.assert_array_equal(data, [[1, 2], [3, 4]])

    def test_buffer_is_tightly_packed(self):
        t = tc.ConstrainedTensor(4, 4, tc.DataType.FLOAT, tc.SymmetryType.U1)
        t.add_block(0, 0, 2, 2, 0, 0, [[1, 2], [3, 4]])
        t.add_block(0, 2, 2, 2, 1, 0, [[5, 6], [7, 8]])
        self.assertEqual(t.num_blocks, 2)
        self.assertEqual(t.block(1).u1_qnum, 1)
        # 2x2 float block = 16 bytes, so the second block starts at offset 16.
        self.assertEqual(t.block(1).data_offset, 16)
        self.assertEqual(t.buffer_size, 32)

    def test_null_data_is_zero_initialized(self):
        t = tc.ConstrainedTensor(3, 3, tc.DataType.FLOAT, tc.SymmetryType.U1)
        t.add_block(0, 0, 3, 3, 0, 0, None)
        _, data = t.get_block(0)
        np.testing.assert_array_equal(data, np.zeros((3, 3), dtype=np.float32))

    def test_out_of_bounds_raises(self):
        t = tc.ConstrainedTensor(4, 4, tc.DataType.FLOAT, tc.SymmetryType.U1)
        with self.assertRaises(OutOfRangeError):
            t.add_block(3, 3, 2, 2, 0, 0, None)

    def test_validate_overlap_rules(self):
        t = tc.ConstrainedTensor(4, 4, tc.DataType.FLOAT, tc.SymmetryType.U1)
        t.add_block(0, 0, 2, 2, 0, 0, None)
        t.add_block(2, 2, 2, 2, 0, 0, None)
        self.assertTrue(t.validate())

        bad = tc.ConstrainedTensor(4, 4, tc.DataType.FLOAT, tc.SymmetryType.U1)
        bad.add_block(0, 0, 3, 3, 0, 0, None)
        bad.add_block(1, 1, 2, 2, 0, 0, None)
        self.assertFalse(bad.validate())

        adjacent = tc.ConstrainedTensor(4, 4, tc.DataType.FLOAT, tc.SymmetryType.U1)
        adjacent.add_block(0, 0, 2, 2, 0, 0, None)
        adjacent.add_block(0, 2, 2, 2, 0, 0, None)
        self.assertTrue(adjacent.validate())

    def test_to_dense(self):
        t = tc.ConstrainedTensor(4, 4, tc.DataType.FLOAT, tc.SymmetryType.U1)
        t.add_block(0, 0, 2, 2, 0, 0, [[1, 2], [3, 4]])
        t.add_block(3, 3, 1, 1, 0, 0, [[9]])
        expected = np.zeros((4, 4), dtype=np.float32)
        expected[:2, :2] = [[1, 2], [3, 4]]
        expected[3, 3] = 9
        np.testing.assert_array_equal(t.to_dense(), expected)

    def test_get_block_writable_view(self):
        t = tc.ConstrainedTensor(2, 2, tc.DataType.FLOAT, tc.SymmetryType.U1)
        t.add_block(0, 0, 2, 2, 0, 0, None)
        _, data = t.get_block(0)
        data[0, 0] = 5.0
        _, again = t.get_block(0)
        self.assertEqual(again[0, 0], 5.0)


if __name__ == "__main__":
    unittest.main()
