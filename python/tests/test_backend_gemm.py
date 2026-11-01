import unittest

import numpy as np

import tapp_con as tc


def make_tensor(rows, cols, symmetry=tc.SymmetryType.U1):
    return tc.ConstrainedTensor(rows, cols, tc.DataType.FLOAT, symmetry)


class GemmTests(unittest.TestCase):
    def test_simple_2x2(self):
        a = make_tensor(2, 2)
        a.add_block(0, 0, 2, 2, 0, 0, [[1, 2], [3, 4]])
        b = make_tensor(2, 2)
        b.add_block(0, 0, 2, 2, 0, 0, [[5, 6], [7, 8]])
        c = make_tensor(2, 2)
        c.add_block(0, 0, 2, 2, 0, 0, None)

        tc.BlockGEMM.multiply(c, a, b, 1.0, 0.0)
        _, data = c.get_block(0)
        np.testing.assert_array_equal(data, [[19, 22], [43, 50]])

    def test_u1_block_selection(self):
        # A: q=0 at (0,0), q=1 at (2,2).  B: q=1 at (0,0), q=0 at (2,2).
        # Only qA + qB == qC pairs run, so C receives A00@B00 and A11@B11.
        a = make_tensor(4, 4)
        a.add_block(0, 0, 2, 2, 0, 0, [[1, 2], [3, 4]])
        a.add_block(2, 2, 2, 2, 1, 0, [[5, 6], [7, 8]])
        b = make_tensor(4, 4)
        b.add_block(0, 0, 2, 2, 1, 0, [[9, 10], [11, 12]])
        b.add_block(2, 2, 2, 2, 0, 0, [[13, 14], [15, 16]])
        c = make_tensor(4, 4)
        c.add_block(0, 0, 2, 2, 1, 0, None)
        c.add_block(2, 2, 2, 2, 1, 0, None)

        tc.BlockGEMM.multiply(c, a, b, 1.0, 0.0)
        _, c00 = c.get_block(0)
        _, c11 = c.get_block(1)
        np.testing.assert_array_equal(c00, [[31, 34], [71, 78]])
        np.testing.assert_array_equal(c11, [[155, 166], [211, 226]])

    def test_alpha_beta(self):
        a = make_tensor(2, 2)
        a.add_block(0, 0, 2, 2, 0, 0, [[1, 1], [1, 1]])
        b = make_tensor(2, 2)
        b.add_block(0, 0, 2, 2, 0, 0, [[1, 1], [1, 1]])
        c = make_tensor(2, 2)
        c.add_block(0, 0, 2, 2, 0, 0, np.full((2, 2), 10.0, dtype=np.float32))

        # A @ B = [[2, 2], [2, 2]]; 2 * that + 0.5 * 10 = 9.
        tc.BlockGEMM.multiply(c, a, b, 2.0, 0.5)
        _, data = c.get_block(0)
        np.testing.assert_allclose(data, np.full((2, 2), 9.0))

    def test_serial_transpose_a(self):
        a = make_tensor(2, 2)
        a.add_block(0, 0, 2, 2, 0, 0, [[1, 2], [3, 4]])
        b = make_tensor(2, 2)
        b.add_block(0, 0, 2, 2, 0, 0, [[5, 6], [7, 8]])
        c = make_tensor(2, 2)
        c.add_block(0, 0, 2, 2, 0, 0, None)

        tc.BlockGEMM.multiply(c, a, b, 1.0, 0.0, trans_a=True)
        _, data = c.get_block(0)
        np.testing.assert_array_equal(data, [[26, 30], [38, 44]])

    def test_type_mismatch_raises(self):
        a = make_tensor(2, 2)
        b = tc.ConstrainedTensor(2, 2, tc.DataType.DOUBLE, tc.SymmetryType.U1)
        c = make_tensor(2, 2)
        with self.assertRaises(tc.InvalidParameterError):
            tc.BlockGEMM.multiply(c, a, b, 1.0, 0.0)

    def test_parallel_matches_serial(self):
        nl = 4
        size = 3
        total = nl * size
        a = make_tensor(total, total)
        b = make_tensor(total, total)
        c1 = make_tensor(total, total)
        c2 = make_tensor(total, total)

        rng = np.random.default_rng(7)
        for i in range(nl):
            blk = rng.standard_normal((size, size)).astype(np.float32)
            a.add_block(i * size, i * size, size, size, 0, 0, blk)
            blk2 = rng.standard_normal((size, size)).astype(np.float32)
            b.add_block(i * size, i * size, size, size, 0, 0, blk2)
            c1.add_block(i * size, i * size, size, size, 0, 0, None)
            c2.add_block(i * size, i * size, size, size, 0, 0, None)

        _, plan = tc.plan_create(a, b, c1)
        tc.gemm(plan, 1.0, 0.0, threads=1)

        # Independent serial scan must agree.
        tc.BlockGEMM.multiply(c2, a, b, 1.0, 0.0)

        np.testing.assert_array_equal(c1.to_dense(), c2.to_dense())


class PlanTests(unittest.TestCase):
    def test_compute_block_map(self):
        a = make_tensor(4, 4)
        a.add_block(0, 0, 2, 2, 0, 0, None)
        a.add_block(2, 2, 2, 2, 1, 0, None)
        b = make_tensor(4, 4)
        b.add_block(0, 0, 2, 2, 1, 0, None)
        b.add_block(2, 2, 2, 2, 0, 0, None)
        c = make_tensor(4, 4)
        c.add_block(0, 0, 2, 2, 1, 0, None)
        c.add_block(2, 2, 2, 2, 1, 0, None)

        plan = tc.PlanData(a, b, c, tc.DataType.FLOAT)
        plan.compute_block_map()
        self.assertTrue(plan.triples_valid)
        pairs = [(t.a_idx, t.b_idx, t.c_idx) for t in plan.compatible_triples]
        self.assertEqual(pairs, [(0, 0, 0), (1, 1, 1)])

    def test_dimension_mismatch_leaves_plan_invalid(self):
        a = make_tensor(4, 4)
        a.add_block(0, 0, 4, 4, 0, 0, None)
        b = make_tensor(2, 2)
        b.add_block(0, 0, 2, 2, 0, 0, None)
        c = make_tensor(4, 2)
        c.add_block(0, 0, 4, 2, 0, 0, None)

        plan = tc.PlanData(a, b, c, tc.DataType.FLOAT)
        plan.compute_block_map()
        self.assertFalse(plan.triples_valid)


if __name__ == "__main__":
    unittest.main()
