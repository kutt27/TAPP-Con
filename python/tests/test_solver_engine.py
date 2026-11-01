import unittest

import numpy as np

import tapp_con as tc
from tapp_con._types import SymmetryType
from tapp_con.constraint import build_tensor_from_strings, generate_blocks, parse_system
from tapp_con.constraint.parser import _generate_charges


class SectorGenerationTests(unittest.TestCase):
    def test_two_branch_generation(self):
        # (num, lo, hi) -> charges, verified against the C++ parser.
        cases = [
            ((3, -1, 1), [-1, 0, 1]),
            ((1, -1, 1), [-1]),
            ((2, -1, 1), [-1, 1]),
            ((4, -1, 1), [-1, 0, 1, 2]),
            ((5, -3, 3), [-3, -2, 0, 1, 3]),
            ((8, -7, 7), [-7, -5, -3, -1, 1, 3, 5, 7]),
            ((10, -1, 1), [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]),
        ]
        for (num, lo, hi), expected in cases:
            self.assertEqual(_generate_charges(num, lo, hi), expected, (num, lo, hi))


class GenerateBlocksTests(unittest.TestCase):
    def setUp(self):
        self.system = parse_system(2, ["q0 + q1 = 0"], [3, 3], SymmetryType.U1, -1, 1).system

    def test_expected_blocks(self):
        blocks = generate_blocks(self.system, [0], [1], [6, 6])
        self.assertEqual(len(blocks), 3)
        # sorted by (row_offset, col_offset); the su2 field carries u1_vals[0].
        self.assertEqual(
            [
                (b.row_offset, b.col_offset, b.num_rows, b.num_cols, b.u1_qnum,
                 b.su2_spin_times_two)
                for b in blocks
            ],
            [(0, 4, 2, 2, 0, -1), (2, 2, 2, 2, 0, 0), (4, 0, 2, 2, 0, 1)],
        )

    def test_impossible_target_yields_no_blocks(self):
        system = parse_system(2, ["q0 + q1 = 10"], [3, 3], SymmetryType.U1, -1, 1).system
        self.assertEqual(generate_blocks(system, [0], [1], [6, 6]), [])


class BuildTensorTests(unittest.TestCase):
    def test_build_from_strings(self):
        t = build_tensor_from_strings(
            2, ["q0 + q1 = 0"], [3, 3], [0], [1], [6, 6],
            tc.DataType.FLOAT, SymmetryType.U1, -1, 1,
        )
        self.assertEqual(t.total_rows, 6)
        self.assertEqual(t.total_cols, 6)
        self.assertEqual(t.num_blocks, 3)
        self.assertTrue(t.validate())
        self.assertEqual(t.buffer_size, 3 * (2 * 2) * 4)  # 48 bytes

    def test_end_to_end_gemm(self):
        # A all-ones blocks, B = 2*identity blocks -> C = 2 everywhere.
        n = 3
        dtype = tc.DataType.FLOAT
        sym = SymmetryType.U1
        a = tc.ConstrainedTensor(6, 6, dtype, sym)
        b = tc.ConstrainedTensor(6, 6, dtype, sym)
        c = tc.ConstrainedTensor(6, 6, dtype, sym)
        for ro, co in [(0, 0), (2, 2), (4, 4)]:
            a.add_block(ro, co, 2, 2, 0, 0, np.ones((2, 2), dtype=np.float32))
            b.add_block(ro, co, 2, 2, 0, 0, 2 * np.eye(2, dtype=np.float32))
            c.add_block(ro, co, 2, 2, 0, 0, None)

        status, plan = tc.plan_create(a, b, c)
        self.assertEqual(status, tc.Status.SUCCESS)
        self.assertEqual(tc.gemm(plan, 1.0, 0.0), tc.Status.SUCCESS)
        for i in range(3):
            _, data = c.get_block(i)
            np.testing.assert_allclose(data, np.full((2, 2), 2.0))


if __name__ == "__main__":
    unittest.main()
