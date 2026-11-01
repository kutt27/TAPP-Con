import unittest

import numpy as np

import tapp_con as tc


class ApiTests(unittest.TestCase):
    def test_lifecycle(self):
        status, t = tc.tensor_create(tc.TAPP_CON_DOUBLE, 4, 4, tc.TAPP_CON_SYMMETRY_U1)
        self.assertEqual(status, tc.Status.SUCCESS)
        self.assertEqual(tc.tensor_num_blocks(t)[1], 0)
        self.assertEqual(tc.tensor_destroy(t), tc.Status.SUCCESS)

    def test_errors(self):
        self.assertEqual(
            tc.tensor_create(tc.TAPP_CON_DOUBLE, 0, 4, tc.TAPP_CON_SYMMETRY_U1)[0],
            tc.Status.INVALID_PARAMETER,
        )
        self.assertEqual(tc.tensor_destroy(None), tc.Status.INVALID_TENSOR)
        self.assertEqual(tc.plan_destroy(None), tc.Status.INVALID_PLAN)
        self.assertEqual(tc.gemm(None, 1.0, 0.0), tc.Status.INVALID_PLAN)

    def test_add_get_block(self):
        _, t = tc.tensor_create(tc.TAPP_CON_FLOAT, 4, 4, tc.TAPP_CON_SYMMETRY_U1)
        status = tc.tensor_add_block(t, 0, 0, 2, 2, 0, 0, [[1, 2], [3, 4]])
        self.assertEqual(status, tc.Status.SUCCESS)
        self.assertEqual(tc.tensor_num_blocks(t)[1], 1)

        status, blk, data = tc.tensor_get_block(t, 0)
        self.assertEqual(status, tc.Status.SUCCESS)
        self.assertEqual(
            (blk.row_offset, blk.col_offset, blk.num_rows, blk.num_cols, blk.u1_qnum),
            (0, 0, 2, 2, 0),
        )
        self.assertEqual(float(data[0, 0]), 1.0)
        self.assertEqual(float(data[1, 1]), 4.0)

        self.assertEqual(tc.tensor_get_block(t, 9)[0], tc.Status.INVALID_PARAMETER)

    def test_validate(self):
        _, t = tc.tensor_create(tc.TAPP_CON_FLOAT, 4, 4, tc.TAPP_CON_SYMMETRY_U1)
        tc.tensor_add_block(t, 0, 0, 2, 2, 0, 0, None)
        tc.tensor_add_block(t, 2, 2, 2, 2, 0, 0, None)
        self.assertEqual(tc.tensor_validate(t), tc.Status.SUCCESS)

        _, bad = tc.tensor_create(tc.TAPP_CON_FLOAT, 4, 4, tc.TAPP_CON_SYMMETRY_U1)
        tc.tensor_add_block(bad, 0, 0, 3, 3, 0, 0, None)
        tc.tensor_add_block(bad, 1, 1, 2, 2, 0, 0, None)
        self.assertEqual(tc.tensor_validate(bad), tc.Status.BLOCK_OVERLAP)

    def test_build_from_constraints(self):
        status, t = tc.build_from_constraints(
            2, ["q0 + q1 = 0"], [3, 3], [0], [1], [6, 6],
            tc.TAPP_CON_FLOAT, tc.TAPP_CON_SYMMETRY_U1, -1, 1,
        )
        self.assertEqual(status, tc.Status.SUCCESS)
        self.assertEqual(tc.tensor_num_blocks(t)[1], 3)

        bad, _ = tc.build_from_constraints(
            2, ["q0 + q1"], [3, 3], [0], [1], [6, 6],
            tc.TAPP_CON_FLOAT, tc.TAPP_CON_SYMMETRY_U1, -1, 1,
        )
        self.assertEqual(bad, tc.Status.INVALID_PARAMETER)

    def test_end_to_end_gemm(self):
        _, a = tc.tensor_create(tc.TAPP_CON_FLOAT, 2, 2, tc.TAPP_CON_SYMMETRY_U1)
        tc.tensor_add_block(a, 0, 0, 2, 2, 0, 0, [[1, 2], [3, 4]])
        _, b = tc.tensor_create(tc.TAPP_CON_FLOAT, 2, 2, tc.TAPP_CON_SYMMETRY_U1)
        tc.tensor_add_block(b, 0, 0, 2, 2, 0, 0, [[5, 6], [7, 8]])
        _, c = tc.tensor_create(tc.TAPP_CON_FLOAT, 2, 2, tc.TAPP_CON_SYMMETRY_U1)
        tc.tensor_add_block(c, 0, 0, 2, 2, 0, 0, None)

        status, plan = tc.plan_create(a, b, c, tc.TAPP_CON_FLOAT, 0, 0)
        self.assertEqual(status, tc.Status.SUCCESS)
        self.assertEqual(tc.gemm(plan, 1.0, 0.0), tc.Status.SUCCESS)

        _, _, data = tc.tensor_get_block(c, 0)
        np.testing.assert_array_equal(data, [[19, 22], [43, 50]])

    def test_error_strings(self):
        self.assertEqual(tc.error_string(tc.TAPP_CON_SUCCESS), "Success")
        self.assertEqual(tc.error_string(tc.TAPP_CON_INVALID_TENSOR), "Invalid tensor handle")
        self.assertEqual(tc.error_string(tc.TAPP_CON_MEMORY_ERROR), "Memory allocation failed")
        self.assertEqual(tc.error_string(-99), "Unknown error")

    def test_c_symbol_aliases_exist(self):
        self.assertIs(tc.tapp_constrained_tensor_create, tc.tensor_create)
        self.assertIs(tc.tapp_constrained_gemm, tc.gemm)


if __name__ == "__main__":
    unittest.main()
