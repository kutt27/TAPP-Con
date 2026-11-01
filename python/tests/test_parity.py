"""Parity gate: run the real C++ library's reference dumper and assert the NumPy
port reproduces every value.

The dumper is built by the repository build step at
``build/tools/dump_reference``. If it is missing the test is skipped, so the pure
Python suite still runs standalone.
"""

import json
import os
import subprocess
import unittest
from pathlib import Path

import numpy as np

import tapp_con as tc
from tapp_con._types import SymmetryType
from tapp_con.constraint import generate_blocks, parse_system, parse_u1
from tapp_con.constraint.parser import _generate_charges

REPO_ROOT = Path(__file__).resolve().parents[2]
DUMPER = Path(
    os.environ.get("TAPP_DUMP_REFERENCE", REPO_ROOT / "build" / "tools" / "dump_reference")
)


def load_reference():
    out = subprocess.run(
        [str(DUMPER)], check=True, capture_output=True, text=True
    ).stdout
    return json.loads(out)


def build_gemm_fixture(u1_blocks, init_c_nonzero):
    """Mirror dump_reference.cpp's dump_gemm_case input construction."""
    sym = tc.SymmetryType.U1
    a = tc.ConstrainedTensor(4, 4, tc.DataType.FLOAT, sym)
    b = tc.ConstrainedTensor(4, 4, tc.DataType.FLOAT, sym)
    c = tc.ConstrainedTensor(4, 4, tc.DataType.FLOAT, sym)

    if u1_blocks:
        a.add_block(0, 0, 2, 2, 0, 0, [[1, 2], [3, 4]])
        a.add_block(2, 2, 2, 2, 1, 0, [[5, 6], [7, 8]])
        b.add_block(0, 0, 2, 2, 1, 0, [[9, 10], [11, 12]])
        b.add_block(2, 2, 2, 2, 0, 0, [[13, 14], [15, 16]])
        c.add_block(0, 0, 2, 2, 1, 0, None)
        c.add_block(2, 2, 2, 2, 1, 0, None)
    else:
        afull = np.array(
            [1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 5, 6, 0, 0, 7, 8], dtype=np.float32
        ).reshape(4, 4)
        bfull = np.array(
            [5, 6, 0, 0, 7, 8, 0, 0, 0, 0, 9, 10, 0, 0, 11, 12], dtype=np.float32
        ).reshape(4, 4)
        a.add_block(0, 0, 4, 4, 0, 0, afull)
        b.add_block(0, 0, 4, 4, 0, 0, bfull)
        c.add_block(0, 0, 4, 4, 0, 0, None)

    if init_c_nonzero:
        for k in range(c.num_blocks):
            _, data = c.get_block(k)
            data[...] = 10.0

    return a, b, c


@unittest.skipUnless(DUMPER.exists(), f"reference dumper not built at {DUMPER}")
class ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref = load_reference()

    def test_parser(self):
        for case in self.ref["parser"]:
            lc, err = parse_u1(case["input"])
            if case["ok"]:
                self.assertEqual(err, "", case["input"])
                self.assertEqual(lc.target, case["target"], case["input"])
                self.assertEqual(
                    [[t.leg_index, t.coefficient] for t in lc.terms],
                    case["terms"],
                    case["input"],
                )
            else:
                self.assertIsNone(lc, case["input"])
                self.assertEqual(err, case["error"], case["input"])

    def test_parse_system_leg_charges(self):
        res = parse_system(2, ["q0 + q1 = 0"], [3, 3], SymmetryType.U1, -1, 1)
        self.assertEqual(res.success, self.ref["parse_system"]["success"])
        self.assertEqual(
            [ls.allowed_charges for ls in res.system.leg_sectors],
            self.ref["parse_system"]["leg_charges"],
        )

    def test_generate_blocks(self):
        system = parse_system(2, ["q0 + q1 = 0"], [3, 3], SymmetryType.U1, -1, 1).system
        blocks = generate_blocks(system, [0], [1], [6, 6])
        got = [
            [b.row_offset, b.col_offset, b.num_rows, b.num_cols, b.u1_qnum,
             b.su2_spin_times_two]
            for b in blocks
        ]
        self.assertEqual(got, self.ref["blocks"])

    def test_sector_generation(self):
        for case in self.ref["sectors"]:
            self.assertEqual(
                _generate_charges(case["num"], case["lo"], case["hi"]),
                case["charges"],
                case,
            )

    def test_build_from_constraints(self):
        ref = self.ref["build"]
        status, t = tc.build_from_constraints(
            2, ["q0 + q1 = 0"], [3, 3], [0], [1], [6, 6],
            tc.TAPP_CON_FLOAT, tc.TAPP_CON_SYMMETRY_U1, -1, 1,
        )
        self.assertEqual(int(status), ref["status"])
        self.assertEqual(t.total_rows, ref["total_rows"])
        self.assertEqual(t.total_cols, ref["total_cols"])
        self.assertEqual(t.num_blocks, ref["num_blocks"])
        self.assertEqual(t.buffer_size, ref["buffer_size"])
        self.assertEqual(t.validate(), ref["valid"])
        got = [
            [b.row_offset, b.col_offset, b.num_rows, b.num_cols, b.u1_qnum]
            for b in t.blocks
        ]
        self.assertEqual(got, ref["blocks"])

    def _check_gemm(self, name, u1_blocks, alpha, beta, transpose_a, init_c):
        ref = self.ref[name]
        a, b, c = build_gemm_fixture(u1_blocks, init_c)
        status, plan = tc.plan_create(
            a, b, c, tc.TAPP_CON_FLOAT, 1 if transpose_a else 0, 0
        )
        self.assertEqual(status, tc.Status.SUCCESS)
        self.assertEqual(tc.gemm(plan, alpha, beta), tc.Status.SUCCESS)

        got_blocks = [
            [blk.row_offset, blk.col_offset, blk.num_rows, blk.num_cols, blk.u1_qnum]
            for blk in c.blocks
        ]
        self.assertEqual(got_blocks, ref["blocks"], name)

        got_data = []
        for k in range(c.num_blocks):
            _, data = c.get_block(k)
            got_data.extend(np.asarray(data, dtype=np.float64).reshape(-1).tolist())
        np.testing.assert_allclose(got_data, ref["c_data"], rtol=1e-6, atol=1e-6, err_msg=name)

    def test_gemm_simple(self):
        self._check_gemm("gemm_simple", False, 1.0, 0.0, False, False)

    def test_gemm_u1_blocks(self):
        self._check_gemm("gemm_u1_blocks", True, 1.0, 0.0, False, False)

    def test_gemm_alpha_beta(self):
        self._check_gemm("gemm_alpha_beta", True, 2.0, 0.5, False, True)

    def test_gemm_transpose_a(self):
        # The plan path (multiply_parallel) ignores transpose; both libraries
        # therefore compute A @ B, not A^T @ B. Reproduced intentionally.
        self._check_gemm("gemm_transpose_a", False, 1.0, 0.0, True, False)

    def test_symmetry_rules(self):
        for a, b, c, expected in self.ref["symmetry"]["u1"]:
            self.assertEqual(tc.U1Handler.compatible(a, b, c), bool(expected))
        for a, b, c, expected in self.ref["symmetry"]["su2"]:
            self.assertEqual(tc.SU2Handler.compatible(a, b, c), bool(expected))


if __name__ == "__main__":
    unittest.main()
