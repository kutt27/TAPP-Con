import unittest

from tapp_con.constraint import LinearConstraint, parse_system, parse_u1, tokenize
from tapp_con._types import SymmetryType

OK_CASES = [
    ("q0 + q1 = 0", 0, [(0, 1), (1, 1)]),
    ("q0 = 5", 5, [(0, 1)]),
    ("2*q0 + q1 - q2 = 3", 3, [(0, 2), (1, 1), (2, -1)]),
    ("-q0 + q1 = 0", 0, [(0, -1), (1, 1)]),
    ("-2*q0 = 0", 0, [(0, -2)]),
]

ERROR_CASES = [
    ("2 q0 = 0", "Expected '*' after coefficient"),
    ("2q0 = 0", "Expected '*' after coefficient"),
    ("q0 + q1", "Expected '='"),
    ("q0 = 5 garbage", "Unexpected trailing characters"),
    ("q0 = -5", "Expected integer target value"),
    ("x0 = 0", "Expected identifier of form 'q<leg_index>'"),
    ("q = 0", "Expected identifier of form 'q<leg_index>'"),
    ("qX = 0", "Invalid leg index in 'qX'"),
    ("", "Expected '='"),
    ("q0 # 1 = 0", "Unexpected character '#'"),
]


class ParserTests(unittest.TestCase):
    def test_accepted(self):
        for expr, target, terms in OK_CASES:
            lc, err = parse_u1(expr)
            self.assertEqual(err, "", expr)
            self.assertIsInstance(lc, LinearConstraint)
            self.assertEqual(lc.target, target, expr)
            self.assertEqual(
                [(t.leg_index, t.coefficient) for t in lc.terms], terms, expr
            )

    def test_rejected(self):
        for expr, message in ERROR_CASES:
            lc, err = parse_u1(expr)
            self.assertIsNone(lc, expr)
            self.assertEqual(err, message, expr)

    def test_tokenizer_unexpected_char(self):
        tokens, err = tokenize("q0 # 1 = 0")
        self.assertIsNone(tokens)
        self.assertEqual(err, "Unexpected character '#'")

    def test_evaluate(self):
        lc, _ = parse_u1("2*q0 + q1 - q2 = 3")
        self.assertTrue(lc.evaluate([2, 1, 2]))  # 4 + 1 - 2 = 3
        self.assertFalse(lc.evaluate([1, 1, 2]))
        # out-of-range leg index -> always false
        lc2, _ = parse_u1("q5 = 0")
        self.assertFalse(lc2.evaluate([0, 0, 0]))


class ParseSystemTests(unittest.TestCase):
    def test_leg_sectors(self):
        res = parse_system(2, ["q0 + q1 = 0"], [3, 3], SymmetryType.U1, -1, 1)
        self.assertTrue(res.success)
        self.assertEqual(len(res.system.u1_constraints), 1)
        self.assertEqual(len(res.system.leg_sectors), 2)
        for ls in res.system.leg_sectors:
            self.assertEqual(ls.allowed_charges, [-1, 0, 1])

    def test_parse_error_propagates(self):
        res = parse_system(2, ["q0 + q1"], [3, 3], SymmetryType.U1, -1, 1)
        self.assertFalse(res.success)
        self.assertEqual(res.error_message, "Expected '='")


if __name__ == "__main__":
    unittest.main()
