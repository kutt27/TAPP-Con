import unittest

from tapp_con.symmetry import SU2Handler, U1Handler


class U1Tests(unittest.TestCase):
    def test_compatible(self):
        for a, b, c, expected in [
            (0, 0, 0, True),
            (1, 2, 3, True),
            (-1, 1, 0, True),
            (5, -3, 2, True),
            (0, 0, 1, False),
            (1, 1, 3, False),
            (-1, -1, 0, False),
            (2, 3, 4, False),
        ]:
            self.assertEqual(U1Handler.compatible(a, b, c), expected, (a, b, c))

    def test_fuse(self):
        self.assertEqual(U1Handler.fuse(0, 0), 0)
        self.assertEqual(U1Handler.fuse(1, 2), 3)
        self.assertEqual(U1Handler.fuse(-1, 1), 0)
        self.assertEqual(U1Handler.fuse(100, -100), 0)


class SU2Tests(unittest.TestCase):
    def test_compatible(self):
        for a, b, c, expected in [
            (1, 1, 0, True),
            (1, 1, 2, True),
            (1, 1, 1, False),
            (1, 1, 4, False),
            (2, 2, 0, True),
            (2, 2, 2, True),
            (2, 2, 4, True),
            (2, 2, 6, False),
            (2, 1, 1, True),
            (2, 1, 3, True),
            (2, 1, 5, False),
            (1, 0, 1, True),
            (1, 0, 3, False),
        ]:
            self.assertEqual(SU2Handler.compatible(a, b, c), expected, (a, b, c))

    def test_allowed_fusions(self):
        self.assertEqual(SU2Handler.allowed_fusions(1, 1), [0, 2])
        self.assertEqual(SU2Handler.allowed_fusions(2, 2), [0, 2, 4])
        self.assertEqual(SU2Handler.allowed_fusions(2, 1), [1, 3])

    def test_clebsch_gordan_placeholder(self):
        self.assertEqual(SU2Handler.clebsch_gordan(1, 1, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
