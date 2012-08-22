import unittest

from tsm_solver import KohonenTSPSolver

class TestKohonenTSPSolver(unittest.TestCase):
    "This tests the TSM Solver class."
    def setUp(self):
        self.dummy_cities = [(1.,1.), (1.,2.), (2.,1.)]
        self.kts = KohonenTSPSolver(self.dummy_cities)

    def test_shuffle(self):
        # test euclidean norm
        result = self.kts._euclidean_norm([3,4])

        self.assertEqual(result, 5)


if __name__ == '__main__':
    unittest.main()
