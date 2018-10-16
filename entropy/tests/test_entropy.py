import numpy as np
import unittest

from entropy import perm_entropy

np.random.seed(1234567)
RANDOM_TS = np.random.rand(1000)
BANDT_PERM = [4, 7, 9, 10, 6, 11, 3]


class TestEntropy(unittest.TestCase):

    def test_permutation_entropy(self):
        self.assertEqual(np.round(perm_entropy(RANDOM_TS, order=3,
                                               delay=1, normalize=True), 3),
                         0.999)
        # Compare with Bandt and Pompe 2002
        self.assertEqual(np.round(perm_entropy(BANDT_PERM, order=2), 3), 0.918)
        self.assertEqual(np.round(perm_entropy(BANDT_PERM, order=3), 3), 1.522)


if __name__ == '__main__':
    unittest.main()
