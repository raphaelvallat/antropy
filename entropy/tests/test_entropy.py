import numpy as np
import unittest

from entropy import perm_entropy, spectral_entropy, svd_entropy

np.random.seed(1234567)
RANDOM_TS = np.random.rand(3000)
SF_TS = 100
BANDT_PERM = [4, 7, 9, 10, 6, 11, 3]
PURE_SINE = np.sin(2 * np.pi * 1 * np.arange(3000) / 100)


class TestEntropy(unittest.TestCase):

    def test_perm_entropy(self):
        self.assertEqual(np.round(perm_entropy(RANDOM_TS, order=3,
                                               delay=1, normalize=True), 1),
                         1.0)
        # Compare with Bandt and Pompe 2002
        self.assertEqual(np.round(perm_entropy(BANDT_PERM, order=2), 3), 0.918)
        self.assertEqual(np.round(perm_entropy(BANDT_PERM, order=3), 3), 1.522)

    def test_spectral_entropy(self):
        spectral_entropy(RANDOM_TS, SF_TS, method='fft')
        spectral_entropy(RANDOM_TS, SF_TS, method='welch')
        spectral_entropy(RANDOM_TS, SF_TS, method='welch', nperseg=400)
        self.assertEqual(np.round(spectral_entropy(RANDOM_TS, SF_TS,
                                                   normalize=True), 1), 0.9)
        self.assertEqual(np.round(spectral_entropy(PURE_SINE, 100), 2), 0.0)

    def test_svd_entropy(self):
        svd_entropy(RANDOM_TS, order=3, delay=1, normalize=False)
        svd_entropy(RANDOM_TS, order=3, delay=1, normalize=True)
        svd_entropy(RANDOM_TS, order=2, delay=1, normalize=False)
        svd_entropy(RANDOM_TS, order=3, delay=2, normalize=False)


if __name__ == '__main__':
    unittest.main()
