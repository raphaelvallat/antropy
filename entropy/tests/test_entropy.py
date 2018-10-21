import numpy as np
import unittest

from entropy import (perm_entropy, spectral_entropy, svd_entropy,
                     sample_entropy, app_entropy)

np.random.seed(1234567)
RANDOM_TS = np.random.rand(3000)
RANDOM_TS_LONG = np.random.rand(6000)
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
        # Error
        with self.assertRaises(ValueError):
            perm_entropy(BANDT_PERM, order=4, delay=3)
        with self.assertRaises(ValueError):
            perm_entropy(BANDT_PERM, order=3, delay=0.5)
        with self.assertRaises(ValueError):
            perm_entropy(BANDT_PERM, order=1, delay=1)

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

    def test_sample_entropy(self):
        se = sample_entropy(RANDOM_TS, order=2)
        sample_entropy(RANDOM_TS_LONG, order=2)
        se_eu_3 = sample_entropy(RANDOM_TS, order=3, metric='euclidean')
        # Compare with MNE-features
        self.assertEqual(np.round(se, 3), 2.192)
        self.assertEqual(np.round(se_eu_3, 3), 2.725)
        sample_entropy(RANDOM_TS, order=3)
        sample_entropy(RANDOM_TS, order=2, metric='euclidean')
        with self.assertRaises(ValueError):
            sample_entropy(RANDOM_TS, order=2, metric='wrong')

    def test_app_entropy(self):
        ae = app_entropy(RANDOM_TS, order=2)
        ae_eu_3 = app_entropy(RANDOM_TS, order=3, metric='euclidean')
        # Compare with MNE-features
        self.assertEqual(np.round(ae, 3), 2.075)
        self.assertEqual(np.round(ae_eu_3, 3), 0.956)
        app_entropy(RANDOM_TS, order=3)
        with self.assertRaises(ValueError):
            app_entropy(RANDOM_TS, order=2, metric='wrong')
