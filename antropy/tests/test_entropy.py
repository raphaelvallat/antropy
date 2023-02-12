"""Test entropy functions."""
import unittest
import numpy as np
from numpy.testing import assert_equal
from numpy import apply_along_axis as aal
from antropy import (
    perm_entropy,
    spectral_entropy,
    svd_entropy,
    sample_entropy,
    app_entropy,
    lziv_complexity,
    num_zerocross,
    hjorth_params,
)

from antropy.utils import _xlogx

from utils import RANDOM_TS, NORMAL_TS, RANDOM_TS_LONG, PURE_SINE, ARANGE, TEST_DTYPES

SF_TS = 100
BANDT_PERM = [4, 7, 9, 10, 6, 11, 3]

# Concatenate 2D data
data = np.vstack((RANDOM_TS, NORMAL_TS, PURE_SINE, ARANGE))


class TestEntropy(unittest.TestCase):
    def test_perm_entropy(self):
        self.assertEqual(
            np.round(perm_entropy(RANDOM_TS, order=3, delay=1, normalize=True), 1), 1.0
        )
        # Compare with Bandt and Pompe 2002
        self.assertEqual(np.round(perm_entropy(BANDT_PERM, order=2), 3), 0.918)
        self.assertEqual(np.round(perm_entropy(BANDT_PERM, order=3), 3), 1.522)

        # Average of multiple delays
        assert isinstance(perm_entropy(RANDOM_TS, order=3, delay=[1, 2, 3]), float)
        # Error
        with self.assertRaises(ValueError):
            perm_entropy(BANDT_PERM, order=4, delay=3)
        with self.assertRaises(ValueError):
            perm_entropy(BANDT_PERM, order=3, delay=0.5)
        with self.assertRaises(ValueError):
            perm_entropy(BANDT_PERM, order=1, delay=1)

    def test_spectral_entropy(self):
        spectral_entropy(RANDOM_TS, SF_TS, method="fft")
        spectral_entropy(RANDOM_TS, SF_TS, method="welch")
        spectral_entropy(RANDOM_TS, SF_TS, method="welch", nperseg=400)
        self.assertEqual(np.round(spectral_entropy(RANDOM_TS, SF_TS, normalize=True), 1), 0.9)
        self.assertEqual(np.round(spectral_entropy(PURE_SINE, 100), 2), 0.0)
        # 2D data
        params = dict(sf=SF_TS, normalize=True, method="welch", nperseg=100)
        assert_equal(
            aal(spectral_entropy, axis=1, arr=data, **params),
            spectral_entropy(data, **params),
        )

    def test_svd_entropy(self):
        svd_entropy(RANDOM_TS, order=3, delay=1, normalize=False)
        svd_entropy(RANDOM_TS, order=3, delay=1, normalize=True)
        svd_entropy(RANDOM_TS, order=2, delay=1, normalize=False)
        svd_entropy(RANDOM_TS, order=3, delay=2, normalize=False)

    def test_sample_entropy(self):
        se = sample_entropy(RANDOM_TS, order=2)
        sample_entropy(RANDOM_TS_LONG, order=2)
        se_eu_3 = sample_entropy(RANDOM_TS, order=3, metric="euclidean")
        # Compare with MNE-features
        # Note that MNE-features uses the sample standard deviation
        # np.std(ddof=1) and not the population standard deviation to define r
        self.assertEqual(np.round(se, 3), 2.192)
        self.assertEqual(np.round(se_eu_3, 3), 2.724)
        sample_entropy(RANDOM_TS, order=3)
        sample_entropy(RANDOM_TS, order=2, metric="euclidean")
        with self.assertRaises(ValueError):
            sample_entropy(RANDOM_TS, order=2, metric="wrong")

    def test_app_entropy(self):
        ae = app_entropy(RANDOM_TS, order=2)
        ae_eu_3 = app_entropy(RANDOM_TS, order=3, metric="euclidean")
        # Compare with MNE-features
        # Note that MNE-features uses the sample standard deviation
        # np.std(ddof=1) and not the population standard deviation to define r
        self.assertEqual(np.round(ae, 3), 2.076)
        self.assertEqual(np.round(ae_eu_3, 3), 0.956)
        app_entropy(RANDOM_TS, order=3)
        with self.assertRaises(ValueError):
            app_entropy(RANDOM_TS, order=2, metric="wrong")

    def test_lziv_complexity(self):
        """Compare to:
        https://www.mathworks.com/matlabcentral/fileexchange/38211-calc_lz_complexity
        """
        s = "010101010101"
        n = len(s)
        assert lziv_complexity(s) == 3
        assert lziv_complexity(s, True) == 3 / (n / np.log2(n))
        assert round(lziv_complexity(s, True), 5) == 0.89624
        assert lziv_complexity([True, False, True, False]) == 3
        assert lziv_complexity(np.array([1, 0, 1, 0, 1, 0])) == 3
        assert lziv_complexity(["1", "0", "1", "0"]) == 3
        assert lziv_complexity(["1010"]) == 3
        assert lziv_complexity("11111") == lziv_complexity("00") == 2
        assert lziv_complexity(np.ones(10000), normalize=True) < 0.01
        assert lziv_complexity([1, 2, 3, 4, 5]) == 5
        assert lziv_complexity([1, 2, 3, 4, 5], normalize=True) == 1.0

        # Test with a random sequence
        random_seq = np.random.randint(0, 2, 1000)
        assert lziv_complexity(random_seq, normalize=True) > 0.8

        # With characters and words
        s = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        assert lziv_complexity(s) == 26
        assert lziv_complexity(s, normalize=True) == 1.0
        assert lziv_complexity(s + s) == 27
        assert lziv_complexity(s + s, normalize=True) < 1.0
        assert lziv_complexity("HELLO WORLD!")
        s = ["A"] * 10000
        assert lziv_complexity(s) == 2
        assert lziv_complexity(s, normalize=True) < 0.01

    def test_num_zerocross(self):
        assert num_zerocross([-1, 0, 1, 2, 3]) == 1
        assert num_zerocross([-1, 1, 2, -1]) == 2
        assert num_zerocross([0, 0, 2, -1, 0, 1, 0, 2]) == 2
        # 2D data
        assert_equal(aal(num_zerocross, axis=0, arr=data), num_zerocross(data, axis=0))
        assert_equal(
            aal(num_zerocross, axis=-1, arr=data, normalize=True),
            num_zerocross(data, axis=-1, normalize=True),
        )

    def test_hjorth_params(self):
        mob, com = hjorth_params(RANDOM_TS)
        mob_sine, com_sine = hjorth_params(PURE_SINE)
        assert mob_sine < mob
        assert com_sine < com
        # 2D data (avoid warning with flat line variance)
        assert_equal(
            aal(hjorth_params, axis=-1, arr=data[:-1, :]).T,
            hjorth_params(data[:-1, :], axis=-1),
        )

    def test_notwritable_dtypes(self):
        entropy_funcs = [
            perm_entropy,
            lambda x: spectral_entropy(x, sf=100),  # sf is required arg
            svd_entropy,
            sample_entropy,
            app_entropy,
            lziv_complexity,
            num_zerocross,
            hjorth_params,
        ]
        # Make sure that the functions can handle non-writable arrays
        for func in entropy_funcs:
            for dtype in TEST_DTYPES:
                x = RANDOM_TS.astype(dtype)
                x.flags.writeable = False
                func(x)

    def test_xlogx_handles_zero(self):
        assert_equal(_xlogx(0), 0)

    def test_xlogx_handles_array(self):
        np.testing.assert_allclose(
            _xlogx(np.array([0, 0.25, 1, 2, 3, 4, -1])),
            np.array([0, -0.5, 0, 2, 4.754887502163468, 8, np.nan]),
        )

    def test_xlogx_handles_2d_array(self):
        np.testing.assert_allclose(
            _xlogx(np.array([[0, 0.25, 1, 2, 3, 4, -1], [-1, 0, 3, 4, -1, 0, 3]])),
            np.array(
                [
                    [0, -0.5, 0, 2, 4.754887502163468, 8, np.nan],
                    [np.nan, 0, 4.754887502163468, 8, np.nan, 0, 4.754887502163468],
                ]
            ),
        )

    def test_xlogx_accepts_other_base(self):
        np.testing.assert_allclose(
            _xlogx(np.array([0, 1, 3, 9, -1]), base=3), np.array([0, 0, 3, 18, np.nan])
        )
