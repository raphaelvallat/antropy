"""Test fractal dimension functions."""
import unittest
import numpy as np
from numpy.testing import assert_equal
from numpy import apply_along_axis as aal
from antropy import petrosian_fd, katz_fd, higuchi_fd, detrended_fluctuation

np.random.seed(1234567)
RANDOM_TS = np.random.rand(3000)
NORMAL_TS = np.random.normal(size=3000)
RANDOM_TS_LONG = np.random.rand(6000)
SF_TS = 100
PURE_SINE = np.sin(2 * np.pi * 1 * np.arange(3000) / 100)
ARANGE = np.arange(3000)
PPG_SIGNAL = np.array([-4.18272436e-07, 4.29464204e-07, -5.14206643e-06, 1.66113061e-05, 2.80120692e-05, 8.83404336e-05, 1.98141181e-04, -2.88309770e-05,
                       1.97888389e-04, -1.23435636e-03, -7.56440358e-04, -2.97200186e-03, -1.34446557e-03, 8.17779135e-04, 2.52697129e-03, 1.52477197e-02,
                       6.17225675e-03, 2.33431348e-02, -1.14744448e-02, -9.02422108e-03, -4.83006717e-02, -6.78126099e-02, -2.59228720e-02, -5.89649339e-02,
                       1.29395071e-01])

# Concatenate 2D data
data = np.vstack((RANDOM_TS, NORMAL_TS, PURE_SINE, ARANGE))


class TestEntropy(unittest.TestCase):

    def test_petrosian_fd(self):
        pfd = petrosian_fd(RANDOM_TS)
        petrosian_fd(list(RANDOM_TS))
        self.assertEqual(np.round(pfd, 3), 1.030)
        # 2D data
        assert_equal(aal(petrosian_fd, axis=1, arr=data), petrosian_fd(data))
        assert_equal(aal(petrosian_fd, axis=0, arr=data), petrosian_fd(data, axis=0))

    def test_katz_fd(self):
        x_k = [0., 0., 2., -2., 0., -1., -1., 0.]
        self.assertEqual(np.round(katz_fd(x_k), 3), 5.783)
        # 2D data
        assert_equal(aal(katz_fd, axis=1, arr=data), katz_fd(data))
        assert_equal(aal(katz_fd, axis=0, arr=data), katz_fd(data, axis=0))

    def test_higuchi_fd(self):
        """Test for function `higuchi_fd`.
        Results have been tested against the MNE-features and pyrem packages.
        """
        # Compare with MNE-features
        self.assertEqual(np.round(higuchi_fd(RANDOM_TS), 8), 1.9914198)
        # Check if readonly arrays can be processed as well
        x = RANDOM_TS.copy().astype(np.float64)
        x.flags.writeable = False
        self.assertEqual(np.round(higuchi_fd(x), 8), 1.9914198)
        higuchi_fd(list(RANDOM_TS), kmax=20)

    def test_detrended_fluctuation(self):
        """Test for function `detrended_fluctuation`.
        Results have been tested against the NOLDS Python package.

        Note: updated in May 2020 following a conversation on GitHub,
        https://github.com/neuropsychology/NeuroKit/issues/206
        """
        self.assertEqual(np.round(detrended_fluctuation(RANDOM_TS), 4), 0.4976)
        self.assertEqual(np.round(detrended_fluctuation(PURE_SINE), 4), 1.5848)
        self.assertEqual(np.round(detrended_fluctuation(PPG_SIGNAL), 4), 0.0)
