import numpy as np
import unittest

from entropy import detrended_fluctuation

np.random.seed(1234567)
RANDOM_TS = np.random.rand(3000)
SF_TS = 100
PURE_SINE = np.sin(2 * np.pi * 1 * np.arange(3000) / 100)


class TestEntropy(unittest.TestCase):

    def test_detrended_fluctuation(self):
        """Test for function `detrended_fluctuation`.
        Results have been tested against the NOLDS Python package.
        """
        self.assertEqual(np.round(detrended_fluctuation(RANDOM_TS), 4), 0.5082)
        self.assertEqual(np.round(detrended_fluctuation(PURE_SINE), 4), 1.6158)
