import numpy as np
import unittest

from entropy import petrosian_fd, katz_fd

np.random.seed(1234567)
RANDOM_TS = np.random.rand(3000)
SF_TS = 100
PURE_SINE = np.sin(2 * np.pi * 1 * np.arange(3000) / 100)


class TestEntropy(unittest.TestCase):

    def test_petrosian_fd(self):
        pfd = petrosian_fd(RANDOM_TS)
        petrosian_fd(list(RANDOM_TS))
        self.assertEqual(np.round(pfd, 3), 1.030)

    def test_katz_fd(self):
        data = [0., 0., 2., -2., 0., -1., -1., 0.]
        self.assertAlmostEqual(np.round(katz_fd(data), 3), 5.783)
