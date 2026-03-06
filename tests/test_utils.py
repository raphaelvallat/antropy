"""Test utility functions."""

import unittest

import numpy as np
from numpy.testing import assert_array_equal

from antropy.utils import _embed


class TestEmbed(unittest.TestCase):
    def test_embed_1d_shape(self):
        x = np.arange(10, dtype=float)
        result = _embed(x, order=3, delay=1)
        # shape: (n_times - (order-1)*delay, order) = (8, 3)
        assert result.shape == (8, 3)

    def test_embed_1d_values(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        result = _embed(x, order=3, delay=1)
        assert_array_equal(result[0], [1.0, 2.0, 3.0])
        assert_array_equal(result[-1], [5.0, 6.0, 7.0])

    def test_embed_1d_delay(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        result = _embed(x, order=3, delay=2)
        # shape: (7 - (3-1)*2, 3) = (3, 3)
        assert result.shape == (3, 3)
        assert_array_equal(result[0], [1.0, 3.0, 5.0])
        assert_array_equal(result[1], [2.0, 4.0, 6.0])

    def test_embed_2d_shape(self):
        rng = np.random.default_rng(42)
        x = rng.random((3, 100))
        result = _embed(x, order=4, delay=2)
        # shape: (n_signals, n_times - (order-1)*delay, order) = (3, 94, 4)
        assert result.shape == (3, 94, 4)

    def test_embed_2d_values(self):
        # Construct a simple 2-signal array for easy manual verification
        x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]])
        result = _embed(x, order=3, delay=1)
        # shape: (2, 3, 3)
        assert result.shape == (2, 3, 3)
        assert_array_equal(result[0, 0], [1.0, 2.0, 3.0])
        assert_array_equal(result[0, -1], [3.0, 4.0, 5.0])
        assert_array_equal(result[1, 0], [10.0, 20.0, 30.0])

    def test_embed_errors_order_times_delay(self):
        x = np.arange(5, dtype=float)
        with self.assertRaises(ValueError):
            _embed(x, order=6, delay=1)  # order * delay > N

    def test_embed_errors_delay_lt_1(self):
        x = np.arange(10, dtype=float)
        with self.assertRaises(ValueError):
            _embed(x, order=3, delay=0)

    def test_embed_errors_order_lt_2(self):
        x = np.arange(10, dtype=float)
        with self.assertRaises(ValueError):
            _embed(x, order=1, delay=1)

    def test_embed_errors_3d_input(self):
        x = np.ones((2, 3, 4))
        with self.assertRaises(AssertionError):
            _embed(x, order=2, delay=1)
