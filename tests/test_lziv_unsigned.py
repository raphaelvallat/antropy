"""Numeric symbols retain their boundaries for unsigned integer inputs."""

import numpy as np
import pytest

from antropy import lziv_complexity


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("offset", [0, 10])
@pytest.mark.parametrize(
    ("symbols", "complexity", "normalized"),
    [("010", 3, np.log2(3)), ("1001111011000010", 6, 1.5)],
)
def test_unsigned_numeric_symbols(dtype, normalize, offset, symbols, complexity, normalized):
    """Relabeling a binary alphabet must not turn each symbol into decimal characters."""
    sequence = np.array([int(symbol) + offset for symbol in symbols], dtype=dtype)
    original = sequence.copy()

    result = lziv_complexity(sequence, normalize=normalize)

    assert result == pytest.approx(normalized if normalize else complexity)
    np.testing.assert_array_equal(sequence, original)
