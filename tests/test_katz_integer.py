"""Katz distances must be computed before narrowing integer differences."""

import math

import numpy as np
import pytest

from antropy import katz_fd

CASES = [np.array([1, 5, 2, 8, 3], dtype=dtype) for dtype in (np.uint8, np.uint16, np.uint32)] + [
    np.array([2**64 - 8 + i for i in (1, 5, 2, 7, 6)], dtype=np.uint64),
    np.array([0, 2**64 - 1, 2**63, 0, 2**63], dtype=np.uint64),
    np.array([-128, 127, 0, -127, 126], dtype=np.int8),
    np.array([2**63 - 8 + i for i in (1, 5, 2, 7, 6)], dtype=np.int64),
    np.array([-(2**63), 2**63 - 1, 0, -(2**63) + 1, 2**63 - 2], dtype=np.int64),
    np.array([True, False, False, False, True]),
]


def integer_reference(sequence):
    """Use Python integers for distances and only then evaluate the existing formula."""
    values = [int(value) for value in sequence]
    length = sum(abs(b - a) for a, b in zip(values, values[1:]))
    average = float(length) / (len(values) - 1)
    diameter = max(abs(value - values[0]) for value in values)
    return math.log10(len(values) - 1) / math.log10(float(diameter) / average)


@pytest.mark.parametrize("sequence", CASES)
@pytest.mark.parametrize("layout", ["vector", "rows", "columns"])
def test_katz_integer_distances(sequence, layout):
    """Support ADC values, signed extrema and nearby large 64-bit integers."""
    rows = np.stack([sequence, sequence[::-1]])
    expected = np.array([integer_reference(row) for row in rows])
    if layout == "vector":
        data, axis, expected = sequence, -1, expected[0]
    elif layout == "rows":
        data, axis = rows, -1
    else:
        data, axis = rows.T, 0
    original = data.copy()
    data.flags.writeable = False

    result = katz_fd(data, axis=axis)

    np.testing.assert_allclose(result, expected, rtol=1e-12)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("axis", [-1, 0])
def test_katz_floating_path_unchanged(dtype, axis):
    """Fractional input follows the existing floating-point calculation."""
    data = np.array([[0.25, 1.5, -2.0, 3.25, -1.0]], dtype=dtype)
    if axis == 0:
        data = data.T
    distances = np.abs(np.diff(data, axis=axis))
    average = np.mean(distances, axis=axis)
    diameter = np.max(np.abs(data - data.take([0], axis=axis)), axis=axis)
    expected = np.log10(np.sum(distances, axis=axis) / average) / np.log10(diameter / average)

    np.testing.assert_array_equal(katz_fd(data, axis=axis), expected)
