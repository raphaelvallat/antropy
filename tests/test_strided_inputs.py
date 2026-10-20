"""Public feature functions accept NumPy views as well as contiguous arrays."""

import numpy as np
import pytest

import antropy as ant


@pytest.mark.parametrize("name", ["sample_entropy", "higuchi_fd", "detrended_fluctuation"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("readonly", [False, True])
@pytest.mark.parametrize("layout", ["step", "reverse", "column", "contiguous"])
def test_numba_features_accept_array_views(name, dtype, readonly, layout):
    """The same samples yield the same feature regardless of physical layout."""
    source = np.random.default_rng(42).normal(size=(512, 3)).astype(dtype)
    if layout == "step":
        data = source[:, 0][::2]
    elif layout == "reverse":
        data = source[:, 0][::-1]
    elif layout == "column":
        data = source[:, 1]
    else:
        data = source[:, 1].copy()
    if layout != "contiguous":
        assert not data.flags.c_contiguous
    data.flags.writeable = not readonly
    original = data.copy()
    feature = getattr(ant, name)
    reference = feature(np.array(data, dtype=np.float64, copy=True, order="C"))

    result = feature(data)

    assert np.isfinite(reference)
    assert result == reference
    np.testing.assert_array_equal(data, original)
    assert data.flags.writeable == (not readonly)


@pytest.mark.parametrize("metric,size", [("chebyshev", 5000), ("euclidean", 512)])
def test_sample_entropy_strided_kdtree_path(metric, size):
    """The non-Numba routes retain their existing behavior."""
    data = np.random.default_rng(52).normal(size=2 * size)[::2]
    reference = ant.sample_entropy(data.copy(), metric=metric)

    assert ant.sample_entropy(data, metric=metric) == reference
