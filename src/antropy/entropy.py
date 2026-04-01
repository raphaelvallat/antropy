"""Entropy functions"""

import itertools
from math import factorial, log

import numpy as np
from numba import jit, types
from numpy.lib.stride_tricks import as_strided
from scipy.signal import periodogram, welch
from sklearn.neighbors import KDTree

from .utils import _embed, _xlogx

__all__ = [
    "perm_entropy",
    "spectral_entropy",
    "svd_entropy",
    "app_entropy",
    "sample_entropy",
    "lziv_complexity",
    "num_zerocross",
    "hjorth_params",
]


# ---------------------------------------------------------------------------
# Fast-path lookup tables for order=3 and order=4 (any delay).
#
# Instead of argsort, every ordinal pattern is identified by encoding all
# C(order, 2) pairwise comparisons as bits of an integer key, then reading
# the ordinal index from a small pre-computed table.  This avoids the O(m
# log m) sort and replaces it with O(m) bitwise operations.
#
# Note: equal values (ties) are encoded as False by '<', which may assign
# them to a different ordinal pattern than argsort would.  For the typical
# case of continuous data this never occurs in practice.
# ---------------------------------------------------------------------------

# order=3 — key bits: bit2=(a<b), bit1=(a<c), bit0=(b<c)
# The 8-entry _PE3_LOOKUP maps the 3-bit key to the hashmult hash value
# (sum of argsort * [1,3,9]) used by the general path; _PE3_REMAP then
# compresses those 6 sparse hash values to the dense range 0..5.
_PE3_LOOKUP = np.zeros(8, dtype=np.int8)
_PE3_LOOKUP[0b111] = 21  # a<b<c  → argsort [0,1,2]
_PE3_LOOKUP[0b110] = 15  # a<c<b  → argsort [0,2,1]
_PE3_LOOKUP[0b100] = 11  # c<a<b  → argsort [2,0,1]
_PE3_LOOKUP[0b000] = 5  # c<b<a  → argsort [2,1,0]
_PE3_LOOKUP[0b001] = 7  # b<c<a  → argsort [1,2,0]
_PE3_LOOKUP[0b011] = 19  # b<a<c  → argsort [1,0,2]
_PE3_REMAP = np.zeros(22, dtype=np.int64)
for _i, _v in enumerate([5, 7, 11, 15, 19, 21]):
    _PE3_REMAP[_v] = _i
del _i, _v

# order=4 — key bits: bit5=(a<b), bit4=(a<c), bit3=(a<d), bit2=(b<c),
#                     bit1=(b<d), bit0=(c<d)
# The 64-entry table directly maps each valid 6-bit key to ordinal 0..23.
# Invalid keys (impossible comparison patterns) keep the sentinel value -1.
_PE4_LOOKUP = np.full(64, -1, dtype=np.int64)
for _idx, _perm in enumerate(itertools.permutations(range(4))):
    _key = (
        ((_perm[0] < _perm[1]) << 5)
        | ((_perm[0] < _perm[2]) << 4)
        | ((_perm[0] < _perm[3]) << 3)
        | ((_perm[1] < _perm[2]) << 2)
        | ((_perm[1] < _perm[3]) << 1)
        | (_perm[2] < _perm[3])
    )
    _PE4_LOOKUP[_key] = _idx
del _idx, _perm, _key


def _perm_entropy_fast(x, order, delay, normalize):
    """Comparison-based fast path for order=3 and order=4, supporting 1D and 2D input.

    Accepts ``x`` of shape ``(n_times,)`` or ``(n_epochs, n_times)``.
    Returns a scalar for 1D input and an array of shape ``(n_epochs,)`` for 2D.

    Instead of argsort, all pairwise comparisons between delayed columns are
    packed into an integer bit-key and looked up in a pre-computed table,
    giving a ~5-7x speed-up over the argsort-based general path.
    """
    is_1d = x.ndim == 1
    if is_1d:
        x = x[np.newaxis, :]  # treat as a single epoch for unified code below

    n, m = x.shape
    n_embed = m - (order - 1) * delay

    # Apply a positional epsilon jitter — adding i*eps to column i — so that
    # any tied values are broken by column index, exactly matching argsort's
    # behaviour.  This handles integer signals, quantized data, zero-padded
    # epochs, and any other input with exact duplicate values.
    # eps is scaled to the signal magnitude so the jitter never affects the
    # ordering of distinct values.
    eps = np.finfo(np.float64).eps * (float(np.abs(x).max()) + 1)
    cols = [
        x[:, i * delay : i * delay + n_embed].astype(np.float64) + i * eps for i in range(order)
    ]

    if order == 3:
        col0, col1, col2 = cols
        # Encode the 3 pairwise comparisons as bits of an integer key
        bit_key = (
            ((col0 < col1).astype(np.uint8) << 2)
            | ((col0 < col2).astype(np.uint8) << 1)
            | (col1 < col2).astype(np.uint8)
        )
        keys = _PE3_REMAP[_PE3_LOOKUP[bit_key]]  # ordinal indices 0..5
        n_perms = 6

    else:  # order == 4
        col0, col1, col2, col3 = cols
        # Encode the 6 pairwise comparisons as bits of an integer key
        bit_key = (
            ((col0 < col1).astype(np.uint8) << 5)
            | ((col0 < col2).astype(np.uint8) << 4)
            | ((col0 < col3).astype(np.uint8) << 3)
            | ((col1 < col2).astype(np.uint8) << 2)
            | ((col1 < col3).astype(np.uint8) << 1)
            | (col2 < col3).astype(np.uint8)
        )
        keys = _PE4_LOOKUP[bit_key]  # ordinal indices 0..23
        n_perms = 24

    # Count pattern occurrences per epoch using a single bincount call.
    # Each epoch's keys are shifted by epoch_index * n_perms so that all
    # epochs can be counted in one pass, then reshaped to (n, n_perms).
    offsets = (np.arange(n, dtype=np.int64) * n_perms)[:, None]
    counts = np.bincount((keys + offsets).ravel(), minlength=n * n_perms).reshape(n, n_perms)
    p = counts / n_embed
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0, np.log2(p), 0.0)
    result = -(p * log_p).sum(axis=1)
    if normalize:
        result /= np.log2(factorial(order))
        result = np.minimum(np.maximum(result, 0.0), 1.0)

    return float(result[0]) if is_1d else result


def perm_entropy(x, order=3, delay=1, normalize=False):
    """Permutation Entropy.

    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape ``(n_times,)``, or a
        two-dimensional array of shape ``(n_epochs, n_times)``.
        2D input is only supported for ``order=3`` or ``order=4``.
    order : int
        Order of permutation entropy. Default is 3.
    delay : int, list, np.ndarray or range
        Time delay (lag). Default is 1. If multiple values are passed
        (e.g. ``[1, 2, 3]``), AntroPy will calculate the average permutation
        entropy across all these delays.
    normalize : bool
        If True, divide by log2(order!) to normalize the entropy between 0
        and 1. Otherwise, return the permutation entropy in bits.

    Returns
    -------
    pe : float or np.array
        Permutation entropy. Returns a scalar for 1D input, or an array of
        shape ``(n_epochs,)`` for 2D input.

    Notes
    -----
    The permutation entropy is a complexity measure for time-series first
    introduced by Bandt and Pompe in 2002.

    The permutation entropy of a signal :math:`x` is defined as:

    .. math:: H = -\\sum p(\\pi)\\log_2 p(\\pi)

    where the sum runs over all :math:`n!` permutations :math:`\\pi` of order
    :math:`n`. This is the information contained in comparing :math:`n`
    consecutive values of the time series. It is clear that
    :math:`0 ≤ H (n) ≤ \\log_2(n!)` where the lower bound is attained for an
    increasing or decreasing sequence of values, and the upper bound for a
    completely random system where all :math:`n!` possible permutations appear
    with the same probability.

    The embedded matrix :math:`Y` is created by:

    .. math::
        y(i)=[x_i,x_{i+\\text{delay}}, ...,x_{i+(\\text{order}-1) *
        \\text{delay}}]

    .. math:: Y=[y(1),y(2),...,y(N-(\\text{order}-1))*\\text{delay})]^T

    For ``order ∈ {3, 4}``, a fast vectorised path based on lookup tables is
    used instead of ``argsort``, giving a 1.5–5× speed-up for 1D input and
    2–6× for 2D input. Higher orders fall back to a standard ``argsort``
    implementation (1D only).

    References
    ----------
    Bandt, Christoph, and Bernd Pompe. "Permutation entropy: a
    natural complexity measure for time series." Physical review letters
    88.17 (2002): 174102.

    Examples
    --------
    Permutation entropy with order 2:

    >>> import numpy as np
    >>> import antropy as ant
    >>> x = [4, 7, 9, 10, 6, 11, 3]
    >>> # Returns a value in bits, between 0 and log2(factorial(order))
    >>> print(f"{ant.perm_entropy(x, order=2):.4f}")
    0.9183

    Normalized permutation entropy with order 3:

    >>> # Returns a value between 0 and 1.
    >>> print(f"{ant.perm_entropy(x, normalize=True):.4f}")
    0.5888

    Average across multiple delays:

    >>> rng = np.random.default_rng(seed=42)
    >>> x = rng.random(1000)
    >>> print(f"{ant.perm_entropy(x, delay=[1, 2, 3], normalize=True):.4f}")
    0.9996

    Pure sine wave (low entropy):

    >>> x = np.sin(2 * np.pi * 1 * np.arange(3000) / 100)
    >>> print(f"{ant.perm_entropy(x, normalize=True):.4f}")
    0.4441

    Linearly-increasing time-series (minimum entropy):

    >>> x = np.arange(1000)
    >>> print(f"{ant.perm_entropy(x, normalize=True):.4f}")
    0.0000

    2D input — vectorized permutation entropy (only supported for ``order=3`` or ``order=4``):

    >>> rng = np.random.default_rng(seed=42)
    >>> x2d = rng.random((4, 1000))
    >>> pe = ant.perm_entropy(x2d, order=3, normalize=True)
    >>> pe.shape
    (4,)
    >>> print(np.round(pe, 4))
    [0.9997 0.9991 0.9988 0.9988]
    """
    # If multiple delays are passed, return the average across all of them
    if isinstance(delay, (list, np.ndarray, range)):
        return np.mean([perm_entropy(x, order=order, delay=d, normalize=normalize) for d in delay])
    x = np.asarray(x)
    if x.ndim not in (1, 2):
        raise ValueError("x must be 1D or 2D.")
    if order < 2:
        raise ValueError("Order has to be at least 2.")
    if delay < 1:
        raise ValueError("delay must be greater than zero.")
    delay = int(delay)
    n_embed = x.shape[-1] - (order - 1) * delay
    if n_embed <= 0:
        raise ValueError("The signal is too short for the given order and delay.")
    if x.ndim == 2 and order not in (3, 4):
        raise ValueError("2D input is only supported for order=3 and order=4.")

    if order in (3, 4):
        return _perm_entropy_fast(x, order, delay, normalize)

    # General path for order > 4 (1D only).
    # as_strided is used instead of _embed because _embed allocates a full
    # (n_windows, order) copy of the data, while as_strided builds a zero-copy
    # view. svd_entropy and app/sample_entropy also call _embed, but their
    # downstream consumers (np.linalg.svd, sklearn.KDTree) require a
    # contiguous array and would copy anyway, so as_strided offers no benefit
    # there. Here argsort works fine on non-contiguous input, so the copy is
    # avoided entirely.
    # @ replaces the elementwise-multiply + sum with a BLAS matrix-vector product.
    n_windows = n_embed  # n_embed = len(x) - (order-1)*delay
    embedded = as_strided(
        x,
        shape=(n_windows, order),
        strides=(x.strides[0], x.strides[0] * delay),
    )
    hashmult = np.power(order, np.arange(order))
    hashval = embedded.argsort(axis=1, kind="stable") @ hashmult
    _, counts = np.unique(hashval, return_counts=True)
    p = counts / counts.sum()
    pe = -_xlogx(p).sum()
    if normalize:
        pe /= np.log2(factorial(order))
        pe = float(np.minimum(np.maximum(pe, 0.0), 1.0))
    return pe


def spectral_entropy(x, sf, method="fft", nperseg=None, normalize=False, axis=-1):
    """Spectral Entropy.

    Parameters
    ----------
    x : list or np.array
        1D or N-D data.
    sf : float
        Sampling frequency, in Hz.
    method : str
        Spectral estimation method:

        * ``'fft'`` : Fourier Transform (:py:func:`scipy.signal.periodogram`)
        * ``'welch'`` : Welch periodogram (:py:func:`scipy.signal.welch`)
    nperseg : int or None
        Length of each FFT segment for Welch method.
        If None (default), uses scipy's default (256 samples, or the length
        of the signal if shorter).
    normalize : bool
        If True, divide by log2(number of frequency bins) to normalize the
        spectral entropy between 0 and 1. Otherwise, return the spectral
        entropy in bit.
    axis : int
        The axis along which the entropy is calculated. Default is -1 (last).

    Returns
    -------
    se : float
        Spectral Entropy

    Notes
    -----
    Spectral Entropy is defined to be the Shannon entropy of the power
    spectral density (PSD) of the data:

    .. math:: H(x, sf) =  -\\sum_{f=0}^{f_s/2} P(f) \\log_2[P(f)]

    Where :math:`P` is the normalised PSD, and :math:`f_s` is the sampling
    frequency.

    .. note::
        The DC component (:math:`f = 0`) is included in the sum.

    References
    ----------
    - Inouye, T. et al. (1991). Quantification of EEG irregularity by
      use of the entropy of the power spectrum. Electroencephalography
      and clinical neurophysiology, 79(3), 204-210.

    - https://en.wikipedia.org/wiki/Spectral_density

    - https://en.wikipedia.org/wiki/Welch%27s_method

    Examples
    --------
    Spectral entropy of a pure sine using FFT

    >>> import numpy as np
    >>> import antropy as ant
    >>> sf, f, dur = 100, 1, 4
    >>> N = sf * dur  # Total number of discrete samples
    >>> t = np.arange(N) / sf  # Time vector
    >>> x = np.sin(2 * np.pi * f * t)
    >>> print(f"{ant.spectral_entropy(x, sf, method='fft'):.2f}")
    0.00

    Spectral entropy of a random signal using Welch's method

    >>> np.random.seed(42)
    >>> x = np.random.rand(3000)
    >>> print(f"{ant.spectral_entropy(x, sf=100, method='welch'):.4f}")
    6.9800

    Normalized spectral entropy

    >>> print(f"{ant.spectral_entropy(x, sf=100, method='welch', normalize=True):.4f}")
    0.9956

    Normalized spectral entropy of 2D data

    >>> np.random.seed(42)
    >>> x = np.random.normal(size=(4, 3000))
    >>> np.round(ant.spectral_entropy(x, sf=100, normalize=True), 4)
    array([0.9464, 0.9428, 0.9431, 0.9417])

    Fractional Gaussian noise with H = 0.5

    >>> import stochastic.processes.noise as sn
    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.5, rng=rng).sample(10000)
    >>> print(f"{ant.spectral_entropy(x, sf=100, normalize=True):.4f}")
    0.9505

    Fractional Gaussian noise with H = 0.9

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.9, rng=rng).sample(10000)
    >>> print(f"{ant.spectral_entropy(x, sf=100, normalize=True):.4f}")
    0.8477

    Fractional Gaussian noise with H = 0.1

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.1, rng=rng).sample(10000)
    >>> print(f"{ant.spectral_entropy(x, sf=100, normalize=True):.4f}")
    0.9248
    """
    x = np.asarray(x)
    # Compute and normalize power spectrum
    if method == "fft":
        _, psd = periodogram(x, sf, axis=axis)
    elif method == "welch":
        _, psd = welch(x, sf, nperseg=nperseg, axis=axis)
    else:
        raise ValueError("method must be 'fft' or 'welch', got '%s'." % method)
    psd_norm = psd / psd.sum(axis=axis, keepdims=True)
    se = -_xlogx(psd_norm).sum(axis=axis)
    if normalize:
        se /= np.log2(psd_norm.shape[axis])
    return se


def svd_entropy(x, order=3, delay=1, normalize=False):
    """Singular Value Decomposition entropy.

    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times)
    order : int
        Order of SVD entropy (= length of the embedding dimension).
        Default is 3.
    delay : int
        Time delay (lag). Default is 1.
    normalize : bool
        If True, divide by log2(order) to normalize the entropy between 0
        and 1 (the maximum is reached when all ``order`` singular values are
        equal). Otherwise, return the SVD entropy in bit.

    Returns
    -------
    svd_e : float
        SVD Entropy

    Notes
    -----
    SVD entropy is an indicator of the number of eigenvectors that are needed
    for an adequate explanation of the data set. In other words, it measures
    the dimensionality of the data.

    The SVD entropy of a signal :math:`x` is defined as:

    .. math::
        H = -\\sum_{i=1}^{M} \\overline{\\sigma}_i log_2(\\overline{\\sigma}_i)

    where :math:`M` is the number of singular values of the embedded matrix
    :math:`Y` and :math:`\\sigma_1, \\sigma_2, ..., \\sigma_M` are the
    normalized singular values of :math:`Y`.

    The embedded matrix :math:`Y` is created by:

    .. math::
        y(i)=[x_i,x_{i+\\text{delay}}, ...,x_{i+(\\text{order}-1) *
        \\text{delay}}]

    .. math:: Y=[y(1),y(2),...,y(N-(\\text{order}-1))*\\text{delay})]^T

    Examples
    --------
    SVD entropy with order 2

    >>> import numpy as np
    >>> import antropy as ant
    >>> import stochastic.processes.noise as sn
    >>> x = [4, 7, 9, 10, 6, 11, 3]
    >>> # Return a value in bit between 0 and log2(factorial(order))
    >>> print(ant.svd_entropy(x, order=2))
    0.7618909465130066

    Normalized SVD entropy with order 3

    >>> x = [4, 7, 9, 10, 6, 11, 3]
    >>> # Return a value comprised between 0 and 1.
    >>> print(ant.svd_entropy(x, order=3, normalize=True))
    0.6870083043946692

    Fractional Gaussian noise with H = 0.5

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.5, rng=rng).sample(10000)
    >>> print(f"{ant.svd_entropy(x, normalize=True):.4f}")
    1.0000

    Fractional Gaussian noise with H = 0.9

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.9, rng=rng).sample(10000)
    >>> print(f"{ant.svd_entropy(x, normalize=True):.4f}")
    0.9080

    Fractional Gaussian noise with H = 0.1

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.1, rng=rng).sample(10000)
    >>> print(f"{ant.svd_entropy(x, normalize=True):.4f}")
    0.9637

    Random

    >>> rng = np.random.default_rng(seed=42)
    >>> print(f"{ant.svd_entropy(rng.random(1000), normalize=True):.4f}")
    0.8527

    Pure sine wave

    >>> x = np.sin(2 * np.pi * 1 * np.arange(3000) / 100)
    >>> print(f"{ant.svd_entropy(x, normalize=True):.4f}")
    0.1775

    Linearly-increasing time-series

    >>> x = np.arange(1000)
    >>> print(f"{ant.svd_entropy(x, normalize=True):.4f}")
    0.0053
    """
    x = np.array(x)
    mat = _embed(x, order=order, delay=delay)
    W = np.linalg.svd(mat, compute_uv=False)
    # Normalize the singular values
    W /= np.sum(W)
    svd_e = -_xlogx(W).sum()
    if normalize:
        svd_e /= np.log2(order)
    return svd_e


def _app_samp_entropy(x, order, r, metric="chebyshev", approximate=True):
    """Utility function for `app_entropy`` and `sample_entropy`."""
    _all_metrics = KDTree.valid_metrics
    _all_metrics = _all_metrics() if callable(_all_metrics) else _all_metrics
    if metric not in _all_metrics:
        raise ValueError(
            "The given metric (%s) is not valid. The valid "
            "metric names are: %s" % (metric, _all_metrics)
        )
    phi = np.zeros(2)

    # compute phi(order, r)
    _emb_data1 = _embed(x, order, 1)
    if approximate:
        emb_data1 = _emb_data1
    else:
        emb_data1 = _emb_data1[:-1]
    count1 = (
        KDTree(emb_data1, metric=metric)
        .query_radius(emb_data1, r, count_only=True)
        .astype(np.float64)
    )
    # compute phi(order + 1, r)
    emb_data2 = _embed(x, order + 1, 1)
    count2 = (
        KDTree(emb_data2, metric=metric)
        .query_radius(emb_data2, r, count_only=True)
        .astype(np.float64)
    )
    if approximate:
        phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
        phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))
    else:
        phi[0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
        phi[1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))
    return phi


@jit(
    (types.Array(types.float64, 1, "C", readonly=True), types.int32, types.float64),
    nopython=True,
)
def _numba_sampen(sequence, order, r):
    """
    Fast evaluation of the sample entropy using Numba.
    """

    size = sequence.size
    # sequence = sequence.tolist()

    numerator = 0
    denominator = 0

    for offset in range(1, size - order):
        n_numerator = int(abs(sequence[order] - sequence[order + offset]) >= r)
        n_denominator = 0

        for idx in range(order):
            n_numerator += abs(sequence[idx] - sequence[idx + offset]) >= r
            n_denominator += abs(sequence[idx] - sequence[idx + offset]) >= r

        if n_numerator == 0:
            numerator += 1
        if n_denominator == 0:
            denominator += 1

        prev_in_diff = int(abs(sequence[order] - sequence[offset + order]) >= r)
        for idx in range(1, size - offset - order):
            out_diff = int(abs(sequence[idx - 1] - sequence[idx + offset - 1]) >= r)
            in_diff = int(abs(sequence[idx + order] - sequence[idx + offset + order]) >= r)
            n_numerator += in_diff - out_diff
            n_denominator += prev_in_diff - out_diff
            prev_in_diff = in_diff

            if n_numerator == 0:
                numerator += 1
            if n_denominator == 0:
                denominator += 1

    if denominator == 0:
        return np.nan  # undefined: no templates of length m matched within r
    elif numerator == 0:
        return np.inf
    else:
        return -log(numerator / denominator)


def app_entropy(x, order=2, tolerance=None, metric="chebyshev"):
    """Approximate Entropy.

    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times).
    order : int
        Embedding dimension. Default is 2.
    tolerance : float
        Tolerance value for acceptance of the template vector. Default is 0.2
        times the standard deviation of x.
    metric : str
        Name of the distance metric function used with
        :py:class:`sklearn.neighbors.KDTree`. Default is to use the
        `Chebyshev <https://en.wikipedia.org/wiki/Chebyshev_distance>`_
        distance.

    Returns
    -------
    ae : float
        Approximate Entropy.

    Notes
    -----
    Approximate entropy is a technique used to quantify the amount of
    regularity and the unpredictability of fluctuations over time-series data.
    Smaller values indicates that the data is more regular and predictable.

    The default tolerance value (:math:`r`) is set to :math:`0.2 * \\text{std}(x)`.

    Code adapted from the `mne-features <https://mne.tools/mne-features/>`_
    package by Jean-Baptiste Schiratti and Alexandre Gramfort.

    References
    ----------
    Richman, J. S. et al. (2000). Physiological time-series analysis
    using approximate entropy and sample entropy. American Journal of
    Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.

    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html

    Examples
    --------
    Fractional Gaussian noise with H = 0.5

    >>> import numpy as np
    >>> import antropy as ant
    >>> import stochastic.processes.noise as sn
    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.5, rng=rng).sample(10000)
    >>> print(f"{ant.app_entropy(x, order=2):.4f}")
    2.1958

    Same with order = 3 and metric = 'euclidean'

    >>> print(f"{ant.app_entropy(x, order=3, metric='euclidean'):.4f}")
    1.5120

    Fractional Gaussian noise with H = 0.9

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.9, rng=rng).sample(10000)
    >>> print(f"{ant.app_entropy(x):.4f}")
    1.9681

    Fractional Gaussian noise with H = 0.1

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.1, rng=rng).sample(10000)
    >>> print(f"{ant.app_entropy(x):.4f}")
    2.0906

    Random

    >>> rng = np.random.default_rng(seed=42)
    >>> print(f"{ant.app_entropy(rng.random(1000)):.4f}")
    1.8177

    Pure sine wave

    >>> x = np.sin(2 * np.pi * 1 * np.arange(3000) / 100)
    >>> print(f"{ant.app_entropy(x):.4f}")
    0.2009

    Linearly-increasing time-series

    >>> x = np.arange(1000)
    >>> print(f"{ant.app_entropy(x):.4f}")
    -0.0010
    """
    # define r
    if tolerance is None:
        r = 0.2 * np.std(x, ddof=0)
    else:
        if not isinstance(tolerance, (float, int)):
            raise TypeError("tolerance must be a float or int, got %s." % type(tolerance).__name__)
        r = tolerance
    phi = _app_samp_entropy(x, order=order, r=r, metric=metric, approximate=True)
    return np.subtract(phi[0], phi[1])


def sample_entropy(x, order=2, tolerance=None, metric="chebyshev"):
    """Sample Entropy.

    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times).
    order : int
        Embedding dimension. Default is 2.
    tolerance : float
        Tolerance value for acceptance of the template vector. Default is 0.2
        times the standard deviation of x.
    metric : str
        Name of the distance metric function used with
        :py:class:`sklearn.neighbors.KDTree`. Default is to use the
        `Chebyshev <https://en.wikipedia.org/wiki/Chebyshev_distance>`_
        distance.

    Returns
    -------
    se : float
        Sample Entropy.

    Notes
    -----
    Sample entropy is a modification of approximate entropy, used for assessing
    the complexity of physiological time-series signals. It has two advantages
    over approximate entropy: data length independence and a relatively
    trouble-free implementation. Large values indicate high complexity whereas
    smaller values characterize more self-similar and regular signals.

    The sample entropy of a signal :math:`x` is defined as:

    .. math:: H(x, m, r) = -\\log\\frac{C(m + 1, r)}{C(m, r)}

    where :math:`m` is the embedding dimension (= order), :math:`r` is
    the radius of the neighbourhood (default = :math:`0.2 * \\text{std}(x)`),
    :math:`C(m + 1, r)` is the number of embedded vectors of length
    :math:`m + 1` having a
    `Chebyshev distance <https://en.wikipedia.org/wiki/Chebyshev_distance>`_
    less than :math:`r` and :math:`C(m, r)` is the number of embedded
    vectors of length :math:`m` having a Chebyshev distance less than
    :math:`r`.

    Note that if ``metric == 'chebyshev'`` and ``len(x) < 5000`` points,
    then the sample entropy is computed using a fast custom Numba script.
    For other distance metric or longer time-series, the sample entropy is
    computed using a code from the
    `mne-features <https://mne.tools/mne-features/>`_ package by Jean-Baptiste
    Schiratti and Alexandre Gramfort (requires sklearn). Both code paths should
    produce equivalent results.

    When no template of length ``order`` matches within ``tolerance``
    (denominator = 0), the function returns ``np.nan`` rather than an
    arbitrary value, as the entropy is mathematically undefined in that case.

    References
    ----------
    Richman, J. S. et al. (2000). Physiological time-series analysis
    using approximate entropy and sample entropy. American Journal of
    Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.

    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html

    Examples
    --------
    Fractional Gaussian noise with H = 0.5

    >>> import numpy as np
    >>> import antropy as ant
    >>> import stochastic.processes.noise as sn
    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.5, rng=rng).sample(10000)
    >>> print(f"{ant.sample_entropy(x, order=2):.4f}")
    2.1819

    Same with order = 3 and using the Euclidean distance

    >>> print(f"{ant.sample_entropy(x, order=3, metric='euclidean'):.4f}")
    2.6806

    Fractional Gaussian noise with H = 0.9

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.9, rng=rng).sample(10000)
    >>> print(f"{ant.sample_entropy(x):.4f}")
    1.9078

    Fractional Gaussian noise with H = 0.1

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.1, rng=rng).sample(10000)
    >>> print(f"{ant.sample_entropy(x):.4f}")
    2.0555

    Random

    >>> rng = np.random.default_rng(seed=42)
    >>> print(f"{ant.sample_entropy(rng.random(1000)):.4f}")
    2.2017

    Pure sine wave

    >>> x = np.sin(2 * np.pi * 1 * np.arange(3000) / 100)
    >>> print(f"{ant.sample_entropy(x):.4f}")
    0.1633

    Linearly-increasing time-series

    >>> x = np.arange(1000)
    >>> print(f"{ant.sample_entropy(x):.4f}")
    -0.0000
    """
    # define r
    if tolerance is None:
        r = 0.2 * np.std(x, ddof=0)
    else:
        if not isinstance(tolerance, (float, int)):
            raise TypeError("tolerance must be a float or int, got %s." % type(tolerance).__name__)
        r = tolerance
    x = np.asarray(x, dtype=np.float64)
    if metric == "chebyshev" and x.size < 5000:
        return _numba_sampen(x, order=order, r=r)
    else:
        phi = _app_samp_entropy(x, order=order, r=r, metric=metric, approximate=False)
        return -np.log(np.divide(phi[1], phi[0]))


@jit("uint32(uint32[:])", nopython=True)
def _lz_complexity(binary_string):
    """Internal Numba implementation of the Lempel-Ziv (LZ) complexity.
    https://github.com/Naereen/Lempel-Ziv_Complexity/blob/master/src/lziv_complexity.py
    - Updated with strict integer typing instead of strings
    - Slight restructuring based on Yacine Mahdid's notebook:
    https://github.com/BIAPT/Notebooks/blob/master/features/Lempel-Ziv%20Complexity.ipynb
    """
    # Initialize variables
    complexity = 1
    prefix_len = 1
    len_substring = 1
    max_len_substring = 1
    pointer = 0

    # Iterate until the entire string has not been parsed
    while prefix_len + len_substring <= len(binary_string):
        # Given a prefix length, find the largest substring
        if (
            binary_string[pointer + len_substring - 1]
            == binary_string[prefix_len + len_substring - 1]  # noqa: W503
        ):
            len_substring += 1  # increase the length of the substring
        else:
            max_len_substring = max(len_substring, max_len_substring)
            pointer += 1
            # Since all pointers have been scanned, pick largest as the jump
            # size
            if pointer == prefix_len:
                # Increment complexity
                complexity += 1
                # Set prefix length to the max substring size found so far
                # (jump size)
                prefix_len += max_len_substring
                # Reset pointer and max substring size
                pointer = 0
                max_len_substring = 1
            # Reset length of current substring
            len_substring = 1

    # Check if final iteration occurred in the middle of a substring
    if len_substring != 1:
        complexity += 1

    return complexity


def lziv_complexity(sequence, normalize=False):
    """
    Lempel-Ziv (LZ) complexity of a sequence.

    .. versionadded:: 0.1.1

    Parameters
    ----------
    sequence : str or array
        A sequence of characters, e.g. ``'1001111011000010'``,
        ``[0, 1, 0, 1, 1]``, or ``'Hello World!'``.
    normalize : bool
        If ``True``, returns the normalized LZ (see Notes).

    Returns
    -------
    lz : int or float
        LZ complexity, which corresponds to the number of different
        substrings encountered as the stream is viewed from the
        beginning to the end. If ``normalize=False``, the output is an
        integer (counts), otherwise the output is a float.

    Notes
    -----
    LZ complexity is defined as the number of different substrings encountered
    as the sequence is viewed from beginning to the end.

    Although the raw LZ is an important complexity indicator, it is heavily
    influenced by sequence length (longer sequence will result in higher LZ).
    Zhang and colleagues (2009) have therefore proposed the normalized LZ,
    which is defined by

    .. math:: \\text{LZn} = \\frac{\\text{LZ}}{(n / \\log_b{n})}

    where :math:`n` is the length of the sequence and :math:`b` the number of
    unique characters in the sequence.

    .. warning::
        Float and integer arrays are cast to ``uint32`` before processing
        (values are truncated, not discretized into bins). For
        continuous-valued signals, binarize the sequence first, e.g.::

            (x >= np.median(x)).astype(int)

    References
    ----------
    * Lempel, A., & Ziv, J. (1976). On the Complexity of Finite Sequences.
      IEEE Transactions on Information Theory / Professional Technical
      Group on Information Theory, 22(1), 75–81.
      https://doi.org/10.1109/TIT.1976.1055501
    * Zhang, Y., Hao, J., Zhou, C., & Chang, K. (2009). Normalized
      Lempel-Ziv complexity and its application in bio-sequence analysis.
      Journal of Mathematical Chemistry, 46(4), 1203–1212.
      https://doi.org/10.1007/s10910-008-9512-2
    * https://en.wikipedia.org/wiki/Lempel-Ziv_complexity
    * https://github.com/Naereen/Lempel-Ziv_Complexity

    Examples
    --------
    >>> from antropy import lziv_complexity
    >>> # Substrings = 1 / 0 / 01 / 1110 / 1100 / 0010
    >>> s = "1001111011000010"
    >>> lziv_complexity(s)
    6

    Using a list of integer / boolean instead of a string

    >>> # 1 / 0 / 10
    >>> lziv_complexity([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    3

    With normalization

    >>> lziv_complexity(s, normalize=True)
    1.5

    This function also works with characters and words

    >>> s = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    >>> lziv_complexity(s), lziv_complexity(s, normalize=True)
    (26, 1.0)

    >>> s = "HELLO WORLD! HELLO WORLD! HELLO WORLD! HELLO WORLD!"
    >>> lziv_complexity(s), lziv_complexity(s, normalize=True)
    (11, 0.38596001132145313)
    """
    if not isinstance(sequence, (str, list, np.ndarray)):
        raise TypeError("sequence must be a str, list, or np.ndarray.")
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a bool.")
    if isinstance(sequence, (list, np.ndarray)):
        sequence = np.asarray(sequence)
        if sequence.dtype.kind in "bfi":
            # Convert [True, False] or [1., 0.] to [1, 0]
            s = sequence.astype("uint32")
        else:
            # Treat as numpy array of strings
            # Map string characters to utf-8 integer representation
            s = np.fromiter(map(ord, "".join(sequence.astype(str))), dtype="uint32")
            # Can't preallocate length (by specifying count) due to string
            # concatenation
    else:
        s = np.fromiter(map(ord, sequence), dtype="uint32")

    if normalize:
        n = len(s)
        base = sum(np.bincount(s) > 0)  # Number of unique characters
        base = 2 if base < 2 else base
        return _lz_complexity(s) / (n / log(n, base))
    else:
        return _lz_complexity(s)


###############################################################################
# OTHER TIME-DOMAIN METRICS
###############################################################################


def num_zerocross(x, normalize=False, axis=-1):
    """Number of zero-crossings.

    .. versionadded:: 0.1.3

    Parameters
    ----------
    x : list or np.array
        1D or N-D data.
    normalize : bool
        If True, divide by the number of samples to normalize the output
        between 0 and 1. Otherwise, return the absolute number of zero
        crossings.
    axis : int
        The axis along which to perform the computation. Default is -1 (last).

    Returns
    -------
    nzc : int or float
        Number of zero-crossings.

    Notes
    -----
    Zero-crossings are detected via :py:func:`numpy.signbit`. A sample that
    is exactly ``0`` is treated as positive (``signbit(0) == False``), so a
    transition ``..., -1, 0, 1, ...`` counts as **one** crossing (at the
    ``-1 → 0`` boundary), not two.

    Examples
    --------
    Simple examples

    >>> import numpy as np
    >>> import antropy as ant
    >>> int(ant.num_zerocross([-1, 0, 1, 2, 3]))
    1

    >>> int(ant.num_zerocross([0, 0, 2, -1, 0, 1, 0, 2]))
    2

    Number of zero crossings of a pure sine

    >>> import numpy as np
    >>> import antropy as ant
    >>> sf, f, dur = 100, 1, 4
    >>> N = sf * dur  # Total number of discrete samples
    >>> t = np.arange(N) / sf  # Time vector
    >>> x = np.sin(2 * np.pi * f * t)
    >>> int(ant.num_zerocross(x))
    7

    Random 2D data

    >>> np.random.seed(42)
    >>> x = np.random.normal(size=(4, 3000))
    >>> ant.num_zerocross(x)
    array([1499, 1528, 1547, 1457])

    Same but normalized by the number of samples

    >>> np.round(ant.num_zerocross(x, normalize=True), 4)
    array([0.4997, 0.5093, 0.5157, 0.4857])

    Fractional Gaussian noise with H = 0.5

    >>> import stochastic.processes.noise as sn
    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.5, rng=rng).sample(10000)
    >>> print(f"{ant.num_zerocross(x, normalize=True):.4f}")
    0.4973

    Fractional Gaussian noise with H = 0.9

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.9, rng=rng).sample(10000)
    >>> print(f"{ant.num_zerocross(x, normalize=True):.4f}")
    0.2615

    Fractional Gaussian noise with H = 0.1

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.1, rng=rng).sample(10000)
    >>> print(f"{ant.num_zerocross(x, normalize=True):.4f}")
    0.6451
    """
    x = np.asarray(x)
    # https://stackoverflow.com/a/29674950/10581531
    nzc = np.diff(np.signbit(x), axis=axis).sum(axis=axis)
    if normalize:
        nzc = nzc / x.shape[axis]
    return nzc


def hjorth_params(x, sf=None, axis=-1):
    """Calculate Hjorth mobility and complexity on given axis.

    .. versionadded:: 0.1.3

    Parameters
    ----------
    x : list or np.array
        1D or N-D data.
    sf : float or None
        Sampling frequency in Hz. If provided, mobility is returned in Hz
        instead of samples⁻¹. Default is None (mobility in samples⁻¹).
    axis : int
        The axis along which to perform the computation. Default is -1 (last).

    Returns
    -------
    mobility, complexity : float
        Mobility and complexity parameters. Mobility is in samples⁻¹ when
        ``sf`` is None, or in Hz when ``sf`` is provided. Complexity is
        dimensionless in both cases.

    Notes
    -----
    Hjorth Parameters are indicators of statistical properties used in signal
    processing in the time domain introduced by Bo Hjorth in 1970. The
    parameters are activity, mobility, and complexity. AntroPy only returns the
    mobility and complexity parameters, since activity is simply the variance
    of :math:`x`, which can be computed easily with :py:func:`numpy.var`.

    The **mobility** parameter represents the mean frequency or the proportion
    of standard deviation of the power spectrum. This is defined as the square
    root of variance of the first derivative of :math:`x` divided by the
    variance of :math:`x`.

    The **complexity** gives an estimate of the bandwidth of the signal, which
    indicates the similarity of the shape of the signal to a pure sine wave
    (where the value converges to 1). Complexity is defined as the ratio of
    the mobility of the first derivative of :math:`x` to the mobility of
    :math:`x`.

    .. note::
        Without a sampling frequency, mobility is expressed in units of
        **samples⁻¹**. Pass ``sf`` to convert to Hz (multiplies mobility by
        ``sf``). Complexity is unaffected because it is a ratio of two
        mobilities.

    References
    ----------
    - https://en.wikipedia.org/wiki/Hjorth_parameters
    - https://doi.org/10.1016%2F0013-4694%2870%2990143-4

    Examples
    --------
    Hjorth parameters of a pure sine (mobility in samples⁻¹)

    >>> import numpy as np
    >>> import antropy as ant
    >>> sf, f, dur = 100, 1, 4
    >>> N = sf * dur  # Total number of discrete samples
    >>> t = np.arange(N) / sf  # Time vector
    >>> x = np.sin(2 * np.pi * f * t)
    >>> np.round(ant.hjorth_params(x), 4)
    array([0.0627, 1.005 ])

    Same signal with sf provided: mobility is now in Hz

    >>> np.round(ant.hjorth_params(x, sf=sf), 4)
    array([6.2743, 1.005 ])

    Random 2D data

    >>> np.random.seed(42)
    >>> x = np.random.normal(size=(4, 3000))
    >>> mob, com = ant.hjorth_params(x)
    >>> print(mob)
    [1.42145064 1.4339572  1.42186993 1.40587512]

    >>> print(com)
    [1.21877527 1.21092261 1.217278   1.22623163]

    Fractional Gaussian noise with H = 0.5

    >>> import stochastic.processes.noise as sn
    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.5, rng=rng).sample(10000)
    >>> np.round(ant.hjorth_params(x), 4)
    array([1.4073, 1.2283])

    Fractional Gaussian noise with H = 0.9

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.9, rng=rng).sample(10000)
    >>> np.round(ant.hjorth_params(x), 4)
    array([0.8395, 1.9143])

    Fractional Gaussian noise with H = 0.1

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.1, rng=rng).sample(10000)
    >>> np.round(ant.hjorth_params(x), 4)
    array([1.6917, 1.0717])
    """
    if sf is not None and not isinstance(sf, (int, float)):
        raise TypeError("sf must be a numeric value (int or float), got %s." % type(sf).__name__)
    x = np.asarray(x)
    # Calculate derivatives
    dx = np.diff(x, axis=axis)
    ddx = np.diff(dx, axis=axis)
    # Calculate variance
    x_var = np.var(x, axis=axis)  # = activity
    dx_var = np.var(dx, axis=axis)
    ddx_var = np.var(ddx, axis=axis)
    # Mobility and complexity
    mob = np.sqrt(dx_var / x_var)
    com = np.sqrt(ddx_var / dx_var) / mob
    if sf is not None:
        mob = mob * sf
    return mob, com
