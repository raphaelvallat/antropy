"""Entropy functions"""
import numpy as np
from numba import jit, types
from math import factorial, log
from sklearn.neighbors import KDTree
from scipy.signal import periodogram, welch

from .utils import _embed, _xlogx

all = [
    "perm_entropy",
    "spectral_entropy",
    "svd_entropy",
    "app_entropy",
    "sample_entropy",
    "lziv_complexity",
    "num_zerocross",
    "hjorth_params",
]


def perm_entropy(x, order=3, delay=1, normalize=False):
    """Permutation Entropy.

    Parameters
    ----------
    x : list or np.array
        One-dimensional time series of shape (n_times)
    order : int
        Order of permutation entropy. Default is 3.
    delay : int, list, np.ndarray or range
        Time delay (lag). Default is 1. If multiple values are passed
        (e.g. [1, 2, 3]), AntroPy will calculate the average permutation
        entropy across all these delays.
    normalize : bool
        If True, divide by log2(order!) to normalize the entropy between 0
        and 1. Otherwise, return the permutation entropy in bit.

    Returns
    -------
    pe : float
        Permutation Entropy.

    Notes
    -----
    The permutation entropy is a complexity measure for time-series first
    introduced by Bandt and Pompe in 2002.

    The permutation entropy of a signal :math:`x` is defined as:

    .. math:: H = -\\sum p(\\pi)\\log_2(\\pi)

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

    References
    ----------
    Bandt, Christoph, and Bernd Pompe. "Permutation entropy: a
    natural complexity measure for time series." Physical review letters
    88.17 (2002): 174102.

    Examples
    --------
    Permutation entropy with order 2

    >>> import numpy as np
    >>> import antropy as ant
    >>> import stochastic.processes.noise as sn
    >>> x = [4, 7, 9, 10, 6, 11, 3]
    >>> # Return a value in bit between 0 and log2(factorial(order))
    >>> print(f"{ant.perm_entropy(x, order=2):.4f}")
    0.9183

    Normalized permutation entropy with order 3

    >>> # Return a value comprised between 0 and 1.
    >>> print(f"{ant.perm_entropy(x, normalize=True):.4f}")
    0.5888

    Fractional Gaussian noise with H = 0.5, averaged across multiple delays
    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.5, rng=rng).sample(10000)
    >>> print(f"{ant.perm_entropy(x, delay=[1, 2, 3], normalize=True):.4f}")
    0.9999

    Fractional Gaussian noise with H = 0.1, averaged across multiple delays

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.1, rng=rng).sample(10000)
    >>> print(f"{ant.perm_entropy(x, delay=[1, 2, 3], normalize=True):.4f}")
    0.9986

    Random

    >>> rng = np.random.default_rng(seed=42)
    >>> print(f"{ant.perm_entropy(rng.random(1000), normalize=True):.4f}")
    0.9997

    Pure sine wave

    >>> x = np.sin(2 * np.pi * 1 * np.arange(3000) / 100)
    >>> print(f"{ant.perm_entropy(x, normalize=True):.4f}")
    0.4463

    Linearly-increasing time-series

    >>> x = np.arange(1000)
    >>> print(f"{ant.perm_entropy(x, normalize=True):.4f}")
    -0.0000
    """
    # If multiple delay are passed, return the average across all d
    if isinstance(delay, (list, np.ndarray, range)):
        return np.mean([perm_entropy(x, order=order, delay=d, normalize=normalize) for d in delay])
    x = np.array(x)
    ran_order = range(order)
    hashmult = np.power(order, ran_order)
    assert delay > 0, "delay must be greater than zero."
    # Embed x and sort the order of permutations
    sorted_idx = _embed(x, order=order, delay=delay).argsort(kind="quicksort")
    # Associate unique integer to each permutations
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    # Return the counts
    _, c = np.unique(hashval, return_counts=True)
    # Use np.true_divide for Python 2 compatibility
    p = np.true_divide(c, c.sum())
    pe = -_xlogx(p).sum()
    if normalize:
        pe /= np.log2(factorial(order))
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
        If None (default), uses scipy default of 256 samples.
    normalize : bool
        If True, divide by log2(psd.size) to normalize the spectral entropy
        between 0 and 1. Otherwise, return the spectral entropy in bit.
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
    >>> N = sf * dur # Total number of discrete samples
    >>> t = np.arange(N) / sf # Time vector
    >>> x = np.sin(2 * np.pi * f * t)
    >>> np.round(ant.spectral_entropy(x, sf, method='fft'), 2)
    0.0

    Spectral entropy of a random signal using Welch's method

    >>> np.random.seed(42)
    >>> x = np.random.rand(3000)
    >>> ant.spectral_entropy(x, sf=100, method='welch')
    6.98004566237139

    Normalized spectral entropy

    >>> ant.spectral_entropy(x, sf=100, method='welch', normalize=True)
    0.9955526198316073

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
        If True, divide by log2(order!) to normalize the entropy between 0
        and 1. Otherwise, return the permutation entropy in bit.

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
    W /= sum(W)
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


@jit((types.Array(types.float64, 1, "C", readonly=True), types.int32, types.float64), nopython=True)
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
        return 0  # use 0/0 == 0
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
        assert isinstance(tolerance, (float, int))
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
    inferior to :math:`r` and :math:`C(m, r)` is the number of embedded
    vectors of length :math:`m` having a Chebyshev distance inferior to
    :math:`r`.

    Note that if ``metric == 'chebyshev'`` and ``len(x) < 5000`` points,
    then the sample entropy is computed using a fast custom Numba script.
    For other distance metric or longer time-series, the sample entropy is
    computed using a code from the
    `mne-features <https://mne.tools/mne-features/>`_ package by Jean-Baptiste
    Schiratti and Alexandre Gramfort (requires sklearn).

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
        assert isinstance(tolerance, (float, int))
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
    Lempel-Ziv (LZ) complexity of (binary) sequence.

    .. versionadded:: 0.1.1

    Parameters
    ----------
    sequence : str or array
        A sequence of character, e.g. ``'1001111011000010'``,
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
    as the sequence is viewed from begining to the end.

    Although the raw LZ is an important complexity indicator, it is heavily
    influenced by sequence length (longer sequence will result in higher LZ).
    Zhang and colleagues (2009) have therefore proposed the normalized LZ,
    which is defined by

    .. math:: \\text{LZn} = \\frac{\\text{LZ}}{(n / \\log_b{n})}

    where :math:`n` is the length of the sequence and :math:`b` the number of
    unique characters in the sequence.

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
    >>> s = '1001111011000010'
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

    >>> s = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    >>> lziv_complexity(s), lziv_complexity(s, normalize=True)
    (26, 1.0)

    >>> s = 'HELLO WORLD! HELLO WORLD! HELLO WORLD! HELLO WORLD!'
    >>> lziv_complexity(s), lziv_complexity(s, normalize=True)
    (11, 0.38596001132145313)
    """
    assert isinstance(sequence, (str, list, np.ndarray))
    assert isinstance(normalize, bool)
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
        # 1) Timmermann et al. 2019
        # The sequence is randomly shuffled, and the normalized LZ
        # is calculated as the ratio of the LZ of the original sequence
        # divided by the LZ of the randomly shuffled LZ. However, the final
        # output is dependent on the random seed.
        # sl_shuffled = list(s)
        # rng = np.random.RandomState(None)
        # rng.shuffle(sl_shuffled)
        # s_shuffled = ''.join(sl_shuffled)
        # return _lz_complexity(s) / _lz_complexity(s_shuffled)
        # 2) Zhang et al. 2009
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

    .. versionadded: 0.1.3

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

    Examples
    --------
    Simple examples

    >>> import numpy as np
    >>> import antropy as ant
    >>> ant.num_zerocross([-1, 0, 1, 2, 3])
    1

    >>> ant.num_zerocross([0, 0, 2, -1, 0, 1, 0, 2])
    2

    Number of zero crossings of a pure sine

    >>> import numpy as np
    >>> import antropy as ant
    >>> sf, f, dur = 100, 1, 4
    >>> N = sf * dur # Total number of discrete samples
    >>> t = np.arange(N) / sf # Time vector
    >>> x = np.sin(2 * np.pi * f * t)
    >>> ant.num_zerocross(x)
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


def hjorth_params(x, axis=-1):
    """Calculate Hjorth mobility and complexity on given axis.

    .. versionadded: 0.1.3

    Parameters
    ----------
    x : list or np.array
        1D or N-D data.
    axis : int
        The axis along which to perform the computation. Default is -1 (last).

    Returns
    -------
    mobility, complexity : float
        Mobility and complexity parameters.

    Notes
    -----
    Hjorth Parameters are indicators of statistical properties used in signal
    processing in the time domain introduced by Bo Hjorth in 1970. The
    parameters are activity, mobility, and complexity. EntroPy only returns the
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

    References
    ----------
    - https://en.wikipedia.org/wiki/Hjorth_parameters
    - https://doi.org/10.1016%2F0013-4694%2870%2990143-4

    Examples
    --------
    Hjorth parameters of a pure sine

    >>> import numpy as np
    >>> import antropy as ant
    >>> sf, f, dur = 100, 1, 4
    >>> N = sf * dur # Total number of discrete samples
    >>> t = np.arange(N) / sf # Time vector
    >>> x = np.sin(2 * np.pi * f * t)
    >>> np.round(ant.hjorth_params(x), 4)
    array([0.0627, 1.005 ])

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
    return mob, com
