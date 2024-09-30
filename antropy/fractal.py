"""Fractal functions"""

import numpy as np
from numba import jit, types
from math import log, floor

from .entropy import num_zerocross
from .utils import _linear_regression, _log_n

all = ["petrosian_fd", "katz_fd", "higuchi_fd", "detrended_fluctuation"]


def petrosian_fd(x, axis=-1):
    """Petrosian fractal dimension.

    Parameters
    ----------
    x : list or np.array
        1D or N-D data.
    axis : int
        The axis along which the FD is calculated. Default is -1 (last).

    Returns
    -------
    pfd : float
        Petrosian fractal dimension.

    Notes
    -----
    The Petrosian fractal dimension of a time-series :math:`x` is defined by:

    .. math:: P = \\frac{\\log_{10}(N)}{\\log_{10}(N) +
              \\log_{10}(\\frac{N}{N+0.4N_{\\delta}})}

    where :math:`N` is the length of the time series, and
    :math:`N_{\\delta}` is the number of sign changes in the signal derivative.

    Original code from the `pyrem <https://github.com/gilestrolab/pyrem>`_
    package by Quentin Geissmann.

    References
    ----------
    * A. Petrosian, Kolmogorov complexity of finite sequences and
      recognition of different preictal EEG patterns, in , Proceedings of the
      Eighth IEEE Symposium on Computer-Based Medical Systems, 1995,
      pp. 212-217.

    * Goh, Cindy, et al. "Comparison of fractal dimension algorithms for
      the computation of EEG biomarkers for dementia." 2nd International
      Conference on Computational Intelligence in Medicine and Healthcare
      (CIMED2005). 2005.

    Examples
    --------
    >>> import numpy as np
    >>> import antropy as ant
    >>> import stochastic.processes.noise as sn
    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.5, rng=rng).sample(10000)
    >>> print(f"{ant.petrosian_fd(x):.4f}")
    1.0264

    Fractional Gaussian noise with H = 0.9

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.9, rng=rng).sample(10000)
    >>> print(f"{ant.petrosian_fd(x):.4f}")
    1.0235

    Fractional Gaussian noise with H = 0.1

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.1, rng=rng).sample(10000)
    >>> print(f"{ant.petrosian_fd(x):.4f}")
    1.0283

    Random

    >>> rng = np.random.default_rng(seed=42)
    >>> print(f"{ant.petrosian_fd(rng.random(1000)):.4f}")
    1.0350

    Pure sine wave

    >>> x = np.sin(2 * np.pi * 1 * np.arange(3000) / 100)
    >>> print(f"{ant.petrosian_fd(x):.4f}")
    1.0010

    Linearly-increasing time-series (should be 1)

    >>> x = np.arange(1000)
    >>> print(f"{ant.petrosian_fd(x):.4f}")
    1.0000
    """
    x = np.asarray(x)
    N = x.shape[axis]
    # Number of sign changes in the first derivative of the signal
    nzc_deriv = num_zerocross(np.diff(x, axis=axis), axis=axis)
    pfd = np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * nzc_deriv)))
    return pfd


def katz_fd(x, axis=-1):
    """Katz Fractal Dimension.

    Parameters
    ----------
    x : list or np.array
        1D or N-D data.
    axis : int
        The axis along which the FD is calculated. Default is -1 (last).

    Returns
    -------
    kfd : float
        Katz fractal dimension.

    Notes
    -----
    Katz’s method calculates the fractal dimension of a sample as follows:
    the sum and average of the Euclidean distances between the successive
    points of the sample (:math:`L` and :math:`a` , resp.) are calculated as
    well as the maximum distance between the first point and any other point
    of the sample (:math:`d`). The fractal dimension of the sample (:math:`D`)
    then becomes:

    .. math::
        D = \\frac{\\log_{10}(L/a)}{\\log_{10}(d/a)} =
        \\frac{\\log_{10}(n)}{\\log_{10}(d/L)+\\log_{10}(n)}

    where :math:`n` is :math:`L` divided by :math:`a`.

    Original code from the `mne-features <https://mne.tools/mne-features/>`_
    package by Jean-Baptiste Schiratti and Alexandre Gramfort.

    References
    ----------
    * https://ieeexplore.ieee.org/abstract/document/904882

    * https://hal.inria.fr/inria-00442374/

    * https://www.hindawi.com/journals/ddns/2011/724697/

    Examples
    --------
    >>> import numpy as np
    >>> import antropy as ant
    >>> import stochastic.processes.noise as sn
    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.5, rng=rng).sample(10000)
    >>> print(f"{ant.katz_fd(x):.4f}")
    6.4713

    Fractional Gaussian noise with H = 0.9

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.9, rng=rng).sample(10000)
    >>> print(f"{ant.katz_fd(x):.4f}")
    4.5720

    Fractional Gaussian noise with H = 0.1

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.1, rng=rng).sample(10000)
    >>> print(f"{ant.katz_fd(x):.4f}")
    7.6540

    Random

    >>> rng = np.random.default_rng(seed=42)
    >>> print(f"{ant.katz_fd(rng.random(1000)):.4f}")
    8.1531

    Pure sine wave

    >>> x = np.sin(2 * np.pi * 1 * np.arange(3000) / 100)
    >>> print(f"{ant.katz_fd(x):.4f}")
    2.4871

    Linearly-increasing time-series (should be 1)

    >>> x = np.arange(1000)
    >>> print(f"{ant.katz_fd(x):.4f}")
    1.0000
    """
    x = np.asarray(x)

    # euclidian distance calculation
    euclidean_distance = np.sqrt(1 + np.square(np.diff(x, axis=axis)))

    # total and average path lengths
    total_path_length = euclidean_distance.sum(axis=axis)
    average_path_length = euclidean_distance.mean(axis=axis)

    # max distance from first to all
    horizontal_diffs = np.arange(1, x.shape[axis])
    vertical_diffs = np.take(x, indices=np.arange(1, x.shape[axis]), axis=axis) - np.take(
        x, indices=[0], axis=axis
    )

    if axis == 1:  # reshape if needed
        horizontal_diffs = horizontal_diffs.reshape(1, -1)
    elif axis == 0:
        horizontal_diffs = horizontal_diffs.reshape(-1, 1)

    # Euclidean distance and max distance
    distances = np.sqrt(np.square(horizontal_diffs) + np.square(vertical_diffs))
    max_distance = np.max(distances, axis=axis)

    # Katz Fractal Dimension Calculation
    full_distance = np.log10(total_path_length / average_path_length)
    kfd = np.squeeze(full_distance / (full_distance + np.log10(max_distance / total_path_length)))

    # ensure scalar output
    if not kfd.ndim:
        kfd = kfd.item()

    return kfd


@jit((types.Array(types.float64, 1, "C", readonly=True), types.int32), nopython=True)
def _higuchi_fd(x, kmax):
    """Utility function for `higuchi_fd`."""
    n_times = x.size
    lk = np.empty(kmax)
    x_reg = np.empty(kmax)
    y_reg = np.empty(kmax)
    for k in range(1, kmax + 1):
        lm = np.empty((k,))
        for m in range(k):
            ll = 0
            n_max = floor((n_times - m - 1) / k)
            n_max = int(n_max)
            for j in range(1, n_max):
                ll += abs(x[m + j * k] - x[m + (j - 1) * k])
            ll /= k
            ll *= (n_times - 1) / (k * n_max)
            lm[m] = ll
        # Mean of lm
        m_lm = 0
        for m in range(k):
            m_lm += lm[m]
        m_lm /= k
        lk[k - 1] = m_lm
        x_reg[k - 1] = log(1.0 / k)
        y_reg[k - 1] = log(m_lm)
    higuchi, _ = _linear_regression(x_reg, y_reg)
    return higuchi


def higuchi_fd(x, kmax=10):
    """Higuchi Fractal Dimension.

    Parameters
    ----------
    x : list or np.array
        One dimensional time series.
    kmax : int
        Maximum delay/offset (in number of samples).

    Returns
    -------
    hfd : float
        Higuchi fractal dimension.

    Notes
    -----
    Original code from the `mne-features <https://mne.tools/mne-features/>`_
    package by Jean-Baptiste Schiratti and Alexandre Gramfort.

    This function uses Numba to speed up the computation.

    References
    ----------
    Higuchi, Tomoyuki. "Approach to an irregular time series on the
    basis of the fractal theory." Physica D: Nonlinear Phenomena 31.2
    (1988): 277-283.

    Examples
    --------
    >>> import numpy as np
    >>> import antropy as ant
    >>> import stochastic.processes.noise as sn
    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.5, rng=rng).sample(10000)
    >>> print(f"{ant.higuchi_fd(x):.4f}")
    1.9983

    Fractional Gaussian noise with H = 0.9

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.9, rng=rng).sample(10000)
    >>> print(f"{ant.higuchi_fd(x):.4f}")
    1.8517

    Fractional Gaussian noise with H = 0.1

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.1, rng=rng).sample(10000)
    >>> print(f"{ant.higuchi_fd(x):.4f}")
    2.0581

    Random

    >>> rng = np.random.default_rng(seed=42)
    >>> print(f"{ant.higuchi_fd(rng.random(1000)):.4f}")
    2.0013

    Pure sine wave

    >>> x = np.sin(2 * np.pi * 1 * np.arange(3000) / 100)
    >>> print(f"{ant.higuchi_fd(x):.4f}")
    1.0091

    Linearly-increasing time-series

    >>> x = np.arange(1000)
    >>> print(f"{ant.higuchi_fd(x):.4f}")
    1.0040
    """
    x = np.asarray(x, dtype=np.float64)
    kmax = int(kmax)
    return _higuchi_fd(x, kmax)


@jit((types.Array(types.float64, 1, "C", readonly=True),), nopython=True)
def _dfa(x):
    """
    Utility function for detrended fluctuation analysis
    """
    N = len(x)
    nvals = _log_n(4, 0.1 * N, 1.2)
    walk = np.cumsum(x - x.mean())
    fluctuations = np.zeros(len(nvals))

    for i_n, n in enumerate(nvals):
        d = np.reshape(walk[: N - (N % n)], (N // n, n))
        ran_n = np.array([float(na) for na in range(n)])
        d_len = len(d)
        trend = np.empty((d_len, ran_n.size))
        for i in range(d_len):
            slope, intercept = _linear_regression(ran_n, d[i])
            trend[i, :] = intercept + slope * ran_n
        # Calculate root mean squares of walks in d around trend
        # Note that np.mean on specific axis is not supported by Numba
        flucs = np.sum((d - trend) ** 2, axis=1) / n
        # https://github.com/neuropsychology/NeuroKit/issues/206
        fluctuations[i_n] = np.sqrt(np.mean(flucs))

    # Filter zero
    nonzero = np.nonzero(fluctuations)[0]
    fluctuations = fluctuations[nonzero]
    nvals = nvals[nonzero]
    if len(fluctuations) == 0:
        # all fluctuations are zero => we cannot fit a line
        dfa = np.nan
    else:
        dfa, _ = _linear_regression(np.log(nvals), np.log(fluctuations))
    return dfa


def detrended_fluctuation(x):
    """
    Detrended fluctuation analysis (DFA).

    Parameters
    ----------
    x : list or np.array
        One-dimensional time-series.

    Returns
    -------
    alpha : float
        the estimate alpha (:math:`\\alpha`) for the Hurst parameter.

        :math:`\\alpha < 1`` indicates a
        stationary process similar to fractional Gaussian noise with
        :math:`H = \\alpha`.

        :math:`\\alpha > 1`` indicates a non-stationary process similar to
        fractional Brownian motion with :math:`H = \\alpha - 1`

    Notes
    -----
    `Detrended fluctuation analysis
    <https://en.wikipedia.org/wiki/Detrended_fluctuation_analysis>`_
    is used to find long-term statistical dependencies in time series.

    The idea behind DFA originates from the definition of self-affine
    processes. A process :math:`X` is said to be self-affine if the standard
    deviation of the values within a window of length n changes with the window
    length factor :math:`L` in a power law:

    .. math:: \\text{std}(X, L * n) = L^H * \\text{std}(X, n)

    where :math:`\\text{std}(X, k)` is the standard deviation of the process
    :math:`X` calculated over windows of size :math:`k`. In this equation,
    :math:`H` is called the Hurst parameter, which behaves indeed very similar
    to the Hurst exponant.

    For more details, please refer to the excellent documentation of the
    `nolds <https://cschoel.github.io/nolds/>`_
    Python package by Christopher Scholzel, from which this function is taken:
    https://cschoel.github.io/nolds/nolds.html#detrended-fluctuation-analysis

    Note that the default subseries size is set to
    entropy.utils._log_n(4, 0.1 * len(x), 1.2)). The current implementation
    does not allow to manually specify the subseries size or use overlapping
    windows.

    The code is a faster (Numba) adaptation of the original code by Christopher
    Scholzel.

    References
    ----------
    * C.-K. Peng, S. V. Buldyrev, S. Havlin, M. Simons,
      H. E. Stanley, and A. L. Goldberger, “Mosaic organization of
      DNA nucleotides,” Physical Review E, vol. 49, no. 2, 1994.

    * R. Hardstone, S.-S. Poil, G. Schiavone, R. Jansen,
      V. V. Nikulin, H. D. Mansvelder, and K. Linkenkaer-Hansen,
      “Detrended fluctuation analysis: A scale-free view on neuronal
      oscillations,” Frontiers in Physiology, vol. 30, 2012.

    Examples
    --------
    Fractional Gaussian noise with H = 0.5

    >>> import numpy as np
    >>> import antropy as ant
    >>> import stochastic.processes.noise as sn
    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.5, rng=rng).sample(10000)
    >>> print(f"{ant.detrended_fluctuation(x):.4f}")
    0.5216

    Fractional Gaussian noise with H = 0.9

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.9, rng=rng).sample(10000)
    >>> print(f"{ant.detrended_fluctuation(x):.4f}")
    0.8833

    Fractional Gaussian noise with H = 0.1

    >>> rng = np.random.default_rng(seed=42)
    >>> x = sn.FractionalGaussianNoise(hurst=0.1, rng=rng).sample(10000)
    >>> print(f"{ant.detrended_fluctuation(x):.4f}")
    0.1262

    Random

    >>> rng = np.random.default_rng(seed=42)
    >>> print(f"{ant.detrended_fluctuation(rng.random(1000)):.4f}")
    0.5276

    Pure sine wave

    >>> x = np.sin(2 * np.pi * 1 * np.arange(3000) / 100)
    >>> print(f"{ant.detrended_fluctuation(x):.4f}")
    1.5848

    Linearly-increasing time-series

    >>> x = np.arange(1000)
    >>> print(f"{ant.detrended_fluctuation(x):.4f}")
    2.0390
    """
    x = np.asarray(x, dtype=np.float64)
    return _dfa(x)
