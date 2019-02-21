import numpy as np
from numba import jit

from .utils import _linear_regression, _log_n

all = ['detrended_fluctuation']


@jit('f8(f8[:])', nopython=True)
def _dfa(x):
    """
    Utility function for detrended fluctuation analysis
    """
    N = len(x)
    nvals = _log_n(4, 0.1 * N, 1.2)
    walk = np.cumsum(x - x.mean())
    fluctuations = np.zeros(len(nvals))

    for i_n, n in enumerate(nvals):
        d = np.reshape(walk[:N - (N % n)], (N // n, n))
        ran_n = np.array([float(na) for na in range(n)])
        d_len = len(d)
        slope = np.empty(d_len)
        intercept = np.empty(d_len)
        trend = np.empty((d_len, ran_n.size))
        for i in range(d_len):
            slope[i], intercept[i] = _linear_regression(ran_n, d[i])
            y = np.zeros_like(ran_n)
            # Equivalent to np.polyval function
            for p in [slope[i], intercept[i]]:
                y = y * ran_n + p
            trend[i, :] = y
        # calculate standard deviation (fluctuation) of walks in d around trend
        flucs = np.sqrt(np.sum((d - trend) ** 2, axis=1) / n)
        # calculate mean fluctuation over all subsequences
        fluctuations[i_n] = flucs.sum() / flucs.size

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
    dfa : float
        the estimate alpha for the Hurst parameter:

        alpha < 1: stationary process similar to fractional Gaussian noise
        with H = alpha

        alpha > 1: non-stationary process similar to fractional Brownian
        motion with H = alpha - 1

    Notes
    -----
    Detrended fluctuation analysis (DFA) is used to find long-term statistical
    dependencies in time series.

    The idea behind DFA originates from the definition of self-affine
    processes. A process :math:`X` is said to be self-affine if the standard
    deviation of the values within a window of length n changes with the window
    length factor L in a power law:

    .. math:: \\text{std}(X, L * n) = L^H * \\text{std}(X, n)

    where :math:`\\text{std}(X, k)` is the standard deviation of the process
    :math:`X` calculated over windows of size :math:`k`. In this equation,
    :math:`H` is called the Hurst parameter, which behaves indeed very similar
    to the Hurst exponent.

    For more details, please refer to the excellent documentation of the nolds
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
    .. [1] C.-K. Peng, S. V. Buldyrev, S. Havlin, M. Simons,
           H. E. Stanley, and A. L. Goldberger, “Mosaic organization of
           DNA nucleotides,” Physical Review E, vol. 49, no. 2, 1994.

    .. [2] R. Hardstone, S.-S. Poil, G. Schiavone, R. Jansen,
           V. V. Nikulin, H. D. Mansvelder, and K. Linkenkaer-Hansen,
           “Detrended fluctuation analysis: A scale-free view on neuronal
           oscillations,” Frontiers in Physiology, vol. 30, 2012.

    Examples
    --------

        >>> import numpy as np
        >>> from entropy import detrended_fluctuation
        >>> np.random.seed(123)
        >>> x = np.random.rand(100)
        >>> print(detrended_fluctuation(x))
            0.761647725305623
    """
    x = np.asarray(x, dtype=np.float64)
    return _dfa(x)
