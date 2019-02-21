import numpy as np
from numba import jit
from math import log, floor

from .utils import _linear_regression

all = ['petrosian_fd', 'katz_fd', 'higuchi_fd']


def petrosian_fd(x):
    """Petrosian fractal dimension.

    Parameters
    ----------
    x : list or np.array
        One dimensional time series

    Returns
    -------
    pfd : float
        Petrosian fractal dimension

    Notes
    -----
    The Petrosian algorithm can be used to provide a fast computation of
    the FD of a signal by translating the series into a binary sequence.

    The Petrosian fractal dimension of a time series :math:`x` is defined by:

    .. math:: \\frac{log_{10}(N)}{log_{10}(N) +
       log_{10}(\\frac{N}{N+0.4N_{\\Delta}})}

    where :math:`N` is the length of the time series, and
    :math:`N_{\\Delta}` is the number of sign changes in the binary sequence.

    Original code from the pyrem package by Quentin Geissmann.

    References
    ----------
    .. [1] A. Petrosian, Kolmogorov complexity of finite sequences and
       recognition of different preictal EEG patterns, in , Proceedings of the
       Eighth IEEE Symposium on Computer-Based Medical Systems, 1995,
       pp. 212-217.

    .. [2] Goh, Cindy, et al. "Comparison of fractal dimension algorithms for
       the computation of EEG biomarkers for dementia." 2nd International
       Conference on Computational Intelligence in Medicine and Healthcare
       (CIMED2005). 2005.

    Examples
    --------
    Petrosian fractal dimension.

        >>> import numpy as np
        >>> from entropy import petrosian_fd
        >>> np.random.seed(123)
        >>> x = np.random.rand(100)
        >>> print(petrosian_fd(x))
            1.0505
    """
    n = len(x)
    # Number of sign changes in the first derivative of the signal
    diff = np.ediff1d(x)
    N_delta = (diff[1:-1] * diff[0:-2] < 0).sum()
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * N_delta)))


def katz_fd(x):
    """Katz Fractal Dimension.

    Parameters
    ----------
    x : list or np.array
        One dimensional time series

    Returns
    -------
    kfd : float
        Katz fractal dimension

    Notes
    -----
    The Katz Fractal dimension is defined by:

    .. math:: FD_{Katz} = \\frac{log_{10}(n)}{log_{10}(d/L)+log_{10}(n)}

    where :math:`L` is the total length of the time series and :math:`d`
    is the Euclidean distance between the first point in the
    series and the point that provides the furthest distance
    with respect to the first point.

    Original code from the mne-features package by Jean-Baptiste Schiratti
    and Alexandre Gramfort.

    References
    ----------
    .. [1] Esteller, R. et al. (2001). A comparison of waveform fractal
           dimension algorithms. IEEE Transactions on Circuits and Systems I:
           Fundamental Theory and Applications, 48(2), 177-183.

    .. [2] Goh, Cindy, et al. "Comparison of fractal dimension algorithms for
           the computation of EEG biomarkers for dementia." 2nd International
           Conference on Computational Intelligence in Medicine and Healthcare
           (CIMED2005). 2005.

    Examples
    --------
    Katz fractal dimension.

        >>> import numpy as np
        >>> from entropy import katz_fd
        >>> np.random.seed(123)
        >>> x = np.random.rand(100)
        >>> print(katz_fd(x))
            5.1214
    """
    x = np.array(x)
    dists = np.abs(np.ediff1d(x))
    ll = dists.sum()
    ln = np.log10(np.divide(ll, dists.mean()))
    aux_d = x - x[0]
    d = np.max(np.abs(aux_d[1:]))
    return np.divide(ln, np.add(ln, np.log10(np.divide(d, ll))))


@jit('float64(float64[:], int32)')
def _higuchi_fd(x, kmax):
    """Utility function for `higuchi_fd`.
    """
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
        x_reg[k - 1] = log(1. / k)
        y_reg[k - 1] = log(m_lm)
    higuchi, _ = _linear_regression(x_reg, y_reg)
    return higuchi


def higuchi_fd(x, kmax=10):
    """Higuchi Fractal Dimension.

    Parameters
    ----------
    x : list or np.array
        One dimensional time series
    kmax : int
        Maximum delay/offset (in number of samples).

    Returns
    -------
    hfd : float
        Higuchi Fractal Dimension

    Notes
    -----
    Original code from the mne-features package by Jean-Baptiste Schiratti
    and Alexandre Gramfort.

    The `higuchi_fd` function uses Numba to speed up the computation.

    References
    ----------
    .. [1] Higuchi, Tomoyuki. "Approach to an irregular time series on the
       basis of the fractal theory." Physica D: Nonlinear Phenomena 31.2
       (1988): 277-283.

    Examples
    --------
    Higuchi Fractal Dimension

        >>> import numpy as np
        >>> from entropy import higuchi_fd
        >>> np.random.seed(123)
        >>> x = np.random.rand(100)
        >>> print(higuchi_fd(x))
            2.051179
    """
    x = np.asarray(x, dtype=np.float64)
    kmax = int(kmax)
    return _higuchi_fd(x, kmax)
