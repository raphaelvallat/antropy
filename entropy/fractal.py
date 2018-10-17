import numpy as np

all = ['petrosian_fd', 'katz_fd']


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
    The Petrosian algorithm [1]_ can be used to provide a fast computation of
    the FD of a signal by translating the series into a binary sequence.

    The Petrosian fractal dimension of a time series :math:`x` is defined by:

    .. math:: \dfrac{log(N)}{log(N) + log(\dfrac{N}{N+0.4N_{\Delta}})}

    where :math:`N` is the length of the time series, and
    :math:`N_{\Delta}` is the number of sign changes in the binary sequence.

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
    return np.log(n) / (np.log(n) + np.log(n / (n + 0.4 * N_delta)))


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

    .. math:: FD_{Katz} = \dfrac{log_{10}(n)}{log_{10}(d/L)+log_{10}(n))}

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
