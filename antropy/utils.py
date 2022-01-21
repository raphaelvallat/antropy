"""Helper functions"""
import numpy as np
from numba import jit
from math import log, floor

all = ['_embed', '_linear_regression', '_log_n', '_xlog2x']


def _embed_modified(x, order=3, delay=1):
    """Time-delay embedding.

    Parameters
    ----------
    x : 1D-array of shape (n_times) or 2D-array of shape (signal_indice, n_times)
    order : int
        Embedding dimension (order).
    delay : int
        Delay.

    Returns
    -------
    embedded : 2D-array (if x is 1D)
        Embedded time-series, of shape (n_times - (order - 1) * delay, order)
    embedded : 3D-array if (x is 2D)
        Embedded time-series, of shape (signal_indice, n_times - (order - 1) * delay, order_num) 
    """

    assert type(order) == int, "order must be integer!"
    # check order is int

    N = x.shape[-1]
    if order * delay > N:
        raise ValueError("Error: order * delay should be lower than x.size")
    if delay < 1:
        raise ValueError("Delay has to be at least 1.")
    if order < 2:
        raise ValueError("Order has to be at least 2.")
    # check parameters

    if x.ndim == 1:
    # pass 1D array

        Y = np.zeros((order, N - (order - 1) * delay))
        for i in range(order):
            Y[i] = x[(i * delay):(i * delay + Y.shape[1])]
        return Y.T

    else:
    # pass 2D array

        Y = []
        # pre-defiend an empty list to store numpy.array (concatenate with a list is faster)

        embed_signal_length = N - (order - 1) * delay
        # define the new signal length

        indice = [[(i * delay), (i * delay+embed_signal_length)] for i in range(order)]
        # generate a list of slice indice on input signal

        for i in range(order):
        # loop with the order
            temp = x[:, indice[i][0]: indice[i][1]].reshape(-1, embed_signal_length, 1)
            # slicing the signal with the indice of each order (vectorized operation)

            Y.append(temp)
            # append the sliced signal to list 

        Y = np.concatenate(Y, axis=-1)
        # concatenate the sliced signal to a 3D array (signal_indice, n_times - (order - 1) * delay, order_num) 
        return Y


@jit('UniTuple(float64, 2)(float64[:], float64[:])', nopython=True)
def _linear_regression(x, y):
    """Fast linear regression using Numba.

    Parameters
    ----------
    x, y : ndarray, shape (n_times,)
        Variables

    Returns
    -------
    slope : float
        Slope of 1D least-square regression.
    intercept : float
        Intercept
    """
    n_times = x.size
    sx2 = 0
    sx = 0
    sy = 0
    sxy = 0
    for j in range(n_times):
        sx2 += x[j] ** 2
        sx += x[j]
        sxy += x[j] * y[j]
        sy += y[j]
    den = n_times * sx2 - (sx ** 2)
    num = n_times * sxy - sx * sy
    slope = num / den
    intercept = np.mean(y) - slope * np.mean(x)
    return slope, intercept


@jit('i8[:](f8, f8, f8)', nopython=True)
def _log_n(min_n, max_n, factor):
    """
    Creates a list of integer values by successively multiplying a minimum
    value min_n by a factor > 1 until a maximum value max_n is reached.

    Used for detrended fluctuation analysis (DFA).

    Function taken from the nolds python package
    (https://github.com/CSchoel/nolds) by Christopher Scholzel.

    Parameters
    ----------
    min_n (float):
        minimum value (must be < max_n)
    max_n (float):
        maximum value (must be > min_n)
    factor (float):
       factor used to increase min_n (must be > 1)

    Returns
    -------
    list of integers:
        min_n, min_n * factor, min_n * factor^2, ... min_n * factor^i < max_n
        without duplicates
    """
    max_i = int(floor(log(1.0 * max_n / min_n) / log(factor)))
    ns = [min_n]
    for i in range(max_i + 1):
        n = int(floor(min_n * (factor ** i)))
        if n > ns[-1]:
            ns.append(n)
    return np.array(ns, dtype=np.int64)


def _xlogx(x, base=2):
    """Returns x log_b x if x is positive, 0 if x == 0, and np.nan
    otherwise. This handles the case when the power spectrum density
    takes any zero value.
    """
    x = np.asarray(x)
    xlogx = np.zeros(x.shape)
    xlogx[x < 0] = np.nan
    valid = x > 0
    xlogx[valid] = x[valid] * np.log(x[valid]) / np.log(base)
    return xlogx
