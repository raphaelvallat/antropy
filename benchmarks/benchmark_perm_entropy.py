"""Benchmark perm_entropy: fast path (order=3/4) vs original argsort implementation.

1D signals
----------
Compares perm_entropy() [fast path] against the original argsort-based
implementation for a single time series.

2D signals
----------
Compares perm_entropy() [fast path, vectorized] against
np.apply_along_axis(original, axis=1, arr=x) for multi-channel data.

Usage
-----
    python benchmarks/benchmark_perm_entropy.py
"""

import timeit

import numpy as np
from numpy import apply_along_axis as aal

import antropy as ant
from antropy.utils import _embed

# ---------------------------------------------------------------------------
# Reference implementation: original argsort path, no fast path
# ---------------------------------------------------------------------------


def _perm_entropy_orig(x, order=3, delay=1, normalize=False):
    """Original argsort-based implementation used as the reference baseline."""
    from math import factorial

    x = np.asarray(x)
    hashmult = np.power(order, np.arange(order))
    sorted_idx = _embed(x, order=order, delay=delay).argsort(kind="quicksort")
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    _, counts = np.unique(hashval, return_counts=True)
    p = counts / counts.sum()
    pe = -(p * np.log2(p)).sum()
    if normalize:
        pe /= np.log2(factorial(order))
    return pe


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _time_ms(fn, number):
    """Return mean wall time in milliseconds over `number` repetitions."""
    return timeit.timeit(fn, number=number) / number * 1e3


def _speedup(t_ref, t_fast):
    return t_ref / t_fast


def _header(title):
    print()
    print(title)
    print("-" * len(title))


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

LENGTHS = (1_000, 10_000)
N_CHANNELS = (5, 50, 500)
ORDERS = (3, 4)
RNG = np.random.default_rng(0)


# Number of repetitions: fewer for large/slow combinations
def _nrep(n_ch, length):
    total = n_ch * length
    if total >= 5_000_000:
        return 5
    if total >= 500_000:
        return 20
    return 100


# ---------------------------------------------------------------------------
# 1D benchmark
# ---------------------------------------------------------------------------


def bench_1d():
    _header("1D signals  (single time series)")
    print(
        f"{'Order':>5}  {'Length':>8}  {'Fast path (µs)':>16}  {'Original (µs)':>15}  {'Speedup':>8}"
    )
    print(f"{'':->5}  {'':->8}  {'':->16}  {'':->15}  {'':->8}")

    for order in ORDERS:
        for length in LENGTHS:
            x = RNG.random(length)
            n = _nrep(1, length)
            t_fast = _time_ms(lambda: ant.perm_entropy(x, order=order), number=n) * 1e3  # µs
            t_orig = _time_ms(lambda: _perm_entropy_orig(x, order=order), number=n) * 1e3
            print(
                f"{order:>5}  {length:>8,}  {t_fast:>16.1f}  {t_orig:>15.1f}  {_speedup(t_orig, t_fast):>7.1f}x"
            )


# ---------------------------------------------------------------------------
# 2D benchmark
# ---------------------------------------------------------------------------


def bench_2d():
    _header("2D signals  (perm_entropy on 2D array  vs  apply_along_axis on original)")
    print(
        f"{'Order':>5}  {'Channels':>8}  {'Length':>8}"
        f"  {'Fast path (ms)':>15}  {'apply_along_axis (ms)':>21}  {'Speedup':>8}"
    )
    print(f"{'':->5}  {'':->8}  {'':->8}  {'':->15}  {'':->21}  {'':->8}")

    for order in ORDERS:
        for n_ch in N_CHANNELS:
            for length in LENGTHS:
                x = RNG.random((n_ch, length))
                n = _nrep(n_ch, length)
                t_fast = _time_ms(lambda: ant.perm_entropy(x, order=order), number=n)
                t_orig = _time_ms(
                    lambda: aal(_perm_entropy_orig, axis=1, arr=x, order=order),
                    number=n,
                )
                print(
                    f"{order:>5}  {n_ch:>8}  {length:>8,}"
                    f"  {t_fast:>15.2f}  {t_orig:>21.2f}  {_speedup(t_orig, t_fast):>7.1f}x"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Benchmarking perm_entropy: fast path vs original implementation")
    print("=" * 63)
    bench_1d()
    bench_2d()
    print()
