"""Benchmark perm_entropy: fast path vs improved general path vs original.

Three implementations are compared:

  original      -- argsort on _embed output + np.unique (Antropy v0.2.1)
  general       -- as_strided zero-copy view + argsort @ hashmult + np.unique
                   (the improved fallback path used for order > 4)
  fast path     -- pairwise comparison bit-keys + lookup table + np.bincount
                   (used by perm_entropy for order=3/4; supports 2D natively)

1D signals
----------
All three implementations are compared on a single time series.

2D signals
----------
fast path  vs  apply_along_axis(original)  vs  apply_along_axis(general).

Usage
-----
    python benchmarks/benchmark_perm_entropy.py
"""

import timeit
from math import factorial

import numpy as np
from numpy import apply_along_axis as aal
from numpy.lib.stride_tricks import as_strided

import antropy as ant
from antropy.utils import _embed

# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _perm_entropy_orig(x, order=3, delay=1, normalize=False):
    """Original: argsort on _embed output + np.unique."""
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


def _perm_entropy_general(x, order=3, delay=1, normalize=False):
    """Improved general path: as_strided (zero-copy) + argsort @ hashmult + np.unique."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    n_embed = n - (order - 1) * delay
    embedded = as_strided(
        x,
        shape=(n_embed, order),
        strides=(x.strides[0], x.strides[0] * delay),
    )
    hashmult = np.power(order, np.arange(order))
    hashval = embedded.argsort(axis=1, kind="quicksort") @ hashmult
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


def _header(title):
    print()
    print(title)
    print("-" * len(title))


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

LENGTHS = (1_000, 5_000, 10_000)
N_CHANNELS = (5, 50, 500)
ORDERS = (3, 4)
RNG = np.random.default_rng(0)


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
        f"{'Order':>5}  {'Length':>8}"
        f"  {'Fast path (µs)':>16}  {'General (µs)':>14}  {'Original (µs)':>15}"
        f"  {'vs General':>11}  {'vs Original':>12}"
    )
    print(f"{'':->5}  {'':->8}  {'':->16}  {'':->14}  {'':->15}  {'':->11}  {'':->12}")

    for order in ORDERS:
        for length in LENGTHS:
            x = RNG.random(length)
            n = _nrep(1, length)
            t_fast = _time_ms(lambda: ant.perm_entropy(x, order=order), number=n) * 1e3  # µs
            t_gen = _time_ms(lambda: _perm_entropy_general(x, order=order), number=n) * 1e3
            t_orig = _time_ms(lambda: _perm_entropy_orig(x, order=order), number=n) * 1e3
            print(
                f"{order:>5}  {length:>8,}"
                f"  {t_fast:>16.1f}  {t_gen:>14.1f}  {t_orig:>15.1f}"
                f"  {t_gen / t_fast:>10.1f}x  {t_orig / t_fast:>11.1f}x"
            )


# ---------------------------------------------------------------------------
# 2D benchmark
# ---------------------------------------------------------------------------


def bench_2d():
    _header(
        "2D signals  (fast path  vs  apply_along_axis(general)  vs  apply_along_axis(original))"
    )
    print(
        f"{'Order':>5}  {'Ch':>4}  {'Length':>8}"
        f"  {'Fast (ms)':>10}  {'General AAL (ms)':>17}  {'Original AAL (ms)':>18}"
        f"  {'vs General':>11}  {'vs Original':>12}"
    )
    print(f"{'':->5}  {'':->4}  {'':->8}  {'':->10}  {'':->17}  {'':->18}  {'':->11}  {'':->12}")

    for order in ORDERS:
        for n_ch in N_CHANNELS:
            for length in LENGTHS:
                x = RNG.random((n_ch, length))
                n = _nrep(n_ch, length)
                t_fast = _time_ms(lambda: ant.perm_entropy(x, order=order), number=n)
                t_gen = _time_ms(
                    lambda: aal(_perm_entropy_general, axis=1, arr=x, order=order),
                    number=n,
                )
                t_orig = _time_ms(
                    lambda: aal(_perm_entropy_orig, axis=1, arr=x, order=order),
                    number=n,
                )
                print(
                    f"{order:>5}  {n_ch:>4}  {length:>8,}"
                    f"  {t_fast:>10.2f}  {t_gen:>17.2f}  {t_orig:>18.2f}"
                    f"  {t_gen / t_fast:>10.1f}x  {t_orig / t_fast:>11.1f}x"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Benchmarking perm_entropy: fast path vs general path vs original")
    print("=" * 65)
    bench_1d()
    bench_2d()
    print()
