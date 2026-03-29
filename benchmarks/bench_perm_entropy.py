"""
Fine-grained perm_entropy benchmark: old _embed (np.zeros copy-loop) vs
new _embed (sliding_window_view, zero-copy).

Sweeps N from 100 to 100_000 to identify the crossover and quantify the
speedup at each length.

Usage:
    python benchmarks/bench_perm_entropy.py
"""

import timeit

import numpy as np

import antropy as ant
from antropy.utils import _xlogx

# ── Old implementation (loop-based _embed) ────────────────────────────────────


def _old_embed(x, order, delay=1):
    N = x.shape[-1]
    Y = np.zeros((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[(i * delay) : (i * delay + Y.shape[1])]
    return Y.T


def _old_perm_entropy(x, order=3, delay=1):
    x = np.array(x)
    hashmult = np.power(order, range(order))
    sorted_idx = _old_embed(x, order=order, delay=delay).argsort(kind="quicksort")
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    _, c = np.unique(hashval, return_counts=True)
    p = c / c.sum()
    return -_xlogx(p).sum()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _timeit(func, n_repeat=7, n_number=50):
    func()  # warm-up
    times = timeit.repeat(func, repeat=n_repeat, number=n_number)
    ms = [t / n_number * 1e3 for t in times]
    return float(np.mean(ms)), float(np.std(ms))


def _params(N):
    if N >= 50_000:
        return 5, 10
    if N >= 10_000:
        return 7, 20
    if N >= 1_000:
        return 7, 50
    return 7, 100


# ── Main ──────────────────────────────────────────────────────────────────────

SIZES = [100, 250, 500, 750, 1_000, 1_500, 2_000, 3_000, 5_000, 10_000, 20_000, 50_000, 100_000]

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    print("=" * 68)
    print("perm_entropy: old _embed (copy-loop) vs new _embed (sliding_window_view)")
    print(f"{'N':>8}  {'old (ms)':>12}  {'new (ms)':>12}  {'speedup':>8}")
    print("-" * 68)

    for N in SIZES:
        x = rng.standard_normal(N)
        n_r, n_n = _params(N)

        old_ms, old_std = _timeit(lambda: _old_perm_entropy(x), n_repeat=n_r, n_number=n_n)
        new_ms, new_std = _timeit(lambda: ant.perm_entropy(x), n_repeat=n_r, n_number=n_n)

        speedup = old_ms / new_ms
        marker = " ✓" if speedup >= 1.0 else ""
        print(
            f"  {N:>6,}"
            f"  {old_ms:8.4f} ± {old_std:.4f}"
            f"  {new_ms:8.4f} ± {new_std:.4f}"
            f"  {speedup:7.2f}x{marker}"
        )

    print("=" * 68)
