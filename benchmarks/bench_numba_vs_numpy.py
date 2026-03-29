"""
Benchmark: old implementations vs new NumPy-only implementations.

Only covers the four functions whose implementation changed to NumPy:

  perm_entropy          _embed: np.zeros copy-loop  →  sliding_window_view
  svd_entropy           _embed: np.zeros copy-loop  →  sliding_window_view
  detrended_fluctuation Numba per-window regression  →  BLAS d_c @ t_c
  sample_entropy        Numba O(N²) sliding window  →  KDTree O(N log N)

higuchi_fd and lziv_complexity remain Numba — they were reverted after
benchmarking showed NumPy offered no speedup at any practical signal length.

Signal lengths tested: N = 100, 500, 1_000, 2_000, 10_000, 100_000
(sample_entropy skipped at N = 100_000 — old implementation is O(N²))

Usage:
    python benchmarks/bench_numba_vs_numpy.py
"""

import timeit
from math import floor, log

import numpy as np
from numba import jit, types

import antropy as ant
from antropy.utils import _xlogx

# ── shared constant ───────────────────────────────────────────────────────────
epsilon = 1e-9


# ══════════════════════════════════════════════════════════════════════════════
# OLD implementations
# ══════════════════════════════════════════════════════════════════════════════

# ── perm_entropy / svd_entropy: old _embed uses np.zeros + copy loop ──────────


def _old_embed(x, order, delay=1):
    N = x.shape[-1]
    Y = np.zeros((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[(i * delay) : (i * delay + Y.shape[1])]
    return Y.T


def _old_perm_entropy(x, order=3, delay=1):
    x = np.array(x)
    ran_order = range(order)
    hashmult = np.power(order, ran_order)
    sorted_idx = _old_embed(x, order=order, delay=delay).argsort(kind="quicksort")
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    _, c = np.unique(hashval, return_counts=True)
    p = c / c.sum()
    return -_xlogx(p).sum()


def _old_svd_entropy(x, order=3, delay=1):
    x = np.array(x)
    mat = _old_embed(x, order=order, delay=delay)
    W = np.linalg.svd(mat, compute_uv=False)
    W /= W.sum()
    return -_xlogx(W).sum()


# ── detrended_fluctuation: Numba per-window linear regression loop ────────────


@jit("UniTuple(float64, 2)(float64[:], float64[:])", nopython=True)
def _old_linear_regression(x, y):
    n_times = x.size
    sx2 = sx = sy = sxy = 0.0
    for j in range(n_times):
        sx2 += x[j] ** 2
        sx += x[j]
        sxy += x[j] * y[j]
        sy += y[j]
    den = n_times * sx2 - sx**2
    num = n_times * sxy - sx * sy
    slope = num / (den + epsilon)
    intercept = np.mean(y) - slope * np.mean(x)
    return slope, intercept


@jit((types.Array(types.float64, 1, "C", readonly=True),), nopython=True)
def _old_dfa(x):
    N = len(x)
    # inline _log_n
    min_n, max_n, factor = 4, 0.1 * N, 1.2
    max_i = int(floor(log(1.0 * max_n / min_n) / log(factor)))
    ns = [min_n]
    for i in range(max_i + 1):
        n = int(floor(min_n * (factor**i)))
        if n > ns[-1]:
            ns.append(n)
    nvals = np.array(ns, dtype=np.int64)

    walk = np.cumsum(x - x.mean())
    fluctuations = np.zeros(len(nvals))
    for i_n, n in enumerate(nvals):
        d = np.reshape(walk[: N - (N % n)], (N // n, n))
        ran_n = np.array([float(na) for na in range(n)])
        d_len = len(d)
        trend = np.empty((d_len, ran_n.size))
        for i in range(d_len):
            slope, intercept = _old_linear_regression(ran_n, d[i])
            trend[i, :] = intercept + slope * ran_n
        flucs = np.sum((d - trend) ** 2, axis=1) / n
        fluctuations[i_n] = np.sqrt(np.mean(flucs))
    nonzero = np.nonzero(fluctuations)[0]
    fluctuations = fluctuations[nonzero]
    nvals = nvals[nonzero]
    if len(fluctuations) == 0:
        return np.nan
    dfa, _ = _old_linear_regression(np.log(nvals.astype(np.float64)), np.log(fluctuations))
    return dfa


# ── sample_entropy: Numba O(N²) sliding window ────────────────────────────────


@jit(
    (types.Array(types.float64, 1, "C", readonly=True), types.int32, types.float64),
    nopython=True,
)
def _old_numba_sampen(sequence, order, r):
    size = sequence.size
    numerator = denominator = 0
    for offset in range(1, size - order):
        n_num = int(abs(sequence[order] - sequence[order + offset]) >= r)
        n_den = 0
        for idx in range(order):
            diff = abs(sequence[idx] - sequence[idx + offset]) >= r
            n_num += diff
            n_den += diff
        if n_num == 0:
            numerator += 1
        if n_den == 0:
            denominator += 1
        prev = int(abs(sequence[order] - sequence[offset + order]) >= r)
        for idx in range(1, size - offset - order):
            out = int(abs(sequence[idx - 1] - sequence[idx + offset - 1]) >= r)
            inn = int(abs(sequence[idx + order] - sequence[idx + offset + order]) >= r)
            n_num += inn - out
            n_den += prev - out
            prev = inn
            if n_num == 0:
                numerator += 1
            if n_den == 0:
                denominator += 1
    if denominator == 0:
        return np.nan
    elif numerator == 0:
        return np.inf
    return -log(numerator / denominator)


# ══════════════════════════════════════════════════════════════════════════════
# Benchmark helpers
# ══════════════════════════════════════════════════════════════════════════════


def _timeit(func, n_repeat=5, n_number=20):
    """Return (mean_ms, std_ms) after one warm-up call."""
    func()
    times = timeit.repeat(func, repeat=n_repeat, number=n_number)
    ms = [t / n_number * 1e3 for t in times]
    return float(np.mean(ms)), float(np.std(ms))


def _row(name, old_ms, old_std, new_ms, new_std):
    speedup = old_ms / new_ms
    marker = " ✓" if speedup >= 0.9 else ""
    print(
        f"  {name:<28s}"
        f"  old: {old_ms:8.3f} ± {old_std:.3f} ms"
        f"  │  new: {new_ms:8.3f} ± {new_std:.3f} ms"
        f"  │  {speedup:5.2f}x{marker}"
    )


def _check(name, old_val, new_val, tol=1e-6):
    if np.isnan(old_val) and np.isnan(new_val):
        print(f"  [ok] {name}: both NaN")
        return
    err = abs(float(old_val) - float(new_val))
    tag = "[ok]" if err <= tol else "*** MISMATCH"
    print(f"  {tag} {name}: Δ = {err:.2e}")


def _params(name, N):
    """(n_repeat, n_number) tuned to keep total run time reasonable."""
    if name == "sample_entropy":
        if N >= 10_000:
            return 3, 3
        if N >= 1_000:
            return 5, 10
        return 5, 30
    if name == "detrended_fluctuation":
        if N >= 100_000:
            return 5, 5
        return 5, 20
    if N >= 100_000:
        return 5, 10
    return 5, 20


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    print("=" * 78)
    print("Warming up Numba JIT (one-time compilation)…")
    _w = rng.standard_normal(200).astype(np.float64)
    _ = _old_linear_regression(np.arange(10, dtype=np.float64), np.arange(10, dtype=np.float64))
    _ = _old_dfa(_w)
    _ = _old_numba_sampen(_w, np.int32(2), np.float64(0.2 * _w.std()))
    print("Done.\n")

    for N in (100, 500, 1_000, 2_000, 10_000, 100_000):
        x = rng.standard_normal(N).astype(np.float64)
        r = float(0.2 * x.std())

        print("=" * 78)
        print(f"N = {N:,}")
        print("-" * 78)

        # ── perm_entropy ──────────────────────────────────────────────────────
        n_r, n_n = _params("perm_entropy", N)
        o = _timeit(lambda: _old_perm_entropy(x), n_repeat=n_r, n_number=n_n)
        nw = _timeit(lambda: ant.perm_entropy(x), n_repeat=n_r, n_number=n_n)
        _check("perm_entropy", _old_perm_entropy(x), ant.perm_entropy(x))
        _row("perm_entropy", *o, *nw)

        # ── svd_entropy ───────────────────────────────────────────────────────
        n_r, n_n = _params("svd_entropy", N)
        o = _timeit(lambda: _old_svd_entropy(x), n_repeat=n_r, n_number=n_n)
        nw = _timeit(lambda: ant.svd_entropy(x), n_repeat=n_r, n_number=n_n)
        _check("svd_entropy", _old_svd_entropy(x), ant.svd_entropy(x))
        _row("svd_entropy", *o, *nw)

        # ── detrended_fluctuation ─────────────────────────────────────────────
        n_r, n_n = _params("detrended_fluctuation", N)
        o = _timeit(lambda: _old_dfa(x), n_repeat=n_r, n_number=n_n)
        nw = _timeit(lambda: ant.detrended_fluctuation(x), n_repeat=n_r, n_number=n_n)
        _check("detrended_fluctuation", float(_old_dfa(x)), ant.detrended_fluctuation(x))
        _row("detrended_fluctuation", *o, *nw)

        # ── sample_entropy (skipped at N = 100_000) ───────────────────────────
        if N <= 10_000:
            n_r, n_n = _params("sample_entropy", N)
            o = _timeit(
                lambda: _old_numba_sampen(x, np.int32(2), np.float64(r)),
                n_repeat=n_r,
                n_number=n_n,
            )
            nw = _timeit(lambda: ant.sample_entropy(x, order=2), n_repeat=n_r, n_number=n_n)
            _check(
                "sample_entropy",
                float(_old_numba_sampen(x, np.int32(2), np.float64(r))),
                ant.sample_entropy(x, order=2),
                tol=1e-4,
            )
            _row("sample_entropy", *o, *nw)
        else:
            print(f"  {'sample_entropy':<28s}  (skipped — old is O(N²))")

        print()

    print("=" * 78)
    print("All benchmarks complete.")
