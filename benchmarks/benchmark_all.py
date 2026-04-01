"""Benchmark all public antropy functions on 1 000- and 10 000-sample signals.

Usage
-----
    python benchmarks/benchmark_all.py
"""

import timeit

import numpy as np

import antropy as ant

RNG = np.random.default_rng(42)
LENGTHS = (1_000, 10_000)


def _time_us(fn, number=500):
    """Return mean wall time in microseconds."""
    # One warm-up call (important for Numba-compiled functions).
    fn()
    return timeit.timeit(fn, number=number) / number * 1e6


def main():
    results = {}

    for n in LENGTHS:
        x = RNG.standard_normal(n)
        x_bin = (x > 0).astype(int)

        timings = {}

        timings["ant.perm_entropy"] = _time_us(lambda: ant.perm_entropy(x, normalize=True))
        timings["ant.spectral_entropy"] = _time_us(
            lambda: ant.spectral_entropy(x, sf=100, method="welch", normalize=True)
        )
        timings["ant.svd_entropy"] = _time_us(lambda: ant.svd_entropy(x, normalize=True))
        timings["ant.app_entropy"] = _time_us(lambda: ant.app_entropy(x), number=20)
        timings["ant.sample_entropy"] = _time_us(lambda: ant.sample_entropy(x), number=20)
        timings["ant.lziv_complexity"] = _time_us(
            lambda: ant.lziv_complexity(x_bin, normalize=True)
        )
        timings["ant.num_zerocross"] = _time_us(lambda: ant.num_zerocross(x))
        timings["ant.hjorth_params"] = _time_us(lambda: ant.hjorth_params(x))
        timings["ant.petrosian_fd"] = _time_us(lambda: ant.petrosian_fd(x))
        timings["ant.katz_fd"] = _time_us(lambda: ant.katz_fd(x))
        timings["ant.higuchi_fd"] = _time_us(lambda: ant.higuchi_fd(x), number=100)
        timings["ant.detrended_fluctuation"] = _time_us(
            lambda: ant.detrended_fluctuation(x), number=100
        )

        results[n] = timings

    # Pretty-print results
    funcs = list(results[LENGTHS[0]].keys())
    col_w = 14

    header = f"{'Function':<30}" + "".join(f"  {n:>{col_w},} samples" for n in LENGTHS)
    print(header)
    print("-" * len(header))
    for fn in funcs:
        row = f"{fn:<30}"
        for n in LENGTHS:
            t = results[n][fn]
            if t >= 1000:
                row += f"  {t / 1000:>{col_w}.2f} ms   "
            else:
                row += f"  {t:>{col_w}.1f} µs   "
        print(row)

    return results


if __name__ == "__main__":
    main()
