# Antropy benchmarks — NumPy rewrite evaluation

This document records the performance evaluation of NumPy-only alternatives to
the Numba implementations tested during the rewrite session.  For each
function, the **old** column is the original Numba JIT code; the **new** column
is the proposed NumPy (or hybrid) alternative.  The *Adopted?* column shows the
final decision and the reason.

---

## Summary

| Function | Proposed change | Crossover N | Adopted? |
|---|---|---|---|
| `perm_entropy` | `sliding_window_view` in `_embed` | ~1 500 | No — ≤ 3 % gain in typical range |
| `svd_entropy` | `sliding_window_view` in `_embed` | ~2 000 | No — ≤ 3 % gain in typical range |
| `detrended_fluctuation` | BLAS batch regression (`d_c @ t_c`) | ~7 000 | No — 2.7× slower at N = 1 000 |
| `sample_entropy` | Pure KDTree (no Numba) | ~2 500 | No — replaced by threshold dispatch |
| `sample_entropy` | Threshold dispatch: Numba < 5 000, KDTree ≥ 5 000 | N/A | **Yes** — 2× faster at N = 10 000, parity below threshold |
| `higuchi_fd` | Batched `np.stack` / `np.diff` | > 100 000 | No — 3–20× slower across all tested N |
| `lziv_complexity` | Python loop + `.tolist()` | Never | No — 33–50× slower than Numba |

---

## `perm_entropy` — `sliding_window_view` in `_embed`

Zero-copy `sliding_window_view` replaces the `np.zeros` + copy loop in
`_embed`.  Tested across a fine-grained sweep to locate the crossover.

| N | Old / ms | New / ms | Speedup |
|---:|---:|---:|---:|
| 100 | 0.0186 | 0.0239 | 0.78× |
| 250 | 0.0244 | 0.0290 | 0.84× |
| 500 | 0.0332 | 0.0371 | 0.90× |
| 750 | 0.0430 | 0.0459 | 0.94× |
| 1 000 | 0.0521 | 0.0542 | 0.96× |
| 1 500 | 0.0724 | 0.0726 | 1.00× ← crossover |
| 2 000 | 0.0869 | 0.0852 | 1.02× |
| 3 000 | 0.1203 | 0.1168 | 1.03× |
| 5 000 | 0.1886 | 0.1823 | 1.03× |
| 10 000 | 0.3737 | 0.3473 | 1.08× |
| 20 000 | 0.7127 | 0.6899 | 1.03× |
| 50 000 | 1.8247 | 1.7310 | 1.05× |
| 100 000 | 3.8075 | 3.5002 | 1.09× |

**Decision: not adopted.**  Crossover at N ≈ 1 500.  The maximum gain is only
1.09× at N = 100 000, and the absolute difference is < 0.3 ms even there.  At
the typical user range (N = 1 000–3 000), the gain is 0–3 %, within measurement
noise.  `_embed` reverted to the loop-based copy.

---

## `svd_entropy` — `sliding_window_view` in `_embed`

Same `_embed` change as above; tested via `bench_numba_vs_numpy.py`.

| N | Old / ms | New / ms | Speedup |
|---:|---:|---:|---:|
| 100 | 0.014 | 0.020 | 0.71× |
| 500 | 0.019 | 0.024 | 0.80× |
| 1 000 | 0.030 | 0.035 | 0.85× |
| 2 000 | 0.043 | 0.046 | 0.94× |
| 10 000 | 0.136 | 0.135 | 1.01× ← crossover |
| 100 000 | 1.538 | 1.210 | 1.27× |

**Decision: not adopted.**  Crossover at N ≈ 10 000 — well above the typical
range.  `_embed` reverted; `svd_entropy` retains only the dtype-safety bugfix
(float16 / int inputs were silently passed to `np.linalg.svd` which rejects
them).

---

## `detrended_fluctuation` — BLAS batch regression

The per-window Numba loop was replaced by a single BLAS call
`slopes = (d_c @ t_c) / t_c_sq` that fits all DFA windows at once.

| N | Old / ms | New / ms | Speedup |
|---:|---:|---:|---:|
| 100 | 0.007 | 0.074 | 0.09× |
| 500 | 0.047 | 0.190 | 0.24× |
| 1 000 | 0.101 | 0.274 | 0.37× |
| 2 000 | 0.226 | 0.399 | 0.57× |
| 10 000 | 1.422 | 1.252 | 1.17× ← crossover |
| 100 000 | 21.295 | 12.852 | 1.66× |

**Decision: not adopted.**  2.7× slower at N = 1 000, 1.75× slower at N = 2 000.
Crossover is around N = 7 000–10 000.  Numba's compiled per-window loop is
cache-friendly on the small sub-arrays that DFA produces; BLAS dispatch overhead
dominates at practical signal lengths.  Reverted to the original Numba
implementation.

---

## `sample_entropy` — threshold dispatch (adopted)

Two alternatives were benchmarked against the original pure-Numba
sliding-window implementation:

**Option A — pure KDTree (no Numba, for all N):**

| N | Old Numba / ms | KDTree / ms | Speedup |
|---:|---:|---:|---:|
| 100 | 0.010 | 0.152 | 0.06× |
| 500 | 0.225 | 0.587 | 0.38× |
| 1 000 | 0.904 | 1.523 | 0.59× |
| 2 000 | 3.600 | 4.009 | 0.90× |
| 10 000 | 92.910 | 48.236 | 1.93× |

Pure KDTree is faster only above N ≈ 2 500 (Numba is O(N²), KDTree is
O(N log N)).

**Option B — threshold dispatch: Numba for N < 5 000, KDTree for N ≥ 5 000
(adopted):**

| N | Old pure-Numba / ms | Threshold dispatch / ms | Speedup |
|---:|---:|---:|---:|
| 100 | 0.009 | 0.015 | 0.64× |
| 500 | 0.244 | 0.253 | 0.97× |
| 1 000 | 0.965 | 0.994 | 0.97× |
| 2 000 | 3.865 | 3.877 | 1.00× |
| 10 000 | 96.603 | 48.370 | **2.00×** |

The small overhead at N = 100 (0.64×, +0.006 ms) is from `np.std` and
`np.asarray` in the public wrapper and is negligible in absolute terms.  The
2× gain at N = 10 000 is retained.

**Decision: threshold dispatch adopted.**  This was the original design of
the codebase, temporarily removed during the session and then restored.

---

## `higuchi_fd` — batched NumPy (`np.stack` / `np.diff`)

The Numba triple loop (k × m × j) was replaced by an outer k-loop that stacks
all m sub-series into a `(k, L+1)` matrix and uses a single `np.diff` call.

| N | Old Numba / ms | Batched NumPy / ms | Speedup |
|---:|---:|---:|---:|
| 1 000 | 0.008 | 0.145 | 0.05× |
| 3 000 | 0.026 | 0.156 | 0.17× |
| 10 000 | 0.090 | 0.253 | 0.36× |
| 100 000 | 0.939 | 1.287 | 0.73× |

**Decision: not adopted.**  The NumPy version is 3–20× slower across all
tested sizes.  Numba's compiled innermost loop is efficient at the short
sub-series lengths that `higuchi_fd` produces (typically a few hundred samples
per sub-series even at large N).  Reverted to the original Numba implementation.

---

## `lziv_complexity` — Python loop + `.tolist()`

The Numba JIT was removed and replaced with a pure Python loop.  A `.tolist()`
call was added to convert the `uint32` array to a native Python list before the
loop, avoiding NumPy scalar boxing overhead (~5 % improvement over plain Python).

| N | Old Numba / ms | Python + `.tolist()` / ms | Speedup |
|---:|---:|---:|---:|
| 100 | 0.002 | 0.139 | 0.01× |
| 1 000 | 0.233 | 9.468 | 0.02× |
| 3 000 | 2.401 | 127.2 | 0.02× |
| 10 000 | 24.689 | 809.8 | 0.03× |

**Decision: not adopted.**  The LZ76 algorithm is an inherently sequential
O(N² / log N) scan with no vectorisable structure.  Python is 33–50× slower
than Numba's compiled loop at every tested size.  Reverted to the original
Numba JIT.
