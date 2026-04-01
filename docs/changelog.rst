.. _Changelog:

What's new
##########

v0.2.2 (April 2026)
--------------------

**Performance**

- :py:func:`antropy.perm_entropy` gains a fast path for ``order`` ∈ {3, 4}: ordinal
  patterns are encoded via pairwise comparison bit-keys and counted with
  ``np.bincount`` rather than ``argsort`` + ``np.unique``, giving a major speed-up
  compared to the previous implementation.
- The general path (``order`` > 4) is also improved: the embedded matrix is now
  built as a zero-copy ``as_strided`` view and the hash is computed via a single
  BLAS ``@`` product instead of an element-wise multiply + sum.

**New features**

- :py:func:`antropy.perm_entropy` now accepts 2-D arrays of shape
  ``(n_channels, n_times)`` for ``order`` ∈ {3, 4}, computing entropy for every
  row in a single vectorised call.

**Bug fixes**

- :py:func:`antropy.perm_entropy` now always applies a small positional epsilon
  jitter before sorting, making results fully deterministic across platforms and
  dtypes. Previously, integer-typed signals (and some float signals with exact
  ties) could produce different outputs depending on the platform's unstable-sort
  tie-breaking behaviour.
- Normalized :py:func:`antropy.perm_entropy` output is now clipped to ``[0, 1]``
  to prevent returning ``-0.0`` for perfectly regular signals (IEEE 754 artefact
  from ``-(1.0 * log2(1.0))``).

**Docs**

- Fixed doctest failures in :py:func:`antropy.spectral_entropy`,
  :py:func:`antropy.num_zerocross`, and :py:func:`antropy.hjorth_params` caused by
  NumPy's new scalar ``repr`` (``np.float64(x)`` / ``np.int64(x)`` instead of
  bare literals in newer NumPy).
- Added a Big O complexity column to the performance tables in ``README.rst`` and
  ``docs/index.rst``.
- Performance tables now show timings at both 1000 and 10000 samples.
- Added ``benchmarks/benchmark_all.py`` script to reproduce all table timings.
- Rewrote ``docs/index.rst`` to match ``README.rst`` in content and style.

**Maintenance**

- Clean up ``docs/conf.py`` (remove redundant ``sys.path`` manipulation) and
  tighten ``pyproject.toml`` (remove stale extras, fix editable-install metadata).

v0.2.1 (March 2026)
--------------------

**Bug fixes**

- Fix off-by-one in :py:func:`antropy.higuchi_fd`: the inner loop summed
  :math:`N_m - 1` differences per sub-series instead of the correct :math:`N_m`,
  causing a systematic underestimate of curve length. Verified against the
  NeuroKit2 reference implementation.
- :py:func:`antropy.sample_entropy` now returns ``np.nan`` (instead of ``0``)
  when no template of length *m* matches within the tolerance, since the entropy
  is mathematically undefined in that case.
- Fix silent ``epsilon = 10e-9`` (= 1e-8) typo in ``utils.py``; corrected to
  ``1e-9``. This constant guards against division-by-zero in the Numba linear
  regression used by :py:func:`antropy.detrended_fluctuation`.
- Fix ``all = [...]`` → ``__all__ = [...]`` in all three source modules, which
  previously leaked imported names (``np``, ``jit``, ``KDTree``, …) into the
  public namespace on ``from antropy import *``.

**New features**

- :py:func:`antropy.hjorth_params` gains an optional ``sf`` parameter. When
  provided, mobility is returned in Hz instead of samples⁻¹ (multiplied by
  ``sf``); complexity is unaffected.

**Validation**

- User-facing ``assert`` statements replaced with proper ``ValueError`` /
  ``TypeError`` raises throughout (asserts are silently stripped by
  ``python -O``): ``perm_entropy(delay=0)``, ``spectral_entropy(method=…)``,
  ``app_entropy``/``sample_entropy(tolerance=…)``, and
  ``lziv_complexity(sequence=…, normalize=…)``.

**Docs**

- Docstring formula for :py:func:`antropy.higuchi_fd` corrected to match the
  fix above (:math:`\sum_{j=1}^{N_m}` not :math:`\sum_{j=1}^{N_m-1}`).
- Added notes to :py:func:`antropy.lziv_complexity` (float arrays are truncated
  to ``uint32``, not discretised — binarise continuous signals first),
  :py:func:`antropy.num_zerocross` (``signbit(0) == False`` behaviour),
  :py:func:`antropy.spectral_entropy` (DC component included),
  :py:func:`antropy.detrended_fluctuation` (:math:`\alpha \approx 1` is
  ambiguous), and :py:func:`antropy.hjorth_params` (mobility units).

v0.2.0 (March 2026)
--------------------

**Build & CI**

- Drop Python 3.9 (EOL), add Python 3.13 support. Minimum is now Python 3.10.
- Switch from ``pip`` to ``uv`` in all GitHub Actions workflows.
- Add explicit minimum versions for core dependencies: ``numpy>=1.22.4``, ``scipy>=1.8.0``, ``scikit-learn>=1.2.0``.
- Migrate ``[project.optional-dependencies]`` to PEP 735 ``[dependency-groups]``.
- Bump ``setuptools>=80.0``.
- Split CI into three jobs: ``test-core`` (3 platforms × 4 Python versions), ``test-dependency-combinations`` (4 dep combos from minimum to latest), and ``coverage``.
- Fix ``test-dependency-combinations`` job: separate antropy install (``--no-deps``) from test-dependency install so that pytest's own dependencies (e.g. ``pluggy``) are always resolved.
- Fix Codecov upload to use ``${{ secrets.CODECOV_TOKEN }}`` instead of a hardcoded token.
- Switch Ruff workflow from ``astral-sh/ruff-action@v1`` to ``uvx ruff`` via ``astral-sh/setup-uv@v7``.
- Extend Ruff rules: add ``W`` (pycodestyle warnings) and ``NPY`` (NumPy rules).

**Tests**

- Increase test coverage from ~54 % to 100 %.
- Add ``tests/test_utils.py`` covering all branches of the ``_embed`` helper (1-D and 2-D paths, all error conditions).
- Add edge-case tests: ``sample_entropy`` returning ``inf`` (m-length matches exist but no (m+1)-length matches); ``detrended_fluctuation`` returning ``NaN`` for a constant signal; ``spectral_entropy`` raising on an invalid method string.
- Set ``NUMBA_DISABLE_JIT=1`` in the ``coverage`` CI job so coverage.py can instrument Numba JIT function bodies; other CI jobs still exercise real compiled code.

**Bug fixes**

- Fix :py:func:`antropy.higuchi_fd` returning a ``ValueError`` (``math domain error``) on constant or integer-typed input arrays: ``log(0)`` is now guarded to return ``-inf``, matching Numba's IEEE 754 behaviour.

**Docs**

- Switch documentation theme from ``sphinx_bootstrap_theme`` to ``pydata-sphinx-theme`` (dark/light toggle, GitHub icon, improved layout).
- Fix three broken ``intersphinx`` URLs: NumPy, SciPy, and MNE-Python.
- Add ``sphinx.ext.mathjax`` for LaTeX math rendering in docstrings.
- Add ``contributing.rst`` guide.
- Fix stale ``:py:func:`` cross-references in changelog entries v0.1.1–v0.1.3 (``entropy.XXX`` → ``antropy.XXX``).
- Fix typo in v0.1.6 changelog ("Fox for KDTree" → "Fix for KDTree").
- Modernize ``README.rst`` and ``docs/index.rst``: add PyPI, conda-forge, downloads, and Ruff badges; add ``uv`` installation instructions; fix broken links.

v0.1.9 (February 2025)
----------------------

- Remove `stochastic` package from dependency to enable support for Numpy 2.x

v0.1.8 (December 2024)
----------------------

- Switch to modern python packaging
- Use ruff instead of black/flake8

v0.1.7 (December 2024)
----------------------

- Simplify Katz FD implementation. https://github.com/raphaelvallat/antropy/pull/30
- Add tolerance parameter to the approximate and sample entropy. https://github.com/raphaelvallat/antropy/pull/32
- Fix for scikit-learn ≥ 1.3 in approximate and sample entropy. https://github.com/raphaelvallat/antropy/pull/36

v0.1.6 (July 2023)
------------------

This version requires numba >= 0.57.

a. Allow readonly arrays in numba jit signature. https://github.com/raphaelvallat/antropy/pull/23
b. Improved sample entropy kernel. https://github.com/raphaelvallat/antropy/pull/25
c. Fix for KDTree.valid_metrics which is method since sklearn 1.3. https://github.com/raphaelvallat/antropy/pull/30

v0.1.5 (December 2022)
----------------------

a. :py:func:`antropy.perm_entropy` will now return the average entropy across all delays if a list or range of delays is provided.
b. Handle the limit of p = 0 in functions that evaluate the product p * log2(p), to give 0 instead of nan (see `PR3 <https://github.com/raphaelvallat/antropy/pull/3>`_).
c. :py:func:`antropy.detrended_fluctuation` will now return alpha = 0 when the correlation coefficient of the fluctuations of an input signal is 0 (see `PR21 <https://github.com/raphaelvallat/antropy/pull/21>`_).

v0.1.4 (April 2021)
-------------------

.. important:: The package has now been renamed AntroPy (previously EntroPy)!

a. Faster implementation of :py:func:`antropy.lziv_complexity` (see `PR1 <https://github.com/raphaelvallat/entropy/pull/1>`_). Among other improvements, strings are now mapped to UTF-8 integer representations.

v0.1.3 (March 2021)
-------------------

a. Added the :py:func:`antropy.num_zerocross` function to calculate the (normalized) number of zero-crossings on N-D data.
b. Added the :py:func:`antropy.hjorth_params` function to calculate the mobility and complexity Hjorth parameters on N-D data.
c. Add support for N-D data in :py:func:`antropy.spectral_entropy`, :py:func:`antropy.petrosian_fd` and :py:func:`antropy.katz_fd`.
d. Use the `stochastic <https://github.com/crflynn/stochastic>`_ package to generate stochastic time-series.

v0.1.2 (May 2020)
-----------------

a. :py:func:`antropy.lziv_complexity` now works with non-binary sequence (e.g. "12345" or "Hello World!")
b. The average fluctuations in :py:func:`antropy.detrended_fluctuation` is now calculated using the root mean square instead of a simple arithmetic. For more details, please refer to `this GitHub issue <https://github.com/neuropsychology/NeuroKit/issues/206>`_.
c. Updated flake8

v0.1.1 (November 2019)
----------------------

a. Added Lempel-Ziv complexity (:py:func:`antropy.lziv_complexity`) for binary sequence.

v0.1.0 (October 2018)
---------------------

Initial release.

a. Permutation entropy
b. Spectral entropy
c. Singular value decomposition entropy
d. Approximate entropy
e. Sample entropy
f. Petrosian Fractal Dimension
g. Katz Fractal Dimension
h. Higuchi Fractal Dimension
i. Detrended fluctuation analysis
