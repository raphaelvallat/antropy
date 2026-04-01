.. -*- mode: rst -*-

|

.. image:: https://badge.fury.io/py/antropy.svg
  :target: https://badge.fury.io/py/antropy

.. image:: https://img.shields.io/conda/vn/conda-forge/antropy.svg
  :target: https://anaconda.org/conda-forge/antropy

.. image:: https://img.shields.io/github/license/raphaelvallat/antropy.svg
  :target: https://github.com/raphaelvallat/antropy/blob/master/LICENSE

.. image:: https://github.com/raphaelvallat/antropy/actions/workflows/python_tests.yml/badge.svg
  :target: https://github.com/raphaelvallat/antropy/actions/workflows/python_tests.yml

.. image:: https://codecov.io/gh/raphaelvallat/antropy/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/raphaelvallat/antropy

.. image:: https://static.pepy.tech/badge/antropy
  :target: https://pepy.tech/projects/antropy

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
  :target: https://github.com/astral-sh/ruff
  :alt: Ruff

----------------

.. figure:: https://raw.githubusercontent.com/raphaelvallat/antropy/master/docs/pictures/logo.png
   :align: center

**AntroPy** is a Python package for computing entropy and fractal dimension measures of
time-series. It is designed for speed (Numba JIT compilation) and ease of use, and works on
both 1-D and N-D arrays. Typical use cases include feature extraction from physiological signals
(e.g. EEG, ECG, EMG), and signal processing research.

- `Documentation <https://raphaelvallat.com/antropy/>`_
- `Changelog <https://raphaelvallat.com/antropy/changelog.html>`_
- `GitHub <https://github.com/raphaelvallat/antropy>`_

----------------

Functions
=========

Entropy
-------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Function
     - Description
   * - ``ant.perm_entropy``
     - Permutation entropy — captures ordinal patterns in the signal.
   * - ``ant.spectral_entropy``
     - Spectral (power-spectrum) entropy via FFT or Welch method.
   * - ``ant.svd_entropy``
     - Singular value decomposition entropy of the time-delay embedding matrix.
   * - ``ant.app_entropy``
     - Approximate entropy (ApEn) — regularity measure sensitive to the length of the signal.
   * - ``ant.sample_entropy``
     - Sample entropy (SampEn) — less biased alternative to ApEn.
   * - ``ant.lziv_complexity``
     - Lempel-Ziv complexity for symbolic / binary sequences.
   * - ``ant.num_zerocross``
     - Number of zero-crossings.
   * - ``ant.hjorth_params``
     - Hjorth mobility and complexity parameters.

Fractal dimension
-----------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Function
     - Description
   * - ``ant.petrosian_fd``
     - Petrosian fractal dimension.
   * - ``ant.katz_fd``
     - Katz fractal dimension.
   * - ``ant.higuchi_fd``
     - Higuchi fractal dimension — slope of log curve-length vs log interval.
   * - ``ant.detrended_fluctuation``
     - Detrended fluctuation analysis (DFA) — estimates the Hurst / scaling exponent.

----------------

Installation
============

AntroPy requires Python 3.10+ and depends on NumPy (≥ 1.22.4), SciPy (≥ 1.8.0),
scikit-learn (≥ 1.2.0), and Numba (≥ 0.57).

.. code-block:: shell

    # pip
    pip install antropy

    # uv
    uv pip install antropy

    # conda
    conda install -c conda-forge antropy

Development installation
------------------------

.. code-block:: shell

    git clone https://github.com/raphaelvallat/antropy.git
    cd antropy
    uv pip install --group=test --editable .
    pytest --verbose

----------------

Quick start
===========

Entropy measures
----------------

.. code-block:: python

    import numpy as np
    import antropy as ant

    np.random.seed(1234567)
    x = np.random.normal(size=3000)

    print(ant.perm_entropy(x, normalize=True))
    print(ant.spectral_entropy(x, sf=100, method='welch', normalize=True))
    print(ant.svd_entropy(x, normalize=True))
    print(ant.app_entropy(x))
    print(ant.sample_entropy(x))
    print(ant.hjorth_params(x))             # mobility in samples⁻¹
    print(ant.hjorth_params(x, sf=100))     # mobility in Hz
    print(ant.num_zerocross(x))
    print(ant.lziv_complexity('01111000011001', normalize=True))

.. parsed-literal::

    0.9995              # perm_entropy        (0 = regular, 1 = random)
    0.9941              # spectral_entropy     (0 = pure tone, 1 = white noise)
    0.9999              # svd_entropy
    2.0152              # app_entropy
    2.1986              # sample_entropy
    (1.4313, 1.2153)    # hjorth (mobility, complexity)
    (143.1339, 1.2153)  # hjorth with sf=100 Hz
    1531                # num_zerocross
    1.3598              # lziv_complexity (normalized)

Fractal dimension
-----------------

.. code-block:: python

    print(ant.petrosian_fd(x))
    print(ant.katz_fd(x))
    print(ant.higuchi_fd(x))
    print(ant.detrended_fluctuation(x))

.. parsed-literal::

    1.0311    # petrosian_fd
    5.9543    # katz_fd
    2.0037    # higuchi_fd   (≈ 2 for white noise)
    0.4790    # DFA alpha    (≈ 0.5 for white noise)

N-D arrays
----------

Some functions accept N-D arrays and an ``axis`` argument, making it easy to process
multi-channel data in a single call:

.. code-block:: python

    import numpy as np
    import antropy as ant

    # 4 channels × 3000 samples
    X = np.random.normal(size=(4, 3000))

    pe   = ant.perm_entropy(X, normalize=True, axis=-1)          # shape (4,)
    mob, com = ant.hjorth_params(X, sf=256, axis=-1)             # shape (4,) each
    nzc  = ant.num_zerocross(X, normalize=True, axis=-1)         # shape (4,)
    se   = ant.spectral_entropy(X, sf=256, normalize=True)       # shape (4,)

----------------

Performance
===========

Benchmarks on a MacBook Pro M1 Max (2021):

.. list-table::
   :widths: 32 20 20 28
   :header-rows: 1

   * - Function
     - 1 000 samples
     - 10 000 samples
     - Complexity
   * - ``ant.perm_entropy``
     - 24 µs
     - 87 µs
     - O(n) ¹
   * - ``ant.spectral_entropy``
     - 141 µs
     - 863 µs
     - O(n log n) ⁴
   * - ``ant.svd_entropy``
     - 35 µs
     - 140 µs
     - O(n·m²) ²
   * - ``ant.app_entropy``
     - 1.5 ms
     - 45.9 ms
     - O(n²) worst ⁵
   * - ``ant.sample_entropy``
     - 917 µs
     - 46.0 ms
     - O(n²) worst ⁵
   * - ``ant.lziv_complexity``
     - 241 µs
     - 25.2 ms
     - O(n²/log n)
   * - ``ant.num_zerocross``
     - 2.5 µs
     - 6 µs
     - O(n)
   * - ``ant.hjorth_params``
     - 19 µs
     - 44 µs
     - O(n)
   * - ``ant.petrosian_fd``
     - 6 µs
     - 14 µs
     - O(n)
   * - ``ant.katz_fd``
     - 9 µs
     - 22 µs
     - O(n)
   * - ``ant.higuchi_fd``
     - 7 µs
     - 92 µs
     - O(n·kmax) ³
   * - ``ant.detrended_fluctuation``
     - 99 µs
     - 1.4 ms
     - O(n log n)

¹ ``perm_entropy``: O(n) for ``order`` ∈ {3, 4} (default), O(n·m·log m) for ``order`` > 4.
² ``svd_entropy``: m = ``order`` (default 3).
³ ``higuchi_fd``: ``kmax`` = max interval (default 10).
⁴ ``spectral_entropy``: O(n log n) for FFT method, O(n) for Welch with fixed ``nperseg`` (default).
⁵ ``app_entropy`` / ``sample_entropy``: O(n²) worst case, empirically ~O(n^1.5) via KDTree average case.

Numba functions (``sample_entropy``, ``higuchi_fd``, ``detrended_fluctuation``) incur a one-time compilation cost on the first call.

----------------

Contributing
============

AntroPy was created and is maintained by `Raphael Vallat <https://raphaelvallat.com>`_.
Contributions are welcome — feel free to open an issue or submit a pull request on
`GitHub <https://github.com/raphaelvallat/antropy>`_.

**Note:** this program is provided with **NO WARRANTY OF ANY KIND**. Always validate
results against known references.

----------------

Acknowledgements
================

Several functions in AntroPy were adapted from:

- `MNE-features <https://github.com/mne-tools/mne-features>`_ — Jean-Baptiste Schiratti & Alexandre Gramfort
- `pyEntropy <https://github.com/nikdon/pyEntropy>`_ — Nikolay Donets
- `pyrem <https://github.com/gilestrolab/pyrem>`_ — Quentin Geissmann
- `nolds <https://github.com/CSchoel/nolds>`_ — Christopher Scholzel
