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
time-series. It is designed for speed (Numba JIT compilation for the most expensive functions)
and ease of use, and works on both 1-D and N-D arrays.

Typical use cases include feature extraction from physiological signals (EEG, ECG, EMG),
neuroscience, and signal processing research.

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
       Supports multiple delays and N-D arrays.
   * - ``ant.spectral_entropy``
     - Spectral (power-spectrum) entropy via FFT or Welch method.
       Supports N-D arrays.
   * - ``ant.svd_entropy``
     - Singular value decomposition entropy of the time-delay embedding matrix.
   * - ``ant.app_entropy``
     - Approximate entropy (ApEn) — regularity measure sensitive to the
       length of the signal.
   * - ``ant.sample_entropy``
     - Sample entropy (SampEn) — less biased alternative to ApEn.
       Numba-accelerated for short series (< 5000 samples).
   * - ``ant.lziv_complexity``
     - Lempel-Ziv complexity for symbolic / binary sequences.
       Works with strings, lists, and arrays.
   * - ``ant.num_zerocross``
     - Number of zero-crossings. Supports N-D arrays.
   * - ``ant.hjorth_params``
     - Hjorth mobility and complexity parameters.
       Optional ``sf`` argument converts mobility to Hz.
       Supports N-D arrays.

Fractal dimension
-----------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Function
     - Description
   * - ``ant.petrosian_fd``
     - Petrosian fractal dimension — fast estimate based on zero-crossings
       of the derivative. Supports N-D arrays.
   * - ``ant.katz_fd``
     - Katz fractal dimension. Supports N-D arrays.
   * - ``ant.higuchi_fd``
     - Higuchi fractal dimension — slope of log curve-length vs log interval,
       Numba-accelerated.
   * - ``ant.detrended_fluctuation``
     - Detrended fluctuation analysis (DFA) — estimates the Hurst / scaling
       exponent, Numba-accelerated.

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

    0.9995371694290869       # perm_entropy        (0 = regular, 1 = random)
    0.9940882825422431       # spectral_entropy     (0 = pure tone, 1 = white noise)
    0.9999110978316078       # svd_entropy
    2.015221318528564        # app_entropy
    2.198595813245399        # sample_entropy
    (1.4313385010057378, 1.215335712274099)   # hjorth (mobility, complexity)
    (143.13385010057377, 1.215335712274099)   # hjorth with sf=100 Hz
    1531                     # num_zerocross
    1.3597696150205727       # lziv_complexity (normalized)

Fractal dimension
-----------------

.. code-block:: python

    print(ant.petrosian_fd(x))
    print(ant.katz_fd(x))
    print(ant.higuchi_fd(x))
    print(ant.detrended_fluctuation(x))

.. parsed-literal::

    1.0310643385753608       # petrosian_fd
    5.9542721566659225       # katz_fd
    2.0036527058413816       # higuchi_fd     (≈ 2 for white noise)
    0.47903505674015406      # DFA alpha      (≈ 0.5 for white noise)

N-D arrays
----------

Most functions accept N-D arrays and an ``axis`` argument, making it easy to process
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

Benchmarks on a 1000-sample signal (MacBook Pro, 2020):

.. list-table::
   :widths: 45 30 25
   :header-rows: 1

   * - Function
     - Time
     - Backend
   * - ``ant.perm_entropy``
     - 106 µs
     - NumPy
   * - ``ant.spectral_entropy``
     - 138 µs
     - NumPy / SciPy
   * - ``ant.svd_entropy``
     - 40.7 µs
     - NumPy
   * - ``ant.app_entropy``
     - 2.44 ms
     - NumPy (slow for long series)
   * - ``ant.sample_entropy``
     - 2.21 ms
     - **Numba** (JIT)
   * - ``ant.petrosian_fd``
     - 23.5 µs
     - NumPy
   * - ``ant.katz_fd``
     - 40.1 µs
     - NumPy
   * - ``ant.higuchi_fd``
     - 13.7 µs
     - **Numba** (JIT)
   * - ``ant.detrended_fluctuation``
     - 315 µs
     - **Numba** (JIT)

Numba functions incur a one-time compilation cost on the first call.

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
