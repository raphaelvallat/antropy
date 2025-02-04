.. -*- mode: rst -*-

|

.. image:: https://img.shields.io/github/license/raphaelvallat/antropy.svg
  :target: https://github.com/raphaelvallat/antropy/blob/master/LICENSE

.. image:: https://github.com/raphaelvallat/antropy/actions/workflows/python_tests.yml/badge.svg
  :target: https://github.com/raphaelvallat/antropy/actions/workflows/python_tests.yml

.. image:: https://codecov.io/gh/raphaelvallat/antropy/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/raphaelvallat/antropy

----------------

.. figure::  https://github.com/raphaelvallat/antropy/blob/master/docs/pictures/logo.png?raw=true
   :align:   center

AntroPy is a Python 3 package providing several time-efficient algorithms for computing the complexity of time-series.
It can be used for example to extract features from EEG signals.

Documentation
=============

- `Link to documentation <https://raphaelvallat.com/antropy/build/html/index.html>`_

Installation
============

AntroPy can be installed with pip

.. code-block:: shell

  pip install antropy

or conda

.. code-block:: shell

  conda config --add channels conda-forge
  conda config --set channel_priority strict
  conda install antropy

To build and install from source, clone this repository or download the source archive and decompress the files

.. code-block:: shell

  cd antropy
  pip install ".[test]"     # install the package
  pip install -e ".[test]"  # or editable install
  pytest

**Dependencies**

- `numpy <https://numpy.org/>`_
- `scipy <https://www.scipy.org/>`_
- `scikit-learn <https://scikit-learn.org/>`_
- `numba <http://numba.pydata.org/>`_

Functions
=========

**Entropy**

.. code-block:: python

    import numpy as np
    import antropy as ant
    np.random.seed(1234567)
    x = np.random.normal(size=3000)
    # Permutation entropy
    print(ant.perm_entropy(x, normalize=True))
    # Spectral entropy
    print(ant.spectral_entropy(x, sf=100, method='welch', normalize=True))
    # Singular value decomposition entropy
    print(ant.svd_entropy(x, normalize=True))
    # Approximate entropy
    print(ant.app_entropy(x))
    # Sample entropy
    print(ant.sample_entropy(x))
    # Hjorth mobility and complexity
    print(ant.hjorth_params(x))
    # Number of zero-crossings
    print(ant.num_zerocross(x))
    # Lempel-Ziv complexity
    print(ant.lziv_complexity('01111000011001', normalize=True))

.. parsed-literal::

    0.9995371694290871
    0.9940882825422431
    0.9999110978316078
    2.015221318528564
    2.198595813245399
    (1.4313385010057378, 1.215335712274099)
    1531
    1.3597696150205727

**Fractal dimension**

.. code-block:: python

    # Petrosian fractal dimension
    print(ant.petrosian_fd(x))
    # Katz fractal dimension
    print(ant.katz_fd(x))
    # Higuchi fractal dimension
    print(ant.higuchi_fd(x))
    # Detrended fluctuation analysis
    print(ant.detrended_fluctuation(x))

.. parsed-literal::

    1.0310643385753608
    5.954272156665926
    2.005040632258251
    0.47903505674073327

Execution time
~~~~~~~~~~~~~~

Here are some benchmarks computed on a MacBook Pro (2020).

.. code-block:: python

    import numpy as np
    import antropy as ant
    np.random.seed(1234567)
    x = np.random.rand(1000)
    # Entropy
    %timeit ant.perm_entropy(x)
    %timeit ant.spectral_entropy(x, sf=100)
    %timeit ant.svd_entropy(x)
    %timeit ant.app_entropy(x)  # Slow
    %timeit ant.sample_entropy(x)  # Numba
    # Fractal dimension
    %timeit ant.petrosian_fd(x)
    %timeit ant.katz_fd(x)
    %timeit ant.higuchi_fd(x) # Numba
    %timeit ant.detrended_fluctuation(x) # Numba

.. parsed-literal::

    106 µs ± 5.49 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    138 µs ± 3.53 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    40.7 µs ± 303 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    2.44 ms ± 134 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    2.21 ms ± 35.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    23.5 µs ± 695 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    40.1 µs ± 2.09 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    13.7 µs ± 251 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    315 µs ± 10.7 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

Development
===========

AntroPy was created and is maintained by `Raphael Vallat <https://raphaelvallat.com>`_. Contributions are more than welcome so feel free to contact me, open an issue or submit a pull request!

To see the code or report a bug, please visit the `GitHub repository <https://github.com/raphaelvallat/antropy>`_.

Note that this program is provided with **NO WARRANTY OF ANY KIND**. Always double check the results.

Acknowledgement
===============

Several functions of AntroPy were adapted from:

- MNE-features: https://github.com/mne-tools/mne-features
- pyEntropy: https://github.com/nikdon/pyEntropy
- pyrem: https://github.com/gilestrolab/pyrem
- nolds: https://github.com/CSchoel/nolds

All the credit goes to the author of these excellent packages.