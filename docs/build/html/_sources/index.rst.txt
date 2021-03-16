.. -*- mode: rst -*-

|

.. image:: https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue.svg
    :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/github/license/raphaelvallat/entropy.svg
  :target: https://github.com/raphaelvallat/entropy/blob/master/LICENSE

.. image:: https://travis-ci.org/raphaelvallat/entropy.svg?branch=master
    :target: https://travis-ci.org/raphaelvallat/entropy

.. image:: https://codecov.io/gh/raphaelvallat/entropy/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/raphaelvallat/entropy

----------------

.. figure::  https://raw.githubusercontent.com/raphaelvallat/entropy/master/docs/pictures/logo.png
   :align:   center

EntroPy is a Python 3 package providing several time-efficient algorithms for computing the complexity of time-series.
It can be used for example to extract features from EEG signals.

Installation
============

.. important::
  EntroPy **CANNOT BE INSTALLED WITH PIP OR CONDA**.
  There is already a package called *entropy* on the `PyPi repository <https://pypi.org/project/entropy/>`_,
  which should NOT be mistaken with the current package.

.. code-block:: shell

  git clone https://github.com/raphaelvallat/entropy.git entropy/
  cd entropy/
  pip install -r requirements.txt
  python setup.py develop

**Dependencies**

- `numpy <https://numpy.org/>`_
- `scipy <https://www.scipy.org/>`_
- `scikit-learn <https://scikit-learn.org/>`_
- `numba <http://numba.pydata.org/>`_
- `stochastic <https://github.com/crflynn/stochastic>`_

Functions
=========

**Entropy**

.. code-block:: python

    import numpy as np
    import entropy as ent
    np.random.seed(1234567)
    x = np.random.normal(size=3000)
    # Permutation entropy
    print(ent.perm_entropy(x, normalize=True))
    # Spectral entropy
    print(ent.spectral_entropy(x, sf=100, method='welch', normalize=True))
    # Singular value decomposition entropy
    print(ent.svd_entropy(x, normalize=True))
    # Approximate entropy
    print(ent.app_entropy(x))
    # Sample entropy
    print(ent.sample_entropy(x))
    # Hjorth mobility and complexity
    print(ent.hjorth_params(x))
    # Number of zero-crossings
    print(ent.num_zerocross(x))
    # Lempel-Ziv complexity
    print(ent.lziv_complexity('01111000011001', normalize=True))

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
    print(ent.petrosian_fd(x))
    # Katz fractal dimension
    print(ent.katz_fd(x))
    # Higuchi fractal dimension
    print(ent.higuchi_fd(x))
    # Detrended fluctuation analysis
    print(ent.detrended_fluctuation(x))

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
    import entropy as ent
    np.random.seed(1234567)
    x = np.random.rand(1000)
    # Entropy
    %timeit ent.perm_entropy(x)
    %timeit ent.spectral_entropy(x, sf=100)
    %timeit ent.svd_entropy(x)
    %timeit ent.app_entropy(x)  # Slow
    %timeit ent.sample_entropy(x)  # Numba
    # Fractal dimension
    %timeit ent.petrosian_fd(x)
    %timeit ent.katz_fd(x)
    %timeit ent.higuchi_fd(x) # Numba
    %timeit ent.detrended_fluctuation(x) # Numba

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

EntroPy was created and is maintained by `Raphael Vallat <https://raphaelvallat.com>`_. Contributions are more than welcome so feel free to contact me, open an issue or submit a pull request!

To see the code or report a bug, please visit the `GitHub repository <https://github.com/raphaelvallat/entropy>`_.

Note that this program is provided with **NO WARRANTY OF ANY KIND**. Always double check the results.

Acknowledgement
===============

Several functions of EntroPy were adapted from:

- MNE-features: https://github.com/mne-tools/mne-features
- pyEntropy: https://github.com/nikdon/pyEntropy
- pyrem: https://github.com/gilestrolab/pyrem
- nolds: https://github.com/CSchoel/nolds

All the credit goes to the author of these excellent packages.
