.. -*- mode: rst -*-

|

.. image:: https://img.shields.io/badge/python-3.6%20%7C%203.7-blue.svg
    :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/github/license/raphaelvallat/entropy.svg
  :target: https://github.com/raphaelvallat/entropy/blob/master/LICENSE

.. image:: https://travis-ci.org/raphaelvallat/entropy.svg?branch=master
    :target: https://travis-ci.org/raphaelvallat/entropy

.. image:: https://ci.appveyor.com/api/projects/status/mukj36n939ftu4io?svg=true
    :target: https://ci.appveyor.com/project/raphaelvallat/entropy

.. image:: https://codecov.io/gh/raphaelvallat/entropy/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/raphaelvallat/entropy

----------------

.. figure::  https://raw.githubusercontent.com/raphaelvallat/entropy/master/docs/pictures/logo.png
   :align:   center

EntroPy is a Python 3 package providing several time-efficient algorithms for computing the complexity of one-dimensional time-series.
It can be used for example to extract features from EEG signals.

Installation
============

.. important::
  Please note that EntroPy **cannot** be installed using pip or conda.
  There is already a package called *entropy* on the `pip repository <https://pypi.org/project/entropy/>`_,
  which should not be mistaken with the current package.

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

Functions
=========

**Entropy**

.. code-block:: python

    from entropy import *
    import numpy as np
    np.random.seed(1234567)
    x = np.random.rand(3000)
    print(perm_entropy(x, order=3, normalize=True))                 # Permutation entropy
    print(spectral_entropy(x, 100, method='welch', normalize=True)) # Spectral entropy
    print(svd_entropy(x, order=3, delay=1, normalize=True))         # Singular value decomposition entropy
    print(app_entropy(x, order=2, metric='chebyshev'))              # Approximate entropy
    print(sample_entropy(x, order=2, metric='chebyshev'))           # Sample entropy
    print(lziv_complexity('01111000011001', normalize=True))        # Lempel-Ziv complexity

.. parsed-literal::

    0.9995858289645746
    0.9945519071575192
    0.8482185855709181
    2.0754913760787277
    2.192416747827227
    0.9425204748625924

**Fractal dimension**

.. code-block:: python

    print(petrosian_fd(x))            # Petrosian fractal dimension
    print(katz_fd(x))                 # Katz fractal dimension
    print(higuchi_fd(x, kmax=10))     # Higuchi fractal dimension
    print(detrended_fluctuation(x))   # Detrended fluctuation analysis

.. parsed-literal::

    1.0303256054255618
    9.496389529050981
    1.9914197968462963
    0.5082304865081877

Execution time
~~~~~~~~~~~~~~

Here are some benchmarks computed on an average PC (i7-7700HQ CPU @ 2.80 Ghz - 8 Go of RAM).

.. code-block:: python

    from entropy import *
    import numpy as np
    np.random.seed(1234567)
    x = np.random.rand(1000)
    # Entropy
    %timeit perm_entropy(x, order=3, delay=1)
    %timeit spectral_entropy(x, 100, method='fft')
    %timeit svd_entropy(x, order=3, delay=1)
    %timeit app_entropy(x, order=2) # Slow
    %timeit sample_entropy(x, order=2) # Slow
    # Fractal dimension
    %timeit petrosian_fd(x)
    %timeit katz_fd(x)
    %timeit higuchi_fd(x) # Numba (fast)
    %timeit detrended_fluctuation(x) # Numba (fast)

.. parsed-literal::

    127 µs ± 3.86 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    150 µs ± 859 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    42.4 µs ± 306 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    4.59 ms ± 62.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    2.03 ms ± 39.5 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    16.4 µs ± 251 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    32.4 µs ± 578 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    17.4 µs ± 274 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    755 µs ± 17.1 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

Development
===========

EntroPy was created and is maintained by `Raphael Vallat <https://raphaelvallat.com>`_. Contributions are more than welcome so feel free to contact me, open an issue or submit a pull request!

To see the code or report a bug, please visit the `GitHub repository <https://github.com/raphaelvallat/entropy>`_.

Note that this program is provided with **NO WARRANTY OF ANY KIND**. If you can, always double check the results.

Acknowledgement
===============

Several functions of EntroPy were adapted from:

- MNE-features: https://github.com/mne-tools/mne-features
- pyEntropy: https://github.com/nikdon/pyEntropy
- pyrem: https://github.com/gilestrolab/pyrem
- nolds: https://github.com/CSchoel/nolds

All the credit goes to the author of these excellent packages.
