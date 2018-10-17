.. -*- mode: rst -*-

|

.. image:: https://img.shields.io/github/license/raphaelvallat/entropy.svg
  :target: https://github.com/raphaelvallat/entropy/blob/master/LICENSE

.. image:: https://travis-ci.org/raphaelvallat/entropy.svg?branch=master
    :target: https://travis-ci.org/raphaelvallat/entropy

.. image:: https://ci.appveyor.com/api/projects/status/mukj36n939ftu4io?svg=true
    :target: https://ci.appveyor.com/project/raphaelvallat/entropy

.. image:: https://codecov.io/gh/raphaelvallat/entropy/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/raphaelvallat/entropy

----------------

.. figure::  https://github.com/raphaelvallat/entropy/blob/master/docs/pictures/logo.png
   :align:   center

EntroPy is a Python 3 package for computing several complexity metrics of one-dimensional time series.

Documentation
=============

- `Link to documentation <https://raphaelvallat.github.io/entropy/build/html/index.html>`_

Installation
============

**Develop mode**

.. code-block:: shell

  git clone https://github.com/raphaelvallat/entropy.git entropy/
  cd entropy/
  pip install -r requirements.txt
  python setup.py develop

**Dependencies**

- numpy
- scipy
- scikit-learn

Functions
=========

**1. Permutation entropy**

.. code-block:: python

    from entropy import perm_entropy
    x = [4, 7, 9, 10, 6, 11, 3]
    print(perm_entropy(x, order=3, normalize=True))

.. parsed-literal::

    0.589

**2. Spectral entropy**

.. code-block:: python

    from entropy import spectral_entropy
    import numpy as np
    np.random.seed(1234567)
    x = np.random.rand(3000)
    print(spectral_entropy(x, 100, method='welch', normalize=True))

.. parsed-literal::

    0.994

**3. Singular value decomposition (SVD) entropy**

.. code-block:: python

    from entropy import svd_entropy
    x = [4, 7, 9, 10, 6, 11, 3]
    print(svd_entropy(x, order=3, delay=1, normalize=True))

.. parsed-literal::

    0.421

**4. Approximate entropy**

.. code-block:: python

    from entropy import app_entropy
    import numpy as np
    np.random.seed(1234567)
    x = np.random.rand(3000)
    print(app_entropy(x, order=2, metric='chebyshev'))

.. parsed-literal::

    2.075

**5. Sample entropy**

.. code-block:: python

    from entropy import sample_entropy
    import numpy as np
    np.random.seed(1234567)
    x = np.random.rand(3000)
    print(sample_entropy(x, order=2, metric='chebyshev'))

.. parsed-literal::

    2.191

**6. Petrosian fractal dimension**

.. code-block:: python

    from entropy import petrosian_fd
    import numpy as np
    np.random.seed(1234567)
    x = np.random.rand(3000)
    print(petrosian_fd(x))

.. parsed-literal::

    1.0303

**7. Katz fractal dimension**

.. code-block:: python

    from entropy import katz_fd
    import numpy as np
    np.random.seed(1234567)
    x = np.random.rand(3000)
    print(katz_fd(x))

.. parsed-literal::

    9.4964


Execution time
==============

Some benchmarks computed on an average PC (i7-7700HQ CPU @ 2.80 Ghz - 8 Go of RAM)

.. code-block:: python

    from entropy import *
    import numpy as np
    np.random.seed(1234567)
    x = np.random.rand(1000)
    # Entropy
    %timeit perm_entropy(x, order=3, delay=1)
    %timeit spectral_entropy(x, 100, method='fft')
    %timeit svd_entropy(x, order=3, delay=1)
    %timeit app_entropy(x, order=2)
    %timeit sample_entropy(x, order=2)
    # Fractal dimension
    %timeit petrosian_fd(x)
    %timeit katz_fd(x)

.. parsed-literal::

    # Entropy
    126 µs ± 3.8 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    137 µs ± 2.1 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    43 µs ± 462 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    4.86 ms ± 107 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    5 ms ± 277 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    # Fractal
    16.8 µs ± 99.5 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
    35.4 µs ± 390 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

Development
===========

EntroPy was created and is maintained by `Raphael Vallat <https://raphaelvallat.github.io>`_. Contributions are more than welcome so feel free to contact me, open an issue or submit a pull request!

To see the code or report a bug, please visit the `GitHub repository <https://github.com/raphaelvallat/entropy>`_.

Note that this program is provided with NO WARRANTY OF ANY KIND. If you can, always double check the results with another software.

Acknowledgement
===============

Several functions of EntroPy were borrowed from:

- MNE-features: https://github.com/mne-tools/mne-features
- pyEntropy: https://github.com/nikdon/pyEntropy
- pyrem: https://github.com/gilestrolab/pyrem
