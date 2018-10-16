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

EntroPy is a Python 3 package for computing several entropy metrics of time series.

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

Functions
=========

**1. Permutation entropy**

.. code-block:: python

    from entropy import perm_entropy
    x = [4, 7, 9, 10, 6, 11, 3]
    print(perm_entropy(x, order=3, normalize=True))

.. parsed-literal::

    0.589

Development
===========

EntroPy was created and is maintained by `Raphael Vallat <https://raphaelvallat.github.io>`_. Contributions are more than welcome so feel free to contact me, open an issue or submit a pull request!

To see the code or report a bug, please visit the `GitHub repository <https://github.com/raphaelvallat/entropy>`_.

Note that this program is provided with NO WARRANTY OF ANY KIND. If you can, always double check the results with another software.

Acknowledgement
===============

Several functions of EntroPy were borrowed from:

- pyEntropy: https://github.com/nikdon/pyEntropy
- MNE-features: https://github.com/mne-tools/mne-features
