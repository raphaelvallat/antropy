.. -*- mode: rst -*-

.. figure::  https://github.com/raphaelvallat/entropy/blob/master/docs/pictures/logo.png
   :align:   center

 EntroPy is a Python 3 package for computing several entropy metrics of time series.

 It currently includes

 - ``perm_entropy``: Permutation entropy (Bandt and Pompe, 2002)

Documentation
=============

.. toctree::
   :maxdepth: 1

   api
   changelog

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
