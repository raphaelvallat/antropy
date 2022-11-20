.. _Changelog:

What's new
##########

v0.1.5 (dev)
------------

a. :py:func:`antropy.perm_entropy` will now return the average entropy across all delays if a list or range of delays is provided.
b. Handle the limit of p = 0 in functions that evaluate the product p * log2(p), to give 0 instead of nan (see `PR3 <https://github.com/raphaelvallat/antropy/pull/3>`_).
c. :py:func:`antropy.detrended_fluctuation` will now return alpha = 0 when the correlation coefficient of the fluctuations of an input signal is 0 (see `PR21 <https://github.com/raphaelvallat/antropy/pull/21>`_).

v0.1.4 (April 2021)
-------------------

.. important:: The package has now been renamed AntroPy (previously EntroPy)!

a. Faster implementation of :py:func:`antropy.lziv_complexity` (see `PR1 <https://github.com/raphaelvallat/entropy/pull/1>`_). Among other improvements, strings are now mapped to UTF-8 integer representations.

v0.1.3 (March 2021)
-------------------

a. Added the :py:func:`entropy.num_zerocross` function to calculate the (normalized) number of zero-crossings on N-D data.
b. Added the :py:func:`entropy.hjorth_params` function to calculate the mobility and complexity Hjorth parameters on N-D data.
c. Add support for N-D data in :py:func:`entropy.spectral_entropy`, :py:func:`entropy.petrosian_fd` and :py:func:`entropy.katz_fd`.
d. Use the `stochastic <https://github.com/crflynn/stochastic>`_ package to generate stochastic time-series.

v0.1.2 (May 2020)
-----------------

a. :py:func:`entropy.lziv_complexity` now works with non-binary sequence (e.g. "12345" or "Hello World!")
b. The average fluctuations in :py:func:`entropy.detrended_fluctuation` is now calculated using the root mean square instead of a simple arithmetic. For more details, please refer to `this GitHub issue <https://github.com/neuropsychology/NeuroKit/issues/206>`_.
c. Updated flake8

v0.1.1 (November 2019)
----------------------

a. Added Lempel-Ziv complexity (:py:func:`entropy.lziv_complexity`) for binary sequence.

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
