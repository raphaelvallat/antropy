.. _Changelog:

What's new
##########

v0.1.3 (dev)
------------

a. Use the `stochastic <https://github.com/crflynn/stochastic>`_ package to generate stochastic time-series.
b. Add support for 2D data in :py:func:`entropy.spectral_entropy`.
c. Changed default method for PSD calculation from ``fft`` to the more stable``welch`` in :py:func:`entropy.spectral_entropy`.

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
