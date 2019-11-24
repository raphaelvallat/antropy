#! /usr/bin/env python
#
# Copyright (C) 2018 Raphael Vallat

DESCRIPTION = "EntroPy: entropy of time-series in Python"
DISTNAME = 'entropy'
MAINTAINER = 'Raphael Vallat'
MAINTAINER_EMAIL = 'raphaelvallat9@gmail.com'
URL = 'https://raphaelvallat.com/entropy/build/html/index.html'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/raphaelvallat/entropy/'
VERSION = '0.1.1'
PACKAGE_DATA = {'entropy.data.icons': ['*.ico']}

try:
    from setuptools import setup
    _has_setuptools = True
except ImportError:
    from distutils.core import setup


def check_dependencies():
    install_requires = []

    try:
        import numpy
    except ImportError:
        install_requires.append('numpy')
    try:
        import scipy
    except ImportError:
        install_requires.append('scipy')

    try:
        import sklearn
    except ImportError:
        install_requires.append('scikit-learn')

    try:
        import numba
    except ImportError:
        install_requires.append('numba')


    return install_requires


if __name__ == "__main__":

    install_requires = check_dependencies()

    setup(name=DISTNAME,
          author=MAINTAINER,
          author_email=MAINTAINER_EMAIL,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          install_requires=install_requires,
          include_package_data=True,
          packages=['entropy'],
          package_data=PACKAGE_DATA,
          classifiers=[
              'Intended Audience :: Science/Research',
              'Programming Language :: Python :: 3.6',
              'License :: OSI Approved :: BSD License',
              'Topic :: Scientific/Engineering :: Mathematics',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS'],
          )
