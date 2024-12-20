#! /usr/bin/env python
#
# Copyright (C) 2018 Raphael Vallat

DESCRIPTION = "AntroPy: entropy and complexity of time-series in Python"
DISTNAME = "antropy"
MAINTAINER = "Raphael Vallat"
MAINTAINER_EMAIL = "raphaelvallat9@gmail.com"
URL = "https://raphaelvallat.com/antropy/build/html/index.html"
LICENSE = "BSD (3-clause)"
DOWNLOAD_URL = "https://github.com/raphaelvallat/antropy/"
VERSION = "0.1.7"
PACKAGE_DATA = {"antropy.data.icons": ["*.ico"]}

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
        install_requires.append("numpy")
    try:
        import scipy
    except ImportError:
        install_requires.append("scipy")

    try:
        import sklearn
    except ImportError:
        install_requires.append("scikit-learn")

    try:
        import numba
    except ImportError:
        install_requires.append("numba>=0.57")

    try:
        import stochastic
    except ImportError:
        install_requires.append("stochastic")

    return install_requires


if __name__ == "__main__":
    install_requires = check_dependencies()

    setup(
        name=DISTNAME,
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
        packages=["antropy"],
        package_data=PACKAGE_DATA,
        classifiers=[
            "Intended Audience :: Science/Research",
            "Programming Language :: Python",
            "License :: OSI Approved :: BSD License",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
        ],
    )
