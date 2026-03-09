"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import antropy

# -- Project information -----------------------------------------------------

project = "antropy"
author = "Raphael Vallat"
project_copyright = "2018-%Y, Raphael Vallat"
version = antropy.__version__
release = antropy.__version__


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",      # Includes documentation from docstrings
    "sphinx.ext.autosummary",  # Generates autodoc summaries
    "sphinx.ext.githubpages",  # Adds .nojekyll file for GitHub Pages
    "sphinx.ext.intersphinx",  # Links to other package docs
    "sphinx.ext.mathjax",      # LaTeX math display
    "sphinx.ext.viewcode",     # Adds "view source" links
    "numpydoc",                # Generates NumPy-style docstrings (load after autodoc)
    "sphinx_copybutton",       # Adds copy-to-clipboard button in code blocks
    "sphinx_design",           # Adds directives for badges, dropdowns, tabs, etc.
]

language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
master_doc = "index"
source_suffix = ".rst"

autosummary_generate = True
numpydoc_show_class_members = False
numpydoc_use_plots = True

# configure sphinx-copybutton
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True


# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "logo": {
        "alt_text": "antropy - Home",
        "image_light": "_static/antropy_128x128.png",
        "image_dark": "_static/antropy_128x128.png",
    },
    "icon_links": [
        {
            "name": "antropy on GitHub",
            "url": "https://github.com/raphaelvallat/antropy",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
    "navbar_align": "left",
    "show_prev_next": False,
    "search_bar_text": "Search",
    "navigation_depth": 2,
    "pygments_light_style": "github-light-colorblind",
    "pygments_dark_style": "github-dark-colorblind",
    "footer_start": ["copyright"],
    "footer_center": [],
    "footer_end": [],
    "secondary_sidebar_items": ["page-toc"],
    "primary_sidebar_end": [],
}

html_context = {
    "github_user": "raphaelvallat",
    "github_repo": "antropy",
    "github_version": "master",
    "doc_path": "docs",
    "default_mode": "auto",
}

html_static_path = ["_static"]
html_favicon = "_static/antropy.ico"
html_sidebars = {"**": []}  # remove left sidebar from all pages
html_show_sourcelink = False
html_show_sphinx = False
html_permalinks_icon = "#"


# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "numba": ("https://numba.readthedocs.io/en/stable", None),
    "mne": ("https://mne.tools/stable", None),
}
