# -- Path setup --------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'xcode_debarcode'
copyright = '2025, Marwane Bourdim'
author = 'Marwane Bourdim'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'myst_nb',
    'sphinx_autodoc_typehints',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'myst-nb',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- MyST-NB settings --------------------------------------------------------
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
]

myst_update_mathjax = False

nb_execution_mode = "cache"
nb_execution_timeout = 300
nb_execution_raise_on_error = True

# -- Autosummary settings ----------------------------------------------------
autosummary_generate = True
