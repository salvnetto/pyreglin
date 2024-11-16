import pyreglin

project = 'pyreglin'
copyright = '2024, Fábio N. Demarqui, Salvador Netto, Tomás Bernardes'
author = 'Fábio N. Demarqui, Salvador Netto, Tomás Bernardes'
version = pyreglin.__version__
release = version


extensions = [
    'sphinx.ext.autodoc',    # Automatically include docstrings
    'sphinx.ext.napoleon',  # Support Google and NumPy docstring styles
    'sphinx.ext.viewcode',  # Include source code links
    'sphinx.ext.intersphinx' # Link to other projects' documentation
]


templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
