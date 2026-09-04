# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "kernelfoundry"
copyright = "2026, Intel"
author = "Intel"

# The version info for the project
from kernelfoundry import __version__

version = __version__
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "myst_parser",  # the user guide under guide/ is authored in markdown
]

# Markdown extras used by the user guide. colon_fence allows ::: directives inside
# markdown; deflist is used for the concept definitions.
myst_enable_extensions = ["colon_fence", "deflist"]

# Turn on sphinx.ext.autosummary
autosummary_generate = True
autosummary_ignore_prefixes = ["kernelfoundry."]

# Shorten the "On this page" TOC entries so members show as e.g. "compiled()"
# instead of "EvalResult.compiled()".
toc_object_entries_show_parents = "hide"

# Mock celery as it transforms decorated functions into Task objects, hiding signatures from autodoc
autodoc_mock_imports = ["celery"]

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# README.md is a build note for maintainers, not a docs page. It must be excluded now that
# myst_parser treats .md files under docs/ as sources, or Sphinx warns that it is orphaned.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "shibuya"
html_static_path = ["_static"]
html_favicon = "_static/favicon.png"
html_theme_options = {
    "globaltoc_expand_depth": 2,
    "show_ai_links": False,
}
html_css_files = [
    "custom.css",
]

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "private-members": False,
}

# -- Intersphinx mapping -----------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
