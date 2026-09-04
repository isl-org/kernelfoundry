# kernelfoundry Documentation

This directory contains the Sphinx documentation for the kernelfoundry package.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install .[docs]
```

That installs Sphinx, the [shibuya](https://shibuya.lepture.com/) theme, `sphinx-copybutton` and
`myst-parser`. Autodoc imports the package to read docstrings, so the package's own dependencies
must be installed too: `pip install -e .` from the repository root is the simplest way.

### Build HTML Documentation

To build the HTML documentation:

```bash
cd docs
sphinx-build -M html . _build
```

`make html` works too where `make` is available, but `make.bat` activates a `.venv` at the
repository root when `VIRTUAL_ENV` is unset, which is wrong for any environment that lives
elsewhere. The `sphinx-build` form above is portable and uses whichever interpreter is active.

The generated HTML documentation will be in `_build/html/`. Open `_build/html/index.html` in your browser to view the documentation.

### Clean Build Artifacts

To remove all generated documentation files:

```bash
cd docs
make clean
```

### Other Output Formats

Sphinx supports multiple output formats. To see all available formats:

```bash
cd docs
make help
```

Some useful formats include:
- `make html` - HTML documentation (default)
- `make singlehtml` - Single HTML page
- `make latexpdf` - PDF documentation (requires LaTeX)
- `make epub` - EPUB documentation

## Documentation Structure

- `conf.py` - Sphinx configuration file
- `index.rst` - Landing page and toctrees
- `guide/` - User guide, written in **Markdown** (via `myst-parser`)
- `api/public.rst` - Curated public API: what a task author actually uses
- `api/modules.rst` - Full recursive module index, including internals
- `_static/` - Static files (CSS, images, etc.)
- `_templates/` - Custom Sphinx templates
- `_build/` - Generated documentation (excluded from git)

The published site is assembled by `.github/workflows/deploy-pages.yml`, which builds this
directory and merges it into the separate website repository under `/docs/`.

Prose belongs in `guide/` as Markdown; the API reference is generated from docstrings. See
[CONTRIBUTING.md](../CONTRIBUTING.md#documentation) for which content belongs here versus in the
README.

## Docstring Format

The kernelfoundry package uses Google-style docstrings. When adding or modifying code, please follow this format:

```python
def example_function(arg1, arg2):
    """Brief description of the function.
    
    More detailed description if needed.
    
    Args:
        arg1 (type): Description of arg1.
        arg2 (type): Description of arg2.
    
    Returns:
        type: Description of return value.
    
    Raises:
        ExceptionType: Description of when this exception is raised.
    """
    pass
```

For more information on Google-style docstrings, see:
https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
