# kernelfoundry Documentation

This directory contains the Sphinx documentation for the kernelfoundry package.

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install .[docs]
```

Or install the required packages directly:

```bash
pip install sphinx sphinx-rtd-theme
```

You also need to install the package dependencies (numpy, pytest) for the autodoc extension to import the modules:

```bash
pip install numpy pytest
```

### Build HTML Documentation

To build the HTML documentation:

```bash
cd docs
make html
```

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
- `index.rst` - Main documentation index
- `api/` - API reference documentation
- `_static/` - Static files (CSS, images, etc.)
- `_templates/` - Custom Sphinx templates
- `_build/` - Generated documentation (excluded from git)

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
