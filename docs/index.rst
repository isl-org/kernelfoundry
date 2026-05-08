.. kernelfoundry documentation master file

Welcome to kernelfoundry's documentation!
==========================================

KernelFoundry is a Python package for defining and evaluating GPU kernel tasks.

This package provides the task-side tooling used to compile candidate kernels,
validate correctness, and measure runtime performance in a consistent pytest-based
workflow.

What kernelfoundry provides
---------------------------

- A base task interface via ``TestBase`` (import with ``from kernelfoundry import TestBase``).
- Build helpers to compile kernel sources (SYCL and CUDA).
- Pytest fixtures for toggling reference implementations, collecting performance
    samples, and handling template parameters.
- Validation utilities for numerical comparison of kernel output.
- Runtime measurement and system-info utilities for reproducible benchmarking.

Typical task workflow
---------------------

1. Implement a task test class derived from ``TestBase``.
2. Build candidate kernel artifacts (optionally using
     ``TestBase.compile_torch_extension``).
3. Run correctness tests against a reference implementation.
4. Run performance tests (``@pytest.mark.performance``) and collect timing data.
5. Optionally export profiling results with ``--performance-out``.

Installation
------------

Install the package and basic dependencies with::

    pip install .

The package requires Python 3.10+ (commonly used with Python 3.12). Install into a virtual environment if desired.

For documentation building, install the docs dependencies::

    pip install .[docs]

API Reference
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
