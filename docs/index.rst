.. kernelfoundry documentation master file

Welcome to KernelFoundry's documentation!
==========================================

KernelFoundry is an open-source framework for hardware-aware GPU kernel
optimization. It provides an evolutionary kernel generation algorithm
alongside a kernel evaluation pipeline for testing and benchmarking.

The core workflow prompts a large language model to transpile or rewrite
operators into optimized GPU kernels.
Each iteration feeds compile results, correctness outcomes, runtime
measurements and profiling data back.

What KernelFoundry Includes
---------------------------

- Kernel generation algorithm based on Map-Elits evolutionary search and meta-prompting.
- Evaluation pipeline for correctness checks, runtime benchmarking, and profiler feedback.
- Web UI for monitoring jobs, comparing kernels, and inspecting optimization progress.
- MCP server to integrate the KF test harness with an agentic workflow.

Installation
------------

Install the base package with::

    pip install .

KernelFoundry requires Python 3.10+.

For full functionality including generation pipeline, install::

    pip install .[all]

For full setup options, see the repository README.

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
