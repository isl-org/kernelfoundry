Public API
==========

The handful of things you use when writing a task package. Everything here is what a task
author touches directly; the rest of the package is internal machinery, browsable under
:doc:`modules`.

See :doc:`../guide/writing-tests` for how these fit together.

Task base class
---------------

Every task defines a class deriving from ``TestBase``. It provides the build hooks and the
compilation helper; you add the tests.

.. autoclass:: kernelfoundry.TestBase
   :members: build, build_reference, compile_torch_extension, get_machine_gpu_arch
   :undoc-members:
   :show-inheritance:
   :no-index:

Assertions
----------

Comparison helpers with tolerances appropriate to GPU floating point. ``assert_allclose`` is the
one to reach for by default.

.. autofunction:: kernelfoundry.testing.assert_allclose
   :no-index:

.. autofunction:: kernelfoundry.testing.all_close_with_slack
   :no-index:

.. autofunction:: kernelfoundry.testing.cosine_similarity
   :no-index:

Benchmarking
------------

``measure_runtime_torch`` is the simplest correct benchmark for torch workloads;
``measure_runtime`` is the general form. If you write a custom benchmark, the timed trials must
run inside ``profiler_session`` or no profiler data is collected.

.. autofunction:: kernelfoundry.eval_pipeline.utils.performance.measure_runtime_torch
   :no-index:

.. autofunction:: kernelfoundry.eval_pipeline.utils.performance.measure_runtime
   :no-index:

.. autofunction:: kernelfoundry.eval_pipeline.utils.performance.profiler_session
   :no-index:

.. autofunction:: kernelfoundry.eval_pipeline.utils.performance.detect_profiler
   :no-index:

Pytest fixtures
---------------

Available in any task that keeps the shipped ``conftest.py``. Request them as test arguments.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Fixture
     - Purpose
   * - ``measure_runtime_torch``
     - Benchmark a torch callable: moves inputs to the device, runs warmup, records runtimes.
   * - ``measure_runtime``
     - Benchmark a general callable, for non-torch workloads.
   * - ``torch_profile``
     - Thin wrapper over ``measure_runtime_torch`` for profiling runs.
   * - ``profile_store``
     - Where measured runtimes are recorded. Required if you write a custom benchmark.
   * - ``use_reference``
     - True when running under ``pytest --ref``, so the reference is exercised instead of the
       kernel.
   * - ``template_args_wrapper``
     - Supports templated kernels, where each parameter combination is benchmarked.
   * - ``profiler_test_label``
     - Labels the profiler region for the current test.

Compilers
---------

Chosen through ``eval_config.kernel_compiler``. ``TorchCompiler`` is the default and builds a
PyTorch extension; ``IcpxCompiler`` invokes ``icpx`` directly.

.. autoclass:: kernelfoundry.compiler.TorchCompiler
   :members:
   :show-inheritance:
   :no-index:

.. autoclass:: kernelfoundry.compiler.IcpxCompiler
   :members:
   :show-inheritance:
   :no-index:

MCP server
----------

The MCP server exposes one tool, ``build_and_test(folder_path)``, which builds and benchmarks a
task package and returns the outcome as structured data a coding agent can act on. Setup and
the full return contract are in the
`MCP server README <https://github.com/isl-org/kernelfoundry/blob/main/kernelfoundry/mcp_server/README.md>`_.
