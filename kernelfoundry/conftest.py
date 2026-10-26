"""This is the conftest file for configuring pytest for kernelfoundry tasks.

Fixtures in this file are available in the `task.py` file and provide utilities
to measure performance, switch between reference and kernel code, and use template arguments.
"""

import base64
import os
import sys
from typing import Callable, Union
import pytest
import numpy as np
import logging
from pathlib import Path
from functools import partial, wraps
import json
import subprocess


def pytest_configure(config):
    """This function registers the "performance" custom marker and configures pytest logging.

    Note:
        This function is not intended to be called directly by test code.
    """
    # register an additional marker
    config.addinivalue_line(
        "markers",
        "performance: Marker for tests that are used to measure the runtime performance and collect trace data",
    )
    config.option.log_cli = True
    if config.option.log_cli_level is None:
        config.option.log_cli_level = "INFO"


def pytest_addoption(parser):
    """Adds CLI options for performance measurements and templating behavior.

    Note:
        This function is not intended to be called directly by test code.
    """
    parser.addoption(
        "--performance-out",
        action="store",
        default=None,
        help="Path to write the runtime measurements as JSON at the end of test run",
    )
    parser.addoption(
        "--correctness-out",
        action="store",
        default=None,
        help="Path to write the per-test correctness outcomes as JSON at the end of test run",
    )
    parser.addoption(
        "--itt",
        action="store_true",
        default=False,
        help="If set annotates the performance trials with ITT",
    )
    parser.addoption(
        "--ref",
        action="store_true",
        default=False,
        help="Boolean flag to run the reference code",
    )
    parser.addoption(
        "--is_templated",
        action="store_true",
        default=False,
        help="Boolean flag to indicate that the kernel is templated",
    )
    parser.addoption(
        "--template_params",
        type=str,
        default=None,
        help="Parameters to plug into the templated kernel",
    )
    parser.addoption(
        "--runtime_params",
        type=str,
        default=None,
        help="JSON-encoded runtime parameters from config.yaml hyperparameters.runtime",
    )


def get_runtime_params_from_argv() -> dict:
    """Parse ``--runtime_params`` from sys.argv into a dict for use in pytest.mark.skipif decorators."""

    for arg in sys.argv:
        if arg.startswith("--runtime_params="):
            encoded = arg.split("=", 1)[1]
            return json.loads(base64.b64decode(encoded).decode())
    return {}


@pytest.fixture(scope="session")
def use_reference(request) -> bool:
    """Fixture to easily switch to the reference implementation.

    This can be used in tests to conditionally run the reference code instead of the kernel code, for example to collect performance data on the reference implementation or to verify that the reference implementation runs correctly.

    Example:

        Run the test with the reference implementation::

            python -m pytest --ref task.py

    """
    return request.config.getoption("--ref")


@pytest.fixture(scope="session")
def profile_store(request) -> dict:
    """This store is used to collect per-test performance results in a dictionary.

    The keys of the dictionary are the test node ids, and the values are lists of measurements collected during the test. Tests can append to the list to collect multiple measurements.

    Returns:
        dict: Mutable store keyed by test node id.
    """
    # Mutable container shared across all tests
    store = {}
    yield store  # Tests run; they can modify store
    # Teardown: write file if option provided
    out_path = request.config.getoption("--performance-out")
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(store, f)


@pytest.fixture(scope="session", autouse=True)
def correctness_store(request) -> dict:
    """This store is used to collect per-test correctness results in a dictionary.

    The keys of the dictionary are the test node ids, and the values are whether the test
    passed. Unlike profile_store, the tests cannot fill this as they run: a failing test
    raises, so it never reaches a line that would record its own verdict. The outcomes
    pytest has already accumulated are read back at teardown instead, which is also why
    this is autouse rather than requested by a test.

    Returns:
        dict: Store keyed by test node id, populated once the tests have run.
    """
    # Mutable container, filled at teardown rather than by the tests
    store = {}
    yield store  # Tests run; pytest records each outcome
    reporter = request.config.pluginmanager.getplugin("terminalreporter")
    if reporter is not None:
        store.update(
            {
                report.nodeid: report.passed
                for reports in reporter.stats.values()
                for report in reports
                # A test skipped in its own body reports as not passed, so recording it
                # would claim the kernel is wrong on a case that was never checked. Tests
                # skipped by a marker never reach the call phase and are already absent.
                if getattr(report, "when", None) == "call" and not report.skipped
            }
        )
    # Teardown: write file if option provided
    out_path = request.config.getoption("--correctness-out")
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(store, f)


@pytest.fixture(autouse=True)
def profiler_test_label(request):
    """Expose the current pytest node id to runtime helpers used during profiling."""
    env_var = "KERNELFOUNDRY_PROFILE_LABEL"
    previous_value = os.environ.get(env_var)
    os.environ[env_var] = request.node.nodeid
    try:
        yield
    finally:
        if previous_value is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = previous_value


@pytest.fixture
def measure_runtime_torch(request, profile_store) -> Callable:
    """Create a torch runtime measurement helper with shared output.

    Args:
        request (pytest.FixtureRequest): Pytest fixture request.
        profile_store (dict): Mutable store for per-test measurements.

    Returns:
        Callable: Wrapped measure_runtime_torch callable.

        The callable is :func:`kernelfoundry.eval_pipeline.utils.performance.measure_runtime_torch`
        with the `use_itt` and `output` parameters pre-configured.
    """
    from kernelfoundry.eval_pipeline.utils.performance import measure_runtime_torch as _measure_runtime_torch

    use_itt = request.config.getoption("--itt")
    profile_store[request.node.nodeid] = []
    fn = partial(
        _measure_runtime_torch,
        use_itt=use_itt,
        output=profile_store[request.node.nodeid],
    )
    return fn


@pytest.fixture
def measure_runtime_benchdnn(request, profile_store) -> Callable:
    """Create a benchdnn runtime measurement helper with shared output."""
    from kernelfoundry.eval_pipeline.utils.benchdnn_helper import measure_runtime_benchdnn as _measure_runtime_benchdnn

    profile_store[request.node.nodeid] = []
    fn = partial(
        _measure_runtime_benchdnn,
        output=profile_store[request.node.nodeid],
    )
    return fn


@pytest.fixture
def check_correctness_benchdnn() -> Callable:
    """Create a benchdnn correctness helper."""
    from kernelfoundry.eval_pipeline.utils.benchdnn_helper import (
        check_correctness_benchdnn as _check_correctness_benchdnn,
    )

    return _check_correctness_benchdnn


# TODO replace this with the measure_runtime_x functions to not confuse this with the torch profiler
@pytest.fixture
def torch_profile(measure_runtime_torch) -> Callable:
    """Deprecated. Alias for the torch runtime measurement helper.

    Attention:
        This fixture is deprecated. Please use :func:`measure_runtime_torch` instead.

    Args:
        measure_runtime_torch (Callable): Runtime measurement helper.

    Returns:
        Callable: Same helper passed in.
    """
    return measure_runtime_torch


@pytest.fixture
def measure_runtime(request, profile_store) -> Callable:
    """Create a generic runtime measurement helper with shared output.

    Args:
        request (pytest.FixtureRequest): Pytest fixture request.
        profile_store (dict): Mutable store for per-test measurements.

    Returns:
        Callable: Wrapped measure_runtime callable.

            The callable is :func:`kernelfoundry.eval_pipeline.utils.performance.measure_runtime`
            with the `use_itt` and `output` parameters pre-configured.
    """
    from kernelfoundry.eval_pipeline.utils.performance import measure_runtime as _measure_runtime

    use_itt = request.config.getoption("--itt")
    profile_store[request.node.nodeid] = []
    fn = partial(
        _measure_runtime,
        use_itt=use_itt,
        output=profile_store[request.node.nodeid],
    )
    return fn


@pytest.fixture(scope="session")
def ocl_queue():
    """Fixture to create an OpenCL command queue for GPU execution."""
    import pyopencl as cl

    # Find the first GPU device across all platforms
    device = None
    for platform in cl.get_platforms():
        devs = platform.get_devices(cl.device_type.GPU)
        if devs:
            device = devs[0]
            break
    assert device is not None, "No OpenCL GPU device found"
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    return queue


@pytest.fixture(scope="session")
def template_args_wrapper(request) -> bool:
    """Fixture to apply template arguments"""

    def create_partial_with_end_args(func):
        param_list = eval(request.config.getoption("--template_params"))  # TODO: eval of string not ideal
        return lambda *args: func(*args, *param_list)

    if request.config.getoption("--is_templated"):
        return lambda func: create_partial_with_end_args(func)
    else:
        return lambda func: func


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    """Session-scoped fixture to add cleanup code after all tests have run."""

    def stop_unitrace_session():
        unitrace_cmd = os.environ.get("KERNELFOUNDRY_unitrace_cmd", "unitrace")
        unitrace_session = os.environ.get("UNITRACE_Session")
        if unitrace_session is not None:
            status = subprocess.run([unitrace_cmd, "--stop", unitrace_session], text=True, capture_output=True)
            logging.debug(
                f"Stopped unitrace session: {unitrace_session}, returncode: {status.returncode}, stdout: {status.stdout}, stderr: {status.stderr}"
            )

    request.addfinalizer(stop_unitrace_session)
