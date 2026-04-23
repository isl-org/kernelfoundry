"""This is the conftest file for configuring pytest for kernelfoundry tasks.

Fixtures in this file are available in the `task.py` file and provide utilities
to measure performance, switch between reference and kernel code, and use template arguments.
"""

import os
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


@pytest.fixture
def measure_runtime_torch(request, profile_store) -> Callable:
    """Create a torch runtime measurement helper with shared output.

    Args:
        request (pytest.FixtureRequest): Pytest fixture request.
        profile_store (dict): Mutable store for per-test measurements.

    Returns:
        Callable: Wrapped measure_runtime_torch callable.

        The callable is :func:`kernelfoundry.utils.performance.measure_runtime_torch`
        with the `use_itt` and `output` parameters pre-configured.
    """
    from kernelfoundry.utils.performance import measure_runtime_torch as _measure_runtime_torch

    use_itt = request.config.getoption("--itt")
    profile_store[request.node.nodeid] = []
    fn = partial(
        _measure_runtime_torch,
        use_itt=use_itt,
        output=profile_store[request.node.nodeid],
    )
    return fn


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

            The callable is :func:`kernelfoundry.utils.performance.measure_runtime`
            with the `use_itt` and `output` parameters pre-configured.
    """
    from kernelfoundry.utils.performance import measure_runtime as _measure_runtime

    use_itt = request.config.getoption("--itt")
    profile_store[request.node.nodeid] = []
    fn = partial(
        _measure_runtime,
        use_itt=use_itt,
        output=profile_store[request.node.nodeid],
    )
    return fn


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
