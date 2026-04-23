"""Runtime measurement utilities."""

import sys
import os
import time
import logging
from typing import Callable, Union
import numpy as np
import subprocess
from contextlib import contextmanager, nullcontext
from collections.abc import Iterable, Mapping
from copy import deepcopy


def detect_profiler() -> str | None:
    """Detect whether the process is running under a profiler.

    Returns:
        str | None: The detected profiler name, or None if not detected.
    """
    # check for the env var set by us
    result = os.environ.get("KERNELFOUNDRY_PROFILER")
    if result is not None:
        return result
    # try detecting the profiler from known env vars set by the profilers
    profiler_vars = [
        ("unitrace", ("UNITRACE_VERSION",)),
        ("ncu", ("NV_NSIGHT_INJECTION_PORT_BASE",)),
    ]
    for profiler_name, env_vars in profiler_vars:
        for var in env_vars:
            if var in os.environ:
                return profiler_name
    return None


@contextmanager
def profiler_session(profiler_name: str, stop_on_exit: bool = False):
    """Context manager to resume, pause and stop profiler sessions.

    Supports only unitrace for now.

    Args:
        profiler_name (str): The name of the profiler to manage.
        stop_on_exit (bool): Whether to stop the profiler session on exit.
        If False, the session will be paused instead of stopped.

    """
    if profiler_name == "unitrace":
        unitrace_cmd = os.environ.get("KERNELFOUNDRY_unitrace_cmd", "unitrace")
        unitrace_session = os.environ.get("UNITRACE_Session")

        if unitrace_session is not None:
            status = subprocess.run([unitrace_cmd, "--resume", unitrace_session], text=True, capture_output=True)
            logging.debug(
                f"Resumed unitrace session: {unitrace_session}, returncode: {status.returncode}, stdout: {status.stdout}, stderr: {status.stderr}"
            )
    try:
        yield
    finally:
        if profiler_name == "unitrace" and unitrace_session is not None:
            if stop_on_exit:
                status = subprocess.run([unitrace_cmd, "--stop", unitrace_session], text=True, capture_output=True)
                logging.debug(
                    f"Stopped unitrace session: {unitrace_session}, returncode: {status.returncode}, stdout: {status.stdout}, stderr: {status.stderr}"
                )
            else:
                status = subprocess.run([unitrace_cmd, "--pause", unitrace_session], text=True, capture_output=True)
                logging.debug(
                    f"Paused unitrace session: {unitrace_session}, returncode: {status.returncode}, stdout: {status.stdout}, stderr: {status.stderr}"
                )


def _get_size_in_bytes(obj) -> int:
    """Recursively get the size of an object in bytes.

    Supports tensors, numpy arrays, mappings and iterables. For other objects, returns 0.
    """
    if hasattr(obj, "nbytes"):  # numpy and torch
        return obj.nbytes
    elif isinstance(obj, (str, bytes)):
        return len(obj)
    elif isinstance(obj, Mapping):
        return sum(_get_size_in_bytes(k) + _get_size_in_bytes(v) for k, v in obj.items())
    elif isinstance(obj, Iterable):
        return sum(_get_size_in_bytes(i) for i in obj)
    else:
        return 0


def _replicate_inputs(args, kwargs, target_size, max_repeats=10000):
    """Replicate the inputs, args and kwargs, to the target size to avoid caching effects for very small inputs.
    Args:
        args: tuple of positional arguments or list of tuples of positional arguments
        kwargs: dict of keyword arguments or list of dicts of keyword arguments
        target_size: the target size to replicate to in bytes.
            If the total size of args and kwargs is already above the target size, no replication will be done.

    Returns:
        Replicated args and kwargs as lists of tuples and dicts, respectively.
    """
    size_in_bytes = _get_size_in_bytes(args) + _get_size_in_bytes(kwargs)
    if size_in_bytes < target_size:
        list_args = [args]
        list_kwargs = [kwargs]
        num_repeats = int(target_size / size_in_bytes)
        if num_repeats > max_repeats:
            logging.warning(
                f"Calculated number of repeats for input replication is {num_repeats}, which is above the max_repeats limit of {max_repeats}. Capping the number of repeats to {max_repeats}."
            )
            num_repeats = max_repeats

        # use the no_grad context manager from torch if available
        torch_mod = sys.modules.get("torch")
        if torch_mod is not None:
            no_grad = torch_mod.no_grad
        else:
            no_grad = nullcontext

        with no_grad():
            for _ in range(num_repeats):
                list_args.append(deepcopy(args))
                list_kwargs.append(deepcopy(kwargs))
        return list_args, list_kwargs
    elif size_in_bytes == 0:
        logging.warning(
            "Args and kwargs appear to be empty or of unsupported types for size calculation and cannot be replicated"
        )
        return [args], [kwargs]
    return [args], [kwargs]


def measure_runtime(
    target: Callable,
    sync_fn: Callable,
    args: list[tuple] | list[list] | tuple | list | None = None,
    kwargs: list[dict] | dict | None = None,
    warmup_min_time: float = 1.0,
    warmup_min_iters: int = 10,
    inner_loop_min_time: float = 0.01,
    perf_trials_min_iters: int = 10,
    perf_trials_min_time: float = 1.0,
    use_itt: bool = False,
    reduce_iterations_for_external_profiler: bool = True,
    auto_replicate_inputs_size: int = 128 * 2**20,  # 128MB
    info_str: str = "",
    output: list[float] | None = None,
) -> list[float]:
    """Measures the runtime of the target callable.

    The function assumes that all invocations of the target are run in order on the same device,
    and that the sync_fn function will synchronize the device to ensure all operations are complete.

    Args:
        target (Callable): The kernel function to be measured.
        sync_fn (Callable): The synchronization function to be called after target execution.
        args: Positional arguments to pass to the target function. This can be a list of positional arguments to iterate
            over different inputs for each call to the target. Note that you must provide a list of kwargs of the same length
            if you provide a list of args. Use `kwargs=len(args)*[{}]` if there are no kwargs to pass.
            Note that arguments for each list entry should have the same shape and structure.
        kwargs: Keyword arguments to pass to the target function. If this is a list of keyword argument dictionaries,
            then this function will iterate through the list to use a different set of kwargs for each call to the
            target with the intent to avoid caching effects. Note that args must be a list of tuples/lists of the same
            length as kwargs in this case.
            Note that arguments for each list entry should have the same shape and structure.
        warmup_min_time (float): Minimum total time for warmup phase in seconds.
        warmup_min_iters (int): Minimum number of iterations for warmup phase.
        inner_loop_min_time (float): Minimum time for inner loop trials in seconds.
        perf_trials_min_iters (int): Minimum number of performance trials.
        perf_trials_min_time (float): Minimum total time for performance trials in seconds.
        use_itt (bool): Whether to use ITT annotations during profiling.
        reduce_iterations_for_external_profiler (bool): If an external profiler is detected,
            reduce the number of iterations to avoid long profiling sessions.
        auto_replicate_inputs_size (int): Replicate the inputs, args and kwargs, to this size to avoid caching effects
            for very small inputs. Set to 0 to disable replication. This option has no effect if args and kwargs are
            lists of arguments.
        info_str (str): Additional info string added before the timing info about warmup and test iterations. Useful for
            adding information about the device.
        output (list[float]): Optional list to store the measured runtimes.

    Returns:
        List of measured runtimes in milliseconds.
    """
    ext_profiler = detect_profiler()
    return_early = reduce_iterations_for_external_profiler and ext_profiler in ("ncu",)

    if use_itt:
        from ittapi import task as itt_task
    else:
        from contextlib import nullcontext as itt_task

    if args is None:
        args = tuple()
    if kwargs is None:
        kwargs = {}

    if isinstance(kwargs, Mapping):
        if auto_replicate_inputs_size > 0:
            args, kwargs = _replicate_inputs(args, kwargs, auto_replicate_inputs_size)
        else:
            args = [args]
            kwargs = [kwargs]
    else:  # kwargs must be a list or tuple
        assert len(args) == len(kwargs), "If args and kwargs are lists, they must have the same length"

    arg_list_size = len(kwargs)
    arg_i = 0

    # for OCL, we need to enable collection before any trial is executed. This is ignored for SYCL (--session control)
    os.environ["PTI_ENABLE_COLLECTION"] = "1"

    # Test trials: run kernel for five trials to get the approximate runtime
    test_trials_time = []
    num_test_trials = {"ncu": 1}.get(ext_profiler, 5)

    with itt_task("model run loop" if return_early else "test trials"):
        for _ in range(num_test_trials):
            arg_i = (arg_i + 1) % arg_list_size
            tic = time.perf_counter()
            target(*args[arg_i], **kwargs[arg_i])
            sync_fn()
            toc = time.perf_counter()
            test_trials_time.append((toc - tic))

    # return after test trials if external profiler is detected and is doing its own sampling like ncu
    if return_early:
        if output is not None:
            output.extend(test_trials_time)
        return test_trials_time

    test_trials_min = np.min(test_trials_time)  # minimum time in seconds
    # e.g. test_trials_min = 0.001s, min_time = 1s -> 1000 iterations
    warmup_time_iters = int(warmup_min_time / test_trials_min)
    # inner loop trials: at least 1, but more if min time of test trials is low
    inner_loop_iters = max([1, int(inner_loop_min_time / test_trials_min)])

    num_warmup = max([warmup_min_iters, warmup_time_iters])

    # Warmup trials (not considered for runtime stats)
    tic = time.perf_counter()
    for _ in range(num_warmup):
        arg_i = (arg_i + 1) % arg_list_size
        target(*args[arg_i], **kwargs[arg_i])
    sync_fn()
    toc = time.perf_counter()
    warmup_time = toc - tic

    # Actual trials
    num_perf_trials = max(
        perf_trials_min_iters, int(perf_trials_min_time / warmup_time * num_warmup / inner_loop_iters)
    )
    if ext_profiler and reduce_iterations_for_external_profiler:
        # Limit the number of trials for external profilers to avoid long profiling sessions,
        # while still having enough iterations for stable measurements
        inner_loop_iters = max(5, min(20, inner_loop_iters))
        num_perf_trials = 1

    logging.info(
        f"[Performance test] {info_str} warm up {num_warmup}, trials {num_perf_trials} ({inner_loop_iters} passes each)"
    )
    elapsed_times = []

    with profiler_session(ext_profiler):
        with itt_task("model run loop"):
            for _ in range(num_perf_trials):
                sync_fn()
                tic = time.perf_counter()
                for _ in range(inner_loop_iters):
                    arg_i = (arg_i + 1) % arg_list_size
                    target(*args[arg_i], **kwargs[arg_i])

                # Synchronize to ensure the events have completed
                sync_fn()
                toc = time.perf_counter()

                elapsed_time_ms = ((toc - tic) / inner_loop_iters) * 1000  # Convert to milliseconds
                elapsed_times.append(elapsed_time_ms)

    os.environ["PTI_ENABLE_COLLECTION"] = "0"
    if output is not None:
        output.extend(elapsed_times)
    logging.info(
        f"[Performance test] Avg: {np.mean(elapsed_times):.3f} ms, Min: {np.min(elapsed_times):.3f} ms, Max: {np.max(elapsed_times):.3f} ms, Median: {np.median(elapsed_times):.3f} ms"
    )
    return elapsed_times


def measure_runtime_torch(
    target: Callable,
    device: Union[str, "torch.device"],
    args: tuple | list | None = None,
    kwargs: dict | None = None,
    warmup_min_time: float = 1.0,
    warmup_min_iters: int = 10,
    inner_loop_min_time: float = 0.01,
    perf_trials_min_iters: int = 10,
    perf_trials_min_time: float = 1.0,
    use_itt: bool = False,
    reduce_iterations_for_external_profiler: bool = True,
    auto_replicate_inputs_size: int = 128 * 2**20,  # 128MB
    output: list[float] | None = None,
) -> list[float]:
    """Measures the runtime of the target callable on the specified torch device.

    Args:
        target (Callable): The kernel function to be measured.
        device (Union[str, torch.device]): The device to use for synchronization.
        args: Positional arguments to pass to the target function. This can be a list of positional arguments to iterate
            over different inputs for each call to the target. Note that you must provide a list of kwargs of the same length
            if you provide a list of args. Use `kwargs=len(args)*[{}]` if there are no kwargs to pass.
            Note that arguments for each list entry should have the same shape and structure.
        kwargs: Keyword arguments to pass to the target function. If this is a list of keyword argument dictionaries,
            then this function will iterate through the list to use a different set of kwargs for each call to the
            target with the intent to avoid caching effects. Note that args must be a list of tuples/lists of the same
            length as kwargs in this case.
            Note that arguments for each list entry should have the same shape and structure.
        warmup_min_time (float): Minimum total time for warmup phase in seconds.
        warmup_min_iters (int): Minimum number of iterations for warmup phase.
        inner_loop_min_time (float): Minimum time for inner loop trials in seconds.
        perf_trials_min_iters (int): Minimum number of performance trials.
        perf_trials_min_time (float): Minimum total time for performance trials in seconds.
        use_itt (bool): Whether to use ITT annotations during profiling.
        reduce_iterations_for_external_profiler (bool): If an external profiler is detected,
            reduce the number of iterations to avoid long profiling sessions.
        auto_replicate_inputs_size (int): Replicate the inputs, args and kwargs, to this size to avoid caching effects
            for very small inputs. Set to 0 to disable replication. This option has no effect if args and kwargs are
            lists of arguments.
        output (list[float]): Optional list to store the measured runtimes.

    Returns:
        list[float]: List of measured runtimes in milliseconds.
    """
    import torch

    if isinstance(device, str):
        device = torch.device(device)

    if torch.xpu.is_available():
        torch_acc = torch.xpu
        assert not torch.cuda.is_available(), "Mixed xpu and cuda GPU in same machine is not supported!"
    elif torch.cuda.is_available():
        torch_acc = torch.cuda
    else:
        raise ValueError("No xpu or cuda device found")

    result = measure_runtime(
        target=target,
        sync_fn=lambda: torch_acc.synchronize(device),
        args=args,
        kwargs=kwargs,
        warmup_min_time=warmup_min_time,
        warmup_min_iters=warmup_min_iters,
        inner_loop_min_time=inner_loop_min_time,
        perf_trials_min_iters=perf_trials_min_iters,
        perf_trials_min_time=perf_trials_min_time,
        use_itt=use_itt,
        reduce_iterations_for_external_profiler=reduce_iterations_for_external_profiler,
        auto_replicate_inputs_size=auto_replicate_inputs_size,
        info_str=f"Using device: {device},",
        output=output,
    )
    return result
