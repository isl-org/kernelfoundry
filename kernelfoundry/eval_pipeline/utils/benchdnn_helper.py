"""Helpers for invoking benchdnn correctness/performance runs and parsing output."""

import csv
import logging
import os
import re
import subprocess


def _setup_benchdnn_env_vars(
    ld_preload: str | None = None,
    use_custom_kernel: bool = False,
    onednn_verbosity_level: int = 0,
) -> dict[str, str]:
    """Build environment variables for a benchdnn subprocess invocation."""
    benchdnn_env = dict(os.environ)
    if ld_preload is not None:
        benchdnn_env["LD_PRELOAD"] = ld_preload
    benchdnn_env["DNNL_USE_CUSTOM_KERNEL"] = "1" if use_custom_kernel else "0"
    benchdnn_env["ONEDNN_VERBOSE"] = str(onednn_verbosity_level)
    return benchdnn_env


def parse_benchdnn_correctness_check(stdout: str) -> bool:
    """Detect correctness success from benchdnn stdout.

    This parser looks for benchdnn result lines in the REPRO format:
    ``case_num:status[ ... ] __REPRO: ...``.

    Expected input includes exactly one REPRO line.

    Args:
        stdout: Raw stdout text produced by benchdnn in correctness mode.

    Returns:
        ``True`` when the single REPRO line has ``PASSED`` status,
        otherwise ``False``.

    Raises:
        ValueError: If stdout contains zero or multiple REPRO lines, or if
            the REPRO line does not match the expected ``case_num:status``
            prefix format.
    """
    repro_lines = [line.strip() for line in stdout.splitlines() if "__REPRO:" in line]
    if len(repro_lines) != 1:
        raise ValueError("Expected exactly 1 benchdnn correctness REPRO line in stdout, " f"got {len(repro_lines)}")

    # Expected prefix: '<case_num>:<status>' before '__REPRO:'.
    repro_prefix = repro_lines[0].split("__REPRO:", 1)[0].strip()
    match = re.match(r"^\d+:([A-Z_]+)\b", repro_prefix)
    if not match:
        raise ValueError(
            "Malformed benchdnn correctness REPRO line; expected "
            "'<case_num>:<status> ... __REPRO: ...'. "
            f"line={repro_lines[0]}"
        )

    return match.group(1) == "PASSED"


def parse_benchdnn_perf_stats(stdout: str) -> dict:
    """Extract ``avg_time`` from benchdnn performance stdout.

    Expected input includes exactly two CSV-like ``perf,`` lines: one header
    row and one value row.

    Args:
        stdout: Raw stdout text produced by benchdnn in performance mode.

    Returns:
        A dictionary containing the parsed ``avg_time`` value as ``float``.

    Raises:
        ValueError: If expected perf rows are missing, malformed, missing
            ``avg_time``, or contain a non-numeric ``avg_time`` value.
    """
    perf_lines = []
    for line in stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        if text.startswith("perf,"):
            perf_lines.append(text)

    if len(perf_lines) != 2:
        raise ValueError(
            f"Expected 2 perf lines in benchdnn stdout (1 for headers and 1 for values), got {len(perf_lines)}"
        )

    headers = next(csv.reader([perf_lines[0]]))
    values = next(csv.reader([perf_lines[1]]))
    if len(headers) != len(values):
        raise ValueError(f"benchdnn perf stats have mismatched headers and values:\n{headers}\n{values}")

    if "avg_time" not in headers:
        raise ValueError(f"benchdnn perf stats missing 'avg_time' column: {headers}")

    avg_time_text = values[headers.index("avg_time")].strip()
    try:
        return {"avg_time": float(avg_time_text)}
    except ValueError:
        raise ValueError(f"benchdnn perf stats contained non-numeric 'avg_time' value: {avg_time_text}")


def check_correctness_benchdnn(
    benchdnn_bin: str,
    driver: str,
    test_shape: str | None = None,
    ld_preload: str | None = None,
    use_custom_kernel: bool = False,
    verbosity_level: int = 0,
    log_stdout: bool = False,
) -> bool:
    """Run benchdnn in correctness mode and report pass/fail.

    Args:
        benchdnn_bin: Path to the benchdnn executable.
        driver: benchdnn driver name (for example, ``--matmul``).
        test_shape: Shape string for the benchdnn CLI call.
        ld_preload: Value for ``LD_PRELOAD`` pointing to custom kernel .so.
        use_custom_kernel: Whether to set ``DNNL_USE_CUSTOM_KERNEL=1``.
        verbosity_level: Value written to ``ONEDNN_VERBOSE``.
        log_stdout: Whether to emit benchdnn stdout through ``logging.info``.

    Returns:
        ``True`` when benchdnn stdout indicates a pass, otherwise ``False``.

    Raises:
        ValueError: If ``test_shape`` is empty.
        RuntimeError: If benchdnn exits with a non-zero return code.
    """
    if not test_shape:
        raise ValueError("check_correctness_benchdnn requires a non-empty test_shape")

    shape_args = test_shape.split()
    benchdnn_env = _setup_benchdnn_env_vars(
        ld_preload=ld_preload,
        use_custom_kernel=use_custom_kernel,
        onednn_verbosity_level=verbosity_level,
    )

    impl_args = ["--global-impl=sycl:ref:any"]

    cmd = [benchdnn_bin, "--mode=c", driver, "--reset", "--perf-template=csv", *impl_args, *shape_args]
    benchdnn_result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=benchdnn_env,
    )

    if log_stdout:
        logging.info("[benchdnn] stdout:\n%s", benchdnn_result.stdout)

    if benchdnn_result.returncode != 0:
        raise RuntimeError(
            "benchdnn correctness execution failed with non-zero exit code "
            f"{benchdnn_result.returncode}. "
            f"cmd={cmd}, "
            f"stderr={benchdnn_result.stderr.strip()}"
        )

    return parse_benchdnn_correctness_check(benchdnn_result.stdout)


def measure_runtime_benchdnn(
    benchdnn_bin: str,
    driver: str,
    test_shape: str | None = None,
    ld_preload: str | None = None,
    use_custom_kernel: bool = False,
    verbosity_level: int = 0,
    log_stdout: bool = False,
    output: list[float] | None = None,
) -> list[float]:
    """Run benchdnn in performance mode and return measured runtime.

    Args:
        benchdnn_bin: Path to the benchdnn executable.
        driver: benchdnn driver name (for example, ``--matmul``).
        test_shape: Shape string for the benchdnn CLI call.
        ld_preload: Value for ``LD_PRELOAD`` pointing to custom kernel .so.
        use_custom_kernel: Whether to set ``DNNL_USE_CUSTOM_KERNEL=1``.
        verbosity_level: Value written to ``ONEDNN_VERBOSE``.
        log_stdout: Whether to emit benchdnn stdout through ``logging.info``.
        output: Optional list that, when provided, is extended in-place with
            the returned runtime values.

    Returns:
        A single-item list containing ``avg_time`` parsed from benchdnn output.

    Raises:
        ValueError: If ``test_shape`` is empty or perf output parsing fails.
        RuntimeError: If benchdnn exits with a non-zero return code.
    """
    if not test_shape:
        raise ValueError("measure_runtime_benchdnn requires a non-empty test_shape")

    shape_args = test_shape.split()
    benchdnn_env = _setup_benchdnn_env_vars(
        ld_preload=ld_preload,
        use_custom_kernel=use_custom_kernel,
        onednn_verbosity_level=verbosity_level,
    )

    impl_args = []
    if use_custom_kernel:
        impl_args.append("--global-impl=sycl:ref:any")

    cmd = [benchdnn_bin, "--mode=p", driver, "--reset", "--perf-template=csv", *impl_args, *shape_args]
    benchdnn_result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=benchdnn_env,
    )

    if log_stdout:
        logging.info("[benchdnn] stdout:\n%s", benchdnn_result.stdout)

    if benchdnn_result.returncode != 0:
        raise RuntimeError(
            "benchdnn performance execution failed with non-zero exit code "
            f"{benchdnn_result.returncode}. "
            f"cmd={cmd}, "
            f"stderr={benchdnn_result.stderr.strip()}"
        )

    perf_stats = parse_benchdnn_perf_stats(benchdnn_result.stdout)

    avg_time = perf_stats["avg_time"]
    runtimes = [float(avg_time)]
    if output is not None:
        output.extend(runtimes)
    return runtimes
