"""Helpers for configuring logging and recording raw process results."""

import logging
from pathlib import Path

from kernelfoundry.eval_pipeline.task import BuildResult, ProcessResult, TestResult


def log_process_result(
    name: str,
    result: ProcessResult,
    kernel_uuid: str,
    iteration: int,
    branch: int,
    worker_info: dict | None = None,
    agent_session_id: str | None = None,
) -> None:
    """Log stdout, stderr, and metadata for a single process execution.

    Args:
        name: Event name emitted through the custom RAW logging level.
        result: Process result containing return code, streams, and optional metadata.
        kernel_uuid: Unique identifier for the kernel associated with this result.
        iteration: Iteration identifier recorded alongside the raw log payload.
        branch: Branch value recorded alongside the raw log payload.
        worker_info: Optional worker metadata to include in the structured log payload.
        agent_session_id: Optional agent session identifier to include in the structured log payload.
    """
    if not result:
        return
    output_string = ""
    if result.returncode is not None:
        output_string += f"Return code: {result.returncode}\n"
    if result.message is not None:
        output_string += f"Message: {result.message}\n"
    if result.stdout is not None and result.stdout.strip() != "":
        output_string += "------ Standard Output ------\n"
        output_string += result.stdout + "\n"
    if result.stderr is not None and result.stderr.strip() != "":
        output_string += "------ Standard Error ------\n"
        output_string += result.stderr + "\n"

    log_extra = {
        "data": {
            "log": output_string,
            "output_data": result.output_data,
            "worker_info": worker_info,
            "trial": iteration,  # keep trial for compatibility with existing log schema
            "version": branch,  # keep version for compatibility with existing log schema
            "iteration": iteration,
            "branch": branch,
        },
        "kernel_uuid": kernel_uuid,
        "agent_session_id": agent_session_id,
    }

    logging.raw(name, extra=log_extra)
    print(output_string)  # also print to console for easier debugging


def log_build_test_result(
    result: BuildResult | TestResult,
    kernel_uuid: str | None = None,
    agent_session_id: str | None = None,
    iteration: int | None = None,
    branch: int | None = None,
    prefix: str | None = None,
) -> None:
    """Log all raw subprocess outputs contained in a build or test result.

    Args:
        result: Build or test result whose subprocess outputs should be emitted.
        kernel_uuid: Unique identifier for the kernel associated with this result.
        iteration: Iteration recorded alongside each emitted raw log payload.
        branch: Branch (formerly version) recorded alongside each emitted raw log payload.
        prefix: Optional prefix used to name emitted log events. Defaults to
            ``build`` or ``test`` based on the result type.
    """
    try:
        worker_info = result.worker_info
    except AttributeError:
        worker_info = {}
    if isinstance(result, BuildResult):
        prefix = prefix or "build"
        log_process_result(
            name=prefix,
            result=result.result,
            kernel_uuid=kernel_uuid,
            iteration=iteration,
            branch=branch,
            worker_info=worker_info,
            agent_session_id=agent_session_id,
        )
    elif isinstance(result, TestResult):
        prefix = prefix or "test"
        log_process_result(
            name=f"{prefix}_correctness",
            result=result.correctness_result,
            kernel_uuid=kernel_uuid,
            iteration=iteration,
            branch=branch,
            worker_info=worker_info,
            agent_session_id=agent_session_id,
        )
        log_process_result(
            name=f"{prefix}_performance",
            result=result.performance_result,
            kernel_uuid=kernel_uuid,
            iteration=iteration,
            branch=branch,
            worker_info=worker_info,
            agent_session_id=agent_session_id,
        )
        for k, v in result.trace_results.items():
            log_process_result(
                name=f"{prefix}_trace_{k}",
                result=v,
                kernel_uuid=kernel_uuid,
                iteration=iteration,
                branch=branch,
                worker_info=worker_info,
                agent_session_id=agent_session_id,
            )


def setup_logging(logfile: str | Path, logging_level=None):
    """Configure the root logger with a file handler and custom RAW level.

    Args:
        logfile: Path to the log file that should receive root logger output.
            This is usually a path to the "controller.log" in the respective run directory.
        logging_level: Logging level to apply to the file handler and root logger.
            If None, the current root logger level is reused.
    """
    if logging_level is None:
        logging_level = logging.getLogger().level

    def raw_root(message, *args, **kwargs):
        logging.log(RAW, message, *args, **kwargs)

    # Define the custom level
    RAW = 25  # between INFO (20) and WARNING (30)
    logging.addLevelName(RAW, "RAW")
    logging.raw = raw_root

    # Add high-level logging functions for build and test results
    logging.build_test_result = log_build_test_result
    logging.process_result = log_process_result

    # Set up logging to console and file
    log_file = Path(logfile)
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging_level)

    # Create formatter
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    # console_handler.setFormatter(formatter)

    # Add handlers to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging_level)
