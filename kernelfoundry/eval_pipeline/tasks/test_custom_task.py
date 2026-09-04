"""Celery task for testing and profiling kernel implementations.

Runs correctness tests, performance benchmarks, and profiler traces on custom
and reference kernel implementations, optionally inside a container.
"""

import os
import sys
import asyncio
from pathlib import Path
from celery import shared_task
from dataclasses import asdict
import logging
import json
import traceback
import time

from kernelfoundry import PACKAGE_ROOT
from kernelfoundry.eval_pipeline.utils.tmp_dir import TemporaryDirectory
from kernelfoundry.eval_pipeline.task import Task, ProcessResult, TestResult
from kernelfoundry.eval_pipeline.utils.custom_task_helper import blocks_to_str
from kernelfoundry.eval_pipeline.profiler_command import ProfilerUnavailable, get_profilers
from kernelfoundry.eval_pipeline.utils.extract_template_parameters import find_template_parameters
from kernelfoundry.eval_pipeline.utils.subprocess import robust_subprocess_run
from kernelfoundry.eval_pipeline.utils.env_helper import safe_copy_env
from kernelfoundry.eval_pipeline.utils.container import get_container_runtime, select_container_image

from kernelfoundry.eval_pipeline.utils.sysinfo import get_worker_info


@shared_task(track_started=True)
def test_custom_task(task: dict) -> dict[str, dict]:
    """Test and profile a kernel implementation.

    Celery task entry point. Decodes the task, runs correctness tests,
    performance benchmarks, and profiler traces (optionally inside a container),
    and returns the encoded test results dict.

    Args:
        task: Encoded Task dict as produced by ``Task.encode()``.

    Returns:
        Dict mapping variant keys (e.g., 'custom', 'reference') to test result
        dicts (one per variant, encoded as dicts with correctness, performance,
        and trace results).
    """
    result = _test_custom_task(Task.decode(task))
    return {key: asdict(test_result) for key, test_result in result.items()}


def _test_custom_task(task: Task) -> dict[str, TestResult]:
    ans = {}
    worker_info = get_worker_info()

    timeout = task.config.get("test_timeout")
    if timeout is None:
        timeout = 120  # seconds
    output_inactivity_timeout = task.config.get("test_output_inactivity_timeout", 30)
    test_custom = task.config.get("test_custom", True)
    test_reference = task.config.get("test_reference", False)
    use_container = task.config.get("use_container", False)
    gpu_arch = task.config["gpu_arch"]
    assert isinstance(gpu_arch, str), f"gpu_arch must be a string, got {type(gpu_arch)}"
    language = task.config.get("language", "").lower()
    environment_build_timeout = task.config.get("environment_build_timeout", 20 * 60)
    container_registry = task.config.get("paths", {}).get("container_registry")
    allowed_container_registries = task.config.get("paths", {}).get("allowed_container_registries")
    container_image = task.config.get("container_image")
    if use_container:
        if isinstance(container_image, dict):
            container_image = select_container_image(container_image, language, gpu_arch)
        elif isinstance(container_image, str):
            pass  # nothing to do here
        if container_image is None:
            test_result = TestResult()
            test_result.correctness_result = ProcessResult(
                returncode=-1,
                stdout="",
                stderr="",
                message=f"No entry for container image for language '{language}' and GPU architecture '{gpu_arch}'",
            )
            test_result.worker_info = worker_info
            ans["container"] = test_result
            return ans

    # Extract template parameters
    code = blocks_to_str(task.blocks["EVOLVE"])
    template_parameters = find_template_parameters(code)

    logging.info(f"[Test Custom Task]")
    if template_parameters is not None:
        logging.info(f"Iterating over template parameters {template_parameters}")

    option_variants = []
    if test_custom and task.test_result is None:
        if template_parameters:
            logging.info(f"Found parameters: {template_parameters}")
            for param_comb in template_parameters:
                param_str = "_".join([str(param) for param in param_comb])
                logging.info(f"Parameter comb {param_str}, passed as --template_params={str(param_comb)}")
                option_variants.append(
                    (f"custom_{param_str}", ["--is_templated", f"--template_params={str(param_comb)}"])
                )
        else:
            option_variants.append(("custom", []))
    if test_reference and task.test_result_reference is None:
        option_variants.append(("reference", ["--ref"]))

    runtime_params = (task.hyperparameters or {}).get("runtime") or {}
    runtime_args = [f"--runtime_params={json.dumps(runtime_params)}"] if runtime_params else []

    env = safe_copy_env(
        add_vars={"KERNELFOUNDRY_TEST": "1"},
        extend_pythonpath=["/kernelfoundry"] if use_container else [PACKAGE_ROOT.parent],
        src={} if use_container else None,  # if using container, start with a clean env
    )

    with TemporaryDirectory(
        prefix="test_custom_task_", delete=not os.environ.get("DBG_KEEP_TEMPDIRS", False)
    ) as task_dir:
        logging.info(f"[Test Custom Task] Executing in temporary directory: {task_dir}")
        task_data_dir = Path(task_dir) / "task_data"  # this is where we extract the task data to

        task.task_data.to_disk(output_dir=task_data_dir)
        if task.build_result is not None and task.build_result.artifacts is not None:
            task.build_result.artifacts.to_disk(output_dir=task_data_dir)

        if task.build_result_reference is not None and task.build_result_reference.artifacts is not None:
            task.build_result_reference.artifacts.to_disk(output_dir=task_data_dir)

        # check if the task has a Dockerfile and if so, build the image
        if use_container:
            tic = time.time()
            try:
                container_runtime = get_container_runtime()(
                    registry=container_registry,
                    allowed_registries=allowed_container_registries,
                )
                image = container_runtime.get_image(container_image)
            except Exception as e:
                test_result = TestResult()
                test_result.correctness_result = ProcessResult(
                    returncode=-1,
                    stdout="",
                    stderr="",
                    message=f"Test failed with error: {str(e)}\n{traceback.format_exc()}",
                )
                test_result.worker_info = worker_info
                ans["container"] = test_result
                return ans

            if image is None:
                test_result = TestResult()
                test_result.correctness_result = ProcessResult(
                    returncode=-1,
                    stdout="",
                    stderr="",
                    message="Failed to find a suitable container image",
                )
                test_result.worker_info = worker_info
                ans["container"] = test_result
                return ans

            # TODO add support for selecting specific GPUs to the container
            toc = time.time()
            logging.info(f"Image setup took {toc - tic:.2f} seconds")

            workspace_dir = Path("/workspace")
            container_run_args = image.default_run_args(
                workdir=workspace_dir / "task_data", workspace_dir=task_dir, gpus="all"
            )

            # Get the worker info from inside the container
            cmd = [
                "python",
                "-c",
                "from kernelfoundry.eval_pipeline.utils.sysinfo import get_worker_info; import json; info = get_worker_info(); json.dump(info, open('/workspace/worker_info.json', 'w'))",
            ]
            container_run_args["env_vars"] = {**env}
            coroutine = image.run_cmd(
                cmd,
                timeout=timeout,
                output_inactivity_timeout=output_inactivity_timeout,
                container_run_args=container_run_args,
            )
            result, result_msg = asyncio.run(coroutine)
            if result.returncode != 0:
                test_result = TestResult()
                test_result.correctness_result = ProcessResult.create(result, result_msg)
                test_result.worker_info = worker_info
                ans["container_build"] = test_result
                return ans

            host_worker_info = worker_info
            with open(Path(task_dir) / "worker_info.json", "r") as f:
                worker_info = json.load(f)
                # copy the hostname from the host info and add a container field for convenience
                worker_info["container"] = worker_info["hostname"]
                worker_info["hostname"] = host_worker_info["hostname"]
                # The gpu name is just empty in some containers, so we copy it from the host info if it's missing
                if worker_info.get("gpu_name", "") == "":
                    worker_info["gpu_name"] = host_worker_info.get("gpu_name", "")
            worker_info["host_info"] = host_worker_info
        else:
            workspace_dir = Path(task_dir)
        workspace_task_data_dir = workspace_dir / "task_data"

        for key, options in option_variants:
            logging.info(f"[Test custom task] Start testing of {key}")
            test_result = TestResult()
            test_result.worker_info = worker_info
            #
            # Run the correctness tests
            #
            try:
                cmd = (
                    [
                        "python" if use_container else sys.executable,
                        "-u",  # force the stdout and stderr streams to be unbuffered to capture all errors
                        "-m",
                        "pytest",
                        "--color=no",
                        "-m",
                        "not performance",
                    ]
                    + options
                    + runtime_args
                    + ["task.py"]
                )

                if use_container:
                    container_run_args["workdir"] = workspace_task_data_dir.as_posix()
                    container_run_args["env_vars"] = {**env}  # pass the env vars to the container
                    coroutine = image.run_cmd(
                        cmd,
                        timeout=timeout,
                        output_inactivity_timeout=output_inactivity_timeout,
                        end_marker="pytest",
                        container_run_args=container_run_args,
                    )
                else:
                    coroutine = robust_subprocess_run(
                        cmd,
                        timeout=timeout,
                        output_inactivity_timeout=output_inactivity_timeout,
                        end_marker="pytest",
                        cwd=workspace_task_data_dir.as_posix(),
                        env=env,
                    )

                tic = time.time()
                result, result_msg = asyncio.run(coroutine)
                test_result.correctness_result = ProcessResult.create(result, message=result_msg)
                toc = time.time()
                logging.info(f"Correctness tests took {toc - tic:.2f} seconds")

            except Exception as e:
                test_result.correctness_result = ProcessResult(
                    returncode=-1,
                    stdout="",
                    stderr="",
                    message=f"Test failed with error: {str(e)}\n{traceback.format_exc()}",
                )

            if test_result.correctness_result.returncode != 0:
                # skip profiling if correctness tests failed
                ans[key] = test_result
                logging.info(f"Skipping profiling for '{key}' due to correctness test failure.")
                continue

            #
            # Run the profiling tests
            #
            profile_output_file = Path(workspace_dir) / f"runtimes_{key}.json"
            try:
                cmd = (
                    [
                        "python" if use_container else sys.executable,
                        "-u",
                        "-m",
                        "pytest",
                        "--color=no",
                        "-m",
                        "performance",
                        f"--performance-out={profile_output_file.as_posix()}",
                    ]
                    + options
                    + runtime_args
                    + [
                        "task.py",
                    ]
                )
                if use_container:
                    container_run_args["workdir"] = workspace_task_data_dir.as_posix()
                    container_run_args["env_vars"] = {**env}  # pass the env vars to the container
                    coroutine = image.run_cmd(
                        cmd,
                        timeout=timeout,
                        output_inactivity_timeout=output_inactivity_timeout,
                        end_marker="pytest",
                        container_run_args=container_run_args,
                    )
                else:
                    coroutine = robust_subprocess_run(
                        cmd,
                        timeout=timeout,
                        output_inactivity_timeout=output_inactivity_timeout,
                        end_marker="pytest",
                        cwd=workspace_task_data_dir.as_posix(),
                        env=env,
                    )

                tic = time.time()
                result, result_msg = asyncio.run(coroutine)
                test_result.performance_result = ProcessResult.create(result, message=result_msg)
                toc = time.time()
                logging.info(f"Performance tests took {toc - tic:.2f} seconds")

                # Read profile results from the output file (if benchmarking worked correctly)
                if test_result.performance_result.returncode == 0:
                    try:
                        with open(Path(task_dir) / f"runtimes_{key}.json", "r") as f:
                            test_result.performance_result.output_data = {"runtimes": json.load(f)}
                    except Exception as e:
                        test_result.performance_result.returncode = -1
                        test_result.performance_result.message = f"Failed to read profile output file: {str(e)}"

            except Exception as e:
                test_result.performance_result = ProcessResult(
                    returncode=-1,
                    stdout="",
                    stderr="",
                    message=f"Profile failed with error: {str(e)}\n{traceback.format_exc()}",
                )

            #
            # Run the tracing profilers
            #
            if "custom" in key and not task.config.get("eval_config", {}).get("profile_custom_model", True):
                # skip profiling custom
                ans[key] = test_result
                continue
            if "reference" in key and not task.config.get("eval_config", {}).get("profile_original_model", True):
                # skip profiling reference
                ans[key] = test_result
                continue
            try:
                # Select profiler based on language (use ref_language for reference) and user selection from config
                language = task.config.get("language", "")
                profiler_type = task.config.get("profiler_kernel")
                if "reference" in key:
                    language = task.config.get("prompt", {}).get("reference_language", language)
                    profiler_type = task.config.get("profiler_reference")

                profiler_classes = get_profilers(language, gpu_arch, profiler_type)

                # Set env var for detecting the profiler in the subprocess
                for profiler_i, (profiler_class) in enumerate(profiler_classes):
                    env_profiler = env.copy()
                    profiler_output_dir = Path(task_dir) / f"profiler_output_{key}_{profiler_i}"
                    profiler_output_dir.mkdir(parents=True, exist_ok=True)
                    if use_container:
                        profiler = profiler_class(workspace_dir / profiler_output_dir.name)
                    else:
                        profiler = profiler_class(profiler_output_dir)
                    profiler.prepare(profiler_output_dir)  # e.g. to copy scripts into container
                    env_profiler["KERNELFOUNDRY_PROFILER"] = profiler.name
                    env_profiler.update(profiler.env_vars())
                    profiler_name = f"{profiler.name}.{profiler_i:03d}"
                    cmd = (
                        [
                            "python" if use_container else sys.executable,
                            "-u",
                            "-m",
                            "pytest",
                            "--color=no",
                            "-m",
                            "performance",
                            "--itt",  # enable ITT markers
                        ]
                        + options
                        + [
                            "task.py",
                        ]
                    )
                    cmd = profiler.wrap_cmd(" ".join(cmd))
                    logging.info(cmd)

                    if use_container:
                        container_run_args["workdir"] = workspace_task_data_dir.as_posix()
                        container_run_args["env_vars"] = {**env_profiler}  # pass the env vars to the container
                        coroutine = image.run_cmd(
                            cmd,
                            timeout=timeout,
                            output_inactivity_timeout=output_inactivity_timeout,
                            end_marker=profiler.end_marker(),
                            container_run_args=container_run_args,
                        )
                    else:
                        coroutine = robust_subprocess_run(
                            cmd,
                            timeout=timeout,
                            output_inactivity_timeout=output_inactivity_timeout,
                            end_marker=profiler.end_marker(),
                            cwd=workspace_task_data_dir.as_posix(),
                            env=env_profiler,
                        )

                    tic = time.time()
                    result, result_msg = asyncio.run(coroutine)
                    test_result.trace_results[profiler_name] = ProcessResult.create(result, message=result_msg)
                    toc = time.time()
                    logging.info(f"Tracing tests pass {profiler_i} took {toc - tic:.2f} seconds")
                    try:
                        # recreate the profiler object with the directory on the host
                        profiler = profiler_class(profiler_output_dir)
                        test_result.trace_results[profiler_name].output_data = profiler.read_output()
                    except ProfilerUnavailable as e:
                        logging.error("Profiler '%s' produced no data: %s", profiler_name, e)
                        test_result.trace_results[profiler_name].output_data = {}
                        test_result.trace_results[profiler_name].message = f"Failed to read {profiler_name} output: {e}"
                    except Exception as e:
                        test_result.trace_results[profiler_name].output_data = {}
                        test_result.trace_results[profiler_name].message = (
                            f"Failed to read {profiler_name} output: {str(e)}\n{traceback.format_exc()}"
                        )

            except Exception as e:
                test_result.trace_results[profiler.name] = ProcessResult(
                    returncode=-1,
                    stdout="",
                    stderr="",
                    message=f"Trace failed with error: {str(e)}\n{traceback.format_exc()}",
                )

            ans[key] = test_result

    _log_outcome_summary(ans)
    return ans


def _log_outcome_summary(results: dict[str, TestResult]) -> None:
    """State each variant's outcome as the last thing the run prints.

    The reference is exercised after the candidate, so a run where the custom kernel failed
    would still have PASSED as the last line if the reference passed. This prints a summary of all variants at the end.
    """
    logging.info("=" * 62)
    logging.info("[Test Custom Task] Outcome by variant:")
    for key, test_result in results.items():
        correctness = test_result.correctness_result
        if correctness is None:
            verdict = "NOT RUN"
        elif correctness.returncode == 0:
            verdict = "PASSED"
        else:
            verdict = f"FAILED (exit {correctness.returncode})"
        logging.info("  %-28s %s", key, verdict)

    failed = [
        k for k, r in results.items() if r.correctness_result is not None and r.correctness_result.returncode != 0
    ]
    if failed:
        # Say it once more in the plainest terms available, because this is the line a log tail shows.
        logging.info(
            "[Test Custom Task] RESULT: %d of %d variant(s) failed: %s", len(failed), len(results), ", ".join(failed)
        )
    else:
        logging.info("[Test Custom Task] RESULT: all %d variant(s) passed", len(results))
    logging.info("=" * 62)
