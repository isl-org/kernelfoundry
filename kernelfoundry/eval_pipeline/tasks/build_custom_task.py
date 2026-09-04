"""Celery task for building custom kernel tasks in a temporary directory, optionally inside a container."""

import os
import sys
import asyncio
from celery import shared_task
from pathlib import Path
import json
import logging
import time
import traceback

from kernelfoundry import PACKAGE_ROOT
from kernelfoundry.eval_pipeline.utils.tmp_dir import TemporaryDirectory
from kernelfoundry.eval_pipeline.task import Task, ProcessResult, BuildResult
from kernelfoundry.eval_pipeline.utils.memory_file_map import MemoryFileMap
from kernelfoundry.eval_pipeline.utils.subprocess import robust_subprocess_run
from kernelfoundry.eval_pipeline.utils.env_helper import safe_copy_env
from kernelfoundry.eval_pipeline.utils.container import get_container_runtime, select_container_image

from kernelfoundry.eval_pipeline.utils.sysinfo import get_worker_info


@shared_task(track_started=True)
def build_custom_task(task: dict) -> dict:
    """Build a custom kernel task and return the encoded result.

    Celery task entry point. Decodes the task, runs the build pipeline
    (optionally inside a container), and returns the encoded result dict.

    Args:
        task: Encoded Task dict as produced by ``Task.encode()``.

    Returns:
        Encoded Task dict with build results attached.
    """
    return _build_custom_task(Task.decode(task)).encode()


def _build_custom_task(task: Task) -> Task:
    worker_info = get_worker_info()

    timeout = task.config.get("build_timeout")
    if timeout is None:
        timeout = 120  # seconds
    output_inactivity_timeout = task.config.get("build_output_inactivity_timeout", 150)
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
            build_result = BuildResult(worker_info=worker_info)
            build_result.result = ProcessResult(
                returncode=-1,
                stdout="",
                stderr="",
                message=f"No entry for container image for language '{language}' and GPU architecture '{gpu_arch}'",
            )
            task.build_result = build_result
            return task

    env = safe_copy_env(
        add_vars={"KERNELFOUNDRY_BUILD": "1"},
        extend_pythonpath=(
            ["/kernelfoundry", "/kernelfoundry/kernelfoundry"] if use_container else [PACKAGE_ROOT.parent]
        ),
        src={} if use_container else None,  # if using container, start with a clean env
    )

    with TemporaryDirectory(
        prefix="build_custom_task_", delete=not os.environ.get("DBG_KEEP_TEMPDIRS", False)
    ) as task_dir:
        logging.info(f"[Build Custom Task] Building in temporary directory: {task_dir}")
        task_data_dir = Path(task_dir) / "task_data"

        inputs = {
            "gpu_arch": task.config.get("gpu_arch"),
            "hyperparameters": task.hyperparameters,
        }
        config_json_path = Path(task_dir) / "input.json"

        task.task_data.to_disk(output_dir=task_data_dir)

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
                build_result = BuildResult(worker_info=worker_info)
                build_result.result = ProcessResult(
                    returncode=-1,
                    stdout="",
                    stderr="",
                    message=f"Build failed with error: {str(e)}\n{traceback.format_exc()}",
                )
                task.build_result = build_result
                return task

            if image is None:
                build_result = BuildResult(worker_info=worker_info)
                build_result.result = ProcessResult(
                    returncode=-1,
                    stdout="",
                    stderr="",
                    message=f"Failed to find a suitable container image. Requested image is {container_image}",
                )
                task.build_result = build_result
                return task

            toc = time.time()
            logging.info(f"Image setup took {toc - tic:.2f} seconds")

            workspace_dir = Path("/workspace")
            container_run_args = image.default_run_args(workdir=workspace_dir / "task_data", workspace_dir=task_dir)

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
                build_result = BuildResult(worker_info=worker_info)
                build_result.result = ProcessResult.create(result, result_msg)
                task.build_result = build_result
                return task

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

        # Mapping from build function to the corresponding result attribute
        build_result_mapping = {"custom": "build_result", "reference": "build_result_reference"}
        build_function_mapping = {"custom": "build", "reference": "build_reference"}

        for mode in ["custom", "reference"]:
            # skip if no build function for this mode OR if test_custom is False
            if (mode == "custom" and (not task.has_build_step or not task.config.get("test_custom", True))) or (
                mode == "reference"
                and (not task.has_reference_build_step or not task.config.get("test_reference", True))
            ):
                # NOTE: reference build is stored for later trials and does not to be rebuilt even if required for testing
                continue

            # Get the appropriate result attribute for this build function
            result_attr = build_result_mapping[mode]
            setattr(task, result_attr, BuildResult(worker_info=worker_info))
            current_build_result = getattr(task, result_attr)

            # try build in subprocess
            try:
                with open(config_json_path, "w") as f:
                    json.dump(inputs, f)

                workspace_config_json_path = workspace_dir / "input.json"
                workspace_artifacts_json_path = workspace_dir / "artifacts.json"

                cmd = [
                    "python" if use_container else sys.executable,
                    "-u",  # force the stdout and stderr streams to be unbuffered to capture all errors
                    "-m",
                    "kernelfoundry.eval_pipeline.tasks.build_custom_task_target",
                    "--config",
                    workspace_config_json_path.as_posix(),
                    "--build_function",
                    build_function_mapping[mode],
                    "--output",
                    workspace_artifacts_json_path.as_posix(),
                ]

                if use_container:
                    container_run_args["workdir"] = workspace_task_data_dir.as_posix()
                    container_run_args["env_vars"] = {**env}
                    coroutine = image.run_cmd(
                        cmd,
                        timeout=timeout,
                        output_inactivity_timeout=output_inactivity_timeout,
                        container_run_args=container_run_args,
                    )
                else:
                    coroutine = robust_subprocess_run(
                        cmd,
                        timeout=timeout,
                        output_inactivity_timeout=output_inactivity_timeout,
                        cwd=task_data_dir.as_posix(),
                        env=env,
                    )

                tic = time.time()
                result, result_msg = asyncio.run(coroutine)
                current_build_result.result = ProcessResult.create(result, message=result_msg)
                toc = time.time()
                logging.info(f"Build '{mode}' took {toc - tic:.2f} seconds")

                if result.returncode != 0:
                    current_build_result.result.message = "Build failed with non-zero return code"
                    # Continue to reference if test_reference = True
                    if task.config.get("test_reference", True):
                        continue
                    else:
                        break

                artifacts_json_path = Path(task_dir) / "artifacts.json"
                with open(artifacts_json_path, "r") as f:
                    artifacts = json.load(f)

                if artifacts.get("artifacts") is None:
                    current_build_result.result.message = "No artifacts were produced during build"
                    continue

                file_map = MemoryFileMap()
                for file_path in artifacts.get("artifacts", []):
                    if os.path.isabs(file_path):
                        full_path = Path(file_path)
                        if use_container:
                            full_path = Path(task_dir) / full_path.relative_to(workspace_dir)
                    else:
                        full_path = task_data_dir / file_path
                    if full_path.exists():
                        st = full_path.stat()
                        with open(full_path, "rb") as f:
                            relative_path = Path(file_path).relative_to(workspace_task_data_dir).as_posix()
                            file_map[relative_path] = f.read()
                            file_map.meta_map[relative_path] = {"mode": st.st_mode & 0o777, "mtime": st.st_mtime}
                current_build_result.artifacts = file_map
                current_build_result.result.message = "Build completed successfully"

            except Exception as e:
                current_build_result.result = ProcessResult(
                    returncode=-1,
                    stdout="",
                    stderr="",
                    message=f"Build failed with error: {str(e)}\n{traceback.format_exc()}",
                )

    return task
