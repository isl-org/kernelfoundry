"""Celery task for building container images from task specifications.

Builds or retrieves container images for custom kernel testing environments,
handling both user-provided Dockerfiles and default images.
"""

import os
import logging
import time
import traceback
from dataclasses import asdict
from celery import shared_task
from pathlib import Path

from kernelfoundry.eval_pipeline.utils.tmp_dir import TemporaryDirectory
from kernelfoundry.eval_pipeline.task import Task, BuildResult, ProcessResult
from kernelfoundry.eval_pipeline.utils.container import get_container_runtime

from kernelfoundry.eval_pipeline.utils.sysinfo import get_worker_info


@shared_task(track_started=True)
def build_image(task: dict) -> dict:
    """Build or retrieve a container image for a task.

    Celery task entry point. Builds a custom container image from a Dockerfile
    (if provided) or retrieves the default image for the specified language and
    GPU architecture. Returns serialized build results.

    Args:
        task: Encoded Task dict as produced by ``Task.encode()``.

    Returns:
        Dict mapping language to dict mapping gpu_arch to build result dict
        containing worker_info, process result, and encoded artifacts.
    """
    result = _build_image(Task.decode(task))
    # Serialize: {language: {gpu_arch: build_result_dict}}
    return {
        language: {
            gpu_arch: {
                "worker_info": build_result.worker_info,
                "result": asdict(build_result.result) if build_result.result is not None else None,
                "artifacts": build_result.artifacts.encode() if build_result.artifacts is not None else None,
            }
            for gpu_arch, build_result in gpu_arch_map.items()
        }
        for language, gpu_arch_map in result.items()
    }


def _build_image(task: Task) -> dict[str, dict[str, BuildResult]]:
    """Builds a container image for the given custom task.

    Returns:
        dict of format {language: {gpu_arch: BuildResult}} where BuildResult.result.output_data
        contains 'image' on success.
    """
    worker_info = get_worker_info()

    language = task.config["language"].lower()
    # For now we just support a single image that works for all GPU architectures for custom Dockerfiles
    # TODO allow multiple user dockerfiles with different GPU architectures in the future
    gpu_arch = "all"
    environment_build_timeout = task.config.get("environment_build_timeout", 20 * 60)
    container_registry = task.config.get("paths", {}).get("container_registry")
    allowed_container_registries = task.config.get("paths", {}).get("allowed_container_registries")

    with TemporaryDirectory(prefix="build_image_", delete=not os.environ.get("DBG_KEEP_TEMPDIRS", False)) as task_dir:
        task_data_dir = Path(task_dir) / "task_data"
        task.task_data.to_disk(output_dir=task_data_dir)

        dockerfile_path = task_data_dir / "environment" / "Dockerfile"
        container_runtime = get_container_runtime()(
            registry=container_registry,
            allowed_registries=allowed_container_registries,
        )

        try:
            tic = time.time()
            if dockerfile_path.exists():
                image, result, result_msg = container_runtime.build_image(
                    environment_path=dockerfile_path.parent, timeout=environment_build_timeout
                )
            else:
                image, result, result_msg = container_runtime.get_default_image(
                    language=language, gpu_arch=gpu_arch, timeout=environment_build_timeout
                )

            toc = time.time()
            logging.info(f"[Build Image] Image build/setup took {toc - tic:.2f} seconds")
        except Exception as e:
            build_result = BuildResult(
                worker_info=worker_info,
                result=ProcessResult(
                    returncode=-1,
                    stdout="",
                    stderr="",
                    message=f"Image build failed with error: {str(e)}\n{traceback.format_exc()}",
                ),
            )
            logging.error(f"[Build Image] Image build failed with error: {str(e)}\n{traceback.format_exc()}")
            return {language: {gpu_arch: build_result}}

        if image is None:
            process_result = (
                ProcessResult.create(result, message=result_msg)
                if result is not None
                else ProcessResult(
                    returncode=-1,
                    stdout="",
                    stderr="",
                    message=result_msg or "Failed to build or find a suitable container image",
                )
            )
            return {language: {gpu_arch: BuildResult(worker_info=worker_info, result=process_result)}}

        if image is not None and task.config.get("use_queue", True):
            result, msg = image.push(timeout=environment_build_timeout)
            process_result = ProcessResult.create(result, message=msg)
            if process_result.returncode != 0:
                return {language: {gpu_arch: BuildResult(worker_info=worker_info, result=process_result)}}

        return {
            language: {
                gpu_arch: BuildResult(
                    worker_info=worker_info,
                    result=ProcessResult(
                        returncode=0,
                        stdout="",
                        stderr="",
                        message="Image built successfully",
                        output_data={"image": image.tag},
                    ),
                )
            }
        }
