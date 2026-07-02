"""Celery task for pulling container images on remote workers."""

import logging
import traceback
from dataclasses import asdict
from celery import shared_task

from kernelfoundry.eval_pipeline.task import ProcessResult
from kernelfoundry.eval_pipeline.utils.container import get_container_runtime

from kernelfoundry.eval_pipeline.utils.sysinfo import get_worker_info


@shared_task(track_started=True)
def pull_image(
    image_id: str,
    registry: str | None = None,
    allowed_container_registries: list[str] | None = None,
    timeout: int | None = None,
) -> dict:
    """Pull a container image on the worker.

    Celery task entry point. Pulls a container image by ID or name and returns
    metadata about the result.

    Args:
        image_id: The image ID or fully-qualified image name to pull.
        registry: The container registry to use for pulling the image.
        allowed_container_registries: List of allowed registries to pull from.
        timeout: Maximum time in seconds to wait for the pull to complete.

    Returns:
        Dict with keys:
            - image_id: The pulled image ID (or None on failure)
            - hostname: Worker hostname where image was pulled
            - process_result: Encoded ProcessResult dict with return code and messages
    """
    image_id_out, hostname, process_result = _pull_image(
        image_id,
        registry=registry,
        allowed_container_registries=allowed_container_registries,
        timeout=timeout,
    )
    return {
        "image_id": image_id_out,
        "hostname": hostname,
        "process_result": asdict(process_result),
    }


def _pull_image(
    image_id: str,
    registry: str | None = None,
    allowed_container_registries: list[str] | None = None,
    timeout: int | None = None,
) -> tuple[str | None, str, ProcessResult]:
    """Pulls a container image by ID or name on the worker.

    Args:
        image_id: The image ID or fully-qualified image name to pull.
        registry: The container registry to use for pulling the image.
        timeout: The maximum time in seconds to wait for the pull to complete.

    Returns:
        A tuple of (image_id or None on failure, worker hostname, ProcessResult).
    """
    worker_info = get_worker_info()
    hostname = worker_info.get("hostname", "")
    container_runtime = get_container_runtime()(
        registry=registry,
        allowed_registries=allowed_container_registries,
    )

    try:
        image, result, result_msg = container_runtime.pull_image(image_id, timeout=timeout)
    except Exception as e:
        return (
            None,
            hostname,
            ProcessResult(
                returncode=-1,
                stdout="",
                stderr="",
                message=f"Image pull failed with error: {str(e)}\n{traceback.format_exc()}",
            ),
        )

    if image is None:
        process_result = ProcessResult.create(result, message=result_msg)
        logging.warning(f"[Pull Image] Failed to pull image {image_id}: {result_msg}")
        return None, hostname, process_result

    logging.info(f"[Pull Image] Successfully pulled image {image_id} on {hostname}")
    return image.image_id, hostname, ProcessResult.create(result, message=result_msg)
