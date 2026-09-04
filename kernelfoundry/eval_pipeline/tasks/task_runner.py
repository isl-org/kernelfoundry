"""Task execution dispatcher for managing kernel compilation and testing jobs.

Provides the TaskRunner class which coordinates task execution either locally
or via Celery queue, handling build and test steps with optional containerization.
"""

from kernelfoundry.eval_pipeline.tasks.build_custom_task import build_custom_task
from kernelfoundry.eval_pipeline.tasks.build_image import build_image
from kernelfoundry.eval_pipeline.tasks.pull_image import pull_image
from kernelfoundry.eval_pipeline.tasks.test_custom_task import test_custom_task
from kernelfoundry.eval_pipeline.tasks.test_sleep import test_sleep
from kernelfoundry.eval_pipeline.task import Task, BuildResult, ProcessResult, TestResult
from kernelfoundry.eval_pipeline.utils.container import select_container_image
from kernelfoundry.eval_pipeline.utils.gpu_specs import ARCH_TO_SPECS, ARCH_TO_PCI_DEVICE_IDS
from kernelfoundry.eval_pipeline.utils.sysinfo import discover_intel_gpus

import logging
import threading

# languages with supported compilers
BUILD_SUPPORTED_LANGUAGES = ["cuda", "sycl", "ocl", "triton"]


class TaskRunner:
    """Task execution manager and dispatcher."""

    app = None
    # Local mode can evaluate multiple candidates concurrently from different threads.
    # Serialize GPU test execution to avoid concurrent runs on a single device.
    _local_test_lock = threading.Lock()

    @classmethod
    def init(cls, use_queue: bool = False, gpu_arch=None):
        if use_queue:
            from kernelfoundry.eval_pipeline.celery_app import celery_app
            from celery.exceptions import TimeoutError

            cls.app = celery_app
        else:
            cls.app = None

        if gpu_arch is None:
            return

        archs = gpu_arch if isinstance(gpu_arch, (list, tuple)) else str(gpu_arch).split(",")
        archs = [a.strip() for a in archs]

        # assert that arch is known and has a corresponding entry in specs dict
        for arch in archs:
            assert (
                arch in ARCH_TO_SPECS
            ), f"Unknown gpu_arch {arch!r}: no entry in ARCH_TO_SPECS. Available: {ARCH_TO_SPECS.keys()}"

        # check whether a worker is listening to the gpu_arch queue
        if use_queue:
            active_queues = {
                q["name"]
                for worker_queues in (cls.app.control.inspect(timeout=5.0).active_queues() or {}).values()
                for q in worker_queues
            }
            for arch in archs:
                queue = f"test_custom_task_{arch}"
                assert queue in active_queues, (
                    f"No worker is currently listening on queue {queue!r} for gpu_arch={arch!r}. "
                    f"Active queues: {sorted(active_queues) or '(none)'}"
                )
        # check if the local gpu corresponds to the specified gpu_arch (only for Intel GPUs)
        else:
            detected_ids = {device_id.lower() for _, device_id, _ in discover_intel_gpus()}
            if detected_ids:
                for arch in archs:
                    valid_ids = ARCH_TO_PCI_DEVICE_IDS.get(arch)
                    if valid_ids is None:
                        continue  # not a known Intel arch (e.g. a CUDA arch); nothing to check locally
                    assert detected_ids & set(valid_ids), (
                        f"Specified gpu_arch={arch!r} does not match any GPU installed on this "
                        f"machine (detected PCI device ids: {sorted(detected_ids)})"
                    )

    @classmethod
    def build_custom_task(cls, task: Task) -> Task:
        """Builds the custom task if it has a build step.

        Args:
            task (Task): The custom task to build.

        Returns:
            Task: The custom task with the build artifacts included.
        """
        if not (task.has_build_step or task.has_reference_build_step):
            logging.warning("Custom task has no build step. Skipping build.")
            return task

        timeout = task.config.get("build_timeout")
        if timeout is None:
            timeout = 120  # seconds
        language = task.config.get("language", "").lower()
        assert language in BUILD_SUPPORTED_LANGUAGES, f"Unsupported language for custom task: {language}"
        gpu_arch = task.config.get("gpu_arch")
        use_container = task.config.get("use_container", False)

        queue = f"build_custom_task_{language}"
        queue_override = task.config.get("queue_override_build")

        if use_container and cls.app:
            # Make sure to pull the image before testing to avoid doing it inside the test task which would cause timeouts
            container_image = task.config.get("container_image")
            if isinstance(container_image, dict):
                container_image = select_container_image(container_image, language, gpu_arch)
            elif isinstance(container_image, str):
                pass  # nothing to do here
            if container_image is None:
                raise ValueError(f"No container_image specified for {language} and {gpu_arch}")

            image_id, hostname, process_result = cls.pull_image(
                container_image,
                registry=task.config.get("paths", {}).get("container_registry"),
                allowed_container_registries=task.config.get("paths", {}).get("allowed_container_registries"),
                queue=queue_override if queue_override else queue,
                timeout=task.config.get("environment_pull_timeout") or 3600,
            )
            if image_id is None:
                raise RuntimeError(
                    f"Failed to pull container image {container_image} for testing: {process_result.message} - {process_result.stderr}"
                )
            # make sure the test runs on the same worker we were pulling the image
            queue = f"worker_{hostname}"

        custom_task_encoded = task.encode()

        if cls.app:
            max_retries = 2
            logging.debug(f"Submitting build task to queue {queue} from thread {threading.get_ident()}")
            for attempt in range(max_retries + 1):
                try:
                    ans = cls.app.send_task(
                        "kernelfoundry.eval_pipeline.tasks.build_custom_task.build_custom_task",
                        args=(custom_task_encoded,),
                        queue=queue_override if queue_override else queue,
                        retries=1,  # worker retries
                        soft_time_limit=timeout + 10,
                        time_limit=timeout + 30,
                    ).get(timeout=timeout + 40)
                    break
                except TimeoutError:
                    if attempt < max_retries:
                        logging.warning(
                            f"Build task timed out (attempt {attempt + 1}/{max_retries + 1}), retrying in thread {threading.get_ident()}"
                        )
                    else:
                        raise
            logging.debug(f"Received build task result for custom task in thread {threading.get_ident()}")
        else:
            ans = build_custom_task.run(custom_task_encoded)

        ans = Task.decode(ans)
        return ans

    @classmethod
    def build_image(cls, task: Task) -> dict[str, dict[str, BuildResult]]:
        """Builds a container image for the custom task.

        Args:
            task (Task): The custom task whose environment image should be built.

        Returns:
            dict of format {language: {gpu_arch: BuildResult}} where BuildResult.result.output_data
            contains 'image_id' on success.

        Raises:
            RuntimeError: If the image build fails.
        """
        timeout = task.config.get("environment_build_timeout", 20 * 60)
        language = task.config.get("language", "").lower()
        assert language in BUILD_SUPPORTED_LANGUAGES, f"Unsupported language for build_image: {language}"

        queue_override = task.config.get("queue_override_build_image")

        custom_task_encoded = task.encode()

        if cls.app:
            logging.debug(f"Submitting build_image task to queue build_image from thread {threading.get_ident()}")
            ans = cls.app.send_task(
                "kernelfoundry.eval_pipeline.tasks.build_image.build_image",
                args=(custom_task_encoded,),
                queue=queue_override if queue_override else f"build_image",
                soft_time_limit=timeout + 10,
                time_limit=timeout + 30,
            ).get(timeout=timeout + 40)
            logging.debug(f"Received build_image result from thread {threading.get_ident()}")
        else:
            ans = build_image.run(custom_task_encoded)

        # Decode serialized {language: {gpu_arch: build_result_dict}} back to BuildResult objects
        return {
            language: {gpu_arch: BuildResult.decode(br) for gpu_arch, br in gpu_arch_map.items()}
            for language, gpu_arch_map in ans.items()
        }

    @classmethod
    def pull_image(
        cls,
        image_id: str,
        registry: str | None,
        allowed_container_registries: list[str] | None,
        queue: str,
        timeout: int = 3600,
    ) -> tuple[str | None, str, ProcessResult]:
        """Pulls a container image on a worker.

        Args:
            image_id: The image ID or fully-qualified image name to pull.
            registry: The container registry to use for pulling the image.
            queue: The queue to submit the pull task to.
            timeout: The maximum time in seconds to wait for the pull to complete.

        Returns:
            A tuple of (image_id or None on failure, worker hostname, ProcessResult).
        """
        if cls.app:
            logging.debug(f"Submitting pull_image task to queue {queue} from thread {threading.get_ident()}")
            ans = cls.app.send_task(
                "kernelfoundry.eval_pipeline.tasks.pull_image.pull_image",
                args=(image_id, registry, allowed_container_registries, timeout),
                queue=queue,
                soft_time_limit=timeout + 10,
                time_limit=timeout + 30,
            ).get(timeout=timeout + 40)
            logging.debug(f"Received pull_image result from thread {threading.get_ident()}")
        else:
            ans = pull_image.run(image_id, registry, allowed_container_registries, timeout)

        return (
            ans["image_id"],
            ans["hostname"],
            ProcessResult(**ans["process_result"]),
        )

    @classmethod
    def test_custom_task(cls, task: Task, gpu_arch: str = None) -> dict[str, TestResult]:
        """Executes the custom task.

        Args:
            task (Task): The custom task to execute.

        Returns:
            dict: The results of the custom task execution.
        """
        timeout = task.config.get("test_timeout", 120)  # 120 s by default
        # multiply with 5 because test_custom_task is running 5 subprocesses (correctness, performance, 3x profiler)
        timeout = timeout * 5
        use_container = task.config.get("use_container", False)

        if gpu_arch is None:
            gpu_arch = task.config.get("gpu_arch")
        language = task.config["language"].lower()

        queue = f"test_custom_task_{gpu_arch}"
        queue_override = task.config.get("queue_override_test")

        if use_container and cls.app:
            # Make sure to pull the image before testing to avoid doing it inside the test task which would cause timeouts
            container_image = task.config.get("container_image")
            if isinstance(container_image, dict):
                container_image = select_container_image(container_image, language, gpu_arch)
            elif isinstance(container_image, str):
                pass  # nothing to do here
            if container_image is None:
                raise ValueError(f"No container_image specified for {language} and {gpu_arch}")

            image_id, hostname, process_result = cls.pull_image(
                container_image,
                registry=task.config.get("paths", {}).get("container_registry"),
                allowed_container_registries=task.config.get("paths", {}).get("allowed_container_registries"),
                queue=queue_override if queue_override else queue,
                timeout=task.config.get("environment_pull_timeout") or 3600,
            )
            if image_id is None:
                raise RuntimeError(
                    f"Failed to pull container image {container_image} for testing: {process_result.message} - {process_result.stderr}"
                )
            # make sure the test runs on the same worker we were pulling the image
            queue = f"worker_{hostname}"

        custom_task_encoded = task.encode()

        if cls.app:
            max_retries = 2
            logging.debug(f"Submitting test task to queue {queue} from thread {threading.get_ident()}")
            for attempt in range(max_retries + 1):
                async_result = cls.app.send_task(
                    "kernelfoundry.eval_pipeline.tasks.test_custom_task.test_custom_task",
                    args=(custom_task_encoded,),
                    queue=queue_override if queue_override else queue,
                    # retries=1,  # worker retries
                    # soft_time_limit=timeout + 10,
                    # time_limit=timeout + 30,
                )
                ans = None
                for i in range(20):
                    tout = timeout + 40 if i == 0 else 60
                    try:
                        ans = async_result.get(timeout=tout)
                        break
                    except TimeoutError:
                        logging.warning(f"TimeoutError when getting result for task {async_result.id}, retrying...")
                if ans is None:
                    if attempt < max_retries:
                        logging.warning(
                            f"Test task {async_result.id} timed out (attempt {attempt + 1}/{max_retries + 1}), retrying in thread {threading.get_ident()}"
                        )
                    else:
                        raise TimeoutError(f"Max retries exceeded for test_custom_task {async_result.id}")
                else:
                    logging.debug(
                        f"Received test task {async_result.id} result for custom task in thread {threading.get_ident()}"
                    )
                    break
        else:
            with cls._local_test_lock:
                ans = test_custom_task.run(custom_task_encoded)

        ans = {key: TestResult.decode(test_result) for key, test_result in ans.items()}
        return ans

    @classmethod
    def test_sleep(cls, duration: float, message: str, queue: str, get_timeout: float = 30) -> str:
        """A simple test function to sleep for a given duration.

        Args:
            duration (float): The duration to sleep in seconds.
            message (str): The message to return after sleeping.
            queue (str): The queue to submit the task to.
            get_timeout (float): Timeout for getting the result.
        Returns:
            str: Hostname of the worker that executed the task.
        """
        import time

        if cls.app:
            args = (duration, message)
            async_result = cls.app.send_task(
                "kernelfoundry.eval_pipeline.tasks.test_sleep.test_sleep",
                args=args,
                queue=queue,
                # soft_time_limit=5,
                # time_limit=duration + 5,
            )

            for i in range(15):
                logging.info(f"{async_result.id} is in state {async_result.state}")
                time.sleep(0.5)
            for i in range(2):
                try:
                    ans = async_result.get(timeout=get_timeout)
                    return ans
                except TimeoutError:
                    logging.warning(f"TimeoutError when getting result for task {async_result.id}, retrying...")

        else:
            return test_sleep.run(duration, message)
