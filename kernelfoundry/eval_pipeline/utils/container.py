"""Container runtime abstraction for Docker and Podman."""

from typing import Coroutine
import asyncio
import tempfile
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from kernelfoundry.eval_pipeline.utils.subprocess import robust_subprocess_run
import kernelfoundry
from kernelfoundry.eval_pipeline.utils import sysinfo
from itertools import product
import subprocess


class Image:
    """Represents a built or resolved container image tied to a specific runtime."""

    def __init__(self, runtime: "ContainerRuntime", image_id: str, tag: str) -> None:
        self._runtime = runtime
        self.image_id = image_id
        self.tag = tag

    @property
    def runtime(self) -> "ContainerRuntime":
        """The container runtime this image is bound to."""
        return self._runtime

    @staticmethod
    def default_run_args(
        workdir: Path | str,
        workspace_dir: Path | str,
        gpus: list[int] | str | None = None,
        kernelfoundry_dir: Path | str | None = None,
    ) -> dict:
        """Returns a default set of arguments for running a container with this image building and testing kernels.

        Args:
            workdir: The working directory to set inside the container.
            workspace_dir: The host directory to mount as /workspace inside the container,
                containing the kernel source and test files.
            gpus: Optional list of GPU indices to make available inside the container,
                or "all" to make all GPUs available. If not specified, no GPUs will be made available.
            kernelfoundry_dir: Optional host directory of the kernelfoundry codebase.

        Returns:
            A dictionary of arguments to pass to the runtime's :meth:`get_run_cmd` method
        """
        if kernelfoundry_dir is None:
            kernelfoundry_dir = Path(kernelfoundry.__file__).parent.parent
        args = {
            "workdir": str(workdir),
            "volumes": [(str(workspace_dir), "/workspace"), (str(kernelfoundry_dir), "/kernelfoundry", "ro")],
            "env_vars": {"PYTHONPATH": "/kernelfoundry"},
            "gpus": gpus,
        }
        return args

    def run_cmd(
        self,
        cmd: list[str] | str,
        timeout: int | None = None,
        output_inactivity_timeout: float | None = None,
        end_marker: str | None = None,
        container_run_args: dict = None,
        **kwargs,
    ) -> Coroutine:
        """Runs a command in a container from this image.

        Args:
            cmd: The command to run inside the container, as a list of strings.
            timeout: The maximum time to wait for the command to complete.
            output_inactivity_timeout: Time in seconds to wait for output before considering the process inactive.
            end_marker: Controls end-of-process monitoring. ``"pytest"`` terminates after the pytest
                summary line (10 s grace). Any other string terminates when that literal appears in
                output (2 s grace). ``None`` disables early termination.
            container_run_args: Optional dictionary of arguments to pass to the runtime's :meth:`get_run_cmd` method
                without the image argument (e.g. workdir, volumes, env_vars, gpus).
            **kwargs: Additional keyword arguments to pass to subprocess.Popen

        Returns:
            Returns a coroutine that, when awaited, runs the command in the container and returns
            a tuple of (CompletedProcess, termination_message)
        """
        container_run_args = container_run_args or {}
        container_cmd = self._runtime.get_run_cmd(self, **container_run_args)
        if isinstance(cmd, str):
            run_cmd = container_cmd + ["sh", "-c", cmd]
            # run_cmd = " ".join(container_cmd + [cmd])
        else:
            run_cmd = container_cmd + cmd
        return robust_subprocess_run(
            run_cmd,
            timeout=timeout,
            output_inactivity_timeout=output_inactivity_timeout,
            end_marker=end_marker,
            **kwargs,
        )

    def push(self, timeout: int | None = None) -> tuple[subprocess.CompletedProcess, str | None]:
        """Pushes the image to its registry.

        Args:
            timeout: The maximum time to wait for the push to complete.

        Returns:
            A tuple of (CompletedProcess, termination_message)
        """
        if not self._runtime.registry:
            return subprocess.CompletedProcess(args=[], returncode=0), None
        result, result_msg = asyncio.run(
            robust_subprocess_run(
                [self._runtime._cmd, "push", self.tag],
                timeout=timeout,
                output_inactivity_timeout=timeout,
            )
        )
        return result, result_msg

    def __repr__(self) -> str:
        return f"{type(self._runtime).__name__}Image({self.image_id!r}, {self.tag!r})"


class ContainerRuntime(ABC):
    """Abstract base class for container runtimes."""

    def __init__(
        self, registry: str | None = None, gpu_type: str | None = None, allowed_registries: list[str] | None = None
    ) -> None:
        """
        Args:
            registry: Optional container registry prefix used when pulling images
                      (e.g. 'registry.example.com:5000'). When set, image names that are not absolutely
                      qualified with the registry will be automatically prefixed when
                      passed to :meth:`get_image`.
            gpu_type: Optional GPU type to use when running containers (e.g. 'nvidia').
                      If not specified, the GPU type is auto-detected based on the system's hardware.
            allowed_registries: Optional list of allowed registry prefixes.
                      If set, any registry used in image names must start with one of these prefixes.
        """
        if gpu_type is None:
            intel_gpus = sysinfo.discover_intel_gpus()
            nvidia_gpu_name = sysinfo.get_nvidia_gpu_name()
            if intel_gpus:
                gpu_type = "intel"
            if nvidia_gpu_name:
                gpu_type = "nvidia"
        assert gpu_type in (None, "intel", "nvidia"), f"Unsupported gpu_type: {gpu_type}"
        self.registry = registry
        self.gpu_type = gpu_type
        self.allowed_registries = allowed_registries

    @property
    @abstractmethod
    def _cmd(self) -> str:
        """The CLI command used to invoke this runtime (e.g. 'docker' or 'podman')."""

    def build_image(
        self,
        environment_path: Path | str,
        dockerfile_path: Path | str | None = "Dockerfile",
        image_name: str | None = None,
        timeout: int | None = None,
        qualify_name: bool = True,
    ) -> tuple[Image | None, subprocess.CompletedProcess, str | None]:
        """Builds a container image from a Dockerfile.

        Args:
            environment_path: The build context directory.
            dockerfile_path: The path to the Dockerfile.
            image_name: The name to give the built image.
            timeout: The maximum time to wait for the build to complete.
            qualify_name: Whether to prepend the registry prefix to bare image names.

        Returns:
            A tuple of (Image or None if build failed, CompletedProcess, termination_message)
        """
        with tempfile.NamedTemporaryFile(suffix=".iid") as iidfile:
            cmd = [
                self._cmd,
                "build",
                "--provenance=false",
                "--sbom=false",
                "--load",
                "-f",
                str(dockerfile_path),
                "--iidfile",
                iidfile.name,
            ]
            # if no image name is provided, we will use a temporary name based on the iidfile
            temp_named = image_name is None
            if temp_named:
                image_name = f"kernelfoundry_task_image/{Path(iidfile.name).stem}"

            if qualify_name:
                image_name = self._qualify(image_name)
            cmd += ["-t", image_name]
            cmd += ["."]
            # build the image
            result, result_msg = asyncio.run(
                robust_subprocess_run(
                    cmd,
                    timeout=timeout,
                    output_inactivity_timeout=timeout,
                    cwd=str(environment_path),
                )
            )
            if result.returncode == 0:
                full_id = Path(iidfile.name).read_text().strip()
                image_id = full_id.split(":")[-1]
                if temp_named:
                    # retag from the temp iidfile-based name to the image-id-based name
                    new_image_name = f"kernelfoundry_task_image/{image_id}"
                    if qualify_name:
                        new_image_name = self._qualify(new_image_name)

                    cmd = [
                        self._cmd,
                        "tag",
                        image_name,
                        new_image_name,
                    ]
                    tag_result, tag_result_msg = asyncio.run(
                        robust_subprocess_run(
                            cmd,
                            timeout=timeout,
                            output_inactivity_timeout=timeout,
                            cwd=str(environment_path),
                        )
                    )
                    if tag_result.returncode != 0:
                        return None, tag_result, tag_result_msg

                    # remove the temp tag now that the image-id-based tag exists
                    asyncio.run(
                        robust_subprocess_run(
                            [self._cmd, "rmi", image_name],
                            timeout=timeout,
                            output_inactivity_timeout=timeout,
                            cwd=str(environment_path),
                        )
                    )
                    image_name = new_image_name

                image = Image(self, image_id, tag=image_name)
                return image, result, None
            else:
                print("Image build failed with result:", result)
                return None, result, result_msg

    def get_image(self, image_name: str) -> tuple[Image | None, subprocess.CompletedProcess, str | None]:
        """Looks up an existing image by name, optionally prefixed with the registry.

        Args:
            image_name: The name of the container image. If a registry was provided
                        to the constructor and the name is not already fully qualified,
                        the registry prefix is prepended automatically.

        Returns:
            An :class:`Image` object for the latest matching image, or ``None`` if
            no such image exists locally.
        """
        qualified_name = self._qualify(image_name)
        result, _ = asyncio.run(
            robust_subprocess_run(
                [self._cmd, "images", "--format", "{{.ID}}", qualified_name],
                timeout=60,
            )
        )
        ids = result.stdout.strip().splitlines()
        if not ids:
            return None
        return Image(self, ids[0], tag=qualified_name)

    def get_default_image(
        self, language: str, gpu_arch: str, timeout: int | None = None
    ) -> tuple[Image | None, subprocess.CompletedProcess | None, str | None]:
        """Gets the default image for the given language and GPU architecture.

        Args:
            language: The programming language for which to get the default image.
            gpu_arch: The GPU architecture for which to get the default image.
        Returns:
            A tuple of (Image or None if no suitable image could be found or built, CompletedProcess, termination_message)
        """
        timeout = timeout or 60 * 60
        language_list = [language.lower()]
        gpu_arch_list = [gpu_arch.lower(), "all"]
        if language.lower() in ["ocl", "triton"]:
            language_list.append("sycl")  # ocl is supported with the sycl image

        for lang, gpu in product(language_list, gpu_arch_list):
            image_name = f"kernelfoundry/{lang}-{gpu}"
            image = self.get_image(image_name)
            if image:
                return image, None, None

            # try pulling the image from the registry
            image, pull_result, pull_result_msg = self.pull_image(image_name, timeout=timeout)
            if image:
                return image, pull_result, pull_result_msg

            # try building the default image
            import kernelgen.docker  # TODO

            docker_images_root = Path(kernelgen.docker.__file__).parent
            docker_build_env_dir = docker_images_root / f"kernelfoundry_{lang}-{gpu}"
            if docker_build_env_dir.is_dir():
                image, build_result, build_result_msg = self.build_image(
                    environment_path=docker_build_env_dir, image_name=image_name, timeout=timeout
                )
                return image, build_result, build_result_msg
        raise RuntimeError(f"No default image found for language={language} gpu_arch={gpu_arch}")

    def get_run_cmd(
        self,
        image: "Image",
        workdir: str | None = None,
        volumes: list[tuple[str, str, str] | tuple[str, str]] | None = None,
        env_vars: dict[str, str] | None = None,
        gpus: list[int] | str | None = None,
        reserved_host_memory_kb: int | None = 4 * 2**20,  # default to reserving 4GB for the host
    ) -> list[str]:
        """Returns the base container run command for the given image.

        Args:
            image: The :class:`Image` to build the run command for.
            workdir: Optional working directory to set inside the container.
            volumes: Optional list of tuples specifying volume mounts. Each tuple can be either
                (host_path, container_path) or (host_path, container_path, mode).
                E.g. [("/host/data", "/container/data"), ("/host/config", "/container/config", "ro")].
            env_vars: Optional dictionary of environment variables to set inside the container.
            gpus: Optional list of GPU indices to make available inside the container,
                or ``"all"`` to make all GPUs available.
            reserved_host_memory_kb: Amount of system memory in kB to leave available for the
                host. When set, the container memory limit is passed as total system
                memory minus this reserved amount. Default is 4GB. None disables the memory limit.

        Returns:
            A list of strings forming the base command, e.g.
            ``['docker', 'run', '--rm', '<image_id>']``.
        """
        cmd = [self._cmd, "run", "--rm", "--init"]
        if env_vars:
            for key, value in env_vars.items():
                cmd += ["-e", f"{key}={value}"]
        if gpus is not None:
            assert self.gpu_type is not None, "gpu_type must be set to use GPU options"
            if self.gpu_type == "intel":
                device_args = ["--device", "/dev/dri:/dev/dri"]
                if gpus != "all":
                    device_args += ["-e", f'''ONEAPI_DEVICE_SELECTOR="*:{','.join(str(i) for i in gpus)}"''']

            elif self.gpu_type == "nvidia":
                if gpus == "all":
                    device_args = ["--gpus", "all"]
                else:
                    gpu_indices = ",".join(str(i) for i in gpus)
                    device_args = ["--gpus", f'"device={gpu_indices}"']
            cmd += device_args

        if workdir:
            cmd += ["-w", workdir]

        if volumes:
            for volume in volumes:
                if len(volume) == 2:
                    host_path, container_path = volume
                    cmd += ["-v", f"{host_path}:{container_path}"]
                elif len(volume) == 3:
                    host_path, container_path, mode = volume
                    cmd += ["-v", f"{host_path}:{container_path}:{mode}"]
                else:
                    raise ValueError(f"Invalid volume spec: {volume!r}")

        if reserved_host_memory_kb is not None:
            if reserved_host_memory_kb < 0:
                raise ValueError("reserved_host_memory_kb must be non-negative")
            total_system_memory_kb = sysinfo.get_total_system_memory_kb()
            if total_system_memory_kb <= 0:
                raise RuntimeError("Unable to determine total system memory in kB")
            container_memory_limit_kb = total_system_memory_kb - reserved_host_memory_kb
            if container_memory_limit_kb <= 0:
                raise ValueError("reserved_host_memory_kb must be smaller than total system memory")
            cmd += ["--memory", f"{container_memory_limit_kb}k"]

        cmd.append(image.image_id)

        return cmd

    def pull_image(
        self, image_id: str, timeout: int | None = None, prepend_registry: bool = True
    ) -> tuple[Image | None, subprocess.CompletedProcess, str | None]:
        """Pulls a container image by ID or name from a registry.

        Args:
            image_id: The image ID or fully-qualified image name to pull.
            timeout: The maximum time to wait for the pull to complete.

        Returns:
            A tuple of (Image or None if pull failed, CompletedProcess, termination_message)
        """
        if prepend_registry:
            image_id = self._qualify(image_id)
        result, result_msg = asyncio.run(
            robust_subprocess_run(
                [self._cmd, "pull", image_id],
                timeout=timeout,
                output_inactivity_timeout=timeout,
            )
        )
        if result.returncode == 0:
            return self.get_image(image_id), result, result_msg
        else:
            return None, result, result_msg

    def _qualify(self, image_name: str) -> str:
        """Prepends the registry prefix when one is configured and the name is bare.

        If ``allowed_registries`` is set and the image name is already absolutely
        qualified (i.e. it already starts with a known registry prefix), this method
        verifies that the registry matches one of the allowed registries.  A
        :class:`ValueError` is raised when the check fails.
        """
        first_component = image_name.split("/", maxsplit=1)[0]
        is_fully_qualified = "/" in image_name and (
            "." in first_component or ":" in first_component or first_component == "localhost"
        )

        if is_fully_qualified:
            allowed = set(self.allowed_registries or [])
            if self.registry:
                allowed.add(self.registry)

            if allowed and not any(image_name == r or image_name.startswith(f"{r}/") for r in allowed):
                raise ValueError(
                    f"Image '{image_name}' uses a registry that is not allowed. "
                    f"Allowed registries: {sorted(allowed)}"
                )
            return image_name

        if self.registry:
            return f"{self.registry}/{image_name}"
        return image_name


class Docker(ContainerRuntime):
    """Container runtime backed by Docker."""

    @property
    def _cmd(self) -> str:
        return "docker"


class Podman(Docker):
    """Container runtime backed by Podman (drop-in Docker replacement)."""

    @property
    def _cmd(self) -> str:
        return "podman"


def get_container_runtime() -> type[ContainerRuntime]:
    """Returns the appropriate container runtime class for the current system.

    Checks for ``docker`` first, then ``podman``. Raises :class:`RuntimeError`
    if neither is available.

    Returns:
        The :class:`Docker` or :class:`Podman` class.
    """
    import shutil

    for cls, cmd in [(Docker, "docker"), (Podman, "podman")]:
        if shutil.which(cmd) is not None:
            return cls

    raise RuntimeError("No container runtime found: neither 'docker' nor 'podman' is available on PATH.")


def select_container_image(container_image_dict: dict, language: str, gpu_arch: str) -> str:
    """Returns the container image for the given language and GPU architecture.

    Args:
        container_image_dict (dict): Dictionary mapping languages to GPU architectures to container images.
        language (str): The programming language.
        gpu_arch (str): The GPU architecture.

    Returns:
        str: The container image.
    """
    tmp_lang = language
    if language not in container_image_dict and len(container_image_dict) > 0:
        logging.warning(
            f"Container image differs from language, probably reference language != kernel language, using first image"
        )
        tmp_lang = list(container_image_dict.keys())[0]

    container_image = container_image_dict.get(tmp_lang, {}).get(gpu_arch) or container_image_dict.get(
        tmp_lang, {}
    ).get("all")
    return container_image
