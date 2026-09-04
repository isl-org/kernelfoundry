"""Base class for custom task tests from which all tests must derive.

Derive a test class from TestBase in your ``task.py`` file and define tests as
methods on that class.
"""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil

import pytest

__all__ = ["TestBase"]


class TestBase(ABC):
    """Base class from which kernel task tests must derive.

    Example:
        The following shows how to define a test by deriving from TestBase::

            # This is a partial example; see templates for a complete task.
            from pathlib import Path
            import torch
            import pytest
            from kernelfoundry import TestBase

            # ... pytest fixtures for device/kernel/data are omitted for brevity.

            class TestRelu(TestBase):
                def build(self, gpu_arch) -> list[str]:
                    return self.compile_torch_extension(
                        extension_name="relu_kernel",
                        src="relu_kernel.sycl",
                        output_dir=Path(__file__).parent,
                        gpu_arch=gpu_arch,
                    )

                def test_correctness(self, data, kernel, device):
                    x, y = data
                    assert torch.allclose(kernel(x), y, rtol=1e-4, atol=1e-4)

                @pytest.mark.performance
                def test_benchmark(self, data, kernel, device, measure_runtime_torch):
                    # measure_runtime_torch fixture is provided by kernelfoundry/conftest.py
                    x, _ = data
                    measure_runtime_torch(kernel, device, args=(x,))

    """

    def build(self, gpu_arch, **buildtime_params) -> list[str]:
        """Builds the kernel and returns a list of build artifacts required for running the tests.

        Returns:
            list[str]: List of paths to the build artifacts. The artifacts must not
                be outside of the task folder structure.
        """
        return []

    def build_reference(self, gpu_arch, **buildtime_params) -> list[str]:
        """Builds the reference code and returns a list of build artifacts required for running the tests.

        Returns:
            list[str]: list of paths to the build artifacts. The artifacts must not
                be outside of the task folder structure.
        """
        return []

    @classmethod
    def _has_build_override(cls) -> bool:
        """Returns whether the build step is implemented for this task.

        Returns:
            bool: True if the build step is implemented, False otherwise.
        """
        return cls.build is not TestBase.build

    @staticmethod
    def compile_torch_extension(
        extension_name: str,
        src: str | Path,
        output_dir: str | Path,
        gpu_arch: str,
        timeout: int = 120,
        backend: str = "torch",
    ) -> list[str]:
        """Compiles the source file to a PyTorch extension.

        Args:
            extension_name (str): Name of the PyTorch extension to build.
            src (str): Path to the source file.
            output_dir (str): Directory to store the compiled outputs.
            gpu_arch (str): GPU architecture string.
            timeout (int): Timeout for each compilation step in seconds.
            backend (str): The backend to use for compilation. This is either 'torch' (default) or 'icpx'.

        Returns:
            list[str]: List of paths to the compiled extensions.
        """
        from kernelfoundry.compiler import IcpxCompiler, TorchCompiler, find_built_extension

        assert gpu_arch, "gpu_arch must be specified"
        assert backend in ["icpx", "torch"], f"Unsupported backend: {backend}"

        Compiler = IcpxCompiler if backend == "icpx" else TorchCompiler

        # temporary directory to store build files
        with TemporaryDirectory() as build_dir:
            compiler = Compiler(
                extension_name=extension_name,
                src=str(src),
                build_dir=build_dir,
                gpu_arch=gpu_arch,
                timeout=timeout,
                verbose=True,
            )
            status = compiler.compile()
            if status["returncode"] != 0:
                if status.get("phase") == "load":
                    summary = (
                        f"Compilation succeeded but the built extension '{extension_name}' could not be "
                        f"imported. The compiler and linker reported no errors; the module itself would "
                        f"not load. A missing PyInit_{extension_name} usually means the sources built "
                        f"nothing that registers the module -- an empty or unreplaced EVOLVE block, for "
                        f"instance -- rather than a toolchain fault."
                    )
                else:
                    summary = f"Compilation failed for extension '{extension_name}'"
                raise RuntimeError(f"{summary}: {status['stdout']}\n{status['stderr']}")

            # copy the extension to the output directory. The suffix is .so on Linux, .pyd on Windows.
            extension_path = find_built_extension(build_dir, extension_name)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy(extension_path, output_dir)
        return [(Path(output_dir) / extension_path.name).as_posix()]

    @staticmethod
    def get_machine_gpu_arch() -> str:
        """Returns the GPU architecture string of the local machine.

        Returns:
            str: GPU architecture string.
        """
        arch = None
        from kernelfoundry.eval_pipeline.utils.sysinfo import get_nvidia_compute_capabilities, select_intel_gpu

        intel_gpu = select_intel_gpu()
        if intel_gpu is not None:
            _, device_id, _ = intel_gpu
            arch = device_id
        else:
            nvidia_caps = get_nvidia_compute_capabilities()
            if nvidia_caps:
                arch = " ".join(nvidia_caps)
        if arch is None:
            raise ValueError("Cannot determine GPU architecture of the local machine.")
        return arch

    @staticmethod
    def validate():
        """Hook for task-specific validation of the task definition.

        Override in a subclass to assert anything that must hold before a job starts -- for
        example that required data files are present, or that a build script is executable.
        The default implementation does nothing.
        """
        pass
