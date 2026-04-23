"""Base class for custom task tests from which all tests must derive.

Derive a test class from CustomTest in your ``task.py`` file and define tests as
methods on that class.
"""

import pytest
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil

__all__ = ["CustomTest"]


class CustomTest(ABC):
    """Base class from which kernel task tests must derive.

    Example:
        The following shows how to define a test by deriving from CustomTest::

            # This is a partial example; see templates for a complete task.
            from pathlib import Path
            import torch
            import pytest
            from kernelfoundry.custom_test import CustomTest

            # ... pytest fixtures for device/kernel/data are omitted for brevity.

            class TestRelu(CustomTest):
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

    def build(self, gpu_arch) -> list[str]:
        """Builds the kernel and returns a list of build artifacts required for running the tests.

        Returns:
            list[str]: List of paths to the build artifacts. The artifacts must not
                be outside of the task folder structure.
        """
        return []

    def build_reference(self, gpu_arch) -> list[str]:
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
        return cls.build is not CustomTest.build

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
        from kernelfoundry.compiler import IcpxCompiler, TorchCompiler

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
                raise RuntimeError(f"Compilation failed: {status['stdout']}\n{status['stderr']}")

            # copy the extension to the output directory
            extension_path = Path(build_dir) / f"{extension_name}.so"
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
        from kernelfoundry.utils.sysinfo import discover_intel_gpus, get_nvidia_compute_capabilities

        intel_gpus = discover_intel_gpus()
        if intel_gpus:
            _, device_id, _ = intel_gpus[0]
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
        pass
