import os
import pytest
import torch
from pathlib import Path
from kernelfoundry import TestBase  # Tests must derive from TestBase

# The documentation and implementation of the fixtures can be found in kernelfoundry/conftest.py


# Markers for annotating the reference implementation for the LLM prompt
# [REFERENCE_START]
def reference_kernel(a, b):
    # Reference implementation of the kernel
    return torch.matmul(a, b)


# [REFERENCE_END]


@pytest.fixture(scope="session")
def device():
    return torch.device("xpu")


@pytest.fixture(scope="session")
def kernel(use_reference):
    if use_reference:  # use_reference is True if --ref is passed as argument to pytest
        return reference_kernel
    else:
        import matmul_kernel

        return matmul_kernel.sycl_matmul


def get_data(size, device):
    # data generation for correctness tests and benchmarking
    torch.manual_seed(42)
    a = torch.randn(size, size).to(device)
    b = torch.randn(size, size).to(device)
    c = reference_kernel(a, b)
    return a, b, c


class TestMatrixMultiplication(TestBase):
    """A custom task for matrix multiplication kernels."""

    def build(self, gpu_arch) -> list[str]:
        """Builds the matmul kernel (using helper from TestBase)"""
        artifacts = self.compile_torch_extension(
            extension_name="matmul_kernel",
            src="matrix_mul_kernel.sycl",
            output_dir=Path(__file__).parent,
            gpu_arch=gpu_arch,
            backend="icpx",
        )
        return artifacts

    def test_correctness_wrt_reference(self, kernel, device):
        """Tests the correctness of the kernel against the reference implementation."""
        a, b, c = get_data(1000, device)
        result = kernel(a, b)
        assert torch.allclose(result, c, rtol=1e-4, atol=1e-4), "Matrix multiplication result is incorrect."

    def test_correctness_identity(self, kernel, device):
        """Tests the correctness of the kernel when multiplying with an identity matrix."""
        a, _, _ = get_data(1000, device)
        b = torch.eye(a.shape[0], device=device)
        result = kernel(a, b)
        assert torch.allclose(result, a), "Multiplication with identity matrix failed."

    @pytest.mark.performance
    @pytest.mark.parametrize("size", [512, 1024, 2048])
    def test_benchmark(self, size, kernel, device, torch_profile):
        """Benchmark the runtime using the torch_profile fixture.
        The runtime measurements are expected to be in milliseconds!
        """
        a, b, _ = get_data(size, device)

        torch_profile(kernel, device, args=(a, b))


if __name__ == "__main__":
    # Optional build when running this file directly during development
    task = TestMatrixMultiplication()
    task.build(TestBase.get_machine_gpu_arch())
