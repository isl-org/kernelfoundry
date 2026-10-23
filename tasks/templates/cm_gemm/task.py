import pytest
import numpy as np
from pathlib import Path
from clops import cl
from kernelfoundry import TestBase  # Tests must derive from TestBase

# Fixtures provided via conftest.py from kernelfoundry: use_reference, profile_store, measure_runtime, request


def load_gemm_kernel(filename: str, cwd: Path):
    """Load and JIT-compile a CM GEMM kernel via clops.

    Args:
        filename: CM source file name (relative to task directory).
        cwd: Path to the task directory.

    Returns:
        A callable run(t_a, t_b, t_c, m, k, n) that enqueues the kernel.
    """
    src = (cwd / filename).read_text()
    # -cmc triggers Intel NEO driver's built-in CM compiler.
    # CM kernel source is passed as an OpenCL program; the driver JIT-compiles it.
    km = cl.kernels(src, "-cmc")

    def run(t_a, t_b, t_c, m: int, k: int, n: int):
        # GWS = (M,): one CM thread per output row
        km.enqueue("gemm", [int(m)], [1], t_a, t_b, t_c, int(m), int(k), int(n))

    return run


@pytest.fixture(scope="session")
def kernel(use_reference):
    """Returns a callable run(t_a, t_b, t_c, m, k, n) that dispatches the CM GEMM kernel."""
    cwd = Path(__file__).parent
    filename = "gemm_reference.cm" if use_reference else "gemm_kernel.cm"
    return load_gemm_kernel(filename, cwd)


def make_device_data(m: int, k: int, n: int):
    """Create random float32 host arrays and upload them to device tensors.

    Returns:
        (t_a, t_b, t_c, expected): device tensors and numpy expected output.
    """
    a = np.random.randn(m, k).astype(np.float32)
    b = np.random.randn(k, n).astype(np.float32)
    expected = np.matmul(a, b)

    t_a = cl.tensor(a)
    t_b = cl.tensor(b)
    t_c = cl.tensor(np.zeros((m, n), dtype=np.float32))
    return t_a, t_b, t_c, expected


class TestGemmCM(TestBase):
    """Custom task for GEMM kernels using CM (C for Metal) via clops."""

    # has_build_step is False (set in config.yaml): clops compiles JIT, no AOT step needed.

    @pytest.mark.parametrize("_run", range(3))
    def test_correctness_wrt_numpy(self, kernel, _run):
        """Verify CM GEMM output matches numpy matmul."""
        m, k, n = 128, 256, 512
        t_a, t_b, t_c, expected = make_device_data(m, k, n)

        kernel(t_a, t_b, t_c, m, k, n)
        cl.finish()

        result = t_c.numpy().reshape(m, n)
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        assert np.allclose(result, expected, rtol=1e-4, atol=1e-4), "GEMM result is incorrect."

    @pytest.mark.parametrize("_run", range(3))
    def test_correctness_wrt_reference(self, kernel, use_reference, _run):
        """Verify optimized kernel output matches the reference implementation."""
        if use_reference:
            pytest.skip("Skipping reference-vs-reference comparison.")

        m, k, n = 128, 256, 512
        t_a, t_b, t_c, _ = make_device_data(m, k, n)

        kernel(t_a, t_b, t_c, m, k, n)
        cl.finish()
        result = t_c.numpy().reshape(m, n)

        # Run reference on same inputs
        cwd = Path(__file__).parent
        ref_kernel = load_gemm_kernel("gemm_reference.cm", cwd)
        t_c_ref = cl.tensor(np.zeros((m, n), dtype=np.float32))
        ref_kernel(t_a, t_b, t_c_ref, m, k, n)
        cl.finish()
        expected = t_c_ref.numpy().reshape(m, n)

        assert np.allclose(result, expected, rtol=1e-4, atol=1e-4), "Optimized result differs from reference."

    @pytest.mark.performance
    @pytest.mark.parametrize("m, k, n", [(128, 256, 512), (256, 512, 1024)])
    def test_benchmark(self, kernel, measure_runtime, m, k, n):
        """Benchmark: time only the CM kernel dispatch, no host<->device copies."""
        t_a, t_b, t_c, _ = make_device_data(m, k, n)
        measure_runtime(
            kernel,
            args=(t_a, t_b, t_c, m, k, n),
            sync_fn=cl.finish,
            auto_replicate_inputs_size=0,
        )


if __name__ == "__main__":
    """Optional quick smoke-test when running this file directly during development."""
    cwd = Path(__file__).parent
    ref_kernel = load_gemm_kernel("gemm_reference.cm", cwd)
    m, k, n = 64, 64, 64
    t_a, t_b, t_c, expected = make_device_data(m, k, n)
    ref_kernel(t_a, t_b, t_c, m, k, n)
    cl.finish()
    result = t_c.numpy().reshape(m, n)
    print("Max abs error:", np.abs(result - expected).max())
