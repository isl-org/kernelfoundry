import pytest
import numpy as np
from typing import Any
import pyopencl as cl
from pathlib import Path
from kernelfoundry import TestBase  # Tests must derive from TestBase

# Fixtures provided via conftest.py from kernelfoundry: use_reference, profile_store, measure_runtime, ocl_queue, request


def get_data_on_device(
    inp_shape: tuple, use_reference_arg: bool, ocl_queue_arg: cl.CommandQueue
) -> tuple[tuple[Any, ...], np.ndarray]:
    """
    Creates random test data and moves it to device

    Args:
        inp_shape (tuple): Shape of the input matrices (m, k, n)
        use_reference_arg (bool): Whether to use the reference implementation
        ocl_queue_arg (cl.CommandQueue): OpenCL command queue for device transfers (ignored if use_reference_arg is True)

    Returns:
        tuple[tuple[Any, ...], np.ndarray]: A tuple containing the input arguments (on device if not using reference) and the expected result as a numpy array
    """
    # create numpy arrays of size n on host
    m, k, n = inp_shape
    a = (np.random.randn(m, k) - 0.5).astype(np.float32)
    b = (np.random.randn(k, n) - 0.5).astype(np.float32)
    # for testing with respect to standard matmul:
    expected = np.matmul(a, b)

    # create device buffers and upload data
    a_flat = a.flatten()
    b_flat = b.flatten()
    a_ocl = cl.Buffer(ocl_queue_arg.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a_flat)
    b_ocl = cl.Buffer(ocl_queue_arg.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b_flat)
    out_ocl = cl.Buffer(ocl_queue_arg.context, cl.mem_flags.WRITE_ONLY, size=expected.nbytes)
    return (a_ocl, b_ocl, out_ocl, np.int32(m), np.int32(k), np.int32(n)), expected


def initialize_gemm_kernel(filename: str, queue: cl.CommandQueue):
    """Helper function to initialize a GEMM OCL kernel (either from kernel fixture or for correctness test)"""
    # load OCL kernel from text
    _CL_SRC = Path(__file__).parent / filename
    program = cl.Program(queue.context, _CL_SRC.read_text()).build()
    knl = program.gemm
    # convert into function that takes device buffers and matrix dimensions
    # Kernel signature: gemm(A, B, C, M, K, N) with global work size (N, M)
    return lambda a, b, c, m, k, n: knl(queue, (n, m), None, a, b, c, np.int32(m), np.int32(k), np.int32(n))


@pytest.fixture(scope="session")
def kernel(use_reference, ocl_queue):
    """Returns a callable kernel(a, b, c, m, k, n) that runs on the GPU."""
    if use_reference:
        filename = "gemm_reference.cl"
    else:
        filename = "gemm_kernel.cl"
    return initialize_gemm_kernel(filename, ocl_queue)


class TestGemmOCL(TestBase):
    """A custom task for gemm kernels using pyopencl."""

    def build(self, gpu_arch) -> list[str]:
        # No compiled extension needed; pyopencl is used directly.
        return []

    @pytest.mark.parametrize("_run", range(3))
    def test_correctness_wrt_pytorch(self, kernel, use_reference, ocl_queue, _run):
        # generate random data
        args, expected = get_data_on_device((128, 256, 512), use_reference, ocl_queue)

        # run kernel
        kernel(*args)

        # move result back to host
        _, _, out_buf, m, _, n = args
        result_flat = np.empty(m * n, dtype=np.float32)
        cl.enqueue_copy(ocl_queue, result_flat, out_buf)
        result = result_flat.reshape((m, n))

        # check shape
        assert result.shape == expected.shape, f"Output shape {result.shape} != expected shape {expected.shape}"
        # check values
        assert np.allclose(result, expected, rtol=1e-4, atol=1e-4), "Matrix multiply result is incorrect."

    @pytest.mark.parametrize("_run", range(3))
    def test_correctness_wrt_reference(self, kernel, use_reference, ocl_queue, _run):
        # generate random data
        args, _ = get_data_on_device((128, 256, 512), use_reference, ocl_queue)

        # run kernel
        kernel(*args)

        # move result back to host
        _, _, out_buf, m, _, n = args
        result_flat = np.empty(m * n, dtype=np.float32)
        cl.enqueue_copy(ocl_queue, result_flat, out_buf)
        result = result_flat.reshape((m, n))

        # inititalize reference
        reference_kernel_function = initialize_gemm_kernel("gemm_reference.cl", ocl_queue)
        # run reference
        reference_kernel_function(*args)

        # move reference result back to host
        reference_result_flat = np.empty(m * n, dtype=np.float32)
        cl.enqueue_copy(ocl_queue, reference_result_flat, out_buf)
        expected = reference_result_flat.reshape((m, n))

        # check values
        assert np.allclose(result, expected, rtol=1e-4, atol=1e-4), "Matrix multiply result is incorrect."

    @pytest.mark.performance
    @pytest.mark.parametrize(
        "shape",
        [(128, 256, 512), (256, 512, 1024)],
        ids=lambda shape: "x".join(map(str, shape)),
    )
    def test_benchmark(self, use_reference, kernel, ocl_queue, measure_runtime, shape):
        """Benchmark: times only the GPU kernel dispatch — no host<->device copies."""
        args, _ = get_data_on_device(shape, use_reference, ocl_queue)
        measure_runtime(
            kernel,
            args=args,
            sync_fn=ocl_queue.finish,
            auto_replicate_inputs_size=0,
        )


if __name__ == "__main__":
    """Optional build when running this file directly during development."""
    task = TestGemmOCL()
    task.build(TestBase.get_machine_gpu_arch())
