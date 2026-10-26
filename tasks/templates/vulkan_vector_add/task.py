import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from kernelfoundry import TestBase  # Tests must derive from TestBase
from kernelfoundry.compiler import compile_glsl_to_spirv
from kernelfoundry.eval_pipeline.utils.vulkan_support import create_vulkan_compute_context

# Fixtures provided via conftest.py from kernelfoundry: use_reference, profile_store, request

LOCAL_SIZE_X = 256
NUM_BUFFERS = 3  # a, b, out, each bound as an SSBO at bindings 0/1/2
PUSH_CONSTANT_SIZE = 4  # single uint32 `n` (element count)


# Markers for annotating the reference implementation for the LLM prompt
# [REFERENCE_START]
def reference_kernel(a, b):
    # Reference implementation of the kernel
    return a + b


# [REFERENCE_END]


@pytest.fixture(scope="session")
def vulkan_ctx(use_reference):
    """Builds the Vulkan compute pipeline for the vector_add shader."""
    if use_reference:
        return None
    spv_path = Path(__file__).parent / "vector_add_kernel.spv"
    return create_vulkan_compute_context(
        spv_path, num_storage_buffers=NUM_BUFFERS, push_constant_size=PUSH_CONSTANT_SIZE
    )


def get_data_on_device(n: int, use_reference_arg: bool, vulkan_ctx_arg):
    """
    Creates random test data and moves it to device

    Args:
        n (int): Size of 1D tensor
        use_reference_arg (bool): Whether to use the reference implementation
        vulkan_ctx_arg (VulkanCtx): Vulkan context for device transfers (ignored if use_reference_arg is True)

    Returns:
        tuple[tuple[Any, ...], np.ndarray]: A tuple containing the input arguments (on device if not using
        reference) and the expected result as a numpy array
    """
    a = (np.random.randn(n) - 0.5).astype(np.float32)
    b = (np.random.randn(n) - 0.5).astype(np.float32)
    if use_reference_arg:
        a_t, b_t = torch.from_numpy(a).to("xpu"), torch.from_numpy(b).to("xpu")
        return (a_t, b_t), reference_kernel(a_t, b_t).cpu().numpy()
    else:
        expected = a + b
        a_buf = vulkan_ctx_arg.make_buffer(a.nbytes)
        b_buf = vulkan_ctx_arg.make_buffer(b.nbytes)
        out_buf = vulkan_ctx_arg.make_buffer(a.nbytes)
        vulkan_ctx_arg.upload(a_buf, a)
        vulkan_ctx_arg.upload(b_buf, b)
        return (a_buf, b_buf, out_buf, n), expected


@pytest.fixture(scope="session")
def kernel(use_reference, vulkan_ctx):
    """Returns a callable that runs the kernel. For the reference path this takes (a, b) tensors
    and returns the result; for the Vulkan path this dispatches and writes into out_buf in place.
    """
    if use_reference:
        return reference_kernel
    else:
        return lambda a_buf, b_buf, out_buf, n: vulkan_ctx.submit(
            [a_buf, b_buf, out_buf], n, push_constants=struct.pack("I", n), local_size_x=LOCAL_SIZE_X
        )


class TestVectorAddVulkan(TestBase):
    """A custom task for vector_add kernels using the Vulkan compute API."""

    def build(self, gpu_arch) -> list[str]:
        """Compiles the GLSL compute shader to SPIR-V via glslangValidator."""
        src = Path(__file__).parent / "vector_add_kernel.comp"
        spv = compile_glsl_to_spirv(src)
        return [str(spv)]

    @pytest.mark.parametrize("_run", range(5))
    def test_correctness(self, kernel, use_reference, vulkan_ctx, _run):
        args, expected = get_data_on_device(1_000_000, use_reference, vulkan_ctx)

        if use_reference:
            result = kernel(*args).cpu().numpy()
        else:
            a_buf, b_buf, out_buf, n = args
            kernel(a_buf, b_buf, out_buf, n)
            result = vulkan_ctx.download(out_buf)

        assert result.shape == expected.shape, f"Output shape {result.shape} != expected shape {expected.shape}"
        assert np.allclose(result, expected, rtol=1e-4, atol=1e-4), "Vector add result is incorrect."

    @pytest.mark.performance
    @pytest.mark.parametrize("shape", [1_000_000, 5_000_000, 10_000_000])
    def test_benchmark(self, use_reference, kernel, vulkan_ctx, measure_runtime, shape):
        """Benchmark: times only the GPU kernel dispatch — no host<->device copies."""
        args, _ = get_data_on_device(shape, use_reference, vulkan_ctx)
        # kernel() only submits (non-blocking); sync_fn waits once per batch of inner-loop
        # iterations, so the CPU can queue multiple dispatches instead of a full GPU
        # round-trip on every call.
        sync_fn = torch.xpu.synchronize if use_reference else vulkan_ctx.wait
        measure_runtime(
            kernel,
            args=args,
            sync_fn=sync_fn,
            auto_replicate_inputs_size=0,
        )


if __name__ == "__main__":
    # Optional build when running this file directly during development
    task = TestVectorAddVulkan()
    task.build(TestBase.get_machine_gpu_arch())
