import os
import pytest
import torch
import random
import numpy as np
from functools import partial
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F

from kernelfoundry.custom_test import CustomTest  # Tests must derive from Task
from kernelfoundry.testing import all_close_with_slack, cosine_similarity

# [REFERENCE_START]
REFERENCE_PLACEHOLDER
# [REFERENCE_END]

NUMBER_ITERS = 5
# correctness tests depend on shape mismatch test
CORRECTNESS_DEPENDENCY = ["TestKernelBench::test_output_shapes_match"]
# # performance test depends on correctness tests -> handled by Task executer
# PERFORMANCE_DEPENDENCIES = [f"TestKernelBench::test_cosine_similarity[{i}]" for i in range(NUMBER_ITERS)] + [
#     f"TestKernelBench::test_all_close[{i}]" for i in range(NUMBER_ITERS)
# ]


@pytest.fixture(scope="session")
def device():
    if torch.xpu.is_available():
        return torch.device("xpu")
    else:
        return torch.device("cuda")


def set_seed(seed: int):
    """Set seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def set_fp32_precision(precision: str):
    if torch.cuda.is_available():
        if precision == "ieee":
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.conv.fp32_precision = "ieee"
        elif precision == "tf32":
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        else:
            raise ValueError(f"Unsupported precision: {precision}")


@pytest.fixture()
def input_tensors(device):
    inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_inputs()]
    return inputs


@pytest.fixture(scope="session")
def reference_model(device):
    set_seed(42)  # set seed for model initialization
    model = Model(*get_init_inputs()).to(device)
    return partial(model.forward, fn=module_fn)


@pytest.fixture(scope="session")
def kernel_model(device, use_reference, template_args_wrapper):
    set_seed(42)  # set seed for model initialization
    model = Model(*get_init_inputs()).to(device)
    if use_reference:
        return partial(model.forward, fn=module_fn)
    else:
        import pytorch_operation

        return partial(model.forward, fn=template_args_wrapper(pytorch_operation.forward))


class TestKernelBench(CustomTest):
    """Testing KernelBench tasks for correctness (shape mismatch and precision) and profile"""

    ### remove for triton ###
    def build(self, gpu_arch) -> list[str]:
        # Task provides a helper to compile the torch extension

        src = list(Path(__file__).parent.glob("pytorch_operation.*"))[0]
        backend = "icpx" if src.suffix == ".sycl" else "torch"  # use icpx if sycl to enable esimd
        artifacts = self.compile_torch_extension(
            extension_name="pytorch_operation",
            src=src,
            output_dir=Path(__file__).parent,
            gpu_arch=gpu_arch,
            backend=backend,
        )
        return artifacts

    ### end remove ###

    @pytest.mark.dependency()
    def test_output_shapes_match(self, reference_model, kernel_model, input_tensors):
        # test whether feeding inputs to model works
        print("[Correctness check] Feeding inputs to standard pytorch model (reference)...")
        out_ref = reference_model(*input_tensors)
        print(
            "[Correctness check] Successfully fed inputs standard pytorch model (reference)",
        )
        print("[Correctness check] Feeding inputs to model with custom kernel ...")
        out_kernel = kernel_model(*input_tensors)
        print(
            "[Correctness check] Successfully fed inputs to new model with custom kernel.",
        )

        # Check output shape mismatch
        assert (
            out_ref.shape == out_kernel.shape
        ), f"Output shape mismatch: Expected {out_ref.shape}, got {out_kernel.shape}\n"

    @pytest.mark.dependency(depends=CORRECTNESS_DEPENDENCY)
    @pytest.mark.parametrize("iteration", range(NUMBER_ITERS))  # Run the test 5 times
    def test_all_close(self, reference_model, kernel_model, input_tensors, iteration):
        set_fp32_precision("tf32")
        out_ref = reference_model(*input_tensors)
        set_fp32_precision("ieee")
        out_ref_ieee = reference_model(*input_tensors)
        out_kernel = kernel_model(*input_tensors)
        assert all_close_with_slack(out_ref, out_kernel) or all_close_with_slack(
            out_ref_ieee, out_kernel
        ), f"Output value mismatch in trial {iteration}: Max diff: {torch.max(torch.abs(out_ref - out_kernel)).item():.6f}, Avg diff: {torch.mean(torch.abs(out_ref - out_kernel)).item():.6f}"

    @pytest.mark.dependency(depends=CORRECTNESS_DEPENDENCY)
    @pytest.mark.parametrize("iteration", range(NUMBER_ITERS))  # Run the test 5 times
    def test_cosine_similarity(self, reference_model, kernel_model, input_tensors, iteration):
        set_fp32_precision("tf32")
        out_ref = reference_model(*input_tensors)
        set_fp32_precision("ieee")
        out_ref_ieee = reference_model(*input_tensors)
        out_kernel = kernel_model(*input_tensors)
        assert cosine_similarity(out_ref, out_kernel) or cosine_similarity(
            out_ref_ieee, out_kernel
        ), f"Output value mismatch in trial {iteration}: Cosine similarity between reference and kernel output too large."

    @pytest.mark.performance
    def test_benchmark(self, kernel_model, input_tensors, device, torch_profile):
        """Profile the kernel performance using the "torch_profile" fixture"""
        set_fp32_precision("tf32")
        torch_profile(kernel_model, device, args=input_tensors)
