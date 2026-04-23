"""Test the auto_replicate_inputs_size feature of measure_runtime_torch."""

import pytest

from kernelfoundry.utils import performance
import torch
import time
import random

INPUTS = [
    ([torch.rand(2**18) for i in range(4)], {f"arg{i}": torch.rand(2**19) for i in range(4)}, 64 * 2**20),
    (None, {f"arg{i}": torch.rand(2**19) for i in range(4)}, 64 * 2**20),
    ([torch.rand(2**18) for i in range(4)], None, 64 * 2**20),
    ([torch.rand(2**18) for i in range(4)], {f"arg{i}": torch.rand(2**19) for i in range(4)}, 0),
]


def target_fn(*args, **kwargs):

    for x in args:
        if isinstance(x, torch.Tensor):
            target_fn.pointers.add(x.data_ptr())
    for k in sorted(kwargs):
        x = kwargs[k]
        target_fn.pointers.add(x.data_ptr())
    time.sleep(0.001 + 0.0005 * random.random())
    # return last tensor
    return x


target_fn.pointers = set()


def expected_num_pointers(args, kwargs, target_size):
    num_args = len(args) if args is not None else 0
    num_args += len(kwargs) if kwargs is not None else 0
    if target_size == 0:
        return num_args
    return num_args * (int(target_size / performance._get_size_in_bytes([args, kwargs])) + 1)


@pytest.mark.parametrize("inputs", INPUTS)
def test_auto_replicate_inputs(inputs):
    target_fn.pointers.clear()

    args, kwargs, target_size = inputs
    expected = expected_num_pointers(args, kwargs, target_size)

    _ = performance.measure_runtime_torch(
        target=target_fn,
        device=torch.device,
        args=args,
        kwargs=kwargs,
        perf_trials_min_time=0.1,
        auto_replicate_inputs_size=target_size,
    )
    actual = len(target_fn.pointers)
    assert actual == expected, f"Expected {expected} unique pointers, but got {actual}."


def test_multiple_inputs():
    """Tests multiple inputs provided by the user"""
    target_fn.pointers.clear()

    target_size = 64 * 2**20
    args = []
    args.append([torch.rand(2**18) for i in range(3)])
    args.append([torch.rand(2**18) for i in range(3)])
    kwargs = len(args) * [{}]
    expected = len(args) * len(args[0])

    _ = performance.measure_runtime_torch(
        target=target_fn,
        device=torch.device,
        args=args,
        kwargs=kwargs,
        perf_trials_min_time=0.1,
        auto_replicate_inputs_size=target_size,
    )
    actual = len(target_fn.pointers)
    assert actual == expected, f"Expected {expected} unique pointers, but got {actual}."
