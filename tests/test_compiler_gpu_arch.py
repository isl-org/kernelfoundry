"""The compiler must hand ocloc a device name it knows.

A worker that owns a Battlemage card serves ``test_custom_task_b580`` (and b70, b60), and
gpu_specs lists those SKUs as arch names, so a task config naming one routes to the right
machine. The build then failed with "Could not determine device target: b580", because the
arch string reached ocloc untouched through both TORCH_XPU_ARCH_LIST and ``-device``, and
ocloc only accepts the generic platform name.
"""

import tempfile

import pytest

from kernelfoundry.compiler import IcpxCompiler


@pytest.fixture
def build_dir():
    with tempfile.TemporaryDirectory() as path:
        yield path


@pytest.mark.parametrize("arch", ["b580", "b70", "b60"])
def test_battlemage_skus_compile_as_bmg(arch, build_dir):
    assert IcpxCompiler("ext", "kernel.sycl", build_dir, arch).gpu_arch == "bmg"


@pytest.mark.parametrize("arch", ["bmg", "lnl", "ptl", "dg2", "pvc", "0xe20b", "Ampere", ""])
def test_other_archs_are_passed_through(arch, build_dir):
    """Including the CUDA archs, which share the attribute via TORCH_CUDA_ARCH_LIST."""
    assert IcpxCompiler("ext", "kernel.sycl", build_dir, arch).gpu_arch == arch


@pytest.mark.parametrize(
    "arch, expected",
    [
        ("b580,lnl", "bmg,lnl"),
        ("lnl,b70", "lnl,bmg"),
        ("lnl,ptl", "lnl,ptl"),
        ("lnl, ptl", "lnl,ptl"),  # ocloc rejects the list over the space alone
    ],
)
def test_every_entry_of_a_target_list_is_translated(arch, expected, build_dir):
    """A gpu_arch may name several targets, and one bad entry fails the whole list."""
    assert IcpxCompiler("ext", "kernel.sycl", build_dir, arch).gpu_arch == expected
